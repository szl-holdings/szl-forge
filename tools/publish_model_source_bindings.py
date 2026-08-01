#!/usr/bin/env python3
"""Publish and verify exact-source contracts for qualified Hub models."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import statistics
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "publishing" / "model-source-bindings.json"
DEFAULT_REPORT = ROOT / "reports" / "model-source-bindings.json"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BindingError(RuntimeError):
    """Raised when a model cannot be bound without weakening evidence."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "szl.model-source-bindings/v2":
        raise BindingError("unsupported binding contract schema")
    repository = payload.get("source_repository")
    if not isinstance(repository, str) or repository.count("/") != 1:
        raise BindingError("source_repository must be an owner/repository name")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BindingError("binding contract requires at least one artifact")
    repo_ids: set[str] = set()
    for artifact in artifacts:
        repo_id = artifact.get("repo_id")
        if not isinstance(repo_id, str) or repo_id in repo_ids:
            raise BindingError("every artifact requires a unique repo_id")
        repo_ids.add(repo_id)
        if not artifact.get("source_path"):
            raise BindingError(f"{repo_id}: source_path is required")
        if artifact.get("artifact_class") not in {
            "fine_tuned_model",
            "quantized_model",
        }:
            raise BindingError(f"{repo_id}: unsupported artifact_class")
        if not artifact.get("promotion_state"):
            raise BindingError(f"{repo_id}: promotion_state is required")
        if not artifact.get("required_hub_files"):
            raise BindingError(f"{repo_id}: required_hub_files is required")
        if not artifact.get("source_files"):
            raise BindingError(f"{repo_id}: source_files is required")
        if not artifact.get("lineage"):
            raise BindingError(f"{repo_id}: exact base-model lineage is required")
        if not artifact.get("signed_receipts"):
            raise BindingError(f"{repo_id}: signed_receipts contract is required")
    return payload


def source_evidence(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative in artifact["source_files"]:
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise BindingError(
                f"{artifact['repo_id']}: source file is missing or outside the repository: {relative}"
            )
        evidence.append(
            {
                "path": Path(relative).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return evidence


def hub_evidence(api: HfApi, artifact: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    info = api.model_info(artifact["repo_id"], files_metadata=True)
    files = {sibling.rfilename: sibling for sibling in info.siblings or []}
    missing = sorted(set(artifact["required_hub_files"]) - set(files))
    if missing:
        raise BindingError(f"{artifact['repo_id']}: missing Hub files: {missing}")

    expected_hashes = artifact.get("expected_weight_sha256") or {}
    for filename, expected in sorted(expected_hashes.items()):
        sibling = files.get(filename)
        observed = getattr(getattr(sibling, "lfs", None), "sha256", None)
        if observed != expected:
            raise BindingError(
                f"{artifact['repo_id']}: {filename} SHA-256 drifted "
                f"(expected {expected}, observed {observed or 'UNAVAILABLE'})"
            )

    evidence: list[dict[str, Any]] = []
    for filename in artifact["required_hub_files"]:
        sibling = files[filename]
        evidence.append(
            {
                "path": filename,
                "bytes": sibling.size,
                "blob_id": sibling.blob_id,
                "lfs_sha256": getattr(getattr(sibling, "lfs", None), "sha256", None),
            }
        )
    return info.sha, evidence


def _card_license(info: Any) -> str | None:
    card_data = getattr(info, "card_data", None)
    value = getattr(card_data, "license", None) if card_data is not None else None
    if isinstance(value, str):
        return value.lower()
    for tag in getattr(info, "tags", None) or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()
    return None


def lineage_evidence(api: HfApi, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for step in artifact["lineage"]:
        repo_id = step.get("repo_id")
        revision = step.get("revision")
        expected_license = str(step.get("license") or "").lower()
        if not isinstance(repo_id, str) or FULL_SHA_RE.fullmatch(str(revision)) is None:
            raise BindingError(f"{artifact['repo_id']}: lineage requires exact repo and revision")
        if not expected_license:
            raise BindingError(f"{artifact['repo_id']}: lineage license is required")
        info = api.model_info(repo_id, revision=revision)
        if info.sha != revision:
            raise BindingError(
                f"{artifact['repo_id']}: lineage revision drift for {repo_id} "
                f"(expected {revision}, observed {info.sha})"
            )
        observed_license = _card_license(info)
        if observed_license != expected_license:
            raise BindingError(
                f"{artifact['repo_id']}: license drift for {repo_id} "
                f"(expected {expected_license}, observed {observed_license or 'UNAVAILABLE'})"
            )
        evidence.append(
            {
                "relation": step["relation"],
                "repo_id": repo_id,
                "revision": revision,
                "license": observed_license,
                "status": "EXACT_REVISION_AND_LICENSE_VERIFIED",
            }
        )
    return evidence


def signed_receipt_evidence(
    artifact: dict[str, Any],
    hub_revision: str,
    token: str | None,
    downloader: Callable[..., str],
) -> dict[str, Any]:
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    contract = artifact["signed_receipts"]
    names = {
        "public_key": contract["public_key"],
        "training": contract["training"],
        "evaluation": contract["evaluation"],
    }
    paths = {
        role: Path(
            downloader(
                repo_id=artifact["repo_id"],
                filename=filename,
                repo_type="model",
                revision=hub_revision,
                token=token,
            )
        )
        for role, filename in names.items()
    }
    declared_key = json.loads(paths["public_key"].read_text(encoding="utf-8"))
    receipts: dict[str, dict[str, Any]] = {}
    file_evidence: dict[str, dict[str, Any]] = {}
    for role in ("training", "evaluation"):
        path = paths[role]
        receipt = json.loads(path.read_text(encoding="utf-8"))
        canonical = json.dumps(
            receipt["payload"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if canonical != receipt.get("canonical"):
            raise BindingError(f"{artifact['repo_id']}: {role} receipt canonical mismatch")
        if receipt.get("publicKeySpkiBase64") != declared_key.get("publicKeySpkiBase64"):
            raise BindingError(f"{artifact['repo_id']}: {role} receipt key mismatch")
        if receipt.get("keyId") != declared_key.get("keyId"):
            raise BindingError(f"{artifact['repo_id']}: {role} receipt key id mismatch")
        public_key = load_der_public_key(base64.b64decode(receipt["publicKeySpkiBase64"]))
        public_key.verify(
            base64.b64decode(receipt["signatureBase64"]), canonical.encode("utf-8")
        )
        receipts[role] = receipt
        file_evidence[names[role]] = {
            "sha256": file_sha256(path),
            "canonical_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "signature": "VALID_AGAINST_REPOSITORY_DECLARED_KEY",
        }
    training_canonical_sha = file_evidence[names["training"]]["canonical_sha256"]
    if receipts["evaluation"]["payload"].get("trainingReceiptSha256") != training_canonical_sha:
        raise BindingError(f"{artifact['repo_id']}: evaluation receipt chain mismatch")
    eval_payload = receipts["evaluation"]["payload"]
    return {
        "status": "DECLARED_KEY_SIGNATURES_VALID",
        "claim_scope": contract["claim_scope"],
        "key_id": declared_key["keyId"],
        "public_key_file": {
            "path": names["public_key"],
            "sha256": file_sha256(paths["public_key"]),
        },
        "receipts": file_evidence,
        "held_out_evaluation": {
            key: eval_payload.get(key)
            for key in (
                "planTotal",
                "planValid",
                "groundingTotal",
                "groundingCorrect",
                "abstainTotal",
                "abstainCorrect",
                "hallucinatedCitationCount",
            )
            if key in eval_payload
        },
        "independent_identity_binding": "NOT_ESTABLISHED",
    }


def _json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 65.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise BindingError(f"runtime probe failed: {url} returned {response.status}")
        return json.loads(response.read())


def runtime_evidence(
    artifact: dict[str, Any],
    requester: Callable[..., dict[str, Any]] = _json_request,
    *,
    expected_source_revision: str | None = None,
) -> dict[str, Any] | None:
    probe = artifact.get("runtime_probe")
    if probe is None:
        return None
    base = probe["base_url"].rstrip("/")
    health = requester(f"{base}/health")
    build = requester(f"{base}/api/build-info")
    identity = requester(f"{base}/api/v1/identity")
    if health != {
        "status": "READY",
        "model_sha256_verified": True,
        "source_integrity": True,
        "receipt_status": "DECLARED_KEY_SIGNATURES_VALID",
        "failure_code": None,
    }:
        raise BindingError(f"{artifact['repo_id']}: live health is not the exact READY contract")
    runtime = build.get("runtime") or {}
    if runtime.get("state") != "READY" or not runtime.get("model_sha256_verified"):
        raise BindingError(f"{artifact['repo_id']}: build-info runtime is not ready")
    build_revision = (build.get("build") or {}).get("revision")
    if FULL_SHA_RE.fullmatch(str(build_revision)) is None:
        raise BindingError(f"{artifact['repo_id']}: runtime source revision is not immutable")
    if expected_source_revision is not None:
        expected_source_revision = expected_source_revision.strip().lower()
        if FULL_SHA_RE.fullmatch(expected_source_revision) is None:
            raise BindingError(
                "expected runtime source revision must be an exact 40-character Git SHA"
            )
        if build_revision != expected_source_revision:
            raise BindingError(
                f"{artifact['repo_id']}: live runtime source revision {build_revision} "
                f"does not match expected {expected_source_revision}"
            )
    model = identity.get("model") or {}
    space = identity.get("space") or {}
    if (
        model.get("repo") != artifact["repo_id"]
        or model.get("revision") != probe["model_revision"]
        or model.get("file") != probe["model_file"]
        or model.get("sha256_loaded") != probe["model_sha256"]
        or model.get("sha256_expected") != probe["model_sha256"]
        or space.get("release_id") != probe["release_id"]
        or space.get("source_integrity") is not True
    ):
        raise BindingError(f"{artifact['repo_id']}: live identity drifted from the pinned contract")
    outputs: list[str] = []
    latencies: list[int] = []
    output_hashes: list[str] = []
    for _ in range(int(probe["repeat_count"])):
        result = requester(
            f"{base}/api/v1/infer",
            payload={
                "prompt": probe["prompt"],
                "max_new_tokens": probe["max_new_tokens"],
            },
        )
        result_model = result.get("model") or {}
        if (
            result_model.get("revision") != probe["model_revision"]
            or result_model.get("sha256") != probe["model_sha256"]
            or not isinstance(result.get("output"), str)
            or probe["required_output_substring"].lower() not in result["output"].lower()
        ):
            raise BindingError(f"{artifact['repo_id']}: live inference contract failed")
        elapsed = result.get("elapsed_ms")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise BindingError(f"{artifact['repo_id']}: live latency is malformed")
        outputs.append(result["output"])
        latencies.append(elapsed)
        output_hashes.append(hashlib.sha256(result["output"].encode("utf-8")).hexdigest())
    if len(set(outputs)) != 1:
        raise BindingError(f"{artifact['repo_id']}: repeated greedy inference was not reproducible")
    hardware = identity.get("hardware") or {}
    return {
        "status": "VERIFIED_OPERATIONAL_LIMITED",
        "clean_load": "VERIFIED",
        "runtime_source_revision": build_revision,
        "runtime_model_revision": probe["model_revision"],
        "runtime_model_sha256": probe["model_sha256"],
        "release_id": probe["release_id"],
        "inference": {
            "runs": len(outputs),
            "deterministic_outputs_equal": True,
            "output_sha256": output_hashes[0],
            "latency_ms": latencies,
            "p50_latency_ms": statistics.median(latencies),
            "measurement_state": "MEASURED_LIVE_SERVICE",
        },
        "memory": {
            "value": hardware.get("memory_observed"),
            "state": "REPORTED_BY_RUNTIME_ENV_NOT_PROCESS_PEAK",
        },
        "energy": {"state": "UNAVAILABLE"},
        "restart_reproducibility": {"state": "UNAVAILABLE_NOT_TESTED_BY_THIS_PROBE"},
        "service_level": "BEST_EFFORT_NO_SLA",
    }


def publication_payload(
    contract: dict[str, Any],
    artifact: dict[str, Any],
    source_revision: str,
    hub_revision_before: str,
    source_files: list[dict[str, Any]],
    hub_files: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    signed_receipts: dict[str, Any],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": "szl.hf-model-source-binding/v2",
        "artifact": {
            "repo_id": artifact["repo_id"],
            "repo_type": "model",
            "artifact_class": artifact["artifact_class"],
            "role": artifact["role"],
            "maturity": artifact["maturity"],
            "promotion_state": artifact["promotion_state"],
        },
        "source_repository": contract["source_repository"],
        "source_revision": source_revision,
        "source": {
            "repository": f"https://github.com/{contract['source_repository']}",
            "revision": source_revision,
            "path": artifact["source_path"],
            "relation": "CANONICAL_SOURCE_CURRICULUM_SCHEMA_AND_SIGNED_RECEIPTS",
        },
        "observed_hub_revision_before_binding": hub_revision_before,
        "source_files": source_files,
        "hub_files": hub_files,
        "lineage": lineage,
        "signed_receipts": signed_receipts,
        "runtime": runtime or {"status": "NOT_QUALIFIED_NO_RUNTIME_PROBE"},
        "autonomy_boundary": artifact.get("autonomy_boundary")
        or {
            "autonomous_execution": False,
            "controller_validation_required": True,
            "reason": "No autonomous authority is granted by this binding.",
        },
        "release_receipt": {
            "status": "UNSIGNED_EXACT_REVISION_READBACK",
            "owner_signed_release_receipt": "UNAVAILABLE",
            "reason": "The publication record is hash-bound and immutable-readback verified; no approved local owner signing key is used by this workflow.",
        },
        "claims": {
            "source_binding": "EXACT_GIT_REVISION",
            "artifact_equivalence": contract["policy"]["artifact_equivalence"],
            "reproducible_build": contract["policy"]["reproducible_build"],
            "independent_quality_certification": "NOT_CLAIMED",
            "energy_measurement": (
                (runtime or {}).get("energy", {}).get("state", "UNAVAILABLE")
            ),
        },
        "limitations": artifact["limitations"],
        "policy_statement": contract["policy"]["statement"],
    }


def prepare_one(
    api: HfApi,
    contract: dict[str, Any],
    artifact: dict[str, Any],
    *,
    source_revision: str,
    token: str | None,
    downloader: Callable[..., str] = hf_hub_download,
    requester: Callable[..., dict[str, Any]] = _json_request,
    expected_runtime_source_revision: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    source_files = source_evidence(artifact)
    hub_revision_before, hub_files = hub_evidence(api, artifact)
    lineage = lineage_evidence(api, artifact)
    receipts = signed_receipt_evidence(
        artifact, hub_revision_before, token, downloader
    )
    runtime = runtime_evidence(
        artifact,
        requester,
        expected_source_revision=expected_runtime_source_revision,
    )
    publication = publication_payload(
        contract,
        artifact,
        source_revision,
        hub_revision_before,
        source_files,
        hub_files,
        lineage,
        receipts,
        runtime,
    )
    result: dict[str, Any] = {
        "repo_id": artifact["repo_id"],
        "artifact_class": artifact["artifact_class"],
        "promotion_state": artifact["promotion_state"],
        "status": "VERIFIED_DRY_RUN",
        "source_revision": source_revision,
        "hub_revision_before": hub_revision_before,
        "lineage_status": "EXACT_REVISIONS_AND_LICENSES_VERIFIED",
        "signed_receipt_status": receipts["status"],
        "held_out_evaluation": receipts["held_out_evaluation"],
        "runtime_status": (runtime or {}).get(
            "status", "NOT_QUALIFIED_NO_RUNTIME_PROBE"
        ),
        "release_receipt_status": "UNSIGNED_EXACT_REVISION_READBACK",
        "publication_sha256": hashlib.sha256(
            canonical_json(publication).encode("utf-8")
        ).hexdigest(),
    }
    body = canonical_json(publication).encode("utf-8")
    return result, body


def publish_prepared(
    api: HfApi,
    artifact: dict[str, Any],
    result: dict[str, Any],
    body: bytes,
    *,
    source_revision: str,
    token: str,
) -> dict[str, Any]:
    commit = api.upload_file(
        path_or_fileobj=io.BytesIO(body),
        path_in_repo="publication.json",
        repo_id=artifact["repo_id"],
        repo_type="model",
        revision="main",
        parent_commit=result["hub_revision_before"],
        token=token,
        commit_message=f"Bind canonical source {source_revision[:12]}",
    )
    revision = getattr(commit, "oid", None) or api.model_info(
        artifact["repo_id"], token=token
    ).sha
    downloaded = Path(
        hf_hub_download(
            artifact["repo_id"],
            "publication.json",
            repo_type="model",
            revision=revision,
            token=token,
        )
    ).read_bytes()
    if downloaded != body:
        raise BindingError(f"{artifact['repo_id']}: publication readback mismatch")
    result.update(
        {
            "status": "PUBLISHED_AND_READBACK_VERIFIED",
            "hub_revision_after": revision,
        }
    )
    return result


def publish_one(
    api: HfApi,
    contract: dict[str, Any],
    artifact: dict[str, Any],
    *,
    source_revision: str,
    publish: bool,
    token: str | None,
    downloader: Callable[..., str] = hf_hub_download,
    requester: Callable[..., dict[str, Any]] = _json_request,
    expected_runtime_source_revision: str | None = None,
) -> dict[str, Any]:
    if publish and not token:
        raise BindingError("HF_TOKEN is required when --publish is used")
    result, body = prepare_one(
        api,
        contract,
        artifact,
        source_revision=source_revision,
        token=token,
        downloader=downloader,
        requester=requester,
        expected_runtime_source_revision=expected_runtime_source_revision,
    )
    if not publish:
        return result
    return publish_prepared(
        api,
        artifact,
        result,
        body,
        source_revision=source_revision,
        token=token,
    )


def run(
    *,
    contract_path: Path,
    report_path: Path,
    source_revision: str,
    publish: bool,
    token: str | None,
    api: HfApi | None = None,
    downloader: Callable[..., str] = hf_hub_download,
    requester: Callable[..., dict[str, Any]] = _json_request,
    expected_runtime_source_revision: str | None = None,
) -> dict[str, Any]:
    source_revision = source_revision.strip().lower()
    if FULL_SHA_RE.fullmatch(source_revision) is None:
        raise BindingError("source revision must be an exact 40-character Git SHA")
    if expected_runtime_source_revision is not None:
        expected_runtime_source_revision = (
            expected_runtime_source_revision.strip().lower()
        )
        if not expected_runtime_source_revision:
            expected_runtime_source_revision = None
        elif FULL_SHA_RE.fullmatch(expected_runtime_source_revision) is None:
            raise BindingError(
                "expected runtime source revision must be an exact 40-character Git SHA"
            )
    contract = load_contract(contract_path)
    api = api or HfApi(token=token)
    if publish and not token:
        raise BindingError("HF_TOKEN is required when --publish is used")

    prepared = [
        prepare_one(
            api,
            contract,
            artifact,
            source_revision=source_revision,
            token=token,
            downloader=downloader,
            requester=requester,
            expected_runtime_source_revision=expected_runtime_source_revision,
        )
        for artifact in contract["artifacts"]
    ]
    results = [result for result, _ in prepared]
    if publish:
        runtime_first = sorted(
            range(len(contract["artifacts"])),
            key=lambda index: (
                contract["artifacts"][index].get("runtime_probe") is None
            ),
        )
        for index in runtime_first:
            artifact = contract["artifacts"][index]
            result, body = prepared[index]
            results[index] = publish_prepared(
                api,
                artifact,
                result,
                body,
                source_revision=source_revision,
                token=token,
            )
    report = {
        "schema": "szl.model-source-binding-report/v2",
        "mode": "PUBLISH" if publish else "DRY_RUN",
        "source_repository": contract["source_repository"],
        "source_revision": source_revision,
        "results": results,
        "status": (
            "PUBLISHED_AND_READBACK_VERIFIED"
            if publish
            else "VERIFIED_DRY_RUN"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_json(report), encoding="utf-8")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--source-revision",
        default=os.getenv("GITHUB_SHA", ""),
        help="Exact canonical Git revision. Defaults to GITHUB_SHA.",
    )
    parser.add_argument(
        "--expected-runtime-source-revision",
        help="Optional exact live runtime revision required before publication.",
    )
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        contract_path=args.contract,
        report_path=args.report,
        source_revision=args.source_revision,
        publish=args.publish,
        token=os.getenv("HF_TOKEN"),
        expected_runtime_source_revision=args.expected_runtime_source_revision,
    )
    print(canonical_json(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
