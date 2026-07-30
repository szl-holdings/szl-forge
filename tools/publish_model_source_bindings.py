#!/usr/bin/env python3
"""Publish and verify exact-source contracts for qualified Hub models."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

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
    if payload.get("schema") != "szl.model-source-bindings/v1":
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
        if not artifact.get("required_hub_files"):
            raise BindingError(f"{repo_id}: required_hub_files is required")
        if not artifact.get("source_files"):
            raise BindingError(f"{repo_id}: source_files is required")
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


def publication_payload(
    contract: dict[str, Any],
    artifact: dict[str, Any],
    source_revision: str,
    hub_revision_before: str,
    source_files: list[dict[str, Any]],
    hub_files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "szl.hf-model-source-binding/v1",
        "artifact": {
            "repo_id": artifact["repo_id"],
            "repo_type": "model",
            "role": artifact["role"],
            "maturity": artifact["maturity"],
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
        "claims": {
            "source_binding": "EXACT_GIT_REVISION",
            "artifact_equivalence": contract["policy"]["artifact_equivalence"],
            "reproducible_build": contract["policy"]["reproducible_build"],
            "independent_quality_certification": "NOT_CLAIMED",
        },
        "limitations": artifact["limitations"],
        "policy_statement": contract["policy"]["statement"],
    }


def publish_one(
    api: HfApi,
    contract: dict[str, Any],
    artifact: dict[str, Any],
    *,
    source_revision: str,
    publish: bool,
    token: str | None,
) -> dict[str, Any]:
    source_files = source_evidence(artifact)
    hub_revision_before, hub_files = hub_evidence(api, artifact)
    publication = publication_payload(
        contract,
        artifact,
        source_revision,
        hub_revision_before,
        source_files,
        hub_files,
    )
    result: dict[str, Any] = {
        "repo_id": artifact["repo_id"],
        "status": "VERIFIED_DRY_RUN",
        "source_revision": source_revision,
        "hub_revision_before": hub_revision_before,
        "publication_sha256": hashlib.sha256(
            canonical_json(publication).encode("utf-8")
        ).hexdigest(),
    }
    if not publish:
        return result
    if not token:
        raise BindingError("HF_TOKEN is required when --publish is used")

    body = canonical_json(publication).encode("utf-8")
    commit = api.upload_file(
        path_or_fileobj=io.BytesIO(body),
        path_in_repo="publication.json",
        repo_id=artifact["repo_id"],
        repo_type="model",
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


def run(
    *,
    contract_path: Path,
    report_path: Path,
    source_revision: str,
    publish: bool,
    token: str | None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    source_revision = source_revision.strip().lower()
    if FULL_SHA_RE.fullmatch(source_revision) is None:
        raise BindingError("source revision must be an exact 40-character Git SHA")
    contract = load_contract(contract_path)
    api = api or HfApi(token=token)
    results = [
        publish_one(
            api,
            contract,
            artifact,
            source_revision=source_revision,
            publish=publish,
            token=token,
        )
        for artifact in contract["artifacts"]
    ]
    report = {
        "schema": "szl.model-source-binding-report/v1",
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
    )
    print(canonical_json(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
