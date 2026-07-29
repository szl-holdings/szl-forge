#!/usr/bin/env python3
"""Fail-closed verification for the SZL Hugging Face model/kernel portfolio."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTFOLIO = ROOT / "portfolio" / "model_portfolio.json"
RECEIPT_FILES = (
    "owner_pubkey.json",
    "training_receipt.signed.json",
    "eval_receipt.signed.json",
)


class PortfolioError(RuntimeError):
    """A portfolio invariant failed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source(path: Path) -> str:
    """Hash committed source bytes, avoiding Windows checkout EOL translation."""

    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return sha256_path(path)
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return hashlib.sha256(completed.stdout).hexdigest()
    return sha256_path(path)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def verify_signed_receipts(receipt_dir: Path) -> dict[str, Any]:
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    missing = [name for name in RECEIPT_FILES if not (receipt_dir / name).is_file()]
    if missing:
        raise PortfolioError(
            f"{receipt_dir.relative_to(ROOT)} missing receipt files: {missing}"
        )
    declared_key = json.loads(
        (receipt_dir / "owner_pubkey.json").read_text(encoding="utf-8")
    )
    wrappers: dict[str, dict[str, Any]] = {}
    for name in RECEIPT_FILES[1:]:
        wrapper = json.loads((receipt_dir / name).read_text(encoding="utf-8"))
        expected = canonical_json(wrapper["payload"])
        if wrapper.get("canonical") != expected:
            raise PortfolioError(f"{name}: canonical payload mismatch")
        if wrapper.get("keyId") != declared_key.get("keyId"):
            raise PortfolioError(f"{name}: keyId differs from owner_pubkey.json")
        if wrapper.get("publicKeySpkiBase64") != declared_key.get(
            "publicKeySpkiBase64"
        ):
            raise PortfolioError(f"{name}: public key differs from owner_pubkey.json")
        public_key = load_der_public_key(
            base64.b64decode(wrapper["publicKeySpkiBase64"])
        )
        public_key.verify(
            base64.b64decode(wrapper["signatureBase64"]),
            wrapper["canonical"].encode("utf-8"),
        )
        wrappers[name] = wrapper

    training = wrappers["training_receipt.signed.json"]
    evaluation = wrappers["eval_receipt.signed.json"]
    training_digest = hashlib.sha256(training["canonical"].encode("utf-8")).hexdigest()
    if evaluation["payload"].get("trainingReceiptSha256") != training_digest:
        raise PortfolioError("evaluation receipt does not hash-chain to training")
    if evaluation["payload"].get("weightsArtifactSha256") != training["payload"].get(
        "weightsArtifactSha256"
    ):
        raise PortfolioError("training and evaluation receipts name different weights")
    for filename, expected in training["payload"].get("datasets", {}).items():
        local = receipt_dir / filename
        if not local.is_file():
            raise PortfolioError(f"receipt-bound dataset missing: {local}")
        if sha256_source(local) != expected:
            raise PortfolioError(f"receipt-bound dataset hash mismatch: {local}")

    result = {
        "status": "DECLARED_KEY_SIGNATURES_VALID",
        "key_id": declared_key["keyId"],
        "training_canonical_sha256": training_digest,
        "evaluation_canonical_sha256": hashlib.sha256(
            evaluation["canonical"].encode("utf-8")
        ).hexdigest(),
        "weights_artifact_sha256": training["payload"]["weightsArtifactSha256"],
        "weights_hash_recomputed": False,
        "weights_hash_boundary": (
            "merged weight directories are not stored in Git; their aggregate hash "
            "is signed but cannot be recomputed from this source checkout"
        ),
    }
    for field in (
        "evalTotal",
        "evalContractValid",
        "adversarialTotal",
        "adversarialRefused",
        "planTotal",
        "planValid",
        "groundingTotal",
        "groundingCorrect",
        "abstainTotal",
        "abstainCorrect",
        "hallucinatedCitationCount",
    ):
        if field in evaluation["payload"]:
            result[field] = evaluation["payload"][field]
    return result


def validate_portfolio(document: dict[str, Any]) -> list[str]:
    if document.get("schema") != "szl.model-kernel-portfolio/v1":
        raise PortfolioError("unsupported portfolio schema")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PortfolioError("portfolio artifacts must be a non-empty list")
    repo_ids = [item.get("repo_id") for item in artifacts]
    if any(not isinstance(repo_id, str) or "/" not in repo_id for repo_id in repo_ids):
        raise PortfolioError("every artifact requires a namespace/repository repo_id")
    if len(repo_ids) != len(set(repo_ids)):
        raise PortfolioError("duplicate repo_id in portfolio")
    allowed = {"trained_model", "quantized_model", "learned_kernel", "software_kernel"}
    for artifact in artifacts:
        if artifact.get("kind") not in allowed:
            raise PortfolioError(f"{artifact['repo_id']}: unsupported kind")
        if artifact.get("autonomy_eligible") is not False:
            raise PortfolioError(
                f"{artifact['repo_id']}: autonomy requires a separate reviewed policy"
            )
    return repo_ids


def sibling_record(sibling: Any) -> dict[str, Any]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        lfs_sha256 = None
    elif isinstance(lfs, dict):
        lfs_sha256 = lfs.get("sha256")
    else:
        lfs_sha256 = getattr(lfs, "sha256", None)
    return {
        "path": sibling.rfilename,
        "size": getattr(sibling, "size", None),
        "lfs_sha256": lfs_sha256,
    }


def audit_live_artifact(
    artifact: dict[str, Any],
    *,
    api: Any,
    weight_extensions: tuple[str, ...],
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    info = api.model_info(artifact["repo_id"], files_metadata=True)
    files = [sibling_record(item) for item in info.siblings]
    paths = {item["path"] for item in files}
    missing = sorted(set(artifact.get("required_files", [])) - paths)
    weight_files = [
        item
        for item in files
        if item["path"].lower().endswith(weight_extensions)
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if missing:
        errors.append(f"required files absent: {missing}")
    license_name = getattr(getattr(info, "card_data", None), "license", None)
    if str(license_name).lower() != "apache-2.0":
        errors.append(f"license is {license_name!r}, expected 'apache-2.0'")
    kind = artifact["kind"]
    if kind in {"trained_model", "quantized_model", "learned_kernel"} and not weight_files:
        errors.append(f"{kind} has no weight artifact")
    if kind == "software_kernel" and weight_files:
        errors.append(
            "software_kernel unexpectedly contains model weight files: "
            + ", ".join(item["path"] for item in weight_files)
        )
    total_weight_bytes = sum(int(item["size"] or 0) for item in weight_files)
    maximum = artifact.get("max_total_weight_bytes")
    if maximum is not None and total_weight_bytes > int(maximum):
        errors.append(
            f"weight bytes {total_weight_bytes} exceed declared maximum {maximum}"
        )
    if artifact.get("github_source") is None:
        warnings.append("canonical GitHub source is unbound")

    receipt_parity: dict[str, Any] | None = None
    local_dir = artifact.get("local_receipt_dir")
    if local_dir:
        local_root = ROOT / local_dir
        receipt_parity = {}
        for name in RECEIPT_FILES:
            remote = Path(
                hf_hub_download(
                    repo_id=artifact["repo_id"],
                    filename=name,
                    repo_type="model",
                    force_download=True,
                )
            )
            local_sha = sha256_path(local_root / name)
            remote_sha = sha256_path(remote)
            receipt_parity[name] = {
                "local_sha256": local_sha,
                "remote_sha256": remote_sha,
                "matched": local_sha == remote_sha,
            }
            if local_sha != remote_sha:
                errors.append(f"{name}: Hub bytes differ from canonical Git source")

    return {
        "repo_id": artifact["repo_id"],
        "hub_revision": info.sha,
        "kind": kind,
        "maturity": artifact["maturity"],
        "license": license_name,
        "downloads": getattr(info, "downloads", None),
        "files": len(files),
        "weight_files": weight_files,
        "total_weight_bytes": total_weight_bytes,
        "receipt_parity": receipt_parity,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def build_report(document: dict[str, Any], *, live: bool) -> dict[str, Any]:
    repo_ids = validate_portfolio(document)
    receipt_evidence: dict[str, Any] = {}
    for artifact in document["artifacts"]:
        local_dir = artifact.get("local_receipt_dir")
        if local_dir:
            receipt_evidence[artifact["repo_id"]] = verify_signed_receipts(
                ROOT / local_dir
            )

    live_results: list[dict[str, Any]] = []
    if live:
        from huggingface_hub import HfApi

        api = HfApi()
        extensions = tuple(document["policy"]["model_weight_extensions"])
        for artifact in document["artifacts"]:
            live_results.append(
                audit_live_artifact(
                    artifact,
                    api=api,
                    weight_extensions=extensions,
                )
            )

    errors = [
        f"{result['repo_id']}: {message}"
        for result in live_results
        for message in result["errors"]
    ]
    return {
        "schema": "szl.model-kernel-portfolio-report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LIVE_PUBLIC_HUB" if live else "OFFLINE_SOURCE",
        "portfolio_size": len(repo_ids),
        "trained_models": sum(
            item["kind"] == "trained_model" for item in document["artifacts"]
        ),
        "quantized_models": sum(
            item["kind"] == "quantized_model" for item in document["artifacts"]
        ),
        "learned_kernels": sum(
            item["kind"] == "learned_kernel" for item in document["artifacts"]
        ),
        "software_kernels": sum(
            item["kind"] == "software_kernel" for item in document["artifacts"]
        ),
        "receipt_evidence": receipt_evidence,
        "live_artifacts": live_results,
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        document = json.loads(Path(args.portfolio).read_text(encoding="utf-8"))
        report = build_report(document, live=args.live)
    except Exception as exc:  # noqa: BLE001 - terminal verifier must emit evidence
        report = {
            "schema": "szl.model-kernel-portfolio-report/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "fatal": f"{type(exc).__name__}: {exc}",
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
