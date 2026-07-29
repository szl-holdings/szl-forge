#!/usr/bin/env python3
"""Mint and verify the Qwen3.5 ReceiptAgent owner-signed evidence chain."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPTS = HERE / "receipts"
OWNER_PUBLIC_KEY = REPO / "receiptagent" / "owner_pubkey.json"
SIGNER_DIR = REPO / "receiptagent"
TRAINING_RECEIPT = RECEIPTS / "training_receipt.signed.json"
EVALUATION_RECEIPT = RECEIPTS / "eval_receipt.signed.json"
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
QUALIFICATION_SOURCE_FILES = (
    "frontier/qwen35-receiptagent-v2/qualify_runtime.py",
    "frontier/qwen35-receiptagent-v2/train_candidate.py",
    "frontier/qwen35-receiptagent-v2/evaluate_candidate.py",
)


class EvidenceError(RuntimeError):
    """A fail-closed evidence-chain validation error."""


def canonical_json(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=False)
                + ":"
                + canonical_json(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    raise TypeError(f"unsupported canonical JSON type: {type(value)!r}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must contain a JSON object")
    return value


def verify_self_digest(report: dict[str, Any], label: str) -> str:
    claimed = report.get("report_sha256")
    if not isinstance(claimed, str):
        raise EvidenceError(f"{label} has no report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    measured = sha256_json(unsigned)
    if measured != claimed:
        raise EvidenceError(
            f"{label} self-digest mismatch: measured {measured}, claimed {claimed}"
        )
    return claimed


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise EvidenceError(f"{label}: measured {actual!r}, expected {expected!r}")


def ensure_source_commit(source_commit: str) -> None:
    if not SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise EvidenceError("source commit must be a lowercase 40-hex SHA")
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise EvidenceError(f"source commit {source_commit} is not in this repository")


def source_bundle(revision: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in QUALIFICATION_SOURCE_FILES:
        completed = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=REPO,
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            raise EvidenceError(f"cannot read {path} from Git revision {revision}")
        result[path] = hashlib.sha256(completed.stdout).hexdigest()
    return result


def source_bundle_sha(bundle: dict[str, str]) -> str:
    return sha256_json(bundle)


def verify_source_binding(payload: dict[str, Any]) -> None:
    source_commit = payload.get("sourceCommit", "")
    if not SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise EvidenceError("receipt source commit must be a lowercase 40-hex SHA")
    declared = payload.get("qualificationSource")
    if not isinstance(declared, dict):
        raise EvidenceError("receipt qualification source bundle is missing")
    require_equal(
        "qualification source paths",
        tuple(sorted(declared)),
        tuple(sorted(QUALIFICATION_SOURCE_FILES)),
    )
    require_equal(
        "qualification source bundle SHA",
        payload.get("qualificationSourceSha256"),
        source_bundle_sha(declared),
    )
    # A protected squash does not retain intermediate commit ancestry. The
    # immutable source bundle therefore remains the durable binding. When the
    # original commit is present, verify it too; always require current Git
    # bytes to match the signed bundle.
    commit_probe = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=REPO,
        check=False,
        capture_output=True,
    )
    if commit_probe.returncode == 0:
        require_equal(
            "source commit bundle",
            source_bundle(source_commit),
            declared,
        )
    require_equal("current source bundle", source_bundle("HEAD"), declared)


def signing_function() -> Any:
    sys.path.insert(0, str(SIGNER_DIR))
    try:
        from sign_receipt import sign_payload
    finally:
        sys.path.pop(0)
    return sign_payload


def training_payload(
    candidate: dict[str, Any],
    report: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    measured = candidate["measured_evidence"]
    require_equal("training state", report.get("state"), "MEASURED_TRAINING_COMPLETED")
    require_equal("candidate id", report.get("candidate_id"), candidate["candidate_id"])
    report_sha = verify_self_digest(report, "training report")
    require_equal(
        "training report SHA",
        report_sha,
        measured["training_report_sha256"],
    )
    require_equal(
        "adapter aggregate SHA",
        report.get("adapter", {}).get("aggregate_sha256"),
        measured["adapter_aggregate_sha256"],
    )
    adapter_files = {
        item["path"]: item
        for item in report.get("adapter", {}).get("files", [])
        if isinstance(item, dict) and "path" in item
    }
    require_equal(
        "adapter weights SHA",
        adapter_files.get("adapter_model.safetensors", {}).get("sha256"),
        measured["adapter_model_sha256"],
    )
    require_equal(
        "optimizer steps",
        report.get("configuration", {}).get("max_steps"),
        measured["optimizer_steps"],
    )
    require_equal("training rows", report.get("training_rows"), measured["training_rows"])
    require_equal(
        "implementation revision",
        report.get("implementation", {}).get("revision"),
        candidate["training_implementation"]["revision"],
    )
    metrics = report.get("training", {}).get("metrics", {})
    source = source_bundle(source_commit)
    return {
        "kind": "szl-frontier-training-receipt",
        "v": 1,
        "candidateId": candidate["candidate_id"],
        "sourceCommit": source_commit,
        "qualificationSource": source,
        "qualificationSourceSha256": source_bundle_sha(source),
        "canonicalBase": candidate["canonical_base"],
        "trainingImplementation": candidate["training_implementation"],
        "datasets": report["dataset_hashes"],
        "trainingReportSha256": report_sha,
        "adapterAggregateSha256": measured["adapter_aggregate_sha256"],
        "adapterModelSha256": measured["adapter_model_sha256"],
        "optimizerSteps": measured["optimizer_steps"],
        "trainingRows": measured["training_rows"],
        "finalTrainLoss": format(float(metrics["train_loss"]), ".12g"),
        "trainedAt": report["measured_at"],
        "host": report["host"],
        "gpu": {
            "name": report["gpu"]["name"],
            "computeCapability": report["gpu"]["compute_capability"],
            "cudaRuntime": report["gpu"]["cuda_runtime"],
            "torchVersion": report["gpu"]["torch_version"],
            "peakReservedBytes": report["gpu"]["peak_reserved_bytes_training"],
            "temperatureCBeforeLoad": report["gpu"]["temperature_c_before_load"],
        },
        "autonomyEligible": False,
        "publicationEligible": False,
    }


def evaluation_payload(
    candidate: dict[str, Any],
    report: dict[str, Any],
    source_commit: str,
    training_receipt: dict[str, Any],
) -> dict[str, Any]:
    measured = candidate["measured_evidence"]
    require_equal(
        "evaluation state",
        report.get("state"),
        "MEASURED_EVALUATION_COMPLETED",
    )
    require_equal("candidate id", report.get("candidate_id"), candidate["candidate_id"])
    require_equal("acceptance", report.get("acceptance_passed"), True)
    report_sha = verify_self_digest(report, "evaluation report")
    require_equal(
        "evaluation report SHA",
        report_sha,
        measured["evaluation_report_sha256"],
    )
    require_equal(
        "training report chain",
        report.get("training_report_sha256"),
        measured["training_report_sha256"],
    )
    require_equal(
        "adapter chain",
        report.get("adapter_aggregate_sha256"),
        measured["adapter_aggregate_sha256"],
    )
    counts = report.get("counts", {})
    for key in (
        "eval_contract_valid",
        "eval_total",
        "adversarial_refused",
        "adversarial_total",
    ):
        require_equal(key, counts.get(key), measured[key])
    training_canonical_sha = sha256_text(training_receipt["canonical"])
    source = source_bundle(source_commit)
    return {
        "kind": "szl-frontier-eval-receipt",
        "v": 1,
        "candidateId": candidate["candidate_id"],
        "sourceCommit": source_commit,
        "qualificationSource": source,
        "qualificationSourceSha256": source_bundle_sha(source),
        "trainingReceiptCanonicalSha256": training_canonical_sha,
        "trainingReportSha256": measured["training_report_sha256"],
        "evaluationReportSha256": report_sha,
        "adapterAggregateSha256": measured["adapter_aggregate_sha256"],
        "datasets": report["dataset_hashes"],
        "evalTotal": measured["eval_total"],
        "evalContractValid": measured["eval_contract_valid"],
        "adversarialTotal": measured["adversarial_total"],
        "adversarialRefused": measured["adversarial_refused"],
        "evaluatedAt": report["measured_at"],
        "host": report["host"],
        "gpu": {
            "name": report["gpu"]["name"],
            "computeCapability": report["gpu"]["compute_capability"],
            "cudaRuntime": report["gpu"]["cuda_runtime"],
            "torchVersion": report["gpu"]["torch_version"],
            "peakReservedBytes": report["gpu"]["peak_reserved_bytes_evaluation"],
            "temperatureCBeforeLoad": report["gpu"]["temperature_c_before_load"],
        },
        "acceptancePassed": True,
        "autonomyEligible": False,
        "publicationEligible": False,
    }


def verify_wrapper(
    wrapper: dict[str, Any],
    owner_key: dict[str, Any],
    label: str,
) -> str:
    payload = wrapper.get("payload")
    if not isinstance(payload, dict):
        raise EvidenceError(f"{label} payload is missing")
    canonical = canonical_json(payload)
    require_equal(f"{label} canonical", wrapper.get("canonical"), canonical)
    require_equal(f"{label} key id", wrapper.get("keyId"), owner_key["keyId"])
    require_equal(f"{label} payload key id", payload.get("keyId"), owner_key["keyId"])
    require_equal(
        f"{label} public key",
        wrapper.get("publicKeySpkiBase64"),
        owner_key["publicKeySpkiBase64"],
    )
    try:
        public_key = serialization.load_der_public_key(
            base64.b64decode(owner_key["publicKeySpkiBase64"], validate=True)
        )
        if not isinstance(public_key, Ed25519PublicKey):
            raise EvidenceError(f"{label} key is not Ed25519")
        public_key.verify(
            base64.b64decode(wrapper["signatureBase64"], validate=True),
            canonical.encode("utf-8"),
        )
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError(f"{label} signature verification failed: {exc}") from exc
    return sha256_text(canonical)


def verify_chain() -> dict[str, Any]:
    candidate = load_json(HERE / "candidate.json")
    owner_key = load_json(OWNER_PUBLIC_KEY)
    training = load_json(TRAINING_RECEIPT)
    evaluation = load_json(EVALUATION_RECEIPT)
    training_sha = verify_wrapper(training, owner_key, "training receipt")
    verify_wrapper(evaluation, owner_key, "evaluation receipt")
    require_equal(
        "receipt chain",
        evaluation["payload"].get("trainingReceiptCanonicalSha256"),
        training_sha,
    )
    for label, wrapper in (("training", training), ("evaluation", evaluation)):
        payload = wrapper["payload"]
        require_equal(
            f"{label} candidate id",
            payload.get("candidateId"),
            candidate["candidate_id"],
        )
        verify_source_binding(payload)
        require_equal(
            f"{label} adapter hash",
            payload.get("adapterAggregateSha256"),
            candidate["measured_evidence"]["adapter_aggregate_sha256"],
        )
        require_equal(f"{label} publication flag", payload.get("publicationEligible"), False)
        require_equal(f"{label} autonomy flag", payload.get("autonomyEligible"), False)
    counts = candidate["measured_evidence"]
    eval_payload = evaluation["payload"]
    require_equal("eval count", eval_payload.get("evalContractValid"), counts["eval_contract_valid"])
    require_equal("eval total", eval_payload.get("evalTotal"), counts["eval_total"])
    require_equal(
        "refusal count",
        eval_payload.get("adversarialRefused"),
        counts["adversarial_refused"],
    )
    require_equal(
        "refusal total",
        eval_payload.get("adversarialTotal"),
        counts["adversarial_total"],
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "key_id": owner_key["keyId"],
        "training_receipt_canonical_sha256": training_sha,
        "evaluation_receipt_canonical_sha256": sha256_text(evaluation["canonical"]),
        "chain_valid": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
    }


def mint(args: argparse.Namespace) -> dict[str, Any]:
    ensure_source_commit(args.source_commit)
    candidate = load_json(HERE / "candidate.json")
    training_report = load_json(args.training_report)
    evaluation_report = load_json(args.evaluation_report)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    sign_payload = signing_function()
    training = sign_payload(
        training_payload(candidate, training_report, args.source_commit),
        str(TRAINING_RECEIPT),
    )
    sign_payload(
        evaluation_payload(
            candidate,
            evaluation_report,
            args.source_commit,
            training,
        ),
        str(EVALUATION_RECEIPT),
    )
    return verify_chain()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mint_parser = sub.add_parser("mint", help="validate reports and mint the receipt chain")
    mint_parser.add_argument("--training-report", type=Path, required=True)
    mint_parser.add_argument("--evaluation-report", type=Path, required=True)
    mint_parser.add_argument("--source-commit", required=True)
    sub.add_parser("verify", help="verify the committed receipt chain")
    args = parser.parse_args()
    try:
        result = mint(args) if args.command == "mint" else verify_chain()
    except EvidenceError as exc:
        print(json.dumps({"chain_valid": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
