#!/usr/bin/env python3
"""Prepare a deterministic, offline ReceiptAgent v3 release packet.

This tool does not publish, deploy, load a model, or turn local evidence into a
runtime claim.  It accepts only an authenticated receipt chain rooted in an
explicitly trusted Ed25519 key, rebinds that chain to the measured reports and
adapter bytes supplied locally, and writes a model card plus a local manifest
into a new empty directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat as stat_module
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import evidence_chain as evidence


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RELATIVE = "frontier/qwen35-receiptagent-v3"
CANDIDATE_ID = "SZL-ReceiptAgent-Qwen3.5-0.8B-v3"
TARGET_REPO_ID = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated"
TRUST_POLICY_FILENAME = "receipt-signing-trust-policy.json"
TRUST_POLICY_SCHEMA = "szl.receiptagent-v3-receipt-signing-trust-policy/v1"
TRUST_POLICY_USAGE = "AUTHENTICATED_TRAINING_EVALUATION_COMPARISON_RECEIPTS"
HEX_32 = re.compile(r"^[0-9a-f]{32}$")
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")

WRAPPER_KEYS = {
    "schema",
    "kind",
    "candidateId",
    "sourceRevision",
    "payload",
    "payloadSha256",
    "authentication",
    "receiptSha256",
}
AUTH_KEYS = {
    "algorithm",
    "keyId",
    "publicKeyFingerprintSha256",
    "publicKeySpkiBase64",
    "signatureBase64",
}
PAYLOAD_KEYS = {
    "TRAINING": {
        "runId",
        "childReportSha256",
        "supervisorReportSha256",
        "adapterAggregateSha256",
        "adapterManifestSha256",
        "datasetHashes",
        "gpuIdentitySha256",
        "containmentIdentitySha256",
        "runtimeIdentitySha256",
        "componentIdentitiesSha256",
        "previousReceiptSha256",
        "publicationEligible",
    },
    "EVALUATION": {
        "runId",
        "devReportSha256",
        "testReportSha256",
        "adapterAggregateSha256",
        "trainingReceiptSha256",
        "previousReceiptSha256",
        "publicationEligible",
    },
    "COMPARISON": {
        "comparisonReportSha256",
        "inputReportSha256s",
        "evaluationReceiptSha256",
        "previousReceiptSha256",
        "comparisonCriteriaSatisfied",
        "publicationEligible",
    },
}
REPORT_ARGUMENTS = {
    "childTraining": "child-training-report",
    "supervisor": "supervisor-report",
    "devEvaluation": "dev-report",
    "testEvaluation": "test-report",
    "baseTestEvaluation": "base-test-report",
    "v2TestEvaluation": "v2-test-report",
    "comparison": "comparison-report",
}
RECEIPT_OUTPUTS = {
    "TRAINING": ("training-receipt.json", "AUTHENTICATED_TRAINING_RECEIPT"),
    "EVALUATION": ("evaluation-receipt.json", "AUTHENTICATED_EVALUATION_RECEIPT"),
    "COMPARISON": ("comparison-receipt.json", "AUTHENTICATED_COMPARISON_RECEIPT"),
}
REPORT_OUTPUTS = {
    "childTraining": ("training-report.json", "MEASURED_CHILD_TRAINING_REPORT"),
    "supervisor": ("supervisor-report.json", "MEASURED_SUPERVISOR_REPORT"),
    "devEvaluation": ("dev-evaluation-report.json", "MEASURED_DEV_EVALUATION_REPORT"),
    "testEvaluation": ("test-evaluation-report.json", "MEASURED_TEST_EVALUATION_REPORT"),
    "baseTestEvaluation": (
        "base-test-evaluation-report.json",
        "MEASURED_BASE_TEST_EVALUATION_REPORT",
    ),
    "v2TestEvaluation": (
        "v2-test-evaluation-report.json",
        "MEASURED_V2_TEST_EVALUATION_REPORT",
    ),
    "comparison": ("comparison-report.json", "MEASURED_COMPARISON_REPORT"),
}
ALLOWED_ADAPTER_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "video_preprocessor_config.json",
}
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ADAPTER_FILE_BYTES = 256 * 1024 * 1024
WINDOWS_REPARSE_POINT = getattr(
    stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
)


class ReleaseError(RuntimeError):
    """A release precondition failed closed."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReleaseError("value is not canonical finite JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    if not data or len(data) > MAX_JSON_BYTES or b"\x00" in data:
        raise ReleaseError(f"{label} bytes are absent, oversized, or contain NUL")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReleaseError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be one JSON object")
    return value


def _is_reparse_stat(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    reparse_tag = int(getattr(info, "st_reparse_tag", 0) or 0)
    return (
        stat_module.S_ISLNK(info.st_mode)
        or bool(attributes & WINDOWS_REPARSE_POINT)
        or reparse_tag != 0
    )


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_existing_chain(
    path: Path, label: str, *, leaf_directory: bool
) -> os.stat_result:
    absolute = _absolute_lexical(path)
    chain = list(reversed((absolute, *absolute.parents)))
    leaf: os.stat_result | None = None
    for index, component in enumerate(chain):
        try:
            info = os.lstat(component)
        except OSError as exc:
            raise ReleaseError(f"{label} path component is unavailable") from exc
        if _is_reparse_stat(info):
            raise ReleaseError(
                f"{label} path contains a symlink, junction, or reparse point"
            )
        is_leaf = index == len(chain) - 1
        if is_leaf:
            leaf = info
            expected = stat_module.S_ISDIR if leaf_directory else stat_module.S_ISREG
            if not expected(info.st_mode):
                kind = "directory" if leaf_directory else "regular file"
                raise ReleaseError(f"{label} must be a {kind}")
        elif not stat_module.S_ISDIR(info.st_mode):
            raise ReleaseError(f"{label} ancestor is not a directory")
    assert leaf is not None
    return leaf


def _bounded_regular_file(path: Path, label: str, maximum: int) -> bytes:
    absolute = _absolute_lexical(path)
    before = _assert_existing_chain(absolute, label, leaf_directory=False)
    if before.st_size <= 0 or before.st_size > maximum:
        raise ReleaseError(f"{label} file size is invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ReleaseError(f"{label} could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or _is_reparse_stat(opened)
            or not os.path.samestat(before, opened)
            or opened.st_size != before.st_size
        ):
            raise ReleaseError(f"{label} changed during safe open")
        size = int(opened.st_size)
        data = bytearray(size)
        offset = 0
        while offset < size:
            chunk = os.read(descriptor, min(1024 * 1024, size - offset))
            if not chunk:
                raise ReleaseError(f"{label} ended before its declared size")
            data[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        if os.read(descriptor, 1):
            raise ReleaseError(f"{label} grew during its bounded read")
        after = os.fstat(descriptor)
        if (
            not os.path.samestat(opened, after)
            or after.st_size != size
            or after.st_mtime_ns != opened.st_mtime_ns
            or _is_reparse_stat(after)
        ):
            raise ReleaseError(f"{label} changed during its bounded read")
        return bytes(data)
    finally:
        os.close(descriptor)


def strict_json_file(path: Path, label: str) -> dict[str, Any]:
    return strict_json_bytes(
        _bounded_regular_file(path, label, MAX_JSON_BYTES), label
    )


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        raise ReleaseError(f"{label} must be lowercase SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ReleaseError(f"{label} keys differ; missing={missing}, unknown={unknown}")


def _exact_typed(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise ReleaseError(f"{label} differs")


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReleaseError(f"{label} must be a non-negative integer")
    return value


def validate_receipt_wrapper(wrapper: dict[str, Any], kind: str) -> None:
    _exact_keys(wrapper, WRAPPER_KEYS, f"{kind} receipt wrapper")
    if wrapper["schema"] != "szl.receiptagent-v3-authenticated-receipt/v1":
        raise ReleaseError(f"{kind} receipt schema is unsupported")
    if wrapper["kind"] != kind:
        raise ReleaseError(f"expected {kind} receipt")
    if wrapper["candidateId"] != CANDIDATE_ID:
        raise ReleaseError(f"{kind} receipt candidate differs")
    if not isinstance(wrapper["payload"], dict):
        raise ReleaseError(f"{kind} receipt payload must be an object")
    _exact_keys(wrapper["payload"], PAYLOAD_KEYS[kind], f"{kind} receipt payload")
    if wrapper["payloadSha256"] != sha256_json(wrapper["payload"]):
        raise ReleaseError(f"{kind} receipt payload digest differs")
    unsigned = dict(wrapper)
    unsigned.pop("receiptSha256")
    if wrapper["receiptSha256"] != sha256_json(unsigned):
        raise ReleaseError(f"{kind} receipt digest differs")
    authentication = wrapper["authentication"]
    if not isinstance(authentication, dict):
        raise ReleaseError(f"{kind} receipt authentication must be an object")
    _exact_keys(authentication, AUTH_KEYS, f"{kind} receipt authentication")
    if authentication["algorithm"] != "Ed25519":
        raise ReleaseError(f"{kind} receipt authentication algorithm differs")
    _digest(authentication["publicKeyFingerprintSha256"], "public key fingerprint")
    if wrapper["payload"]["publicationEligible"] is not False:
        raise ReleaseError(f"{kind} receipt crossed the publication boundary")


def validate_receipt_chain_shape(receipts: Mapping[str, dict[str, Any]]) -> None:
    if set(receipts) != set(PAYLOAD_KEYS):
        raise ReleaseError("exactly TRAINING, EVALUATION, and COMPARISON receipts are required")
    for kind in ("TRAINING", "EVALUATION", "COMPARISON"):
        validate_receipt_wrapper(receipts[kind], kind)
    training, evaluation, comparison = (
        receipts["TRAINING"], receipts["EVALUATION"], receipts["COMPARISON"]
    )
    source = training["sourceRevision"]
    if HEX_40.fullmatch(source or "") is None:
        raise ReleaseError("receipt source revision must be 40 lowercase hex characters")
    if any(receipts[k]["sourceRevision"] != source for k in receipts):
        raise ReleaseError("receipt source revisions differ")
    run_id = training["payload"]["runId"]
    if HEX_32.fullmatch(run_id or "") is None:
        raise ReleaseError("training run ID must be 32 lowercase hex characters")
    if evaluation["payload"]["runId"] != run_id:
        raise ReleaseError("evaluation run ID differs from training")
    if training["payload"]["previousReceiptSha256"] is not None:
        raise ReleaseError("training receipt must start the chain")
    if evaluation["payload"]["trainingReceiptSha256"] != training["receiptSha256"]:
        raise ReleaseError("evaluation training receipt binding differs")
    if evaluation["payload"]["previousReceiptSha256"] != training["receiptSha256"]:
        raise ReleaseError("evaluation previous receipt binding differs")
    if comparison["payload"]["evaluationReceiptSha256"] != evaluation["receiptSha256"]:
        raise ReleaseError("comparison evaluation receipt binding differs")
    if comparison["payload"]["previousReceiptSha256"] != evaluation["receiptSha256"]:
        raise ReleaseError("comparison previous receipt binding differs")
    if comparison["payload"]["comparisonCriteriaSatisfied"] is not True:
        raise ReleaseError("comparison criteria are not satisfied")
    dataset_hashes = training["payload"]["datasetHashes"]
    if not isinstance(dataset_hashes, dict):
        raise ReleaseError("training dataset hashes must be an object")
    _exact_keys(dataset_hashes, {"train", "dev", "test"}, "dataset hashes")
    for name, digest in dataset_hashes.items():
        _digest(digest, f"{name} dataset digest")
    inputs = comparison["payload"]["inputReportSha256s"]
    if not isinstance(inputs, dict):
        raise ReleaseError("comparison input report hashes must be an object")
    _exact_keys(inputs, {"base", "v2", "v3"}, "comparison input report hashes")
    for name, digest in inputs.items():
        _digest(digest, f"{name} input report digest")
    for kind, wrapper in receipts.items():
        _digest(wrapper["receiptSha256"], f"{kind} receipt digest")


def verify_report(report: dict[str, Any], label: str) -> str:
    declared = _digest(report.get("reportSha256"), f"{label} report digest")
    unsigned = dict(report)
    unsigned.pop("reportSha256")
    if sha256_json(unsigned) != declared:
        raise ReleaseError(f"{label} report self-digest differs")
    if report.get("publicationEligible") is not False:
        raise ReleaseError(f"{label} report crossed the publication boundary")
    return declared


def adapter_inventory(
    directory: Path,
) -> tuple[str, list[dict[str, Any]], dict[str, bytes]]:
    directory = _absolute_lexical(directory)
    directory_before = _assert_existing_chain(
        directory, "adapter", leaf_directory=True
    )
    combined = hashlib.sha256()
    files: list[dict[str, Any]] = []
    snapshot: dict[str, bytes] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        name = path.name
        if name not in ALLOWED_ADAPTER_FILES or Path(name).as_posix() != name:
            raise ReleaseError(f"adapter artifact file is not allowlisted: {name}")
        data = _bounded_regular_file(
            path, f"adapter artifact {name}", MAX_ADAPTER_FILE_BYTES
        )
        if name.endswith(".json"):
            strict_json_bytes(data, f"adapter {name}")
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        snapshot[name] = data
        files.append({"path": name, "bytes": len(data), "sha256": sha256_bytes(data)})
    observed = {item["path"] for item in files}
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(observed):
        raise ReleaseError("adapter omits required config or SafeTensors weights")
    directory_after = os.lstat(directory)
    if (
        _is_reparse_stat(directory_after)
        or not os.path.samestat(directory_before, directory_after)
        or directory_after.st_mtime_ns != directory_before.st_mtime_ns
    ):
        raise ReleaseError("adapter directory changed during inventory")
    return combined.hexdigest(), files, snapshot


def _report_source(report: Mapping[str, Any]) -> Any:
    source = report.get("source")
    return source.get("revision") if isinstance(source, dict) else None


def validate_adapter_config(
    candidate: Mapping[str, Any], adapter_config_bytes: bytes
) -> None:
    config = strict_json_bytes(adapter_config_bytes, "adapter config")
    implementation = candidate.get("actual_training_base")
    recipe = candidate.get("training_recipe")
    if not isinstance(implementation, dict) or not isinstance(recipe, dict):
        raise ReleaseError("candidate adapter binding contract is absent")
    required = {
        "base_model_name_or_path": implementation.get("repo_id"),
        "revision": implementation.get("revision"),
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": recipe.get("lora_r"),
        "lora_alpha": recipe.get("lora_alpha"),
        "bias": "none",
        "inference_mode": True,
        "use_rslora": False,
    }
    if HEX_40.fullmatch(required["revision"] or "") is None:
        raise ReleaseError("candidate implementation revision is invalid")
    for key, expected in required.items():
        _exact_typed(config.get(key), expected, f"adapter config {key}")
    dropout = config.get("lora_dropout")
    expected_dropout = recipe.get("lora_dropout")
    if (
        type(dropout) not in (int, float)
        or type(expected_dropout) not in (int, float)
        or dropout != expected_dropout
    ):
        raise ReleaseError("adapter config lora_dropout differs")
    target_modules = config.get("target_modules")
    if isinstance(target_modules, str):
        if not target_modules or target_modules.strip() != target_modules:
            raise ReleaseError("adapter config target_modules string is malformed")
    elif isinstance(target_modules, list):
        if not target_modules:
            raise ReleaseError("adapter config target_modules list is empty")
        if any(
            not isinstance(module, str)
            or not module
            or module.strip() != module
            for module in target_modules
        ):
            raise ReleaseError("adapter config target_modules list is malformed")
        if len(set(target_modules)) != len(target_modules):
            raise ReleaseError("adapter config target_modules list contains duplicates")
    else:
        raise ReleaseError("adapter config target_modules is absent or malformed")


def validate_trust_policy(policy: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "candidateId",
        "algorithm",
        "keyId",
        "publicKeyFingerprintSha256",
        "usage",
        "state",
    }
    _exact_keys(policy, expected_keys, "receipt signing trust policy")
    required = {
        "schema": TRUST_POLICY_SCHEMA,
        "candidateId": CANDIDATE_ID,
        "algorithm": "Ed25519",
        "usage": TRUST_POLICY_USAGE,
        "state": "ACTIVE",
    }
    for key, expected in required.items():
        _exact_typed(policy.get(key), expected, f"receipt signing trust policy {key}")
    key_id = policy.get("keyId")
    if (
        not isinstance(key_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", key_id) is None
    ):
        raise ReleaseError("receipt signing trust policy keyId is malformed")
    fingerprint = _digest(
        policy.get("publicKeyFingerprintSha256"),
        "receipt signing trust policy public key fingerprint",
    )
    return {
        "policySchema": TRUST_POLICY_SCHEMA,
        "policySha256": sha256_json(policy),
        "candidateId": CANDIDATE_ID,
        "algorithm": "Ed25519",
        "keyId": key_id,
        "publicKeyFingerprintSha256": fingerprint,
        "usage": TRUST_POLICY_USAGE,
        "state": "ACTIVE",
    }


def bind_reports_and_adapter(
    candidate: dict[str, Any],
    source_revision: str,
    receipts: Mapping[str, dict[str, Any]],
    reports: Mapping[str, dict[str, Any]],
    adapter_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, bytes]]:
    if set(reports) != set(REPORT_ARGUMENTS):
        raise ReleaseError("the exact measured report set is required")
    digests = {name: verify_report(report, name) for name, report in reports.items()}
    training_payload = receipts["TRAINING"]["payload"]
    evaluation_payload = receipts["EVALUATION"]["payload"]
    comparison_payload = receipts["COMPARISON"]["payload"]
    child = reports["childTraining"]
    supervisor = reports["supervisor"]
    dev = reports["devEvaluation"]
    test = reports["testEvaluation"]
    base = reports["baseTestEvaluation"]
    v2 = reports["v2TestEvaluation"]
    comparison = reports["comparison"]
    run_id = training_payload["runId"]

    required_child = {
        "schema": "szl.frontier-training-run/v3",
        "candidateId": CANDIDATE_ID,
        "supervisorRunId": run_id,
        "state": "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
        "runKind": "FULL",
        "qualificationEligible": True,
        "receiptEligible": False,
        "publicationEligible": False,
    }
    for key, expected in required_child.items():
        _exact_typed(child.get(key), expected, f"child training report {key}")
    if _report_source(child) != source_revision:
        raise ReleaseError("child training source revision differs")
    if digests["childTraining"] != training_payload["childReportSha256"]:
        raise ReleaseError("authenticated child training report digest differs")

    required_supervisor = {
        "schema": "szl.frontier-training-supervisor/v1",
        "candidateId": CANDIDATE_ID,
        "runId": run_id,
        "runKind": "FULL",
        "state": "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
        "primaryCause": "SUCCESS",
        "localEvaluationInputBindingSatisfied": True,
        "authenticatedSupervisorEnvelopePresent": False,
        "receiptEligible": False,
        "publicationEligible": False,
    }
    for key, expected in required_supervisor.items():
        _exact_typed(supervisor.get(key), expected, f"supervisor report {key}")
    if _report_source(supervisor) != source_revision:
        raise ReleaseError("supervisor source revision differs")
    if digests["supervisor"] != training_payload["supervisorReportSha256"]:
        raise ReleaseError("authenticated supervisor report digest differs")

    aggregate, local_files, adapter_bytes = adapter_inventory(adapter_dir)
    if aggregate != training_payload["adapterAggregateSha256"]:
        raise ReleaseError("local adapter aggregate differs from training receipt")
    if evaluation_payload["adapterAggregateSha256"] != aggregate:
        raise ReleaseError("evaluation adapter binding differs")
    child_adapter = child.get("adapter")
    if not isinstance(child_adapter, dict) or child_adapter.get("aggregateSha256") != aggregate:
        raise ReleaseError("child training adapter aggregate differs")
    supervisor_adapter = supervisor.get("adapter")
    if not isinstance(supervisor_adapter, dict):
        raise ReleaseError("supervisor adapter evidence is absent")
    required_adapter = {
        "aggregateSha256": aggregate,
        "matchesTrainingReport": True,
        "safeTensorsParsed": True,
        "allowlistedFilesOnly": True,
        "symlinksAbsent": True,
    }
    for key, expected in required_adapter.items():
        _exact_typed(
            supervisor_adapter.get(key), expected, f"supervisor adapter {key}"
        )
    if sha256_json(supervisor_adapter) != training_payload["adapterManifestSha256"]:
        raise ReleaseError("authenticated adapter manifest digest differs")
    attested_files = supervisor_adapter.get("files")
    if not isinstance(attested_files, list):
        raise ReleaseError("supervisor adapter manifest is absent")
    child_files = child_adapter.get("files")
    if child_files != attested_files:
        raise ReleaseError("child and supervisor adapter manifests differ")
    by_path: dict[str, dict[str, Any]] = {}
    attested_paths: list[str] = []
    for item in attested_files:
        if not isinstance(item, dict):
            raise ReleaseError("attested adapter manifest entry is malformed")
        path = item.get("path")
        if not isinstance(path, str) or path in by_path or path not in ALLOWED_ADAPTER_FILES:
            raise ReleaseError("attested adapter manifest path is duplicate or invalid")
        expected_keys = {"path", "bytes", "sha256"}
        if path.endswith(".json"):
            expected_keys.add("jsonKeys")
        elif path.endswith(".safetensors"):
            expected_keys.add("tensorCount")
        _exact_keys(item, expected_keys, f"adapter manifest entry {path}")
        _exact_nonnegative_int(item["bytes"], f"adapter bytes for {path}")
        _digest(item["sha256"], f"adapter digest for {path}")
        if path.endswith(".json"):
            _exact_nonnegative_int(item["jsonKeys"], f"adapter JSON key count for {path}")
        elif path.endswith(".safetensors"):
            tensor_count = _exact_nonnegative_int(
                item["tensorCount"], f"adapter tensor count for {path}"
            )
            if tensor_count == 0:
                raise ReleaseError(f"adapter tensor count for {path} must be positive")
        by_path[path] = item
        attested_paths.append(path)
    if attested_paths != sorted(attested_paths):
        raise ReleaseError("attested adapter manifest paths are not sorted")
    if set(by_path) != {item["path"] for item in local_files}:
        raise ReleaseError("local and attested adapter file rosters differ")
    for item in local_files:
        expected = by_path[item["path"]]
        if expected["bytes"] != item["bytes"] or expected["sha256"] != item["sha256"]:
            raise ReleaseError(f"adapter bytes differ for {item['path']}")
    validate_adapter_config(candidate, adapter_bytes["adapter_config.json"])

    for label, report, split in (
        ("devEvaluation", dev, "DEV"),
        ("testEvaluation", test, "TEST"),
    ):
        required = {
            "schema": "szl.frontier-eval-run/v3",
            "candidateId": CANDIDATE_ID,
            "modelKind": "v3",
            "split": split,
            "state": "MEASURED_EVALUATION_COMPLETED_UNATTESTED",
            "absoluteGatePassed": True,
            "receiptEligible": False,
            "publicationEligible": False,
        }
        for key, expected in required.items():
            _exact_typed(report.get(key), expected, f"{label} report {key}")
        if _report_source(report) != source_revision:
            raise ReleaseError(f"{label} source revision differs")
        if report.get("trainingReportSha256") != digests["childTraining"]:
            raise ReleaseError(f"{label} training report binding differs")
        linkage = report.get("supervisionLinkage")
        model = report.get("model")
        if not isinstance(linkage, dict) or linkage.get("runId") != run_id:
            raise ReleaseError(f"{label} supervisor run binding differs")
        if not isinstance(model, dict) or model.get("adapterAggregateSha256") != aggregate:
            raise ReleaseError(f"{label} adapter binding differs")
    if digests["devEvaluation"] != evaluation_payload["devReportSha256"]:
        raise ReleaseError("authenticated dev report digest differs")
    if digests["testEvaluation"] != evaluation_payload["testReportSha256"]:
        raise ReleaseError("authenticated test report digest differs")

    input_reports = {"base": base, "v2": v2, "v3": test}
    input_digests = {
        "base": digests["baseTestEvaluation"],
        "v2": digests["v2TestEvaluation"],
        "v3": digests["testEvaluation"],
    }
    for kind, report in input_reports.items():
        if report.get("schema") != "szl.frontier-eval-run/v3" or report.get("modelKind") != kind:
            raise ReleaseError(f"{kind} comparison input identity differs")
        if report.get("split") != "TEST" or _report_source(report) != source_revision:
            raise ReleaseError(f"{kind} comparison input benchmark binding differs")
        if report.get("state") != "MEASURED_EVALUATION_COMPLETED_UNATTESTED":
            raise ReleaseError(f"{kind} comparison input is not measured and complete")
    if comparison_payload["inputReportSha256s"] != input_digests:
        raise ReleaseError("authenticated comparison input report digests differ")
    if comparison.get("inputReports") != input_digests:
        raise ReleaseError("comparison report input bindings differ")
    required_comparison = {
        "schema": "szl.frontier-comparison/v2",
        "candidateId": CANDIDATE_ID,
        "state": "UNAUTHENTICATED_COMPARISON_CRITERIA_SATISFIED",
        "sourceRevision": source_revision,
        "absoluteGatePassed": True,
        "authoritySafetyNoRegression": True,
        "comparisonCriteriaSatisfied": True,
        "receiptEligible": False,
        "publicationEligible": False,
    }
    for key, expected in required_comparison.items():
        _exact_typed(comparison.get(key), expected, f"comparison report {key}")
    _exact_nonnegative_int(
        comparison.get("strictCaseImprovementOverV2"),
        "strict-case improvement over v2",
    )
    _exact_nonnegative_int(
        comparison.get("requiredStrictCaseImprovementOverV2"),
        "required strict-case improvement over v2",
    )
    if digests["comparison"] != comparison_payload["comparisonReportSha256"]:
        raise ReleaseError("authenticated comparison report digest differs")
    return local_files, digests, adapter_bytes


def render_model_card(
    candidate: Mapping[str, Any],
    source_revision: str,
    run_id: str,
    receipts: Mapping[str, dict[str, Any]],
    reports: Mapping[str, dict[str, Any]],
) -> str:
    implementation = candidate["actual_training_base"]
    upstream = candidate["upstream_lineage"]
    predecessor = candidate["predecessor"]
    training = receipts["TRAINING"]["payload"]
    comparison = reports["comparison"]
    try:
        measured = json.dumps(
            comparison.get("recomputedResults", {}),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReleaseError("comparison results are not finite JSON") from exc
    return f"""---
license: apache-2.0
base_model: {implementation['repo_id']}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- receipt-agent
- evidence-bound
---

# SZL ReceiptAgent Qwen3.5 0.8B v3

This card was generated from an offline, authenticated local evidence chain for
the declared target `{TARGET_REPO_ID}`. It is a release *preparation* artifact:
it does not prove that any repository was published, that immutable hub bytes
were read back, that a hosted runtime loaded them, or that a third party
validated the model.

## Evidence state

- Candidate: `{CANDIDATE_ID}`
- Exact source revision: `{source_revision}`
- Exact supervised run: `{run_id}`
- Adapter aggregate SHA-256: `{training['adapterAggregateSha256']}`
- Training receipt: `{receipts['TRAINING']['receiptSha256']}`
- Evaluation receipt: `{receipts['EVALUATION']['receiptSha256']}`
- Comparison receipt: `{receipts['COMPARISON']['receiptSha256']}`
- Authenticated evidence chain valid: `true`
- Local receipt eligible: `true`
- Publication eligible: `false`
- Deployment/runtime validated: `false`
- Third-party validated: `false`

## Measured comparison

The frozen comparison revalidated one project-authored public test suite. The
stored results below are measurements bound by digest; they are not a blind or
independent benchmark and do not authorize publication by themselves.

```json
{measured}
```

Strict-case improvement over v2: `{comparison['strictCaseImprovementOverV2']}`
(required: `{comparison['requiredStrictCaseImprovementOverV2']}`). Authority
safety no-regression: `{str(comparison['authoritySafetyNoRegression']).lower()}`.

## Lineage and format

This artifact is a LoRA/PEFT adapter and requires the pinned implementation base
`{implementation['repo_id']}` at revision `{implementation['revision']}`. The
declared upstream lineage is `{upstream['repo_id']}` at revision
`{upstream['revision']}`; byte equivalence with that upstream repository was not
verified. The frozen v2 repository `{predecessor['repo_id']}` at revision
`{predecessor['release_revision']}` was a comparator, not weight initialization.

Training used project-authored synthetic policy/schema rows only. The declared
recipe used LoRA rank `{candidate['training_recipe']['lora_r']}`, alpha
`{candidate['training_recipe']['lora_alpha']}`, seed
`{candidate['training_recipe']['seed']}`, and
`{candidate['training_recipe']['full_optimizer_steps']}` optimizer steps.

## Intended use

Receipt-oriented structured assistance under an external authority and evidence
policy. Outputs require application-side schema validation and human or policy
review. This adapter is not an autonomous approver, deployment controller, or
source of authority.

## Limitations and claim boundary

- The benchmark is project-authored, public, and preregistered; it is not blind
  testing or independent certification.
- Local authenticated receipts bind source, run, reports, adapter bytes, and
  benchmark inputs. They do not establish immutable Hugging Face readback.
- No hosted inference, deployment health, production traffic, or third-party
  runtime receipt is included.
- The 4-bit Unsloth implementation base is pinned; upstream byte equivalence is
  explicitly unverified.
- The adapter can inherit base-model limitations and may fail outside the
  measured receipt-policy distribution.
- `publicationEligible` remains `false` until a separate publisher verifies an
  immutable hub revision and a separate runtime witness verifies those exact
  bytes. This packet cannot flip that state.
"""


def build_release_packet(
    *,
    candidate: dict[str, Any],
    source_revision: str,
    receipts: Mapping[str, dict[str, Any]],
    reports: Mapping[str, dict[str, Any]],
    adapter_dir: Path,
    trusted_public_key: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if HEX_40.fullmatch(source_revision) is None:
        raise ReleaseError("source revision must be 40 lowercase hex characters")
    if candidate.get("candidate_id") != CANDIDATE_ID:
        raise ReleaseError("candidate identity differs")
    if candidate.get("target_repo_id") != TARGET_REPO_ID:
        raise ReleaseError("declared target repository differs")
    trust_policy = load_committed_trust_policy(source_revision)
    trust_identity = validate_trust_policy(trust_policy)
    validate_receipt_chain_shape(receipts)
    if any(receipts[k]["sourceRevision"] != source_revision for k in receipts):
        raise ReleaseError("requested source revision differs from receipt chain")
    chain = evidence.verify_chain(
        receipts["TRAINING"],
        receipts["EVALUATION"],
        receipts["COMPARISON"],
        trusted_public_key=trusted_public_key,
    )
    expected_chain = {
        "candidateId": CANDIDATE_ID,
        "sourceRevision": source_revision,
        "trainingReceiptSha256": receipts["TRAINING"]["receiptSha256"],
        "evaluationReceiptSha256": receipts["EVALUATION"]["receiptSha256"],
        "comparisonReceiptSha256": receipts["COMPARISON"]["receiptSha256"],
        "authenticatedEvidenceChainValid": True,
        "receiptEligible": True,
        "publicationEligible": False,
    }
    _exact_keys(
        chain,
        set(expected_chain) | {"keyId"},
        "authenticated chain summary",
    )
    if not isinstance(chain["keyId"], str) or not chain["keyId"]:
        raise ReleaseError("authenticated chain summary keyId differs")
    for key, expected in expected_chain.items():
        _exact_typed(
            chain.get(key), expected, f"authenticated chain summary {key}"
        )
    if chain["keyId"] != trust_identity["keyId"]:
        raise ReleaseError("receipt key ID is not approved by the source trust policy")
    receipt_fingerprint = receipts["TRAINING"]["authentication"][
        "publicKeyFingerprintSha256"
    ]
    if receipt_fingerprint != trust_identity["publicKeyFingerprintSha256"]:
        raise ReleaseError(
            "receipt public key is not approved by the source trust policy"
        )
    local_files, report_digests, adapter_bytes = bind_reports_and_adapter(
        candidate, source_revision, receipts, reports, adapter_dir
    )
    run_id = receipts["TRAINING"]["payload"]["runId"]
    card = render_model_card(candidate, source_revision, run_id, receipts, reports)
    public_files = {
        name: data for name, data in adapter_bytes.items() if name != "README.md"
    }
    public_roles = {name: "ADAPTER_ARTIFACT" for name in public_files}
    local_files_bytes: dict[str, bytes] = {}
    local_roles: dict[str, str] = {}
    if "README.md" in adapter_bytes:
        local_files_bytes["adapter-source-readme.md"] = adapter_bytes["README.md"]
        local_roles["adapter-source-readme.md"] = "ADAPTER_SOURCE_README"
    public_files["README.md"] = card.encode("utf-8")
    public_roles["README.md"] = "MEASURED_MODEL_CARD"
    for kind, (name, role) in RECEIPT_OUTPUTS.items():
        public_files[name] = (canonical_json(receipts[kind]) + "\n").encode(
            "utf-8"
        )
        public_roles[name] = role
    for report_key, (name, role) in REPORT_OUTPUTS.items():
        local_files_bytes[name] = (
            canonical_json(reports[report_key]) + "\n"
        ).encode(
            "utf-8"
        )
        local_roles[name] = role
    intended = [
        {
            "path": name,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "role": public_roles[name],
        }
        for name, data in public_files.items()
    ]
    intended.sort(key=lambda item: item["path"])
    local_evidence = [
        {
            "path": name,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "role": local_roles[name],
            "disposition": "NOT_FOR_PUBLICATION",
        }
        for name, data in local_files_bytes.items()
    ]
    local_evidence.sort(key=lambda item: item["path"])
    packet_files = dict(public_files)
    packet_files.update(local_files_bytes)
    training_payload = receipts["TRAINING"]["payload"]
    comparison = reports["comparison"]
    manifest: dict[str, Any] = {
        "schema": "szl.receiptagent-v3-release-manifest/v1",
        "candidateId": CANDIDATE_ID,
        "state": "OFFLINE_RELEASE_PACKET_PREPARED_NOT_PUBLISHED",
        "targetRepository": {
            "repoId": TARGET_REPO_ID,
            "repoType": "model",
            "private": False,
            "declaredOnly": True,
            "immutableRevision": None,
            "immutableReadbackVerified": False,
        },
        "sourceRevision": source_revision,
        "runId": run_id,
        "authenticatedEvidenceChainValid": True,
        "receiptSigningTrustPolicy": trust_identity,
        "receiptEligible": True,
        "publicationEligible": False,
        "deploymentEligible": False,
        "runtimeValidated": False,
        "thirdPartyValidated": False,
        "lineage": {
            "adapterFormat": "LORA_PEFT_SAFETENSORS",
            "implementationBase": candidate["actual_training_base"],
            "declaredUpstream": candidate["upstream_lineage"],
            "predecessorComparator": candidate["predecessor"],
        },
        "adapter": {
            "aggregateSha256": training_payload["adapterAggregateSha256"],
            "manifestSha256": training_payload["adapterManifestSha256"],
            "sourceFiles": local_files,
            "sourceReadmeDisposition": (
                "LOCAL_EVIDENCE_ONLY_NOT_FOR_PUBLICATION"
                if "README.md" in adapter_bytes
                else "SOURCE_README_ABSENT"
            ),
        },
        "benchmark": {
            "datasetHashes": training_payload["datasetHashes"],
            "comparisonReportSha256": report_digests["comparison"],
            "inputReportSha256s": receipts["COMPARISON"]["payload"][
                "inputReportSha256s"
            ],
            "protocolSha256": comparison["protocolSha256"],
            "comparisonCriteriaSatisfied": True,
            "strictCaseImprovementOverV2": comparison[
                "strictCaseImprovementOverV2"
            ],
            "requiredStrictCaseImprovementOverV2": comparison[
                "requiredStrictCaseImprovementOverV2"
            ],
            "authoritySafetyNoRegression": True,
            "claimBoundary": "Project-authored public benchmark; not blind or independent certification.",
        },
        "reports": report_digests,
        "receipts": {
            kind.lower(): wrapper["receiptSha256"] for kind, wrapper in receipts.items()
        },
        "intendedRepositoryFiles": intended,
        "localEvidenceDisposition": "NOT_FOR_PUBLICATION",
        "localEvidenceFiles": local_evidence,
        "releaseManifestIsTargetRepositoryFile": True,
        "unmetPublicationGates": [
            "IMMUTABLE_HUGGING_FACE_REVISION_READBACK_ABSENT",
            "EXACT_HUB_FILE_INVENTORY_READBACK_ABSENT",
            "EXACT_REVISION_RUNTIME_LOAD_RECEIPT_ABSENT",
        ],
        "claimBoundary": (
            "Deterministic offline preparation only. The publisher must append the "
            "self-excluded release manifest after verifying this inventory. No "
            "publication, deployment, runtime health, third-party validation, or "
            "autonomy claim is authorized."
        ),
    }
    manifest["manifestSha256"] = sha256_json(manifest)
    validate_release_manifest(manifest)
    return manifest, packet_files


def validate_release_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ReleaseError("release manifest must be one JSON object")
    canonical_json(manifest)
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ModuleNotFoundError as exc:
        raise ReleaseError("jsonschema is required for release validation") from exc
    schema = strict_json_file(HERE / "release.schema.json", "release schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ReleaseError("release schema is invalid") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ReleaseError(f"release manifest schema differs at {location}")
    unsigned = dict(manifest)
    declared = unsigned.pop("manifestSha256", None)
    if declared != sha256_json(unsigned):
        raise ReleaseError("release manifest digest differs")


def _assert_safe_new_output(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ReleaseError("output directory must be an absolute traversal-free path")
    if os.path.lexists(path):
        raise ReleaseError("output directory must not already exist")
    parent = path.parent
    _assert_existing_chain(parent, "output parent", leaf_directory=True)


def _safe_flat_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ReleaseError(f"packet output path is not safe and flat: {name!r}")


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_packet(
    path: Path,
    manifest: dict[str, Any],
    packet_files: Mapping[str, bytes],
) -> None:
    _assert_safe_new_output(path)
    validate_release_manifest(manifest)
    intended = manifest.get("intendedRepositoryFiles")
    if not isinstance(intended, list):
        raise ReleaseError("intended repository file inventory is absent")
    local_evidence = manifest.get("localEvidenceFiles")
    if not isinstance(local_evidence, list):
        raise ReleaseError("local evidence file inventory is absent")
    public_expected: dict[str, dict[str, Any]] = {}
    for item in intended:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256", "role"}
        ):
            raise ReleaseError("intended repository file entry is malformed")
        name = item["path"]
        _safe_flat_name(name)
        if name == "release-manifest.json" or name in public_expected:
            raise ReleaseError(
                "manifest self-inventory or duplicate output is forbidden"
            )
        public_expected[name] = item
    local_expected: dict[str, dict[str, Any]] = {}
    for item in local_evidence:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"path", "bytes", "sha256", "role", "disposition"}
            or item["disposition"] != "NOT_FOR_PUBLICATION"
        ):
            raise ReleaseError("local evidence file entry is malformed")
        name = item["path"]
        _safe_flat_name(name)
        if (
            name == "release-manifest.json"
            or name in local_expected
            or name in public_expected
        ):
            raise ReleaseError("local evidence output is duplicate or public")
        local_expected[name] = item
    expected = dict(public_expected)
    expected.update(local_expected)
    if set(packet_files) != set(expected):
        raise ReleaseError(
            "packet file snapshot differs from public and local inventories"
        )
    for name, data in packet_files.items():
        _safe_flat_name(name)
        if not isinstance(data, bytes):
            raise ReleaseError(f"packet file bytes are invalid: {name}")
        item = expected[name]
        _exact_nonnegative_int(item["bytes"], f"packet byte count for {name}")
        if item["bytes"] != len(data) or item["sha256"] != sha256_bytes(data):
            raise ReleaseError(f"packet file snapshot digest differs: {name}")
    try:
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseError("release manifest is not finite JSON") from exc
    os.mkdir(path, 0o700)
    try:
        _assert_existing_chain(path, "packet output", leaf_directory=True)
        for name in sorted(packet_files):
            _assert_existing_chain(path, "packet output", leaf_directory=True)
            _write_exclusive(path / name, packet_files[name])
        _write_exclusive(path / "release-manifest.json", manifest_bytes)
    except Exception:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
        raise


def load_committed_json(source_revision: str, filename: str, label: str) -> dict[str, Any]:
    if HEX_40.fullmatch(source_revision) is None:
        raise ReleaseError("source revision must be 40 lowercase hex characters")
    if filename not in {"candidate.json", TRUST_POLICY_FILENAME}:
        raise ReleaseError("committed JSON filename is not allowlisted")
    result = subprocess.run(
        ["git", "show", f"{source_revision}:{RELATIVE}/{filename}"],
        cwd=REPOSITORY,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode != 0:
        raise ReleaseError(f"{label} bytes are unavailable at the exact source revision")
    return strict_json_bytes(result.stdout, label)


def load_committed_candidate(source_revision: str) -> dict[str, Any]:
    return load_committed_json(source_revision, "candidate.json", "committed candidate")


def load_committed_trust_policy(source_revision: str) -> dict[str, Any]:
    return load_committed_json(
        source_revision,
        TRUST_POLICY_FILENAME,
        "committed receipt signing trust policy",
    )


def _bounded_error(exc: Exception) -> str:
    message = str(exc).replace(str(REPOSITORY), "<REPOSITORY>")
    return re.sub(r"[A-Za-z]:\\[^\s]+", "<LOCAL_PATH>", message)[:500]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-receipt", type=Path, required=True)
    parser.add_argument("--comparison-receipt", type=Path, required=True)
    parser.add_argument("--trusted-public-key", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for option in REPORT_ARGUMENTS.values():
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    try:
        candidate = load_committed_candidate(args.source_commit)
        receipts = {
            "TRAINING": strict_json_file(args.training_receipt, "training receipt"),
            "EVALUATION": strict_json_file(args.evaluation_receipt, "evaluation receipt"),
            "COMPARISON": strict_json_file(args.comparison_receipt, "comparison receipt"),
        }
        reports = {
            key: strict_json_file(getattr(args, option.replace("-", "_")), key)
            for key, option in REPORT_ARGUMENTS.items()
        }

        manifest, packet_files = build_release_packet(
            candidate=candidate,
            source_revision=args.source_commit,
            receipts=receipts,
            reports=reports,
            adapter_dir=args.adapter_dir,
            trusted_public_key=args.trusted_public_key,
        )
        write_packet(args.output_dir, manifest, packet_files)
        print(json.dumps({
            "state": manifest["state"],
            "manifestSha256": manifest["manifestSha256"],
            "outputDirectory": str(args.output_dir),
            "publicationEligible": False,
        }, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - one bounded fail-closed CLI path
        print(json.dumps({
            "state": "RELEASE_PACKET_BLOCKED",
            "fatal": _bounded_error(exc),
            "publicationEligible": False,
            "deploymentEligible": False,
            "runtimeValidated": False,
        }, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
