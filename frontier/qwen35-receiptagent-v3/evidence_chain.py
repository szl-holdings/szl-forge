#!/usr/bin/env python3
"""Create and verify owner-authenticated ReceiptAgent v3 evidence receipts.

The module never creates measurements or key material.  It accepts measured
JSON reports and a caller-supplied Ed25519 key, validates the report bindings,
and signs a small canonical envelope.  Authentication is deliberately kept
separate from publication: every receipt and every verification result remains
``publicationEligible = false`` because Hub byte readback and a running-runtime
witness are outside this three-receipt chain.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


SCHEMA = "szl.receiptagent-v3-authenticated-receipt/v1"
CANDIDATE_ID = "SZL-ReceiptAgent-Qwen3.5-0.8B-v3"
KINDS = ("TRAINING", "EVALUATION", "COMPARISON")
TOP_LEVEL_KEYS = {
    "schema",
    "kind",
    "candidateId",
    "sourceRevision",
    "payload",
    "payloadSha256",
    "authentication",
    "receiptSha256",
}
AUTHENTICATION_KEYS = {
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
HEX_32 = re.compile(r"^[0-9a-f]{32}$")
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_KEY_BYTES = 64 * 1024
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class EvidenceError(RuntimeError):
    """A fail-closed evidence-chain error."""


def canonical_json(value: Any) -> str:
    """Return the repository's deterministic UTF-8 JSON representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"value is not canonical-JSON-compatible: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_reparse_point(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _checked_input_path(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, os.stat_result]:
    """Inspect an input path without following the final filesystem object.

    Directory symlinks and Windows junctions can occur in any path component,
    so each component is rejected when it is a symlink or carries the Windows
    reparse-point flag.
    POSIX final-component races receive an additional ``O_NOFOLLOW`` guard in
    :func:`_bounded_regular_file_read`.
    """

    source = Path(os.path.abspath(path))
    components = (source, *source.parents)
    final_metadata: os.stat_result | None = None
    for component in components:
        try:
            metadata = os.lstat(component)
        except OSError as exc:
            raise EvidenceError(
                f"cannot inspect {label}: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
            raise EvidenceError(f"{label} must not traverse a symlink or reparse point")
        if component == source:
            final_metadata = metadata
    if final_metadata is None or not stat.S_ISREG(final_metadata.st_mode):
        raise EvidenceError(f"{label} must be one regular file")
    return source, final_metadata


def _bounded_regular_file_read(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read one stable regular file through one descriptor after sizing it."""

    source, before = _checked_input_path(path, label=label)
    flags = os.O_RDONLY
    for optional_flag in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, optional_flag, 0)
    descriptor = -1
    try:
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse_point(opened):
            raise EvidenceError(f"{label} must be one regular non-reparse file")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise EvidenceError(f"{label} changed while it was opened")
        size = opened.st_size
        if size <= 0:
            raise EvidenceError(f"{label} is empty")
        if size > max_bytes:
            raise EvidenceError(f"{label} exceeds the {max_bytes}-byte input limit")
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        with handle:
            raw = handle.read(size + 1)
            after = os.fstat(handle.fileno())
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError(f"cannot read {label}: {type(exc).__name__}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        len(raw) != size
        or after.st_size != size
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        raise EvidenceError(f"{label} changed while it was read")
    return raw


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one bounded JSON object while rejecting duplicates and NaN values."""

    source = Path(path)
    raw = _bounded_regular_file_read(
        source,
        max_bytes=MAX_JSON_BYTES,
        label=f"evidence JSON {source}",
    )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError(f"cannot parse {source}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{source} must contain one JSON object")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be one object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise EvidenceError(f"{label} keys differ; missing={missing}, extra={extra}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _source_revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_40.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a lowercase 40-hex Git commit")
    return value


def _run_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_32.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be lowercase 32-hex")
    return value


def _require(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise EvidenceError(f"{label} differs: observed {value!r}, expected {expected!r}")


def verify_report_digest(report: Mapping[str, Any], label: str) -> str:
    claimed = _sha(report.get("reportSha256"), f"{label} reportSha256")
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    measured = sha256_json(unsigned)
    _require(claimed, measured, f"{label} self-digest")
    return claimed


def _source_from_report(report: Mapping[str, Any], label: str) -> str:
    source = _mapping(report.get("source"), f"{label} source")
    return _source_revision(source.get("revision"), f"{label} source revision")


def _public_der(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _read_key_material(value: bytes | str | Path, label: str) -> bytes:
    if isinstance(value, bytes):
        if not value:
            raise EvidenceError(f"{label} is empty")
        if len(value) > MAX_KEY_BYTES:
            raise EvidenceError(
                f"{label} exceeds the {MAX_KEY_BYTES}-byte input limit"
            )
        return value
    return _bounded_regular_file_read(
        value,
        max_bytes=MAX_KEY_BYTES,
        label=label,
    )


def _load_private_key(
    value: Ed25519PrivateKey | bytes | str | Path,
) -> Ed25519PrivateKey:
    if isinstance(value, Ed25519PrivateKey):
        return value
    material = _read_key_material(value, "private key")
    loaders = (
        lambda: serialization.load_pem_private_key(material, password=None),
        lambda: serialization.load_der_private_key(material, password=None),
        lambda: Ed25519PrivateKey.from_private_bytes(material),
    )
    for loader in loaders:
        try:
            key = loader()
        except (TypeError, ValueError):
            continue
        if isinstance(key, Ed25519PrivateKey):
            return key
    raise EvidenceError("private key is not an unencrypted Ed25519 PEM, DER, or raw key")


def _load_public_key(
    value: Ed25519PublicKey | bytes | str | Path,
) -> Ed25519PublicKey:
    if isinstance(value, Ed25519PublicKey):
        return value
    material = _read_key_material(value, "trusted public key")
    loaders = (
        lambda: serialization.load_pem_public_key(material),
        lambda: serialization.load_der_public_key(material),
        lambda: Ed25519PublicKey.from_public_bytes(material),
    )
    for loader in loaders:
        try:
            key = loader()
        except (TypeError, ValueError):
            continue
        if isinstance(key, Ed25519PublicKey):
            return key
    raise EvidenceError("trusted key is not an Ed25519 PEM, DER, or raw public key")


def _embedded_public_key(authentication: Mapping[str, Any]) -> Ed25519PublicKey:
    try:
        der = base64.b64decode(
            authentication["publicKeySpkiBase64"], validate=True
        )
        key = serialization.load_der_public_key(der)
    except Exception as exc:  # noqa: BLE001 - normalize cryptographic parser failures
        raise EvidenceError("embedded public key is not valid base64 DER SPKI") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise EvidenceError("embedded public key is not Ed25519")
    return key


def _signature_document(wrapper: Mapping[str, Any]) -> dict[str, Any]:
    authentication = dict(_mapping(wrapper.get("authentication"), "authentication"))
    authentication.pop("signatureBase64", None)
    return {
        "schema": wrapper.get("schema"),
        "kind": wrapper.get("kind"),
        "candidateId": wrapper.get("candidateId"),
        "sourceRevision": wrapper.get("sourceRevision"),
        "payload": wrapper.get("payload"),
        "payloadSha256": wrapper.get("payloadSha256"),
        "authentication": authentication,
    }


def _mint(
    kind: str,
    payload: dict[str, Any],
    *,
    source_revision: str,
    private_key: Ed25519PrivateKey | bytes | str | Path,
    key_id: str,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise EvidenceError(f"unsupported receipt kind: {kind}")
    _source_revision(source_revision, "receipt source revision")
    if not isinstance(key_id, str) or KEY_ID.fullmatch(key_id) is None:
        raise EvidenceError("key ID must be 3-128 safe identifier characters")
    _exact_keys(payload, PAYLOAD_KEYS[kind], f"{kind} payload")
    _require(payload.get("publicationEligible"), False, "receipt publication flag")
    key = _load_private_key(private_key)
    public_der = _public_der(key.public_key())
    authentication = {
        "algorithm": "Ed25519",
        "keyId": key_id,
        "publicKeyFingerprintSha256": sha256_bytes(public_der),
        "publicKeySpkiBase64": base64.b64encode(public_der).decode("ascii"),
    }
    wrapper: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": kind,
        "candidateId": CANDIDATE_ID,
        "sourceRevision": source_revision,
        "payload": payload,
        "payloadSha256": sha256_json(payload),
        "authentication": authentication,
    }
    signature = key.sign(canonical_json(_signature_document(wrapper)).encode("utf-8"))
    authentication["signatureBase64"] = base64.b64encode(signature).decode("ascii")
    wrapper["receiptSha256"] = sha256_json(wrapper)
    return wrapper


def _training_payload(
    child_report: dict[str, Any],
    supervisor_report: dict[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    child_sha = verify_report_digest(child_report, "child training report")
    supervisor_sha = verify_report_digest(supervisor_report, "supervisor report")
    for label, report in (("child", child_report), ("supervisor", supervisor_report)):
        _require(report.get("candidateId"), CANDIDATE_ID, f"{label} candidate")
        _require(_source_from_report(report, label), source_revision, f"{label} source")
        _require(report.get("receiptEligible"), False, f"{label} receipt flag")
        _require(report.get("publicationEligible"), False, f"{label} publication flag")
    _require(child_report.get("schema"), "szl.frontier-training-run/v3", "child schema")
    _require(
        child_report.get("state"),
        "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
        "child state",
    )
    _require(child_report.get("runKind"), "FULL", "child run kind")
    _require(
        supervisor_report.get("schema"),
        "szl.frontier-training-supervisor/v1",
        "supervisor schema",
    )
    _require(
        supervisor_report.get("state"),
        "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
        "supervisor state",
    )
    for key, expected in {
        "runKind": "FULL",
        "primaryCause": "SUCCESS",
        "localEvaluationInputBindingSatisfied": True,
        "authenticatedSupervisorEnvelopePresent": False,
    }.items():
        _require(supervisor_report.get(key), expected, f"supervisor {key}")
    run_id = _run_id(supervisor_report.get("runId"), "supervisor run ID")
    _require(child_report.get("supervisorRunId"), run_id, "child/supervisor run ID")
    training_binding = _mapping(
        supervisor_report.get("trainingReport"), "supervisor training report binding"
    )
    _require(
        training_binding.get("canonicalReportSha256"),
        child_sha,
        "supervisor child report digest",
    )
    child_adapter = _mapping(child_report.get("adapter"), "child adapter")
    supervisor_adapter = _mapping(supervisor_report.get("adapter"), "supervisor adapter")
    adapter_sha = _sha(
        child_adapter.get("aggregateSha256"), "child adapter aggregate"
    )
    _require(
        supervisor_adapter.get("aggregateSha256"),
        adapter_sha,
        "supervisor adapter aggregate",
    )
    _require(supervisor_adapter.get("matchesTrainingReport"), True, "adapter binding")
    source_bundle = _mapping(child_report.get("sourceBundle"), "child source bundle")
    held_out = _mapping(source_bundle.get("heldOutCommitments"), "held-out commitments")
    _exact_keys(held_out, {"dev.jsonl", "test.jsonl"}, "held-out commitments")
    dataset_hashes = {
        "train": _sha(source_bundle.get("trainSha256"), "train dataset digest"),
        "dev": _sha(
            _mapping(held_out["dev.jsonl"], "dev commitment").get("sha256"),
            "dev dataset digest",
        ),
        "test": _sha(
            _mapping(held_out["test.jsonl"], "test commitment").get("sha256"),
            "test dataset digest",
        ),
    }
    identities = _mapping(supervisor_report.get("identities"), "component identities")
    containment = _mapping(supervisor_report.get("containment"), "containment identity")
    telemetry = _mapping(supervisor_report.get("telemetry"), "supervisor telemetry")
    gpu = _mapping(child_report.get("gpu"), "child GPU identity")
    runtime = _mapping(child_report.get("runtimePackages"), "runtime identity")
    return {
        "runId": run_id,
        "childReportSha256": child_sha,
        "supervisorReportSha256": supervisor_sha,
        "adapterAggregateSha256": adapter_sha,
        "adapterManifestSha256": sha256_json(supervisor_adapter),
        "datasetHashes": dataset_hashes,
        "gpuIdentitySha256": sha256_json({"child": gpu, "supervisor": telemetry}),
        "containmentIdentitySha256": sha256_json(containment),
        "runtimeIdentitySha256": sha256_json(runtime),
        "componentIdentitiesSha256": sha256_json(identities),
        "previousReceiptSha256": None,
        "publicationEligible": False,
    }


def mint_training_receipt(
    child_report: dict[str, Any],
    supervisor_report: dict[str, Any],
    *,
    source_revision: str,
    private_key: Ed25519PrivateKey | bytes | str | Path,
    key_id: str,
) -> dict[str, Any]:
    """Validate measured child/supervisor reports and sign the training receipt."""

    payload = _training_payload(child_report, supervisor_report, source_revision)
    return _mint(
        "TRAINING",
        payload,
        source_revision=source_revision,
        private_key=private_key,
        key_id=key_id,
    )


def _validate_evaluation_report(
    report: dict[str, Any],
    *,
    split: str,
    source_revision: str,
    training_payload: Mapping[str, Any],
) -> str:
    report_sha = verify_report_digest(report, f"{split} evaluation report")
    for key, expected in {
        "schema": "szl.frontier-eval-run/v3",
        "candidateId": CANDIDATE_ID,
        "modelKind": "v3",
        "split": split,
        "state": "MEASURED_EVALUATION_COMPLETED_UNATTESTED",
        "absoluteGatePassed": True,
        "comparisonEligible": False,
        "authenticatedEvaluationEnvelopePresent": False,
        "receiptEligible": False,
        "publicationEligible": False,
    }.items():
        _require(report.get(key), expected, f"{split} evaluation {key}")
    _require(
        _source_from_report(report, f"{split} evaluation"),
        source_revision,
        f"{split} evaluation source",
    )
    _require(
        report.get("trainingReportSha256"),
        training_payload["childReportSha256"],
        f"{split} training report binding",
    )
    model = _mapping(report.get("model"), f"{split} model identity")
    _require(
        model.get("adapterAggregateSha256"),
        training_payload["adapterAggregateSha256"],
        f"{split} adapter binding",
    )
    linkage = _mapping(report.get("supervisionLinkage"), f"{split} supervision linkage")
    _require(linkage.get("runId"), training_payload["runId"], f"{split} run ID")
    _require(
        linkage.get("reportSha256"),
        training_payload["supervisorReportSha256"],
        f"{split} supervisor receipt binding",
    )
    _require(
        linkage.get("adapterAggregateSha256"),
        training_payload["adapterAggregateSha256"],
        f"{split} supervisor adapter binding",
    )
    _require(linkage.get("sourceRevision"), source_revision, f"{split} linkage source")
    return report_sha


def mint_evaluation_receipt(
    dev_report: dict[str, Any],
    test_report: dict[str, Any],
    training_receipt: dict[str, Any],
    *,
    private_key: Ed25519PrivateKey | bytes | str | Path,
    key_id: str,
) -> dict[str, Any]:
    """Bind DEV and TEST measurements to an authenticated training receipt."""

    key = _load_private_key(private_key)
    verify_receipt(
        training_receipt,
        trusted_public_key=key.public_key(),
        expected_kind="TRAINING",
    )
    source_revision = training_receipt["sourceRevision"]
    training_payload = _mapping(training_receipt["payload"], "training payload")
    dev_sha = _validate_evaluation_report(
        dev_report,
        split="DEV",
        source_revision=source_revision,
        training_payload=training_payload,
    )
    test_sha = _validate_evaluation_report(
        test_report,
        split="TEST",
        source_revision=source_revision,
        training_payload=training_payload,
    )
    payload = {
        "runId": training_payload["runId"],
        "devReportSha256": dev_sha,
        "testReportSha256": test_sha,
        "adapterAggregateSha256": training_payload["adapterAggregateSha256"],
        "trainingReceiptSha256": training_receipt["receiptSha256"],
        "previousReceiptSha256": training_receipt["receiptSha256"],
        "publicationEligible": False,
    }
    return _mint(
        "EVALUATION",
        payload,
        source_revision=source_revision,
        private_key=key,
        key_id=key_id,
    )


def mint_comparison_receipt(
    comparison_report: dict[str, Any],
    evaluation_receipt: dict[str, Any],
    *,
    private_key: Ed25519PrivateKey | bytes | str | Path,
    key_id: str,
) -> dict[str, Any]:
    """Validate comparison criteria and bind them after the evaluation receipt."""

    key = _load_private_key(private_key)
    verify_receipt(
        evaluation_receipt,
        trusted_public_key=key.public_key(),
        expected_kind="EVALUATION",
    )
    source_revision = evaluation_receipt["sourceRevision"]
    report_sha = verify_report_digest(comparison_report, "comparison report")
    for field, expected in {
        "schema": "szl.frontier-comparison/v2",
        "candidateId": CANDIDATE_ID,
        "sourceRevision": source_revision,
        "state": "UNAUTHENTICATED_COMPARISON_CRITERIA_SATISFIED",
        "comparisonCriteriaSatisfied": True,
        "absoluteGatePassed": True,
        "authoritySafetyNoRegression": True,
        "authenticatedComparisonEnvelopePresent": False,
        "receiptEligible": False,
        "publicationEligible": False,
    }.items():
        _require(comparison_report.get(field), expected, f"comparison {field}")
    inputs = _mapping(comparison_report.get("inputReports"), "comparison inputs")
    _exact_keys(inputs, {"base", "v2", "v3"}, "comparison inputs")
    input_hashes = {name: _sha(inputs[name], f"comparison input {name}") for name in inputs}
    evaluation_payload = _mapping(
        evaluation_receipt["payload"], "evaluation receipt payload"
    )
    _require(
        input_hashes["v3"],
        evaluation_payload["testReportSha256"],
        "comparison v3 TEST binding",
    )
    payload = {
        "comparisonReportSha256": report_sha,
        "inputReportSha256s": input_hashes,
        "evaluationReceiptSha256": evaluation_receipt["receiptSha256"],
        "previousReceiptSha256": evaluation_receipt["receiptSha256"],
        "comparisonCriteriaSatisfied": True,
        "publicationEligible": False,
    }
    return _mint(
        "COMPARISON",
        payload,
        source_revision=source_revision,
        private_key=key,
        key_id=key_id,
    )


def verify_receipt(
    wrapper: dict[str, Any],
    *,
    trusted_public_key: Ed25519PublicKey | bytes | str | Path | None = None,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Verify one strict wrapper; embedded-only verification is explicitly untrusted."""

    _exact_keys(wrapper, TOP_LEVEL_KEYS, "receipt")
    _require(wrapper.get("schema"), SCHEMA, "receipt schema")
    _require(wrapper.get("candidateId"), CANDIDATE_ID, "receipt candidate")
    source_revision = _source_revision(wrapper.get("sourceRevision"), "receipt source")
    kind = wrapper.get("kind")
    if kind not in KINDS:
        raise EvidenceError("receipt kind is unsupported")
    if expected_kind is not None:
        _require(kind, expected_kind, "receipt kind")
    payload = _mapping(wrapper.get("payload"), "receipt payload")
    _exact_keys(payload, PAYLOAD_KEYS[kind], f"{kind} payload")
    _require(payload.get("publicationEligible"), False, "receipt publication flag")
    _require(
        _sha(wrapper.get("payloadSha256"), "payload digest"),
        sha256_json(payload),
        "payload digest",
    )
    if kind == "TRAINING":
        _require(payload.get("previousReceiptSha256"), None, "training chain root")
        _run_id(payload.get("runId"), "training run ID")
        _exact_keys(
            _mapping(payload.get("datasetHashes"), "dataset hashes"),
            {"train", "dev", "test"},
            "dataset hashes",
        )
        for name, digest in payload["datasetHashes"].items():
            _sha(digest, f"{name} dataset digest")
    elif kind == "EVALUATION":
        _run_id(payload.get("runId"), "evaluation run ID")
        _require(
            payload.get("previousReceiptSha256"),
            payload.get("trainingReceiptSha256"),
            "evaluation previous receipt",
        )
    else:
        _require(
            payload.get("previousReceiptSha256"),
            payload.get("evaluationReceiptSha256"),
            "comparison previous receipt",
        )
        _require(
            payload.get("comparisonCriteriaSatisfied"),
            True,
            "comparison criteria",
        )
        _exact_keys(
            _mapping(payload.get("inputReportSha256s"), "comparison inputs"),
            {"base", "v2", "v3"},
            "comparison inputs",
        )
        for name in ("base", "v2", "v3"):
            _sha(
                payload["inputReportSha256s"][name],
                f"comparison input {name}",
            )
    for key, value in payload.items():
        if key.endswith("Sha256") and value is not None:
            _sha(value, f"payload {key}")
    authentication = _mapping(wrapper.get("authentication"), "authentication")
    _exact_keys(authentication, AUTHENTICATION_KEYS, "authentication")
    _require(authentication.get("algorithm"), "Ed25519", "signature algorithm")
    key_id = authentication.get("keyId")
    if not isinstance(key_id, str) or KEY_ID.fullmatch(key_id) is None:
        raise EvidenceError("receipt key ID is malformed")
    embedded = _embedded_public_key(authentication)
    embedded_der = _public_der(embedded)
    fingerprint = _sha(
        authentication.get("publicKeyFingerprintSha256"), "public key fingerprint"
    )
    _require(fingerprint, sha256_bytes(embedded_der), "public key fingerprint")
    trusted_match = False
    if trusted_public_key is not None:
        trusted = _load_public_key(trusted_public_key)
        _require(_public_der(trusted), embedded_der, "trusted public key")
        trusted_match = True
    try:
        signature = base64.b64decode(authentication["signatureBase64"], validate=True)
        embedded.verify(
            signature,
            canonical_json(_signature_document(wrapper)).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 - normalize crypto/base64 failures
        raise EvidenceError("receipt Ed25519 signature verification failed") from exc
    unsigned = dict(wrapper)
    claimed_receipt_sha = _sha(unsigned.pop("receiptSha256"), "receipt digest")
    _require(claimed_receipt_sha, sha256_json(unsigned), "receipt digest")
    return {
        "kind": kind,
        "candidateId": CANDIDATE_ID,
        "sourceRevision": source_revision,
        "payloadSha256": wrapper["payloadSha256"],
        "receiptSha256": claimed_receipt_sha,
        "keyId": key_id,
        "publicKeyFingerprintSha256": fingerprint,
        "signatureValid": True,
        "trustedKeyMatched": trusted_match,
        "publicationEligible": False,
    }


def verify_chain(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    comparison: dict[str, Any],
    *,
    trusted_public_key: Ed25519PublicKey | bytes | str | Path | None = None,
) -> dict[str, Any]:
    """Verify the strict TRAINING -> EVALUATION -> COMPARISON receipt chain."""

    if trusted_public_key is None:
        raise EvidenceError("a trusted Ed25519 public key is required for chain verification")
    summaries = [
        verify_receipt(
            receipt,
            trusted_public_key=trusted_public_key,
            expected_kind=kind,
        )
        for receipt, kind in zip(
            (training, evaluation, comparison), KINDS, strict=True
        )
    ]
    for field in ("candidateId", "sourceRevision", "keyId", "publicKeyFingerprintSha256"):
        _require(summaries[1][field], summaries[0][field], f"evaluation {field}")
        _require(summaries[2][field], summaries[0][field], f"comparison {field}")
    training_sha = summaries[0]["receiptSha256"]
    evaluation_sha = summaries[1]["receiptSha256"]
    _require(
        evaluation["payload"].get("previousReceiptSha256"),
        training_sha,
        "evaluation chain link",
    )
    _require(
        comparison["payload"].get("previousReceiptSha256"),
        evaluation_sha,
        "comparison chain link",
    )
    _require(
        evaluation["payload"].get("runId"),
        training["payload"].get("runId"),
        "training/evaluation run ID",
    )
    _require(
        evaluation["payload"].get("adapterAggregateSha256"),
        training["payload"].get("adapterAggregateSha256"),
        "training/evaluation adapter",
    )
    _require(
        comparison["payload"]["inputReportSha256s"].get("v3"),
        evaluation["payload"].get("testReportSha256"),
        "evaluation/comparison TEST report",
    )
    return {
        "candidateId": summaries[0]["candidateId"],
        "sourceRevision": summaries[0]["sourceRevision"],
        "keyId": summaries[0]["keyId"],
        "trainingReceiptSha256": training_sha,
        "evaluationReceiptSha256": evaluation_sha,
        "comparisonReceiptSha256": summaries[2]["receiptSha256"],
        "authenticatedEvidenceChainValid": True,
        "receiptEligible": True,
        "publicationEligible": False,
    }
