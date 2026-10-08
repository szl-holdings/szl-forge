"""Fail-closed public projection of governed inference results.

Tests use only synthetic fixtures. The controller's original result,
including its private continuation, is never modified. Unknown provider fields
are held instead of being serialized to a public response.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


class PublicExportHold(ValueError):
    """The response cannot be projected under the pinned public schema."""


SOURCE_SCHEMA = "szl.forge.production-governed-inference/v2"
PUBLIC_SCHEMA = "szl.forge.public-governed-inference/v3"


_TEXT = (str,)
_OPTIONAL_TEXT = (str, type(None))
_BOOL = (bool,)
_NUMBER = (int, float, type(None))
_STRINGS = [_TEXT]

_MODEL = {
    "id": _TEXT,
    "revision": _TEXT,
    "adapter_revision": _TEXT,
    "tokenizer_revision": _TEXT,
    "template_revision": _TEXT,
    "quantization_revision": _TEXT,
}
_RUNTIME = {"engine": _TEXT, "version": _TEXT, "hardware_fingerprint": _TEXT}
_FORMULA = {
    "requested_ids": _STRINGS,
    "authorization_basis_ids": _STRINGS,
    "applications_sha256": _TEXT,
    "locked_proven_ids": _STRINGS,
    "formal_source_commit": _TEXT,
    "kernel_source_commit": _TEXT,
    "lambda_status": _TEXT,
}
_NEMO = {
    "stage": _TEXT,
    "decision": _TEXT,
    "rule_version": _TEXT,
    "input_hash": _TEXT,
    "violated_rules": _STRINGS,
    "reasons": _STRINGS,
}
_CLAIM = {
    "label": _TEXT,
    "statement_sha256": _TEXT,
    "supporting_node_ids": _STRINGS,
}
_RECEIPT_PAYLOAD = {
    "schema": _TEXT,
    "request_id": _TEXT,
    "state": _TEXT,
    "authority_state": _TEXT,
    "prompt_sha256": _TEXT,
    "principal_id_sha256": _TEXT,
    "tenant_id_sha256": _TEXT,
    "policy_revision": _TEXT,
    "retrieval_query_sha256": _TEXT,
    "ranking_receipt_sha256": _TEXT,
    "evidence_set_sha256": _TEXT,
    "formula_binding": _FORMULA,
    "model": (_MODEL, type(None)),
    "runtime": (_RUNTIME, type(None)),
    "output_sha256": _OPTIONAL_TEXT,
    "claims_sha256": _OPTIONAL_TEXT,
    "citations_sha256": _OPTIONAL_TEXT,
    "nemo": [_NEMO],
    "tool_intent_sha256": _OPTIONAL_TEXT,
    "action_admission_sha256": _OPTIONAL_TEXT,
    "action_receipt_sha256": _OPTIONAL_TEXT,
    "executed": _BOOL,
}
_RECEIPT = {
    "schema": _TEXT,
    "canonicalization": _TEXT,
    "algorithm": _TEXT,
    "payload": _RECEIPT_PAYLOAD,
    "receipt_sha256": _TEXT,
    "signature": {
        "status": _TEXT,
        "must_be_signed_before_consequential_action": _BOOL,
    },
}
_ANATOMY_EVENT = {
    "schema": _TEXT,
    "request_id": _TEXT,
    "state": _TEXT,
    "authority_state": _TEXT,
    "prompt_sha256": _TEXT,
    "principal_id_sha256": _TEXT,
    "tenant_id_sha256": _TEXT,
    "policy_revision": _TEXT,
    "evidence_set_sha256": _TEXT,
    "formula_ids": _STRINGS,
    "model_revision": _OPTIONAL_TEXT,
    "runtime_engine": _OPTIONAL_TEXT,
    "output_sha256": _OPTIONAL_TEXT,
    "claims_sha256": _OPTIONAL_TEXT,
    "nemo_decisions": [{"stage": _TEXT, "decision": _TEXT}],
    "tool_intent_sha256": _OPTIONAL_TEXT,
    "raw_prompt_present": _BOOL,
    "hydrated_content_present": _BOOL,
    "private_reasoning_present": _BOOL,
    "observer_authority": _TEXT,
}
_PUBLIC_RESULT = {
    "schema": _TEXT,
    "request_id": _TEXT,
    "prompt_sha256": _TEXT,
    "principal_id_sha256": _TEXT,
    "tenant_id_sha256": _TEXT,
    "policy_revision": _TEXT,
    "retrieval_query_sha256": _TEXT,
    "ranking_receipt_sha256": _TEXT,
    "evidence_set_sha256": _TEXT,
    "formula_binding": _FORMULA,
    "executed": _BOOL,
    "state": _TEXT,
    "authority_state": _TEXT,
    "reason_codes": _STRINGS,
    "nemo": [_NEMO],
    "output": _OPTIONAL_TEXT,
    "output_sha256": _OPTIONAL_TEXT,
    "output_schema": _TEXT,
    "claims": [_CLAIM],
    "claims_sha256": _TEXT,
    "citations": _STRINGS,
    "citations_sha256": _TEXT,
    "model": (_MODEL, type(None)),
    "runtime": (_RUNTIME, type(None)),
    "metrics": {
        "elapsed_ms": _NUMBER,
        "prompt_tokens": _NUMBER,
        "completion_tokens": _NUMBER,
        "finish_reason": _OPTIONAL_TEXT,
        "nemo_text_witness": {
            "decision": _TEXT,
            "rule_version": _TEXT,
            "input_hash": _TEXT,
            "violated_rules": _STRINGS,
            "reasons": _STRINGS,
        },
    },
    "evidence_handles": [{"nodeId": _TEXT, "nodeKind": _TEXT, "label": _TEXT, "note": _TEXT}],
    "tool_intent_sha256": _OPTIONAL_TEXT,
    "action_admission_sha256": _OPTIONAL_TEXT,
    "action_receipt_sha256": _OPTIONAL_TEXT,
    "anatomy_observation": {"delivery": _TEXT, "event": _ANATOMY_EVENT},
    "receipt": _RECEIPT,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _project(value: Any, spec: Any) -> Any:
    if isinstance(spec, dict):
        if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
            raise PublicExportHold("schema mismatch")
        if set(value) - set(spec):
            raise PublicExportHold("unknown public field")
        return {key: _project(item, spec[key]) for key, item in value.items()}
    if isinstance(spec, list):
        if not isinstance(value, list) or len(spec) != 1:
            raise PublicExportHold("schema mismatch")
        return [_project(item, spec[0]) for item in value]
    if isinstance(spec, tuple):
        for candidate in spec:
            if isinstance(candidate, dict) and isinstance(value, Mapping):
                return _project(value, candidate)
            if isinstance(candidate, type) and type(value) is candidate:
                if type(value) is float and not math.isfinite(value):
                    break
                return value
        raise PublicExportHold("schema mismatch")
    raise PublicExportHold("invalid policy")


def validate_public_body(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact recursive public schema; return an independent copy."""
    public = _project(result, _PUBLIC_RESULT)
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or public.get("state") not in {"PROPOSAL", "ABSTAIN", "REVIEW", "BLOCKED"}
        or public.get("executed") is not False
    ):
        raise PublicExportHold("unsupported inference state")
    state = public["state"]
    expected_fields = set(_PUBLIC_RESULT)
    if state == "PROPOSAL":
        if (
            set(public) != expected_fields
            or not isinstance(public["output"], str)
            or not public["output"]
            or public["output_sha256"]
            != hashlib.sha256(public["output"].encode("utf-8")).hexdigest()
            or not isinstance(public["model"], Mapping)
            or set(public["model"]) != set(_MODEL)
            or not isinstance(public["runtime"], Mapping)
            or set(public["runtime"]) != set(_RUNTIME)
            or set(public["metrics"])
            != {"elapsed_ms", "prompt_tokens", "completion_tokens", "finish_reason", "nemo_text_witness"}
            or public["metrics"]["finish_reason"] not in {None, "stop", "length", "time_budget"}
            or set(public["metrics"]["nemo_text_witness"])
            != {"decision", "rule_version", "input_hash", "violated_rules", "reasons"}
            or not public["claims"]
            or not public["evidence_handles"]
        ):
            raise PublicExportHold("incomplete proposal")
    elif (
        set(public) != expected_fields - {"output_schema", "evidence_handles"}
        or public["output"] is not None
        or public["output_sha256"] is not None
        or public["model"] is not None
        or public["runtime"] is not None
        or public["metrics"] != {}
        or public["claims"] != []
        or public["citations"] != []
    ):
        raise PublicExportHold("incomplete terminal result")
    if set(public["formula_binding"]) != set(_FORMULA):
        raise PublicExportHold("formula binding mismatch")
    if any(set(item) != set(_NEMO) for item in public["nemo"]):
        raise PublicExportHold("witness shape mismatch")
    if any(set(item) != set(_CLAIM) for item in public["claims"]):
        raise PublicExportHold("claim shape mismatch")
    if "evidence_handles" in public and any(
        set(item) != {"nodeId", "nodeKind", "label", "note"}
        for item in public["evidence_handles"]
    ):
        raise PublicExportHold("evidence handle shape mismatch")
    receipt = public["receipt"]
    if (
        set(receipt) != set(_RECEIPT)
        or receipt["schema"] != "szl.forge.production-inference-receipt/v2"
        or receipt["algorithm"] != "sha256"
        or receipt["canonicalization"] != "utf8-json-sort-keys-compact"
        or set(receipt["payload"]) != set(_RECEIPT_PAYLOAD)
        or receipt["payload"]["schema"] != SOURCE_SCHEMA
        or receipt["payload"]["state"] != state
        or receipt["payload"]["executed"] is not False
        or receipt["receipt_sha256"]
        != hashlib.sha256(_canonical(receipt["payload"])).hexdigest()
        or receipt["signature"]
        != {"status": "UNSIGNED_LOCAL", "must_be_signed_before_consequential_action": True}
    ):
        raise PublicExportHold("receipt mismatch")
    payload = receipt["payload"]
    if (
        set(payload["formula_binding"]) != set(_FORMULA)
        or any(set(item) != set(_NEMO) for item in payload["nemo"])
        or (payload["model"] is not None and set(payload["model"]) != set(_MODEL))
        or (payload["runtime"] is not None and set(payload["runtime"]) != set(_RUNTIME))
    ):
        raise PublicExportHold("receipt payload shape mismatch")
    # The v2 receipt remains source-bound, but every field it shares with the
    # v3 public body must describe the same result. A self-consistent receipt
    # hash alone does not establish that binding.
    if any(
        payload[key] != public[key]
        for key in _RECEIPT_PAYLOAD
        if key != "schema"
    ):
        raise PublicExportHold("receipt payload result mismatch")
    if (
        public["claims_sha256"] != hashlib.sha256(_canonical(public["claims"])).hexdigest()
        or public["citations_sha256"]
        != hashlib.sha256(_canonical(public["citations"])).hexdigest()
    ):
        raise PublicExportHold("claim or citation digest mismatch")
    observation = public["anatomy_observation"]
    if set(observation) != {"delivery", "event"}:
        raise PublicExportHold("observation shape mismatch")
    anatomy = observation["event"]
    if (
        set(anatomy) != set(_ANATOMY_EVENT)
        or any(
            set(item) != {"stage", "decision"}
            for item in anatomy["nemo_decisions"]
        )
        or anatomy["raw_prompt_present"] is not False
        or anatomy["hydrated_content_present"] is not False
        or anatomy["private_reasoning_present"] is not False
        or anatomy["observer_authority"] != "NONE"
    ):
        raise PublicExportHold("observation mismatch")
    for key in (
        "request_id", "state", "authority_state", "prompt_sha256",
        "principal_id_sha256", "tenant_id_sha256", "policy_revision",
        "evidence_set_sha256", "output_sha256", "claims_sha256",
        "tool_intent_sha256",
    ):
        if anatomy[key] != public[key]:
            raise PublicExportHold("observation result mismatch")
    if (
        anatomy["formula_ids"] != public["formula_binding"]["requested_ids"]
        or anatomy["model_revision"]
        != (public["model"] or {}).get("revision")
        or anatomy["runtime_engine"]
        != (public["runtime"] or {}).get("engine")
        or anatomy["nemo_decisions"]
        != [
            {"stage": item["stage"], "decision": item["decision"]}
            for item in public["nemo"]
        ]
    ):
        raise PublicExportHold("observation result mismatch")
    return public


def project_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return only public, schema-known fields; leave private continuation intact."""
    if not isinstance(result, Mapping):
        raise PublicExportHold("schema mismatch")
    if result.get("schema") != SOURCE_SCHEMA:
        raise PublicExportHold("unsupported source schema")
    if set(result) - set(_PUBLIC_RESULT) - {"continuation"}:
        raise PublicExportHold("unknown public field")
    public = _project(
        {key: value for key, value in result.items() if key != "continuation"},
        _PUBLIC_RESULT,
    )
    public["schema"] = PUBLIC_SCHEMA
    public = validate_public_body(public)
    if public["state"] == "PROPOSAL":
        continuation = result.get("continuation")
        if (
            not isinstance(continuation, Mapping)
            or continuation.get("schema") != "szl.forge.external-execution-continuation/v2"
        ):
            raise PublicExportHold("private continuation mismatch")
    public["public_export_receipt"] = {
        "schema": "szl.forge.public-transcript-export-receipt/v1",
        "state": "EXPORTED",
        "policy": "strict-v1-private-continuation-omitted",
        "source_schema": SOURCE_SCHEMA,
        "public_sha256": hashlib.sha256(_canonical(public)).hexdigest(),
        "signature_status": "UNSIGNED_LOCAL",
    }
    return public


def hold_public_result() -> dict[str, Any]:
    """Return a generic HOLD without echoing the rejected input or field names."""
    return {
        "schema": "szl.forge.public-transcript-export/v1",
        "state": "HOLD",
        "publication_eligible": False,
        "public_export_receipt": {
            "schema": "szl.forge.public-transcript-export-receipt/v1",
            "state": "HOLD",
            "policy": "strict-v1-private-continuation-omitted",
            "reason_code": "PUBLIC_SCHEMA_UNSAFE_OR_UNKNOWN",
            "signature_status": "UNSIGNED_LOCAL",
        },
    }
