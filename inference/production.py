"""Production-grade proof-carrying inference for SZL Forge.

This module coordinates a replaceable proposal model with:
- Second Brain handles-only retrieval and controller-only authorized hydration;
- exact, revision-pinned formula authority and applicability receipts;
- SZL Nemo's deterministic E1-E10 structured witness;
- A11oy-only admission for consequential tool intent;
- deterministic inference receipts and sanitized Living Anatomy observations.

The module never executes tools and never persists private chain-of-thought.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Protocol

CONTRACT_PATH = Path(__file__).resolve().with_name("production_control_plane.v2.json")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
PIN_RE = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")
LOCKED_EIGHT = ("F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22")
ADVISORY_FORMULAS = ("F23",)
TRUTH_LABELS = frozenset(
    {"MEASURED", "REPORTED", "MODELED", "CONJECTURE", "UNKNOWN", "UNAVAILABLE"}
)
PRIVATE_REASONING_MARKERS = (
    "<think>",
    "</think>",
    "<analysis>",
    "</analysis>",
    "chain_of_thought",
    "private_chain_of_thought",
    "hidden_reasoning",
)
FORBIDDEN_GENERATOR_KEYS = frozenset(
    {
        "chain_of_thought",
        "private_chain_of_thought",
        "hidden_reasoning",
        "reasoning_trace",
        "raw_private_graph",
        "private_graph",
    }
)


class Retriever(Protocol):
    def __call__(self, query: str, k: int = 6) -> Mapping[str, Any]: ...


class Hydrator(Protocol):
    def __call__(
        self,
        handles: Sequence[Mapping[str, Any]],
        *,
        principal_id: str,
        tenant_id: str,
        policy_revision: str,
    ) -> Mapping[str, Any]: ...


class Generator(Protocol):
    def identity(self) -> Mapping[str, Any]: ...

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


class EnvelopeWitness(Protocol):
    def __call__(self, envelope: Mapping[str, Any]) -> Mapping[str, Any]: ...


Observer = Callable[[Mapping[str, Any]], Any]


class ProductionBoundaryError(ValueError):
    """Raised when a component crosses the production inference boundary."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_hex64(value: Any) -> bool:
    return bool(HEX64_RE.fullmatch(str(value or "").strip().lower()))


def _is_pinned(value: Any, *, allow_none: bool = False) -> bool:
    token = str(value or "").strip().lower()
    if allow_none and token == "none":
        return True
    return bool(PIN_RE.fullmatch(token))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    return ()


def load_production_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionBoundaryError(
            f"production contract unavailable: {type(exc).__name__}"
        ) from exc
    if not isinstance(document, dict):
        raise ProductionBoundaryError("production contract must be an object")
    if document.get("schema") != "szl.forge.production-control-plane/v2":
        raise ProductionBoundaryError("unsupported production contract schema")
    if document.get("status") != "OPERATIONAL_GOVERNED_BOUNDARY":
        raise ProductionBoundaryError("production boundary is not operational")
    sources = _mapping(document.get("sources"))
    expected_sources = {
        "formal_formula": "szl-holdings/lutar-lean",
        "formula_kernel": "szl-holdings/szl-formulas",
        "second_brain": "szl-holdings/szl-second-brain",
        "nemo_witness": "szl-holdings/szl-nemo",
        "action_admission": "szl-holdings/a11oy",
    }
    for name, repository in expected_sources.items():
        source = _mapping(sources.get(name))
        if source.get("repository") != repository or not _is_pinned(
            source.get("commit")
        ):
            raise ProductionBoundaryError(f"source binding drift: {name}")
    formula = _mapping(document.get("formula_authority"))
    if tuple(formula.get("locked_proven_ids") or ()) != LOCKED_EIGHT:
        raise ProductionBoundaryError("locked-proven formula set drift")
    if formula.get("locked_proven_count") != 8:
        raise ProductionBoundaryError("locked-proven formula count drift")
    if formula.get("f_id_to_callable_mapping") != "UNKNOWN_NOT_ASSERTED":
        raise ProductionBoundaryError("unproved formula namespace mapping")
    lambda_rule = _mapping(formula.get("lambda"))
    if lambda_rule != {
        "formula_id": "F23",
        "status": "CONJECTURE_1_ADVISORY",
        "can_authorize": False,
        "can_be_sole_allow_basis": False,
    }:
        raise ProductionBoundaryError("Lambda authority drift")
    authority = _mapping(document.get("authority"))
    if authority != {
        "model": "PROPOSAL_ONLY",
        "second_brain": "READ_ONLY_EVIDENCE",
        "nemo": "INDEPENDENT_WITNESS",
        "a11oy": "ACTION_ADMISSION",
        "anatomy": "OBSERVER_ONLY",
    }:
        raise ProductionBoundaryError("authority map drift")
    runtime = _mapping(document.get("runtime_selection"))
    if runtime.get("winner") != "UNSELECTED":
        raise ProductionBoundaryError(
            "runtime winner declared without measured promotion"
        )
    if set(document.get("truth_labels") or ()) != TRUTH_LABELS:
        raise ProductionBoundaryError("truth-label vocabulary drift")
    return copy.deepcopy(document)


def _normalize_formula_applications(
    raw: Any,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    applications: list[dict[str, str]] = []
    requested: list[str] = []
    authorization_basis: list[str] = []
    seen: set[str] = set()
    allowed = set(LOCKED_EIGHT) | set(ADVISORY_FORMULAS)
    for item in _sequence(raw):
        if not isinstance(item, Mapping):
            raise ProductionBoundaryError("formula applications must be objects")
        formula_id = str(item.get("formula_id") or "").strip()
        applicability = str(item.get("applicability") or "").strip()
        basis = str(item.get("basis_sha256") or "").strip().lower()
        can_authorize = bool(item.get("authorization_basis", False))
        if formula_id not in allowed or formula_id in seen:
            raise ProductionBoundaryError(
                f"unknown or duplicate formula application: {formula_id or 'EMPTY'}"
            )
        if applicability != "APPLIES" or not _is_hex64(basis):
            raise ProductionBoundaryError(
                f"formula applicability evidence invalid: {formula_id}"
            )
        if can_authorize and formula_id not in LOCKED_EIGHT:
            raise ProductionBoundaryError(
                f"advisory formula cannot authorize: {formula_id}"
            )
        applications.append(
            {
                "formula_id": formula_id,
                "applicability": "APPLIES",
                "basis_sha256": basis,
            }
        )
        requested.append(formula_id)
        if can_authorize:
            authorization_basis.append(formula_id)
        seen.add(formula_id)
    return applications, requested, authorization_basis


def _normalize_request(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProductionBoundaryError("request must be an object")
    prompt = str(raw.get("prompt") or "").strip()
    principal_id = str(raw.get("principal_id") or "").strip()
    tenant_id = str(raw.get("tenant_id") or "").strip()
    policy_revision = str(raw.get("policy_revision") or "").strip().lower()
    if not prompt:
        raise ProductionBoundaryError("prompt is required")
    if not principal_id or not tenant_id:
        raise ProductionBoundaryError("principal_id and tenant_id are required")
    if not _is_pinned(policy_revision):
        raise ProductionBoundaryError("policy_revision must be immutable")
    applications, requested, authorization_basis = _normalize_formula_applications(
        raw.get("formula_applications", ())
    )
    request_id = str(raw.get("request_id") or "").strip()
    if not request_id:
        request_id = text_sha256(
            f"{tenant_id}\0{principal_id}\0{policy_revision}\0{prompt}"
        )[:24]
    tool_intent = raw.get("tool_intent")
    if tool_intent is not None and not isinstance(tool_intent, Mapping):
        raise ProductionBoundaryError("tool_intent must be an object")
    action_admission = raw.get("action_admission")
    action_receipt = raw.get("action_receipt")
    if action_admission is not None and not isinstance(action_admission, Mapping):
        raise ProductionBoundaryError("action_admission must be an object")
    if action_receipt is not None and not isinstance(action_receipt, Mapping):
        raise ProductionBoundaryError("action_receipt must be an object")
    if tool_intent is None and (action_admission is not None or action_receipt is not None):
        raise ProductionBoundaryError(
            "action admission and receipt require tool_intent"
        )
    try:
        k = int(raw.get("k", 6))
    except (TypeError, ValueError) as exc:
        raise ProductionBoundaryError("k must be an integer") from exc
    return {
        "request_id": request_id,
        "prompt": prompt,
        "principal_id": principal_id,
        "tenant_id": tenant_id,
        "policy_revision": policy_revision,
        "grounding_required": bool(raw.get("grounding_required", True)),
        "k": max(1, min(k, 12)),
        "formula_applications": applications,
        "formula_ids": requested,
        "authorization_basis_ids": authorization_basis,
        "tool_intent": copy.deepcopy(dict(tool_intent))
        if tool_intent is not None
        else None,
        "action_admission": copy.deepcopy(dict(action_admission))
        if action_admission is not None
        else None,
        "action_receipt": copy.deepcopy(dict(action_receipt))
        if action_receipt is not None
        else None,
    }


def _validate_retrieval(
    result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str, str]:
    if not isinstance(result, Mapping):
        raise ProductionBoundaryError("retriever must return an object")
    if result.get("content_access") != "HANDLES_ONLY":
        raise ProductionBoundaryError("retrieval must remain HANDLES_ONLY")
    handles_raw = result.get("handles") or []
    evidence_raw = result.get("evidence") or []
    if not isinstance(handles_raw, list) or not isinstance(evidence_raw, list):
        raise ProductionBoundaryError("retrieval handles and evidence must be lists")
    handles: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    for raw in handles_raw:
        if not isinstance(raw, Mapping):
            raise ProductionBoundaryError("malformed retrieval handle")
        node_id = str(raw.get("nodeId") or "").strip()
        if not node_id:
            raise ProductionBoundaryError("retrieval handle missing nodeId")
        handles.append(
            {
                "nodeId": node_id,
                "nodeKind": str(raw.get("nodeKind") or "INDEX"),
                "label": str(raw.get("label") or "DECLARED"),
                "note": str(raw.get("note") or "")[:160],
            }
        )
    for raw in evidence_raw:
        if not isinstance(raw, Mapping):
            raise ProductionBoundaryError("malformed retrieval evidence")
        node_id = str(raw.get("node_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        digest = str(raw.get("sha256") or "").strip().lower()
        if not node_id or not source or not _is_hex64(digest):
            raise ProductionBoundaryError(
                "evidence requires node_id, source, and sha256"
            )
        evidence.append(
            {"node_id": node_id, "source": source, "sha256": digest}
        )
    handle_ids = [item["nodeId"] for item in handles]
    evidence_ids = [item["node_id"] for item in evidence]
    if handle_ids != evidence_ids or len(handle_ids) != len(set(handle_ids)):
        raise ProductionBoundaryError("retrieval handle/evidence identity mismatch")
    ready = bool(result.get("ready"))
    if ready != bool(handles):
        raise ProductionBoundaryError("retrieval ready state drift")
    evidence_digest = canonical_sha256(evidence)
    if str(result.get("evidence_set_sha256") or "").lower() != evidence_digest:
        raise ProductionBoundaryError("retrieval evidence-set digest mismatch")
    ranking_receipt = result.get("ranking_receipt") or {}
    if not isinstance(ranking_receipt, Mapping):
        raise ProductionBoundaryError("ranking_receipt must be an object")
    ranking_digest = canonical_sha256(dict(ranking_receipt))
    return handles, evidence, evidence_digest, ranking_digest


def _validate_hydration(
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    evidence: Sequence[Mapping[str, str]],
    evidence_digest: str,
) -> list[dict[str, str]]:
    if not isinstance(result, Mapping):
        raise ProductionBoundaryError("hydrator must return an object")
    if result.get("content_access") != "CONTROLLER_ONLY":
        raise ProductionBoundaryError("hydrated content must remain controller-only")
    if result.get("state") != "AUTHORIZED_CONTENT_READY":
        raise ProductionBoundaryError("authorized hydration is not ready")
    if str(result.get("principal_id_sha256") or "").lower() != text_sha256(
        request["principal_id"]
    ):
        raise ProductionBoundaryError("hydration principal binding mismatch")
    if str(result.get("tenant_id_sha256") or "").lower() != text_sha256(
        request["tenant_id"]
    ):
        raise ProductionBoundaryError("hydration tenant binding mismatch")
    if str(result.get("policy_revision") or "").lower() != request["policy_revision"]:
        raise ProductionBoundaryError("hydration policy binding mismatch")
    if str(result.get("evidence_set_sha256") or "").lower() != evidence_digest:
        raise ProductionBoundaryError("hydration evidence-set digest mismatch")
    documents_raw = result.get("documents") or []
    if not isinstance(documents_raw, list):
        raise ProductionBoundaryError("hydrated documents must be a list")
    expected = {item["node_id"]: item for item in evidence}
    documents: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in documents_raw:
        if not isinstance(raw, Mapping):
            raise ProductionBoundaryError("malformed hydrated document")
        node_id = str(raw.get("node_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        digest = str(raw.get("sha256") or "").strip().lower()
        content = str(raw.get("content") or "")
        if node_id not in expected or node_id in seen:
            raise ProductionBoundaryError("unknown or duplicate hydrated document")
        if source != expected[node_id]["source"] or digest != expected[node_id]["sha256"]:
            raise ProductionBoundaryError("hydrated source/digest binding mismatch")
        if text_sha256(content) != digest:
            raise ProductionBoundaryError("hydrated content digest mismatch")
        documents.append(
            {
                "node_id": node_id,
                "source": source,
                "sha256": digest,
                "content": content,
            }
        )
        seen.add(node_id)
    if seen != set(expected):
        raise ProductionBoundaryError("incomplete authorized hydration")
    if bool(result.get("ready")) != bool(documents):
        raise ProductionBoundaryError("hydration ready state drift")
    if result.get("raw_graph_nodes_admitted_to_gradients") != 0:
        raise ProductionBoundaryError("private graph gradient boundary drift")
    return documents


def _validate_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProductionBoundaryError("generator identity must be an object")
    model = _mapping(raw.get("model"))
    runtime = _mapping(raw.get("runtime"))
    normalized_model = {
        "id": str(model.get("id") or "").strip(),
        "revision": str(model.get("revision") or "").strip().lower(),
        "adapter_revision": str(
            model.get("adapter_revision") or "NONE"
        ).strip().lower(),
        "tokenizer_revision": str(
            model.get("tokenizer_revision") or ""
        ).strip().lower(),
        "template_revision": str(
            model.get("template_revision") or ""
        ).strip().lower(),
        "quantization_revision": str(
            model.get("quantization_revision") or "NONE"
        ).strip().lower(),
    }
    if not normalized_model["id"]:
        raise ProductionBoundaryError("model id is required")
    for key in ("revision", "tokenizer_revision", "template_revision"):
        if not _is_pinned(normalized_model[key]):
            raise ProductionBoundaryError(f"{key} must be pinned")
    for key in ("adapter_revision", "quantization_revision"):
        if not _is_pinned(normalized_model[key], allow_none=True):
            raise ProductionBoundaryError(f"{key} must be NONE or pinned")
    normalized_runtime = {
        "engine": str(runtime.get("engine") or "").strip(),
        "version": str(runtime.get("version") or "").strip(),
        "hardware_fingerprint": str(
            runtime.get("hardware_fingerprint") or ""
        ).strip(),
    }
    if not all(normalized_runtime.values()):
        raise ProductionBoundaryError(
            "runtime engine, version, and hardware fingerprint are required"
        )
    return {"model": normalized_model, "runtime": normalized_runtime}


def _validate_claims(
    raw: Any, *, available_nodes: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ProductionBoundaryError("generator must emit at least one claim")
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProductionBoundaryError("claims must be objects")
        label = str(item.get("label") or "").strip().upper()
        statement_digest = str(
            item.get("statement_sha256") or ""
        ).strip().lower()
        support = [
            str(node_id)
            for node_id in _sequence(item.get("supporting_node_ids"))
        ]
        if label not in TRUTH_LABELS or not _is_hex64(statement_digest):
            raise ProductionBoundaryError("claim label or digest invalid")
        if statement_digest in seen:
            raise ProductionBoundaryError("duplicate claim digest")
        if len(support) != len(set(support)) or not set(support).issubset(
            available_nodes
        ):
            raise ProductionBoundaryError("claim evidence binding invalid")
        claims.append(
            {
                "label": label,
                "statement_sha256": statement_digest,
                "supporting_node_ids": support,
            }
        )
        seen.add(statement_digest)
    return claims


def _validate_generation(
    raw: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    available_nodes: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProductionBoundaryError("generator must return an object")
    forbidden = set(raw) & FORBIDDEN_GENERATOR_KEYS
    if forbidden:
        raise ProductionBoundaryError(
            f"generator exposed private state: {sorted(forbidden)}"
        )
    text = str(raw.get("text") or "").strip()
    if not text:
        raise ProductionBoundaryError("generator returned an empty answer")
    lowered = text.lower()
    if any(marker in lowered for marker in PRIVATE_REASONING_MARKERS):
        raise ProductionBoundaryError("generator emitted private reasoning")
    identity = _validate_identity(
        _mapping(raw.get("identity")) or expected_identity
    )
    if identity != expected_identity:
        raise ProductionBoundaryError("generator identity changed during request")
    output_schema = str(raw.get("output_schema") or "").strip()
    if not output_schema:
        raise ProductionBoundaryError("output_schema is required")
    claims = _validate_claims(
        raw.get("claims"), available_nodes=available_nodes
    )
    citations = [str(node_id) for node_id in _sequence(raw.get("citations"))]
    if len(citations) != len(set(citations)) or not set(citations).issubset(
        available_nodes
    ):
        raise ProductionBoundaryError("citation binding invalid")
    metrics = raw.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        raise ProductionBoundaryError("metrics must be an object")
    return {
        "text": text,
        "identity": identity,
        "output_schema": output_schema,
        "claims": claims,
        "citations": citations,
        "metrics": copy.deepcopy(dict(metrics)),
    }


def _formula_envelope(
    request: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    sources = contract["sources"]
    formula = contract["formula_authority"]
    return {
        "locked_proven_ids": list(LOCKED_EIGHT),
        "locked_proven_count": 8,
        "formal_source_repository": sources["formal_formula"]["repository"],
        "formal_source_commit": sources["formal_formula"]["commit"],
        "kernel_source_repository": sources["formula_kernel"]["repository"],
        "kernel_source_commit": sources["formula_kernel"]["commit"],
        "f_id_to_callable_mapping": formula["f_id_to_callable_mapping"],
        "requested_formula_ids": list(request["formula_ids"]),
        "applications": copy.deepcopy(request["formula_applications"]),
        "authorization_basis_ids": list(request["authorization_basis_ids"]),
        "lambda": copy.deepcopy(formula["lambda"]),
    }


def _base_nemo_envelope(
    *,
    stage: str,
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    identity: Mapping[str, Any],
    handles: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, str]],
    evidence_digest: str,
    claims: Sequence[Mapping[str, Any]],
    witness_history: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "szl.nemo.inference-envelope.v1",
        "stage": stage,
        "witness_identity": {
            "artifact_kind": "SOFTWARE_KERNEL",
            "generative": False,
            "not_nemotron": True,
        },
        "scope": {
            "principal_id_sha256": text_sha256(request["principal_id"]),
            "tenant_id_sha256": text_sha256(request["tenant_id"]),
            "access_decision": "ALLOW",
            "policy_revision": request["policy_revision"],
        },
        "model": {
            "id": identity["model"]["id"],
            "revision": identity["model"]["revision"],
            "adapter_revision": identity["model"]["adapter_revision"],
            "tokenizer_revision": identity["model"]["tokenizer_revision"],
            "template_revision": identity["model"]["template_revision"],
        },
        "runtime": copy.deepcopy(identity["runtime"]),
        "evidence": {
            "content_access": "HANDLES_ONLY",
            "grounding_required": request["grounding_required"],
            "handles": [{"nodeId": item["nodeId"]} for item in handles],
            "items": copy.deepcopy(list(evidence)),
            "evidence_set_sha256": evidence_digest,
        },
        "formulas": _formula_envelope(request, contract),
        "authority": {
            "model_authority": "PROPOSAL_ONLY",
            "executed": False,
            "execution_authority": "NONE",
        },
        "witness_history": list(witness_history),
        "claims": [
            {
                "label": item["label"],
                "statement_sha256": item["statement_sha256"],
            }
            for item in claims
        ],
        "tool_intent": None,
        "action_admission": None,
        "receipt": None,
        "tool_result": None,
        "postcondition": None,
    }


def _normalize_witness(
    raw: Mapping[str, Any], *, stage: str, envelope_digest: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProductionBoundaryError("Nemo witness must return an object")
    decision = str(raw.get("decision") or "").strip().upper()
    if decision not in {"ALLOW", "BLOCK", "REVIEW"}:
        raise ProductionBoundaryError("Nemo witness decision invalid")
    rule_version = str(raw.get("rule_version") or "").strip()
    input_hash = str(raw.get("input_hash") or "").strip().lower()
    violated_rules = [
        str(item) for item in _sequence(raw.get("violated_rules"))
    ]
    reasons = [str(item) for item in _sequence(raw.get("reasons"))]
    if not rule_version:
        raise ProductionBoundaryError("Nemo rule_version is required")
    if input_hash != f"sha256:{envelope_digest}":
        raise ProductionBoundaryError("Nemo input hash does not bind envelope")
    return {
        "stage": stage,
        "decision": decision,
        "rule_version": rule_version,
        "input_hash": input_hash,
        "violated_rules": violated_rules,
        "reasons": reasons,
    }


def _run_witness(
    witness: EnvelopeWitness,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    digest = canonical_sha256(envelope)
    try:
        raw = witness(copy.deepcopy(dict(envelope)))
    except Exception as exc:
        raise ProductionBoundaryError(
            f"Nemo witness unavailable: {type(exc).__name__}"
        ) from exc
    return _normalize_witness(
        raw, stage=str(envelope["stage"]), envelope_digest=digest
    )


def _pending_action_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "authority": "A11OY",
            "human_approval": "PENDING",
            "signed_receipt_required": True,
        },
        {"signature_status": "PENDING"},
    )


def _normalize_action_contract(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    admission_raw = request.get("action_admission")
    receipt_raw = request.get("action_receipt")
    if admission_raw is None and receipt_raw is None:
        return _pending_action_contract()
    admission = copy.deepcopy(dict(_mapping(admission_raw)))
    receipt = copy.deepcopy(dict(_mapping(receipt_raw)))
    if admission.get("authority") != "A11OY":
        raise ProductionBoundaryError("action authority must be A11OY")
    if admission.get("human_approval") not in {"PENDING", "APPROVED"}:
        raise ProductionBoundaryError("human approval state invalid")
    if admission.get("signed_receipt_required") is not True:
        raise ProductionBoundaryError("signed action receipt is required")
    if admission["human_approval"] == "APPROVED":
        if (
            receipt.get("signature_status") != "SIGNED_VERIFIED"
            or not _is_hex64(receipt.get("sha256"))
        ):
            raise ProductionBoundaryError(
                "approved action requires a verified signed receipt"
            )
    elif receipt.get("signature_status") not in {
        "PENDING",
        "UNSIGNED_HONEST",
    }:
        raise ProductionBoundaryError("pending action receipt state invalid")
    return admission, receipt


def _inference_receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": result["schema"],
        "request_id": result["request_id"],
        "state": result["state"],
        "authority_state": result["authority_state"],
        "prompt_sha256": result["prompt_sha256"],
        "principal_id_sha256": result["principal_id_sha256"],
        "tenant_id_sha256": result["tenant_id_sha256"],
        "policy_revision": result["policy_revision"],
        "retrieval_query_sha256": result["retrieval_query_sha256"],
        "ranking_receipt_sha256": result["ranking_receipt_sha256"],
        "evidence_set_sha256": result["evidence_set_sha256"],
        "formula_binding": result["formula_binding"],
        "model": result.get("model"),
        "runtime": result.get("runtime"),
        "output_sha256": result.get("output_sha256"),
        "claims_sha256": result.get("claims_sha256"),
        "citations_sha256": result.get("citations_sha256"),
        "nemo": result.get("nemo", []),
        "tool_intent_sha256": result.get("tool_intent_sha256"),
        "action_admission_sha256": result.get("action_admission_sha256"),
        "action_receipt_sha256": result.get("action_receipt_sha256"),
        "executed": False,
    }
    digest = canonical_sha256(payload)
    return {
        "schema": "szl.forge.production-inference-receipt/v2",
        "canonicalization": "utf8-json-sort-keys-compact",
        "algorithm": "sha256",
        "payload": payload,
        "receipt_sha256": digest,
        "signature": {
            "status": "UNSIGNED_LOCAL",
            "must_be_signed_before_consequential_action": True,
        },
    }


def _anatomy_event(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "szl.anatomy.production-inference-observation/v2",
        "request_id": result["request_id"],
        "state": result["state"],
        "authority_state": result["authority_state"],
        "prompt_sha256": result["prompt_sha256"],
        "principal_id_sha256": result["principal_id_sha256"],
        "tenant_id_sha256": result["tenant_id_sha256"],
        "policy_revision": result["policy_revision"],
        "evidence_set_sha256": result["evidence_set_sha256"],
        "formula_ids": result["formula_binding"]["requested_ids"],
        "model_revision": _mapping(result.get("model")).get("revision"),
        "runtime_engine": _mapping(result.get("runtime")).get("engine"),
        "output_sha256": result.get("output_sha256"),
        "claims_sha256": result.get("claims_sha256"),
        "nemo_decisions": [
            {"stage": item["stage"], "decision": item["decision"]}
            for item in result.get("nemo", [])
        ],
        "tool_intent_sha256": result.get("tool_intent_sha256"),
        "raw_prompt_present": False,
        "hydrated_content_present": False,
        "private_reasoning_present": False,
        "observer_authority": "NONE",
    }


def _finalize(
    result: dict[str, Any], observer: Observer | None
) -> dict[str, Any]:
    event = _anatomy_event(result)
    delivery = "NOT_CONFIGURED"
    if observer is not None:
        try:
            observer(copy.deepcopy(event))
            delivery = "DELIVERED"
        except Exception:
            delivery = "UNAVAILABLE"
    result["anatomy_observation"] = {
        "delivery": delivery,
        "event": event,
    }
    result["receipt"] = _inference_receipt(result)
    return result


def _base_result(
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    evidence_digest: str,
    ranking_digest: str,
) -> dict[str, Any]:
    return {
        "schema": "szl.forge.production-governed-inference/v2",
        "request_id": request["request_id"],
        "prompt_sha256": text_sha256(request["prompt"]),
        "principal_id_sha256": text_sha256(request["principal_id"]),
        "tenant_id_sha256": text_sha256(request["tenant_id"]),
        "policy_revision": request["policy_revision"],
        "retrieval_query_sha256": text_sha256(request["prompt"]),
        "ranking_receipt_sha256": ranking_digest,
        "evidence_set_sha256": evidence_digest,
        "formula_binding": {
            "requested_ids": list(request["formula_ids"]),
            "authorization_basis_ids": list(
                request["authorization_basis_ids"]
            ),
            "applications_sha256": canonical_sha256(
                request["formula_applications"]
            ),
            "locked_proven_ids": list(LOCKED_EIGHT),
            "formal_source_commit": contract["sources"]["formal_formula"][
                "commit"
            ],
            "kernel_source_commit": contract["sources"]["formula_kernel"][
                "commit"
            ],
            "lambda_status": contract["formula_authority"]["lambda"]["status"],
        },
        "executed": False,
    }


def _terminal(
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    state: str,
    authority_state: str,
    reason_codes: Sequence[str],
    evidence_digest: str,
    ranking_digest: str,
    nemo: Sequence[Mapping[str, Any]] = (),
    observer: Observer | None = None,
) -> dict[str, Any]:
    result = _base_result(
        request,
        contract,
        evidence_digest=evidence_digest,
        ranking_digest=ranking_digest,
    )
    result.update(
        {
            "state": state,
            "authority_state": authority_state,
            "reason_codes": list(reason_codes),
            "nemo": [copy.deepcopy(dict(item)) for item in nemo],
            "output": None,
            "output_sha256": None,
            "claims": [],
            "claims_sha256": canonical_sha256([]),
            "citations": [],
            "citations_sha256": canonical_sha256([]),
            "model": None,
            "runtime": None,
            "metrics": {},
            "tool_intent_sha256": canonical_sha256(
                request["tool_intent"]
            )
            if request.get("tool_intent")
            else None,
            "action_admission_sha256": None,
            "action_receipt_sha256": None,
        }
    )
    return _finalize(result, observer)


def production_infer(
    request: Mapping[str, Any],
    *,
    retriever: Retriever,
    hydrator: Hydrator,
    generator: Generator,
    witness: EnvelopeWitness,
    observer: Observer | None = None,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, Any]:
    """Return a governed proposal or action-ready package without executing a tool."""
    contract = load_production_contract(contract_path)
    try:
        safe = _normalize_request(request)
    except ProductionBoundaryError as exc:
        fallback = {
            "request_id": text_sha256(canonical_bytes(dict(request)).decode())[
                :24
            ],
            "prompt": str(request.get("prompt") or ""),
            "principal_id": str(request.get("principal_id") or "UNAVAILABLE"),
            "tenant_id": str(request.get("tenant_id") or "UNAVAILABLE"),
            "policy_revision": (
                str(request.get("policy_revision") or "").lower()
                if _is_pinned(request.get("policy_revision"))
                else "0" * 40
            ),
            "grounding_required": True,
            "k": 6,
            "formula_applications": [],
            "formula_ids": [],
            "authorization_basis_ids": [],
            "tool_intent": None,
            "action_admission": None,
            "action_receipt": None,
        }
        return _terminal(
            fallback,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["REQUEST_BOUNDARY_INVALID", str(exc)],
            evidence_digest=canonical_sha256([]),
            ranking_digest=canonical_sha256({}),
            observer=observer,
        )

    try:
        retrieval_raw = retriever(safe["prompt"], safe["k"])
        handles, evidence, evidence_digest, ranking_digest = (
            _validate_retrieval(retrieval_raw)
        )
    except Exception as exc:
        return _terminal(
            safe,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=[
                "SECOND_BRAIN_RETRIEVAL_INVALID",
                type(exc).__name__,
            ],
            evidence_digest=canonical_sha256([]),
            ranking_digest=canonical_sha256({}),
            observer=observer,
        )
    if safe["grounding_required"] and not handles:
        return _terminal(
            safe,
            contract,
            state="ABSTAIN",
            authority_state="NONE",
            reason_codes=["NO_GROUNDED_EVIDENCE"],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            observer=observer,
        )

    try:
        hydration_raw = hydrator(
            handles,
            principal_id=safe["principal_id"],
            tenant_id=safe["tenant_id"],
            policy_revision=safe["policy_revision"],
        )
        documents = _validate_hydration(
            hydration_raw,
            request=safe,
            evidence=evidence,
            evidence_digest=evidence_digest,
        )
    except Exception as exc:
        return _terminal(
            safe,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=[
                "SECOND_BRAIN_AUTHORIZATION_OR_HYDRATION_INVALID",
                type(exc).__name__,
            ],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            observer=observer,
        )

    try:
        identity = _validate_identity(generator.identity())
    except Exception as exc:
        return _terminal(
            safe,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["GENERATOR_IDENTITY_INVALID", type(exc).__name__],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            observer=observer,
        )

    nemo_records: list[dict[str, Any]] = []
    pre_envelope = _base_nemo_envelope(
        stage="PRE_GENERATION",
        request=safe,
        contract=contract,
        identity=identity,
        handles=handles,
        evidence=evidence,
        evidence_digest=evidence_digest,
        claims=[],
        witness_history=[],
    )
    try:
        pre = _run_witness(witness, pre_envelope)
    except Exception as exc:
        return _terminal(
            safe,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["NEMO_PRE_GENERATION_INVALID", type(exc).__name__],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            observer=observer,
        )
    nemo_records.append(pre)
    if pre["decision"] != "ALLOW":
        return _terminal(
            safe,
            contract,
            state="BLOCKED" if pre["decision"] == "BLOCK" else "REVIEW",
            authority_state="NONE",
            reason_codes=["NEMO_PRE_GENERATION_" + pre["decision"]]
            + pre["violated_rules"],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            nemo=nemo_records,
            observer=observer,
        )

    generation_context = {
        "schema": "szl.forge.authorized-generation-context/v2",
        "request_id": safe["request_id"],
        "prompt": safe["prompt"],
        "evidence": documents,
        "formula_applications": copy.deepcopy(
            safe["formula_applications"]
        ),
        "authority": "PROPOSAL_ONLY",
        "instructions": {
            "cite_node_ids": True,
            "emit_claim_digests": True,
            "expose_private_chain_of_thought": False,
            "execute_tools": False,
            "lambda_can_authorize": False,
        },
    }
    try:
        generated_raw = generator(generation_context)
        generated = _validate_generation(
            generated_raw,
            expected_identity=identity,
            available_nodes={item["node_id"] for item in evidence},
        )
    except Exception as exc:
        return _terminal(
            safe,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=[
                "GENERATOR_OUTPUT_INVALID",
                type(exc).__name__,
            ],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            nemo=nemo_records,
            observer=observer,
        )

    post_envelope = _base_nemo_envelope(
        stage="POST_GENERATION",
        request=safe,
        contract=contract,
        identity=identity,
        handles=handles,
        evidence=evidence,
        evidence_digest=evidence_digest,
        claims=generated["claims"],
        witness_history=["PRE_GENERATION"],
    )
    try:
        post = _run_witness(witness, post_envelope)
    except Exception as exc:
        return _terminal(
            safe,
            contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["NEMO_POST_GENERATION_INVALID", type(exc).__name__],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            nemo=nemo_records,
            observer=observer,
        )
    nemo_records.append(post)
    if post["decision"] != "ALLOW":
        return _terminal(
            safe,
            contract,
            state="BLOCKED" if post["decision"] == "BLOCK" else "REVIEW",
            authority_state="NONE",
            reason_codes=["NEMO_POST_GENERATION_" + post["decision"]]
            + post["violated_rules"],
            evidence_digest=evidence_digest,
            ranking_digest=ranking_digest,
            nemo=nemo_records,
            observer=observer,
        )

    tool_digest: str | None = None
    admission_digest: str | None = None
    action_receipt_digest: str | None = None
    state = "PROPOSAL"
    authority_state = "NO_ACTION_AUTHORITY"
    reason_codes: list[str] = []

    if safe["tool_intent"] is not None:
        tool_digest = canonical_sha256(safe["tool_intent"])
        try:
            admission, action_receipt = _normalize_action_contract(safe)
        except Exception as exc:
            return _terminal(
                safe,
                contract,
                state="BLOCKED",
                authority_state="NONE",
                reason_codes=[
                    "A11OY_ACTION_CONTRACT_INVALID",
                    type(exc).__name__,
                ],
                evidence_digest=evidence_digest,
                ranking_digest=ranking_digest,
                nemo=nemo_records,
                observer=observer,
            )
        pre_tool_envelope = _base_nemo_envelope(
            stage="PRE_TOOL",
            request=safe,
            contract=contract,
            identity=identity,
            handles=handles,
            evidence=evidence,
            evidence_digest=evidence_digest,
            claims=generated["claims"],
            witness_history=["PRE_GENERATION", "POST_GENERATION"],
        )
        pre_tool_envelope["tool_intent"] = {"sha256": tool_digest}
        pre_tool_envelope["action_admission"] = admission
        pre_tool_envelope["receipt"] = action_receipt
        try:
            pre_tool = _run_witness(witness, pre_tool_envelope)
        except Exception as exc:
            return _terminal(
                safe,
                contract,
                state="BLOCKED",
                authority_state="NONE",
                reason_codes=["NEMO_PRE_TOOL_INVALID", type(exc).__name__],
                evidence_digest=evidence_digest,
                ranking_digest=ranking_digest,
                nemo=nemo_records,
                observer=observer,
            )
        nemo_records.append(pre_tool)
        admission_digest = canonical_sha256(admission)
        action_receipt_digest = canonical_sha256(action_receipt)
        if pre_tool["decision"] == "BLOCK":
            return _terminal(
                safe,
                contract,
                state="BLOCKED",
                authority_state="NONE",
                reason_codes=["NEMO_PRE_TOOL_BLOCK"]
                + pre_tool["violated_rules"],
                evidence_digest=evidence_digest,
                ranking_digest=ranking_digest,
                nemo=nemo_records,
                observer=observer,
            )
        if pre_tool["decision"] == "REVIEW":
            state = "REVIEW"
            authority_state = "HUMAN_APPROVAL_AND_SIGNED_RECEIPT_REQUIRED"
            reason_codes = ["A11OY_ADMISSION_PENDING"]
        else:
            state = "ACTION_READY_EXTERNAL_EXECUTION"
            authority_state = "A11OY_SIGNED_ADMISSION_VERIFIED"
            reason_codes = ["FORGE_DID_NOT_EXECUTE_TOOL"]

    result = _base_result(
        safe,
        contract,
        evidence_digest=evidence_digest,
        ranking_digest=ranking_digest,
    )
    result.update(
        {
            "state": state,
            "authority_state": authority_state,
            "reason_codes": reason_codes,
            "nemo": nemo_records,
            "output": generated["text"],
            "output_sha256": text_sha256(generated["text"]),
            "output_schema": generated["output_schema"],
            "claims": generated["claims"],
            "claims_sha256": canonical_sha256(generated["claims"]),
            "citations": generated["citations"],
            "citations_sha256": canonical_sha256(
                generated["citations"]
            ),
            "model": identity["model"],
            "runtime": identity["runtime"],
            "metrics": generated["metrics"],
            "evidence_handles": handles,
            "tool_intent_sha256": tool_digest,
            "action_admission_sha256": admission_digest,
            "action_receipt_sha256": action_receipt_digest,
            "continuation": {
                "schema": "szl.forge.external-execution-continuation/v2",
                "request_id": safe["request_id"],
                "scope": {
                    "principal_id_sha256": text_sha256(
                        safe["principal_id"]
                    ),
                    "tenant_id_sha256": text_sha256(safe["tenant_id"]),
                    "policy_revision": safe["policy_revision"],
                },
                "model": identity["model"],
                "runtime": identity["runtime"],
                "evidence": {
                    "handles": [{"nodeId": h["nodeId"]} for h in handles],
                    "items": evidence,
                    "evidence_set_sha256": evidence_digest,
                    "grounding_required": safe["grounding_required"],
                },
                "formulas": _formula_envelope(safe, contract),
                "claims": [
                    {
                        "label": item["label"],
                        "statement_sha256": item["statement_sha256"],
                    }
                    for item in generated["claims"]
                ],
                "tool_intent": {"sha256": tool_digest}
                if tool_digest
                else None,
                "action_admission_sha256": admission_digest,
                "action_receipt_sha256": action_receipt_digest,
                "witness_history": [
                    item["stage"] for item in nemo_records
                ],
                "raw_prompt_present": False,
                "hydrated_content_present": False,
                "private_reasoning_present": False,
            },
        }
    )
    return _finalize(result, observer)


def verify_external_execution(
    inference_result: Mapping[str, Any],
    *,
    tool_result_sha256: str,
    postcondition_status: str,
    postcondition_details_sha256: str,
    action_admission: Mapping[str, Any],
    action_receipt: Mapping[str, Any],
    witness: EnvelopeWitness,
    observer: Observer | None = None,
) -> dict[str, Any]:
    """Witness an A11oy-executed action and its bound postcondition."""
    continuation = _mapping(inference_result.get("continuation"))
    if (
        inference_result.get("state") != "ACTION_READY_EXTERNAL_EXECUTION"
        or continuation.get("schema")
        != "szl.forge.external-execution-continuation/v2"
    ):
        raise ProductionBoundaryError(
            "execution verification requires an action-ready inference result"
        )
    if not _is_hex64(tool_result_sha256) or not _is_hex64(
        postcondition_details_sha256
    ):
        raise ProductionBoundaryError("tool result and postcondition digests required")
    status = str(postcondition_status or "").strip().upper()
    if status not in {"PASS", "FAIL"}:
        raise ProductionBoundaryError("postcondition status must be PASS or FAIL")
    admission = copy.deepcopy(dict(action_admission))
    receipt = copy.deepcopy(dict(action_receipt))
    if canonical_sha256(admission) != continuation.get(
        "action_admission_sha256"
    ):
        raise ProductionBoundaryError("action admission changed after inference")
    if canonical_sha256(receipt) != continuation.get("action_receipt_sha256"):
        raise ProductionBoundaryError("action receipt changed after inference")
    if (
        admission.get("authority") != "A11OY"
        or admission.get("human_approval") != "APPROVED"
        or admission.get("signed_receipt_required") is not True
        or receipt.get("signature_status") != "SIGNED_VERIFIED"
        or not _is_hex64(receipt.get("sha256"))
    ):
        raise ProductionBoundaryError(
            "post-tool verification requires approved signed A11oy admission"
        )

    scope = _mapping(continuation.get("scope"))
    evidence = _mapping(continuation.get("evidence"))
    formulas = _mapping(continuation.get("formulas"))
    envelope = {
        "schema": "szl.nemo.inference-envelope.v1",
        "stage": "POST_TOOL",
        "witness_identity": {
            "artifact_kind": "SOFTWARE_KERNEL",
            "generative": False,
            "not_nemotron": True,
        },
        "scope": {
            "principal_id_sha256": scope.get("principal_id_sha256"),
            "tenant_id_sha256": scope.get("tenant_id_sha256"),
            "access_decision": "ALLOW",
            "policy_revision": scope.get("policy_revision"),
        },
        "model": {
            key: continuation["model"][key]
            for key in (
                "id",
                "revision",
                "adapter_revision",
                "tokenizer_revision",
                "template_revision",
            )
        },
        "runtime": continuation["runtime"],
        "evidence": {
            "content_access": "HANDLES_ONLY",
            "grounding_required": evidence.get("grounding_required"),
            "handles": evidence.get("handles"),
            "items": evidence.get("items"),
            "evidence_set_sha256": evidence.get("evidence_set_sha256"),
        },
        "formulas": formulas,
        "authority": {
            "model_authority": "PROPOSAL_ONLY",
            "executed": True,
            "execution_authority": "A11OY",
        },
        "witness_history": [
            "PRE_GENERATION",
            "POST_GENERATION",
            "PRE_TOOL",
        ],
        "claims": continuation.get("claims"),
        "tool_intent": continuation.get("tool_intent"),
        "action_admission": admission,
        "receipt": receipt,
        "tool_result": {"sha256": tool_result_sha256.lower()},
        "postcondition": {
            "status": status,
            "details_sha256": postcondition_details_sha256.lower(),
        },
    }
    decision = _run_witness(witness, envelope)
    result = {
        "schema": "szl.forge.external-execution-verification/v2",
        "request_id": continuation.get("request_id"),
        "state": "VERIFIED" if decision["decision"] == "ALLOW" else "REVIEW",
        "authority_state": "A11OY_EXECUTED_FORGE_DID_NOT_EXECUTE",
        "executed": True,
        "tool_result_sha256": tool_result_sha256.lower(),
        "postcondition": {
            "status": status,
            "details_sha256": postcondition_details_sha256.lower(),
        },
        "nemo": decision,
        "source_inference_receipt_sha256": _mapping(
            inference_result.get("receipt")
        ).get("receipt_sha256"),
        "raw_prompt_present": False,
        "hydrated_content_present": False,
        "private_reasoning_present": False,
    }
    event = {
        "schema": "szl.anatomy.external-execution-observation/v2",
        "request_id": result["request_id"],
        "state": result["state"],
        "authority_state": result["authority_state"],
        "tool_result_sha256": result["tool_result_sha256"],
        "postcondition_status": status,
        "postcondition_details_sha256": postcondition_details_sha256.lower(),
        "nemo_decision": decision["decision"],
        "observer_authority": "NONE",
        "raw_prompt_present": False,
        "hydrated_content_present": False,
        "private_reasoning_present": False,
    }
    delivery = "NOT_CONFIGURED"
    if observer is not None:
        try:
            observer(copy.deepcopy(event))
            delivery = "DELIVERED"
        except Exception:
            delivery = "UNAVAILABLE"
    result["anatomy_observation"] = {"delivery": delivery, "event": event}
    result["verification_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in result.items()
            if key not in {"verification_sha256"}
        }
    )
    return result


def make_second_brain_retriever() -> Retriever:
    try:
        from second_brain import hybrid_context  # type: ignore
    except ImportError as exc:
        raise ProductionBoundaryError(
            "szl-second-brain is not installed"
        ) from exc
    return hybrid_context


def make_second_brain_hydrator(
    authorizer: Callable[[str, str, str, str, str], bool],
    *,
    path: Path | None = None,
) -> Hydrator:
    try:
        from second_brain import AuthorizedHydrator  # type: ignore
    except ImportError as exc:
        raise ProductionBoundaryError(
            "szl-second-brain is not installed"
        ) from exc
    instance = AuthorizedHydrator(authorizer, path=path)

    def hydrate(
        handles: Sequence[Mapping[str, Any]],
        *,
        principal_id: str,
        tenant_id: str,
        policy_revision: str,
    ) -> Mapping[str, Any]:
        return instance.hydrate(
            handles,
            principal_id=principal_id,
            tenant_id=tenant_id,
            policy_revision=policy_revision,
        )

    return hydrate


def make_szl_nemo_envelope_witness() -> EnvelopeWitness:
    try:
        from szl_nemo import evaluate_envelope  # type: ignore
    except ImportError as exc:
        raise ProductionBoundaryError("szl-nemo is not installed") from exc

    def witness(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
        return evaluate_envelope(envelope).to_dict()

    return witness


class OpenAICompatibleProductionGenerator:
    """Pinned adapter for llama.cpp, vLLM, or SGLang compatible servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        template_revision: str,
        adapter_revision: str = "NONE",
        quantization_revision: str = "NONE",
        engine: str,
        engine_version: str,
        hardware_fingerprint: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 45.0,
        max_tokens: int = 384,
    ) -> None:
        base = base_url.rstrip("/")
        if not (
            base.startswith("https://")
            or base.startswith("http://127.0.0.1")
            or base.startswith("http://localhost")
        ):
            raise ProductionBoundaryError(
                "inference endpoint must be HTTPS or loopback HTTP"
            )
        self.base_url = base
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)
        self._identity = _validate_identity(
            {
                "model": {
                    "id": model_id,
                    "revision": model_revision,
                    "adapter_revision": adapter_revision,
                    "tokenizer_revision": tokenizer_revision,
                    "template_revision": template_revision,
                    "quantization_revision": quantization_revision,
                },
                "runtime": {
                    "engine": engine,
                    "version": engine_version,
                    "hardware_fingerprint": hardware_fingerprint,
                },
            }
        )

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        evidence = context.get("evidence") or []
        evidence_block = "\n\n".join(
            (
                f"[{row['node_id']}] source={row['source']} "
                f"sha256={row['sha256']}\n{row['content']}"
            )
            for row in evidence
        )
        system = (
            "You are the proposal-only SZL Forge inference engine. Use only the "
            "authorized evidence supplied below. Cite node IDs in square brackets. "
            "Do not execute tools, invent evidence, expose private chain-of-thought, "
            "or describe F23/Lambda as proven. Return a concise answer; the controller "
            "will create claim digests and governance receipts.\n\n"
            f"Formula applications: {json.dumps(context.get('formula_applications') or [], sort_keys=True)}\n\n"
            f"Evidence:\n{evidence_block}"
        )
        body = {
            "model": self._identity["model"]["id"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context["prompt"]},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise ProductionBoundaryError(
                    f"missing API key environment variable: {self.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=canonical_bytes(body),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = str(payload["choices"][0]["message"]["content"]).strip()
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise ProductionBoundaryError(
                f"inference transport failed: {type(exc).__name__}"
            ) from exc
        cited = [
            row["node_id"]
            for row in evidence
            if f"[{row['node_id']}]" in text
        ]
        support = cited or [row["node_id"] for row in evidence]
        claim = {
            "label": "MODELED",
            "statement_sha256": text_sha256(text),
            "supporting_node_ids": support,
        }
        return {
            "text": text,
            "output_schema": "szl.answer-with-node-citations/v2",
            "claims": [claim],
            "citations": cited,
            "identity": self.identity(),
            "metrics": {},
        }


__all__ = [
    "ADVISORY_FORMULAS",
    "CONTRACT_PATH",
    "LOCKED_EIGHT",
    "OpenAICompatibleProductionGenerator",
    "ProductionBoundaryError",
    "TRUTH_LABELS",
    "canonical_sha256",
    "load_production_contract",
    "make_second_brain_hydrator",
    "make_second_brain_retriever",
    "make_szl_nemo_envelope_witness",
    "production_infer",
    "text_sha256",
    "verify_external_execution",
]
