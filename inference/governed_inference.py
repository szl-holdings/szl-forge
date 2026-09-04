"""Proof-carrying inference coordinator for SZL Forge.

The coordinator is engine-neutral. It binds four independently testable planes:
Second Brain retrieval/hydration, a proposal-only model, the deterministic Nemo
witness, and a sanitized Anatomy observer. It never executes tools.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from inference.validate_control_plane import DEFAULT_CONTRACT, load as load_contract

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PINNED_REVISION = re.compile(r"^[0-9a-f]{7,64}$|^sha256:[0-9a-f]{64}$")
LOCKED_EIGHT = ("F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22")
ADVISORY_FORMULAS = ("F23",)
WITNESS_DECISIONS = {"ALLOW", "BLOCK", "REVIEW"}
FORBIDDEN_GENERATOR_KEYS = {
    "chain_of_thought",
    "private_chain_of_thought",
    "hidden_reasoning",
    "reasoning_trace",
}
PRIVATE_REASONING_MARKERS = ("<think>", "</think>", "<analysis>", "</analysis>")


class Retriever(Protocol):
    def __call__(self, query: str, k: int = 6) -> Mapping[str, Any]: ...


class Hydrator(Protocol):
    def __call__(
        self, handles: Sequence[Mapping[str, Any]], request: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]: ...


class Generator(Protocol):
    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


class Witness(Protocol):
    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


Observer = Callable[[Mapping[str, Any]], Any]


class InferenceBoundaryError(ValueError):
    """Raised when untrusted component output violates the inference boundary."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_pinned_revision(value: Any) -> bool:
    token = str(value or "").strip().lower()
    return bool(PINNED_REVISION.fullmatch(token)) and token not in {"main", "master", "latest"}


def _safe_request(request: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(request.get("prompt") or "").strip()
    principal_id = str(request.get("principal_id") or "").strip()
    tenant_id = str(request.get("tenant_id") or "").strip()
    if not prompt:
        raise InferenceBoundaryError("prompt is required")
    if not principal_id or not tenant_id:
        raise InferenceBoundaryError("principal_id and tenant_id are required")
    formula_ids = tuple(str(x) for x in request.get("formula_ids", ()))
    allowed = set(LOCKED_EIGHT) | set(ADVISORY_FORMULAS)
    unknown = sorted(set(formula_ids) - allowed)
    if unknown:
        raise InferenceBoundaryError(f"unknown or unbound formula ids: {unknown}")
    if len(formula_ids) != len(set(formula_ids)):
        raise InferenceBoundaryError("formula_ids must be unique")
    request_id = str(request.get("request_id") or "").strip()
    if not request_id:
        request_id = text_sha256(f"{tenant_id}\0{principal_id}\0{prompt}")[:24]
    tool_intent = request.get("tool_intent")
    if tool_intent is not None and not isinstance(tool_intent, Mapping):
        raise InferenceBoundaryError("tool_intent must be an object when present")
    return {
        "request_id": request_id,
        "principal_id": principal_id,
        "tenant_id": tenant_id,
        "prompt": prompt,
        "formula_ids": formula_ids,
        "requires_grounding": bool(request.get("requires_grounding", True)),
        "tool_intent": copy.deepcopy(dict(tool_intent)) if tool_intent is not None else None,
    }


def _validate_retrieval(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if not isinstance(result, Mapping):
        raise InferenceBoundaryError("retriever must return an object")
    if result.get("content_access") != "HANDLES_ONLY":
        raise InferenceBoundaryError("Second Brain retrieval must remain HANDLES_ONLY")
    handles_raw = result.get("handles") or []
    evidence_raw = result.get("evidence") or []
    if not isinstance(handles_raw, list) or not isinstance(evidence_raw, list):
        raise InferenceBoundaryError("retrieval handles and evidence must be lists")
    handles: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for raw in handles_raw:
        if not isinstance(raw, Mapping):
            raise InferenceBoundaryError("malformed retrieval handle")
        node_id = str(raw.get("nodeId") or "")
        if not node_id:
            raise InferenceBoundaryError("retrieval handle missing nodeId")
        handles.append({
            "nodeId": node_id,
            "nodeKind": str(raw.get("nodeKind") or "INDEX"),
            "label": str(raw.get("label") or "DECLARED"),
            "note": str(raw.get("note") or "")[:160],
        })
    for raw in evidence_raw:
        if not isinstance(raw, Mapping):
            raise InferenceBoundaryError("malformed retrieval evidence")
        node_id = str(raw.get("node_id") or "")
        digest = str(raw.get("sha256") or "").lower()
        source = str(raw.get("source") or "")
        if not node_id or not source or not HEX64.fullmatch(digest):
            raise InferenceBoundaryError("retrieval evidence requires node_id, source, and sha256")
        evidence.append({"node_id": node_id, "sha256": digest, "source": source})
    if [h["nodeId"] for h in handles] != [e["node_id"] for e in evidence]:
        raise InferenceBoundaryError("handle/evidence identity mismatch")
    if bool(result.get("ready")) != bool(handles):
        raise InferenceBoundaryError("retrieval ready state disagrees with evidence")
    digest = canonical_sha256(evidence)
    declared = result.get("evidence_set_sha256")
    if declared is not None and str(declared).lower() != digest:
        raise InferenceBoundaryError("retrieval evidence-set digest mismatch")
    return handles, evidence, digest


def _validate_hydration(
    documents: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = {str(e["node_id"]): e for e in evidence}
    hydrated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in documents:
        if not isinstance(raw, Mapping):
            raise InferenceBoundaryError("hydrator returned a malformed document")
        node_id = str(raw.get("node_id") or "")
        content = str(raw.get("content") or "")
        source = str(raw.get("source") or "")
        if node_id not in expected or node_id in seen:
            raise InferenceBoundaryError("hydrator returned an unknown or duplicate node")
        if source != expected[node_id]["source"]:
            raise InferenceBoundaryError("hydrated source does not match retrieval evidence")
        digest = text_sha256(content)
        if digest != expected[node_id]["sha256"]:
            raise InferenceBoundaryError("hydrated content digest mismatch")
        hydrated.append({
            "node_id": node_id,
            "source": source,
            "sha256": digest,
            "content": content,
        })
        seen.add(node_id)
    if seen != set(expected):
        raise InferenceBoundaryError("hydrator did not resolve the complete evidence set")
    return hydrated


def _validate_witness(result: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise InferenceBoundaryError("witness must return an object")
    decision = str(result.get("decision") or "").upper()
    if decision not in WITNESS_DECISIONS:
        raise InferenceBoundaryError("witness decision must be ALLOW, BLOCK, or REVIEW")
    rule_version = str(result.get("rule_version") or "").strip()
    if not rule_version:
        raise InferenceBoundaryError("witness rule_version is required")
    reason_codes = result.get("reason_codes") or []
    if not isinstance(reason_codes, list) or any(not isinstance(x, str) for x in reason_codes):
        raise InferenceBoundaryError("witness reason_codes must be a string list")
    return {
        "stage": stage,
        "decision": decision,
        "rule_version": rule_version,
        "reason_codes": list(reason_codes),
        "input_sha256": str(result.get("input_sha256") or ""),
    }


def _validate_generator(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise InferenceBoundaryError("generator must return an object")
    forbidden = FORBIDDEN_GENERATOR_KEYS & set(result)
    if forbidden:
        raise InferenceBoundaryError(f"generator attempted to persist private reasoning: {sorted(forbidden)}")
    text = str(result.get("text") or "").strip()
    if not text:
        raise InferenceBoundaryError("generator returned an empty answer")
    lowered = text.lower()
    if any(marker in lowered for marker in PRIVATE_REASONING_MARKERS):
        raise InferenceBoundaryError("generator emitted a private-reasoning marker")
    model = result.get("model")
    runtime = result.get("runtime")
    if not isinstance(model, Mapping) or not isinstance(runtime, Mapping):
        raise InferenceBoundaryError("generator must return model and runtime identity")
    model_id = str(model.get("id") or "").strip()
    model_revision = str(model.get("revision") or "").strip().lower()
    adapter_revision = str(model.get("adapter_revision") or "NONE").strip().lower()
    if not model_id or not _is_pinned_revision(model_revision):
        raise InferenceBoundaryError("model id and pinned model revision are required")
    if adapter_revision != "none" and not _is_pinned_revision(adapter_revision):
        raise InferenceBoundaryError("adapter revision must be NONE or revision-pinned")
    engine = str(runtime.get("engine") or "").strip()
    engine_version = str(runtime.get("version") or "").strip()
    hardware = str(runtime.get("hardware_fingerprint") or "").strip()
    if not engine or not engine_version or not hardware:
        raise InferenceBoundaryError("runtime engine, version, and hardware fingerprint are required")
    output_schema = str(result.get("output_schema") or "").strip()
    if not output_schema:
        raise InferenceBoundaryError("output_schema is required")
    return {
        "text": text,
        "model": {
            "id": model_id,
            "revision": model_revision,
            "adapter_revision": adapter_revision,
        },
        "runtime": {
            "engine": engine,
            "version": engine_version,
            "hardware_fingerprint": hardware,
        },
        "output_schema": output_schema,
        "metrics": copy.deepcopy(dict(result.get("metrics") or {})),
    }


def _receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": result["schema"],
        "request_id": result["request_id"],
        "state": result["state"],
        "authority_state": result["authority_state"],
        "prompt_sha256": result["prompt_sha256"],
        "retrieval_query_sha256": result["retrieval_query_sha256"],
        "evidence_set_sha256": result["evidence_set_sha256"],
        "formula_binding": result["formula_binding"],
        "model": result.get("model"),
        "runtime": result.get("runtime"),
        "output_sha256": result.get("output_sha256"),
        "witness": result.get("witness", []),
        "reason_codes": result.get("reason_codes", []),
        "tool_intent_sha256": result.get("tool_intent_sha256"),
        "executed": False,
    }
    digest = canonical_sha256(payload)
    return {
        "schema": "szl.forge.inference-receipt/v1",
        "canonicalization": "utf8-json-sort-keys-compact",
        "algorithm": "sha256",
        "payload": payload,
        "receipt_sha256": digest,
        "signature": {
            "status": "UNSIGNED_LOCAL",
            "must_be_signed_before_consequential_action": True,
        },
    }


def _base_result(
    request: Mapping[str, Any], contract: Mapping[str, Any], evidence_digest: str = canonical_sha256([])
) -> dict[str, Any]:
    formula = contract["formula_binding"]
    return {
        "schema": "szl.forge.governed-inference/v1",
        "request_id": request["request_id"],
        "principal_id_sha256": text_sha256(request["principal_id"]),
        "tenant_id_sha256": text_sha256(request["tenant_id"]),
        "prompt_sha256": text_sha256(request["prompt"]),
        "retrieval_query_sha256": text_sha256(request["prompt"]),
        "evidence_set_sha256": evidence_digest,
        "formula_binding": {
            "requested_ids": list(request["formula_ids"]),
            "locked_proven_ids": list(LOCKED_EIGHT),
            "formal_source_commit": formula["formal_source"]["commit"],
            "kernel_source_commit": formula["kernel_source"]["commit"],
            "lambda_status": formula["lambda"]["status"],
        },
        "executed": False,
    }


def _finalize(result: dict[str, Any], observer: Observer | None = None) -> dict[str, Any]:
    event = {
        "schema": "szl.anatomy.inference-observation/v1",
        "request_id": result["request_id"],
        "state": result["state"],
        "authority_state": result["authority_state"],
        "prompt_sha256": result["prompt_sha256"],
        "evidence_set_sha256": result["evidence_set_sha256"],
        "formula_ids": result["formula_binding"]["requested_ids"],
        "output_sha256": result.get("output_sha256"),
        "reason_codes": result.get("reason_codes", []),
        "raw_prompt_present": False,
        "private_reasoning_present": False,
        "observer_authority": "NONE",
    }
    delivery = "NOT_CONFIGURED"
    if observer is not None:
        try:
            observer(copy.deepcopy(event))
            delivery = "DELIVERED"
        except Exception:
            delivery = "UNAVAILABLE"
    result["anatomy_observation"] = {"delivery": delivery, "event": event}
    result["receipt"] = _receipt(result)
    return result


def _terminal(
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    state: str,
    authority_state: str,
    reason_codes: Sequence[str],
    witness: Sequence[Mapping[str, Any]] = (),
    evidence_digest: str = canonical_sha256([]),
    observer: Observer | None = None,
) -> dict[str, Any]:
    result = _base_result(request, contract, evidence_digest)
    result.update({
        "state": state,
        "authority_state": authority_state,
        "reason_codes": list(reason_codes),
        "witness": [copy.deepcopy(dict(x)) for x in witness],
        "output": None,
        "output_sha256": None,
        "tool_intent_sha256": canonical_sha256(request["tool_intent"]) if request["tool_intent"] else None,
    })
    return _finalize(result, observer)


def governed_infer(
    request: Mapping[str, Any],
    *,
    retriever: Retriever,
    hydrator: Hydrator,
    generator: Generator,
    witness: Witness,
    observer: Observer | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
    k: int = 6,
) -> dict[str, Any]:
    """Produce a governed proposal and unsigned local receipt; never execute a tool."""
    contract = load_contract(contract_path)
    try:
        safe = _safe_request(request)
    except InferenceBoundaryError as exc:
        fallback = {
            "request_id": text_sha256(canonical_bytes(dict(request)).decode("utf-8"))[:24],
            "principal_id": str(request.get("principal_id") or "UNAVAILABLE"),
            "tenant_id": str(request.get("tenant_id") or "UNAVAILABLE"),
            "prompt": str(request.get("prompt") or ""),
            "formula_ids": (),
            "requires_grounding": True,
            "tool_intent": None,
        }
        return _terminal(
            fallback, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["REQUEST_BOUNDARY_INVALID", str(exc)],
            observer=observer,
        )

    if safe["tool_intent"] is not None and safe["formula_ids"] and set(safe["formula_ids"]) <= {"F23"}:
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["LAMBDA_CANNOT_AUTHORIZE_ACTION"],
            observer=observer,
        )

    try:
        retrieved = retriever(safe["prompt"], max(1, min(int(k), 12)))
        handles, evidence, evidence_digest = _validate_retrieval(retrieved)
    except Exception as exc:
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["RETRIEVAL_BOUNDARY_INVALID", type(exc).__name__],
            observer=observer,
        )
    if safe["requires_grounding"] and not handles:
        return _terminal(
            safe, contract,
            state="ABSTAIN",
            authority_state="NONE",
            reason_codes=["NO_GROUNDED_EVIDENCE"],
            evidence_digest=evidence_digest,
            observer=observer,
        )

    try:
        hydrated = _validate_hydration(hydrator(handles, safe), evidence)
    except Exception as exc:
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["HYDRATION_BOUNDARY_INVALID", type(exc).__name__],
            evidence_digest=evidence_digest,
            observer=observer,
        )

    witness_records: list[dict[str, Any]] = []
    pre_payload = {
        "prompt": safe["prompt"],
        "prompt_sha256": text_sha256(safe["prompt"]),
        "formula_ids": list(safe["formula_ids"]),
        "evidence_set_sha256": evidence_digest,
        "tool_intent": safe["tool_intent"],
    }
    try:
        pre = _validate_witness(witness("PRE_GENERATION", pre_payload), "PRE_GENERATION")
    except Exception as exc:
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["NEMO_PRE_WITNESS_INVALID", type(exc).__name__],
            evidence_digest=evidence_digest,
            observer=observer,
        )
    witness_records.append(pre)
    if pre["decision"] == "BLOCK":
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["NEMO_PRE_GENERATION_BLOCK"] + pre["reason_codes"],
            witness=witness_records,
            evidence_digest=evidence_digest,
            observer=observer,
        )

    generation_context = {
        "schema": "szl.forge.generation-context/v1",
        "request_id": safe["request_id"],
        "prompt": safe["prompt"],
        "evidence": hydrated,
        "formula_binding": {
            "requested_ids": list(safe["formula_ids"]),
            "locked_proven_ids": list(LOCKED_EIGHT),
            "advisory_ids": list(ADVISORY_FORMULAS),
            "lambda_can_authorize": False,
            "formal_source_commit": contract["formula_binding"]["formal_source"]["commit"],
            "kernel_source_commit": contract["formula_binding"]["kernel_source"]["commit"],
        },
        "instructions": {
            "authority": "PROPOSAL_ONLY",
            "cite_node_ids": True,
            "expose_private_chain_of_thought": False,
            "output_contract": "answer plus cited node IDs; no action execution",
        },
    }
    try:
        generated = _validate_generator(generator(generation_context))
    except Exception as exc:
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["GENERATOR_BOUNDARY_INVALID", type(exc).__name__],
            witness=witness_records,
            evidence_digest=evidence_digest,
            observer=observer,
        )

    post_payload = {
        "prompt": safe["prompt"],
        "answer": generated["text"],
        "answer_sha256": text_sha256(generated["text"]),
        "formula_ids": list(safe["formula_ids"]),
        "evidence_set_sha256": evidence_digest,
        "tool_intent": safe["tool_intent"],
    }
    try:
        post = _validate_witness(witness("POST_GENERATION", post_payload), "POST_GENERATION")
    except Exception as exc:
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["NEMO_POST_WITNESS_INVALID", type(exc).__name__],
            witness=witness_records,
            evidence_digest=evidence_digest,
            observer=observer,
        )
    witness_records.append(post)
    if post["decision"] == "BLOCK":
        return _terminal(
            safe, contract,
            state="BLOCKED",
            authority_state="NONE",
            reason_codes=["NEMO_POST_GENERATION_BLOCK"] + post["reason_codes"],
            witness=witness_records,
            evidence_digest=evidence_digest,
            observer=observer,
        )

    tool_intent_sha = canonical_sha256(safe["tool_intent"]) if safe["tool_intent"] else None
    requires_review = (
        safe["tool_intent"] is not None
        or pre["decision"] == "REVIEW"
        or post["decision"] == "REVIEW"
    )
    state = "REVIEW" if requires_review else "PROPOSAL"
    authority_state = "HUMAN_APPROVAL_REQUIRED" if requires_review else "NO_ACTION_AUTHORITY"
    result = _base_result(safe, contract, evidence_digest)
    result.update({
        "state": state,
        "authority_state": authority_state,
        "reason_codes": ["TOOL_INTENT_REQUIRES_A11OY_ADMISSION"] if safe["tool_intent"] else [],
        "witness": witness_records,
        "output": generated["text"],
        "output_sha256": text_sha256(generated["text"]),
        "output_schema": generated["output_schema"],
        "model": generated["model"],
        "runtime": generated["runtime"],
        "metrics": generated["metrics"],
        "evidence_handles": handles,
        "tool_intent_sha256": tool_intent_sha,
    })
    return _finalize(result, observer)


def make_second_brain_retriever() -> Retriever:
    """Bind to the installed szl-second-brain package without adding a hard dependency."""
    try:
        from second_brain import navigator_context  # type: ignore
    except ImportError as exc:
        raise InferenceBoundaryError("szl-second-brain is not installed") from exc
    return navigator_context


def make_public_jsonl_hydrator(corpus_path: Path) -> Hydrator:
    """Create a digest-verifying hydrator for the public Second Brain projection."""
    rows: dict[str, dict[str, Any]] = {}
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("id"):
            continue
        content = str(row.get("text") or "")
        rows[str(row["id"])] = {
            "node_id": str(row["id"]),
            "source": str(row.get("source") or "unknown"),
            "sha256": text_sha256(content),
            "content": content,
        }

    def hydrate(handles: Sequence[Mapping[str, Any]], request: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        del request
        return [copy.deepcopy(rows[h["nodeId"]]) for h in handles if h.get("nodeId") in rows]

    return hydrate


def make_szl_nemo_witness() -> Witness:
    """Adapt the installed deterministic szl-nemo engine to staged witness calls."""
    try:
        from szl_nemo import evaluate  # type: ignore
    except ImportError as exc:
        raise InferenceBoundaryError("szl-nemo is not installed") from exc

    def staged(stage: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = str(payload.get("prompt") or "")
        answer = str(payload.get("answer") or "")
        if stage == "PRE_GENERATION" and not answer:
            answer = (
                "PRE_GENERATION policy check only. Proposal generation has no action authority; "
                "no live, benchmark, proof, or fine-tuning claim is made at this stage."
            )
        decision = evaluate(prompt, answer)
        reason_codes = list(decision.violated_rules) or list(decision.reasons)
        return {
            "decision": str(decision.decision),
            "rule_version": str(decision.rule_version),
            "reason_codes": reason_codes,
            "input_sha256": str(decision.input_hash),
            "stage": stage,
        }

    return staged


class OpenAICompatibleGenerator:
    """Small adapter for llama.cpp, vLLM, or SGLang OpenAI-compatible servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str = "NONE",
        engine: str,
        engine_version: str,
        hardware_fingerprint: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 45.0,
        max_tokens: int = 256,
    ) -> None:
        base = base_url.rstrip("/")
        if not (base.startswith("https://") or base.startswith("http://127.0.0.1") or base.startswith("http://localhost")):
            raise InferenceBoundaryError("inference endpoint must be HTTPS or loopback HTTP")
        if not _is_pinned_revision(model_revision):
            raise InferenceBoundaryError("model_revision must be pinned")
        if adapter_revision.lower() != "none" and not _is_pinned_revision(adapter_revision):
            raise InferenceBoundaryError("adapter_revision must be NONE or pinned")
        self.base_url = base
        self.model_id = model_id
        self.model_revision = model_revision
        self.adapter_revision = adapter_revision
        self.engine = engine
        self.engine_version = engine_version
        self.hardware_fingerprint = hardware_fingerprint
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_tokens = int(max_tokens)

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        evidence = context.get("evidence") or []
        evidence_block = "\n\n".join(
            f"[{row['node_id']}] source={row['source']} sha256={row['sha256']}\n{row['content']}"
            for row in evidence
        )
        formula = context.get("formula_binding") or {}
        system = (
            "You are a proposal-only SZL inference engine. Use only the supplied evidence. "
            "Cite supporting node IDs in square brackets. Do not execute actions, invent evidence, "
            "or expose private chain-of-thought. Formula IDs are constraints with the exact status "
            "provided; F23/Lambda is advisory and cannot authorize an action.\n\n"
            f"Formula binding: {json.dumps(formula, sort_keys=True)}\n\nEvidence:\n{evidence_block}"
        )
        body = {
            "model": self.model_id,
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
                raise InferenceBoundaryError(f"missing API key environment variable: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=canonical_bytes(body),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise InferenceBoundaryError(f"inference transport failed: {type(exc).__name__}") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceBoundaryError("OpenAI-compatible response shape invalid") from exc
        return {
            "text": str(text),
            "output_schema": "szl.answer-with-node-citations/v1",
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "adapter_revision": self.adapter_revision,
            },
            "runtime": {
                "engine": self.engine,
                "version": self.engine_version,
                "hardware_fingerprint": self.hardware_fingerprint,
            },
            "metrics": {},
        }
