from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

import app as legacy
from inference import (
    ProductionBoundaryError,
    load_production_contract,
    make_second_brain_hydrator,
    make_second_brain_retriever,
    make_szl_nemo_envelope_witness,
    production_infer,
)
from inference.production import canonical_sha256, text_sha256


FORGE_CONTROLLER_REVISION = "943f6ab987bbe120cae32649c46c3a5f0b6f9e9b"
SECOND_BRAIN_REVISION = "fa3e4605344b13db220a79f9dcd267ee5725c87e"
NEMO_REVISION = "810231a531188bb569e3faa17396386eb0a5e260"
CONTROLLER_VERSION = "0.2.0"
SECOND_BRAIN_VERSION = "1.2.0"
NEMO_VERSION = "0.4.0"
PUBLIC_PRINCIPAL = "public-anonymous"
PUBLIC_TENANT = "public"
MAX_GOVERNED_K = 4
PUBLIC_POLICY_REVISION = "sha256:" + canonical_sha256(
    {
        "schema": "szl.public-memory-policy/v1",
        "principal": PUBLIC_PRINCIPAL,
        "tenant": PUBLIC_TENANT,
        "second_brain_revision": SECOND_BRAIN_REVISION,
        "access": "PUBLIC_PROJECTION_ONLY",
        "tool_authority": "NONE",
    }
)

app = legacy.app
app.version = "2.0.0"

_components_lock = threading.Lock()
_components_cache: dict[str, Any] | None = None
_anatomy_lock = threading.Lock()
_anatomy_state: dict[str, Any] = {
    "observation_count": 0,
    "last": None,
}


class GovernedInferenceRequest(BaseModel):
    """Public, proposal-only governed inference request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=legacy.MAX_INPUT_CHARS)
    max_new_tokens: int = Field(default=24, ge=1, le=legacy.MAX_NEW_TOKENS)
    k: int = Field(default=3, ge=1, le=MAX_GOVERNED_K)

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("prompt must contain visible text and no NUL bytes")
        if any(token in value for token in legacy.RESERVED_CHAT_TOKENS):
            raise ValueError("prompt contains a reserved chat control token")
        return value


def _public_authorizer(
    principal_id: str,
    tenant_id: str,
    policy_revision: str,
    node_id: str,
    source: str,
) -> bool:
    """Authorize only the packaged public projection under one immutable policy."""

    return bool(
        principal_id == PUBLIC_PRINCIPAL
        and tenant_id == PUBLIC_TENANT
        and policy_revision == PUBLIC_POLICY_REVISION
        and node_id
        and source
    )


def _components() -> dict[str, Any]:
    global _components_cache
    if _components_cache is None:
        with _components_lock:
            if _components_cache is None:
                contract = load_production_contract()
                _components_cache = {
                    "contract": contract,
                    "retriever": make_second_brain_retriever(),
                    "hydrator": make_second_brain_hydrator(_public_authorizer),
                    "witness": make_szl_nemo_envelope_witness(),
                }
    return _components_cache


def _distribution_direct_url(name: str) -> dict[str, Any] | None:
    """Return PEP 610 source metadata when the installer recorded it."""

    try:
        distribution = importlib.metadata.distribution(name)
        for relative in distribution.files or ():
            if str(relative).endswith(".dist-info/direct_url.json"):
                path = distribution.locate_file(relative)
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else None
    except (
        importlib.metadata.PackageNotFoundError,
        OSError,
        json.JSONDecodeError,
    ):
        return None
    return None


def _dependency_status() -> dict[str, Any]:
    expected = {
        "szl-forge-inference": CONTROLLER_VERSION,
        "szl-second-brain": SECOND_BRAIN_VERSION,
        "szl-nemo": NEMO_VERSION,
    }
    observed: dict[str, Any] = {}
    ready = True
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        match = actual == version
        ready = ready and match
        observed[name] = {
            "expected": version,
            "observed": actual,
            "match": match,
            "direct_url": _distribution_direct_url(name),
        }

    contract_ready = False
    contract_error = None
    try:
        contract = load_production_contract()
        contract_ready = bool(
            contract["sources"]["second_brain"]["commit"] == SECOND_BRAIN_REVISION
            and contract["sources"]["nemo_witness"]["commit"] == NEMO_REVISION
            and contract["formula_authority"]["locked_proven_count"] == 8
            and contract["runtime_selection"]["winner"] == "UNSELECTED"
        )
    except Exception as exc:
        contract_error = type(exc).__name__
    ready = ready and contract_ready
    return {
        "ready": ready,
        "packages": observed,
        "contract_ready": contract_ready,
        "contract_error": contract_error,
    }


def _hardware_fingerprint() -> str:
    facts = {
        "space_id": os.getenv("SPACE_ID", legacy.SPACE_ID),
        "accelerator": os.getenv("ACCELERATOR", "none"),
        "cpu_cores": os.getenv("CPU_CORES", "unknown"),
        "memory": os.getenv("MEMORY", "unknown"),
        "machine": platform.machine(),
        "model_sha256": legacy.MODEL_SHA256,
    }
    return "sha256:" + canonical_sha256(facts)


def _clean_fragment(value: str, limit: int) -> str:
    value = value.replace("\x00", " ").replace("\r", " ").strip()
    for token in legacy.RESERVED_CHAT_TOKENS:
        value = value.replace(token, "[reserved-token]")
    value = " ".join(value.split())
    return value[:limit]


def _compose_grounded_prompt(context: Mapping[str, Any]) -> str:
    question = _clean_fragment(str(context.get("prompt") or ""), 420)
    rows: list[str] = []
    for raw in list(context.get("evidence") or ())[:MAX_GOVERNED_K]:
        if not isinstance(raw, Mapping):
            continue
        node_id = _clean_fragment(str(raw.get("node_id") or ""), 100)
        source = _clean_fragment(str(raw.get("source") or ""), 60)
        digest = _clean_fragment(str(raw.get("sha256") or ""), 64)
        content = _clean_fragment(str(raw.get("content") or ""), 220)
        rows.append(f"[{node_id}] source={source} sha256={digest}\n{content}")

    formulas = json.dumps(
        context.get("formula_applications") or [],
        sort_keys=True,
        separators=(",", ":"),
    )
    prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Authorized evidence:\n"
        f"{chr(10).join(rows)}\n\n"
        f"Formula applications: {formulas[:220]}\n\n"
        "Answer briefly from the authorized evidence. Cite node IDs in square "
        "brackets. Do not execute actions, invent benchmarks, claim perfect trust, "
        "or describe Lambda/F23 as proven."
    )
    return prompt[: legacy.MAX_INPUT_CHARS]


def _normalize_text_decision(raw: Any) -> dict[str, Any]:
    if hasattr(raw, "to_dict"):
        value = raw.to_dict()
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise ProductionBoundaryError("Nemo text witness returned an invalid decision")
    decision = str(value.get("decision") or "").upper()
    if decision not in {"ALLOW", "BLOCK", "REVIEW"}:
        raise ProductionBoundaryError("Nemo text witness decision is invalid")
    return {
        "decision": decision,
        "rule_version": str(value.get("rule_version") or ""),
        "input_hash": str(value.get("input_hash") or ""),
        "violated_rules": [
            str(item) for item in value.get("violated_rules") or ()
        ],
        "reasons": [str(item) for item in value.get("reasons") or ()],
    }


class SpaceLlamaGenerator:
    """Adapter over the one existing verified llama.cpp generation path."""

    def __init__(
        self,
        max_new_tokens: int,
        *,
        text_witness: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.max_new_tokens = max(1, min(int(max_new_tokens), legacy.MAX_NEW_TOKENS))
        self._text_witness = text_witness

    def identity(self) -> Mapping[str, Any]:
        source_revision = legacy.observed_source_revision()
        if source_revision is None:
            raise ProductionBoundaryError(
                "governed inference requires an exact deployed source revision"
            )
        if legacy.state["status"] != "READY" or not legacy.state["llama_cpp_version"]:
            raise ProductionBoundaryError("llama.cpp runtime is not ready")
        return {
            "model": {
                "id": legacy.OPENAI_MODEL_ID,
                "revision": legacy.MODEL_REVISION,
                "adapter_revision": "NONE",
                "tokenizer_revision": legacy.MODEL_REVISION,
                "template_revision": source_revision,
                "quantization_revision": "sha256:" + legacy.MODEL_SHA256,
            },
            "runtime": {
                "engine": "llama-cpp-python",
                "version": str(legacy.state["llama_cpp_version"]),
                "hardware_fingerprint": _hardware_fingerprint(),
            },
        }

    def _evaluate_text(self, prompt: str, answer: str) -> dict[str, Any]:
        evaluator = self._text_witness
        if evaluator is None:
            from szl_nemo import evaluate

            evaluator = evaluate
        return _normalize_text_decision(evaluator(prompt, answer))

    def __call__(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        grounded_prompt = _compose_grounded_prompt(context)
        result = legacy.run_bounded_completion(
            legacy.formatted_chat(grounded_prompt),
            self.max_new_tokens,
        )
        answer = str(result.get("output") or "").strip()
        text_decision = self._evaluate_text(
            str(context.get("prompt") or ""),
            answer,
        )
        if text_decision["decision"] != "ALLOW":
            raise ProductionBoundaryError(
                "Nemo R1-R5 blocked or reviewed the generated answer"
            )

        evidence = [
            item
            for item in context.get("evidence") or ()
            if isinstance(item, Mapping) and item.get("node_id")
        ]
        cited = [
            str(item["node_id"])
            for item in evidence
            if f"[{item['node_id']}]" in answer
        ]
        support = cited or [str(item["node_id"]) for item in evidence[:1]]
        return {
            "text": answer,
            "output_schema": "szl.answer-with-node-citations/v2",
            "claims": [
                {
                    "label": "MODELED",
                    "statement_sha256": text_sha256(answer),
                    "supporting_node_ids": support,
                }
            ],
            "citations": cited,
            "identity": self.identity(),
            "metrics": {
                "elapsed_ms": result.get("elapsed_ms"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "finish_reason": result.get("finish_reason"),
                "nemo_text_witness": text_decision,
            },
        }


def _observe_anatomy(event: Mapping[str, Any]) -> None:
    serialized = json.dumps(event, sort_keys=True)
    forbidden = (
        "chain_of_thought",
        "private_chain_of_thought",
        '"prompt":',
        '"content":',
    )
    if any(token in serialized for token in forbidden):
        raise ProductionBoundaryError("unsafe Anatomy observation")
    with _anatomy_lock:
        _anatomy_state["observation_count"] += 1
        _anatomy_state["last"] = copy.deepcopy(dict(event))


def _governed_headers() -> dict[str, str]:
    return {
        **legacy.provenance_headers(),
        "X-SZL-Governed-Inference": "v2",
        "X-SZL-Forge-Controller-Revision": FORGE_CONTROLLER_REVISION,
        "X-SZL-Second-Brain-Revision": SECOND_BRAIN_REVISION,
        "X-SZL-Nemo-Revision": NEMO_REVISION,
        "Cache-Control": "no-store",
    }


@app.get("/api/v2/governed-health")
def governed_health() -> JSONResponse:
    dependency = _dependency_status()
    brain_ready = False
    brain_chunks = None
    component_error = None
    try:
        components = _components()
        retriever = components["retriever"]
        probe = retriever("Lambda proof status", 1)
        brain_ready = bool(
            probe.get("ready")
            and probe.get("content_access") == "HANDLES_ONLY"
            and probe.get("handles")
        )
        brain_chunks = probe.get("corpus_n")
    except Exception as exc:
        component_error = type(exc).__name__

    source_revision = legacy.observed_source_revision()
    ready = bool(
        legacy.state["status"] == "READY"
        and source_revision is not None
        and dependency["ready"]
        and brain_ready
    )
    payload = {
        "schema": "szl.model-inference-lab.governed-health/v2",
        "status": "READY" if ready else "UNAVAILABLE",
        "legacy_model_runtime": legacy.state["status"],
        "source_revision": source_revision,
        "controller_revision": FORGE_CONTROLLER_REVISION,
        "dependency_status": dependency,
        "second_brain": {
            "ready": brain_ready,
            "public_chunk_count": brain_chunks,
            "content_access": "HANDLES_ONLY",
            "private_graph_present": False,
        },
        "nemo": {
            "version": NEMO_VERSION,
            "envelope_rules": "doctrine-v11/E1-E10",
            "text_rules": "doctrine-v11/R1-R5",
        },
        "component_error": component_error,
    }
    return JSONResponse(
        payload,
        status_code=200 if ready else 503,
        headers=_governed_headers(),
    )


@app.get("/.well-known/szl-governed-inference-contract.json")
def governed_contract() -> JSONResponse:
    contract = load_production_contract()
    return JSONResponse(
        {
            "schema": "szl.model-inference-lab.governed-contract/v2",
            "status": "OPERATIONAL_WHEN_GOVERNED_HEALTH_IS_READY",
            "endpoint": {
                "method": "POST",
                "path": "/api/v2/governed-infer",
                "authentication": "PUBLIC_PROJECTION_FIXED_SCOPE",
                "tools": False,
                "max_input_chars": legacy.MAX_INPUT_CHARS,
                "max_completion_tokens": legacy.MAX_NEW_TOKENS,
                "max_retrieval_handles": MAX_GOVERNED_K,
            },
            "controller": {
                "repository": "szl-holdings/szl-forge",
                "revision": FORGE_CONTROLLER_REVISION,
                "version": CONTROLLER_VERSION,
            },
            "second_brain": {
                "revision": SECOND_BRAIN_REVISION,
                "version": SECOND_BRAIN_VERSION,
                "content_access": "HANDLES_ONLY_PUBLIC",
                "authorized_hydration": "IN_PROCESS_CONTROLLER_ONLY",
                "private_graph_present": False,
            },
            "nemo": {
                "revision": NEMO_REVISION,
                "version": NEMO_VERSION,
                "structured_witness": "doctrine-v11/E1-E10",
                "text_witness": "doctrine-v11/R1-R5",
            },
            "formula_authority": contract["formula_authority"],
            "model": {
                "id": legacy.OPENAI_MODEL_ID,
                "revision": legacy.MODEL_REVISION,
                "gguf_sha256": legacy.MODEL_SHA256,
            },
            "authority": {
                "model": "PROPOSAL_ONLY",
                "public_endpoint_tool_execution": "DISABLED",
                "a11oy": "NOT_INVOKED_BY_PUBLIC_ENDPOINT",
                "anatomy": "LOCAL_SANITIZED_OBSERVER_ONLY",
            },
            "runtime_selection": contract["runtime_selection"],
            "privacy": {
                "raw_prompt_persisted": False,
                "hydrated_content_persisted": False,
                "private_reasoning_persisted": False,
                "platform_logging_outside_source": "NOT_ASSERTED",
            },
        },
        headers=_governed_headers(),
    )


@app.get("/api/v2/anatomy/last")
def anatomy_last() -> JSONResponse:
    with _anatomy_lock:
        payload = {
            "schema": "szl.anatomy.local-observer-state/v2",
            "observer_authority": "NONE",
            "observation_count": _anatomy_state["observation_count"],
            "last": copy.deepcopy(_anatomy_state["last"]),
        }
    return JSONResponse(payload, headers=_governed_headers())


@app.post("/api/v2/governed-infer")
def governed_infer(request: GovernedInferenceRequest) -> JSONResponse:
    if legacy.state["status"] != "READY":
        raise HTTPException(status_code=503, detail="model runtime is not ready")
    if legacy.observed_source_revision() is None:
        raise HTTPException(
            status_code=503,
            detail="exact governed source revision is unavailable",
        )
    components = _components()
    result = production_infer(
        {
            "prompt": request.prompt,
            "principal_id": PUBLIC_PRINCIPAL,
            "tenant_id": PUBLIC_TENANT,
            "policy_revision": PUBLIC_POLICY_REVISION,
            "grounding_required": True,
            "k": request.k,
            "formula_applications": [],
        },
        retriever=components["retriever"],
        hydrator=components["hydrator"],
        generator=SpaceLlamaGenerator(request.max_new_tokens),
        witness=components["witness"],
        observer=_observe_anatomy,
    )
    status_code = {
        "PROPOSAL": 200,
        "ABSTAIN": 422,
        "REVIEW": 409,
        "BLOCKED": 503,
    }.get(str(result.get("state")), 500)
    return JSONResponse(
        result,
        status_code=status_code,
        headers=_governed_headers(),
    )


__all__ = [
    "CONTROLLER_VERSION",
    "FORGE_CONTROLLER_REVISION",
    "GovernedInferenceRequest",
    "MAX_GOVERNED_K",
    "NEMO_REVISION",
    "NEMO_VERSION",
    "PUBLIC_POLICY_REVISION",
    "PUBLIC_PRINCIPAL",
    "PUBLIC_TENANT",
    "SECOND_BRAIN_REVISION",
    "SECOND_BRAIN_VERSION",
    "SpaceLlamaGenerator",
    "anatomy_last",
    "app",
    "governed_contract",
    "governed_health",
    "governed_infer",
]
