"""Fail-closed host probe for exact-commit TRL AsyncGRPO LoRA / vLLM evaluation.

This module does not train, serve, install packages, open network sockets,
write adapters, or invent GPU measurements. Missing hardware is UNAVAILABLE,
never PASS.
"""
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
from typing import Any, Mapping

from frontier.asyncgrpo_lora_contract import (
    DENIED_AUTHORITY,
    EvaluationContractError,
    TRL_BASELINE_VERSION,
    TRL_REPOSITORY,
    TRL_REVISION,
    evaluate_plan,
)

SCHEMA = "szl.asyncgrpo-lora-vllm-evaluation/v1"
ALLOWED_DISPOSITIONS = frozenset({"UNAVAILABLE", "HOLD", "EVALUATION"})
REQUIRED_EVIDENCE = (
    "exact-trl-commit-install-receipt",
    "pep610-direct-url-provenance",
    "compatible-exact-peft-vllm-closure",
    "job-local-env-not-production-runtime",
    "transferred-sync-bytes",
    "sync-pause",
    "generation-throughput",
    "policy-version-correctness",
    "max-staleness",
    "adapter-rank-and-capacity",
    "loaded-evicted-adapter-identities",
    "checkpoint-presence",
    "resource-use",
    "adapter-only-vs-merged-equivalent-conditions",
    "vllm-without-lora-support",
    "runtime-adapter-update-unavailable",
    "unsupported-or-insufficient-max-rank",
    "insufficient-max-loras",
    "shared-path-unavailable",
    "path-escape-symlink",
    "partially-written-adapter",
    "load-failure",
    "load-before-evict-ordering",
    "server-restart",
    "serving-cache-eviction",
    "checkpoint-missing",
    "hybrid-recurrent-state-logprob-mismatch",
    "serving-cache-ephemeral-not-durable-artifact",
    "adapter-only-sync-api-at-pinned-trl-revision",
)
MEASURED_FIELDS = (
    "adapterOnlyTransferredBytes",
    "mergedTransferredBytes",
    "adapterOnlySyncPauseSeconds",
    "mergedSyncPauseSeconds",
    "adapterOnlyGenerationThroughput",
    "mergedGenerationThroughput",
    "policyVersionCorrect",
    "resourceUse",
)
DEFAULT_PLAN = {
    "lora_rank": 32,
    "max_lora_rank": 32,
    "max_loras": 6,
    "max_staleness": 4,
    "lora_server_enabled": True,
    "shared_storage_verified": True,
    "checkpoints_enabled": True,
    "adapter_servable": True,
}
STABLE_BASELINE_REVISION = "3d9261f1fec9f9a8140099c78a65c7da73dce79c"
ADAPTER_ONLY_UPSTREAM = "https://github.com/huggingface/trl/issues/5975"


class EvaluationRunnerError(EvaluationContractError):
    """Host probe or measurement admission failed closed."""


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise EvaluationRunnerError(f"{name} must be an explicit boolean")
    return value


def _optional_number(name: str, value: object) -> int | float:
    if type(value) is bool or type(value) not in (int, float):
        raise EvaluationRunnerError(f"{name} must be a builtin int or float")
    return value


def package_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def installed_revision(package: str) -> str | None:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return None
    document = distribution.read_text("direct_url.json")
    if not document or len(document) > 16384:
        return None
    try:
        payload = json.loads(document)
    except json.JSONDecodeError:
        return None
    vcs = payload.get("vcs_info") if isinstance(payload, dict) else None
    if not isinstance(vcs, dict):
        return None
    commit = vcs.get("commit_id")
    return commit if isinstance(commit, str) else None


def cuda_available() -> bool:
    if not package_present("torch"):
        return False
    try:
        import torch
    except Exception:
        return False
    available = getattr(getattr(torch, "cuda", None), "is_available", None)
    if not callable(available):
        return False
    try:
        return _boolean("torch.cuda.is_available", available())
    except Exception:
        return False


def probe_host() -> dict[str, object]:
    trl_revision = installed_revision("trl")
    return {
        "trlPresent": package_present("trl"),
        "peftPresent": package_present("peft"),
        "vllmPresent": package_present("vllm"),
        "torchPresent": package_present("torch"),
        "cudaAvailable": cuda_available(),
        "trlInstalledRevision": trl_revision,
        "exactTrlRevisionMatched": trl_revision == TRL_REVISION,
        "adapterOnlySyncApiAssumed": False,
    }


def hardware_ready(probe: Mapping[str, object]) -> bool:
    return (
        probe.get("cudaAvailable") is True
        and probe.get("vllmPresent") is True
        and probe.get("exactTrlRevisionMatched") is True
        and probe.get("peftPresent") is True
    )


def admit_measurements(probe: Mapping[str, object], measurements: object) -> dict[str, object]:
    admitted = {field: "UNAVAILABLE" for field in MEASURED_FIELDS}
    if measurements is None:
        return admitted
    if not isinstance(measurements, dict):
        raise EvaluationRunnerError("measurements must be a mapping or None")
    if not hardware_ready(probe):
        raise EvaluationRunnerError(
            "measurements are not admissible without exact TRL, PEFT, vLLM and CUDA"
        )
    for field in MEASURED_FIELDS:
        if field not in measurements:
            raise EvaluationRunnerError("measured comparison is incomplete")
        value = measurements[field]
        if field == "policyVersionCorrect":
            admitted[field] = _boolean(field, value)
        elif field == "resourceUse":
            if not isinstance(value, dict):
                raise EvaluationRunnerError("resourceUse must be a mapping")
            admitted[field] = {
                str(key): _optional_number(str(key), item) for key, item in value.items()
            }
        else:
            admitted[field] = _optional_number(field, value)
    return admitted


def evaluate_host(
    *,
    plan: Mapping[str, Any] | None = None,
    measurements: Mapping[str, object] | None = None,
) -> dict[str, object]:
    probe = probe_host()
    plan_result = evaluate_plan(**dict(DEFAULT_PLAN if plan is None else plan))
    reasons: list[str] = list(plan_result["reasons"])
    if probe["trlPresent"] is not True:
        reasons.append("trl_uninstalled")
    elif probe["exactTrlRevisionMatched"] is not True:
        reasons.append("trl_revision_unmatched")
    if probe["peftPresent"] is not True:
        reasons.append("peft_uninstalled")
    if probe["vllmPresent"] is not True:
        reasons.append("vllm_uninstalled")
    if probe["cudaAvailable"] is not True:
        reasons.append("cuda_unavailable")
    reasons.append("adapter_only_sync_api_unverified_at_pin")
    if measurements is None:
        reasons.append("measured_comparison_absent")
    admitted = admit_measurements(probe, measurements)
    if hardware_ready(probe) and measurements is not None and not plan_result["reasons"]:
        disposition = "EVALUATION"
        satisfied: list[str] = []
    elif hardware_ready(probe):
        disposition = "HOLD"
        satisfied = []
    else:
        disposition = "UNAVAILABLE"
        satisfied = []
    if disposition not in ALLOWED_DISPOSITIONS:
        raise EvaluationRunnerError("disposition escaped the allowed set")
    return {
        "schema": SCHEMA,
        "sourceRepository": TRL_REPOSITORY,
        "sourceRevision": TRL_REVISION,
        "stableBaselineVersion": TRL_BASELINE_VERSION,
        "stableBaselineRevision": STABLE_BASELINE_REVISION,
        "upstreamAdapterOnlyIssue": ADAPTER_ONLY_UPSTREAM,
        "probe": probe,
        "plan": plan_result,
        "measured": admitted,
        "requiredEvidence": list(REQUIRED_EVIDENCE),
        "satisfiedEvidence": satisfied,
        "missingEvidence": list(REQUIRED_EVIDENCE),
        "reasons": reasons,
        "disposition": disposition,
        "evaluationOnly": True,
        "productionEligible": False,
        **DENIED_AUTHORITY,
    }
