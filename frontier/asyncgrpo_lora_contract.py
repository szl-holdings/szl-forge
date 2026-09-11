"""Fail-closed contract for the exact-commit TRL AsyncGRPO LoRA evaluation.

This module performs no training, serving, network calls, provider writes, or
runtime adapter updates. It only validates a bounded evaluation plan so missing
preconditions cannot be misread as production authority.
"""
from __future__ import annotations

from pathlib import Path

TRL_REPOSITORY = "huggingface/trl"
TRL_REVISION = "f540773f5250c816e992ae3d35a41142ef3625c0"
TRL_BASELINE_VERSION = "1.13.0"
ALLOWED_MAX_LORA_RANKS = frozenset({1, 8, 16, 32, 64, 128, 256, 320, 512})
DENIED_AUTHORITY = {
    "productionTrainingAuthorized": False,
    "productionServingAuthorized": False,
    "providerCredentialUseAuthorized": False,
    "externalProviderWriteAuthorized": False,
    "hubPublicationAuthorized": False,
    "automaticPromotionAuthorized": False,
    "mergeDeployAuthority": False,
}


class EvaluationContractError(ValueError):
    """Malformed evaluation configuration is not evidence."""


def required_max_loras(max_staleness: int) -> int:
    if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
        raise EvaluationContractError("max_staleness must be a non-negative integer")
    return max_staleness + 2


def serving_cache_path(output_dir: Path, candidate: Path) -> Path:
    """Require the serving cache to stay below the declared output directory."""
    root = output_dir.resolve()
    target = candidate.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise EvaluationContractError("serving cache escapes output directory") from exc
    return target


def evaluate_plan(
    *,
    lora_rank: int,
    max_lora_rank: int,
    max_loras: int,
    max_staleness: int,
    lora_server_enabled: bool,
    shared_storage_verified: bool,
    checkpoints_enabled: bool,
    adapter_servable: bool,
) -> dict[str, object]:
    if not isinstance(lora_rank, int) or isinstance(lora_rank, bool) or lora_rank <= 0:
        raise EvaluationContractError("lora_rank must be a positive integer")
    if max_lora_rank not in ALLOWED_MAX_LORA_RANKS:
        raise EvaluationContractError("max_lora_rank is not an upstream-supported capacity")
    if max_lora_rank < lora_rank:
        raise EvaluationContractError("max_lora_rank is below the adapter rank")
    if not isinstance(max_loras, int) or isinstance(max_loras, bool) or max_loras <= 0:
        raise EvaluationContractError("max_loras must be a positive integer")

    reasons: list[str] = []
    if not lora_server_enabled:
        reasons.append("vllm_lora_not_enabled")
    if max_loras < required_max_loras(max_staleness):
        reasons.append("insufficient_version_capacity")
    if not shared_storage_verified:
        reasons.append("shared_storage_unverified")
    if not checkpoints_enabled:
        reasons.append("durable_checkpoint_missing")
    if not adapter_servable:
        reasons.append("adapter_not_vllm_servable")

    sync_mode = "adapter-only" if not reasons else "merged-fallback-or-unavailable"
    return {
        "sourceRepository": TRL_REPOSITORY,
        "sourceRevision": TRL_REVISION,
        "stableBaselineVersion": TRL_BASELINE_VERSION,
        "syncModeCandidate": sync_mode,
        "requiredMaxLoras": required_max_loras(max_staleness),
        "reasons": reasons,
        "disposition": "EVALUATION" if not reasons else "HOLD",
        **DENIED_AUTHORITY,
    }
