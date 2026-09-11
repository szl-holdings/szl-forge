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


def _integer(name: str, value: object, *, minimum: int) -> int:
    # bool is an int subclass; floats can compare equal to integer set members.
    # This JSON-facing contract admits builtin integers, never implicit coercion.
    if type(value) is not int or value < minimum:
        raise EvaluationContractError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    # In particular, the nonempty string "false" must not pass a safety check.
    # Do not render the supplied value: invalid inputs may contain sensitive text.
    if type(value) is not bool:
        raise EvaluationContractError(f"{name} must be an explicit boolean")
    return value


def required_max_loras(max_staleness: int) -> int:
    return _integer("max_staleness", max_staleness, minimum=0) + 2


def serving_cache_path(output_dir: Path, candidate: Path) -> Path:
    """Resolve a strict descendant for planning, without creating any files.

    This is a point-in-time path check, NOT a filesystem sandbox or protection
    from concurrent symlink swaps. The runner must enforce its own isolated
    writable directory and revalidate at use; this helper grants no I/O authority.
    """
    if not isinstance(output_dir, Path) or not isinstance(candidate, Path):
        raise EvaluationContractError("cache paths must be pathlib.Path objects")
    try:
        root = output_dir.resolve()
        target = candidate.resolve()
        relative = target.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationContractError("serving cache cannot resolve inside output directory") from exc
    if not relative.parts:
        raise EvaluationContractError("serving cache must be strictly below output directory")
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
    # Validate every supplied field before any condition can produce EVALUATION.
    # False is a valid negative observation; malformed types are contract errors.
    _integer("lora_rank", lora_rank, minimum=1)
    _integer("max_lora_rank", max_lora_rank, minimum=1)
    _integer("max_loras", max_loras, minimum=1)
    required_capacity = required_max_loras(max_staleness)
    _boolean("lora_server_enabled", lora_server_enabled)
    _boolean("shared_storage_verified", shared_storage_verified)
    _boolean("checkpoints_enabled", checkpoints_enabled)
    _boolean("adapter_servable", adapter_servable)
    if max_lora_rank not in ALLOWED_MAX_LORA_RANKS:
        raise EvaluationContractError("max_lora_rank is not an upstream-supported capacity")
    if max_lora_rank < lora_rank:
        raise EvaluationContractError("max_lora_rank is below the adapter rank")

    reasons: list[str] = []
    if not lora_server_enabled:
        reasons.append("vllm_lora_not_enabled")
    if max_loras < required_capacity:
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
        "requiredMaxLoras": required_capacity,
        "reasons": reasons,
        "disposition": "EVALUATION" if not reasons else "HOLD",
        **DENIED_AUTHORITY,
    }
