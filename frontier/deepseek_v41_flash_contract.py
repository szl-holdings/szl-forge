"""Fail-closed qualification contract for DeepSeek-V4.1-Flash.

This module binds the exact Hugging Face source admitted by szl-frontier#134/#135.
It validates evidence supplied by source-preflight, protocol-fixture, runtime,
hardware, rollback and rights checks. It never grants production authority.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

SOURCE_REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
SOURCE_REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
FRONTIER_ADMISSION_MERGE = "5839c1b9d2fae5477d5c693a0d1684398c75224b"
MODEL_TYPE = "deepseek_v41"
REQUIRED_SOURCE_FILES = (
    "README.md",
    "config.json",
    "LICENSE",
    "encoding/README.md",
    "encoding/encoding.py",
)
OBSERVATION_STATES = frozenset({"NOT_RUN", "UNAVAILABLE", "PASS", "FAIL"})
PROTOCOL_CASES = (
    "text_multiturn",
    "image_text_interleaved",
    "numeric_reasoning_effort",
    "tool_call_roundtrip",
    "malformed_tool_negative",
    "mid_conversation_system",
    "chat_completions_complete",
    "chat_completions_streamed",
    "responses_complete",
    "responses_streamed",
    "generic_jinja_rejected",
)
RUNTIME_STATES = (
    "baseline_decode",
    "vision_input",
    "ced",
    "csa2",
    "swa_bounded_replay",
    "fp4_kv",
    "engram",
    "dspark",
    "bounded_context_ladder",
    "oom_cancellation_restart",
    "cache_mismatch_rollback",
)
DENIED_AUTHORITY = {
    "productionTrainingAuthorized": False,
    "productionServingAuthorized": False,
    "hubPublicationAuthorized": False,
    "weightRehostingAuthorized": False,
    "automaticPromotionAuthorized": False,
    "productionDefaultChangeAuthorized": False,
    "toolExecutionAuthorized": False,
    "mergeDeployAuthority": False,
}


class QualificationContractError(ValueError):
    """Malformed, contradictory or source-drifted evidence."""


def _state(name: str, value: object) -> str:
    if type(value) is not str or value not in OBSERVATION_STATES:
        raise QualificationContractError(
            f"{name} must be one of {sorted(OBSERVATION_STATES)}"
        )
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise QualificationContractError(f"{name} must be an explicit boolean")
    return value


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise QualificationContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _digest_map(name: str, value: object, required_keys: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise QualificationContractError(f"{name} must be a mapping")
    keys = set(value)
    if keys != set(required_keys):
        raise QualificationContractError(
            f"{name} keys must exactly match {sorted(required_keys)}"
        )
    return {key: _sha256(f"{name}[{key}]", value[key]) for key in required_keys}


def _state_map(name: str, value: object, required_keys: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise QualificationContractError(f"{name} must be a mapping")
    keys = set(value)
    if keys != set(required_keys):
        raise QualificationContractError(
            f"{name} keys must exactly match {sorted(required_keys)}"
        )
    return {key: _state(f"{name}[{key}]", value[key]) for key in required_keys}


def evaluate_observation(
    *,
    observed_source_revision: str,
    observed_model_type: str,
    source_file_sha256: Mapping[str, str],
    source_bytes_independently_verified: bool,
    checkpoint_inventory_captured: bool,
    checkpoint_inventory_source_bound: bool,
    protocol_states: Mapping[str, str],
    generic_jinja_fallback_possible: bool,
    runtime_states: Mapping[str, str],
    runtime_identity_sha256: str | None,
    dependency_closure_captured: bool,
    hardware_identity_state: str,
    rollback_state: str,
    model_rights_state: str,
    measured_claims_source_bound: bool,
) -> dict[str, object]:
    """Validate bounded evidence for the admitted DeepSeek V4.1 Flash source."""
    if observed_source_revision != SOURCE_REVISION:
        raise QualificationContractError("source revision does not match admitted revision")
    if observed_model_type != MODEL_TYPE:
        raise QualificationContractError("model_type does not match deepseek_v41")

    source_hashes = _digest_map("source_file_sha256", source_file_sha256, REQUIRED_SOURCE_FILES)
    protocol = _state_map("protocol_states", protocol_states, PROTOCOL_CASES)
    runtime = _state_map("runtime_states", runtime_states, RUNTIME_STATES)

    _boolean("source_bytes_independently_verified", source_bytes_independently_verified)
    _boolean("checkpoint_inventory_captured", checkpoint_inventory_captured)
    _boolean("checkpoint_inventory_source_bound", checkpoint_inventory_source_bound)
    _boolean("generic_jinja_fallback_possible", generic_jinja_fallback_possible)
    _boolean("dependency_closure_captured", dependency_closure_captured)
    _boolean("measured_claims_source_bound", measured_claims_source_bound)

    hardware_identity_state = _state("hardware_identity_state", hardware_identity_state)
    rollback_state = _state("rollback_state", rollback_state)
    model_rights_state = _state("model_rights_state", model_rights_state)

    if runtime_identity_sha256 is not None:
        runtime_identity_sha256 = _sha256("runtime_identity_sha256", runtime_identity_sha256)

    if checkpoint_inventory_source_bound and not checkpoint_inventory_captured:
        raise QualificationContractError(
            "source-bound checkpoint inventory requires captured inventory"
        )
    if any(state == "PASS" for state in protocol.values()) and not source_bytes_independently_verified:
        raise QualificationContractError("protocol PASS requires independently verified source bytes")
    if generic_jinja_fallback_possible:
        raise QualificationContractError("generic Jinja fallback must fail closed")

    runtime_passes = [name for name, state in runtime.items() if state == "PASS"]
    if runtime_passes:
        if runtime_identity_sha256 is None:
            raise QualificationContractError("runtime PASS requires immutable runtime identity")
        if not dependency_closure_captured:
            raise QualificationContractError("runtime PASS requires dependency closure")
        if hardware_identity_state != "PASS":
            raise QualificationContractError("runtime PASS requires hardware identity")
        if rollback_state != "PASS":
            raise QualificationContractError("runtime PASS requires rollback evidence")
        if model_rights_state != "PASS":
            raise QualificationContractError("runtime PASS requires model-rights evidence")
        if not checkpoint_inventory_source_bound:
            raise QualificationContractError("runtime PASS requires source-bound checkpoint inventory")

    if runtime["dspark"] == "PASS" and runtime["baseline_decode"] != "PASS":
        raise QualificationContractError("DSpark PASS requires baseline decode PASS first")
    if runtime["bounded_context_ladder"] == "PASS" and runtime["baseline_decode"] != "PASS":
        raise QualificationContractError("context-ladder PASS requires baseline decode PASS")
    if measured_claims_source_bound and not runtime_passes:
        raise QualificationContractError("measured claims require at least one runtime PASS")

    blockers: list[str] = []
    if not source_bytes_independently_verified:
        blockers.append("source_bytes_not_independently_verified")
    if not checkpoint_inventory_captured:
        blockers.append("checkpoint_inventory_not_captured")
    if not checkpoint_inventory_source_bound:
        blockers.append("checkpoint_inventory_not_source_bound")
    blockers.extend(f"protocol:{k}:{v}" for k, v in protocol.items() if v != "PASS")
    blockers.extend(f"runtime:{k}:{v}" for k, v in runtime.items() if v != "PASS")
    if runtime_identity_sha256 is None:
        blockers.append("runtime_identity_not_pinned")
    if not dependency_closure_captured:
        blockers.append("dependency_closure_not_captured")
    if hardware_identity_state != "PASS":
        blockers.append(f"hardware_identity:{hardware_identity_state}")
    if rollback_state != "PASS":
        blockers.append(f"rollback:{rollback_state}")
    if model_rights_state != "PASS":
        blockers.append(f"model_rights:{model_rights_state}")
    if not measured_claims_source_bound:
        blockers.append("measured_claims_not_source_bound")

    return {
        "schema": "szl.forge.deepseek-v41-flash-qualification.v1",
        "sourceRepository": SOURCE_REPOSITORY,
        "sourceRevision": SOURCE_REVISION,
        "frontierAdmissionMerge": FRONTIER_ADMISSION_MERGE,
        "modelType": MODEL_TYPE,
        "sourceFileSha256": source_hashes,
        "sourceBytesIndependentlyVerified": source_bytes_independently_verified,
        "checkpointInventoryCaptured": checkpoint_inventory_captured,
        "checkpointInventorySourceBound": checkpoint_inventory_source_bound,
        "protocolObservations": protocol,
        "genericJinjaFallbackPossible": generic_jinja_fallback_possible,
        "runtimeObservations": runtime,
        "runtimeIdentitySha256": runtime_identity_sha256,
        "dependencyClosureCaptured": dependency_closure_captured,
        "hardwareIdentity": hardware_identity_state,
        "rollback": rollback_state,
        "modelRights": model_rights_state,
        "measuredClaimsSourceBound": measured_claims_source_bound,
        "blockers": blockers,
        "disposition": "EVALUATION" if not blockers else "HOLD",
        **DENIED_AUTHORITY,
    }
