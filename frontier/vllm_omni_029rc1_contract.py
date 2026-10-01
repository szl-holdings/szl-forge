"""Fail-closed qualification contract for vLLM-Omni 0.29.0rc1.

This module is deliberately network-free and execution-free. It binds the exact
upstream prerelease source admitted by szl-frontier#129/#130, then validates
runtime observations without granting training, serving, publication, or
promotion authority.
"""
from __future__ import annotations

import re

VLLM_OMNI_REPOSITORY = "vllm-project/vllm-omni"
VLLM_OMNI_TAG = "v0.29.0rc1"
VLLM_OMNI_REVISION = "aff7d64948f6c0e81e9b3272234623044e215967"
FRONTIER_ADMISSION_MERGE = "fa16050c930a2f5a258619101a98bfc2013cb783"
BASE_RUNTIME = "vLLM 0.29.0"
OBSERVATION_STATES = frozenset({"NOT_RUN", "UNAVAILABLE", "PASS", "FAIL"})
RUNTIME_STATE_NAMES = (
    "duplexRuntime",
    "serverVAD",
    "nemotronTalker",
    "diffusionBatching",
    "diffusionParallelism",
    "componentOffload",
    "parallelStageInit",
    "mooncakeEndToEnd",
)
DENIED_AUTHORITY = {
    "productionTrainingAuthorized": False,
    "productionServingAuthorized": False,
    "hubPublicationAuthorized": False,
    "weightRehostingAuthorized": False,
    "automaticPromotionAuthorized": False,
    "productionDefaultChangeAuthorized": False,
    "mergeDeployAuthority": False,
}


class QualificationContractError(ValueError):
    """Malformed or contradictory evidence is not qualification evidence."""


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise QualificationContractError(f"{name} must be an explicit boolean")
    return value


def _state(name: str, value: object) -> str:
    if type(value) is not str or value not in OBSERVATION_STATES:
        raise QualificationContractError(
            f"{name} must be one of {sorted(OBSERVATION_STATES)}"
        )
    return value


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise QualificationContractError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def evaluate_observation(
    *,
    observed_source_revision: str,
    artifact_sha256: str | None,
    artifact_bytes_independently_verified: bool,
    artifact_source_bound: bool,
    dependency_closure_captured: bool,
    hardware_identity_state: str,
    duplex_runtime_state: str,
    server_vad_state: str,
    nemotron_talker_state: str,
    diffusion_batching_state: str,
    diffusion_parallelism_state: str,
    component_offload_state: str,
    parallel_stage_init_state: str,
    mooncake_end_to_end_state: str,
    legacy_batching_migration_state: str,
    rollback_state: str,
    model_rights_state: str,
    turn_based_vad_interrupt_disabled_verified: bool,
    mooncake_completion_witness: bool,
    legacy_config_rejected_or_migrated: bool,
    max_num_seqs_semantics_preserved: bool,
) -> dict[str, object]:
    """Validate bounded vLLM-Omni RC evidence.

    ``PASS`` is scoped to one named observation. Even a complete observation
    matrix yields ``EVALUATION`` and never production authority.
    """
    if observed_source_revision != VLLM_OMNI_REVISION:
        raise QualificationContractError(
            "observed source revision does not match the admitted vLLM-Omni tag"
        )

    _boolean(
        "artifact_bytes_independently_verified",
        artifact_bytes_independently_verified,
    )
    _boolean("artifact_source_bound", artifact_source_bound)
    _boolean("dependency_closure_captured", dependency_closure_captured)
    _boolean(
        "turn_based_vad_interrupt_disabled_verified",
        turn_based_vad_interrupt_disabled_verified,
    )
    _boolean("mooncake_completion_witness", mooncake_completion_witness)
    _boolean(
        "legacy_config_rejected_or_migrated",
        legacy_config_rejected_or_migrated,
    )
    _boolean(
        "max_num_seqs_semantics_preserved",
        max_num_seqs_semantics_preserved,
    )

    if artifact_sha256 is not None:
        artifact_sha256 = _sha256("artifact_sha256", artifact_sha256)

    if artifact_bytes_independently_verified and artifact_sha256 is None:
        raise QualificationContractError(
            "independent byte verification requires an observed artifact digest"
        )
    if artifact_source_bound and not artifact_bytes_independently_verified:
        raise QualificationContractError(
            "source-bound artifact evidence requires independently verified bytes"
        )

    states = {
        "hardwareIdentity": _state("hardware_identity_state", hardware_identity_state),
        "duplexRuntime": _state("duplex_runtime_state", duplex_runtime_state),
        "serverVAD": _state("server_vad_state", server_vad_state),
        "nemotronTalker": _state("nemotron_talker_state", nemotron_talker_state),
        "diffusionBatching": _state(
            "diffusion_batching_state", diffusion_batching_state
        ),
        "diffusionParallelism": _state(
            "diffusion_parallelism_state", diffusion_parallelism_state
        ),
        "componentOffload": _state(
            "component_offload_state", component_offload_state
        ),
        "parallelStageInit": _state(
            "parallel_stage_init_state", parallel_stage_init_state
        ),
        "mooncakeEndToEnd": _state(
            "mooncake_end_to_end_state", mooncake_end_to_end_state
        ),
        "legacyBatchingMigration": _state(
            "legacy_batching_migration_state", legacy_batching_migration_state
        ),
        "rollback": _state("rollback_state", rollback_state),
        "modelRights": _state("model_rights_state", model_rights_state),
    }

    runtime_passes = [
        name for name in RUNTIME_STATE_NAMES if states[name] == "PASS"
    ]
    if runtime_passes:
        if not artifact_bytes_independently_verified:
            raise QualificationContractError(
                "runtime PASS requires independently verified artifact bytes"
            )
        if not artifact_source_bound:
            raise QualificationContractError(
                "runtime PASS requires source-to-artifact binding"
            )
        if not dependency_closure_captured:
            raise QualificationContractError(
                "runtime PASS requires captured dependency closure"
            )
        if states["hardwareIdentity"] != "PASS":
            raise QualificationContractError(
                "runtime PASS requires observed hardware identity"
            )
        if states["rollback"] != "PASS":
            raise QualificationContractError(
                "runtime PASS requires rollback evidence"
            )
        if states["modelRights"] != "PASS":
            raise QualificationContractError(
                "model-specific runtime PASS requires model-rights evidence"
            )

    if (
        states["serverVAD"] == "PASS"
        and not turn_based_vad_interrupt_disabled_verified
    ):
        raise QualificationContractError(
            "server VAD PASS requires verification of the turn-based interrupt constraint"
        )

    if states["mooncakeEndToEnd"] == "PASS" and not mooncake_completion_witness:
        raise QualificationContractError(
            "Mooncake end-to-end PASS requires an actual completion witness"
        )

    if states["legacyBatchingMigration"] == "PASS":
        if not legacy_config_rejected_or_migrated:
            raise QualificationContractError(
                "migration PASS requires explicit legacy rejection or migration"
            )
        if not max_num_seqs_semantics_preserved:
            raise QualificationContractError(
                "migration PASS requires preserved max_num_seqs semantics"
            )

    blockers: list[str] = []
    if not artifact_bytes_independently_verified:
        blockers.append("artifact_bytes_not_independently_verified")
    if not artifact_source_bound:
        blockers.append("artifact_not_source_bound")
    if not dependency_closure_captured:
        blockers.append("dependency_closure_not_captured")
    blockers.extend(
        f"{name}:{state}" for name, state in states.items() if state != "PASS"
    )
    if not turn_based_vad_interrupt_disabled_verified:
        blockers.append("turn_based_vad_interrupt_constraint_not_verified")
    if not mooncake_completion_witness:
        blockers.append("mooncake_end_to_end_completion_not_witnessed")
    if not legacy_config_rejected_or_migrated:
        blockers.append("legacy_diffusion_batch_size_migration_not_proven")
    if not max_num_seqs_semantics_preserved:
        blockers.append("max_num_seqs_semantics_not_proven")

    complete = not blockers
    return {
        "schema": "szl.forge.vllm-omni-029rc1-qualification.v1",
        "sourceRepository": VLLM_OMNI_REPOSITORY,
        "sourceTag": VLLM_OMNI_TAG,
        "sourceRevision": VLLM_OMNI_REVISION,
        "frontierAdmissionMerge": FRONTIER_ADMISSION_MERGE,
        "baseRuntime": BASE_RUNTIME,
        "prerelease": True,
        "artifactSha256": artifact_sha256,
        "artifactBytesIndependentlyVerified": artifact_bytes_independently_verified,
        "artifactSourceBound": artifact_source_bound,
        "dependencyClosureCaptured": dependency_closure_captured,
        "observations": states,
        "constraints": {
            "turnBasedVADInterruptDisabledVerified": (
                turn_based_vad_interrupt_disabled_verified
            ),
            "mooncakeCompletionWitness": mooncake_completion_witness,
            "legacyConfigRejectedOrMigrated": legacy_config_rejected_or_migrated,
            "maxNumSeqsSemanticsPreserved": max_num_seqs_semantics_preserved,
        },
        "blockers": blockers,
        "disposition": "EVALUATION" if complete else "HOLD",
        **DENIED_AUTHORITY,
    }
