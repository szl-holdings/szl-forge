"""Fail-closed qualification contract for Tencent AuK / AuK-Flash.

Binds the exact Hugging Face snapshots admitted by szl-frontier#136/#137 and
validates task-separated, consent-aware speech evidence. It grants no production
or publication authority.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

SOURCE_REVISIONS = {
    "tencent/AuK": "43c9447eb0a8085d1694b2eb4963051dda8ed231",
    "tencent/AuK-Flash": "575b92f0895f75180bf2cbd35f2e176c5732b8ed",
}
TASKS = (
    "tts",
    "content_editing",
    "acoustic_editing",
    "paralinguistic_editing",
    "speech_enhancement",
    "source_separation",
)
OBSERVATION_STATES = frozenset({"NOT_RUN", "UNAVAILABLE", "PASS", "FAIL"})
DENIED_AUTHORITY = {
    "productionServingAuthorized": False,
    "publicVoiceCloningAuthorized": False,
    "hubPublicationAuthorized": False,
    "weightRehostingAuthorized": False,
    "automaticPromotionAuthorized": False,
    "productionDefaultChangeAuthorized": False,
    "autonomousAudioActionAuthorized": False,
    "mergeDeployAuthority": False,
}


class QualificationContractError(ValueError):
    """Malformed or rights-unsafe evidence is not qualification evidence."""


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


def _state_map(name: str, value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(TASKS):
        raise QualificationContractError(f"{name} keys must exactly match {sorted(TASKS)}")
    return {task: _state(f"{name}[{task}]", value[task]) for task in TASKS}


def evaluate_observation(
    *,
    observed_revisions: Mapping[str, str],
    source_manifest_sha256: str,
    source_bytes_independently_verified: bool,
    artifact_inventory_captured: bool,
    artifact_inventory_source_bound: bool,
    base_task_states: Mapping[str, str],
    flash_task_states: Mapping[str, str],
    consented_fixture_set_verified: bool,
    speaker_provenance_captured: bool,
    sensitive_raw_audio_public: bool,
    runtime_identity_sha256: str | None,
    dependency_closure_captured: bool,
    hardware_identity_state: str,
    malformed_audio_state: str,
    cancellation_restart_state: str,
    rollback_state: str,
    model_rights_state: str,
    provenance_receipt_state: str,
    flash_four_step_claim_measured: bool,
) -> dict[str, object]:
    if not isinstance(observed_revisions, Mapping) or dict(observed_revisions) != SOURCE_REVISIONS:
        raise QualificationContractError("observed revisions must exactly match admitted AuK snapshots")

    source_manifest_sha256 = _sha256("source_manifest_sha256", source_manifest_sha256)
    base = _state_map("base_task_states", base_task_states)
    flash = _state_map("flash_task_states", flash_task_states)

    _boolean("source_bytes_independently_verified", source_bytes_independently_verified)
    _boolean("artifact_inventory_captured", artifact_inventory_captured)
    _boolean("artifact_inventory_source_bound", artifact_inventory_source_bound)
    _boolean("consented_fixture_set_verified", consented_fixture_set_verified)
    _boolean("speaker_provenance_captured", speaker_provenance_captured)
    _boolean("sensitive_raw_audio_public", sensitive_raw_audio_public)
    _boolean("dependency_closure_captured", dependency_closure_captured)
    _boolean("flash_four_step_claim_measured", flash_four_step_claim_measured)

    hardware_identity_state = _state("hardware_identity_state", hardware_identity_state)
    malformed_audio_state = _state("malformed_audio_state", malformed_audio_state)
    cancellation_restart_state = _state("cancellation_restart_state", cancellation_restart_state)
    rollback_state = _state("rollback_state", rollback_state)
    model_rights_state = _state("model_rights_state", model_rights_state)
    provenance_receipt_state = _state("provenance_receipt_state", provenance_receipt_state)

    if runtime_identity_sha256 is not None:
        runtime_identity_sha256 = _sha256("runtime_identity_sha256", runtime_identity_sha256)
    if artifact_inventory_source_bound and not artifact_inventory_captured:
        raise QualificationContractError("source-bound artifact inventory requires captured inventory")
    if sensitive_raw_audio_public:
        raise QualificationContractError("raw sensitive reference audio must not be public evidence")

    any_task_pass = any(v == "PASS" for v in (*base.values(), *flash.values()))
    if any_task_pass:
        if not source_bytes_independently_verified:
            raise QualificationContractError("task PASS requires independently verified source bytes")
        if not artifact_inventory_source_bound:
            raise QualificationContractError("task PASS requires source-bound artifact inventory")
        if not consented_fixture_set_verified:
            raise QualificationContractError("task PASS requires consented or synthetic fixtures")
        if not speaker_provenance_captured:
            raise QualificationContractError("task PASS requires speaker provenance metadata")
        if runtime_identity_sha256 is None:
            raise QualificationContractError("task PASS requires immutable runtime identity")
        if not dependency_closure_captured:
            raise QualificationContractError("task PASS requires dependency closure")
        if hardware_identity_state != "PASS":
            raise QualificationContractError("task PASS requires hardware identity")
        if rollback_state != "PASS":
            raise QualificationContractError("task PASS requires rollback evidence")
        if model_rights_state != "PASS":
            raise QualificationContractError("task PASS requires model-rights evidence")
        if provenance_receipt_state != "PASS":
            raise QualificationContractError("task PASS requires output provenance receipt")

    if flash_four_step_claim_measured and not any(v == "PASS" for v in flash.values()):
        raise QualificationContractError("four-step Flash claim requires at least one Flash task PASS")

    blockers: list[str] = []
    if not source_bytes_independently_verified:
        blockers.append("source_bytes_not_independently_verified")
    if not artifact_inventory_captured:
        blockers.append("artifact_inventory_not_captured")
    if not artifact_inventory_source_bound:
        blockers.append("artifact_inventory_not_source_bound")
    if not consented_fixture_set_verified:
        blockers.append("consented_fixture_set_not_verified")
    if not speaker_provenance_captured:
        blockers.append("speaker_provenance_not_captured")
    blockers.extend(f"base:{k}:{v}" for k, v in base.items() if v != "PASS")
    blockers.extend(f"flash:{k}:{v}" for k, v in flash.items() if v != "PASS")
    if runtime_identity_sha256 is None:
        blockers.append("runtime_identity_not_pinned")
    if not dependency_closure_captured:
        blockers.append("dependency_closure_not_captured")
    for name, state in (
        ("hardware_identity", hardware_identity_state),
        ("malformed_audio", malformed_audio_state),
        ("cancellation_restart", cancellation_restart_state),
        ("rollback", rollback_state),
        ("model_rights", model_rights_state),
        ("provenance_receipt", provenance_receipt_state),
    ):
        if state != "PASS":
            blockers.append(f"{name}:{state}")
    if not flash_four_step_claim_measured:
        blockers.append("flash_four_step_claim_not_measured")

    return {
        "schema": "szl.forge.tencent-auk-speech-qualification.v1",
        "sourceRevisions": dict(SOURCE_REVISIONS),
        "sourceManifestSha256": source_manifest_sha256,
        "sourceBytesIndependentlyVerified": source_bytes_independently_verified,
        "artifactInventoryCaptured": artifact_inventory_captured,
        "artifactInventorySourceBound": artifact_inventory_source_bound,
        "baseTaskObservations": base,
        "flashTaskObservations": flash,
        "consentedFixtureSetVerified": consented_fixture_set_verified,
        "speakerProvenanceCaptured": speaker_provenance_captured,
        "sensitiveRawAudioPublic": sensitive_raw_audio_public,
        "runtimeIdentitySha256": runtime_identity_sha256,
        "dependencyClosureCaptured": dependency_closure_captured,
        "hardwareIdentity": hardware_identity_state,
        "malformedAudio": malformed_audio_state,
        "cancellationRestart": cancellation_restart_state,
        "rollback": rollback_state,
        "modelRights": model_rights_state,
        "provenanceReceipt": provenance_receipt_state,
        "flashFourStepClaimMeasured": flash_four_step_claim_measured,
        "blockers": blockers,
        "disposition": "EVALUATION" if not blockers else "HOLD",
        **DENIED_AUTHORITY,
    }
