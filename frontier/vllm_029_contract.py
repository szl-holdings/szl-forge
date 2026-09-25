"""Fail-closed qualification contract for vLLM 0.29.0.

This module is deliberately network-free and execution-free. It binds the exact
upstream source and selected release artifacts admitted by szl-frontier#121/#123,
then validates observations without granting training, serving, publication, or
promotion authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

VLLM_REPOSITORY = "vllm-project/vllm"
VLLM_REVISION = "98dff2a81d747d1dba01a47f939f48c3526d4206"
TRL_COMPAT_REVISION = "488a0d34be07c7cb2c130e8ba44cb9a4103717b2"
FRONTIER_SELECTION_MERGE = "e4661fe971fc222c7b91c42851d3bb43f89f68f9"
UPSTREAM_METADATA_EVIDENCE = "UPSTREAM_RELEASE_METADATA_NOT_INDEPENDENT_BYTE_VERIFICATION"
INDEPENDENT_BYTE_EVIDENCE = "INDEPENDENT_BYTE_VERIFIED"
OBSERVATION_STATES = frozenset({"NOT_RUN", "UNAVAILABLE", "PASS", "FAIL"})
DENIED_AUTHORITY = {
    "productionTrainingAuthorized": False,
    "productionServingAuthorized": False,
    "hubPublicationAuthorized": False,
    "weightRehostingAuthorized": False,
    "automaticPromotionAuthorized": False,
    "productionDefaultChangeAuthorized": False,
    "mergeDeployAuthority": False,
}


@dataclass(frozen=True)
class Artifact:
    asset_id: int
    size_bytes: int
    sha256: str
    accelerator: str
    platform: str
    abi: str


ARTIFACTS = {
    "cpu-linux-x86_64-glibc234": Artifact(
        asset_id=552379258,
        size_bytes=137_859_053,
        sha256="4d22dac8259e7e24dbc615531c73cbd1a6d4b4214a3329ccddb7854f26c9dac8",
        accelerator="cpu",
        platform="linux-x86_64-glibc2.34",
        abi="cp38-abi3",
    ),
    "cuda129-linux-x86_64-glibc228": Artifact(
        asset_id=552379252,
        size_bytes=548_493_389,
        sha256="22e8d8fec755986b3ad964004a1f8c65a55626ec948354bdbe95993e6b0289fe",
        accelerator="cuda12.9",
        platform="linux-x86_64-glibc2.28",
        abi="cp38-abi3",
    ),
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
        raise QualificationContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def evaluate_observation(
    *,
    artifact_key: str,
    observed_size_bytes: int | None,
    observed_sha256: str | None,
    independent_hash_verified: bool,
    dependency_closure_captured: bool,
    hardware_identity_state: str,
    model_runner_v2_state: str,
    model_runner_v1_fallback_state: str,
    correctness_state: str,
    failure_restart_state: str,
    matched_performance_state: str,
    rollback_target_bound: bool,
) -> dict[str, object]:
    """Validate a bounded vLLM 0.29 observation and return its disposition.

    `PASS` means only that the named bounded observation passed. Even a complete
    set of PASS observations yields EVALUATION, never production authority.
    """
    if artifact_key not in ARTIFACTS:
        raise QualificationContractError("artifact_key is not an admitted artifact")
    artifact = ARTIFACTS[artifact_key]

    _boolean("independent_hash_verified", independent_hash_verified)
    _boolean("dependency_closure_captured", dependency_closure_captured)
    _boolean("rollback_target_bound", rollback_target_bound)
    hardware_identity_state = _state("hardware_identity_state", hardware_identity_state)
    model_runner_v2_state = _state("model_runner_v2_state", model_runner_v2_state)
    model_runner_v1_fallback_state = _state(
        "model_runner_v1_fallback_state", model_runner_v1_fallback_state
    )
    correctness_state = _state("correctness_state", correctness_state)
    failure_restart_state = _state("failure_restart_state", failure_restart_state)
    matched_performance_state = _state("matched_performance_state", matched_performance_state)

    if observed_size_bytes is not None and type(observed_size_bytes) is not int:
        raise QualificationContractError("observed_size_bytes must be an integer or None")
    if observed_sha256 is not None:
        observed_sha256 = _sha256("observed_sha256", observed_sha256)

    if independent_hash_verified:
        if observed_size_bytes != artifact.size_bytes:
            raise QualificationContractError("independently verified artifact size does not match pin")
        if observed_sha256 != artifact.sha256:
            raise QualificationContractError("independently verified artifact digest does not match pin")
    elif observed_sha256 == artifact.sha256:
        # Matching an expected string without actually hashing acquired bytes is
        # still upstream metadata evidence, never independent verification.
        pass

    if model_runner_v2_state == "PASS":
        if not independent_hash_verified:
            raise QualificationContractError("Model Runner V2 PASS requires verified artifact bytes")
        if not dependency_closure_captured:
            raise QualificationContractError("Model Runner V2 PASS requires captured dependency closure")
        if hardware_identity_state != "PASS":
            raise QualificationContractError("Model Runner V2 PASS requires observed hardware identity")
        if not rollback_target_bound:
            raise QualificationContractError("Model Runner V2 PASS requires a bound rollback target")

    states = {
        "hardwareIdentity": hardware_identity_state,
        "modelRunnerV2": model_runner_v2_state,
        "modelRunnerV1Fallback": model_runner_v1_fallback_state,
        "correctness": correctness_state,
        "failureRestart": failure_restart_state,
        "matchedPerformance": matched_performance_state,
    }
    blockers: list[str] = []
    if not independent_hash_verified:
        blockers.append("artifact_bytes_not_independently_verified")
    if not dependency_closure_captured:
        blockers.append("dependency_closure_not_captured")
    if not rollback_target_bound:
        blockers.append("rollback_target_not_bound")
    blockers.extend(
        f"{name}:{state}"
        for name, state in states.items()
        if state != "PASS"
    )

    complete = not blockers
    return {
        "schema": "szl.forge.vllm-029-qualification.v1",
        "sourceRepository": VLLM_REPOSITORY,
        "sourceRevision": VLLM_REVISION,
        "trlCompatibilityRevision": TRL_COMPAT_REVISION,
        "frontierSelectionMerge": FRONTIER_SELECTION_MERGE,
        "artifactKey": artifact_key,
        "artifact": {
            "assetId": artifact.asset_id,
            "sizeBytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "accelerator": artifact.accelerator,
            "platform": artifact.platform,
            "abi": artifact.abi,
        },
        "evidenceClass": (
            INDEPENDENT_BYTE_EVIDENCE
            if independent_hash_verified
            else UPSTREAM_METADATA_EVIDENCE
        ),
        "dependencyClosureCaptured": dependency_closure_captured,
        "rollbackTargetBound": rollback_target_bound,
        "observations": states,
        "blockers": blockers,
        "disposition": "EVALUATION" if complete else "HOLD",
        **DENIED_AUTHORITY,
    }
