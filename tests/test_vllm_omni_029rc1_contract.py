from __future__ import annotations

import pytest

from frontier.vllm_omni_029rc1_contract import (
    BASE_RUNTIME,
    DENIED_AUTHORITY,
    FRONTIER_ADMISSION_MERGE,
    QualificationContractError,
    VLLM_OMNI_REPOSITORY,
    VLLM_OMNI_REVISION,
    VLLM_OMNI_TAG,
    evaluate_observation,
)

GOOD_DIGEST = "a" * 64


def evidence(**overrides):
    values = {
        "observed_source_revision": VLLM_OMNI_REVISION,
        "artifact_sha256": None,
        "artifact_bytes_independently_verified": False,
        "artifact_source_bound": False,
        "dependency_closure_captured": False,
        "hardware_identity_state": "NOT_RUN",
        "duplex_runtime_state": "NOT_RUN",
        "server_vad_state": "NOT_RUN",
        "nemotron_talker_state": "NOT_RUN",
        "diffusion_batching_state": "NOT_RUN",
        "diffusion_parallelism_state": "NOT_RUN",
        "component_offload_state": "NOT_RUN",
        "parallel_stage_init_state": "NOT_RUN",
        "mooncake_end_to_end_state": "NOT_RUN",
        "legacy_batching_migration_state": "NOT_RUN",
        "rollback_state": "NOT_RUN",
        "model_rights_state": "NOT_RUN",
        "turn_based_vad_interrupt_disabled_verified": False,
        "mooncake_completion_witness": False,
        "legacy_config_rejected_or_migrated": False,
        "max_num_seqs_semantics_preserved": False,
    }
    values.update(overrides)
    return values


def complete_evidence(**overrides):
    values = {
        "artifact_sha256": GOOD_DIGEST,
        "artifact_bytes_independently_verified": True,
        "artifact_source_bound": True,
        "dependency_closure_captured": True,
        "hardware_identity_state": "PASS",
        "duplex_runtime_state": "PASS",
        "server_vad_state": "PASS",
        "nemotron_talker_state": "PASS",
        "diffusion_batching_state": "PASS",
        "diffusion_parallelism_state": "PASS",
        "component_offload_state": "PASS",
        "parallel_stage_init_state": "PASS",
        "mooncake_end_to_end_state": "PASS",
        "legacy_batching_migration_state": "PASS",
        "rollback_state": "PASS",
        "model_rights_state": "PASS",
        "turn_based_vad_interrupt_disabled_verified": True,
        "mooncake_completion_witness": True,
        "legacy_config_rejected_or_migrated": True,
        "max_num_seqs_semantics_preserved": True,
    }
    values.update(overrides)
    return evidence(**values)


def test_exact_identity_and_authority_are_fixed():
    assert VLLM_OMNI_REPOSITORY == "vllm-project/vllm-omni"
    assert VLLM_OMNI_TAG == "v0.29.0rc1"
    assert VLLM_OMNI_REVISION == "aff7d64948f6c0e81e9b3272234623044e215967"
    assert FRONTIER_ADMISSION_MERGE == "fa16050c930a2f5a258619101a98bfc2013cb783"
    assert BASE_RUNTIME == "vLLM 0.29.0"
    assert all(value is False for value in DENIED_AUTHORITY.values())


def test_unexecuted_observation_stays_hold():
    result = evaluate_observation(**evidence())
    assert result["disposition"] == "HOLD"
    assert result["prerelease"] is True
    assert "artifact_bytes_not_independently_verified" in result["blockers"]
    assert "duplexRuntime:NOT_RUN" in result["blockers"]
    assert result["productionServingAuthorized"] is False


def test_wrong_source_revision_is_rejected():
    with pytest.raises(QualificationContractError, match="source revision"):
        evaluate_observation(**evidence(observed_source_revision="0" * 40))


def test_states_and_booleans_are_strict():
    with pytest.raises(QualificationContractError, match="duplex_runtime_state"):
        evaluate_observation(**evidence(duplex_runtime_state="pass"))
    with pytest.raises(QualificationContractError, match="explicit boolean"):
        evaluate_observation(
            **evidence(artifact_bytes_independently_verified="true")
        )


def test_independent_bytes_require_digest_and_source_binding_requires_bytes():
    with pytest.raises(QualificationContractError, match="artifact digest"):
        evaluate_observation(
            **evidence(artifact_bytes_independently_verified=True)
        )
    with pytest.raises(QualificationContractError, match="source-bound"):
        evaluate_observation(**evidence(artifact_source_bound=True))


def test_runtime_pass_requires_full_execution_prerequisites():
    with pytest.raises(QualificationContractError, match="verified artifact bytes"):
        evaluate_observation(**evidence(duplex_runtime_state="PASS"))

    with pytest.raises(QualificationContractError, match="hardware identity"):
        evaluate_observation(
            **evidence(
                artifact_sha256=GOOD_DIGEST,
                artifact_bytes_independently_verified=True,
                artifact_source_bound=True,
                dependency_closure_captured=True,
                duplex_runtime_state="PASS",
                rollback_state="PASS",
                model_rights_state="PASS",
            )
        )


def test_vad_pass_requires_turn_based_interrupt_constraint():
    with pytest.raises(QualificationContractError, match="interrupt constraint"):
        evaluate_observation(
            **complete_evidence(
                turn_based_vad_interrupt_disabled_verified=False
            )
        )


def test_mooncake_pass_requires_actual_completion_witness():
    with pytest.raises(QualificationContractError, match="completion witness"):
        evaluate_observation(
            **complete_evidence(mooncake_completion_witness=False)
        )


def test_migration_pass_requires_explicit_legacy_and_new_semantics_proof():
    with pytest.raises(QualificationContractError, match="legacy rejection"):
        evaluate_observation(
            **complete_evidence(legacy_config_rejected_or_migrated=False)
        )

    with pytest.raises(QualificationContractError, match="max_num_seqs"):
        evaluate_observation(
            **complete_evidence(max_num_seqs_semantics_preserved=False)
        )


def test_model_rights_cannot_be_unavailable_for_runtime_pass():
    with pytest.raises(QualificationContractError, match="model-rights"):
        evaluate_observation(
            **complete_evidence(model_rights_state="UNAVAILABLE")
        )


def test_complete_evidence_is_only_evaluation():
    result = evaluate_observation(**complete_evidence())
    assert result["disposition"] == "EVALUATION"
    assert result["blockers"] == []
    assert result["artifactSha256"] == GOOD_DIGEST
    assert result["observations"]["mooncakeEndToEnd"] == "PASS"
    for key, expected in DENIED_AUTHORITY.items():
        assert result[key] is expected
