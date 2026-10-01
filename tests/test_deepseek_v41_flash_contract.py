from __future__ import annotations

import pytest

from frontier.deepseek_v41_flash_contract import (
    DENIED_AUTHORITY,
    FRONTIER_ADMISSION_MERGE,
    MODEL_TYPE,
    PROTOCOL_CASES,
    REQUIRED_SOURCE_FILES,
    RUNTIME_STATES,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    QualificationContractError,
    evaluate_observation,
)

GOOD = "a" * 64


def source_hashes():
    return {name: GOOD for name in REQUIRED_SOURCE_FILES}


def protocol_states(value="NOT_RUN"):
    return {name: value for name in PROTOCOL_CASES}


def runtime_states(value="NOT_RUN"):
    return {name: value for name in RUNTIME_STATES}


def evidence(**overrides):
    values = {
        "observed_source_revision": SOURCE_REVISION,
        "observed_model_type": MODEL_TYPE,
        "source_file_sha256": source_hashes(),
        "source_bytes_independently_verified": False,
        "checkpoint_inventory_captured": False,
        "checkpoint_inventory_source_bound": False,
        "protocol_states": protocol_states(),
        "generic_jinja_fallback_possible": False,
        "runtime_states": runtime_states(),
        "runtime_identity_sha256": None,
        "dependency_closure_captured": False,
        "hardware_identity_state": "NOT_RUN",
        "rollback_state": "NOT_RUN",
        "model_rights_state": "NOT_RUN",
        "measured_claims_source_bound": False,
    }
    values.update(overrides)
    return values


def complete_evidence(**overrides):
    values = {
        "source_bytes_independently_verified": True,
        "checkpoint_inventory_captured": True,
        "checkpoint_inventory_source_bound": True,
        "protocol_states": protocol_states("PASS"),
        "runtime_states": runtime_states("PASS"),
        "runtime_identity_sha256": "b" * 64,
        "dependency_closure_captured": True,
        "hardware_identity_state": "PASS",
        "rollback_state": "PASS",
        "model_rights_state": "PASS",
        "measured_claims_source_bound": True,
    }
    values.update(overrides)
    return evidence(**values)


def test_exact_identity_and_denied_authority_are_fixed():
    assert SOURCE_REPOSITORY == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert SOURCE_REVISION == "df42c109f1defefcbfcedbe7d905718a12266e40"
    assert FRONTIER_ADMISSION_MERGE == "5839c1b9d2fae5477d5c693a0d1684398c75224b"
    assert MODEL_TYPE == "deepseek_v41"
    assert all(value is False for value in DENIED_AUTHORITY.values())


def test_unexecuted_observation_stays_hold():
    result = evaluate_observation(**evidence())
    assert result["disposition"] == "HOLD"
    assert "source_bytes_not_independently_verified" in result["blockers"]
    assert "runtime:baseline_decode:NOT_RUN" in result["blockers"]
    assert result["productionServingAuthorized"] is False


def test_source_revision_and_model_type_fail_closed():
    with pytest.raises(QualificationContractError, match="source revision"):
        evaluate_observation(**evidence(observed_source_revision="0" * 40))
    with pytest.raises(QualificationContractError, match="model_type"):
        evaluate_observation(**evidence(observed_model_type="deepseek_v4"))


def test_source_digest_map_must_be_exact_and_valid():
    bad = source_hashes()
    bad.pop("LICENSE")
    with pytest.raises(QualificationContractError, match="keys must exactly match"):
        evaluate_observation(**evidence(source_file_sha256=bad))
    bad = source_hashes()
    bad["LICENSE"] = "ABC"
    with pytest.raises(QualificationContractError, match="SHA-256"):
        evaluate_observation(**evidence(source_file_sha256=bad))


def test_protocol_pass_requires_verified_source_bytes():
    states = protocol_states()
    states["text_multiturn"] = "PASS"
    with pytest.raises(QualificationContractError, match="protocol PASS"):
        evaluate_observation(**evidence(protocol_states=states))


def test_generic_jinja_fallback_is_never_admissible():
    with pytest.raises(QualificationContractError, match="Jinja"):
        evaluate_observation(**evidence(generic_jinja_fallback_possible=True))


def test_runtime_pass_requires_full_identity_closure():
    states = runtime_states()
    states["baseline_decode"] = "PASS"
    with pytest.raises(QualificationContractError, match="runtime identity"):
        evaluate_observation(
            **evidence(
                source_bytes_independently_verified=True,
                checkpoint_inventory_captured=True,
                checkpoint_inventory_source_bound=True,
                runtime_states=states,
            )
        )


def test_dspark_cannot_outrun_baseline_decode():
    states = runtime_states()
    states["dspark"] = "PASS"
    with pytest.raises(QualificationContractError, match="baseline decode"):
        evaluate_observation(
            **evidence(
                source_bytes_independently_verified=True,
                checkpoint_inventory_captured=True,
                checkpoint_inventory_source_bound=True,
                runtime_states=states,
                runtime_identity_sha256="b" * 64,
                dependency_closure_captured=True,
                hardware_identity_state="PASS",
                rollback_state="PASS",
                model_rights_state="PASS",
            )
        )


def test_context_ladder_requires_baseline_decode():
    states = runtime_states()
    states["bounded_context_ladder"] = "PASS"
    with pytest.raises(QualificationContractError, match="context-ladder"):
        evaluate_observation(
            **evidence(
                source_bytes_independently_verified=True,
                checkpoint_inventory_captured=True,
                checkpoint_inventory_source_bound=True,
                runtime_states=states,
                runtime_identity_sha256="b" * 64,
                dependency_closure_captured=True,
                hardware_identity_state="PASS",
                rollback_state="PASS",
                model_rights_state="PASS",
            )
        )


def test_measured_claims_require_real_runtime_pass():
    with pytest.raises(QualificationContractError, match="measured claims"):
        evaluate_observation(
            **evidence(
                source_bytes_independently_verified=True,
                measured_claims_source_bound=True,
            )
        )


def test_complete_evidence_is_still_only_evaluation():
    result = evaluate_observation(**complete_evidence())
    assert result["disposition"] == "EVALUATION"
    assert result["blockers"] == []
    assert result["runtimeObservations"]["dspark"] == "PASS"
    for key, expected in DENIED_AUTHORITY.items():
        assert result[key] is expected
