from __future__ import annotations

import pytest

from frontier.tencent_auk_speech_contract import (
    DENIED_AUTHORITY,
    SOURCE_REVISIONS,
    TASKS,
    QualificationContractError,
    evaluate_observation,
)

GOOD = "a" * 64


def task_states(value="NOT_RUN"):
    return {name: value for name in TASKS}


def evidence(**overrides):
    values = {
        "observed_revisions": dict(SOURCE_REVISIONS),
        "source_manifest_sha256": GOOD,
        "source_bytes_independently_verified": False,
        "artifact_inventory_captured": False,
        "artifact_inventory_source_bound": False,
        "base_task_states": task_states(),
        "flash_task_states": task_states(),
        "consented_fixture_set_verified": False,
        "speaker_provenance_captured": False,
        "sensitive_raw_audio_public": False,
        "runtime_identity_sha256": None,
        "dependency_closure_captured": False,
        "hardware_identity_state": "NOT_RUN",
        "malformed_audio_state": "NOT_RUN",
        "cancellation_restart_state": "NOT_RUN",
        "rollback_state": "NOT_RUN",
        "model_rights_state": "NOT_RUN",
        "provenance_receipt_state": "NOT_RUN",
        "flash_four_step_claim_measured": False,
    }
    values.update(overrides)
    return values


def complete_evidence(**overrides):
    values = {
        "source_bytes_independently_verified": True,
        "artifact_inventory_captured": True,
        "artifact_inventory_source_bound": True,
        "base_task_states": task_states("PASS"),
        "flash_task_states": task_states("PASS"),
        "consented_fixture_set_verified": True,
        "speaker_provenance_captured": True,
        "runtime_identity_sha256": "b" * 64,
        "dependency_closure_captured": True,
        "hardware_identity_state": "PASS",
        "malformed_audio_state": "PASS",
        "cancellation_restart_state": "PASS",
        "rollback_state": "PASS",
        "model_rights_state": "PASS",
        "provenance_receipt_state": "PASS",
        "flash_four_step_claim_measured": True,
    }
    values.update(overrides)
    return evidence(**values)


def test_exact_source_revisions_and_denied_authority_are_fixed():
    assert SOURCE_REVISIONS == {
        "tencent/AuK": "43c9447eb0a8085d1694b2eb4963051dda8ed231",
        "tencent/AuK-Flash": "575b92f0895f75180bf2cbd35f2e176c5732b8ed",
    }
    assert all(value is False for value in DENIED_AUTHORITY.values())


def test_unexecuted_observation_stays_hold():
    result = evaluate_observation(**evidence())
    assert result["disposition"] == "HOLD"
    assert "source_bytes_not_independently_verified" in result["blockers"]
    assert "base:tts:NOT_RUN" in result["blockers"]
    assert result["publicVoiceCloningAuthorized"] is False


def test_source_revision_map_is_exact():
    wrong = dict(SOURCE_REVISIONS)
    wrong["tencent/AuK"] = "0" * 40
    with pytest.raises(QualificationContractError, match="admitted AuK snapshots"):
        evaluate_observation(**evidence(observed_revisions=wrong))


def test_task_maps_must_be_complete_and_strict():
    bad = task_states()
    bad.pop("source_separation")
    with pytest.raises(QualificationContractError, match="keys must exactly match"):
        evaluate_observation(**evidence(base_task_states=bad))
    bad = task_states()
    bad["tts"] = "pass"
    with pytest.raises(QualificationContractError, match="must be one of"):
        evaluate_observation(**evidence(base_task_states=bad))


def test_sensitive_raw_reference_audio_is_never_public_evidence():
    with pytest.raises(QualificationContractError, match="raw sensitive reference audio"):
        evaluate_observation(**evidence(sensitive_raw_audio_public=True))


def test_task_pass_requires_consent_provenance_and_runtime_closure():
    base = task_states()
    base["tts"] = "PASS"
    with pytest.raises(QualificationContractError, match="source bytes"):
        evaluate_observation(**evidence(base_task_states=base))

    with pytest.raises(QualificationContractError, match="consented or synthetic"):
        evaluate_observation(
            **evidence(
                base_task_states=base,
                source_bytes_independently_verified=True,
                artifact_inventory_captured=True,
                artifact_inventory_source_bound=True,
            )
        )


def test_source_bound_inventory_requires_capture():
    with pytest.raises(QualificationContractError, match="requires captured inventory"):
        evaluate_observation(**evidence(artifact_inventory_source_bound=True))


def test_flash_four_step_claim_requires_flash_task_pass():
    with pytest.raises(QualificationContractError, match="Flash task PASS"):
        evaluate_observation(
            **evidence(
                source_bytes_independently_verified=True,
                flash_four_step_claim_measured=True,
            )
        )


def test_complete_evidence_is_still_only_evaluation():
    result = evaluate_observation(**complete_evidence())
    assert result["disposition"] == "EVALUATION"
    assert result["blockers"] == []
    assert result["baseTaskObservations"]["source_separation"] == "PASS"
    assert result["flashTaskObservations"]["speech_enhancement"] == "PASS"
    for key, expected in DENIED_AUTHORITY.items():
        assert result[key] is expected
