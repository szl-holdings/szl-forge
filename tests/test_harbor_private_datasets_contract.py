from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "harbor_private_datasets_contract",
    ROOT / "frontier" / "harbor_private_datasets_contract.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def evidence(**overrides: object) -> object:
    values: dict[str, object] = {
        "upstream_repository": MODULE.UPSTREAM_REPOSITORY,
        "upstream_revision": MODULE.UPSTREAM_REVISION,
        "dataset_repo_id": "SZLHOLDINGS/private-harbor-eval-fixture",
        "dataset_revision": "a" * 40,
        "dataset_content_sha256": "b" * 64,
        "synthetic_or_non_sensitive_fixture": True,
        "credential_host_restricted_to_huggingface": True,
        "credential_in_clone_url": False,
        "credential_in_argv": False,
        "credential_in_persisted_git_config": False,
        "credential_in_output_or_logs": False,
        "control_token_visible_to_catalog_discovery": False,
        "control_token_visible_to_trial_environment": False,
        "inference_or_github_credential_reused": False,
        "missing_token_fails_before_inference": True,
        "denied_access_fails_before_inference": True,
        "missing_git_lfs_fails_before_source_resolution": True,
        "lfs_materialization_failure_fails_before_inference": True,
        "revision_drift_fails_closed": True,
        "read_only_access_proved": True,
        "mutation_attempt_denied": True,
        "revocation_denies_new_retrieval": True,
        "cached_checkout_cleanup_proved": True,
        "jobs_scan_bounded": True,
        "partial_history_explicit": True,
    }
    values.update(overrides)
    return MODULE.PrivateDatasetEvidence(**values)


def test_exact_source_is_fixed() -> None:
    assert MODULE.UPSTREAM_REVISION == "8685727908e8d9ee5ccde48586864a851e24ff79"
    assert MODULE.PRIOR_UPSTREAM_REVISION == "271303d7a21fa595a5a71e46ce5524b0aff6bb9f"
    assert MODULE.FRONTIER_ISSUE == 83
    assert MODULE.FORGE_ISSUE == 227


def test_bounded_pass_stays_evaluation_not_production() -> None:
    result = MODULE.evaluate(evidence())
    assert result["boundedContractPassed"] is True
    assert result["disposition"] == "EVALUATION"
    assert result["productionAuthorized"] is False
    assert result["automaticPromotionAuthorized"] is False


def test_fixture_must_be_non_sensitive() -> None:
    result = MODULE.evaluate(evidence(synthetic_or_non_sensitive_fixture=False))
    assert result["disposition"] == "HOLD"
    assert "fixture_not_bounded_non_sensitive" in result["blockers"]


def test_any_credential_leak_or_reuse_fails_closed() -> None:
    for field in (
        "credential_host_restricted_to_huggingface",
        "credential_in_clone_url",
        "credential_in_argv",
        "credential_in_persisted_git_config",
        "credential_in_output_or_logs",
        "control_token_visible_to_catalog_discovery",
        "control_token_visible_to_trial_environment",
        "inference_or_github_credential_reused",
    ):
        bad_value = False if field == "credential_host_restricted_to_huggingface" else True
        result = MODULE.evaluate(evidence(**{field: bad_value}))
        assert result["disposition"] == "HOLD", field
        assert "credential_isolation_or_non_disclosure_failed" in result["blockers"]


def test_source_prerequisites_fail_closed_before_inference() -> None:
    for field in (
        "missing_token_fails_before_inference",
        "denied_access_fails_before_inference",
        "missing_git_lfs_fails_before_source_resolution",
        "lfs_materialization_failure_fails_before_inference",
        "revision_drift_fails_closed",
    ):
        result = MODULE.evaluate(evidence(**{field: False}))
        assert result["disposition"] == "HOLD", field
        assert "source_prerequisite_failure_not_fail_closed" in result["blockers"]


def test_read_only_boundary_requires_denied_mutation() -> None:
    assert MODULE.evaluate(evidence(read_only_access_proved=False))["disposition"] == "HOLD"
    result = MODULE.evaluate(evidence(mutation_attempt_denied=False))
    assert result["disposition"] == "HOLD"
    assert "read_only_boundary_not_proved" in result["blockers"]


def test_revocation_and_cache_cleanup_are_required() -> None:
    assert MODULE.evaluate(evidence(revocation_denies_new_retrieval=False))["disposition"] == "HOLD"
    result = MODULE.evaluate(evidence(cached_checkout_cleanup_proved=False))
    assert result["disposition"] == "HOLD"
    assert "revocation_or_cleanup_not_proved" in result["blockers"]


def test_moving_jobs_history_never_implies_completeness() -> None:
    assert MODULE.evaluate(evidence(jobs_scan_bounded=False))["disposition"] == "HOLD"
    result = MODULE.evaluate(evidence(partial_history_explicit=False))
    assert result["disposition"] == "HOLD"
    assert "moving_jobs_observation_not_bounded" in result["blockers"]


def test_malformed_identity_and_operational_authority_are_rejected() -> None:
    for field, value in (
        ("dataset_revision", "main"),
        ("dataset_content_sha256", "abc"),
        ("inference_performed", True),
        ("training_authorized", True),
        ("publication_authorized", True),
        ("deployment_authorized", True),
        ("production_corpus_authorized", True),
        ("automatic_promotion_authorized", True),
    ):
        try:
            evidence(**{field: value})
        except MODULE.HarborPrivateDatasetContractError:
            pass
        else:
            raise AssertionError(f"invalid evidence escaped contract: {field}")
