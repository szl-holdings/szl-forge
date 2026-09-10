from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "harbor_provider_refs_contract",
    ROOT / "frontier" / "harbor_provider_refs_contract.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def evidence(**overrides: object) -> object:
    values: dict[str, object] = {
        "upstream_repository": MODULE.UPSTREAM_REPOSITORY,
        "upstream_revision": MODULE.UPSTREAM_REVISION,
        "reference_name": "MOCK_PROVIDER_KEY",
        "contains_secret_value": False,
        "attempted_value_persistence_rejected": True,
        "review_invalidated_recipe_change": True,
        "review_invalidated_model_change": True,
        "review_invalidated_connection_change": True,
        "review_invalidated_registry_change": True,
        "review_invalidated_presence_change": True,
        "expected_revision_conflict_denied": True,
        "ambiguous_save_followed_by_readback": True,
        "blind_retry_used": False,
        "legacy_fallback_used": False,
        "dispatch_grant_rechecked": True,
        "restart_grant_rechecked": True,
        "independent_egress_policy_enforced": True,
        "controller_count": 1,
        "protocol": "native",
        "reviewed_base_url": False,
        "model_base_url_bound": False,
        "running_worker_remote_revocation_supported": False,
    }
    values.update(overrides)
    return MODULE.ProviderReferenceEvidence(**values)


def test_exact_source_refresh_is_fixed() -> None:
    assert MODULE.UPSTREAM_REVISION == "271303d7a21fa595a5a71e46ce5524b0aff6bb9f"
    assert MODULE.PRIOR_UPSTREAM_REVISION == "ba67b21d625227abf084eba7b605737e4a057575"
    assert MODULE.FRONTIER_ISSUE == 72
    assert MODULE.FORGE_ISSUE == 222


def test_bounded_contract_pass_remains_evaluation_and_records_revocation_gap() -> None:
    result = MODULE.evaluate(evidence())
    assert result["boundedContractPassed"] is True
    assert result["disposition"] == "EVALUATION"
    assert "already_running_worker_remote_revocation_unavailable" in result["blockers"]
    assert result["productionAuthorized"] is False
    assert result["automaticPromotionAuthorized"] is False


def test_secret_value_presence_fails_closed() -> None:
    result = MODULE.evaluate(evidence(contains_secret_value=True))
    assert result["boundedContractPassed"] is False
    assert result["disposition"] == "HOLD"
    assert "reference_only_contract_failed" in result["blockers"]


def test_review_change_matrix_must_be_complete() -> None:
    result = MODULE.evaluate(evidence(review_invalidated_presence_change=False))
    assert result["boundedContractPassed"] is False
    assert "review_invalidation_incomplete" in result["blockers"]


def test_revision_conflict_and_ambiguous_save_controls_fail_closed() -> None:
    assert MODULE.evaluate(evidence(expected_revision_conflict_denied=False))["disposition"] == "HOLD"
    assert MODULE.evaluate(evidence(ambiguous_save_followed_by_readback=False))["disposition"] == "HOLD"
    assert MODULE.evaluate(evidence(blind_retry_used=True))["disposition"] == "HOLD"


def test_registered_reference_never_uses_legacy_fallback_and_rechecks_grants() -> None:
    assert MODULE.evaluate(evidence(legacy_fallback_used=True))["disposition"] == "HOLD"
    assert MODULE.evaluate(evidence(dispatch_grant_rechecked=False))["disposition"] == "HOLD"
    assert MODULE.evaluate(evidence(restart_grant_rechecked=False))["disposition"] == "HOLD"


def test_host_metadata_is_not_accepted_without_independent_egress_policy() -> None:
    result = MODULE.evaluate(evidence(independent_egress_policy_enforced=False))
    assert result["disposition"] == "HOLD"
    assert "independent_egress_policy_missing" in result["blockers"]


def test_single_writer_invariant_is_explicit() -> None:
    result = MODULE.evaluate(evidence(controller_count=2))
    assert result["disposition"] == "HOLD"
    assert "single_writer_invariant_failed" in result["blockers"]


def test_native_key_only_binding_may_have_no_url() -> None:
    result = MODULE.evaluate(evidence(protocol="native", reviewed_base_url=False, model_base_url_bound=False))
    assert result["protocolBindingValid"] is True
    assert result["disposition"] == "EVALUATION"


def test_chat_protocols_require_reviewed_base_url_and_model_binding() -> None:
    for protocol in ("chat-completions", "responses"):
        bad = MODULE.evaluate(
            evidence(protocol=protocol, reviewed_base_url=False, model_base_url_bound=False)
        )
        assert bad["disposition"] == "HOLD"
        good = MODULE.evaluate(
            evidence(protocol=protocol, reviewed_base_url=True, model_base_url_bound=True)
        )
        assert good["disposition"] == "EVALUATION"


def test_operational_authority_is_rejected_at_construction() -> None:
    for field in (
        "provider_call_performed",
        "deployment_authorized",
        "worker_launch_authorized",
        "production_route_authorized",
        "automatic_promotion_authorized",
    ):
        try:
            evidence(**{field: True})
        except MODULE.HarborProviderContractError:
            pass
        else:
            raise AssertionError(f"operational authority escaped contract: {field}")
