from __future__ import annotations

import json

import pytest

from inference.hf_frontier import (
    CANDIDATES,
    Candidate,
    Disposition,
    FrontierError,
    build_receipt,
    gate,
    normalize_metadata,
    plan,
    sha256,
)


def metadata(repo_id: str, *, license_id: str = "apache-2.0", tags=None, gated=False):
    return {
        "id": repo_id,
        "modelId": repo_id,
        "sha": "a" * 40,
        "tags": list(tags or []) + [f"license:{license_id}"],
        "gated": gated,
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "lastModified": "2026-09-09T00:00:00.000Z",
        "cardData": {"license": license_id},
    }


def test_registry_never_pre_authorizes_production():
    assert CANDIDATES
    assert all(candidate.production_eligible is False for candidate in CANDIDATES)
    assert plan()["automaticPromotionAuthorized"] is False


def test_bodhan_other_license_and_gating_hold_evaluation():
    candidate = CANDIDATES[0]
    normalized = normalize_metadata(
        candidate,
        metadata(candidate.repo_id, license_id="other", tags=["custom_code"], gated="manual"),
    )
    decision = gate(normalized)
    assert decision["effectiveDisposition"] == "HOLD"
    assert decision["deploymentAuthorized"] is False
    assert set(decision["reasonCodes"]) == {
        "gated_repository",
        "license_not_admitted",
        "remote_code_signal",
    }


def test_amx_remains_watch_even_with_admitted_license():
    candidate = CANDIDATES[2]
    normalized = normalize_metadata(candidate, metadata(candidate.repo_id))
    decision = gate(normalized)
    assert decision["effectiveDisposition"] == "WATCH"
    assert decision["deploymentAuthorized"] is False
    assert decision["automaticPromotionAuthorized"] is False


def test_receipt_is_self_hashing_without_mutating_source():
    candidate = CANDIDATES[2]
    receipt = build_receipt(candidate, metadata(candidate.repo_id), observed_at="2026-09-09T12:00:00+00:00")
    claimed = receipt.pop("receiptSha256")
    assert claimed == sha256(receipt)


def test_wrong_repo_identity_fails_closed():
    with pytest.raises(FrontierError):
        normalize_metadata(CANDIDATES[0], metadata("other/model"))


def test_missing_revision_fails_closed():
    raw = metadata(CANDIDATES[0].repo_id)
    raw.pop("sha")
    with pytest.raises(FrontierError):
        normalize_metadata(CANDIDATES[0], raw)


def test_candidate_cannot_be_constructed_as_production_eligible():
    with pytest.raises(FrontierError):
        Candidate(
            repo_id="example/model",
            category="test",
            disposition=Disposition.EVALUATE,
            rationale="test",
            production_eligible=True,
        )


def test_plan_is_json_serializable_and_metadata_only():
    encoded = json.dumps(plan(), sort_keys=True)
    assert "metadata-only" in encoded
    assert '"deploymentAuthorized": false' in encoded
