from __future__ import annotations

import pytest

from tools.hf_kernels_017_evidence import EvidenceError, REQUIRED_CASES, REVISION, plan, validate


def _payload():
    return {
        "schema": "szl.forge.hf-kernels-0.17-evidence.v1",
        "version": "0.17.0",
        "revision": REVISION,
        "artifact": {"sha256": "a" * 64},
        "environment": {
            "python": "3.12.0",
            "torch": "2.x",
            "backend": "cuda",
            "device": "cuda:0",
            "driver": "measured",
            "compiler": "measured",
            "helionUnavailableReason": "supported Helion toolchain not present on runner",
        },
        "checks": {name: "PASS" for name in REQUIRED_CASES} | {"helion": "UNAVAILABLE"},
    }


def test_plan_is_exact_source_and_hold():
    p = plan()
    assert p["revision"] == REVISION
    assert p["version"] == "0.17.0"
    assert p["disposition"] == "HOLD"
    assert p["authority"]["automaticPromotion"] is False


def test_valid_bounded_evidence_remains_hold():
    result = validate(_payload())
    assert result["disposition"] == "HOLD"
    assert result["allRequiredAvailableChecksPass"] is True


def test_wrong_revision_fails_closed():
    payload = _payload()
    payload["revision"] = "0" * 40
    with pytest.raises(EvidenceError, match="revision mismatch"):
        validate(payload)


def test_artifact_digest_is_required():
    payload = _payload()
    payload["artifact"]["sha256"] = "unknown"
    with pytest.raises(EvidenceError, match="artifact sha256"):
        validate(payload)


def test_negative_trust_case_is_mandatory():
    payload = _payload()
    payload["checks"]["trusted_repo_allowlist_negative"] = "FAIL"
    with pytest.raises(EvidenceError, match="negative trust test"):
        validate(payload)


def test_incompatible_capability_must_fail_before_execution():
    payload = _payload()
    payload["checks"]["min_version_incompatible_fails_before_execution"] = "FAIL"
    with pytest.raises(EvidenceError, match="fail before execution"):
        validate(payload)


def test_helion_unavailable_requires_reason():
    payload = _payload()
    payload["environment"].pop("helionUnavailableReason")
    with pytest.raises(EvidenceError, match="requires reason"):
        validate(payload)
