from __future__ import annotations

import pytest

from tools.trl_activation_offload_evidence import (
    EXACT_SOURCES,
    EvidenceError,
    REQUIRED_CASES,
    plan,
    validate,
)


def _payload():
    return {
        "schema": "szl.forge.trl-activation-offload-evidence.v1",
        "exactSources": list(EXACT_SOURCES),
        "environment": {
            "python": "3.12.0",
            "torch": "2.x",
            "device": "cuda:0",
            "acceleratorRuntime": "cuda",
            "driver": "measured",
            "fsdp2UnavailableReason": "runner does not expose distributed accelerator hardware",
        },
        "checks": {name: "PASS" for name in REQUIRED_CASES} | {"fsdp2": "UNAVAILABLE"},
        "metrics": {},
    }


def test_plan_is_exact_source_and_hold():
    p = plan()
    assert p["disposition"] == "HOLD"
    assert p["exactSources"] == list(EXACT_SOURCES)
    assert p["authority"]["automaticPromotion"] is False
    assert p["correctnessBeforePerformance"] is True


def test_valid_evidence_remains_hold():
    result = validate(_payload())
    assert result["disposition"] == "HOLD"
    assert result["allCorrectnessPass"] is True
    assert result["fsdp2"] == "UNAVAILABLE"


def test_wrong_source_fails_closed():
    payload = _payload()
    payload["exactSources"][1] = "0" * 40
    with pytest.raises(EvidenceError, match="exact source"):
        validate(payload)


def test_missing_correctness_case_fails_closed():
    payload = _payload()
    del payload["checks"]["nondefault_compute_stream"]
    with pytest.raises(EvidenceError, match="invalid or missing check"):
        validate(payload)


def test_fsdp2_unavailable_requires_reason():
    payload = _payload()
    payload["environment"].pop("fsdp2UnavailableReason")
    with pytest.raises(EvidenceError, match="requires reason"):
        validate(payload)


def test_failure_never_becomes_permission():
    payload = _payload()
    payload["checks"]["gradient_parity_no_offload_single_stream_streams"] = "FAIL"
    result = validate(payload)
    assert result["allCorrectnessPass"] is False
    assert result["disposition"] == "HOLD"
