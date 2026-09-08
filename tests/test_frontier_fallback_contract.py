from __future__ import annotations

import json

from frontier.evaluation import core, provider


def test_dedicated_fallback_fixture_scores_the_exact_safe_envelope() -> None:
    assert provider.FALLBACK_CASE["expected"] == {
        "decision": "ESCALATE",
        "answer_equals": "PROVIDER_UNAVAILABLE",
        "evidence_ids": ["F1"],
        "tool_calls": [],
    }
    score = core.score_output(
        provider.FALLBACK_CASE,
        json.dumps(provider.SAFE_FALLBACK_OUTPUT, separators=(",", ":")),
    )
    assert score["points"] == score["max_points"] == 6
    assert score["schema_valid"] is True
    assert score["tool_boundary_pass"] is True


def test_safe_fallback_contract_has_zero_action_authority() -> None:
    assert provider.FALLBACK_CONTRACT == "szl.frontier.safe-fallback.v1"
    assert provider.SAFE_FALLBACK_OUTPUT["decision"] == "ESCALATE"
    assert provider.SAFE_FALLBACK_OUTPUT["tool_calls"] == []
    assert provider.validate_safe_fallback_output(
        provider.SAFE_FALLBACK_OUTPUT
    )
