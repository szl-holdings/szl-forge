import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "frontier" / "tau_042_evaluation.json"
EXACT_REVISION = "55df51608b8b2d172c4bbac2cd11e8345e307476"


def load_contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_tau_release_is_exact_and_held():
    contract = load_contract()
    assert contract["candidate"] == "huggingface/tau"
    assert contract["release"] == "v0.4.2"
    assert contract["upstream_revision"] == EXACT_REVISION
    assert len(contract["upstream_revision"]) == 40
    assert contract["license"] == "MIT"
    assert contract["disposition"] == "HOLD"
    assert contract["production_eligible"] is False
    assert contract["evaluation_only"] is True


def test_tau_evaluation_grants_no_effectors():
    authority = load_contract()["authority"]
    expected = {
        "autonomous_merge",
        "deploy",
        "secret_access",
        "branch_protection_mutation",
        "upstream_write",
        "production_route_change",
        "live_catalog_authority",
    }
    assert set(authority) == expected
    assert all(value is False for value in authority.values())


def test_tau_release_specific_evidence_is_required():
    evidence = set(load_contract()["required_evidence"])
    assert {
        "pre-0.4.2-session-replay",
        "session-label-and-custom-message-persistence",
        "stream-ordering-and-incomplete-sse-negative-path",
        "live-codex-catalog-nonauthority-test",
        "zai-thinking-serialization",
        "shell-stdin-timeout-path-command-isolation",
        "baseline-comparison",
        "rollback-disable",
    }.issubset(evidence)
