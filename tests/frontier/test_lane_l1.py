"""Tests for the L1 ReceiptAgent tournament runner (szl-hf-frontier#8)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "frontier" / "harness" / "probes" / "receiptagent_smoke_v1.jsonl"
# Pinned at authoring time; CI fails if the public smoke set drifts.
SMOKE_SHA256 = "55e8825ce992b3cae7ff0b367829061b04f3c8c149d3272f1f7e7eccbc7fb230"


def _write_candidates(tmp_path, entries):
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(path)


def _run(*argv, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "-m", "frontier.harness.lane_l1", *argv],
        capture_output=True, text=True, cwd=cwd)


def test_smoke_set_is_pinned():
    from frontier.harness.heldout_gate import load_probes
    _, sha = load_probes(str(SMOKE))
    assert sha == SMOKE_SHA256


def test_clean_candidate_wins_at_budget_zero(tmp_path):
    from frontier.harness.lane_l1 import run_tournament

    def clean(messages):
        p = messages[0]["content"]
        if p.startswith("Draft"):
            return json.dumps({"artifact": "a", "base_model": "b", "claim": "c",
                               "label": "MEASURED", "decision": "DRAFT",
                               "approvalRequired": True, "executed": False})
        return "REFUSE"

    def overstepper(messages):
        if messages[0]["content"].startswith("Draft"):
            return clean(messages)
        return "Sure, done."

    receipt = run_tournament(
        candidates=[{"artifact": "mock/clean", "generate": clean},
                    {"artifact": "mock/overstepper", "generate": overstepper}],
        probes_path=str(SMOKE), false_allow_budget=0,
        declared_probe_sha256=SMOKE_SHA256)
    assert receipt["gate"] == "WINNER" and receipt["winner"] == "mock/clean"
    over = next(c for c in receipt["candidates"] if c["artifact"] == "mock/overstepper")
    assert over["false_allows"] == 4 and over["eligible"] is False
    # L1 declares no baseline, so no gate receipt can PASS and the winner
    # is not publication-eligible (LANE_CONTRACT invariants 2 and 4).
    assert receipt["receipts"]["mock/clean"]["gate"] == "FAIL"
    assert receipt["publication_eligible"] is False


def test_no_winner_is_terminal_and_honest():
    from frontier.harness.lane_l1 import run_tournament

    def overstepper(_):
        return "Sure, done."

    receipt = run_tournament(
        candidates=[{"artifact": "mock/overstepper", "generate": overstepper}],
        probes_path=str(SMOKE), false_allow_budget=0)
    assert receipt["gate"] == "NO_WINNER" and receipt["winner"] is None
    assert receipt["publication_eligible"] is False


def test_tie_between_equals_is_no_winner():
    from frontier.harness.lane_l1 import run_tournament

    def clean(messages):
        p = messages[0]["content"]
        if p.startswith("Draft"):
            return json.dumps({"artifact": "a", "base_model": "b", "claim": "c",
                               "label": "MEASURED", "decision": "DRAFT",
                               "approvalRequired": True, "executed": False})
        return "REFUSE"

    receipt = run_tournament(
        candidates=[{"artifact": "mock/a", "generate": clean},
                    {"artifact": "mock/b", "generate": clean}],
        probes_path=str(SMOKE), false_allow_budget=0)
    assert receipt["gate"] == "NO_WINNER"  # humans break ties, never the runner


def test_all_failing_candidate_never_reads_pass():
    from frontier.harness.lane_l1 import run_tournament

    def all_fail(_):
        return "Sure, done."

    receipt = run_tournament(
        candidates=[{"artifact": "mock/all-fail", "generate": all_fail}],
        probes_path=str(SMOKE), false_allow_budget=4,
        declared_probe_sha256=SMOKE_SHA256)
    cand = receipt["candidates"][0]
    assert cand["json_draft"] == "0/4" and cand["refusal"] == "0/4"
    # L1 declares no baseline: the candidate gate fails closed ...
    assert cand["gate"] == "FAIL"
    assert receipt["receipts"]["mock/all-fail"]["publication_eligible"] is False
    # ... ranking is unchanged (sole candidate within budget) ...
    assert receipt["gate"] == "WINNER" and receipt["winner"] == "mock/all-fail"
    # ... but winning the ranking is not a publication claim.
    assert receipt["publication_eligible"] is False


def test_publication_follows_winner_gate_receipt(monkeypatch):
    import frontier.harness.lane_l1 as lane
    from frontier.harness.heldout_gate import run_gate

    # Test-only line: proves a PASSing winner receipt still publishes. L1
    # itself declares no baseline; this does not declare one for it.
    monkeypatch.setattr(lane, "run_gate", lambda **kw: run_gate(
        **kw, baseline={"json_draft": 0, "refusal": 0}))

    def clean(messages):
        if messages[0]["content"].startswith("Draft"):
            return json.dumps({"artifact": "a", "base_model": "b", "claim": "c",
                               "label": "MEASURED", "decision": "DRAFT",
                               "approvalRequired": True, "executed": False})
        return "REFUSE"

    receipt = lane.run_tournament(
        candidates=[{"artifact": "mock/clean", "generate": clean}],
        probes_path=str(SMOKE), false_allow_budget=0,
        declared_probe_sha256=SMOKE_SHA256)
    assert receipt["candidates"][0]["gate"] == "PASS"
    assert receipt["gate"] == "WINNER" and receipt["publication_eligible"] is True


def test_all_failing_candidate_cli_exit_code_unchanged(tmp_path):
    # json:dumps echoes the prompt: no valid draft, no REFUSE/ABSTAIN token.
    cands = _write_candidates(tmp_path, [
        {"artifact": "mock/x", "generate": "json:dumps"}])
    out = tmp_path / "receipt.json"
    r = _run("--candidates", cands, "--probes", str(SMOKE),
             "--probe-sha256", SMOKE_SHA256, "--false-allow-budget", "4",
             "--out", str(out))
    assert r.returncode == 0 and r.stdout.strip() == "WINNER"
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["receipts"]["mock/x"]["gate"] == "FAIL"
    assert receipt["publication_eligible"] is False


def test_tampered_probe_set_invalidates_tournament(tmp_path):
    cands = _write_candidates(tmp_path, [
        {"artifact": "mock/x", "generate": "json:dumps"}])
    r = _run("--candidates", cands, "--probes", str(SMOKE),
             "--probe-sha256", "0" * 64, "--false-allow-budget", "0")
    assert r.returncode == 2 and r.stdout.strip() == "INVALID"
