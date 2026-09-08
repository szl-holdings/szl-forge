"""Tests for the L3 lane runner CLI (szl-hf-frontier#10)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "frontier" / "harness" / "probes" / "chaski_smoke_v1.jsonl"
# Pinned at authoring time; CI fails if the public smoke set drifts.
SMOKE_SHA256 = "1b5afe991c73ebcd0a67c0fd1ffd09f10e5c132981dbf50b5a620fb21b850c0d"


def _run(*argv):
    return subprocess.run(
        [sys.executable, "-m", "frontier.harness.lane_l3", *argv],
        capture_output=True, text=True, cwd=ROOT)


def test_smoke_set_is_pinned():
    from frontier.harness.heldout_gate import load_probes
    _, sha = load_probes(str(SMOKE))
    assert sha == SMOKE_SHA256


def test_mock_run_passes_baseline_and_writes_receipt(tmp_path):
    out = tmp_path / "receipt.json"
    r = _run("--artifact", "SZLHOLDINGS/chaski-r2", "--probes", str(SMOKE),
             "--mock", "--probe-sha256", SMOKE_SHA256, "--out", str(out))
    assert r.returncode == 0 and r.stdout.strip() == "PASS"
    receipt = json.loads(out.read_text())
    assert receipt["json_draft"] == "5/5" and receipt["refusal"] == "6/6"
    assert receipt["baseline"] == {"json_draft": 0, "refusal": 2}
    assert receipt["publication_eligible"] is True  # mock only; real candidates need GPU


def test_wrong_declared_hash_invalidates(tmp_path):
    r = _run("--artifact", "x", "--probes", str(SMOKE), "--mock",
             "--probe-sha256", "0" * 64)
    assert r.returncode == 2 and r.stdout.strip() == "INVALID"


def test_requires_generate_or_mock():
    r = _run("--artifact", "x", "--probes", str(SMOKE))
    assert r.returncode != 0
