"""Tests for the L2 abstention bench runner (szl-hf-frontier#9)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "frontier" / "harness" / "probes" / "khipu_abstain_smoke_v1.jsonl"
# Pinned at authoring time; CI fails if the public smoke set drifts.
SMOKE_SHA256 = "138c14e64c76f7392c69adc89528b8add5dea9b0451cbf48ae85f9b85201f8dc"


def _run(*argv):
    return subprocess.run(
        [sys.executable, "-m", "frontier.harness.lane_l2", *argv],
        capture_output=True, text=True, cwd=ROOT)


def test_smoke_set_is_pinned():
    from frontier.harness.heldout_gate import load_probes
    _, sha = load_probes(str(SMOKE))
    assert sha == SMOKE_SHA256


def test_mock_abstains_and_beats_disclosed_line(tmp_path):
    out = tmp_path / "receipt.json"
    r = _run("--artifact", "SZLHOLDINGS/khipu-r3", "--probes", str(SMOKE),
             "--mock", "--probe-sha256", SMOKE_SHA256, "--out", str(out))
    assert r.returncode == 0 and r.stdout.strip() == "PASS"
    receipt = json.loads(out.read_text())
    assert receipt["abstain"] == "6/6"
    assert receipt["baseline"] == {"abstain": 3}  # KHIPU-R2 disclosed line


def test_navigate_instead_of_abstain_fails():
    # Reproduces the khipu-r3 failure mode: NAVIGATE instead of ABSTAIN.
    import frontier.harness.lane_l2 as lane
    def navigator(messages):
        return "NAVIGATE to the nearest handle."
    from frontier.harness.heldout_gate import run_gate
    receipt = run_gate(artifact="mock/navigator", probes_path=str(SMOKE),
                       generate=navigator, baseline=lane.KHIPU_BASELINE)
    assert receipt["abstain"] == "0/6" and receipt["gate"] == "FAIL"


def test_tampered_handle_set_invalidates():
    r = _run("--artifact", "x", "--probes", str(SMOKE), "--mock",
             "--probe-sha256", "0" * 64)
    assert r.returncode == 2 and r.stdout.strip() == "INVALID"
