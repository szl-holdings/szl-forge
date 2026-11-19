"""Runs the OAC receipt-bound verifier selftest and the LTT reproduction in CI (unittest discover), so their results are gated, not snapshots.
Stdlib only; each tool runs in a subprocess from a scratch cwd because both write outputs to the cwd. Synthetic data only."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _run(script, *args):
    with tempfile.TemporaryDirectory() as cwd:
        p = subprocess.run([sys.executable, "-B", str(TOOLS / script), *args], capture_output=True, text=True, cwd=cwd, timeout=600)
        out = (Path(cwd) / "oac_ltt.json").read_text() if (Path(cwd) / "oac_ltt.json").exists() else None
    return p, out


class OacRbcgSelftest(unittest.TestCase):
    def test_selftest_passes_every_case(self):
        p, _ = _run("oac_rbcg.py", "selftest")
        self.assertEqual(p.returncode, 0, p.stdout[-4000:] + p.stderr[-4000:])
        res = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(res["result"], "PASS")
        self.assertEqual([k for k, v in res["cases"].items() if not v], [])
        self.assertGreaterEqual(len(res["cases"]), 33)

    def test_ltt_reproduces_published_numbers(self):
        p, out = _run("oac_ltt.py")
        self.assertEqual(p.returncode, 0, p.stdout[-4000:] + p.stderr[-4000:])
        o = json.loads(out)
        self.assertTrue(o["gate"]["reproduced"])
        self.assertAlmostEqual(o["ltt_fnr"]["0.2"]["threshold"], 0.137721, places=6)
        self.assertEqual((o["ltt_fnr"]["0.2"]["val_misses"], o["ltt_fnr"]["0.2"]["val_positives"]), (3, 33))
        self.assertEqual((o["abstention_alpha0.10"]["test"]["answered"], o["abstention_alpha0.10"]["test"]["wrong"]), (135, 8))


if __name__ == "__main__":
    unittest.main()
