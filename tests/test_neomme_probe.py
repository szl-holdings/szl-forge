# SPDX-License-Identifier: Apache-2.0
"""Offline harness tests; these do not claim to exercise model inference."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "neomme_probe.py"
SPEC = importlib.util.spec_from_file_location("neomme_probe", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class NeoMMEProbeTests(unittest.TestCase):
    def test_plan_is_inert_and_non_promotional(self):
        value = probe.plan()
        self.assertEqual(value["status"], "NOT_EXECUTED")
        for field in ("production_promotion", "training_performed", "routing_changed", "private_brain_loaded", "trust_remote_code"):
            self.assertIs(value[field], False)
        self.assertEqual(value["runtime_qualification"], "UNQUALIFIED")
        self.assertEqual(value["receipt_status"], "UNSIGNED_HONEST")
        self.assertNotIn("cases", value)

    def test_source_pins_are_exact(self):
        self.assertRegex(probe.REVISION, r"^[a-f0-9]{40}$")
        self.assertRegex(probe.TRANSFORMERS_REVISION, r"^[a-f0-9]{40}$")
        self.assertRegex(probe.WEIGHT_SHA256, r"^[a-f0-9]{64}$")
        self.assertLess(probe.WEIGHT_BYTES, probe.plan()["download_budget_bytes"])
        self.assertFalse(any(name.endswith((".py", ".pkl", ".bin")) for name in probe.BLOB_PINS))

    def test_fixture_ids_and_ground_truth_are_complete(self):
        ids = [doc[0] for doc in probe.DOCUMENTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(expected in ids for _, expected in probe.QUERIES))
        self.assertEqual(probe.plan()["documents"], 6)
        self.assertEqual(probe.plan()["queries"], 8)

    def test_ranking_ties_are_deterministic(self):
        self.assertEqual(probe.rank([1.0, 1.0, -2.0], ["z", "a", "b"]), ["a", "z", "b"])

    def test_rank_rejects_invalid_shapes(self):
        for scores, names in (([], []), ([1.0], ["a", "b"]), ([1.0, 2.0], ["a", "a"])):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                probe.rank(scores, names)

    def test_rank_rejects_non_finite_and_bool(self):
        for invalid in (math.nan, math.inf, -math.inf, True, "1"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                probe.rank([invalid], ["a"])

    def test_perfect_metrics(self):
        self.assertEqual(probe.metrics([["a", "b"]], ["a"]),
                         {"top1": 1.0, "recall_at_3": 1.0, "mrr": 1.0, "ndcg_at_3": 1.0})

    def test_metric_denominators_include_misses(self):
        value = probe.metrics([["a", "b", "c", "d"], ["a", "b", "c", "d"]], ["a", "d"])
        self.assertEqual(value["top1"], 0.5)
        self.assertEqual(value["recall_at_3"], 0.5)
        self.assertEqual(value["mrr"], 0.625)
        self.assertEqual(value["ndcg_at_3"], 0.5)

    def test_metrics_reject_missing_or_duplicate_documents(self):
        for rankings, expected in (([], []), ([["a"]], []), ([["a"]], ["b"]), ([["a", "a"]], ["a"])):
            with self.subTest(rankings=rankings), self.assertRaises(ValueError):
                probe.metrics(rankings, expected)

    def test_canonical_digest_is_order_independent(self):
        self.assertEqual(probe.digest({"a": 1, "b": 2}), probe.digest({"b": 2, "a": 1}))
        self.assertNotEqual(probe.digest({"a": 1}), probe.digest({"a": 2}))
        with self.assertRaises(ValueError):
            probe.digest({"metric": math.nan})

    def test_failure_replaces_old_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            probe.write_report(path, {"status": "SMOKE_EXECUTED"})
            probe.write_report(path, {"status": "FAILED"})
            self.assertEqual(json.loads(path.read_text()), {"status": "FAILED"})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_cli_defaults_to_plan_without_site_packages(self):
        run = subprocess.run([sys.executable, "-S", str(SOURCE)], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "NOT_EXECUTED")


if __name__ == "__main__":
    unittest.main()
