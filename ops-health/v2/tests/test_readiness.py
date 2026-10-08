"""Section 11 readiness aggregation (v2/research/mutation.py): aggregators, separation measures,
the Lambda decision rule, and a recomputation of the recorded run (if it exists).

SYNTHETIC.  Hand-made gate vectors only; the recorded run is re-read from
v2/results/mutation_readiness.json (no split is opened here).
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest

from v2.research import gates, metrics, mutation as M

ONES = {name: 1.0 for name in gates.GATE_NAMES}


def vec(**changes) -> dict:
    out = dict(ONES)
    out.update(changes)
    return out


class AggregatorTest(unittest.TestCase):
    def test_all_ones(self):
        for name in M.AGGREGATORS:
            with self.subTest(aggregator=name):
                self.assertEqual(M.aggregate(ONES, name), 1.0)
                self.assertTrue(M.is_ready(M.aggregate(ONES, name)))

    def test_one_partial_gate(self):
        v = vec(g_calibration=0.5)
        self.assertEqual(M.aggregate(v, "AND"), 0.0)
        self.assertEqual(M.aggregate(v, "min"), 0.5)
        self.assertAlmostEqual(M.aggregate(v, "mean"), 6.5 / 7, places=15)
        self.assertAlmostEqual(M.aggregate(v, "lambda"), 0.5 ** (1 / 7), places=14)

    def test_one_failing_gate_zeroes_lambda_and_min_only(self):
        v = vec(g_truth_sign=0.0)
        self.assertEqual(M.aggregate(v, "lambda"), 0.0)
        self.assertEqual(M.aggregate(v, "min"), 0.0)
        self.assertAlmostEqual(M.aggregate(v, "mean"), 6 / 7, places=15)

    def test_lambda_is_the_reference_kernel_with_uniform_weights(self):
        lam = M.lambda_module()
        self.assertEqual(M.LAMBDA_PARAMS, {"weights": None, "semantics": "kernel"})
        v = vec(g_truth_sign=0.875, g_monotone=0.99, g_validation=0.9)
        values = [v[g] for g in gates.GATE_NAMES]
        self.assertEqual(M.aggregate(v, "lambda"), lam.lambda_aggregate(values))
        self.assertAlmostEqual(M.aggregate(v, "lambda"),
                               math.exp(math.fsum(math.log(x) for x in values) / 7), places=14)
        # min <= Lambda <= mean on the fixed vectors below (lambda_reference.md section 7)
        for v in (vec(g_failclosed=0.3), vec(g_monotone=0.99, g_calibration=0.7),
                  {g: 0.1 * (i + 3) for i, g in enumerate(gates.GATE_NAMES)}):
            lo, mid, hi = (M.aggregate(v, n) for n in ("min", "lambda", "mean"))
            self.assertLessEqual(lo, mid + 1e-15)
            self.assertLessEqual(mid, hi + 1e-15)

    def test_cut(self):
        self.assertEqual(M.READINESS_CUT, 0.99)
        self.assertTrue(M.is_ready(0.99))
        self.assertFalse(M.is_ready(0.98999999))
        with self.assertRaises(ValueError):
            M.aggregate(ONES, "median")


class SeparationTest(unittest.TestCase):
    def test_hand_case(self):
        s = M.separation([1.0, 0.995], [0.5, 0.992, 0.2])
        self.assertEqual(s["broken_marked_ready"], 1)
        self.assertEqual(s["good_marked_not_ready"], 0)
        self.assertEqual(s["misclassifications"], 1)
        self.assertAlmostEqual(s["margin"], 0.003, places=15)
        self.assertEqual(s["auroc"], 1.0)

    def test_ties_count_half(self):
        s = M.separation([1.0, 1.0], [1.0, 0.0])
        self.assertEqual(s["auroc"], 0.75)
        self.assertEqual(s["margin"], 0.0)
        self.assertEqual(s["misclassifications"], 1)
        self.assertEqual(s["auroc"], metrics.roc_auc([1, 1, 0, 0], [1.0, 1.0, 1.0, 0.0]))

    def test_good_below_cut_is_a_misclassification(self):
        s = M.separation([0.5, 1.0], [0.0])
        self.assertEqual(s["good_marked_not_ready"], 1)
        self.assertEqual(s["misclassifications"], 1)

    def test_lambda_rule(self):
        def sep(margin, errors):
            return {"margin": margin, "misclassifications": errors}

        base = {"AND": sep(0.05, 2), "min": sep(0.05, 2), "mean": sep(0.02, 3)}
        self.assertEqual(M.lambda_verdict({**base, "lambda": sep(0.1, 2)})["verdict"], "KEEP")
        # equal margin to one baseline is not strictly better
        self.assertEqual(M.lambda_verdict({**base, "lambda": sep(0.05, 0)})["verdict"], "REJECTED")
        # better margin but more misclassifications than one baseline
        self.assertEqual(M.lambda_verdict({**base, "lambda": sep(0.2, 3)})["verdict"], "REJECTED")

    def test_readiness_block(self):
        good = {"g1": ONES, "g2": vec(g_calibration=0.995)}
        broken = {"b1": vec(g_truth_sign=0.875), "b2": ONES}
        r = M.readiness(good, broken)
        self.assertEqual(r["per_fixture"]["g2"]["ready"], {"AND": False, "min": True,
                                                            "mean": True, "lambda": True})
        self.assertEqual(r["separation"]["AND"]["misclassifications"], 2)  # g2 and b2
        self.assertEqual(r["separation"]["min"]["misclassifications"], 1)  # b2
        self.assertEqual(r["lambda"]["verdict"], "REJECTED")  # b2 = all ones: margin <= 0


class RecordedRunTest(unittest.TestCase):
    """Recompute everything derivable from v2/results/mutation_readiness.json."""

    @classmethod
    def setUpClass(cls):
        if not M.RESULTS_PATH.exists():
            raise unittest.SkipTest("no recorded mutation run")
        cls.r = json.loads(M.RESULTS_PATH.read_text(encoding="utf-8"))
        if cls.r.get("status") != "COMPLETE":
            raise unittest.SkipTest("recorded mutation run is not COMPLETE")

    def test_scope_and_sealing(self):
        r = self.r
        self.assertIs(r["synthetic"], True)
        self.assertIs(r["sealed"]["test_opened_marker_before"], False)
        self.assertIs(r["sealed"]["test_opened_marker_after"], False)
        self.assertEqual(r["validation_split"]["purpose"], "GATES")
        self.assertEqual(r["validation_split"]["rows"], 4096)
        self.assertEqual(sorted(r["known_good"]), ["k1", "k2", "k3", "k4", "k5", "v2"])
        self.assertEqual(len(r["artifact_mutants"]), 36)
        self.assertEqual(len(r["receipt_mutants"]), 5)
        self.assertEqual(len(r["kernel_mutants"]), 5)
        self.assertEqual(r["protocol"]["preregistration_sha256"],
                         "145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b")

    def test_catch_statuses_are_consistent(self):
        r = self.r
        ref = r["reference_v2_gates"]
        self.assertEqual(ref, r["known_good"]["v2"]["gates"])
        for m in r["artifact_mutants"]:
            with self.subTest(mutant=m["id"]):
                self.assertTrue(m["pass1_integrity"]["caught"])
                self.assertTrue(m["pass2_integrity_bypassed"]["kernel_accepts"])
                self.assertTrue(m["pass2_integrity_bypassed"]["verify_origin_accepts"])
                below = M.gates_below(m["pass2_behavior"]["gates"], ref)
                self.assertEqual(below, m["pass2_behavior"]["gates_below_reference"])
                self.assertEqual(m["status"], "CAUGHT" if below else "BLIND_SPOT")
        for m in r["receipt_mutants"]:
            layer = m["integrity"]
            caught = not layer["kernel"]["accepted"] or not layer["verify_origin"]["accepted"]
            self.assertEqual(m["status"], "CAUGHT" if caught else "BLIND_SPOT")
        self.assertTrue(r["kernel_control"]["test_harness_valid"])
        self.assertTrue(r["kernel_control"]["integrity"]["accepted"])
        for m in r["kernel_mutants"]:
            caught = m["caught_by_integrity"] or bool(m["caught_by_tests"])
            self.assertEqual(m["status"], "CAUGHT" if caught else "BLIND_SPOT")
        listed = {b["id"] for b in r["blind_spots"]}
        statuses = {m["id"] for group in ("artifact_mutants", "receipt_mutants", "kernel_mutants")
                    for m in r[group] if m["status"] == "BLIND_SPOT"}
        self.assertEqual(listed, statuses)
        self.assertEqual(r["counts"]["blind_spots"], len(listed))

    def test_aggregation_recomputes(self):
        r = self.r
        good = {fid: fx["gates"] for fid, fx in r["known_good"].items()}
        broken = {m["id"]: m["pass2_behavior"]["gates"] for m in r["artifact_mutants"]}
        again = M.readiness(good, broken)
        self.assertEqual(json.loads(json.dumps(again)), r["readiness"])

    def test_lambda_implementation_is_the_recorded_one(self):
        # math/lambda_stdlib.py was untracked at the run (committed later, byte-identical, in
        # d9c5fb3): the bytes on disk must be the ones the recorded run imported.
        self.assertEqual(hashlib.sha256(M.LAMBDA_PATH.read_bytes()).hexdigest(),
                         self.r["protocol"]["lambda_stdlib_sha256"])

    def test_recorded_test_layer_wording_is_the_disclosed_one(self):
        # The report quotes the recorded (incorrect) wording and the corrected one.
        for group in ("artifact_mutants", "receipt_mutants"):
            for m in self.r[group]:
                with self.subTest(mutant=m["id"]):
                    self.assertEqual(m["caught_by_tests"], M.RECORDED_TEST_LAYER_WORDING)
        self.assertNotIn("never reads", M.TEST_LAYER_NOT_RUN)

    def test_supplementary_inplace_run_is_consistent(self):
        sup = M.load_inplace_tests()
        if sup is None:
            self.skipTest("no supplementary in-place test run")
        self.assertIs(sup["synthetic"], True)
        self.assertIs(sup["part_of_registered_verdict"], False)
        self.assertTrue(sup["control_ok"])
        self.assertFalse(sup["tree_has_sealed_or_jsonl"])
        self.assertEqual(sup["recorded_results_sha256"],
                         hashlib.sha256(M.RESULTS_PATH.read_bytes()).hexdigest())
        self.assertEqual(sorted(v for v in sup["variants"] if v != "control"),
                         sorted(b["id"] for b in self.r["blind_spots"]))
        caught = [v for v, r in sup["variants"].items() if v != "control" and r["extra_vs_control"]]
        self.assertEqual(sorted(caught), sorted(sup["summary"]["in_place_suite_fails"]))

    def test_report_is_the_render_of_the_recorded_run(self):
        page = M.REPORT_PATH.read_text(encoding="utf-8")
        self.assertEqual(page, M.render(self.r))
        self.assertIn("**Post-run corrections (wording, render and test fixes only).**", page)

    def test_broken_baselines_report_no_per_class_rate(self):
        for name, b in self.r["broken_baselines"].items():
            with self.subTest(fixture=name):
                self.assertTrue(b["invalid_baseline"])
                self.assertTrue(b["no_per_class_rate_reported"])
                self.assertEqual(b["report"]["per_class_rates"], metrics.INVALID_BASELINE)
                self.assertEqual(b["report"]["baseline_validity"]["status"], metrics.INVALID_BASELINE)


if __name__ == "__main__":
    unittest.main()
