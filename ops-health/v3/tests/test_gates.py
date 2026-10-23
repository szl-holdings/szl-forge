"""Gates v3 (prereg s.11.2), broken baselines (s.11.1), metric recomputation (s.6.7) and the
tolerance rule (s.11.3) on HAND-MADE numbers. SYNTHETIC; DEV data only.

The gate plumbing runs on the DEV fit and on DEV gate rows. Every tolerance here is hand-made
(PLUMBING_TOLERANCES); none is computed on DEV data, and no detection ratio or catch rate is
computed (s.15.5).
"""

from __future__ import annotations

import copy
import json
import math
import unittest
from fractions import Fraction
from pathlib import Path

from v2.research import metrics as v2_metrics
from v3.research import bootstrap, freeze, gates, guards, mutation, representation, tolerances
from v3.tests import _support as S

K = S.kernel()
PLUMBING_TOLERANCES = {"theta": [1e-6] * 9, "k": 2, "ba": 0.005}  # hand-made, not a v3 tolerance


class GateFormulaTest(unittest.TestCase):
    def test_gate_names_and_types(self):
        self.assertEqual(gates.GATE_NAMES, ("g_integrity", "g_failclosed", "g_truth_sign", "g_monotone",
                                            "g_validation", "g_calibration", "g_baseline_valid",
                                            "g_coef_band", "g_threshold_band", "g_ba_rel"))
        self.assertEqual(set(gates.PARAMETER_SPACE_GATES), {"g_coef_band", "g_threshold_band", "g_truth_sign"})
        self.assertEqual(set(gates.OUTPUT_GATES), {"g_ba_rel", "g_validation", "g_calibration", "g_monotone"})

    def test_kept_v2_formulas(self):
        self.assertEqual(gates.g_validation(0.35), 0.5)
        self.assertEqual(gates.g_validation(0.9), 1.0)
        self.assertEqual(gates.g_validation(None), 0.0)
        self.assertEqual(gates.g_calibration(0.05), 1.0)
        self.assertAlmostEqual(gates.g_calibration(0.10), 0.5, places=15)
        self.assertEqual(gates.g_calibration(0.30), 0.0)
        self.assertEqual(gates.g_baseline_valid({"valid": True}), 1.0)
        self.assertEqual(gates.g_baseline_valid({"valid": "yes"}), 0.0)

    def test_coef_band(self):
        ref = [1.0] * 9
        tol = [0.1] * 9
        self.assertEqual(gates.g_coef_band([1.1] * 9, ref, tol), 1.0)  # boundary: |d| <= tol
        self.assertEqual(gates.g_coef_band([1.0] * 8 + [1.2], ref, tol), 0.0)
        self.assertEqual(gates.g_coef_band([1.0] * 8 + [math.nan], ref, tol), 0.0)
        self.assertEqual(gates.g_coef_band(None, ref, tol), 0.0)
        self.assertEqual(gates.g_coef_band([1.0] * 8, ref, tol), 0.0)

    def test_threshold_band_is_integer(self):
        self.assertEqual(gates.g_threshold_band(22, 20, 2), 1.0)
        self.assertEqual(gates.g_threshold_band(23, 20, 2), 0.0)
        self.assertEqual(gates.g_threshold_band(17, 20, 3), 1.0)
        for bad in (20.0, True, None, "20"):
            self.assertEqual(gates.g_threshold_band(bad, 20, 2), 0.0)

    def test_ba_rel(self):
        self.assertEqual(gates.g_ba_rel(0.795, 0.80, 0.005), 1.0)
        self.assertEqual(gates.g_ba_rel(0.7949, 0.80, 0.005), 0.0)
        self.assertEqual(gates.g_ba_rel(None, 0.80, 0.005), 0.0)

    def test_vector_and_catch_rule(self):
        full = {name: 1.0 for name in gates.GATE_NAMES}
        self.assertEqual(gates.gate_vector(full), full)
        with self.assertRaises(ValueError):
            gates.gate_vector({k: v for k, v in full.items() if k != "g_ba_rel"})
        with self.assertRaises(ValueError):
            gates.gate_vector(dict(full, g_integrity=1.5))
        ref = dict(full, g_validation=0.9)
        self.assertEqual(gates.caught_by(dict(full, g_validation=0.9), ref), [])
        self.assertEqual(gates.caught_by(dict(full, g_validation=0.89, g_coef_band=0.0), ref),
                         ["g_validation", "g_coef_band"])


class RepresentationTest(unittest.TestCase):
    def test_truth_on_the_r_scale(self):
        truth = representation.truth_r()
        self.assertEqual(truth["weights"]["seconds_since_last_success"], 2.2 * 900.0 / 86400.0)
        self.assertAlmostEqual(truth["weights"]["seconds_since_last_success"], 0.0229167, places=7)
        self.assertEqual(truth["intercept"], 9.8)
        for name in representation.FEATURE_NAMES:
            if name != "seconds_since_last_success":
                self.assertEqual(truth["weights"][name], representation.TRUE_WEIGHTS_V1[name])

    def test_mapping_is_exact_on_the_linear_predictor(self):
        weights = {n: 0.1 * (i + 1) for i, n in enumerate(representation.FEATURE_NAMES)}
        table = dict(representation.TABLE_V1_MINMAX, tls_enabled=(0.5, 2.0))
        mapped = representation.to_r_scale(1.5, weights, table)
        x = {n: 0.3 + 0.01 * i for i, n in enumerate(representation.FEATURE_NAMES)}
        z_table = 1.5 + sum(weights[n] * (x[n] - table[n][0]) / table[n][1] for n in representation.FEATURE_NAMES)
        r = representation.TABLE_R
        z_r = mapped["intercept"] + sum(mapped["weights"][n] * (x[n] - r[n][0]) / r[n][1]
                                        for n in representation.FEATURE_NAMES)
        self.assertAlmostEqual(z_table, z_r, places=12)

    def test_table_mutant_moves_theta_on_the_r_scale(self):
        art = S.candidate()
        mutant = [m for m in mutation.artifact_mutants(art) if m["id"] == mutation.TABLE_MUTANT_ID][0]["artifact"]
        a = representation.artifact_theta_r(art)
        b = representation.artifact_theta_r(mutant)
        self.assertEqual(a[:6] + a[7:], b[:6] + b[7:])
        w = art["weights"]["seconds_since_last_success"]
        self.assertEqual(a[6], w * 900.0 / 900.0)
        self.assertEqual(b[6], w * 900.0 / 86400.0)
        self.assertNotEqual(a[6], b[6])

    def test_malformed_tables_are_refused(self):
        good = representation.table_block(representation.TABLE_R)
        for change in (lambda t: t.pop("tls_enabled"), lambda t: t.__setitem__("x", {"offset": 0, "scale": 1}),
                       lambda t: t["tls_enabled"].__setitem__("scale", 0.0),
                       lambda t: t["tls_enabled"].__setitem__("scale", float("inf")),
                       lambda t: t["tls_enabled"].__setitem__("offset", True)):
            table = json.loads(json.dumps(good))
            change(table)
            with self.assertRaises(representation.TableError):
                representation.parse_table(table)


class _GateFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = S.TempDir()
        cls.tmp = cls._tmp.__enter__()
        cls.info = S.freeze_dev(cls.tmp / "ref")
        cls.model = Path(cls.info["model_path"])
        cls.receipt_path = Path(cls.info["receipt_path"])
        cls.artifact = json.loads(cls.model.read_bytes())
        cls.receipt = json.loads(cls.receipt_path.read_text(encoding="utf-8"))
        cls.ctx = gates.GateContext(S.gate_rows(), "dev-gates (DEV rows 684..1023)")
        cls.ref_result = gates.evaluate_fixture(cls.model, cls.receipt_path, cls.ctx, reference=None, tolerances=None)
        cls.reference = gates.reference_of(cls.ref_result)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def pass2(self, name: str, artifact: dict):
        data = (json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        return S.write_consistent_pair(self.tmp / name, data, self.receipt)


class FixtureEvaluationTest(_GateFixtures):
    def test_reference_plumbing(self):
        g = self.ref_result["gates"]
        self.assertEqual(g["g_integrity"], 1.0)
        self.assertEqual(g["g_failclosed"], 1.0)
        self.assertEqual(self.ref_result["details"]["failclosed"]["total"], 76)
        for name in ("g_coef_band", "g_threshold_band", "g_ba_rel"):
            self.assertEqual(g[name], 1.0)
        for name in gates.GATE_NAMES:
            self.assertTrue(0.0 <= g[name] <= 1.0)
        self.assertEqual(self.ref_result["rows"], 340)

    def test_band_gates_compare_against_the_reference(self):
        same = gates.evaluate_fixture(self.model, self.receipt_path, self.ctx, reference=self.reference,
                                      tolerances=PLUMBING_TOLERANCES)
        self.assertEqual(gates.caught_by(same["gates"], self.ref_result["gates"]), [])
        art = copy.deepcopy(self.artifact)
        art["threshold_index"] = art["threshold_index"] + 3 if art["threshold_index"] <= 96 else art["threshold_index"] - 3
        model, rec = self.pass2("thr3", art)
        result = gates.evaluate_fixture(model, rec, self.ctx, reference=self.reference, tolerances=PLUMBING_TOLERANCES)
        self.assertEqual(result["gates"]["g_threshold_band"], 0.0)
        self.assertEqual(result["gates"]["g_integrity"], 1.0)
        self.assertIn("g_threshold_band", gates.caught_by(result["gates"], self.ref_result["gates"]))

    def test_refused_artifact_scores_zero_on_kernel_output_gates(self):
        mutant = [m for m in mutation.artifact_mutants(self.artifact) if m["id"] == mutation.TABLE_MUTANT_ID][0]
        model, rec = self.pass2("table", mutant["artifact"])
        result = gates.evaluate_fixture(model, rec, self.ctx, reference=self.reference, tolerances=PLUMBING_TOLERANCES)
        g = result["gates"]
        self.assertEqual(g["g_integrity"], 0.0)
        self.assertIn("differs from the registered representation R", result["details"]["integrity_refusal"])
        for name in gates.KERNEL_OUTPUT_GATES:
            self.assertEqual(g[name], 0.0, name)
        self.assertEqual(g["g_coef_band"], 0.0)  # the declared table alone moves theta on R
        self.assertEqual(g["g_threshold_band"], 1.0)
        self.assertEqual(g["g_truth_sign"], self.ref_result["gates"]["g_truth_sign"])

    def test_unreadable_bytes_fail_closed_everywhere(self):
        model, rec = S.write_consistent_pair(self.tmp / "junk", b"{not json", self.receipt)
        result = gates.evaluate_fixture(model, rec, self.ctx, reference=self.reference, tolerances=PLUMBING_TOLERANCES)
        self.assertEqual(set(result["gates"].values()), {0.0})

    def test_non_reference_needs_tolerances(self):
        with self.assertRaises(ValueError):
            gates.evaluate_fixture(self.model, self.receipt_path, self.ctx, reference=self.reference, tolerances=None)


class BrokenBaselineTest(_GateFixtures):
    """s.11.1: constant-ALERT, a fail-open v3 kernel and majority are each INVALID_BASELINE."""

    def test_constant_alert_is_invalid(self):
        art = copy.deepcopy(self.artifact)
        art["intercept"] = 100.0
        art["weights"] = {n: 0.0 for n in art["weights"]}
        model, rec = self.pass2("const_alert", art)
        result = gates.evaluate_fixture(model, rec, self.ctx, reference=self.reference, tolerances=PLUMBING_TOLERANCES)
        validity = result["details"]["baseline_validity"]
        self.assertEqual(validity["status"], v2_metrics.INVALID_BASELINE)
        self.assertIn("iii_decisions_not_constant", validity["failed_preconditions"])
        self.assertEqual(result["gates"]["g_baseline_valid"], 0.0)
        self.assertEqual(result["details"]["binary_report"]["per_class_rates"], v2_metrics.INVALID_BASELINE)
        self.assertNotIn("accuracy", result["details"]["binary_report"])

    def test_fail_open_kernel_is_invalid(self):
        module, path = S.load_kernel_copy(self.tmp, mutate=S.fail_open_mutation)
        rec = S.rebind_receipt(self.receipt_path, path, self.tmp / "fail_open_receipt.json")
        result = gates.evaluate_fixture(self.model, rec, self.ctx, reference=self.reference,
                                        tolerances=PLUMBING_TOLERANCES, kernel_module=module)
        self.assertEqual(result["gates"]["g_integrity"], 1.0)
        self.assertLess(result["gates"]["g_failclosed"], 1.0)
        validity = result["details"]["baseline_validity"]
        self.assertEqual(validity["status"], v2_metrics.INVALID_BASELINE)
        self.assertIn("iv_refused_all_probes", validity["failed_preconditions"])
        self.assertEqual(result["gates"]["g_baseline_valid"], 0.0)

    def test_majority_is_invalid(self):
        decisions = [0] * len(self.ctx.labels)
        validity = v2_metrics.baseline_validity(self.ctx.labels, decisions, receipt_verified=False,
                                                probes_refused=0, probes_total=gates.N_PROBES)
        self.assertEqual(validity["status"], v2_metrics.INVALID_BASELINE)
        self.assertTrue({"i_receipt_verifies", "iii_decisions_not_constant", "iv_refused_all_probes"}
                        <= set(validity["failed_preconditions"]))
        report = v2_metrics.binary_report(self.ctx.labels, decisions, validity)
        self.assertEqual(report["balanced_accuracy"], 0.5)
        self.assertEqual(report["per_class_rates"], v2_metrics.INVALID_BASELINE)


class MetricRecomputationTest(_GateFixtures):
    def test_gate_metrics_equal_a_recomputation_from_the_rows(self):
        kernel = K.OpsHealthKernel(self.model, self.receipt_path)
        scores, decisions = [], []
        for row in self.ctx.rows:
            out = kernel.advise({"features": row["features"]})
            scores.append(out["operator_attention_score"])
            decisions.append(1 if out["threshold_decision"] else 0)
        d = self.ref_result["details"]
        c = v2_metrics.confusion(self.ctx.labels, decisions)
        ba = (Fraction(c["tp"], c["tp"] + c["fn"]) + Fraction(c["tn"], c["tn"] + c["fp"])) / 2
        self.assertEqual(d["balanced_accuracy_exact"], [ba.numerator, ba.denominator])
        self.assertEqual(d["balanced_accuracy"], float(ba))
        self.assertAlmostEqual(d["ece"], v2_metrics.ece(self.ctx.labels, scores), places=11)
        self.assertAlmostEqual(d["brier"], v2_metrics.brier(self.ctx.labels, scores), places=11)
        ones = [1] * len(scores)
        self.assertEqual(bootstrap.BinaryStats(self.ctx.labels, decisions).balanced_accuracy(ones),
                         v2_metrics.balanced_accuracy(self.ctx.labels, decisions))
        stats = bootstrap.ScoreStats(self.ctx.labels, scores)
        self.assertAlmostEqual(stats.ece(ones), v2_metrics.ece(self.ctx.labels, scores), places=12)
        self.assertAlmostEqual(stats.brier(ones), v2_metrics.brier(self.ctx.labels, scores), places=12)
        self.assertEqual(gates.g_validation(d["balanced_accuracy"]), self.ref_result["gates"]["g_validation"])
        self.assertEqual(gates.g_calibration(d["ece"]), self.ref_result["gates"]["g_calibration"])


class ToleranceRuleTest(unittest.TestCase):
    """s.11.3 on hand-made numbers only (no tolerance is computed on DEV data)."""

    @staticmethod
    def band(theta_of, k_of, ba_of):
        return [{"train_split": name, "theta_r": [float(theta_of(i, j)) for j in range(9)],
                 "threshold_index": k_of(i), "ba_gates": float(ba_of(i)), "converged": True}
                for i, name in enumerate(tolerances.BAND_NAMES)]

    def test_constant_band_gives_the_floors(self):
        vec = tolerances.compute(self.band(lambda i, j: 0.5, lambda i: 20, lambda i: 0.8))
        self.assertEqual(vec["theta"], [0.0] * 9)
        self.assertEqual(vec["k"], 2)
        self.assertEqual(vec["ba"], 0.005)
        self.assertEqual(vec["c"], 4.0)

    def test_known_sd(self):
        # alternating +-a around a mean: SD = a * sqrt(n / (n - 1)) with n = 100
        a = 0.01
        vec = tolerances.compute(self.band(lambda i, j: (a if i % 2 else -a) * (j + 1),
                                           lambda i: 20 + (1 if i % 2 else -1), lambda i: 0.8 + (0.01 if i % 2 else -0.01)))
        sd = a * math.sqrt(100 / 99)
        for j in range(9):
            self.assertAlmostEqual(vec["theta"][j], 4 * math.sqrt(2) * sd * (j + 1), places=12)
        self.assertEqual(vec["k"], max(2, math.ceil(4 * math.sqrt(2) * math.sqrt(100 / 99))))
        self.assertAlmostEqual(vec["ba"], 4 * math.sqrt(2) * 0.01 * math.sqrt(100 / 99), places=12)
        self.assertEqual(len(tolerances.digest(vec)), 64)

    def test_refusals(self):
        good = self.band(lambda i, j: 0.5, lambda i: 20, lambda i: 0.8)
        with self.assertRaises(tolerances.ToleranceRefused):
            tolerances.compute(good[:99])
        with self.assertRaises(tolerances.ToleranceRefused):
            tolerances.compute(list(reversed(good)))
        bad = copy.deepcopy(good)
        bad[5]["theta_r"][0] = math.nan
        with self.assertRaises(tolerances.ToleranceRefused):
            tolerances.compute(bad)
        bad = copy.deepcopy(good)
        bad[5]["threshold_index"] = 20.0
        with self.assertRaises(tolerances.ToleranceRefused):
            tolerances.compute(bad)
        dev = copy.deepcopy(good)
        dev[0]["train_split"] = "dev"
        with self.assertRaises(guards.DevRuleError):
            tolerances.compute(dev)


if __name__ == "__main__":
    unittest.main()
