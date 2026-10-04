"""Descriptive selection invariants; no threshold fitting or runtime authority."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oac_operational_risk_coverage", ROOT / "tools/evaluate_operational_health_model.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class RiskCoverageTests(unittest.TestCase):
    labels = [0, 0, 1, 1]
    scores = [0.0, 0.625, 1.0, 0.25]

    def report(self, labels=None, scores=None, threshold=0.25):
        return evaluation.calculate_risk_coverage(
            self.labels if labels is None else labels,
            self.scores if scores is None else scores,
            threshold,
        )

    def test_known_answer_uses_accepted_not_full_corpus_denominator(self):
        report = self.report()
        full = report["curve"][0]
        half = next(point for point in report["curve"] if point["cutoff"] == 0.5)
        self.assertEqual(1, full["accepted_errors"])
        self.assertEqual(0.25, full["accepted_error_rate"]["value"])
        self.assertEqual(3, half["accepted_rows"])
        self.assertEqual(1, half["accepted_errors"])
        self.assertEqual(3, half["accepted_error_rate"]["denominator"])
        self.assertAlmostEqual(1 / 3, half["accepted_error_rate"]["value"])
        self.assertEqual({"true_positive": 1, "true_negative": 1,
                          "false_positive": 1, "false_negative": 0}, half["accepted_confusion"])

    def test_cutoff_one_retains_endpoints_but_control_abstains_all(self):
        report = self.report()
        self.assertEqual(2, report["curve"][-1]["accepted_rows"])
        self.assertEqual(1.0, report["curve"][-1]["cutoff"])
        control = report["abstain_all_control"]
        self.assertIsNone(control["cutoff"])
        self.assertEqual(0, control["accepted_rows"])
        self.assertEqual(4, control["abstained_rows"])
        self.assertEqual(0.0, control["coverage"]["value"])
        self.assertEqual(1.0, control["abstention"]["value"])
        self.assertIsNone(control["accepted_error_rate"]["value"])
        self.assertIsNone(control["accepted_error_rate"]["wilson_95"])
        self.assertEqual(0, control["accepted_error_rate"]["denominator"])

    def test_natural_zero_coverage_is_unknown_risk_not_perfect(self):
        report = self.report([0, 1], [0.25, 0.25])
        for point in report["curve"][1:]:
            self.assertEqual(0, point["accepted_rows"])
            self.assertIsNone(point["accepted_error_rate"]["value"])
            self.assertIsNone(point["accepted_error_rate"]["wilson_95"])
        json.dumps(report, allow_nan=False)

    def test_boundary_and_equal_margin_ties_are_inclusive(self):
        # Both sides have normalized distance 0.5 from the fixed 0.25 decision boundary.
        report = self.report([0, 1, 1], [0.125, 0.625, 0.25])
        half = next(point for point in report["curve"] if point["cutoff"] == 0.5)
        self.assertEqual(2, half["accepted_rows"])
        self.assertEqual(0, half["accepted_errors"])
        self.assertEqual(3, report["curve"][0]["accepted_rows"])
        self.assertEqual(2, report["curve"][0]["accepted_predicted_attention_rows"])

    def test_real_decimal_boundary_retains_symmetric_ties(self):
        report = self.report([0, 1], [0.08, 0.58], 0.16)
        half = next(point for point in report["curve"] if point["cutoff"] == 0.5)
        self.assertEqual(2, half["accepted_rows"])
        self.assertEqual(1.0, half["positive_coverage"]["value"])
        self.assertEqual(1.0, half["negative_coverage"]["value"])

    def test_decimal_tie_neighbors_are_not_admitted_by_a_tolerance(self):
        scores = [math.nextafter(0.08, 0.0), 0.08, math.nextafter(0.08, 1.0),
                  math.nextafter(0.58, 0.0), 0.58, math.nextafter(0.58, 1.0)]
        report = self.report([0, 0, 0, 1, 1, 1], scores, 0.16)
        half = next(point for point in report["curve"] if point["cutoff"] == 0.5)
        self.assertEqual(4, half["accepted_rows"])
        self.assertEqual({"positive_rows": 2, "negative_rows": 2}, half["accepted_class_support"])

    def test_order_invariance_and_label_blind_selection(self):
        expected = self.report()
        order = [2, 0, 3, 1]
        self.assertEqual(expected, self.report([self.labels[i] for i in order],
                                             [self.scores[i] for i in order]))
        changed = self.report([1 - label for label in self.labels])
        for first, second in zip(expected["curve"], changed["curve"], strict=True):
            for field in ("accepted_rows", "abstained_rows", "coverage", "abstention",
                          "accepted_predicted_attention_rows", "accepted_predicted_clear_rows"):
                self.assertEqual(first[field], second[field])
        self.assertNotEqual(expected["curve"][0]["accepted_errors"],
                            changed["curve"][0]["accepted_errors"])

    def test_selective_class_support_and_rejected_errors_remain_visible(self):
        point = next(point for point in self.report()["curve"] if point["cutoff"] == 0.5)
        self.assertEqual({"positive_rows": 1, "negative_rows": 2}, point["accepted_class_support"])
        self.assertEqual(0.5, point["positive_coverage"]["value"])
        self.assertEqual(1.0, point["negative_coverage"]["value"])
        self.assertEqual(1, sum(point["abstained_confusion"].values()))
        self.assertEqual(0, point["abstained_errors"])

    def test_risk_is_not_forced_to_improve_as_coverage_falls(self):
        curve = self.report()["curve"]
        self.assertGreater(curve[1]["accepted_error_rate"]["value"],
                           curve[0]["accepted_error_rate"]["value"])
        self.assertEqual(sorted([point["accepted_rows"] for point in curve], reverse=True),
                         [point["accepted_rows"] for point in curve])

    def test_zero_observed_errors_retains_nonzero_uncertainty(self):
        point = self.report()["curve"][-1]
        self.assertEqual(0.0, point["accepted_error_rate"]["value"])
        self.assertGreater(point["accepted_error_rate"]["wilson_95"]["upper"], 0.0)

    def test_one_class_support_is_null_not_false_certainty(self):
        report = self.report([0, 0], [0.0, 1.0])
        for point in report["curve"]:
            self.assertIsNone(point["positive_coverage"]["value"])
            self.assertEqual(0, point["positive_coverage"]["denominator"])

    def test_descriptive_scope_never_becomes_confidence_or_authorization(self):
        report = self.report()
        self.assertEqual("SAMPLE", report["evidence_class"])
        self.assertFalse(report["cutoff_selected"])
        self.assertFalse(report["runtime_selection_applied"])
        self.assertFalse(report["production_promotion_allowed"])
        self.assertIn("not_confidence", report["selection_semantics"])
        self.assertIn("not_trust", report["selection_semantics"])
        self.assertEqual(list(evaluation.RISK_COVERAGE_CUTOFFS),
                         [point["cutoff"] for point in report["curve"]])
        json.dumps(report, allow_nan=False)

    def test_invalid_inputs_fail_closed_with_fixed_code(self):
        cases = [([], []), ([0], []), ([2], [0.1]), ([True], [0.1]),
                 ([0.0], [0.1]), (["0"], [0.1]), ([0], [True]), ([0], ["0.1"]),
                 ([0], [-0.1]), ([0], [1.1]), ([0], [10**1000]),
                 ([0], [float("nan")]), ([0], [float("inf")])]
        for labels, scores in cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaisesRegex(evaluation.EvaluationError, "INVALID_RISK_COVERAGE_INPUTS"):
                    self.report(labels, scores)
        for threshold in (0, 1, True, "0.25", -1, 2, 10**1000, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(evaluation.EvaluationError, "INVALID_RISK_COVERAGE_INPUTS"):
                    self.report(threshold=threshold)


class RiskCoverageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluation.evaluate()

    def test_fixed_test_curve_retains_all_original_error_counts(self):
        full = self.report["risk_coverage"]["curve"][0]
        self.assertEqual(240, full["accepted_rows"])
        self.assertEqual(68, full["accepted_errors"])
        self.assertEqual(self.report["metrics"]["confusion"], full["accepted_confusion"])
        self.assertEqual(0.16, self.report["risk_coverage"]["decision_threshold"])
        self.assertFalse(self.report["threshold_tuned"])
        self.assertFalse(self.report["weights_updated"])
        self.assertEqual(evaluation.FIXED_MODEL_SHA256,
                         self.report["source_files_sha256"][evaluation.MODEL_FILE])

    def test_new_analysis_is_inside_existing_hashed_source(self):
        source = ROOT / "tools/evaluate_operational_health_model.py"
        self.assertEqual(evaluation.digest(source.read_bytes()),
                         self.report["source_files_sha256"]["tools/evaluate_operational_health_model.py"])

    def test_fresh_lanes_keep_queue_failure_and_add_descriptive_curves(self):
        kernel = evaluation.OperationalHealthKernel(ROOT / evaluation.MODEL_FILE,
                                                    ROOT / evaluation.MODEL_RECEIPT_FILE)
        report = evaluation.fresh_seed_lanes((ROOT / evaluation.TRAINER_FILE).read_bytes(),
                                             kernel, 0.16, rows=100, replicates=2)
        self.assertEqual(5, len(report["lanes"]))
        for lane in report["lanes"].values():
            full = lane["risk_coverage"]["curve"][0]
            self.assertEqual(100, full["accepted_rows"])
            self.assertEqual(lane["confusion"], full["accepted_confusion"])
            self.assertEqual(lane["confusion"]["false_positive"] + lane["confusion"]["false_negative"],
                             full["accepted_errors"])
            self.assertIsNone(lane["risk_coverage"]["abstain_all_control"]["accepted_error_rate"]["value"])
        queue = report["lanes"]["covariate_queue_saturation"]
        self.assertEqual(0.5, queue["balanced_accuracy"]["value"])
        self.assertEqual(100, queue["risk_coverage"]["curve"][0]["accepted_predicted_attention_rows"])

    def test_summary_does_not_fit_or_rescore(self):
        with mock.patch.object(evaluation.OperationalHealthKernel, "score",
                               side_effect=AssertionError("must_not_rescore")):
            self.assertEqual(self.report["risk_coverage"], evaluation.calculate_risk_coverage(
                [int(row["label"]) for row in self.report["observations"]],
                [row["score"] for row in self.report["observations"]], 0.16))


if __name__ == "__main__":
    unittest.main()
