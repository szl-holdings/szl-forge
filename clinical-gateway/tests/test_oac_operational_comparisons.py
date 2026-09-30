"""Reference comparisons: floors, generator ceiling, and paired bootstrap intervals."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "evaluate_operational_health_model.py"
SPEC = importlib.util.spec_from_file_location("oac_operational_evaluation_comparisons", TOOL)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class FixedSplitComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluation.evaluate()
        cls.comparisons = cls.report["comparisons"]

    def test_known_answer_confusions_on_fixed_split(self) -> None:
        predictors = self.comparisons["predictors"]
        self.assertEqual({"true_positive": 0, "false_positive": 0, "false_negative": 51, "true_negative": 189},
                         predictors["train_majority_constant"]["confusion"])
        self.assertEqual({"true_positive": 15, "false_positive": 49, "false_negative": 36, "true_negative": 140},
                         predictors["rule_consecutive_failures_gt_0"]["confusion"])
        self.assertEqual(self.report["metrics"]["confusion"], predictors["fixed_model"]["confusion"])
        self.assertEqual(self.report["metrics"]["roc_auc"],
                         predictors["fixed_model"]["metrics"]["roc_auc"]["value"])

    def test_accuracy_inversion_against_constant_is_reported_not_hidden(self) -> None:
        versus_constant = self.comparisons["paired_differences"]["fixed_model_minus_train_majority_constant"]
        self.assertLess(versus_constant["accuracy"]["value"], 0)
        self.assertFalse(versus_constant["accuracy"]["interval_95"]["excludes_zero"])
        self.assertTrue(versus_constant["balanced_accuracy"]["interval_95"]["excludes_zero"])
        self.assertIsNone(versus_constant["precision"]["value"])

    def test_generator_ceiling_is_threshold_free_and_bounds_the_model(self) -> None:
        ceiling = self.comparisons["generator_bayes_optimal_ceiling"]
        self.assertEqual(0.841477331673, ceiling["roc_auc"])
        self.assertGreaterEqual(ceiling["roc_auc"], ceiling["fixed_model_roc_auc"])
        self.assertAlmostEqual(ceiling["roc_auc"] - ceiling["fixed_model_roc_auc"], ceiling["roc_auc_headroom"],
                               places=11)
        for forbidden in ("decision_threshold", "confusion", "accuracy", "f1"):
            self.assertNotIn(forbidden, ceiling)

    def test_intervals_are_deterministic_and_fully_populated(self) -> None:
        self.assertEqual(evaluation.BOOTSTRAP_REPLICATES, self.comparisons["usable_replicates"])
        self.assertEqual(self.comparisons, evaluation.evaluate()["comparisons"])


class ComparisonFailClosedTests(unittest.TestCase):
    LABELS = [0, 1, 0, 1, 0, 0]
    SCORES = [0.1, 0.9, 0.2, 0.7, 0.3, 0.05]
    FAILURES = [0, 2, 0, 1, 0, 0]
    ORACLE = [0.1, 0.8, 0.2, 0.6, 0.3, 0.1]

    def test_single_class_or_misaligned_inputs_fail_closed(self) -> None:
        for labels, failures in (([0] * 6, self.FAILURES), ([1] * 6, self.FAILURES), (self.LABELS, [0] * 5)):
            with self.subTest(labels=labels, failures=len(failures)), \
                    self.assertRaisesRegex(evaluation.EvaluationError, "INVALID_COMPARISON_INPUTS"):
                evaluation.compare_against_references(labels, self.SCORES, 0.5, failures, self.ORACLE, False)

    def test_perfect_oracle_has_unit_auc_and_nonnegative_headroom(self) -> None:
        result = evaluation.compare_against_references(
            self.LABELS, self.SCORES, 0.5, self.FAILURES, [float(y) for y in self.LABELS], False)
        self.assertEqual(1.0, result["generator_bayes_optimal_ceiling"]["roc_auc"])
        self.assertGreaterEqual(result["generator_bayes_optimal_ceiling"]["roc_auc_headroom"], 0)


class GeneratorRuleMutationTests(unittest.TestCase):
    """A ceiling that ignores the generator would pass the known-answer test by constant.

    These mutants prove the ceiling is computed from the trainer's label rule bytes.
    """

    SOURCE = (ROOT / evaluation.TRAINER_FILE).read_bytes()
    FEATURES = {"listener_running": 1, "tls_enabled": 1, "peer_allowlist_configured": 1,
                "queue_utilization": 0.5, "consecutive_failures": 3, "seconds_since_last_success": 600.0,
                "ledger_integrity_ok": 0, "configuration_valid": 1}

    def test_mutated_generator_weight_changes_the_ceiling_rule(self) -> None:
        original = evaluation._generator_label_probability(self.SOURCE)(self.FEATURES)
        self.assertIn(b"4.4 * (1.0 - values", self.SOURCE)
        mutant = self.SOURCE.replace(b"4.4 * (1.0 - values", b"0.0 * (1.0 - values", 1)
        mutated = evaluation._generator_label_probability(mutant)(self.FEATURES)
        self.assertGreater(abs(original - mutated), 0.1)

    def test_generator_without_label_rule_fails_closed(self) -> None:
        mutant = self.SOURCE.replace(b"def _synthetic_label_probability", b"def _renamed_label_probability", 1)
        with self.assertRaisesRegex(evaluation.EvaluationError, "GENERATOR_LABEL_RULE_UNAVAILABLE"):
            evaluation._generator_label_probability(mutant)


if __name__ == "__main__":
    unittest.main()
