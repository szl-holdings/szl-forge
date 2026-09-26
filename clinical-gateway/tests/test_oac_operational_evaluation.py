"""Offline evaluation invariants: fixed inputs and no qualification authority."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "evaluate_operational_health_model.py"
SPEC = importlib.util.spec_from_file_location("oac_operational_evaluation", TOOL)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def admission_manifest() -> dict:
    return {
        "schema": evaluation.ADMISSION_SCHEMA,
        "data_class": "nonsensitive_operational_only",
        "contains_phi": False,
        "contains_clinical_results": False,
        "authorized_for_evaluation": True,
        **{key: "a" * 64 for key in evaluation.ADMISSION_HASH_FIELDS},
    }


class EvaluationMetricTests(unittest.TestCase):
    def test_wilson_interval_and_empty_support(self) -> None:
        estimate = evaluation.proportion(5, 10)
        self.assertEqual(0.5, estimate["value"])
        self.assertAlmostEqual(0.2365930905, estimate["wilson_95"]["lower"], places=9)
        self.assertAlmostEqual(0.7634069095, estimate["wilson_95"]["upper"], places=9)
        self.assertEqual(
            {"value": None, "numerator": 0, "denominator": 0, "wilson_95": None},
            evaluation.proportion(0, 0),
        )

    def test_metrics_known_answer(self) -> None:
        metrics = evaluation.calculate_metrics([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], 0.5, 0.5)
        self.assertEqual(
            {"true_positive": 1, "true_negative": 1, "false_positive": 1, "false_negative": 1},
            metrics["confusion"],
        )
        self.assertEqual(0.5, metrics["precision"]["value"])
        self.assertEqual(0.325, metrics["brier_score"])
        self.assertEqual(0.25, metrics["train_prevalence_baseline_brier"])
        self.assertAlmostEqual(-0.3, metrics["brier_skill_vs_train_prevalence"])
        self.assertEqual(0.75, metrics["roc_auc"])

    def test_ties_endpoints_and_empty_calibration_bins(self) -> None:
        metrics = evaluation.calculate_metrics([0, 1], [0.0, 1.0], 0.5, 0.5)
        self.assertEqual(0.0, metrics["brier_score"])
        self.assertEqual(0.0, metrics["expected_calibration_error"])
        self.assertEqual(1, metrics["calibration_bins"][0]["rows"])
        self.assertEqual(1, metrics["calibration_bins"][-1]["rows"])
        self.assertIsNone(metrics["calibration_bins"][1]["observed_positive_rate"]["value"])
        self.assertEqual(0.5, evaluation.calculate_metrics([0, 1], [0.5, 0.5], 0.5, 0.5)["roc_auc"])

    def test_single_class_is_not_reported_as_perfect_or_zero_auc(self) -> None:
        metrics = evaluation.calculate_metrics([0, 0], [0.1, 0.2], 0.5, 0.5)
        self.assertIsNone(metrics["precision"]["value"])
        self.assertIsNone(metrics["recall"]["value"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["balanced_accuracy"])

    def test_invalid_metric_inputs_are_rejected(self) -> None:
        cases = [([], []), ([0], []), ([2], [0.1]), ([True], [0.1]),
                 ([0], [float("nan")]), ([0], [float("inf")]), ([0], [-0.1]), ([0], [1.1]),
                 ([0], [10**1000]), ([0], [True])]
        for labels, scores in cases:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(evaluation.EvaluationError):
                evaluation.calculate_metrics(labels, scores, 0.5, 0.5)

    def test_invalid_threshold_prevalence_and_proportions_are_rejected(self) -> None:
        for invalid in (True, "0.5", -1, 2, 10**1000, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(evaluation.EvaluationError):
                    evaluation.calculate_metrics([0, 1], [0.2, 0.8], invalid, 0.5)
                with self.assertRaises(evaluation.EvaluationError):
                    evaluation.calculate_metrics([0, 1], [0.2, 0.8], 0.5, invalid)
        for numerator, denominator in ((-1, 1), (2, 1), (True, 1), (1, -1), (1.0, 2)):
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.proportion(numerator, denominator)


class EvaluationInputTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_nonfinite_overflow_and_deep_input(self) -> None:
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e400}', b'[]',
                    b'\xff', b'{"a":' + b'[' * 64 + b'0' + b']' * 64 + b'}',
                    b'{"a":' + b'[' * 1100 + b'0' + b']' * 1100 + b'}'):
            with self.subTest(raw=raw[:30]), self.assertRaises(evaluation.EvaluationError):
                evaluation.strict_object(raw)

    def test_json_depth_limit_is_explicit_and_not_runtime_recursion_limit(self) -> None:
        allowed = b'{"a":' + b'[' * 63 + b'0' + b']' * 63 + b'}'
        self.assertIsInstance(evaluation.strict_object(allowed), dict)

    def test_closed_row_schema_and_types(self) -> None:
        raw = (ROOT / "operational-model/data/test.jsonl").read_bytes().splitlines()[0]
        row = json.loads(raw)
        evaluation.validate_row(row, "test", 0)
        cases = []
        for key, value in (("synthetic", 1), ("sample_id", "train-000000"),
                           ("schema", "unknown"), ("patient_id", "must-not-be-echoed")):
            case = copy.deepcopy(row)
            case[key] = value
            cases.append(case)
        case = copy.deepcopy(row)
        case["label"]["operator_attention_required"] = 1
        cases.append(case)
        case = copy.deepcopy(row)
        case["features"]["listener_running"] = True
        cases.append(case)
        case = copy.deepcopy(row)
        case["features"]["queue_utilization"] = 2
        cases.append(case)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(evaluation.EvaluationError):
                evaluation.validate_row(case, "test", 0)

    def test_admission_declarations_never_become_verified_evidence(self) -> None:
        result = evaluation.check_admission(admission_manifest())
        self.assertEqual("MANIFEST_VALID_EVIDENCE_UNVERIFIED", result["status"])
        self.assertFalse(result["evidence_verified"])
        self.assertFalse(result["external_corpus_admitted"])
        self.assertFalse(result["production_promotion_allowed"])

    def test_admission_unknown_extra_sensitive_or_malformed_fields_fail(self) -> None:
        cases = []
        for key, value in (("contains_phi", True), ("contains_clinical_results", 0),
                           ("authorized_for_evaluation", False), ("data_class", "unknown"),
                           ("privacy_review_sha256", "UNKNOWN"), ("corpus_sha256", "A" * 64),
                           ("notes", "extra")):
            case = admission_manifest()
            case[key] = value
            cases.append(case)
        missing = admission_manifest()
        missing.pop("permission_evidence_sha256")
        cases.append(missing)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(evaluation.EvaluationError):
                evaluation.check_admission(case)

    def test_bounded_regular_file_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.read_bounded(root)
            path = root / "large.json"
            path.write_bytes(b"x" * 11)
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.read_bounded(path, maximum=10)


class CommittedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluation.evaluate()

    def test_fixed_heldout_confusion_threshold_and_sha(self) -> None:
        self.assertEqual(240, self.report["metrics"]["rows"])
        self.assertEqual(0.16, self.report["decision_threshold"])
        self.assertEqual(
            {"true_positive": 40, "true_negative": 132, "false_positive": 57, "false_negative": 11},
            self.report["metrics"]["confusion"],
        )
        self.assertEqual(evaluation.FIXED_MODEL_SHA256,
                         self.report["source_files_sha256"]["operational-model/artifacts/model.json"])
        self.assertEqual(evaluation.FIXED_TEST_SHA256,
                         self.report["source_files_sha256"]["operational-model/data/test.jsonl"])

    def test_success_does_not_authorize_use_or_claim_training(self) -> None:
        self.assertTrue(self.report["complete"])
        for field in ("production_promotion_allowed", "clinical_use_authorized", "training_rerun",
                      "threshold_tuned", "weights_updated", "source_signature_verified"):
            self.assertIs(self.report[field], False)
        self.assertEqual("NOT_SUPPLIED", self.report["external_corpus_admission"]["status"])
        self.assertIn("public", self.report["evaluation_scope"])
        self.assertIn("synthetic", self.report["evaluation_scope"])
        self.assertEqual(240, len(self.report["observations"]))
        self.assertTrue(all(len(row["observation_sha256"]) == 64 for row in self.report["observations"]))

    def test_evaluation_is_deterministic(self) -> None:
        self.assertEqual(self.report, evaluation.evaluate())

    def test_published_test_metric_drift_is_rejected(self) -> None:
        published = json.loads((ROOT / evaluation.MODEL_RECEIPT_FILE).read_bytes())["metrics"]
        evaluation.verify_receipt_metrics(published, self.report["metrics"])
        for field, value in (("rows", True), ("precision", 0.99), ("roc_auc", float("nan")),
                             ("recall", 10**1000), ("specificity", None), ("confusion", {})):
            changed = copy.deepcopy(published)
            changed["test"][field] = value
            with self.subTest(field=field), self.assertRaises(evaluation.EvaluationError):
                evaluation.verify_receipt_metrics(changed, self.report["metrics"])

    def test_source_change_during_evaluation_is_rejected(self) -> None:
        original = evaluation.read_bounded
        reads = 0

        def unstable(path: Path, maximum: int = evaluation.MAX_BYTES) -> bytes:
            nonlocal reads
            result = original(path, maximum)
            if path == ROOT / evaluation.MODEL_FILE:
                reads += 1
                if reads > 1:
                    return result + b" "
            return result

        with mock.patch.object(evaluation, "read_bounded", side_effect=unstable):
            with self.assertRaisesRegex(evaluation.EvaluationError, "INPUT_CHANGED_DURING_EVALUATION"):
                evaluation.evaluate()

    def test_fixed_model_and_test_split_drift_are_rejected(self) -> None:
        original = evaluation.read_bounded
        for changed_path in (evaluation.MODEL_FILE, "operational-model/data/test.jsonl"):
            def changed(path: Path, maximum: int = evaluation.MAX_BYTES) -> bytes:
                result = original(path, maximum)
                return result + b" " if path == ROOT / changed_path else result

            with self.subTest(path=changed_path), mock.patch.object(evaluation, "read_bounded", side_effect=changed):
                with self.assertRaisesRegex(evaluation.EvaluationError, "FIXED_BASELINE_DRIFT"):
                    evaluation.evaluate()

    def test_cli_writes_once_and_retains_authority_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            command = [sys.executable, "-I", "-B", str(TOOL), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(0, first.returncode, first.stderr)
            original = output.read_bytes()
            self.assertEqual(self.report, json.loads(original))
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(2, second.returncode)
            self.assertEqual(original, output.read_bytes())
            failure = json.loads(second.stderr)
            self.assertFalse(failure["complete"])
            self.assertFalse(failure["production_promotion_allowed"])
            self.assertFalse(failure["clinical_use_authorized"])

    def test_cli_invalid_admission_is_a_closed_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text('{"private":"do-not-echo"}', encoding="utf-8")
            output = Path(temporary) / "failed.json"
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(TOOL), "--admission-manifest", str(path),
                 "--output", str(output)], capture_output=True, text=True, check=False,
            )
            self.assertEqual(2, proc.returncode)
            self.assertNotIn("do-not-echo", proc.stdout + proc.stderr + output.read_text())
            self.assertFalse(json.loads(output.read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
