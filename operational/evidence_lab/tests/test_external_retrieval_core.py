"""Synthetic offline regression tests; not external model-evaluation evidence."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "operational" / "external_retrieval_core.py"
SPEC = importlib.util.spec_from_file_location("external_core_under_test", SOURCE)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def record(margin, answerable, prediction="Paris", gold=None):
    return {"margin": margin, "answerable": answerable, "prediction": prediction,
            "gold": (["Paris"] if answerable else []) if gold is None else gold}


class RankingTests(unittest.TestCase):
    def test_bm25_matches_toy_math(self):
        ranker = core.BM25(["cat cat dog", "dog", ""])
        idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
        expected = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / (4 / 3)))
        self.assertAlmostEqual(expected, ranker.scores("CAT")[0])
        self.assertEqual([0.0, 0.0], ranker.scores("cat")[1:])
        self.assertAlmostEqual(expected * 2, ranker.scores("cat cat")[0])

    def test_bm25_empty_documents_query_and_unseen_terms(self):
        for query in ("", "unseen", "!!!"):
            self.assertEqual([0.0, 0.0], core.BM25(["", "!"]).scores(query))
        self.assertEqual([0.0], core.BM25(["cat"]).scores("dog"))

    def test_bm25_extreme_positive_finite_k1(self):
        for k1 in (math.nextafter(0, 1), sys.float_info.max):
            with self.subTest(k1=k1):
                score = core.BM25(["cat"], k1=k1).scores("cat")[0]
                self.assertTrue(math.isfinite(score))
                self.assertAlmostEqual(math.log1p(0.5 / 1.5), score)

    def test_bm25_rejects_bad_texts_and_parameters(self):
        for texts in ([], "cat", [1], None):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                core.BM25(texts)
        for kwargs in ({"k1": 0}, {"k1": math.nan}, {"b": True}, {"b": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                core.BM25(["cat"], **kwargs)
        with self.assertRaises(ValueError):
            core.BM25(["cat"]).scores(None)

    def test_stable_order_ties_negative_and_invalid_scores(self):
        self.assertEqual([2, 0, 1, 3], core.stable_order([-1.0, -1.0, 0.0, -2.0]))
        self.assertEqual([1, 0], core.stable_order([2 ** 53, 2 ** 53 + 1]))
        for value in (math.nan, math.inf, -math.inf, True, "1", None, 10 ** 1000):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                core.stable_order([0, value])
        with self.assertRaises(ValueError):
            core.stable_order([])

    def test_fixed_rrf_full_vectors_and_stable_ties(self):
        result = core.rrf_fusion([[3, 2, 1], [1, 2, 3]])
        self.assertAlmostEqual(1 / 61 + 1 / 63, result[0])
        self.assertAlmostEqual(2 / 62, result[1])
        self.assertEqual(result[0], result[2])
        self.assertEqual([0, 1, 2], core.stable_order(core.rrf_fusion([[0, 0, 0]])))
        self.assertEqual([2 / 61, 2 / 62], core.rrf_fusion([[1, 0], [1, 0]]))

    def test_rrf_rejects_mismatched_empty_nonfinite_inputs(self):
        for vectors in ([], [[]], [[1], [1, 2]], [[1, math.nan]], [[True]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                core.rrf_fusion(vectors)

    def test_multi_positive_recall_not_hit_rate_and_ideal_cutoff(self):
        scores = list(range(12, 0, -1))
        result = core.retrieval_metrics([scores], [[0, 11]])
        self.assertEqual(1, result["mrr_at_10"])
        for cutoff in (1, 5, 10):
            self.assertEqual(0.5, result[f"recall_at_{cutoff}"])
        self.assertAlmostEqual(1 / (1 + 1 / math.log2(3)), result["ndcg_at_10"])
        all_relevant = core.retrieval_metrics([scores], [list(range(12))])
        self.assertEqual(1, all_relevant["ndcg_at_10"])
        self.assertEqual(10 / 12, all_relevant["recall_at_10"])

    def test_macro_average_cutoff_and_ties(self):
        result = core.retrieval_metrics([[0] * 12, [0] * 12], [[0, 11], [1]])
        self.assertEqual(0.75, result["mrr_at_10"])
        self.assertEqual(0.25, result["recall_at_1"])
        self.assertEqual(0.75, result["recall_at_10"])
        self.assertEqual(0, core.retrieval_metrics([[0] * 12], [[11]])["mrr_at_10"])

    def test_metrics_reject_duplicate_invalid_and_missing_labels(self):
        for positives in ([], [[0, 0]], [[True]], [[-1]], [[2]], [[0.0]], "0", [[]]):
            with self.subTest(positives=positives), self.assertRaises(ValueError):
                core.retrieval_metrics([[2, 1]], positives)
        for scores, positives in (([], []), ([[]], [[0]]), ([[math.nan]], [[0]]),
                                  ([[1], [1]], [[0]])):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                core.retrieval_metrics(scores, positives)


class QATests(unittest.TestCase):
    def test_squad_normalization(self):
        self.assertEqual("quick foxs", core.normalize_answer(" A QUICK,  fox's! the an "))
        self.assertEqual("café", core.normalize_answer("The Café."))

    def test_qa_max_over_references_and_multiset_tokens(self):
        self.assertEqual({"exact_match": 1, "f1": 1},
                         core.qa_exact_f1("The Paris!", ["London", "Paris"]))
        self.assertEqual({"exact_match": 0, "f1": 0.8},
                         core.qa_exact_f1("red red blue", ["red blue"]))
        self.assertEqual(1, core.qa_exact_f1("", [])["f1"])
        self.assertEqual(0, core.qa_exact_f1("wrong", [""])["f1"])
        self.assertEqual(0, core.qa_exact_f1("", ["Paris"])["f1"])

    def test_qa_invalid_input(self):
        for prediction, golds in ((1, ["x"]), ("x", "x"), ("x", [None])):
            with self.subTest(prediction=prediction, golds=golds), self.assertRaises(ValueError):
                core.qa_exact_f1(prediction, golds)


class AbstentionTests(unittest.TestCase):
    def test_strict_threshold_and_metrics(self):
        rows = [record(2, True), record(1, True), record(1, False), record(3, False)]
        metrics = core.answerability_metrics(rows, 1)
        self.assertEqual(0.5, metrics["coverage"])
        self.assertEqual(0.5, metrics["false_answer_rate"])
        self.assertEqual(0.5, metrics["supported_recall"])
        self.assertEqual(0.5, metrics["abstention_accuracy"])
        self.assertEqual(0.5, metrics["qa_f1"])
        self.assertEqual(0.5, metrics["qa_exact_match"])
        self.assertEqual([0.09453120573423074, 0.9054687942657693],
                         metrics["false_answer_rate_wilson_ci"])

    def test_threshold_optimizes_qa_under_empirical_cap(self):
        rows = [record(5, True), record(4, True), record(3, False), record(2, False)]
        result = core.choose_threshold(rows)
        self.assertEqual(3, result["threshold"])
        self.assertEqual(1, result["metrics"]["qa_f1"])
        self.assertEqual(0, result["metrics"]["false_answer_rate"])
        self.assertEqual("calibration_only", result["calibration_label"])
        self.assertEqual("margin > threshold", result["threshold_rule"])

    def test_qa_objective_can_prefer_abstaining_over_more_coverage(self):
        rows = [record(3, True, "wrong"), record(2, True, "wrong"), record(1, False)]
        result = core.choose_threshold(rows)
        self.assertEqual(3, result["threshold"])
        self.assertEqual(0, result["metrics"]["coverage"])

    def test_tied_qa_prefers_lower_false_answer_rate_then_higher_threshold(self):
        # Answering the positive and negative gains and loses one exact answer.
        rows = [record(3, True), record(3, False)]
        result = core.choose_threshold(rows, max_false_answer_rate=1)
        self.assertEqual(3, result["threshold"])
        self.assertEqual(0, result["metrics"]["false_answer_rate"])

    def test_five_percent_cap_uses_unanswerable_denominator(self):
        rows = [record(10, True), record(9, True), record(8, False)]
        rows.extend(record(0, False) for _ in range(19))
        result = core.choose_threshold(rows)
        self.assertEqual(8, result["threshold"])
        self.assertLessEqual(result["metrics"]["false_answer_rate"], 0.05)
        observed = core.answerability_metrics(rows, 0)
        self.assertEqual(0.05, observed["false_answer_rate"])

    def test_cap_boundary_allows_one_in_twenty_but_not_a_lower_cap(self):
        rows = [record(9, True), record(8, True), record(10, False)]
        rows.extend(record(0, False) for _ in range(19))
        permitted = core.choose_threshold(rows)
        restricted = core.choose_threshold(rows, max_false_answer_rate=0.049)
        self.assertEqual(0, permitted["threshold"])
        self.assertEqual(0.05, permitted["metrics"]["false_answer_rate"])
        self.assertEqual(10, restricted["threshold"])
        self.assertEqual(0, restricted["metrics"]["false_answer_rate"])

    def test_answerability_threshold_and_records_are_validated(self):
        for threshold in (math.nan, math.inf, True, "0"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                core.answerability_metrics([record(1, True)], threshold)
        with self.assertRaises(ValueError):
            core.answerability_metrics([], 0)

    def test_only_passed_calibration_data_used_and_not_mutated(self):
        calibration = [record(5, True), record(2, False)]
        before = copy.deepcopy(calibration)
        test_data = mock.Mock(side_effect=AssertionError("test data must not be touched"))
        with mock.patch("builtins.open", side_effect=AssertionError("no file reads")), \
             mock.patch.object(core, "test_data", test_data, create=True):
            first = core.choose_threshold(calibration)
            test_data.margin = 1000000
            second = core.choose_threshold(calibration)
        self.assertEqual(first, second)
        self.assertEqual(before, calibration)
        test_data.assert_not_called()

    def test_requires_both_calibration_classes_and_rejects_bad_records(self):
        cases = [[], [record(1, True)], [record(1, False)], [None], [{}]]
        for update in ({"margin": math.nan}, {"margin": math.inf}, {"margin": True},
                       {"answerable": 1}, {"prediction": None}, {"gold": "Paris"},
                       {"gold": []}, {"gold": [""]}, {"gold": ["Paris", ""]}):
            cases.append([{**record(1, True), **update}, record(0, False)])
        cases.append([record(1, True), record(0, False, gold=["Paris"])])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                core.choose_threshold(rows)
        for limit in (-1, 2, True, math.nan):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                core.choose_threshold([record(1, True), record(0, False)], limit)

    def test_extreme_finite_margins_have_finite_threshold(self):
        result = core.choose_threshold([record(-sys.float_info.max, False),
                                        record(sys.float_info.max, True)])
        self.assertTrue(math.isfinite(result["threshold"]))
        self.assertEqual(1, result["metrics"]["qa_f1"])

    def test_single_class_metrics_report_missing_denominator(self):
        positive = core.answerability_metrics([record(1, True)], 0)
        self.assertIsNone(positive["false_answer_rate"])
        self.assertIsNone(positive["false_answer_rate_wilson_ci"])
        negative = core.answerability_metrics([record(1, False)], 0)
        self.assertIsNone(negative["supported_recall"])

    def test_zero_false_answers_do_not_claim_zero_population_upper_bound(self):
        interval = core.wilson_interval(0, 20)
        self.assertAlmostEqual(0, interval[0])
        self.assertGreater(interval[1], 0.05)
        self.assertAlmostEqual(0.16112515805281938, interval[1])
        for counts in ((True, 2), (1, 0), (-1, 2), (0, -1), (1.0, 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                core.wilson_interval(*counts)


if __name__ == "__main__":
    unittest.main()
