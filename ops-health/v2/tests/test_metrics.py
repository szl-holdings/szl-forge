"""M6: metric recomputation, hand cases, and INVALID_BASELINE preconditions (SYNTHETIC)."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from v2.research import bootstrap, gates, metrics, pipeline, registry
from v2.research.dataio import labels_of
from v2.tests import _support as S

K = S.kernel()

# REPORTED: SZLHOLDINGS/oac-system-health-v1 artifact_receipt.json at Hub revision
# dd7d109813abcd90c5250106dc2eabd2804e7ac3, metrics.test (12-decimal values as published).
V1_TEST_COUNTS = {"tp": 40, "fp": 57, "tn": 132, "fn": 11}
V1_TEST_PUBLISHED = {
    "accuracy": 0.716666666667,
    "balanced_accuracy": 0.741363211951,
    "f1": 0.540540540541,
    "precision": 0.412371134021,
    "recall": 0.78431372549,
    "specificity": 0.698412698413,
}
# REPORTED: the v1 model card's rounded values (CONTRACT section 2 step 7).
V1_CARD = {"precision": 0.4124, "recall": 0.7843, "balanced_accuracy": 0.7414, "f1": 0.5405}


class HandCaseTest(unittest.TestCase):
    def test_confusion_and_rates(self):
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        d = [1, 0, 1, 1, 0, 0, 0, 0]
        c = metrics.confusion(y, d)
        self.assertEqual(c, {"tp": 2, "fp": 1, "tn": 3, "fn": 2})
        r = metrics.rates_from_counts(**c)
        self.assertEqual(r["precision"], 2 / 3)
        self.assertEqual(r["recall"], 0.5)
        self.assertEqual(r["specificity"], 0.75)
        self.assertEqual(r["balanced_accuracy"], 0.625)
        self.assertEqual(r["f1"], 4 / 7)
        self.assertEqual(r["false_alert_share"], 1 / 3)
        self.assertEqual(r["accuracy"], 5 / 8)
        self.assertEqual(r["prevalence"], 0.5)

    def test_undefined_is_null_not_zero(self):
        r = metrics.rates_from_counts(0, 0, 10, 0)  # no positives, no alerts
        self.assertIsNone(r["precision"])
        self.assertIsNone(r["recall"])
        self.assertIsNone(r["balanced_accuracy"])
        self.assertIsNone(r["false_alert_share"])
        self.assertIsNone(r["f1"])
        self.assertEqual(r["specificity"], 1.0)
        self.assertIsNone(metrics.roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))
        self.assertIsNone(metrics.brier([], []))
        self.assertEqual(json.loads(json.dumps(r))["precision"], None)

    def test_brier_and_log_loss(self):
        y = [1, 0, 1, 0]
        s = [0.9, 0.2, 0.6, 0.0]
        self.assertAlmostEqual(metrics.brier(y, s), (0.01 + 0.04 + 0.16 + 0.0) / 4, places=15)
        expected = -(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(1.0 - 1e-15)) / 4
        self.assertAlmostEqual(metrics.log_loss(y, s), expected, places=15)
        with self.assertRaises(ValueError):
            metrics.brier([1], [1.5])

    def test_ece_bins_last_closed(self):
        for k in range(10):
            self.assertEqual(metrics.ece_bin(k / 10), k)
            if k:
                self.assertEqual(metrics.ece_bin(math.nextafter(k / 10, 0.0)), k - 1)
        self.assertEqual(metrics.ece_bin(1.0), 9)
        self.assertEqual(metrics.ece_bin(0.0), 0)

    def test_ece_hand_case(self):
        y = [0, 1, 1, 1]
        s = [0.05, 0.15, 0.95, 1.0]
        # bins 0, 1, 9, 9: gaps |0.05 - 0| + |0.15 - 1| + |1.95 - 2| = 0.95, weighted by count / n
        self.assertAlmostEqual(metrics.ece(y, s), 0.95 / 4, places=15)
        rel = metrics.reliability(y, s)
        self.assertEqual([b["count"] for b in rel["table"]], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(rel["table"][5]["mean_score"])
        self.assertTrue(rel["table"][9]["upper_closed"])
        self.assertEqual(metrics.ece([1, 0], [0.5, 0.5]), 0.0)

    def test_auroc_average_ranks(self):
        self.assertEqual(metrics.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(metrics.roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(metrics.roc_auc([0, 1, 0, 1], [0.2, 0.2, 0.2, 0.9]), 0.75)
        self.assertEqual(metrics.roc_auc([1, 0], [0.1, 0.9]), 0.0)
        # brute-force pair count on a tie-heavy toy
        y = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        s = [0.3, 0.3, 0.7, 0.1, 0.7, 0.2, 0.3, 0.9, 0.5, 0.1]
        pairs = [(a, b) for a, ya in zip(s, y) if ya == 1 for b, yb in zip(s, y) if yb == 0]
        brute = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
        self.assertAlmostEqual(metrics.roc_auc(y, s), brute, places=15)

    def test_selective_and_coverage(self):
        y = [1, 0, 1, 0, 1]
        adv = ["ALERT", "NO_ALERT", "ABSTAIN", "ABSTAIN", "NO_ALERT"]
        why = [None, None, "BOTH", "EMPTY", None]
        out = metrics.selective(y, adv, why)
        self.assertEqual(out["abstention_rate"], 0.4)
        self.assertEqual(out["abstention_rate_both"], 0.2)
        self.assertEqual(out["abstention_rate_empty"], 0.2)
        self.assertEqual(out["selective_balanced_accuracy"], (0.5 + 1.0) / 2)
        cov = metrics.coverage(y, [[1], [0], [0, 1], [], [0]])
        self.assertEqual(cov["marginal"], 3 / 5)
        self.assertEqual(cov["per_class"], {"0": 0.5, "1": 2 / 3})


class V1RecomputationTest(unittest.TestCase):
    def test_v1_counts_reproduce_published_rates(self):
        r = metrics.rates_from_counts(**V1_TEST_COUNTS)
        for key, published in V1_TEST_PUBLISHED.items():
            with self.subTest(metric=key):
                self.assertEqual(round(r[key], 12), published)
        for key, card in V1_CARD.items():
            with self.subTest(card=key):
                self.assertLessEqual(abs(r[key] - card), 0.0005)
        self.assertEqual(sum(V1_TEST_COUNTS.values()), 240)

    def test_published_values_match_local_hub_snapshot_if_present(self):
        path = registry.V1_HUB_DIR / "artifact_receipt.json"
        if not path.exists():
            self.skipTest("hub/oac-v1 snapshot not present")
        test = json.loads(path.read_text(encoding="utf-8"))["metrics"]["test"]
        c = test["confusion"]
        self.assertEqual(
            {"tp": c["true_positive"], "fp": c["false_positive"], "tn": c["true_negative"],
             "fn": c["false_negative"]},
            V1_TEST_COUNTS,
        )
        for key, value in V1_TEST_PUBLISHED.items():
            self.assertEqual(test[key], value)

    def test_per_row_metrics_equal_count_metrics(self):
        y = [1] * 51 + [0] * 189
        d = [1] * 40 + [0] * 11 + [1] * 57 + [0] * 132
        self.assertEqual(metrics.confusion(y, d), V1_TEST_COUNTS)
        self.assertEqual(metrics.balanced_accuracy(y, d),
                         metrics.rates_from_counts(**V1_TEST_COUNTS)["balanced_accuracy"])


class InvalidBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dev = list(S.dev_rows())
        cls.rows = dev[S.EVAL][:4000]
        cls.y = labels_of(cls.rows)
        cls.vectors = pipeline.normalized_vectors(cls.rows)
        cls.scorer = K.ArtifactScorer(S.candidate())
        cls.d_good = [1 if cls.scorer.evaluate(v)["threshold_decision"] else 0 for v in cls.vectors]

    def _report(self, decisions, y=None, **kw):
        y = self.y if y is None else y
        validity = metrics.baseline_validity(y, decisions, **kw)
        return metrics.binary_report(y, decisions, validity)

    def assertInvalid(self, report, failed):
        self.assertEqual(report["baseline_validity"]["status"], metrics.INVALID_BASELINE)
        self.assertEqual(report["baseline_validity"]["failed_preconditions"], failed)
        self.assertEqual(report["per_class_rates"], metrics.INVALID_BASELINE)
        self.assertNotIn("confusion", report)
        text = json.dumps(report)
        for key in metrics.PER_CLASS_RATE_KEYS:
            self.assertNotIn(f'"{key}"', text)

    def test_valid_fixture_reports_rates(self):
        rep = self._report(self.d_good, receipt_verified=True, probes_refused=5, probes_total=5)
        self.assertEqual(rep["baseline_validity"]["status"], "VALID")
        self.assertEqual(set(rep["per_class_rates"]), set(metrics.PER_CLASS_RATE_KEYS))
        self.assertIn("confusion", rep)

    def test_constant_alert(self):
        rep = self._report([1] * len(self.y), receipt_verified=True, probes_refused=5, probes_total=5)
        self.assertInvalid(rep, ["iii_decisions_not_constant"])
        self.assertEqual(rep["balanced_accuracy"], 0.5)

    def test_constant_no_alert_majority(self):
        rep = self._report([0] * len(self.y), receipt_verified=True, probes_refused=5, probes_total=5)
        self.assertInvalid(rep, ["iii_decisions_not_constant"])
        self.assertEqual(rep["balanced_accuracy"], 0.5)

    def test_fewer_than_30_rows_of_a_class(self):
        pos = [i for i, v in enumerate(self.y) if v == 1][:29]
        neg = [i for i, v in enumerate(self.y) if v == 0][:500]
        idx = sorted(pos + neg)
        y = [self.y[i] for i in idx]
        d = [self.d_good[i] for i in idx]
        self.assertEqual(sum(y), 29)
        rep = self._report(d, y=y, receipt_verified=True, probes_refused=5, probes_total=5)
        self.assertInvalid(rep, ["ii_min_rows_per_class"])
        pos30 = [i for i, v in enumerate(self.y) if v == 1][:30]
        idx = sorted(pos30 + neg)
        rep = self._report([self.d_good[i] for i in idx], y=[self.y[i] for i in idx],
                           receipt_verified=True, probes_refused=5, probes_total=5)
        self.assertEqual(rep["baseline_validity"]["status"], "VALID")

    def test_unverified_receipt(self):
        rep = self._report(self.d_good, receipt_verified=False, probes_refused=5, probes_total=5)
        self.assertInvalid(rep, ["i_receipt_verifies"])

    def test_fail_open_kernel_fixture(self):
        """A deliberately broken kernel that accepts prohibited or malformed input, with a
        consistently regenerated receipt (so precondition (i) holds and only (iv) fails)."""
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "art")
            mutant, path = S.load_kernel_copy(tmp, mutate=S.fail_open_mutation)
            receipt = json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))
            receipt["kernel_sha256"] = S.sha256_file(path)
            regenerated = tmp / "regenerated_receipt.json"
            regenerated.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            kernel = mutant.OpsHealthKernel(info["model_path"], regenerated)  # (i) verifies
            probes = gates.run_failclosed_probes(kernel.advise)
            self.assertGreater(len(probes["escaped"]), 0)
            for accepted in ("extra_field", "exact_patient", "substring_hl7", "nan_continuous",
                             "text_instead_of_number", "prohibited_replaces_field"):
                self.assertIn(accepted, probes["escaped"])
            decisions = [
                1 if kernel.advise({"features": row["features"]})["threshold_decision"] else 0
                for row in self.rows
            ]
            self.assertEqual(decisions, self.d_good)  # same model: it scores valid rows alike
            rep = self._report(decisions, receipt_verified=True, probes_refused=probes["refused"],
                               probes_total=probes["total"])
            self.assertInvalid(rep, ["iv_refused_all_probes"])
            # the real kernel with the same model refuses every probe -> VALID
            good = K.OpsHealthKernel(info["model_path"], info["receipt_path"])
            ok = gates.run_failclosed_probes(good.advise)
            rep = self._report(self.d_good, receipt_verified=True, probes_refused=ok["refused"],
                               probes_total=ok["total"])
            self.assertEqual(rep["baseline_validity"]["status"], "VALID")
            self.assertEqual(gates.g_baseline_valid(rep["baseline_validity"]), 1.0)

    def test_multiple_failures_are_all_listed(self):
        rep = self._report([1] * 10, y=[1] * 5 + [0] * 5, receipt_verified=False,
                           probes_refused=0, probes_total=0)
        self.assertInvalid(rep, ["i_receipt_verifies", "ii_min_rows_per_class",
                                 "iii_decisions_not_constant", "iv_refused_all_probes"])


class WeightedEqualsPlainTest(unittest.TestCase):
    """The bootstrap's count-weighted metrics equal metrics.py at unit weights (DEV rows)."""

    def test_unit_weights(self):
        dev = list(S.dev_rows())
        rows = dev[S.EVAL][:5000]
        y = labels_of(rows)
        scorer = K.ArtifactScorer(S.candidate())
        ev = [scorer.evaluate(v) for v in pipeline.normalized_vectors(rows)]
        s = [e["calibrated_score"] for e in ev]
        d = [1 if e["threshold_decision"] else 0 for e in ev]
        ones = [1] * len(y)
        ss = bootstrap.ScoreStats(y, s)
        bs = bootstrap.BinaryStats(y, d)
        self.assertEqual(bs.balanced_accuracy(ones), metrics.balanced_accuracy(y, d))
        self.assertEqual(ss.brier(ones), metrics.brier(y, s))
        self.assertEqual(ss.log_loss(ones), metrics.log_loss(y, s))
        self.assertEqual(ss.ece(ones), metrics.ece(y, s))
        self.assertAlmostEqual(ss.auroc(ones), metrics.roc_auc(y, s), places=14)
        # doubling every weight leaves every statistic unchanged
        twos = [2] * len(y)
        self.assertAlmostEqual(ss.ece(twos), ss.ece(ones), places=14)
        self.assertAlmostEqual(ss.auroc(twos), ss.auroc(ones), places=14)


if __name__ == "__main__":
    unittest.main()
