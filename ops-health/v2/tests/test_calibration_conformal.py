"""M6: calibrators (Platt, isotonic PAV) and Mondrian split conformal (SYNTHETIC, DEV data)."""

from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from v2.research import calibration, conformal, generator, metrics, pipeline
from v2.research.dataio import labels_of
from v2.tests import _support as S

K = S.kernel()


def pav_reference(scores, labels):
    """Min-max formula for the (count-weighted) isotonic regression over tie blocks."""
    keys = sorted(set(scores))
    pos = [sum(y for s, y in zip(scores, labels) if s == k) for k in keys]
    cnt = [sum(1 for s in scores if s == k) for k in keys]
    out = []
    for i in range(len(keys)):
        best = None
        for j in range(i + 1):
            inner = min(Fraction(sum(pos[j:k + 1]), sum(cnt[j:k + 1])) for k in range(i, len(keys)))
            best = inner if best is None or inner > best else best
        out.append(best)
    return keys, out


class IsotonicTest(unittest.TestCase):
    def test_toy_pav(self):
        cal, diag = calibration.fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
        self.assertEqual(cal["params"]["thresholds"], [0.1, 0.2, 0.4])
        self.assertEqual(cal["params"]["values"], [0.0, 0.5, 1.0])
        self.assertEqual(diag["block_counts"], [1, 2, 1])
        apply = lambda s: calibration.apply(cal, s)  # noqa: E731
        self.assertEqual(apply(0.05), 0.0)  # below every block -> first block's value
        self.assertEqual(apply(0.1), 0.0)
        self.assertEqual(apply(0.25), 0.5)  # step prediction, no interpolation
        self.assertEqual(apply(0.3), 0.5)
        self.assertEqual(apply(0.4), 1.0)
        self.assertEqual(apply(0.99), 1.0)

    def test_ties_pooled_first(self):
        cal, diag = calibration.fit_isotonic([0.5, 0.5, 0.5, 0.7], [1, 0, 0, 1])
        self.assertEqual(diag["tie_blocks"], 2)
        self.assertEqual(cal["params"]["thresholds"], [0.5, 0.7])
        self.assertEqual(cal["params"]["values"], [0.333333333333, 1.0])
        # ties are pooled regardless of row order
        cal2, _ = calibration.fit_isotonic([0.7, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        self.assertEqual(cal, cal2)
        # equal-mean neighbours are merged, so fitted values are strictly increasing
        cal3, _ = calibration.fit_isotonic([0.1, 0.2, 0.3], [1, 1, 0])
        self.assertEqual(cal3["params"], {"thresholds": [0.1], "values": [0.666666666667]})

    def test_matches_min_max_reference_on_random_toys(self):
        rng = random.Random(generator.derive_seed("dev"))
        for trial in range(40):
            n = rng.randint(1, 25)
            scores = [rng.choice((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) for _ in range(n)]
            labels = [1 if rng.random() < s else 0 for s in scores]
            cal, _ = calibration.fit_isotonic(scores, labels)
            keys, ref = pav_reference(scores, labels)
            with self.subTest(trial=trial):
                for key, value in zip(keys, ref):
                    self.assertEqual(calibration.apply(cal, key), round(float(value), 12))

    def test_pav_monotone_on_dev(self):
        dev = list(S.dev_rows())
        art = S.candidate(calibrator="none")
        scorer = K.ArtifactScorer(art)
        raw = [scorer.evaluate(v)["raw_score"] for v in pipeline.normalized_vectors(dev[S.CALIBRATION])]
        cal, diag = calibration.fit_isotonic(raw, labels_of(dev[S.CALIBRATION]))
        values = cal["params"]["values"]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))
        self.assertEqual(sum(diag["block_counts"]), len(raw))
        grid = [i / 1000 for i in range(1001)]
        out = calibration.apply_many(cal, grid)
        self.assertTrue(all(b >= a for a, b in zip(out, out[1:])))
        self.assertEqual(K.validate_calibrator(cal), cal)


class PlattTest(unittest.TestCase):
    """DEV labels are Bernoulli draws from the generator-truth probability p, so Platt on
    logit(p) must recover a = 1, b = 0, and on a known distortion of p it must undo it."""

    @classmethod
    def setUpClass(cls):
        rows = list(S.dev_rows())[S.EVAL]
        cls.y = labels_of(rows)
        cls.p = [generator.truth_probability(r["features"]) for r in rows]  # nominal: observed = latent

    def test_recovers_identity(self):
        cal, diag = calibration.fit_platt(self.p, self.y)
        self.assertTrue(diag["converged"])
        self.assertLess(diag["gradient_max_abs"], 1e-9)
        self.assertLess(abs(cal["params"]["a"] - 1.0), 0.1)
        self.assertLess(abs(cal["params"]["b"] - 0.0), 0.1)

    def test_recovers_known_distortion(self):
        a0, b0 = 2.0, -0.5
        distorted = [K.sigmoid((K.logit_clipped(p) - b0) / a0) for p in self.p]
        cal, diag = calibration.fit_platt(distorted, self.y)
        self.assertTrue(diag["converged"])
        self.assertLess(abs(cal["params"]["a"] - a0), 0.1)
        self.assertLess(abs(cal["params"]["b"] - b0), 0.1)
        # frozen resolution: 12 decimals
        self.assertEqual(cal["params"]["a"], round(cal["params"]["a"], 12))

    def test_exact_toy_and_errors(self):
        # symmetric toy: logit(0.25) = -log 3 with 1/4 positive, logit(0.75) = log 3 with 3/4
        # positive, so the MLE is exactly a = 1, b = 0
        scores = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
        labels = [0, 0, 0, 1, 0, 1, 1, 1]
        cal, _ = calibration.fit_platt(scores, labels)
        self.assertAlmostEqual(cal["params"]["a"], 1.0, places=10)
        self.assertAlmostEqual(cal["params"]["b"], 0.0, places=10)
        with self.assertRaises(calibration.CalibrationError):
            calibration.fit_platt([0.2, 0.3], [1, 1])
        with self.assertRaises(calibration.CalibrationError):
            calibration.fit("beta", [0.2], [1])


class ConformalTest(unittest.TestCase):
    def test_quantile_rank_exact(self):
        self.assertEqual(conformal.quantile_rank(99, 0.1), 90)
        self.assertEqual(conformal.quantile_rank(9, 0.1), 9)
        self.assertEqual(conformal.quantile_rank(8, 0.1), 9)  # > n -> +inf
        self.assertEqual(conformal.quantile_rank(3348, 0.1), math.ceil(3349 * Fraction(9, 10)))
        with self.assertRaises(conformal.ConformalError):
            conformal.quantile_rank(10, 1.0)

    def test_toy_quantiles_and_infinite_rank(self):
        # class 1: p1 = 0.05..0.95 (10 rows) -> scores 1 - p1; rank ceil(11 * 0.9) = 10 -> max
        p1_pos = [0.05 + 0.1 * i for i in range(10)]
        # class 0: 8 rows -> rank 9 > 8 -> +inf, stored as 1.0
        p1_neg = [0.1 * i for i in range(8)]
        fit = conformal.fit_mondrian(p1_pos + p1_neg, [1] * 10 + [0] * 8, 0.10)
        block, diag = fit["block"], fit["diagnostics"]
        self.assertEqual(diag["rank"], {"1": 10, "0": 9})
        self.assertTrue(diag["infinite"]["0"])
        self.assertEqual(block["q_hat"]["0"], 1.0)
        self.assertEqual(block["q_hat"]["1"], conformal.frozen_q_hat(max(1.0 - p for p in p1_pos)))
        self.assertGreaterEqual(block["q_hat"]["1"], diag["q_hat_exact"]["1"])
        self.assertEqual(block["n_calibration"], {"0": 8, "1": 10})

    def test_rank_selects_kth_smallest(self):
        scores_neg = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0]  # p1 of class-0 rows
        fit = conformal.fit_mondrian(scores_neg + [0.9] * 10, [0] * 10 + [1] * 10, 0.2)
        # n = 10, alpha 0.2: rank ceil(11 * 0.8) = ceil(8.8) = 9 -> 9th smallest of s(x,0) = p1
        self.assertEqual(fit["diagnostics"]["rank"]["0"], 9)
        self.assertEqual(fit["diagnostics"]["q_hat_exact"]["0"], 0.8)

    def test_frozen_q_hat_rounds_up(self):
        for q in (0.1234567890123456, 0.9999999999994, 1e-13, 0.5, 0.3):
            stored = conformal.frozen_q_hat(q)
            self.assertGreaterEqual(stored, q)
            self.assertLess(stored - q, 1.01e-12)
        self.assertEqual(conformal.frozen_q_hat(math.inf), 1.0)

    def test_advisory_mapping(self):
        block = {"method": "mondrian_split_conformal", "alpha": 0.1,
                 "q_hat": {"0": 0.3, "1": 0.4}, "n_calibration": {"0": 10, "1": 10}}
        cases = {
            0.9: ([1], "ALERT", None),        # s0 = 0.9 > 0.3, s1 = 0.1 <= 0.4
            0.1: ([0], "NO_ALERT", None),     # s0 = 0.1 <= 0.3, s1 = 0.9 > 0.4
            0.65: ([1], "ALERT", None),       # s0 = 0.65 > 0.3, s1 = 0.35 <= 0.4
            0.5: ([], "ABSTAIN", "EMPTY"),    # s0 = 0.5 > 0.3, s1 = 0.5 > 0.4
        }
        for p1, (expected_set, advisory, reason) in cases.items():
            with self.subTest(p1=p1):
                pset = conformal.prediction_set(block, p1)
                self.assertEqual(pset, expected_set)
                self.assertEqual(conformal.advisory(pset), (advisory, reason))
        wide = dict(block, q_hat={"0": 0.7, "1": 0.7})
        self.assertEqual(conformal.prediction_set(wide, 0.5), [0, 1])
        self.assertEqual(conformal.advisory([0, 1]), ("ABSTAIN", "BOTH"))

    def test_mondrian_per_class_coverage_on_20k_dev_rows(self):
        dev = list(S.dev_rows())
        eval_rows = dev[S.EVAL]
        self.assertEqual(len(eval_rows), 20000)
        y = labels_of(eval_rows)
        vectors = pipeline.normalized_vectors(eval_rows)
        for kind in calibration.KINDS:
            scorer = K.ArtifactScorer(S.candidate(calibrator=kind))
            sets = [scorer.evaluate(v)["prediction_set"] for v in vectors]
            cov = metrics.coverage(y, sets)
            with self.subTest(calibrator=kind):
                self.assertGreaterEqual(cov["per_class"]["0"], 0.9 - 0.02)
                self.assertGreaterEqual(cov["per_class"]["1"], 0.9 - 0.02)
                self.assertGreaterEqual(cov["marginal"], 0.9 - 0.02)


if __name__ == "__main__":
    unittest.main()
