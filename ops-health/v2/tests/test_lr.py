"""M6: Newton/IRLS logistic regression (section 6) and v1's gradient-descent reproduction (s.9)."""

from __future__ import annotations

import math
import unittest

from v2.research import lr, pipeline, truth, v1_gd
from v2.research.dataio import labels_of
from v2.tests import _support as S


class IrlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = list(S.dev_rows())[S.TRAIN]
        cls.columns = lr.columns_from_vectors(pipeline.normalized_vectors(rows))
        cls.y = labels_of(rows)
        cls.fits = {l2: lr.fit(cls.columns, cls.y, l2) for l2 in (0.0, 1e-4, 1e-3, 1e-2)}

    def test_converges_with_zero_gradient(self):
        for l2, fit in self.fits.items():
            with self.subTest(l2=l2):
                self.assertTrue(fit["converged"])
                self.assertEqual(fit["stop_reason"], "CONVERGED")
                self.assertLessEqual(fit["iterations"], lr.MAX_ITER)
                self.assertLess(fit["gradient_max_abs"], 1e-9)
                self.assertLess(fit["last_newton_step_max_abs"], lr.TOL)

    def test_analytic_gradient_matches_finite_differences(self):
        theta = [0.3, -0.7, 0.2, -0.4, 1.1, 0.9, 0.5, -1.2, -0.8]
        l2, h = 1e-2, 1e-6
        n = len(self.y)

        def obj(t):
            z = lr.linear_predictor(t[0], t[1:], self.columns, n)
            return lr.objective(z, self.y, t[1:], l2)

        z = lr.linear_predictor(theta[0], theta[1:], self.columns, n)
        grad = lr.gradient(z, self.y, theta[1:], self.columns, l2)
        for j in range(len(theta)):
            up = list(theta)
            dn = list(theta)
            up[j] += h
            dn[j] -= h
            with self.subTest(coordinate=j):
                self.assertAlmostEqual(grad[j], (obj(up) - obj(dn)) / (2 * h), places=6)

    def test_optimum_is_a_minimum(self):
        fit = self.fits[1e-3]
        n = len(self.y)
        theta = [fit["intercept"]] + list(fit["weights"])

        def obj(t):
            return lr.objective(lr.linear_predictor(t[0], t[1:], self.columns, n), self.y, t[1:], 1e-3)

        best = obj(theta)
        self.assertAlmostEqual(best, fit["objective"], places=14)
        for j in range(len(theta)):
            for step in (1e-3, -1e-3):
                moved = list(theta)
                moved[j] += step
                self.assertGreater(obj(moved), best)

    def test_intercept_is_unpenalized(self):
        fit = lr.fit(self.columns, self.y, 1e3)
        prevalence = sum(self.y) / len(self.y)
        self.assertLess(max(abs(w) for w in fit["weights"]), 1e-2)
        self.assertAlmostEqual(fit["intercept"], math.log(prevalence / (1 - prevalence)), places=2)

    def test_deterministic(self):
        again = lr.fit(self.columns, self.y, 1e-3)
        self.assertEqual(again["intercept"], self.fits[1e-3]["intercept"])
        self.assertEqual(again["weights"], self.fits[1e-3]["weights"])

    def test_cap_is_reported(self):
        fit = lr.fit(self.columns, self.y, 1e-3, max_iter=2)
        self.assertFalse(fit["converged"])
        self.assertEqual(fit["stop_reason"], "CAP_REACHED")
        self.assertEqual(fit["iterations"], 2)

    def test_step_halving_on_separable_data(self):
        # Separable 1-D toy without penalty: the objective keeps decreasing, the fit must stay
        # finite and report either the cap or a line-search stop, never a false convergence.
        fit = lr.fit([[0.0, 0.1, 0.2, 0.8, 0.9, 1.0]], [0, 0, 0, 1, 1, 1], 0.0, max_iter=30)
        self.assertFalse(fit["converged"])
        self.assertIn(fit["stop_reason"], ("CAP_REACHED", "LINE_SEARCH_FAILED"))
        self.assertTrue(all(math.isfinite(v) for v in [fit["intercept"], *fit["weights"]]))

    def test_invalid_inputs(self):
        for bad in (-1.0, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(l2=bad), self.assertRaises(ValueError):
                lr.fit(self.columns, self.y, bad)
        with self.assertRaises(ValueError):
            lr.fit([[0.1, 0.2]], [0, 2], 0.0)

    def test_solvers_agree(self):
        a = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
        b = [1.0, -2.0, 0.5]
        x1 = lr.cholesky_solve(a, b)
        x2 = lr.gauss_solve(a, b)
        for u, v in zip(x1, x2):
            self.assertAlmostEqual(u, v, places=12)
        self.assertIsNone(lr.cholesky_solve([[1.0, 2.0], [2.0, 1.0]], [1.0, 1.0]))

    def test_dev_fit_recovers_truth_signs(self):
        # SYNTHETIC sanity: the model class is correctly specified (section 2), so a 4,096-row DEV
        # fit recovers every sign.
        fit = self.fits[1e-3]
        self.assertEqual(truth.sign_agreement(fit["weights"])["k"], 8)


class V1GradientDescentTest(unittest.TestCase):
    def test_vectorized_equals_literal_transcription(self):
        x, y = v1_gd.legacy_train_xy()
        fast = v1_gd.train_v1_gd(x, y, epochs=25)
        slow = v1_gd.train_v1_gd_reference(x, y, epochs=25)
        self.assertEqual(fast, slow)  # bitwise

    def test_reproduces_published_v1_weights_to_12_decimals(self):
        report = v1_gd.reproduce()
        self.assertEqual(report["train_rows"], 768)
        for name, entry in report["parameters"].items():
            with self.subTest(parameter=name):
                self.assertTrue(entry["equal_at_12dp"], entry)
        self.assertTrue(report["all_equal_at_12dp"])
        self.assertLess(report["max_abs_diff_unrounded_vs_published"], 1e-12)


if __name__ == "__main__":
    unittest.main()
