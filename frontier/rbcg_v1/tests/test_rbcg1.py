#!/usr/bin/env python3
"""test_rbcg1.py -- unit tests for rbcg1.py (stdlib unittest). SYNTHETIC DATA ONLY.
Binomial/CP/Wilson and crypto tests are carried over from v0 unchanged (section 1 and 4 code is
unchanged); scene-level loss, betting, HB, LTT, monitor and audit tests are new in v1."""
import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import unittest
from fractions import Fraction
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rbcg1  # noqa: E402
import rbcg1 as rbcg  # noqa: E402  (alias so the carried-over v0 tests read unchanged)
from synth import synth_rows, crafted_units, BINDINGS, PROTOCOL_SHA  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
A6, D6 = 100000, 100000


def load_kat():
    with open(os.path.join(FIX, "kat_vectors.json"), encoding="utf-8") as fh:
        return json.load(fh)


def exact_pmf(n, p):
    return [comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n + 1)]


def ref_prpl(xs, delta):
    """Independent restatement of the WSR predictable plug-in (one-sided): explicit lists, no running sums."""
    mus = [(0.5 + sum(xs[:t])) / (t + 1) for t in range(1, len(xs) + 1)]          # mu_1..mu_n
    out = []
    for t in range(1, len(xs) + 1):
        sig2_prev = (0.25 + sum((xs[i] - mus[i]) ** 2 for i in range(t - 1))) / t   # sigma^2_{t-1}
        out.append(math.sqrt(2 * math.log(1 / delta) / (sig2_prev * t * math.log(1 + t))))
    return out


def ref_log_kmax(xs, m, delta, c):
    k, best = 1.0, 1.0
    for lam, x in zip(ref_prpl(xs, delta), xs):
        k *= 1 + min(lam, c / (1 - m)) * (m - x)
        best = max(best, k)
    return math.log(best)


class TestBinomial(unittest.TestCase):
    def test_cdf_sf_match_exact_rationals(self):
        worst = 0.0
        for n in range(0, 31):
            for pf in (Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(9, 10)):
                pmf = exact_pmf(n, pf)
                for k in range(-1, n + 2):
                    ec = float(sum(pmf[:k + 1])) if k >= 0 else 0.0
                    es = float(sum(pmf[max(k, 0):])) if k <= n else 0.0
                    worst = max(worst, abs(rbcg.binom_cdf(k, n, float(pf)) - ec),
                                abs(rbcg.binom_sf(k, n, float(pf)) - es))
        self.assertLess(worst, 1e-12)

    def test_tail_includes_k(self):
        # P[X <= 2 | 50, 0.1] = 0.1117..., P[X <= 1] = 0.0338...; the tail must include k itself
        pmf = exact_pmf(50, Fraction(1, 10))
        self.assertAlmostEqual(rbcg.binom_cdf(2, 50, 0.1), float(sum(pmf[:3])), places=13)
        self.assertGreater(rbcg.binom_cdf(2, 50, 0.1), 0.1)
        self.assertLess(rbcg.binom_cdf(1, 50, 0.1), 0.1)

    def test_large_n_tails_finite_and_ordered(self):
        a = rbcg.binom_cdf(150, 1900, 0.1)
        b = rbcg.binom_cdf(190, 1900, 0.1)
        self.assertTrue(0.0 < a < b < 1.0)
        self.assertAlmostEqual(rbcg.binom_cdf(150, 1900, 0.1) + rbcg.binom_sf(151, 1900, 0.1), 1.0, places=12)

    def test_edges_and_validation(self):
        self.assertEqual(rbcg.binom_cdf(0, 10, 0.0), 1.0)
        self.assertEqual(rbcg.binom_sf(1, 10, 0.0), 0.0)
        self.assertEqual(rbcg.binom_cdf(9, 10, 1.0), 0.0)
        self.assertEqual(rbcg.binom_sf(10, 10, 1.0), 1.0)
        for bad in ((1.0, 10, 0.1), (True, 10, 0.1), (1, 10.0, 0.1)):
            with self.assertRaises(TypeError):
                rbcg.binom_cdf(*bad)
        for badp in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                rbcg.binom_cdf(1, 10, badp)
        with self.assertRaises(TypeError):
            rbcg.binom_cdf(1, 10, "0.1")


class TestBounds(unittest.TestCase):
    def test_cp_defining_property_and_order(self):
        for n in (1, 5, 20, 60):
            for k in range(0, n + 1):
                for d in (0.05, 0.01):
                    L, U = rbcg.cp_lower(k, n, d), rbcg.cp_upper(k, n, d)
                    self.assertLessEqual(L, k / n + 1e-15)
                    self.assertGreaterEqual(U, k / n - 1e-15)
                    if k > 0:
                        self.assertAlmostEqual(rbcg.binom_sf(k, n, L), d, delta=1e-9)
                    else:
                        self.assertEqual(L, 0.0)
                    if k < n:
                        self.assertAlmostEqual(rbcg.binom_cdf(k, n, U), d, delta=1e-9)
                    else:
                        self.assertEqual(U, 1.0)

    def test_cp_lower_exact_coverage(self):
        n, d = 30, 0.05
        for pf in (Fraction(1, 20), Fraction(1, 5), Fraction(1, 2)):
            pmf = exact_pmf(n, pf)
            cover = sum(pmf[k] for k in range(n + 1) if rbcg.cp_lower(k, n, d) <= float(pf))
            self.assertGreaterEqual(float(cover), 1 - d - 1e-12)

    def test_wilson_endpoints_solve_score_equation(self):
        z = 1.959963984540054
        for n, k in ((50, 5), (200, 3), (17, 16)):
            for p in rbcg.wilson_interval(k, n):
                self.assertAlmostEqual((k / n - p) ** 2, z * z * p * (1 - p) / n, delta=1e-9)
        self.assertEqual(rbcg.wilson_interval(0, 40)[0], 0.0)
        self.assertEqual(rbcg.wilson_interval(40, 40)[1], 1.0)


def _const_rows(n, k, pmax=0.995, ood=False):
    rows = []
    for i in range(n):
        rows.append({"cid": f"r/{i}", "unit": f"u/{i // 24}", "inf": True, "ood": ood,
                     "pmax_e12": int(pmax * rbcg.E12), "pred": "opioid",
                     "label": "none" if i < k else "opioid"})
    return rows


class TestSceneLoss(unittest.TestCase):
    def test_loss_bounds_and_exactness(self):
        for n_max in (1, 7, 24):
            for a in range(n_max + 1):
                for e in range(a + 1):
                    num = rbcg1.loss_num(a, e, A6, n_max)
                    x = Fraction(num, n_max * 10 ** 6)
                    self.assertEqual(x, Fraction(e - Fraction(1, 10) * a, n_max) + Fraction(1, 10))
                    self.assertTrue(0 <= x <= 1)
        self.assertEqual(rbcg1.loss_num(24, 0, A6, 24), 0)
        self.assertEqual(rbcg1.loss_num(24, 24, A6, 24), 24 * 10 ** 6)
        for bad in ((3, 4, 24), (25, 0, 24), (-1, 0, 24)):
            with self.assertRaises(ValueError):
                rbcg1.loss_num(bad[0], bad[1], A6, bad[2])

    def test_mean_loss_sign_equals_risk_side(self):
        # E[L] >= 0 iff E[e]/E[a] >= alpha (with fixed n_max)
        for ea, ee in ((10, 1), (10, 2), (5, 0), (8, 3)):
            num = rbcg1.loss_num(ea, ee, A6, 24) - A6 * 24       # 24e6 * L
            self.assertEqual(num >= 0, Fraction(ee, ea) >= Fraction(1, 10))

    def test_unit_counts_match_direct_definition(self):
        rows, _ = synth_rows(11, 30)
        units = rbcg1.group_units(rows)
        grid = rbcg1.GRID_C_DEFAULT
        fast = rbcg1.unit_counts_by_tau(units, grid, 81)
        for ti, t in enumerate(grid):
            for ui, (_, ur) in enumerate(units):
                a = sum(1 for r in ur if rbcg1.accepts(r, t, 81))
                e = sum(1 for r in ur if rbcg1.accepts(r, t, 81) and r["pred"] != r["label"])
                self.assertEqual(fast[ti][ui], (a, e))

    def test_ood_floor(self):
        r = {"inf": True, "ood": True, "pmax_e12": 80 * 10 ** 10}
        self.assertFalse(rbcg1.accepts(r, 70, 81))
        self.assertTrue(rbcg1.accepts(dict(r, ood=False), 70, 81))
        self.assertTrue(rbcg1.accepts(dict(r, pmax_e12=81 * 10 ** 10), 70, 81))
        self.assertFalse(rbcg1.accepts(dict(r, inf=False, ood=False), 30, 81))

    def test_calibrate_refuses_oversized_units(self):
        rows, _ = synth_rows(1, 3, unit_size=25)
        with self.assertRaises(ValueError):
            rbcg1.calibrate(rows, A6, D6)


class TestBetting(unittest.TestCase):
    def test_prpl_matches_independent_restatement(self):
        rng = random.Random(5)
        xs = [rng.random() * 0.3 for _ in range(60)]
        for d in (0.1, 0.05):
            for a, b in zip(rbcg1.prpl_lambdas(xs, d), ref_prpl(xs, d)):
                self.assertAlmostEqual(a, b, places=10)
            self.assertAlmostEqual(rbcg1.bet_log_kmax(xs, 0.1, d, 0.75), ref_log_kmax(xs, 0.1, d, 0.75), places=10)

    def test_hand_example(self):
        # X = (0, 0), m = 0.1, c = 3/4: both bets are capped at c/(1-m) = 5/6, each factor is 1 + (5/6)(0.1).
        p = rbcg1.bet_pvalue([0.0, 0.0], 0.1, 0.1, 0.75)
        self.assertAlmostEqual(p, 1.0 / (1.0 + 0.75 / 0.9 * 0.1) ** 2, places=12)
        # a scene at the worst bound X = 1 multiplies the capital by 1 - c exactly when capped
        self.assertAlmostEqual(rbcg1.bet_log_kmax([1.0], 0.1, 0.1, 0.75), 0.0, places=15)
        lam1 = min(rbcg1.prpl_lambdas([1.0, 0.0], 0.1)[0], 0.75 / 0.9)
        self.assertAlmostEqual(lam1, 0.75 / 0.9, places=15)

    def test_direction(self):
        rng = random.Random(9)
        low = [0.04 + 0.02 * rng.random() for _ in range(300)]          # E[X] ~ 0.05 < alpha
        high = [0.15 + 0.1 * rng.random() for _ in range(300)]          # E[X] ~ 0.20 > alpha
        self.assertLess(rbcg1.bet_pvalue(low, 0.1, 0.1, 0.75), 1e-3)
        self.assertEqual(rbcg1.bet_pvalue(high, 0.1, 0.1, 0.75), 1.0)
        self.assertEqual(rbcg1.bet_pvalue([], 0.1, 0.1, 0.75), 1.0)

    def test_capital_stays_positive_at_bound(self):
        xs = [1.0] * 40
        k = rbcg1.bet_log_kmax(xs, 0.1, 0.1, 0.75)
        self.assertTrue(math.isfinite(k))
        self.assertEqual(k, 0.0)

    def test_validity_monte_carlo_at_boundary(self):
        # X in {0, 0.2} with mean exactly alpha = 0.1, and X = scaled Bernoulli(0.1): type-I error <= delta
        rng = random.Random(2026)
        for gen in (lambda: 0.2 * (rng.random() < 0.5), lambda: float(rng.random() < 0.1)):
            rej = sum(rbcg1.bet_pvalue([gen() for _ in range(120)], 0.1, 0.1, 0.75) <= 0.1 for _ in range(1500))
            self.assertLessEqual(rej / 1500, 0.10 + 3 * math.sqrt(0.09 / 1500), rej)


class TestHB(unittest.TestCase):
    def test_matches_direct_formula(self):
        den = 24 * 10 ** 6
        rng = random.Random(1)
        for _ in range(20):
            n = rng.randrange(1, 60)
            nums = [rbcg1.loss_num(a, rng.randrange(a + 1), A6, 24) for a in (rng.randrange(25) for _ in range(n))]
            mu = Fraction(sum(nums), n * den)
            a_ = min(float(mu), 0.1)
            h1 = (-math.log(0.9) if a_ == 0 else a_ * math.log(a_ / 0.1) + (1 - a_) * math.log((1 - a_) / 0.9))
            kc = math.ceil(mu * n)
            bent = math.e * sum(comb(n, j) * 0.1 ** j * 0.9 ** (n - j) for j in range(0, min(kc, n) + 1))
            self.assertAlmostEqual(rbcg1.hb_pvalue(nums, den, A6), min(1.0, math.exp(-n * h1), bent), places=9)

    def test_exact_ceiling(self):
        den = 24 * 10 ** 6
        nums = [den // 2] * 5                       # n * mu = 2.5 -> ceil 3
        b3 = math.e * rbcg1.binom_cdf(3, 5, 0.1)
        self.assertAlmostEqual(rbcg1.hb_pvalue(nums, den, A6), min(1.0, b3), places=12)
        self.assertEqual(rbcg1.hb_pvalue([], den, A6), 1.0)

    def test_bentkus_term_binds_with_exact_ceiling(self):
        # n = 50 scenes, n * mu = 2.5 (mu = 0.05 < alpha): Hoeffding exp(-50 h1(0.05, 0.1)) ~ 0.434 and
        # Bentkus e * P(Bin(50, 0.1) <= ceil(2.5) = 3) ~ 0.680, so HB = Hoeffding; with a floor (2) the
        # Bentkus term would be ~0.304 and would wrongly bind.
        den = 24 * 10 ** 6
        nums = [den // 2] * 5 + [0] * 45
        h = math.exp(-50 * (0.05 * math.log(0.5) + 0.95 * math.log(0.95 / 0.9)))
        b = math.e * sum(comb(50, j) * 0.1 ** j * 0.9 ** (50 - j) for j in range(4))
        self.assertLess(h, b)
        self.assertAlmostEqual(rbcg1.hb_pvalue(nums, den, A6), h, places=12)
        nums2 = [0] * 47 + [den] * 1 + [den // 4] * 2    # n * mu = 1.5 -> ceil 2
        b2 = math.e * sum(comb(50, j) * 0.1 ** j * 0.9 ** (50 - j) for j in range(3))
        h2 = math.exp(-50 * (0.03 * math.log(0.3) + 0.97 * math.log(0.97 / 0.9)))
        self.assertAlmostEqual(rbcg1.hb_pvalue(nums2, den, A6), min(h2, b2), places=12)

    def test_hb_is_conservative_here(self):
        # with range-1 scene losses and 80 scenes HB cannot certify the low-variance regime
        rows, _ = synth_rows(3, 120)
        cal = rbcg1.calibrate(rows, A6, D6)
        self.assertIsNone(cal["secondary_tau_star_c"])
        self.assertIsNotNone(cal["tau_star_c"])


class TestLTT(unittest.TestCase):
    def setUp(self):
        self.rows, _ = synth_rows(3, 120)

    def test_table_matches_direct_definition(self):
        cal = rbcg1.calibrate(self.rows, A6, D6)
        units = rbcg1.group_units(self.rows)
        for e in cal["table"]:
            t = e["tau_c"]
            cs = [(sum(1 for r in u if rbcg1.accepts(r, t, 81)),
                   sum(1 for r in u if rbcg1.accepts(r, t, 81) and r["pred"] != r["label"])) for _, u in units]
            nums = [rbcg1.loss_num(a, k, A6, 24) for a, k in cs]
            self.assertEqual(e["n_acc"], sum(a for a, _ in cs))
            self.assertEqual(e["k_err"], sum(k for _, k in cs))
            self.assertEqual(e["sum_x_num"], sum(nums))
            lk = ref_log_kmax([v / 24e6 for v in nums], 0.1, 0.1, 0.75)
            self.assertLessEqual(abs(e["logk_e9"] - lk * 1e9), 2)
            self.assertGreaterEqual(e["p_bet_e9"] / 1e9, min(1.0, math.exp(-lk)) - 1e-9)

    def test_fixed_sequence_stops_at_first_failure(self):
        tab = [{"tau_c": 99, "p_bet_e9": 1}, {"tau_c": 98, "p_bet_e9": 200000000}, {"tau_c": 97, "p_bet_e9": 1}]
        self.assertEqual(rbcg1.ltt_fixed_sequence(tab, D6), (99, 1))

    def test_fixed_sequence_rejects_at_equality(self):
        tab = [{"tau_c": 99, "p_bet_e9": 100000000}, {"tau_c": 98, "p_bet_e9": 100000001}]
        self.assertEqual(rbcg1.ltt_fixed_sequence(tab, D6), (99, 1))

    def test_tau_star_is_last_rejection(self):
        cal = rbcg1.calibrate(self.rows, A6, D6)
        t = cal["table"]
        n = cal["n_rejected"]
        self.assertGreater(n, 0)
        self.assertTrue(all(e["p_bet_e9"] <= D6 * 1000 for e in t[:n]))
        if n < len(t):
            self.assertGreater(t[n]["p_bet_e9"], D6 * 1000)
        self.assertEqual(cal["tau_star_c"], t[n - 1]["tau_c"])

    def test_high_risk_data_certifies_nothing(self):
        rows, _ = synth_rows(4, 120, err_scale=3.0, err_add=0.1)
        self.assertIsNone(rbcg1.calibrate(rows, A6, D6)["tau_star_c"])

    def test_ceil_e9_is_conservative(self):
        for p in (0.1, 1e-12, 0.333333333333, 1.0):
            self.assertGreaterEqual(rbcg1.ceil_e9(p) / 1e9, p)


class TestMonitor(unittest.TestCase):
    def test_smoothed_p_hand_examples(self):
        self.assertAlmostEqual(rbcg1.smoothed_p([1, 2, 3], 2, 0.5), (1 + 0.5 * 2) / 4)
        self.assertAlmostEqual(rbcg1.smoothed_p([], 7, 0.25), 0.25)
        self.assertAlmostEqual(rbcg1.smoothed_p([5, 5, 5], 5, 1.0), 1.0)

    def test_theta_range_and_determinism(self):
        for j in range(1, 50):
            t = rbcg1.theta("ab" * 32, j)
            self.assertTrue(0 < t <= 1)
            self.assertEqual(t, rbcg1.theta("ab" * 32, j))
        self.assertNotEqual(rbcg1.theta_seed_for(rbcg1.THETA_DOMAIN_PRIMARY, "a" * 64),
                            rbcg1.theta_seed_for(rbcg1.THETA_DOMAIN_SECONDARY, "a" * 64))
        self.assertNotEqual(rbcg1.theta_seed_for(rbcg1.THETA_DOMAIN_PRIMARY, "a" * 64),
                            rbcg1.theta_seed_for(rbcg1.THETA_DOMAIN_PRIMARY, "b" * 64))

    def test_one_step_martingale_value(self):
        m = rbcg1.CTM([0, 1, 2, 3], (500,), 10000, "c" * 64)
        s = m.step(10)
        p = s["p"]
        self.assertAlmostEqual(s["log_m"], math.log(0.5 * p ** -0.5), places=12)

    def test_thresholds_are_one_over_delta(self):
        self.assertAlmostEqual(rbcg1.log_alarm_threshold(10000), math.log(100.0))
        self.assertAlmostEqual(rbcg1.log_alarm_threshold(rbcg1.DELTA_S_PRIMARY_E6_DEFAULT), math.log(1e6 / 7500))
        self.assertAlmostEqual(rbcg1.log_alarm_threshold(rbcg1.DELTA_S_SECONDARY_E6_DEFAULT), math.log(400.0))
        self.assertEqual(rbcg1.DELTA_S_PRIMARY_E6_DEFAULT + rbcg1.DELTA_S_SECONDARY_E6_DEFAULT, 10000)

    def test_scores(self):
        rows = [{"ood": True, "pmax_e12": 9 * 10 ** 11}, {"ood": False, "pmax_e12": 10 ** 12},
                {"ood": True, "pmax_e12": 5 * 10 ** 11}]
        self.assertEqual(rbcg1.unit_score_ood_count(rows), 2)
        self.assertEqual(rbcg1.unit_score_conf(rows), (10 ** 11 + 0 + 5 * 10 ** 11) // 3)

    def _monitor(self, cal_units):
        return rbcg1.TwoScoreMonitor(cal_units, rbcg1.monitor_spec("a" * 64, len(cal_units)))

    def test_primary_detects_ood_shift_and_secondary_confidence_shift(self):
        cal, _ = synth_rows(21, 200, ood_unit_rate=0.3)
        cu = rbcg1.group_units(cal)
        dep, _ = synth_rows(22, 30, prefix="d", ood_unit_rate=0.95, with_label=False)
        m = self._monitor(cu)
        steps = [m.step(u) for _, u in rbcg1.group_units(dep)]
        self.assertTrue(any(s["primary"]["alarm"] for s in steps))
        conf, _ = synth_rows(23, 40, prefix="c", ood_unit_rate=0.3, b_a=2.0, b_b=4.0, with_label=False)
        m2 = self._monitor(cu)
        steps2 = [m2.step(u) for _, u in rbcg1.group_units(conf)]
        self.assertTrue(steps2[-1]["secondary"]["alarm"])
        self.assertTrue(steps2[-1]["alarm"])

    def test_alarm_latches(self):
        cu = [(f"u{i}", [{"ood": False, "pmax_e12": 10 ** 12}] * 3) for i in range(50)]
        m = self._monitor(cu)
        hot = [{"ood": True, "pmax_e12": 10 ** 11}] * 3
        cold = [{"ood": False, "pmax_e12": 10 ** 12}] * 3
        seq = [m.step(hot)["alarm"] for _ in range(12)] + [m.step(cold)["alarm"] for _ in range(30)]
        self.assertTrue(seq[-1])

    def test_false_alarm_rate_under_exchangeability(self):
        rng = random.Random(77)
        alarms = 0
        n_sims = 1500
        for s in range(n_sims):
            spec = rbcg1.monitor_spec(hashlib.sha256(b"%d" % s).hexdigest(), 30)
            mk = lambda: [{"ood": rng.random() < 0.3, "pmax_e12": rng.randrange(10 ** 12)} for _ in range(4)]
            cu = [(f"c{i}", mk()) for i in range(30)]
            m = rbcg1.TwoScoreMonitor(cu, spec)
            alarms += any(m.step(mk())["alarm"] for _ in range(30))
        self.assertLessEqual(alarms / n_sims, 0.01 + 3 * math.sqrt(0.01 * 0.99 / n_sims), alarms)

    def test_group_units(self):
        rows = [{"unit": "a"}, {"unit": "a"}, {"unit": "b"}]
        self.assertEqual([u for u, _ in rbcg1.group_units(rows)], ["a", "b"])
        with self.assertRaises(ValueError):
            rbcg1.group_units(rows + [{"unit": "a"}])


class TestAudit(unittest.TestCase):
    def test_hand_example_and_threshold(self):
        a = rbcg1.BettingAudit(A6, 50000, 750)
        self.assertAlmostEqual(a.log_thr, math.log(20.0), places=12)
        lam = math.sqrt(2 * math.log(20) / (0.25 * 1 * math.log(2)))
        a.add(0.5)
        self.assertAlmostEqual(a.logk, math.log1p(min(lam, 7.5) * 0.4), places=12)
        self.assertIsNone(a.violated_at)          # K = 1 + 4.0 * 0.4 < 20

    def test_lower_bound_matches_independent_grid(self):
        rng = random.Random(3)
        a = rbcg1.BettingAudit(A6, 50000, 750)
        for _ in range(60):
            a.add(0.12 + 0.06 * rng.random())
        lb = a.lower_bound()
        grid = [i / 20000 for i in range(1, 20001)]

        def logk(m):
            return sum(math.log1p(min(l, 0.75 / m) * (x - m)) for l, x in zip(a.lams, a.xs))
        inside = [m for m in grid if logk(m) < math.log(20)]
        self.assertLessEqual(abs(lb - inside[0]), 1 / 20000 + 1e-9)
        self.assertLessEqual(lb, inside[0] + 1e-12)
        self.assertEqual(a.violated_at is not None, any(
            sum(math.log1p(min(l, 7.5) * (x - 0.1)) for l, x in zip(a.lams[:t], a.xs[:t])) >= math.log(20)
            for t in range(1, 61)))

    def test_decision_consistent_with_lower_bound(self):
        a = rbcg1.BettingAudit(A6, 50000, 750)
        for _ in range(80):
            a.add(0.2)
        self.assertIsNotNone(a.violated_at)
        self.assertGreater(a.lower_bound(), 0.1)
        b = rbcg1.BettingAudit(A6, 50000, 750)
        for _ in range(80):
            b.add(0.09)
        self.assertIsNone(b.violated_at)
        self.assertLess(b.lower_bound(), 0.1)

    def test_false_flag_rate_under_null(self):
        rng = random.Random(4)
        flags, n_sims = 0, 1500
        for _ in range(n_sims):
            a = rbcg1.BettingAudit(A6, 50000, 750)
            for _ in range(100):
                if a.add(0.2 * (rng.random() < 0.5)):
                    break
            flags += a.violated_at is not None
        self.assertLessEqual(flags / n_sims, 0.05 + 3 * math.sqrt(0.05 * 0.95 / n_sims), flags)

    def test_scene_loss_requires_complete_labels(self):
        rows, labels = crafted_units("d", 1, 0.95, 3)
        x, a, e = rbcg1.scene_loss(rows, labels, 90, 81, A6, 24)
        self.assertEqual((a, e), (24, 3))
        self.assertAlmostEqual(x, (3 - 2.4) / 24 + 0.1, places=12)
        del labels[rows[5]["cid"]]
        self.assertIsNone(rbcg1.scene_loss(rows, labels, 90, 81, A6, 24))


class TestCanonicalAndPAE(unittest.TestCase):
    def test_canonical_json(self):
        self.assertEqual(rbcg.canonical_json({"b": 1, "a": [1, 2], "\u00e9": "\u00fc"}),
                         '{"a":[1,2],"b":1,"\u00e9":"\u00fc"}'.encode("utf-8"))
        with self.assertRaises(ValueError):
            rbcg.canonical_json({"x": float("nan")})

    def test_pae_dsse_spec_vector(self):
        k = load_kat()["dsse_protocol"]
        self.assertEqual(rbcg.pae(k["payload_type"], k["body"].encode("utf-8")), k["pae"].encode("utf-8"))

    def test_pae_counts_utf8_bytes(self):
        self.assertEqual(rbcg.pae("\u00fc", "\u00e9".encode("utf-8")),
                         b"DSSEv1 2 \xc3\xbc 2 \xc3\xa9")


class TestECDSA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kat = load_kat()

    def test_openssl_curve_parameters(self):
        with open(os.path.join(FIX, "openssl_p256_explicit.txt"), encoding="utf-8") as fh:
            txt = fh.read()

        def field(name):
            m = re.search(re.escape(name) + r"\s*\n((?:\s+[0-9a-f:]+\n)+)", txt)
            return int("".join(m.group(1).split()).replace(":", ""), 16)
        self.assertEqual(field("Prime:"), rbcg.P256_P)
        self.assertEqual(field("A:"), rbcg.P256_A)
        self.assertEqual(field("B:"), rbcg.P256_B)
        self.assertEqual(field("Order:"), rbcg.P256_N)
        g = field("Generator (uncompressed):")
        self.assertEqual(g, (4 << 512) | (rbcg.P256_GX << 256) | rbcg.P256_GY)

    def test_rfc6979_a25_public_key(self):
        k = self.kat["rfc6979_a25"]
        self.assertEqual(int(k["q"], 16), rbcg.P256_N)
        sk = rbcg.SigningKey(int(k["x"], 16))
        self.assertEqual(sk.public.point, (int(k["Ux"], 16), int(k["Uy"], 16)))

    def test_rfc6979_a25_signatures(self):
        k = self.kat["rfc6979_a25"]
        d = int(k["x"], 16)
        for msg in ("sample", "test"):
            v = k["sha256"][msg]
            h1 = hashlib.sha256(msg.encode()).digest()
            self.assertEqual(next(rbcg.rfc6979_nonces(d, h1)), int(v["k"], 16), msg)
            self.assertEqual(rbcg.ecdsa_sign_digest(d, h1), (int(v["r"], 16), int(v["s"], 16)), msg)
            self.assertTrue(rbcg.ecdsa_verify_digest(rbcg.SigningKey(d).public.point, h1,
                                                     int(v["r"], 16), int(v["s"], 16)))

    def test_dsse_spec_signature_vector(self):
        k = self.kat["dsse_protocol"]
        sk = rbcg.SigningKey(int(k["d"]))
        self.assertEqual(sk.public.point, (int(k["X"]), int(k["Y"])))
        raw = base64.b64decode(k["sig_b64"])
        self.assertEqual(len(raw), 64)
        msg = rbcg.pae(k["payload_type"], k["body"].encode("utf-8"))
        r, s = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
        self.assertEqual(sk.sign_rs(msg), (r, s))
        self.assertTrue(sk.public.verify(msg, rbcg.der_encode_sig(r, s)))

    def test_distinct_messages_distinct_nonces(self):
        sk = rbcg.SigningKey.generate()
        r1, _ = sk.sign_rs(b"message one")
        r2, _ = sk.sign_rs(b"message two")
        self.assertNotEqual(r1, r2)

    def test_verify_rejects_forgeries_and_out_of_range(self):
        sk, other = rbcg.SigningKey.generate(), rbcg.SigningKey.generate()
        msg = b"rbcg test message"
        r, s = sk.sign_rs(msg)
        pub = sk.public
        self.assertTrue(pub.verify(msg, rbcg.der_encode_sig(r, s)))
        self.assertFalse(pub.verify(msg + b"!", rbcg.der_encode_sig(r, s)))
        self.assertFalse(other.public.verify(msg, rbcg.der_encode_sig(r, s)))
        self.assertFalse(pub.verify(msg, rbcg.der_encode_sig(r, (s + 1) % rbcg.P256_N)))
        self.assertFalse(pub.verify(msg, rbcg.der_encode_sig(r, s + rbcg.P256_N)))
        self.assertFalse(pub.verify(msg, rbcg.der_encode_sig(r + rbcg.P256_N, s)))
        h = hashlib.sha256(msg).digest()
        self.assertFalse(rbcg.ecdsa_verify_digest(pub.point, h, 0, s))
        self.assertFalse(rbcg.ecdsa_verify_digest(pub.point, h, r, 0))

    def test_der_strictness(self):
        good = rbcg.der_encode_sig(5, 2 ** 255 + 7)
        self.assertEqual(rbcg.der_decode_sig(good), (5, 2 ** 255 + 7))
        bad = [b"\x30\x07\x02\x02\x00\x05\x02\x01\x07",          # non-minimal INTEGER
               b"\x30\x06\x02\x01\x85\x02\x01\x07",              # negative INTEGER
               good + b"\x00",                                   # trailing byte
               b"\x30\x81\x06\x02\x01\x05\x02\x01\x07"]          # long-form length
        for b in bad:
            with self.assertRaises(ValueError):
                rbcg.der_decode_sig(b)

    def test_spki_pem_roundtrip_and_curve_check(self):
        sk = rbcg.SigningKey.generate()
        pem = sk.public.pem()
        self.assertTrue(pem.startswith("-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE"))
        self.assertEqual(rbcg.PublicKey.from_pem(pem).point, sk.public.point)
        self.assertEqual(rbcg.PublicKey.from_sec1_hex(sk.public.sec1_hex()).point, sk.public.point)
        self.assertEqual(sk.public.keyid(), hashlib.sha256(pem.strip().encode()).hexdigest())
        x, y = sk.public.point
        with self.assertRaises(ValueError):
            rbcg.PublicKey(x, (y + 1) % rbcg.P256_P)
        self.assertNotIn(str(sk._d), repr(sk))

    def test_group_law(self):
        self.assertIsNone(rbcg._scalar_mult(rbcg.P256_N, rbcg.G))
        self.assertEqual(rbcg._scalar_mult(rbcg.P256_N - 1, rbcg.G), (rbcg.P256_GX, rbcg.P256_P - rbcg.P256_GY))
        a, b = 0x1234567890ABCDEF, 0xFEDCBA0987654321
        A, B = rbcg._scalar_mult(a, rbcg.G), rbcg._scalar_mult(b, rbcg.G)
        self.assertEqual(rbcg._to_affine(rbcg._jadd((A[0], A[1], 1), (B[0], B[1], 1))),
                         rbcg._scalar_mult(a + b, rbcg.G))


class TestReceipts(unittest.TestCase):
    def test_issue_and_chain(self):
        rows, _ = synth_rows(3, 120)
        key, lkey = rbcg1.SigningKey.generate(), rbcg1.SigningKey.generate()
        env, info = rbcg1.issue_guarantee(rows, key=key, bindings=BINDINGS, seed_family="synth-cal",
                                          env_profile="synthetic", log_id=lkey.public.keyid(),
                                          protocol_sha256=PROTOCOL_SHA, issued_at_utc="2026-11-03T00:00:00Z")
        g = info["payload"]
        self.assertEqual(g["procedure"]["tau_star_c"], rbcg1.calibrate(rows, A6, D6)["tau_star_c"])
        self.assertEqual(g["calibration"]["rows_sha256"], rbcg1.rows_sha256(rows))
        self.assertEqual(g["monitor"]["primary"]["theta_seed"],
                         rbcg1.theta_seed_for(rbcg1.THETA_DOMAIN_PRIMARY, rbcg1.rows_sha256(rows)))
        self.assertEqual(g["bindings"]["rbcg_code_sha256"], rbcg1.rbcg_code_sha256())
        self.assertEqual(g["protocol_sha256"], PROTOCOL_SHA)
        self.assertEqual(g["anchoring"]["log_id"], lkey.public.keyid())
        self.assertIn("not for clinical use", g["claim"])
        body = base64.b64decode(env["payload"])
        self.assertEqual(body, rbcg1.canonical_json(g))
        self.assertTrue(key.public.verify(rbcg1.pae(rbcg1.PT_GUARANTEE, body),
                                          base64.b64decode(env["signatures"][0]["sig"])))
        dep, _ = synth_rows(4, 3, prefix="dep", with_label=False)
        ch = rbcg1.MonitorChain(env, rows, key)
        e1, _ = ch.add_unit("dep/U0000", dep[:24], "src")
        e2, _ = ch.add_unit("dep/U0001", dep[24:48], "src")
        p1 = json.loads(base64.b64decode(e1["payload"]))
        p2 = json.loads(base64.b64decode(e2["payload"]))
        self.assertEqual(p1["prev_sha256"], info["digest"])
        self.assertEqual(p2["prev_sha256"], rbcg1.envelope_digest(e1))
        self.assertEqual((p1["seq"], p2["seq"]), (1, 2))
        self.assertEqual(set(p1["primary"]), {"score", "p_e12", "log_m_e9", "alarm"})

    def test_no_guarantee_fails_closed(self):
        rows, _ = synth_rows(4, 120, err_scale=3.0, err_add=0.1)
        with self.assertRaises(rbcg1.NoGuarantee):
            rbcg1.issue_guarantee(rows, key=rbcg1.SigningKey.generate(), bindings=BINDINGS, seed_family="s",
                                  env_profile="e", log_id="0" * 64, protocol_sha256=PROTOCOL_SHA)


if __name__ == "__main__":
    unittest.main()
