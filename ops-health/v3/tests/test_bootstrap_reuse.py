"""Bootstrap reuse (prereg s.8; v2's paired percentile bootstrap). SYNTHETIC.

The stream layout is checked on DEV-sized hand-made inputs: no DEV row and no P or H statistic.
"""

from __future__ import annotations

import random
import unittest

from v2.research import bootstrap as v2b
from v2.research import metrics as v2_metrics
from v3.research import bootstrap, registry


class BootstrapReuseTest(unittest.TestCase):
    def test_registered_constants(self):
        self.assertEqual((bootstrap.SEED, bootstrap.RESAMPLES, bootstrap.LEVEL), (20261014, 2000, 0.95))
        self.assertEqual((bootstrap.TEST_ROWS, bootstrap.SHIFT_ROWS, bootstrap.H3_PAIRS, bootstrap.LEGACY_ROWS),
                         (100000, 25000, 100000, 240))
        self.assertIs(bootstrap.Resampler, v2b.Resampler)
        self.assertIs(bootstrap.interval, v2b.interval)
        self.assertEqual(bootstrap.fragile_seeds(), [20261014 + i for i in range(1, 101)])

    def test_count_vectors_equal_v2s_stream(self):
        mine = list(bootstrap.resampler("test", sizes=[342], resamples=25))
        theirs = list(v2b.Resampler(342, seed=20261014, resamples=25))
        self.assertEqual(mine, theirs)
        for vector in mine:
            self.assertEqual(sum(vector), 342)

    def test_draw_rule_by_hand(self):
        vector = next(iter(bootstrap.resampler("legacy_v1_test", sizes=[7], resamples=1)))
        rng = random.Random(20261014)
        counts = [0] * 7
        for _ in range(7):
            counts[int(rng.random() * 7)] += 1
        self.assertEqual(vector, counts)

    def test_pooled_strata_order_and_sizes(self):
        sizes = (5, 7, 3, 4)
        vectors = list(bootstrap.resampler("pooled_shift", sizes=sizes, resamples=30))
        strata = [list(s) for s in bootstrap.pooled_strata(sizes)]
        self.assertEqual(vectors, list(v2b.Resampler(19, seed=20261014, resamples=30, strata=strata)))
        for vector in vectors:
            per = bootstrap.regime_counts(vector, sizes)
            self.assertEqual(list(per), list(registry.POOLED_SHIFT_SUITE))
            self.assertEqual([sum(v) for v in per.values()], list(sizes))

    def test_one_fresh_stream_per_evaluation_set(self):
        a = list(bootstrap.resampler("test", sizes=[11], resamples=5))
        b = list(bootstrap.resampler("h3_twins", sizes=[11], resamples=5))
        c = list(bootstrap.resampler("legacy_v1_test", sizes=[11], resamples=5))
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        other = list(bootstrap.resampler("test", sizes=[11], resamples=5, seed=bootstrap.fragile_seeds()[0]))
        self.assertNotEqual(a, other)

    def test_pairs_count_for_both_twins(self):
        for vector in bootstrap.resampler("h3_twins", sizes=[9], resamples=10):
            observed, latent = bootstrap.pair_counts(vector)
            self.assertEqual(observed, latent)
            self.assertEqual(observed, vector)
            self.assertIsNot(observed, latent)

    def test_weighted_stats_equal_metrics_with_unit_weights(self):
        labels = [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        decisions = [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        scores = [0.9, 0.1, 0.6, 0.8, 0.4, 0.2, 0.05, 0.55, 0.7, 0.3, 0.95, 0.45]
        ones = [1] * len(labels)
        self.assertEqual(bootstrap.BinaryStats(labels, decisions).balanced_accuracy(ones),
                         v2_metrics.balanced_accuracy(labels, decisions))
        stats = bootstrap.ScoreStats(labels, scores)
        self.assertAlmostEqual(stats.brier(ones), v2_metrics.brier(labels, scores), places=15)
        self.assertAlmostEqual(stats.ece(ones), v2_metrics.ece(labels, scores), places=15)
        self.assertAlmostEqual(stats.auroc(ones), v2_metrics.roc_auc(labels, scores), places=15)

    def test_interval_type7_and_undefined_rule(self):
        values = [None, 0.1, 0.4, None, 0.2, 0.3]
        out = bootstrap.interval(values)
        self.assertEqual((out["n_defined"], out["n_undefined"], out["method"]), (4, 2, "percentile_type7"))
        self.assertAlmostEqual(out["lower"], 0.1 + 0.075 * 0.1, places=15)
        self.assertAlmostEqual(out["upper"], 0.3 + 0.925 * 0.1, places=15)
        self.assertEqual(bootstrap.differences([1.0, None], [0.5, 0.5]), [0.5, None])


if __name__ == "__main__":
    unittest.main()
