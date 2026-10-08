"""M2: generator parity with v1 and v2 generator invariants (SYNTHETIC data only)."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from pathlib import Path

from v2.research import generator as gen

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "v2" / "data" / "legacy_v1"


class V1ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = gen.generate_v1_parity()
        cls.source = json.loads((LEGACY / "SOURCE.json").read_text("utf-8"))

    def test_v1_parity_byte_for_byte(self):
        for split, rows in self.splits.items():
            with self.subTest(split=split):
                produced = gen.jsonl_bytes(rows)
                legacy = (LEGACY / f"{split}.jsonl").read_bytes()
                self.assertEqual(len(rows), {"train": 768, "validation": 192, "test": 240}[split])
                self.assertEqual(produced, legacy)
                self.assertEqual(
                    hashlib.sha256(produced).hexdigest(),
                    self.source["files"][f"{split}.jsonl"]["sha256"],
                )

    def test_v1_split_order_and_sizes(self):
        self.assertEqual(list(self.splits), ["train", "validation", "test"])


class V2GeneratorTest(unittest.TestCase):
    def test_seed_derivation_formula(self):
        for split in ("train", "dev", "shift_clock_skew"):
            material = f"oac-ops-health-v2:20260926:{split}".encode()
            expected = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            self.assertEqual(gen.derive_seed(split), expected)
        self.assertNotEqual(gen.derive_seed("train"), gen.derive_seed("calibration"))

    def test_row_schema_and_regimes(self):
        for regime in gen.REGIMES:
            rows = gen.generate_rows("dev", 50, regime)
            for index, row in enumerate(rows):
                self.assertEqual(
                    set(row), {"schema", "synthetic", "sample_id", "regime", "features", "label"}
                )
                self.assertEqual(row["schema"], "szl-oac/ops-health-observation/v2")
                self.assertIs(row["synthetic"], True)
                self.assertEqual(row["sample_id"], f"dev-{index:06d}")
                self.assertEqual(row["regime"], regime)
                self.assertEqual(set(row["features"]), set(gen.FEATURE_NAMES))
                self.assertEqual(set(row["label"]), {"operator_attention_required"})
                for name, minimum, maximum, kind in gen.FEATURE_SPECS:
                    value = row["features"][name]
                    self.assertGreaterEqual(value, minimum)
                    self.assertLessEqual(value, maximum)
                    if kind == "binary":
                        self.assertIn(value, (0, 1))

    def test_nominal_draws_match_v1_draw_order(self):
        # One nominal row from a fresh Random(2500) equals legacy train-000000.
        observed, latent, label = gen.draw_row(random.Random(2500), "nominal")
        first = json.loads((LEGACY / "train.jsonl").read_bytes().split(b"\n", 1)[0])
        self.assertEqual(observed, first["features"])
        self.assertEqual(observed, latent)
        self.assertEqual(label, first["label"]["operator_attention_required"])

    def test_clock_skew_hides_latent_and_only_moves_forward(self):
        triples = gen.generate_with_latent("dev", 400, "clock_skew")
        rows = gen.generate_rows("dev", 400, "clock_skew")
        moved = 0
        for (observed, latent, label), row in zip(triples, rows, strict=True):
            self.assertEqual(row["features"], observed)
            self.assertEqual(row["label"]["operator_attention_required"], label)
            for name in gen.FEATURE_NAMES:
                if name != "seconds_since_last_success":
                    self.assertEqual(observed[name], latent[name])
            obs_s = observed["seconds_since_last_success"]
            lat_s = latent["seconds_since_last_success"]
            self.assertGreaterEqual(obs_s, lat_s)
            self.assertLessEqual(obs_s, min(86400.0, lat_s + 7200.0) + 1e-6)
            moved += obs_s > lat_s
            # The latent value never appears in the serialized row.
            self.assertNotIn("latent", json.dumps(row))
        self.assertGreater(moved, 390)

    def test_label_uses_latent_features_under_clock_skew(self):
        # Replaying the draw order by hand: 8 nominal features, skew, then the label draw.
        rng_a = random.Random(gen.derive_seed("dev"))
        observed, latent, label = gen.draw_row(rng_a, "clock_skew")
        rng_b = random.Random(gen.derive_seed("dev"))
        latent_b, _, _ = gen.draw_row(rng_b, "nominal")  # consumes features + label draw
        self.assertEqual(latent, latent_b)
        rng_c = random.Random(gen.derive_seed("dev"))
        for _ in range(3):
            rng_c.random()
        rng_c.betavariate(1.25, 4.5)
        # consecutive failures: replicate the weighted draw's consumption
        gen._weighted_failure_count(rng_c)
        rng_c.expovariate(1.0 / 900.0)
        rng_c.random()
        rng_c.random()
        skew = rng_c.uniform(0.0, 7200.0)
        expected_label = rng_c.random() < gen.truth_probability(latent)
        self.assertEqual(label, expected_label)
        self.assertEqual(
            observed["seconds_since_last_success"],
            round(min(86400.0, latent["seconds_since_last_success"] + skew), 6),
        )

    def test_shift_regimes_move_their_feature(self):
        n = 3000
        nominal = gen.generate_rows("dev", n, "nominal")
        mean = lambda rows, key: sum(r["features"][key] for r in rows) / len(rows)  # noqa: E731
        sat = gen.generate_rows("dev", n, "queue_saturation")
        self.assertGreater(mean(sat, "queue_utilization"), mean(nominal, "queue_utilization") + 0.4)
        bursts = gen.generate_rows("dev", n, "failure_bursts")
        self.assertGreater(
            mean(bursts, "consecutive_failures"), mean(nominal, "consecutive_failures") + 2.0
        )
        tls = gen.generate_rows("dev", n, "missing_tls")
        self.assertLess(mean(tls, "tls_enabled"), 0.40)
        self.assertGreater(mean(nominal, "tls_enabled"), 0.85)

    def test_truth_rule_matches_preregistered_parameters(self):
        from v2.research import truth

        rng = random.Random(gen.derive_seed("dev"))
        for _ in range(200):
            observed, latent, _ = gen.draw_row(rng, "nominal")
            x = gen.normalize(latent)
            z = truth.TRUE_INTERCEPT + sum(
                w * v for w, v in zip(truth.TRUE_WEIGHT_VECTOR, x, strict=True)
            )
            self.assertAlmostEqual(z, gen.truth_linear(latent), places=9)

    def test_determinism_and_canonical_writer(self):
        a = gen.jsonl_bytes(gen.generate_rows("dev", 200, "failure_bursts"))
        b = gen.jsonl_bytes(gen.generate_rows("dev", 200, "failure_bursts"))
        self.assertEqual(a, b)
        self.assertTrue(a.endswith(b"\n"))
        self.assertNotIn(b"\r", a)
        self.assertNotIn(b", ", a)
        first = a.split(b"\n", 1)[0].decode()
        self.assertEqual(first, json.dumps(json.loads(first), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
