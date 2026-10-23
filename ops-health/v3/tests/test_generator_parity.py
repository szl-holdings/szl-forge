"""Generator v3 parity and mode tests (prereg s.3.1-s.3.3). SYNTHETIC; DEV data only.

v1 parity regenerates v1's public rows (Random(2500)); v2 parity regenerates v2's four non-sealed
splits only to compare digests (rows are hashed as drawn, never kept); every other row here is a
DEV row (split name 'dev') inside the DEV cap.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
import types
import unittest
from unittest import mock

from v2.research import generator as v2gen
from v3.research import generator as gen
from v3.research import guards, registry
from v3.tests import _support as S

SECONDS = "seconds_since_last_success"


def _jsonl(rows) -> bytes:
    return gen.jsonl_bytes(rows)


class V1ParityTest(unittest.TestCase):
    def test_v1_parity_byte_for_byte(self):
        splits = gen.generate_v1_parity()
        source = json.loads(guards.read_bytes_guarded(registry.V1_LEGACY_DIR / "SOURCE.json").decode("utf-8"))
        self.assertEqual(list(splits), ["train", "validation", "test"])
        for split, rows in splits.items():
            with self.subTest(split=split):
                produced = _jsonl(rows)
                legacy = guards.read_bytes_guarded(registry.V1_LEGACY_DIR / f"{split}.jsonl")
                self.assertEqual(len(rows), {"train": 768, "validation": 192, "test": 240}[split])
                self.assertEqual(produced, legacy)
                self.assertEqual(hashlib.sha256(produced).hexdigest(),
                                 source["files"][f"{split}.jsonl"]["sha256"])


class V2ParityTest(unittest.TestCase):
    def test_v2_non_sealed_digests(self):
        manifest = json.loads(guards.read_bytes_guarded(registry.V2_MANIFEST_PATH).decode("utf-8"))
        for split in registry.V2_PARITY_SPLITS:
            with self.subTest(split=split):
                entry = manifest["splits"][split]
                self.assertIs(entry["sealed"], False)
                result = gen.v2_parity_digest(split)
                self.assertEqual(set(result), {"split", "rows", "sha256", "bytes"})
                self.assertEqual(result["sha256"], entry["sha256"])
                self.assertEqual(result["bytes"], entry["bytes"])
                self.assertEqual(result["rows"], entry["rows"])

    def test_parity_mode_refuses_every_other_split(self):
        for split in registry.V2_SEALED_SPLITS + ("dev", "gates", "train_noD"):
            with self.subTest(split=split), self.assertRaises(gen.GeneratorRefused):
                gen.v2_parity_digest(split)

    def test_reused_code_is_v2s_code(self):
        for name in ("normalize", "truth_linear", "truth_probability", "_weighted_failure_count",
                     "draw_row"):
            with self.subTest(function=name):
                v2_src = inspect.getsource(getattr(v2gen, name))
                v3_src = inspect.getsource(getattr(gen, name))
                strip = lambda s: [l for l in s.splitlines() if not l.strip().startswith('"""')]  # noqa: E731
                self.assertEqual(strip(v3_src)[1:], strip(v2_src)[1:])
        self.assertEqual(gen.FEATURE_SPECS, v2gen.FEATURE_SPECS)
        self.assertEqual(gen.REGIMES, v2gen.REGIMES)
        self.assertEqual(gen.ROW_SCHEMA, v2gen.ROW_SCHEMA)
        self.assertEqual(gen.V1_ROW_SCHEMA, v2gen.V1_ROW_SCHEMA)
        self.assertEqual(gen.V1_SEED, v2gen.V1_SEED)

    def test_v2_generator_source_is_the_registered_one(self):
        digest = hashlib.sha256(guards.read_bytes_guarded(registry.V2 / "research" / "generator.py")).hexdigest()
        self.assertEqual(digest, "a502d3644534ec599663d8678b8066b390e612396b25350f33fa987194b44384")


class SeedTest(unittest.TestCase):
    def test_seed_derivation_formula(self):
        for split in ("train", "dev", "gates", "train_band_001", "train_kg_20"):
            material = f"oac-ops-health-v3:20261013:{split}".encode()
            self.assertEqual(gen.derive_seed(split), int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
        self.assertEqual(registry.SEED_NAMESPACE, "oac-ops-health-v3")
        self.assertEqual(registry.MASTER_SEED, 20261013)

    def test_sealed_splits_have_no_unsalted_seed(self):
        for split in registry.SEALED_SPLITS:
            with self.subTest(split=split), self.assertRaises(gen.GeneratorRefused):
                gen.derive_seed(split)
        with self.assertRaises(gen.GeneratorRefused):
            gen.derive_seed("train", master_seed=20261014)

    def test_sealed_seed_material_template(self):
        salt = "0f" * 32
        for split in registry.SEALED_SPLITS:
            self.assertEqual(gen.sealed_seed_material(split, salt), f"oac-ops-health-v3:20261013:{split}:{salt}")
        for bad in ("0F" * 32, "0f" * 31, "0f" * 33, None, 7, "zz" * 32):
            with self.subTest(salt=bad), self.assertRaises(gen.GeneratorRefused):
                gen.sealed_seed_material("test", bad)
        with self.assertRaises(gen.GeneratorRefused):
            gen.sealed_seed_material("train", salt)


class TwinParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # v2's own generator draws the same DEV stream (v2 namespace, split 'dev'); its rows and
        # latent rows count against this DEV run's cap like every other DEV row.
        guards.DEV_LEDGER.record(namespace="v2-module", master_seed=registry.V2_MASTER_SEED,
                                 regime="clock_skew", mode="v2-module-with-latent",
                                 draws=S.DEV_TWIN, rows_per_draw=2)

    def test_twin_observed_equals_v2_clock_skew_bytes(self):
        observed, _latent = gen.dev_rows_twin(S.DEV_TWIN, namespace=registry.V2_SEED_NAMESPACE)
        reference = v2gen.jsonl_bytes(v2gen.generate_rows("dev", S.DEV_TWIN, "clock_skew"))
        self.assertEqual(_jsonl(observed), reference)

    def test_twin_latent_equals_v2_latent_draws(self):
        _observed, latent = gen.dev_rows_twin(S.DEV_TWIN, namespace=registry.V2_SEED_NAMESPACE)
        triples = v2gen.generate_with_latent("dev", S.DEV_TWIN, "clock_skew")
        self.assertEqual([r["features"] for r in latent], [t[1] for t in triples])
        self.assertEqual([r["label"]["operator_attention_required"] for r in latent], [t[2] for t in triples])

    def test_twin_observed_equals_plain_clock_skew_on_v3_dev_seed(self):
        observed, latent = S.twin_rows()
        self.assertEqual(_jsonl(observed), _jsonl(S.shift_rows("clock_skew")))
        self.assertEqual(len(latent), S.DEV_TWIN)

    def test_latent_twin_differs_only_in_seconds(self):
        observed, latent = S.twin_rows()
        moved = 0
        for index, (o, l) in enumerate(zip(observed, latent, strict=True)):
            self.assertEqual(o["sample_id"], f"dev-{index:06d}")
            self.assertEqual(l["sample_id"], o["sample_id"])
            self.assertEqual(o["label"], l["label"])
            self.assertEqual(o["regime"], "clock_skew")
            self.assertEqual(l["regime"], "clock_skew")
            for name in gen.FEATURE_NAMES:
                if name != SECONDS:
                    self.assertEqual(o["features"][name], l["features"][name])
            self.assertGreaterEqual(o["features"][SECONDS], l["features"][SECONDS])
            self.assertLessEqual(o["features"][SECONDS], min(86400.0, l["features"][SECONDS] + 7200.0) + 1e-6)
            moved += o["features"][SECONDS] > l["features"][SECONDS]
            # neither the skew value nor any latent field is written: the rows carry exactly the
            # v2 keys and the eight fields
            for row in (o, l):
                self.assertEqual(set(row), {"schema", "synthetic", "sample_id", "regime", "features", "label"})
                self.assertEqual(set(row["features"]), set(gen.FEATURE_NAMES))
                self.assertEqual(set(row["label"]), {"operator_attention_required"})
        self.assertGreater(moved, S.DEV_TWIN - 10)

    def test_twin_label_comes_from_latent_features(self):
        observed, latent = S.twin_rows()
        rng = random.Random(gen.derive_seed("dev"))
        for o, l in zip(observed[:32], latent[:32], strict=True):
            obs, lat, label = gen.draw_row(rng, "clock_skew")
            self.assertEqual((obs, lat, label), (o["features"], l["features"], o["label"]["operator_attention_required"]))


class DModeTest(unittest.TestCase):
    def test_arms_identical_except_seconds(self):
        d_arm, nod_arm = S.d_rows()
        self.assertEqual(len(d_arm), S.DEV_D)
        differ = 0
        for a, b in zip(d_arm, nod_arm, strict=True):
            self.assertEqual(a["sample_id"], b["sample_id"])
            self.assertEqual(a["regime"], "nominal")
            for name in gen.FEATURE_NAMES:
                if name != SECONDS:
                    self.assertEqual(a["features"][name], b["features"][name])
            if a["features"][SECONDS] == b["features"][SECONDS]:
                self.assertEqual(a["label"], b["label"])
            else:
                differ += 1
        self.assertGreater(differ, 0)
        self.assertLess(differ, S.DEV_D)

    def test_no_d_seconds_equal_s_nom_draws(self):
        draws = gen.dev_draws_d(S.DEV_D)
        d_arm, nod_arm = S.d_rows()
        wide = 0
        for draw, a, b in zip(draws, d_arm, nod_arm, strict=True):
            self.assertEqual(b["features"][SECONDS], draw["s_nom"])
            expected = draw["s_wide"] if draw["b"] < 0.25 else draw["s_nom"]
            self.assertEqual(a["features"][SECONDS], expected)
            self.assertEqual(draw["D"][0], a["features"])
            self.assertEqual(draw["noD"][0], b["features"])
            self.assertTrue(0.0 <= draw["s_wide"] <= 86400.0)
            wide += draw["b"] < 0.25
        self.assertGreater(wide, 0)

    def test_d_mode_draw_order_by_hand(self):
        rng = random.Random(gen.derive_seed("dev"))
        listener = int(rng.random() < 0.94)
        tls = int(rng.random() < 0.90)
        peer = int(rng.random() < 0.93)
        queue = round(min(1.0, rng.betavariate(1.25, 4.5)), 6)
        failures = gen._weighted_failure_count(rng)
        s_nom = round(min(86400.0, rng.expovariate(1.0 / 900.0)), 6)
        b = rng.random()
        s_wide = round(rng.uniform(0.0, 86400.0), 6)
        ledger_ok = int(rng.random() < 0.985)
        config_ok = int(rng.random() < 0.96)
        u = rng.random()
        features = {"listener_running": listener, "tls_enabled": tls, "peer_allowlist_configured": peer,
                    "queue_utilization": queue, "consecutive_failures": failures,
                    SECONDS: s_wide if b < 0.25 else s_nom, "ledger_integrity_ok": ledger_ok,
                    "configuration_valid": config_ok}
        d_arm, nod_arm = S.d_rows()
        self.assertEqual(d_arm[0]["features"], features)
        self.assertEqual(d_arm[0]["label"]["operator_attention_required"], u < gen.truth_probability(features))
        self.assertEqual(nod_arm[0]["features"], dict(features, **{SECONDS: s_nom}))

    def test_paired_registered_splits_share_one_stream(self):
        self.assertEqual(gen.split_definition("train_noD")["stream"], "train")
        self.assertEqual(gen.split_definition("train_kg_07_noD")["stream"], "train_kg_07")
        self.assertEqual(gen.split_definition("h3_clock_skew_latent")["stream"], "h3_clock_skew")
        with self.assertRaises(gen.GeneratorRefused):
            list(gen._stream(1, "x", 1, "clock_skew", "D"))
        with self.assertRaises(gen.GeneratorRefused):
            list(gen._stream(1, "x", 1, "nominal", "twin_observed"))


class RowFormatTest(unittest.TestCase):
    def test_row_schema_and_regimes(self):
        sets = {"nominal": S.nominal_rows(), "clock_skew": S.shift_rows("clock_skew")}
        for regime in ("queue_saturation", "failure_bursts", "missing_tls"):
            sets[regime] = S.shift_rows(regime)
        for regime, rows in sets.items():
            for index, row in enumerate(rows[:200]):
                self.assertEqual(set(row), {"schema", "synthetic", "sample_id", "regime", "features", "label"})
                self.assertEqual(row["schema"], "szl-oac/ops-health-observation/v2")
                self.assertIs(row["synthetic"], True)
                self.assertEqual(row["sample_id"], f"dev-{index:06d}")
                self.assertEqual(row["regime"], regime)
                self.assertEqual(set(row["features"]), set(gen.FEATURE_NAMES))
                for name, minimum, maximum, kind in gen.FEATURE_SPECS:
                    value = row["features"][name]
                    self.assertTrue(minimum <= value <= maximum)
                    if kind == "binary":
                        self.assertIn(value, (0, 1))

    def test_shift_regimes_move_their_feature(self):
        nominal = S.nominal_rows()
        mean = lambda rows, key: sum(r["features"][key] for r in rows) / len(rows)  # noqa: E731
        self.assertGreater(mean(S.shift_rows("queue_saturation"), "queue_utilization"),
                           mean(nominal, "queue_utilization") + 0.4)
        self.assertGreater(mean(S.shift_rows("failure_bursts"), "consecutive_failures"),
                           mean(nominal, "consecutive_failures") + 2.0)
        self.assertLess(mean(S.shift_rows("missing_tls"), "tls_enabled"), 0.40)

    def test_determinism_and_canonical_writer(self):
        a = _jsonl(gen.dev_rows(200, "failure_bursts"))
        b = _jsonl(S.shift_rows("failure_bursts")[:200])
        self.assertEqual(a, b)
        self.assertTrue(a.endswith(b"\n"))
        self.assertNotIn(b"\r", a)
        first = a.split(b"\n", 1)[0].decode()
        self.assertEqual(first, json.dumps(json.loads(first), sort_keys=True, separators=(",", ":")))

    def test_truth_rule_is_the_registered_one(self):
        from v3.research import representation as rep

        for row in S.nominal_rows()[:200]:
            x = gen.normalize(row["features"])
            z = rep.TRUE_INTERCEPT_V1 + sum(rep.TRUE_WEIGHTS_V1[n] * v for n, v in zip(gen.FEATURE_NAMES, x))
            self.assertAlmostEqual(z, gen.truth_linear(row["features"]), places=9)


class SplitTableTest(unittest.TestCase):
    def test_registered_split_table(self):
        t = registry.SPLIT_TABLE
        expect = {"train": ("nominal", "D", 32768, False), "train_noD": ("nominal", "noD", 32768, False),
                  "calibration": ("nominal", "plain", 4096, False), "conformal": ("nominal", "plain", 8192, False),
                  "validation": ("nominal", "plain", 8192, False), "gates": ("nominal", "plain", 8192, False),
                  "test": ("nominal", "plain", 100000, True),
                  "h3_clock_skew": ("clock_skew", "twin_observed", 100000, True),
                  "h3_clock_skew_latent": ("clock_skew", "twin_latent", 100000, True)}
        for name, (regime, mode, rows, sealed) in expect.items():
            self.assertEqual(t[name][:4], (regime, mode, rows, sealed), name)
        for name in registry.POOLED_SHIFT_SUITE:
            self.assertEqual(t[name][2:4], (25000, True))
        self.assertEqual(sum(1 for n in t if n.startswith("train_band_")), 100)
        self.assertEqual(sum(1 for n in t if n.startswith("train_kg_") and not n.endswith("_noD")), 20)
        self.assertEqual(sum(1 for n in t if n.startswith("train_kg_") and n.endswith("_noD")), 20)
        self.assertEqual(sum(t[n][2] for n in registry.SEALED_SPLITS), 400000)
        self.assertEqual(sum(t[n][2] for n in registry.STORED_SPLITS), 94208)
        self.assertEqual(len(registry.RETRAIN_SPLITS), 140)

    def test_non_sealed_paths_refuse_sealed_splits(self):
        for name in registry.SEALED_SPLITS:
            with self.subTest(split=name):
                with self.assertRaises(gen.GeneratorRefused):
                    gen.generate_split(name)
                with self.assertRaises(gen.GeneratorRefused):
                    gen.split_sha256(name)


class _FakeVerifiedRound:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class SealedModeTest(unittest.TestCase):
    """Sealed generation accepts SALT only as the frozen verifier's VerifiedRound for the
    registered chain and round R'. No accepting input is ever constructed here, and the row
    streams are patched to fail if reached, so no sealed-name row can be drawn."""

    def setUp(self):
        guard = AssertionError("no sealed row may be drawn in a test")
        self._patches = [mock.patch.object(gen, "_stream", side_effect=guard),
                         mock.patch.object(gen, "_paired_stream", side_effect=guard)]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        sys.modules.pop("v3.research.beacon", None)

    def test_refuses_without_a_verified_round(self):
        salt = "ab" * 32
        for bad in (None, salt, {"randomness": salt, "round": registry.BEACON_ROUND},
                    _FakeVerifiedRound(chain_hash=registry.CHAIN_HASH, round=registry.BEACON_ROUND,
                                       randomness=salt, signature="00", verifier_sha256="0" * 64)):
            for split in ("test", "h3_clock_skew", "shift_clock_skew"):
                with self.subTest(value=type(bad).__name__, split=split), self.assertRaises(gen.GeneratorRefused):
                    gen.generate_sealed(split, bad)

    def test_latent_twin_is_never_generated_alone(self):
        with self.assertRaises(gen.GeneratorRefused):
            gen.generate_sealed("h3_clock_skew_latent", None)
        with self.assertRaises(gen.GeneratorRefused):
            gen.generate_sealed("train", None)

    def test_wrong_chain_round_or_relation_is_refused(self):
        fake = types.ModuleType("v3.research.beacon")

        class VerifiedRound:  # stands in for the frozen verifier's output type
            def __init__(self, **fields):
                self.__dict__.update(fields)

        fake.VerifiedRound = VerifiedRound
        fake.source_sha256 = lambda: "1" * 64
        sys.modules["v3.research.beacon"] = fake
        signature = "8f" * 48
        good_salt = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        cases = {
            "first registered round": dict(round=registry.FIRST_REGISTERED_ROUND),
            "neighbour round": dict(round=registry.BEACON_ROUND + 1),
            "bool round": dict(round=True),
            "other chain": dict(chain_hash="0" * 64),
            "randomness not sha256(signature)": dict(randomness="cd" * 32),
            "upper-case randomness": dict(randomness=good_salt.upper()),
            "odd signature": dict(signature="abc"),
            "other verifier": dict(verifier_sha256="2" * 64),
        }
        base = dict(chain_hash=registry.CHAIN_HASH, round=registry.BEACON_ROUND, randomness=good_salt,
                    signature=signature, verifier_sha256="1" * 64)
        for label, change in cases.items():
            with self.subTest(case=label), self.assertRaises(gen.GeneratorRefused):
                gen.generate_sealed("test", VerifiedRound(**dict(base, **change)))
        # a subclass is not the verifier's type, even with every field right
        sub = type("Sub", (VerifiedRound,), {})
        with self.assertRaises(gen.GeneratorRefused):
            gen.generate_sealed("test", sub(**base))
        # control: the same fields in the verifier's own type pass every check and reach the row
        # stream, which is patched to raise, so no sealed-name row is drawn (and the fake SALT is
        # not the randomness of any round)
        for split in ("test", "h3_clock_skew"):
            with self.subTest(control=split), self.assertRaises(AssertionError) as ctx:
                gen.generate_sealed(split, VerifiedRound(**base))
            self.assertIn("no sealed row may be drawn", str(ctx.exception))

    def test_registered_round_is_r_prime(self):
        self.assertEqual(registry.BEACON_ROUND, 6659606)
        self.assertEqual(registry.BEACON_ROUND_TIME, 1795219200)  # 2026-11-21T00:00:00Z
        self.assertEqual(registry.CHAIN_GENESIS_TIME + (registry.FIRST_REGISTERED_ROUND - 1) * 30,
                         registry.FIRST_REGISTERED_ROUND_TIME)


if __name__ == "__main__":
    unittest.main()
