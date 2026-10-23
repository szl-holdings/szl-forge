"""Guard tests (prereg s.3.2 item 5, s.3.3 "every refusal in item 5 fires", s.15.5 DEV rule).

No row of any v2 sealed split is generated, and nothing under v2/data/sealed/ is opened: every
refusal here fires before a seed is derived or a file is read.
"""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v3.research import generator as gen
from v3.research import guards, pipeline, registry, tolerances
from v3.tests import _support as S

V2NS = registry.V2_SEED_NAMESPACE
V2M = registry.V2_MASTER_SEED


class V2SealGuardTest(unittest.TestCase):
    VARIANTS = ("test", "shift_queue_saturation", "shift_failure_bursts", "shift_clock_skew",
                "shift_missing_tls", " test", "TEST", "Test ", "te​st", "ｔｅｓｔ",
                "SHIFT-clock-skew", "shift clock skew", "Shift_Missing_TLS", "shift_anything")

    def test_v2_namespace_with_any_sealed_name_is_refused(self):
        for split in self.VARIANTS:
            with self.subTest(split=ascii(split)):
                with self.assertRaises(guards.V2SealRefused):
                    guards.check_namespace(V2NS, split)
                with self.assertRaises(guards.V2SealRefused):
                    gen.derive_seed(split, namespace=V2NS, master_seed=V2M)
                with self.assertRaises(guards.V2SealRefused):
                    gen.derive_seed(split, namespace=V2NS, master_seed=12345)

    def test_dev_entry_points_refuse_v2_sealed_names(self):
        # the DEV helpers always use split 'dev'; the parity helper refuses sealed names itself
        for split in registry.V2_SEALED_SPLITS:
            with self.subTest(split=split), self.assertRaises(gen.GeneratorRefused):
                gen.v2_parity_digest(split)

    def test_v2_namespace_spelling_variants_are_refused(self):
        for namespace in ("OAC-OPS-HEALTH-V2", " oac-ops-health-v2", "oac_ops_health_v2"):
            with self.subTest(namespace=namespace):
                with self.assertRaises(guards.V2SealRefused):
                    guards.check_namespace(namespace, "train")
                with self.assertRaises(guards.V2SealRefused):
                    guards.check_namespace(namespace, "test")

    def test_unregistered_namespaces_are_refused(self):
        for namespace in ("oac-ops-health-v1", "oac-ops-health-v4", "", "OAC-OPS-HEALTH-V3"):
            with self.subTest(namespace=namespace), self.assertRaises(guards.V2SealRefused):
                gen.derive_seed("train", namespace=namespace)

    def test_v2_non_sealed_names_pass_the_guard(self):
        for split in registry.V2_PARITY_SPLITS + ("dev",):
            self.assertEqual(guards.check_namespace(V2NS, split), V2NS)
            gen.derive_seed(split, namespace=V2NS, master_seed=V2M)
        with self.assertRaises(gen.GeneratorRefused):
            gen.derive_seed("train", namespace=V2NS, master_seed=registry.MASTER_SEED)


class SealedPathGuardTest(unittest.TestCase):
    def assertRefused(self, path):
        self.assertTrue(guards.is_sealed_path(path), path)
        with self.assertRaises(guards.SealedPathRefused):
            guards.refuse_sealed_path(path)
        with self.assertRaises(guards.SealedPathRefused):
            guards.read_bytes_guarded(path)

    def test_paths_under_v2_sealed_are_refused_before_reading(self):
        sealed = registry.V2_SEALED_DIR
        for path in (sealed, sealed / "test.jsonl", sealed / "shift_clock_skew.jsonl",
                     sealed / "nested" / "x.bin", registry.V2_DATA_DIR / "legacy_v1" / ".." / "sealed" / "t.jsonl",
                     str(sealed) + os.sep + "test.jsonl"):
            with self.subTest(path=str(path)):
                self.assertRefused(path)

    def test_relative_and_case_variants_are_refused(self):
        cwd = os.getcwd()
        try:
            os.chdir(registry.ROOT)
            self.assertRefused(Path("v2/data/sealed/test.jsonl"))
            self.assertRefused("v2/data/./sealed/test.jsonl")
        finally:
            os.chdir(cwd)
        self.assertRefused(registry.V2_DATA_DIR / "SEALED" / "test.jsonl")
        self.assertRefused(registry.V2_DATA_DIR / "Sealed" / "test.jsonl")

    def test_any_sealed_component_is_refused(self):
        with S.TempDir() as tmp:
            self.assertRefused(tmp / "sealed" / "rows.jsonl")
            self.assertRefused(tmp / "a" / "SEALED")
            self.assertFalse(guards.is_sealed_path(tmp / "unsealed" / "rows.jsonl"))
            self.assertFalse(guards.is_sealed_path(tmp / "rows_sealed.jsonl"))

    def test_symlink_into_sealed_is_refused_when_links_exist(self):
        with S.TempDir() as tmp:
            target = tmp / "sealed_target_dir"
            link = tmp / "innocent"
            target.mkdir()
            with mock.patch.object(registry, "V2_SEALED_DIR", target):
                try:
                    os.symlink(target, link, target_is_directory=True)
                except (OSError, NotImplementedError):
                    # the host does not allow symbolic links: the resolved-path rule is still
                    # exercised by the '..' case above; record the host fact in the test output
                    self.assertTrue(guards.is_sealed_path(target / "x"))
                    return
                self.assertTrue(guards.is_sealed_path(link / "test.jsonl"))

    def test_v3_writers_refuse_sealed_paths(self):
        with self.assertRaises(guards.SealedPathRefused):
            gen.write_jsonl(registry.V2_SEALED_DIR / "x.jsonl", [])
        self.assertFalse(registry.V2_SEALED_DIR.joinpath("x.jsonl").exists())

    def test_only_guards_and_registry_name_the_v2_sealed_dir(self):
        offenders = []
        for path in sorted((registry.V3 / "research").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "V2_SEALED_DIR":
                    if path.name not in ("guards.py", "registry.py"):
                        offenders.append(f"{path.name}:{node.lineno}")
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "data/sealed" in node.value:
                    if path.name not in ("guards.py", "registry.py"):
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])


class DevRuleTest(unittest.TestCase):
    def test_dev_split_name_only(self):
        guards.check_dev_split("dev")
        for name in ("train", "Dev", "dev2", "validation", None):
            with self.subTest(name=name), self.assertRaises(guards.DevRuleError):
                guards.check_dev_split(name)

    def test_ledger_cap_per_regime(self):
        ledger = guards.DevLedger()
        ledger.record(namespace="n", master_seed=1, regime="nominal", mode="plain", draws=1024, rows_per_draw=1)
        ledger.record(namespace="n", master_seed=1, regime="nominal", mode="plain", draws=1024, rows_per_draw=1)
        ledger.record(namespace="n", master_seed=1, regime="nominal", mode="D+noD", draws=512, rows_per_draw=2)
        self.assertEqual(ledger.total("nominal"), 2048)
        with self.assertRaises(guards.DevRuleError):
            ledger.record(namespace="n", master_seed=1, regime="nominal", mode="plain", draws=1025, rows_per_draw=1)
        with self.assertRaises(guards.DevRuleError):
            ledger.record(namespace="m", master_seed=1, regime="nominal", mode="plain", draws=1, rows_per_draw=1)
        ledger.record(namespace="n", master_seed=1, regime="clock_skew", mode="twin", draws=1024, rows_per_draw=2)
        with self.assertRaises(guards.DevRuleError):
            ledger.record(namespace="n", master_seed=1, regime="clock_skew", mode="plain", draws=1, rows_per_draw=1)
        self.assertEqual(ledger.snapshot()["rows_per_regime"], {"clock_skew": 2048, "nominal": 2048})

    def test_generator_dev_helpers_respect_the_cap(self):
        fresh = guards.DevLedger()
        with mock.patch.object(guards, "DEV_LEDGER", fresh):
            with self.assertRaises(guards.DevRuleError):
                gen.dev_rows(2049, "missing_tls")
            with self.assertRaises(guards.DevRuleError):
                gen.dev_rows_d(1025)
            with self.assertRaises(guards.DevRuleError):
                gen.dev_rows_twin(1025)

    def test_this_runs_ledger_is_within_the_cap(self):
        S.nominal_rows(), S.d_rows(), S.twin_rows()
        snapshot = S.ledger_snapshot()
        for regime, rows in snapshot["rows_per_regime"].items():
            self.assertLessEqual(rows, registry.DEV_MAX_ROWS_PER_REGIME, regime)

    def test_no_fit_on_more_than_2048_dev_rows(self):
        fake = [{"sample_id": f"dev-{i:06d}", "features": {}, "label": {"operator_attention_required": False}}
                for i in range(2049)]
        with self.assertRaises(guards.DevRuleError):
            guards.check_dev_fit(fake)
        self.assertEqual(guards.check_dev_fit(fake[:2048]), 2048)
        with self.assertRaises(guards.DevRuleError):
            guards.check_dev_fit(fake[:10] + [{"sample_id": "train-000000"}])
        with self.assertRaises(guards.DevRuleError):
            pipeline.build_candidate(train_rows=fake, threshold_rows=[], conformal_rows=[],
                                     split_names={"train": "dev", "threshold": "dev", "conformal": "dev"})

    def test_no_tolerance_on_dev_data(self):
        band = [{"train_split": "dev", "theta_r": [0.0] * 9, "threshold_index": 20, "ba_gates": 0.7}] * 100
        with self.assertRaises(guards.DevRuleError):
            tolerances.compute(band)
        with self.assertRaises(guards.DevRuleError):
            guards.refuse_dev_statistic(["train_band_001", "DEV"], "H1")


class RuntimePinTest(unittest.TestCase):
    def test_status_reports_every_pin(self):
        status = guards.runtime_pin_status()
        self.assertEqual(set(status), {"implementation", "version", "version_pinned", "isolated",
                                       "pythonhashseed", "pythonhashseed_pinned", "ignore_environment",
                                       "hash_randomization", "pinned"})
        self.assertEqual(registry.PINNED_PYTHON, (3, 12, 10))
        self.assertEqual(status["pinned"], status["version_pinned"] and status["isolated"]
                         and status["pythonhashseed_pinned"])

    def test_isolated_mode_ignores_pythonhashseed(self):
        """MEASURED fact behind AR-4: under -I the interpreter ignores PYTHONHASHSEED, so the
        effective string hash seed is random; the pin check is the registered text item by item."""
        import subprocess
        import sys

        code = "import sys; print(sys.flags.isolated, sys.flags.ignore_environment, sys.flags.hash_randomization)"
        env = dict(os.environ, PYTHONHASHSEED="0")
        isolated = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                                  env=env, timeout=120, check=True).stdout.split()
        self.assertEqual(isolated, ["1", "1", "1"])
        child = ("import json, sys; sys.path.insert(0, sys.argv[1]); from v3.research import guards; "
                 "print(json.dumps(guards.runtime_pin_status(), sort_keys=True))")
        status = __import__("json").loads(subprocess.run(
            [sys.executable, "-I", "-c", child, str(registry.ROOT)], capture_output=True, text=True,
            env=env, timeout=120, check=True).stdout)
        self.assertTrue(status["isolated"])
        self.assertTrue(status["pythonhashseed_pinned"])
        self.assertEqual(status["hash_randomization"], 1)
        self.assertEqual(status["pinned"], status["version_pinned"])

    def test_registered_computations_refuse_without_the_pins(self):
        with mock.patch.object(guards, "runtime_pin_status", return_value={"pinned": False}):
            with self.assertRaises(guards.RuntimePinError):
                guards.require_runtime_pins()


if __name__ == "__main__":
    unittest.main()
