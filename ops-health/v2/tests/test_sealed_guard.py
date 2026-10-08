"""M6: the sealed-split guard and the sealed-path refusal (sections 3, 4, 7).

No registered split is read and no file under v2/data/sealed/ is parsed here: refusals of the
real sealed splits are checked against a guard whose data directory does not exist (so a refusal
cannot come from reading anything), and the accepting paths use a temporary fixture data
directory filled with a few DEV rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from v2.research import dataio, generator, registry, sealed_guard
from v2.research.sealed_guard import FINAL_TEST_OPENING, SealedAccessError, SealedGuard
from v2.tests import _support as S

NON_FINAL_PURPOSES = ("TRAIN", "CALIBRATION", "CONFORMAL", "REGISTERED_SEARCH", "ABLATION",
                      "GATES", "", "final_test_opening")


def make_fixture(root: Path) -> tuple[Path, Path, dict]:
    """A fake data dir with every registered split name holding 3 DEV rows each."""
    data_dir = root / "data"
    runs_dir = root / "runs"
    rows = generator.dev_rows(3 * len(registry.SPLIT_TABLE))
    manifest = {"splits": {}}
    by_name = {}
    for i, name in enumerate(registry.SPLIT_TABLE):
        chunk = rows[3 * i: 3 * i + 3]
        digest = generator.write_jsonl(registry.split_path(name, data_dir), chunk)
        manifest["splits"][name] = {"sha256": digest}
        by_name[name] = chunk
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir, runs_dir, by_name


class SealedGuardTest(unittest.TestCase):
    def test_real_sealed_splits_refused_without_reading(self):
        with S.TempDir() as tmp:
            guard = SealedGuard(data_dir=tmp / "does-not-exist", runs_dir=tmp / "runs",
                                repo_root=registry.ROOT)
            for name in registry.SEALED_SPLITS + ("sealed",):
                for purpose in NON_FINAL_PURPOSES:
                    with self.subTest(split=name, purpose=purpose):
                        with self.assertRaises(SealedAccessError) as ctx:
                            guard.open_split(name, purpose)
                        self.assertIn("is sealed", str(ctx.exception))
            self.assertFalse((tmp / "runs").exists())
        # the module-level guard bound to the repository refuses the same way
        with self.assertRaises(SealedAccessError):
            sealed_guard.open_split("test", "TRAIN")

    def test_registered_splits_and_sizes(self):
        self.assertEqual(set(registry.SEALED_SPLITS), {"test", "shift_queue_saturation",
                                                       "shift_failure_bursts", "shift_clock_skew",
                                                       "shift_missing_tls"})
        self.assertEqual(registry.SPLIT_TABLE["test"][1], 50000)
        for name in registry.SEALED_SPLITS:
            self.assertEqual(registry.split_path(name).parent.name, "sealed")

    def test_unknown_split_and_wrong_purpose_are_refused(self):
        with S.TempDir() as tmp:
            data_dir, runs_dir, _ = make_fixture(tmp)
            guard = SealedGuard(data_dir=data_dir, runs_dir=runs_dir, repo_root=registry.ROOT)
            with self.assertRaises(SealedAccessError):
                guard.open_split("dev", "TRAIN")
            wrong = {"train": "CALIBRATION", "calibration": "TRAIN", "conformal": "REGISTERED_SEARCH",
                     "validation": "TRAIN"}
            for name, purpose in wrong.items():
                with self.subTest(split=name), self.assertRaises(SealedAccessError):
                    guard.open_split(name, purpose)

    def test_open_splits_verify_hash_before_parsing(self):
        with S.TempDir() as tmp:
            data_dir, runs_dir, by_name = make_fixture(tmp)
            guard = SealedGuard(data_dir=data_dir, runs_dir=runs_dir, repo_root=registry.ROOT)
            self.assertEqual(guard.open_split("train", "TRAIN"), by_name["train"])
            self.assertEqual(guard.open_split("conformal", "CONFORMAL"), by_name["conformal"])
            path = registry.split_path("calibration", data_dir)
            path.write_bytes(path.read_bytes().replace(b'"synthetic":true', b'"synthetic":false', 1))
            with self.assertRaises(SealedAccessError) as ctx:
                guard.open_split("calibration", "CALIBRATION")
            self.assertIn("sha256 does not match", str(ctx.exception))

    def test_final_opening_is_single_and_atomic(self):
        with S.TempDir() as tmp:
            data_dir, runs_dir, by_name = make_fixture(tmp)
            guard = SealedGuard(data_dir=data_dir, runs_dir=runs_dir, repo_root=registry.ROOT)
            frozen = tmp / "model.json"
            frozen.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SealedAccessError):
                guard.open_split("sealed", FINAL_TEST_OPENING)  # no contenders
            # tampered sealed file -> refused, and no opening is recorded
            victim = registry.split_path("shift_clock_skew", data_dir)
            original = victim.read_bytes()
            victim.write_bytes(original + b"\n")
            with self.assertRaises(SealedAccessError):
                guard.open_split("sealed", FINAL_TEST_OPENING, contenders={"v2": [frozen]})
            self.assertFalse((runs_dir / registry.TEST_OPENED_NAME).exists())
            victim.write_bytes(original)
            opened = guard.open_split("test", FINAL_TEST_OPENING, contenders={"v2": [frozen]})
            self.assertEqual(set(opened), set(registry.SEALED_SPLITS))
            self.assertEqual(opened["test"], by_name["test"])
            record = json.loads((runs_dir / registry.TEST_OPENED_NAME).read_text(encoding="utf-8"))
            self.assertEqual(record["purpose"], FINAL_TEST_OPENING)
            self.assertRegex(record["git_head"], r"^[0-9a-f]{40}$")
            self.assertEqual(record["contenders"]["v2"][frozen.as_posix()],
                             hashlib.sha256(frozen.read_bytes()).hexdigest())
            with self.assertRaises(SealedAccessError) as ctx:
                guard.open_split("shift_missing_tls", FINAL_TEST_OPENING, contenders={"v2": [frozen]})
            self.assertIn("already opened", str(ctx.exception))

    def test_dataio_refuses_sealed_paths_before_reading(self):
        for path in (registry.SEALED_DIR / "test.jsonl",
                     registry.DATA_DIR / "SEALED" / "anything.jsonl",
                     Path("elsewhere") / "sealed" / "x.jsonl",
                     registry.SEALED_DIR / ".." / "sealed" / "shift_clock_skew.jsonl"):
            with self.subTest(path=str(path)):
                self.assertTrue(dataio.is_sealed_path(path))
                with self.assertRaises(dataio.SealedPathError):
                    dataio.load_rows(path)
        self.assertFalse(dataio.is_sealed_path(registry.LEGACY_DIR / "train.jsonl"))

    def test_only_the_guard_and_make_data_name_sealed_splits(self):
        # final_score.py is the single final-opening caller (added before the opening, with its
        # DEV stand-in dry run in test_final_score.py); every other module, including
        # final_analyze.py, must stay unable to name the sealed set.
        allowed = {"sealed_guard.py", "make_data.py", "registry.py", "dataio.py", "final_score.py"}
        pattern = re.compile(r"SEALED_DIR|SEALED_SPLITS|sealed/|FINAL_TEST_OPENING")
        for path in sorted((registry.V2 / "research").glob("*.py")):
            if path.name in allowed:
                continue
            with self.subTest(module=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
