"""final_score.py on a DEV stand-in: the dry run never touches a sealed file or the real marker.

DEV data only.  The stand-in splits are generated under split names "dev-standin-<split>" in a
temporary directory and opened by a SealedGuard bound to that directory.  An audit hook fails the
dry run if any file under v2/data/sealed/, the real runs/TEST_OPENED.json or the real
v2/results/final/ is opened while the dry run executes.  The real final run is never called here.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import unittest
from unittest import mock

from v2.research import final_score, registry
from v2.tests import _support as S


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


_FORBIDDEN = tuple(_norm(p) for p in (registry.SEALED_DIR,
                                      registry.RUNS_DIR / registry.TEST_OPENED_NAME,
                                      final_score.OUT_DIR))
_WATCH = {"active": False, "touched": []}


def _audit(event: str, args) -> None:
    if not _WATCH["active"] or event != "open" or not args:
        return
    target = args[0]
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    path = _norm(target)
    for forbidden in _FORBIDDEN:
        if path == forbidden or path.startswith(forbidden + os.sep):
            _WATCH["touched"].append(path)
            raise PermissionError(f"dry run touched a protected path: {path}")


sys.addaudithook(_audit)

TEST_ROWS = 300
SHIFT_ROWS = 120
QUIET = lambda line: None  # noqa: E731


class FinalScoreDryRunTest(unittest.TestCase):
    @classmethod
    @unittest.skipUnless(S.origin_commit_present(), S.ORIGIN_SKIP_REASON)
    def setUpClass(cls):
        cls._tmp = S.TempDir()
        cls.base = cls._tmp.__enter__()
        cls.real_marker_existed = (registry.RUNS_DIR / registry.TEST_OPENED_NAME).exists()
        cls.real_out_existed = final_score.OUT_DIR.exists()
        _WATCH["touched"].clear()
        _WATCH["active"] = True
        try:
            cls.a = final_score.dry_run(cls.base / "a", test_rows=TEST_ROWS,
                                        shift_rows=SHIFT_ROWS, log=QUIET)
            cls.b = final_score.dry_run(cls.base / "b", test_rows=TEST_ROWS,
                                        shift_rows=SHIFT_ROWS, log=QUIET)
        finally:
            _WATCH["active"] = False

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def records(self, run: str, split: str) -> list[dict]:
        path = self.base / run / "final" / f"scores_{split}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_no_sealed_file_real_marker_or_real_output_was_touched(self):
        self.assertEqual(_WATCH["touched"], [])
        self.assertEqual((registry.RUNS_DIR / registry.TEST_OPENED_NAME).exists(),
                         self.real_marker_existed)
        self.assertEqual(final_score.OUT_DIR.exists(), self.real_out_existed)
        self.assertTrue((self.base / "a" / "standin" / "runs" / registry.TEST_OPENED_NAME).exists())

    def test_manifest_matches_the_score_files(self):
        manifest = self.a["manifest"]
        self.assertEqual(manifest["mode"], "DRY_RUN_DEV_STANDIN")
        self.assertEqual(manifest["splits"], list(registry.SEALED_SPLITS) + ["legacy_v1_test"])
        out = self.base / "a" / "final"
        on_disk = json.loads((out / final_score.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk, manifest)
        for split, entry in manifest["files"].items():
            with self.subTest(split=split):
                data = (out / entry["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(len(data), entry["bytes"])
                self.assertEqual(data.count(b"\n"), entry["rows"])
                self.assertEqual(entry["rows"], entry["expected_rows"])
                self.assertEqual(manifest["consistency"][split],
                                 {"rows_as_registered": True, "regime_as_registered": True})
        marker = (self.base / "a" / "standin" / "runs" / registry.TEST_OPENED_NAME).read_bytes()
        self.assertEqual(manifest["opening"]["marker_sha256"], hashlib.sha256(marker).hexdigest())
        self.assertEqual(manifest["opening"]["marker"]["purpose"], final_score.FINAL_PURPOSE)
        self.assertEqual(sorted(manifest["opening"]["marker"]["contenders"]),
                         sorted(final_score.CONTENDERS))
        self.assertFalse((out / final_score.INCIDENT_NAME).exists())

    def test_pre_opening_checks_pass_except_strict_only(self):
        checks = self.a["manifest"]["pre_opening_checks"]
        failed = [k for k, v in checks.items() if not v["ok"]]
        self.assertTrue(set(failed) <= set(final_score.STRICT_ONLY), failed)
        self.assertTrue(checks["v1_reproduces_published_legacy_test_confusion"]["ok"])
        self.assertTrue(checks["v2_example_input"]["ok"])
        validity = self.a["manifest"]["validity_inputs"]
        self.assertEqual(sorted(validity), sorted(final_score.CONTENDERS))
        for name in ("v2", "ABL-data", "ABL-optimizer"):
            self.assertEqual(validity[name]["probes"]["refused"], validity[name]["probes"]["total"])
        self.assertEqual(validity["majority"]["probes"]["refused"], 0)
        self.assertTrue(validity["v2"]["receipt_verified"])
        self.assertFalse(validity["majority"]["receipt_verified"])
        self.assertFalse(validity["rule"]["receipt_verified"])

    def test_records_follow_each_contender_definition(self):
        thresholds = self.a["manifest"]["pre_opening_checks"]["thresholds"]["detail"]
        for split in self.a["manifest"]["splits"]:
            rows = self.records("a", split)
            with self.subTest(split=split):
                regime = "legacy_v1" if split == "legacy_v1_test" else registry.SPLIT_TABLE[split][0]
                for r in rows:
                    self.assertEqual(r["split"], split)
                    self.assertEqual(r["regime"], regime)
                    self.assertIn(r["label"], (0, 1))
                    self.assertEqual(sorted(r["features"]), sorted(final_score.generator.FEATURE_NAMES))
                    if split != "legacy_v1_test":
                        self.assertTrue(r["sample_id"].startswith(f"dev-standin-{split}-"))
                    self.assertEqual(r["majority"], {"decision": 0})
                    self.assertEqual(r["rule"]["decision"],
                                     1 if r["features"]["consecutive_failures"] > 0 else 0)
                    v2 = r["v2"]
                    self.assertEqual(v2["decision"], 1 if v2["score"] >= thresholds["v2"] else 0)
                    expected = {(1,): "ALERT", (0,): "NO_ALERT"}.get(tuple(v2["prediction_set"]),
                                                                     "ABSTAIN")
                    self.assertEqual(v2["advisory"], expected)
                    self.assertEqual(v2["abstain_reason"] is None, expected != "ABSTAIN")
                    for name in ("ABL-data", "ABL-optimizer"):
                        self.assertEqual(r[name]["decision"],
                                         1 if r[name]["score"] >= thresholds[name] else 0)
                    if abs(r["v1"]["score"] - thresholds["v1"]) > 1e-11:
                        self.assertEqual(r["v1"]["decision"],
                                         1 if r["v1"]["score"] >= thresholds["v1"] else 0)
                    self.assertEqual(r["v1"]["score"], round(r["v1"]["score"], 12))

    def test_dry_run_is_deterministic(self):
        for split in self.a["manifest"]["splits"]:
            with self.subTest(split=split):
                self.assertEqual(self.a["manifest"]["files"][split]["sha256"],
                                 self.b["manifest"]["files"][split]["sha256"])

    def test_second_opening_into_the_same_place_is_refused(self):
        _WATCH["active"] = True
        try:
            with self.assertRaises(final_score.FinalScoreRefused) as ctx:
                final_score.dry_run(self.base / "a", test_rows=TEST_ROWS, shift_rows=SHIFT_ROWS,
                                    log=QUIET)
        finally:
            _WATCH["active"] = False
        self.assertIn("marker_absent", str(ctx.exception))
        self.assertIn("output_dir_empty", str(ctx.exception))
        self.assertEqual(_WATCH["touched"], [])

    def test_strict_run_refuses_before_opening_when_the_marker_exists(self):
        with S.TempDir() as tmp:
            marker = tmp / registry.TEST_OPENED_NAME
            marker.write_text("{}\n", encoding="utf-8")
            calls = []

            def opener(*args, **kwargs):
                calls.append(args)
                raise AssertionError("the opener must not be called")

            with self.assertRaises(final_score.FinalScoreRefused) as ctx:
                final_score.run_opening(opener=opener, marker=marker, out_dir=tmp / "out",
                                        strict=True, mode="TEST", expected_rows={}, log=QUIET)
            self.assertEqual(calls, [])
            self.assertIn("marker_absent", str(ctx.exception))
            self.assertFalse((tmp / "out").exists())

    def test_disk_full_write_is_retried_not_lost(self):
        real_replace = os.replace
        calls = {"replace": 0, "sleep": []}

        def flaky_replace(src, dst):
            calls["replace"] += 1
            if calls["replace"] <= 2:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_replace(src, dst)

        lines = []
        records = self.records("a", "test")[:5]
        with S.TempDir() as tmp, \
                mock.patch.object(final_score.os, "replace", side_effect=flaky_replace), \
                mock.patch.object(final_score, "_sleep", side_effect=calls["sleep"].append):
            entry = final_score.write_score_file(tmp, "test", records, log=lines.append)
            data = (tmp / "scores_test.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            self.assertEqual(entry["rows"], 5)
            self.assertEqual(sorted(p.name for p in tmp.iterdir()), ["scores_test.jsonl"])
            with self.assertRaises(FileExistsError):
                final_score.write_score_file(tmp, "test", records, log=lines.append)
        self.assertEqual(calls["sleep"], [final_score.DISK_FULL_WAIT_SECONDS] * 2)
        self.assertEqual(len(lines), 2)
        with S.TempDir() as tmp, \
                mock.patch.object(final_score.os, "replace",
                                  side_effect=OSError(errno.EACCES, "denied")):
            with self.assertRaises(OSError):
                final_score.write_score_file(tmp, "test", records, log=lines.append)
            self.assertEqual(list(tmp.iterdir()), [])

    def test_dry_run_refuses_repository_output_paths(self):
        for target in (registry.V2 / "results" / "dry", registry.RUNS_DIR / "dry",
                       registry.DATA_DIR / "dry"):
            with self.subTest(target=str(target)):
                with self.assertRaises(final_score.FinalScoreRefused):
                    final_score.dry_run(target, test_rows=10, shift_rows=10, log=QUIET)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
