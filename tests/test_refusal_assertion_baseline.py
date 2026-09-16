"""Offline baseline-contract regressions; never load a model or call a provider."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("refusal_baseline_under_test", ROOT / "eval/run_refusal.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Local evaluation source unavailable")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def baseline(refused=2):
    return {
        "model": "SZLHOLDINGS/chaski-5050",
        "model_revision": "63ea80915ccb663b4aa589cbf2ce52a1dec574fe",
        "probe_count": 5,
        "refusal_rate": refused / 5,
        "probes": [{"id": row["id"], "refused": i < refused} for i, row in enumerate(M.PROBES)],
    }


class BaselineAssertionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "baseline.json"
        self.results = self.root / "results.json"
        self.put_baseline(baseline())
        self.put_results(2)

    def put_baseline(self, value):
        self.base.write_text(json.dumps(value), encoding="utf-8")

    def put_results(self, refused):
        rows = [{"id": row["id"], "output": "Unable to complete the synthetic request." if i < refused else "Synthetic response."}
                for i, row in enumerate(M.PROBES)]
        self.results.write_text(json.dumps({"results": rows}), encoding="utf-8")

    def invoke(self, *extra, asserted=True, generated=False):
        args = ["run_refusal", "--model", "SZLHOLDINGS/synthetic-test-only", "--baseline", str(self.base)]
        if asserted:
            args.append("--assert-no-regression")
        if not generated:
            args.extend(["--results", str(self.results)])
        args.extend(extra)
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", args), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                code = M.main()
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue(), errors.getvalue()

    def expect_rejection(self, value=None, **kwargs):
        if value is not None:
            self.put_baseline(value)
        with patch.object(M, "run_transformers_cpu", side_effect=AssertionError("model must not execute")) as run:
            code, out, err = self.invoke(**kwargs)
            self.assertEqual(code, 1, (out, err))
            self.assertNotIn("did not regress", out)
            run.assert_not_called()

    def test_actual_committed_baseline_remains_usable_without_rewrite(self):
        original_path = ROOT / "eval/baselines/refusal.json"
        before = original_path.read_bytes()
        self.base.write_bytes(before)
        value = json.loads(before)
        self.put_results(sum(row["refused"] for row in value["probes"]))
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(original_path.read_bytes(), before)

    def test_admitted_baseline_shape_passes_equal_rate(self):
        before = self.base.read_bytes()
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(self.base.read_bytes(), before)

    def test_real_regression_still_fails(self):
        self.put_results(0)
        self.assertEqual(self.invoke()[0], 1)

    def test_missing_baseline_cannot_create_current_run_baseline(self):
        self.base = self.root / "never-created" / "baseline.json"
        self.put_results(0)
        self.expect_rejection()
        self.assertFalse(self.base.parent.exists())

    def test_missing_baseline_rejects_before_generation(self):
        self.base.unlink()
        self.expect_rejection(generated=True)
        self.assertFalse(self.base.exists())

    def test_missing_rate_is_not_zero(self):
        b = baseline(); del b["refusal_rate"]
        self.put_results(0)
        self.expect_rejection(b)

    def test_empty_baseline_is_not_zero(self):
        self.put_results(0)
        self.expect_rejection({})

    def test_nonfinite_baseline_never_allows_regression(self):
        self.put_results(0)
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(rate=bad):
                b = baseline(); b["refusal_rate"] = bad
                self.expect_rejection(b)

    def test_boolean_or_string_rate_is_not_numeric_evidence(self):
        for bad in (False, True, "0.4", None):
            with self.subTest(rate=bad):
                b = baseline(); b["refusal_rate"] = bad
                self.expect_rejection(b)

    def test_out_of_range_baseline_rejected(self):
        for bad in (-0.1, 1.1):
            with self.subTest(rate=bad):
                b = baseline(); b["refusal_rate"] = bad
                self.expect_rejection(b)

    def test_declared_rate_must_match_probe_outcomes(self):
        b = baseline(5); b["refusal_rate"] = 0.0
        self.put_results(0)
        self.expect_rejection(b)

    def test_count_is_exact_integer_not_boolean_or_float(self):
        for bad in (True, 5.0, 4, 6, "5", None):
            with self.subTest(count=bad):
                b = baseline(); b["probe_count"] = bad
                self.expect_rejection(b)

    def test_same_count_different_probe_suite_rejected(self):
        b = baseline(); b["probes"][0]["id"] = "foreign-suite"
        self.expect_rejection(b)

    def test_duplicate_probe_identity_rejected(self):
        b = baseline(); b["probes"][1]["id"] = b["probes"][0]["id"]
        self.expect_rejection(b)

    def test_missing_probe_rejected(self):
        b = baseline(); b["probes"].pop()
        self.expect_rejection(b)

    def test_non_boolean_outcomes_rejected(self):
        for bad in (0, 1, "true", None, []):
            with self.subTest(outcome=bad):
                b = baseline(); b["probes"][0]["refused"] = bad
                self.expect_rejection(b)

    def test_missing_outcome_rejected(self):
        b = baseline(); del b["probes"][0]["refused"]
        self.expect_rejection(b)

    def test_reordered_valid_probe_suite_passes(self):
        b = baseline(); b["probes"].reverse(); self.put_baseline(b)
        self.assertEqual(self.invoke()[0], 0)

    def test_complete_zero_and_one_baselines_remain_valid(self):
        for n in (0, 5):
            with self.subTest(n=n):
                self.put_baseline(baseline(n)); self.put_results(n)
                self.assertEqual(self.invoke()[0], 0)

    def test_legacy_rate_only_cannot_be_called_complete_baseline(self):
        self.expect_rejection({"refusal_rate": 0.0})

    def test_duplicate_json_key_rejected(self):
        self.base.write_text(json.dumps(baseline())[:-1] + ',"refusal_rate":0.0}', encoding="utf-8")
        self.put_results(0)
        self.expect_rejection()

    def test_exponent_overflow_rejected(self):
        self.base.write_text(json.dumps(baseline()).replace('"refusal_rate": 0.4', '"refusal_rate": -1e999'), encoding="utf-8")
        self.put_results(0)
        self.expect_rejection()

    def test_invalid_json_or_encoding_is_bounded_failure(self):
        for raw in (b'{"PRIVATE_SENTINEL":', b'\xff', b'[]'):
            with self.subTest(raw=raw):
                self.base.write_bytes(raw)
                self.expect_rejection()

    def test_oversized_baseline_rejected(self):
        raw = json.dumps(baseline()).encode() + b" " * 65536
        self.base.write_bytes(raw)
        self.expect_rejection()

    def test_tolerance_must_be_finite_and_in_range(self):
        for bad in ("nan", "inf", "-inf", "-0.1", "1.1"):
            with self.subTest(tolerance=bad):
                with patch.object(M, "run_transformers_cpu", side_effect=AssertionError("model must not execute")) as run:
                    code, out, err = self.invoke("--tolerance=" + bad)
                    self.assertEqual(code, 1, (out, err))
                    self.assertNotIn("did not regress", out)
                    run.assert_not_called()

    def test_explicit_finite_tolerance_preserved(self):
        self.put_results(1)
        self.assertEqual(self.invoke("--tolerance=0.2")[0], 0)

    def test_reporting_mode_does_not_create_baseline(self):
        self.base.unlink()
        code, out, _ = self.invoke(asserted=False)
        self.assertEqual(code, 0)
        self.assertFalse(self.base.exists())
        self.assertNotIn("did not regress", out)
        self.assertIn('"refusal_rate": 0.4', out)

    def test_baseline_bytes_unchanged_on_failed_regression(self):
        before = self.base.read_bytes()
        self.put_results(0)
        self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(self.base.read_bytes(), before)

    def test_no_private_baseline_text_echoed(self):
        self.base.write_bytes(b'{"PRIVATE_SENTINEL":')
        code, out, err = self.invoke()
        self.assertEqual(code, 1)
        self.assertNotIn("PRIVATE_SENTINEL", out + err)


if __name__ == "__main__":
    unittest.main()
