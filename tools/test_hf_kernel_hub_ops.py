"""Offline regressions: syntax is not admission; old claims are not live data."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# Works both with root `unittest tools/test_...py` and discovery in tools/.
SPEC = importlib.util.spec_from_file_location("hf_kernel_hub_ops", Path(__file__).with_name("hf_kernel_hub_ops.py"))
assert SPEC is not None and SPEC.loader is not None
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)


def declaration(**changes):
    value = {"repo_id": "SZLHOLDINGS/szl-kernels", "revision": "c" * 40, "repo_type": "kernel"}
    value.update(changes)
    return value


class ClassifyTests(unittest.TestCase):
    def test_exact_revision_is_only_pinned(self):
        result = ops.classify_consumer_call(declaration())
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"], "PINNED")
        self.assertEqual(result["validation_scope"], "STATIC_DECLARATION_ONLY")
        for key in ("executed", "provider_resolved", "publisher_trust_verified", "production_authorization"):
            self.assertIs(result[key], False)

    def test_full_sha256_syntax(self):
        self.assertTrue(ops.classify_consumer_call(declaration(revision="a" * 64))["ok"])

    def test_model_type_forbidden(self):
        self.assertIn("LEGACY_MODEL_TYPE_FORBIDDEN", ops.classify_consumer_call(declaration(repo_type="model"))["blockers"])

    def test_no_implicit_repo_type(self):
        for value in (None, "", False, 0, [], "space", "Kernel"):
            with self.subTest(value=value):
                self.assertFalse(ops.classify_consumer_call(declaration(repo_type=value))["ok"])
        call = declaration()
        del call["repo_type"]
        self.assertIn("EXPLICIT_KERNEL_TYPE_REQUIRED", ops.classify_consumer_call(call)["blockers"])

    def test_missing_revision(self):
        for value in (None, ""):
            self.assertIn("REVISION_REQUIRED", ops.classify_consumer_call(declaration(revision=value))["blockers"])

    def test_floating_main(self):
        self.assertIn("FLOATING_MAIN_NOT_ADMITTED_AT_RUNTIME", ops.classify_consumer_call(declaration(revision="main"))["blockers"])

    def test_nonimmutable_revisions(self):
        for value in ("v1", "release", "refs/heads/main", "deadbeef", "a" * 39, "a" * 41, "a" * 63, "a" * 65, "g" * 40, "A" * 40, "a" * 40 + "\n", " a" * 20, 1, True, {}, []):
            with self.subTest(value=value):
                result = ops.classify_consumer_call(declaration(revision=value))
                self.assertFalse(result["ok"])
                self.assertIn("IMMUTABLE_REVISION_REQUIRED", result["blockers"])

    def test_invalid_repo_ids(self):
        for value in (None, "", True, 3, [], {}, "no-owner", "SZLHOLDINGS/../evil", "https://huggingface.co/a/b", "SZLHOLDINGS/a\n", "SZLHOLDINGS/*", "SZLHOLDINGS/a b", "SZLHOLDINGS/a.git", "SZLHOLDINGS/" + "a" * 97):
            with self.subTest(value=value):
                self.assertIn("VALID_REPO_ID_REQUIRED", ops.classify_consumer_call(declaration(repo_id=value))["blockers"])

    def test_broad_trust_blocked(self):
        self.assertIn("TRUST_REMOTE_CODE_REQUIRES_REVIEW", ops.classify_consumer_call(declaration(trust_remote_code=True))["blockers"])

    def test_exact_allowlist_is_only_syntax(self):
        result = ops.classify_consumer_call(declaration(trust_remote_code=["SZLHOLDINGS/szl-kernels"]))
        self.assertTrue(result["ok"])
        self.assertIs(result["publisher_trust_verified"], False)

    def test_other_allowlists_fail(self):
        for value in ([], ["SZLHOLDINGS/*"], ["szlholdings/szl-kernels"], ["OTHER/szl-kernels"], ["SZLHOLDINGS/szl-kernels", "OTHER/repo"], ["SZLHOLDINGS/szl-kernels"] * 2, [True], [[]]):
            with self.subTest(value=value):
                self.assertIn("EXACT_REPOSITORY_ALLOWLIST_REQUIRED", ops.classify_consumer_call(declaration(trust_remote_code=value))["blockers"])

    def test_malformed_trust_not_coerced(self):
        for value in (None, 0, 1, "false", "true", {}, ("SZLHOLDINGS/szl-kernels",)):
            with self.subTest(value=value):
                self.assertIn("INVALID_TRUST_REMOTE_CODE", ops.classify_consumer_call(declaration(trust_remote_code=value))["blockers"])

    def test_all_blockers_preserved(self):
        result = ops.classify_consumer_call(declaration(repo_id=None, repo_type="model", revision="v1", trust_remote_code=True))
        self.assertEqual(len(result["blockers"]), 4)
        self.assertEqual(result["state"], "HOLD")

    def test_nondictionary_rejected(self):
        for value in (None, [], "", 1):
            with self.assertRaises(TypeError):
                ops.classify_consumer_call(value)


class EvidenceTests(unittest.TestCase):
    def test_historical_report_cannot_be_live(self):
        report = ops.historical_snapshot(generated_at="2026-09-13T12:00:00Z")
        self.assertEqual(report["generated_at"], "2026-09-13T12:00:00Z")
        self.assertIsNone(report["observed_at"])
        self.assertIsNone(report["counts"])
        self.assertIs(report["collection_performed"], False)
        self.assertEqual(report["state"], "HOLD")
        self.assertEqual(report["historical_claims"]["list_and_tags"]["claimed_observed_at"], "2026-09-11T23:07Z")
        self.assertIsNone(report["historical_claims"]["version_branches"]["claimed_observed_at"])
        self.assertIs(report["historical_claims"]["independently_verified"], False)
        self.assertIs(report["runtime_loaded"], False)
        self.assertIs(report["production_authorization"], False)

    def test_report_generation_does_not_refresh_history(self):
        a = ops.historical_snapshot(generated_at="2026-09-13T12:00:00Z")
        b = ops.historical_snapshot(generated_at="2026-09-14T12:00:00Z")
        self.assertEqual(a["historical_claims"], b["historical_claims"])

    def test_date_does_not_invent_midnight(self):
        report = ops.historical_snapshot()
        self.assertEqual(report["takedown_starts_date"], "2026-09-13")
        self.assertNotIn("takedown_starts", report)

    def test_legacy_name_has_no_live_authority(self):
        self.assertEqual(ops.live_snapshot()["evidence_kind"], "HISTORICAL_SOURCE_CLAIMS_ONLY")
        with self.assertRaises(ValueError):
            ops.live_snapshot("2026-09-13T12:00:00Z")

    def test_reports_are_independent_values(self):
        a = ops.historical_snapshot()
        a["historical_claims"]["list_and_tags"]["counts"]["kernels_api_list"] = 999
        self.assertEqual(ops.historical_snapshot()["historical_claims"]["list_and_tags"]["counts"]["kernels_api_list"], 14)

    def test_no_network_or_kernel_loading(self):
        with mock.patch("socket.socket", side_effect=AssertionError("network forbidden")):
            self.assertEqual(ops.live_snapshot()["state"], "HOLD")
            self.assertEqual(ops.classify_consumer_call(declaration())["state"], "PINNED")
        self.assertNotIn("kernels", ops.__dict__)


class CliTests(unittest.TestCase):
    def test_stdout_json_and_hold_exit(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = ops.main([])
        self.assertEqual(code, 2)
        self.assertIsNone(json.loads(stream.getvalue())["counts"])

    def test_new_file_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            self.assertEqual(ops.main(["--output", str(path)]), 2)
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                ops.main(["--output", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_module_exits_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(Path(ops.__file__).resolve())], cwd=folder, capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["state"], "HOLD")
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
