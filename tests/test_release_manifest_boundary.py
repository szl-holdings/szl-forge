"""Exercise the real manifest assembler and optimized CLI; never grant publication."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "emit_run_manifest.py"
RUN = "https://github.com/szl-holdings/szl-forge/actions/runs/123456"
VALID = {"pass_rate": 0.9, "heldout_passed": True, "refusal_no_regression": True}
SPEC = importlib.util.spec_from_file_location("release_manifest_boundary_subject", SCRIPT)
EMITTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EMITTER)


class ReleaseManifestBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
        ).strip()

    def invoke(self, *, evaluation=None, raw=None, sha=None, run=RUN, previous=None,
               previous_output=None, optimized=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            ev = root / "eval.json"
            if raw is None:
                raw = json.dumps(VALID if evaluation is None else evaluation).encode()
            ev.write_bytes(raw)
            bom = root / "bom.json"
            bom.write_text('{"bomFormat":"CycloneDX"}', encoding="utf-8")
            output = root / "manifest.json"
            if previous_output is not None:
                output.write_bytes(previous_output)
            argv = [sys.executable] + (["-O"] if optimized else []) + [
                str(SCRIPT), "--model-dir", str(model), "--eval", str(ev),
                "--bom", str(bom), "--git-sha", self.revision if sha is None else sha,
                "--workflow-run", run, "--out", str(output),
            ]
            if previous is not None:
                argv.extend(["--prev-hash", previous])
            # Actual assembler entrypoint; the optimized case runs its CLI too.
            env = os.environ.copy()
            if optimized:
                result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True,
                                        timeout=10, check=False)
            else:
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", argv[1:]), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        code = EMITTER.main()
                    except SystemExit as exc:
                        code = exc.code
                    except Exception:
                        code = 1
                        traceback.print_exc()
                result = SimpleNamespace(returncode=code, stdout=stdout.getvalue().encode(),
                                         stderr=stderr.getvalue().encode())
            body = output.read_bytes() if output.exists() else None
            return result, body

    def assert_rejected(self, **kwargs):
        result, body = self.invoke(**kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout.decode(errors="replace"))
        self.assertEqual(body, kwargs.get("previous_output"))
        self.assertNotIn(b"Traceback", result.stderr)

    def test_valid_report_stays_unsigned_and_nonconformant(self):
        result, body = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(body)
        self.assertEqual(output["schema"], "szl.run-manifest/v1")
        self.assertEqual(output["eval"], VALID)
        self.assertEqual(output["subject"]["git_sha"], self.revision)
        self.assertEqual(output["subject"]["workflow_run"], RUN)
        self.assertIs(output["signatures"]["manifest"], False)
        self.assertIs(output["conformance"]["all_vectors_passed"], False)
        self.assertEqual(output["chain"]["prev_hash"], "genesis")

    def test_false_is_preserved_not_promoted(self):
        result, body = self.invoke(evaluation={**VALID, "heldout_passed": False,
                                              "refusal_no_regression": False})
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(body)
        self.assertIs(output["eval"]["heldout_passed"], False)
        self.assertIs(output["eval"]["refusal_no_regression"], False)

    def test_truthy_nonboolean_flags_cannot_become_pass(self):
        for key in ("heldout_passed", "refusal_no_regression"):
            for value in ("false", "true", 1, 0, [], [False], {}, {"x": False}, None):
                with self.subTest(key=key, value=value):
                    self.assert_rejected(evaluation={**VALID, key: value})

    def test_pass_rate_requires_finite_unit_interval_number(self):
        for value in (True, False, "0.9", None, [], {}, -0.001, 1.001,
                      float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assert_rejected(evaluation={**VALID, "pass_rate": value})

    def test_numeric_boundaries_are_accepted_without_inventing_threshold(self):
        for value in (0, 1, 0.0, 1.0):
            with self.subTest(value=value):
                result, body = self.invoke(evaluation={**VALID, "pass_rate": value,
                                                      "heldout_passed": False})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(body)["eval"]["pass_rate"], value)

    def test_missing_fields_and_wrong_root_fail_closed(self):
        for key in VALID:
            with self.subTest(key=key):
                self.assert_rejected(evaluation={k: v for k, v in VALID.items() if k != key})
        for value in ([], "text", None, 1, True):
            with self.subTest(root=value):
                self.assert_rejected(raw=json.dumps(value).encode())

    def test_duplicate_keys_and_overflow_cannot_be_qualified(self):
        for raw in (
            b'{"pass_rate":0.9,"heldout_passed":false,"heldout_passed":true,"refusal_no_regression":true}',
            b'{"pass_rate":1e400,"heldout_passed":true,"refusal_no_regression":true}',
            b'{"pass_rate":0.9,"heldout_passed":true,"refusal_no_regression":true,"extra":{"x":1,"x":2}}',
            b'{"pass_rate":0.9,"heldout_passed":true,"refusal_no_regression":true,"extra":NaN}',
        ):
            with self.subTest(raw=raw):
                self.assert_rejected(raw=raw)

    def test_invalid_utf8_json_depth_and_size_fail_without_output(self):
        for raw in (b"\xff", b"{", b"[" * 1500 + b"]" * 1500, b" " * (1024 * 1024 + 1)):
            with self.subTest(size=len(raw)):
                self.assert_rejected(raw=raw)

    def test_exact_lowercase_git_sha_required(self):
        for value in ("abcd", "a" * 39, "a" * 41, "a" * 64, "A" * 40,
                      " " + self.revision, self.revision + "\n", "main"):
            with self.subTest(sha=value):
                self.assert_rejected(sha=value)

    def test_full_length_wrong_source_is_rejected(self):
        wrong = "b" * 40 if self.revision != "b" * 40 else "c" * 40
        self.assert_rejected(sha=wrong)

    def test_workflow_identity_is_canonical_and_repository_specific(self):
        for value in (
            "http://github.com/szl-holdings/szl-forge/actions/runs/123456",
            "https://github.com/other/repo/actions/runs/123456",
            "https://github.com.example/szl-holdings/szl-forge/actions/runs/123456",
            "https://user@github.com/szl-holdings/szl-forge/actions/runs/123456",
            RUN + "?branch=main", RUN + "#proof", RUN + "/jobs/1", RUN + "\n",
            "https://github.com/szl-holdings/szl-forge/actions/runs/0",
        ):
            with self.subTest(run=value):
                self.assert_rejected(run=value)

    def test_chain_accepts_only_genesis_or_exact_digest(self):
        for previous in ("genesis", "a" * 64):
            with self.subTest(previous=previous):
                result, body = self.invoke(previous=previous)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(body)["chain"]["prev_hash"], previous)
        for previous in ("", "abc", "a" * 63, "A" * 64, "genesis "):
            with self.subTest(previous=previous):
                self.assert_rejected(previous=previous)

    def test_invalid_input_does_not_overwrite_previous_evidence(self):
        self.assert_rejected(evaluation={**VALID, "heldout_passed": "false"},
                             previous_output=b"preserved-existing-evidence")

    def test_dirty_source_is_not_assigned_a_clean_revision(self):
        with patch.object(EMITTER.subprocess, "check_output", side_effect=[self.revision + "\n", " M scripts/emit_run_manifest.py\n"]):
            self.assert_rejected()

    def test_unavailable_checkout_is_not_accepted(self):
        with patch.object(EMITTER.subprocess, "check_output", side_effect=OSError("unavailable")):
            self.assert_rejected()

    def test_validation_also_runs_with_python_optimized(self):
        self.assert_rejected(evaluation={**VALID, "refusal_no_regression": "false"}, optimized=True)


if __name__ == "__main__":
    unittest.main()
