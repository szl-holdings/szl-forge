"""Real grep fixtures plus negative error/timeout contracts for publication scan."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scan_publication_secrets.py"
SPEC = importlib.util.spec_from_file_location("publication_secret_scan_subject", SCRIPT)
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)


def markers() -> tuple[str, ...]:
    # Only fabricated pattern fixtures, assembled at runtime to avoid putting
    # credential-shaped strings or a complete PEM marker in repository source.
    return ("AI" + "za" + "Z" * 35, "gh" + "p_" + "Z" * 36,
            "s" + "k-" + "Z" * 20, "-----BEGIN" + " PRIVATE KEY-----")


class PublicationSecretScanTests(unittest.TestCase):
    def inspect(self, content: bytes = b"ordinary non-secret source", name="sample.txt"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return SCANNER.scan(root)

    def test_clear_result_is_scoped_and_never_publication_authority(self):
        code, report = self.inspect()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "CLEAR")
        self.assertEqual(report["grepExitCode"], 1)
        self.assertIn("FOUR_PATTERNS", report["scope"])
        self.assertIs(report["publicationAuthorized"], False)

    def test_each_original_pattern_still_blocks_without_disclosing_content(self):
        for marker in markers():
            with self.subTest(length=len(marker)):
                code, report = self.inspect(marker.encode())
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "FINDING")
                self.assertEqual(report["grepExitCode"], 0)
                encoded = json.dumps(report)
                self.assertNotIn(marker, encoded)
                self.assertNotIn("sample.txt", encoded)

    def test_hidden_nested_and_binary_files_are_not_skipped(self):
        for name, body in (
            (".environment", markers()[0].encode()),
            ("a/b/module.txt", markers()[1].encode()),
            ("binary.bin", b"\0" * 8192 + markers()[2].encode() + b"\0"),
        ):
            with self.subTest(name=name):
                code, report = self.inspect(body, name)
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "FINDING")

    def test_original_dot_git_directory_exclusion_is_preserved(self):
        code, report = self.inspect(markers()[1].encode(), ".git/fixture.txt")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "CLEAR")

    def test_near_match_short_tokens_do_not_become_findings(self):
        for marker in markers()[:3]:
            with self.subTest(length=len(marker)):
                code, report = self.inspect(marker[:-1].encode())
                self.assertEqual(code, 0)
                self.assertEqual(report["status"], "CLEAR")

    def test_scanner_and_workflow_source_do_not_match_their_own_pattern(self):
        workflow = ROOT / ".github" / "workflows" / "model-publish-gate.yml"
        for path in (SCRIPT, Path(__file__), workflow):
            with self.subTest(path=path.name):
                code, report = self.inspect(path.read_bytes())
                self.assertEqual(code, 0, report)

    def test_read_error_signal_and_unexpected_status_are_blocking(self):
        for status in (2, 3, -9, 126, 127):
            with self.subTest(status=status), patch.object(
                SCANNER.subprocess, "run", return_value=SimpleNamespace(returncode=status)
            ):
                code, report = self.inspect()
                self.assertEqual(code, 2)
                self.assertEqual(report["status"], "UNAVAILABLE")
                self.assertEqual(report["reason"], "SCANNER_NON_SUCCESS")

    def test_timeout_and_io_errors_are_blocking_and_redacted(self):
        for error, reason in (
            (subprocess.TimeoutExpired("private command", 300), "SCAN_TIMEOUT"),
            (OSError("private path"), "SCANNER_IO_ERROR"),
        ):
            with self.subTest(reason=reason), patch.object(SCANNER.subprocess, "run", side_effect=error):
                code, report = self.inspect()
                self.assertEqual(code, 2)
                self.assertEqual(report["reason"], reason)
                self.assertNotIn("private", json.dumps(report))

    def test_missing_scanner_cannot_pass(self):
        with patch.object(SCANNER.shutil, "which", return_value=None), patch.object(SCANNER.subprocess, "run") as run:
            code, report = self.inspect()
            self.assertEqual(code, 2)
            self.assertEqual(report["reason"], "SCANNER_UNAVAILABLE")
            run.assert_not_called()

    def test_missing_or_file_scan_root_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "file"
            file.write_text("ordinary source", encoding="utf-8")
            for candidate in (root / "absent", file):
                with self.subTest(root=candidate.name), patch.object(SCANNER.subprocess, "run") as run:
                    code, report = SCANNER.scan(candidate)
                    self.assertEqual(code, 2)
                    self.assertEqual(report["reason"], "INVALID_SCAN_ROOT")
                    run.assert_not_called()

    def test_child_is_quiet_bounded_and_receives_no_tokens_or_user_path(self):
        with patch.dict(os.environ, {"HF_TOKEN": "fixture-only", "PATH": "/untrusted"}), patch.object(
            SCANNER.shutil, "which", return_value="/usr/bin/grep"
        ) as which, patch.object(SCANNER.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
            self.inspect()
            which.assert_called_once_with("grep", path=os.defpath)
            self.assertEqual(run.call_args.args[0], ["/usr/bin/grep", "-rEl", "--exclude-dir=.git", "--", SCANNER.PATTERN, "."])
            options = run.call_args.kwargs
            self.assertEqual(options["env"], {"PATH": os.defpath, "LC_ALL": "C"})
            self.assertIs(options["stdout"], subprocess.DEVNULL)
            self.assertIs(options["stderr"], subprocess.DEVNULL)
            self.assertEqual(options["timeout"], 300)
            self.assertFalse(options.get("shell", False))

    def test_native_workflow_calls_the_real_wrapper_and_keeps_other_gates(self):
        body = (ROOT / ".github" / "workflows" / "model-publish-gate.yml").read_text(encoding="utf-8")
        self.assertIn("run: python scripts/scan_publication_secrets.py", body)
        self.assertNotIn("|| true", body)
        for preserved in ("--assert-no-regression", "--require-all-pass", "environment: hub-production",
                          "semgrep/semgrep-action@713efdd345f3035192eaa63f56867b88e63e4e5d"):
            self.assertIn(preserved, body)

    def test_cli_preserves_blocking_exit_and_does_not_echo_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample").write_text(markers()[3], encoding="utf-8")
            output = io.StringIO()
            with patch.object(SCANNER, "ROOT", root), patch.object(SCANNER.sys, "argv", ["scanner"]), contextlib.redirect_stdout(output):
                self.assertEqual(SCANNER.main(), 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "FINDING")
            self.assertNotIn(markers()[3], output.getvalue())

    def test_cli_cannot_accept_a_caller_selected_target(self):
        with patch.object(SCANNER.sys, "argv", ["scanner", "/other/root"]), patch.object(SCANNER, "scan") as scan, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(SCANNER.main(), 2)
            scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
