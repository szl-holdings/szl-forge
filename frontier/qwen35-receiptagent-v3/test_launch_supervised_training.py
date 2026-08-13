from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import signal
import subprocess
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "v3_supervised_launcher", HERE / "launch_supervised_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = load_launcher()
SOURCE = "a" * 40
PYTHON = "/home/rosie/.venvs/szl-unsloth/bin/python"
RUNS_ROOT = "/home/rosie/szl-runs/receiptagent-v3-supervised"


def candidate() -> dict:
    return {
        "supervision_policy": {
            "python_executable": PYTHON,
            "systemd_run_executable": "/usr/bin/systemd-run",
            "systemctl_executable": "/usr/bin/systemctl",
            "runs_root": RUNS_ROOT,
        }
    }


class FlushTrackingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


class ArgumentContractTests(unittest.TestCase):
    def parse_failure(self, argv: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                launcher.argument_parser().parse_args(argv)
        self.assertEqual(2, raised.exception.code)

    def test_accepts_exactly_one_source_and_run_kind(self):
        args = launcher.argument_parser().parse_args(
            ["--source-commit", SOURCE, "--run-kind", "smoke"]
        )
        self.assertEqual(SOURCE, args.source_commit)
        self.assertEqual("smoke", args.run_kind)

    def test_rejects_duplicate_options(self):
        self.parse_failure(
            [
                "--source-commit",
                SOURCE,
                "--source-commit",
                SOURCE,
                "--run-kind",
                "smoke",
            ]
        )
        self.parse_failure(
            [
                "--source-commit",
                SOURCE,
                "--run-kind",
                "smoke",
                "--run-kind",
                "full",
            ]
        )

    def test_rejects_missing_unknown_abbreviated_and_malformed_arguments(self):
        self.parse_failure(["--source-commit", SOURCE])
        self.parse_failure(["--run-kind", "smoke"])
        self.parse_failure(
            ["--source-commit", SOURCE, "--run-kind", "smoke", "--unknown"]
        )
        self.parse_failure(["--source", SOURCE, "--run-kind", "smoke"])
        self.parse_failure(["--source-commit", "A" * 40, "--run-kind", "smoke"])
        self.parse_failure(["--source-commit", SOURCE, "--run-kind", "training"])


class LauncherContractTests(unittest.TestCase):
    def test_committed_python_must_exactly_match_launcher(self):
        self.assertEqual(
            PYTHON,
            launcher.committed_python_path(candidate(), observed_executable=PYTHON),
        )
        with self.assertRaisesRegex(launcher.LauncherError, "exact Python path"):
            launcher.committed_python_path(
                candidate(), observed_executable="/usr/bin/python3"
            )

    def test_service_name_is_random_shape_and_systemd_safe(self):
        with mock.patch.object(
            launcher.secrets, "token_hex", return_value="ab" * 16
        ) as token_hex:
            name = launcher.generate_service_name("full")
        token_hex.assert_called_once_with(16)
        self.assertEqual(
            "szl-ra3-supervisor-" + "ab" * 16,
            name,
        )
        self.assertIsNotNone(launcher.SERVICE_NAME.fullmatch(name))
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)

    def test_attempt_identity_is_bound_to_safe_unit_and_committed_runs_root(self):
        name = "szl-ra3-supervisor-" + "ab" * 16
        run_id, path = launcher.attempt_identity(candidate(), name)
        self.assertEqual("ab" * 16, run_id)
        self.assertEqual(f"{RUNS_ROOT}/{run_id}", path)
        with self.assertRaisesRegex(launcher.LauncherError, "normalized absolute"):
            malformed = candidate()
            malformed["supervision_policy"]["runs_root"] = "relative/runs"
            launcher.attempt_identity(malformed, name)

    def test_systemd_command_has_exact_service_sandbox_and_worker_argv(self):
        name = "szl-ra3-supervisor-" + "cd" * 16
        command = launcher.systemd_command(
            service_name=name,
            python_executable=PYTHON,
            source_commit=SOURCE,
            run_kind="smoke",
        )
        self.assertEqual("/usr/bin/systemd-run", command[0])
        self.assertIn("--user", command)
        self.assertIn("--wait", command)
        self.assertIn("--pipe", command)
        self.assertIn("--collect", command)
        self.assertIn(f"--unit={name}", command)
        self.assertIn("--service-type=exec", command)
        self.assertIn(f"--working-directory={launcher.ROOT}", command)
        for property_value in launcher.SYSTEMD_PROPERTIES:
            self.assertIn(f"--property={property_value}", command)
        separator = command.index("--")
        self.assertEqual(
            [
                "/usr/bin/env",
                "-i",
                *(
                    f"{key}={value}"
                    for key, value in sorted(launcher.SUPERVISOR_ENVIRONMENT.items())
                ),
                PYTHON,
                "-I",
                "-B",
                str(launcher.SUPERVISOR),
                "--source-commit",
                SOURCE,
                "--run-kind",
                "smoke",
                "--unit-name",
                name,
            ],
            command[separator + 1 :],
        )

    def test_main_emits_identity_propagates_exit_and_cleans_without_a_shell(self):
        candidate_bytes = json.dumps(candidate()).encode("utf-8")
        managed_calls: list[tuple[list[str], dict]] = []

        def fake_run(command, **kwargs):
            if command[0] == "/usr/bin/git":
                self.assertEqual(
                    [
                        "/usr/bin/git",
                        "show",
                        f"{SOURCE}:{launcher.RELATIVE_CANDIDATE}",
                    ],
                    command,
                )
                return subprocess.CompletedProcess(
                    command, 0, stdout=candidate_bytes, stderr=b""
                )
            managed_calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                command, 73 if command[0] == "/usr/bin/systemd-run" else 0
            )

        output = FlushTrackingStream()
        with (
            mock.patch.object(launcher.subprocess, "run", side_effect=fake_run),
            mock.patch.object(launcher.sys, "executable", PYTHON),
            mock.patch.object(launcher.Path, "is_file", return_value=True),
            mock.patch.object(launcher.os, "access", return_value=True),
            mock.patch.object(launcher.secrets, "token_hex", return_value="ef" * 16),
            mock.patch.object(launcher.sys, "stdout", output),
        ):
            code = launcher.main(["--source-commit", SOURCE, "--run-kind", "full"])

        self.assertEqual(73, code)
        self.assertGreaterEqual(output.flush_count, 1)
        self.assertEqual(2, len(managed_calls))
        command, kwargs = managed_calls[0]
        self.assertEqual("/usr/bin/systemd-run", command[0])
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["check"], False)
        self.assertEqual(launcher.ROOT, kwargs["cwd"])
        service_name = "szl-ra3-supervisor-" + "ef" * 16
        run_id = "ef" * 16
        self.assertEqual(
            (
                f"supervisorUnit={service_name}.service\n"
                f"supervisorRunId={run_id}\n"
                f"supervisorAttemptPath={RUNS_ROOT}/{run_id}\n"
            ),
            output.getvalue(),
        )
        cleanup_command, cleanup_kwargs = managed_calls[1]
        self.assertEqual(
            ["/usr/bin/systemctl", "--user", "stop", f"{service_name}.service"],
            cleanup_command,
        )
        self.assertIs(cleanup_kwargs["shell"], False)
        self.assertIs(cleanup_kwargs["check"], False)
        self.assertEqual(launcher.CLEANUP_TIMEOUT_SECONDS, cleanup_kwargs["timeout"])
        self.assertEqual(launcher.SUPERVISOR_ENVIRONMENT, cleanup_kwargs["env"])

    def test_interruptions_stop_exact_unit_restore_handlers_and_return_signal_code(
        self,
    ):
        candidate_bytes = json.dumps(candidate()).encode("utf-8")
        service_name = "szl-ra3-supervisor-" + "cd" * 16
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                managed_calls: list[tuple[list[str], dict]] = []

                def fake_run(command, **kwargs):
                    if command[0] == "/usr/bin/git":
                        return subprocess.CompletedProcess(
                            command, 0, stdout=candidate_bytes, stderr=b""
                        )
                    managed_calls.append((command, kwargs))
                    if command[0] == "/usr/bin/systemd-run":
                        launcher._raise_launcher_interrupted(signum, None)
                    return subprocess.CompletedProcess(command, 0)

                handlers_before = {
                    observed: signal.getsignal(observed)
                    for observed in (signal.SIGINT, signal.SIGTERM)
                }
                with (
                    mock.patch.object(launcher.subprocess, "run", side_effect=fake_run),
                    mock.patch.object(launcher.sys, "executable", PYTHON),
                    mock.patch.object(launcher.Path, "is_file", return_value=True),
                    mock.patch.object(launcher.os, "access", return_value=True),
                    mock.patch.object(
                        launcher.secrets, "token_hex", return_value="cd" * 16
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = launcher.main(
                        ["--source-commit", SOURCE, "--run-kind", "smoke"]
                    )
                self.assertEqual(128 + signum, code)
                self.assertEqual(
                    [
                        "/usr/bin/systemctl",
                        "--user",
                        "stop",
                        f"{service_name}.service",
                    ],
                    managed_calls[-1][0],
                )
                self.assertEqual(
                    handlers_before,
                    {
                        observed: signal.getsignal(observed)
                        for observed in (signal.SIGINT, signal.SIGTERM)
                    },
                )

    def test_python_mismatch_refuses_before_systemd_launch(self):
        candidate_bytes = json.dumps(candidate()).encode("utf-8")
        calls: list[list[str]] = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(
                command, 0, stdout=candidate_bytes, stderr=b""
            )

        with (
            mock.patch.object(launcher.subprocess, "run", side_effect=fake_run),
            mock.patch.object(launcher.sys, "executable", "/usr/bin/python3"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                launcher.main(["--source-commit", SOURCE, "--run-kind", "full"])
        self.assertEqual(2, raised.exception.code)
        self.assertEqual(1, len(calls))
        self.assertEqual("/usr/bin/git", calls[0][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
