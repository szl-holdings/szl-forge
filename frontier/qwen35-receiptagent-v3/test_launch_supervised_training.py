from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import signal
import stat
import subprocess
import tempfile
import types
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


def supervisor_report(
    run_id: str,
    run_kind: str,
    source_commit: str,
    *,
    state: str | None = None,
    cause: str = "SUCCESS",
    schema: str = "szl.frontier-training-supervisor/v1",
    receipt_eligible: bool = False,
) -> bytes:
    report = {
        "schema": schema,
        "runId": run_id,
        "runKind": run_kind.upper(),
        "source": {"revision": source_commit},
        "primaryCause": cause,
        "state": state or launcher.SUPERVISOR_SUCCESS_STATE_BY_KIND[run_kind],
        "receiptEligible": receipt_eligible,
        "publicationEligible": False,
        "autonomyEligible": False,
    }
    def canonical(value):
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    report["reportSha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return (canonical(report) + "\n").encode()


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

    def test_component_preflight_uses_fixed_git_and_rejects_local_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            committed = {
                filename: f"exact:{filename}\n".encode("utf-8")
                for filename in launcher.SUPERVISED_EXECUTABLE_COMPONENTS
            }
            for filename, data in committed.items():
                (root / filename).write_bytes(data)
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                filename = command[-1].rsplit("/", 1)[-1]
                self.assertEqual(
                    [
                        "/usr/bin/git",
                        "show",
                        f"{SOURCE}:{launcher.RELATIVE}/{filename}",
                    ],
                    command,
                )
                self.assertEqual(root, kwargs["cwd"])
                self.assertIs(kwargs["shell"], False)
                self.assertIs(kwargs["check"], False)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=committed[filename],
                    stderr=b"",
                )

            with mock.patch.object(launcher.subprocess, "run", side_effect=fake_run):
                launcher.verify_local_components(
                    SOURCE,
                    component_dir=root,
                    repo_root=root,
                )
                (root / "containment_probe.py").write_bytes(b"tampered\n")
                with self.assertRaisesRegex(
                    launcher.LauncherError,
                    "containment_probe.py",
                ):
                    launcher.verify_local_components(
                        SOURCE,
                        component_dir=root,
                        repo_root=root,
                    )
            self.assertGreaterEqual(
                len(calls),
                len(launcher.SUPERVISED_EXECUTABLE_COMPONENTS) + 4,
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
            mock.patch.object(launcher, "verify_local_components") as verify_components,
            mock.patch.object(launcher, "require_native_interop", return_value="/run/WSL/123_interop"),
            mock.patch.object(launcher.secrets, "token_hex", return_value="ef" * 16),
            mock.patch.object(launcher.sys, "stdout", output),
            mock.patch.object(
                launcher,
                "record_missing_supervisor_report",
                return_value="NO_ATTEMPT",
            ),
        ):
            code = launcher.main(["--source-commit", SOURCE, "--run-kind", "full"])

        self.assertEqual(73, code)
        verify_components.assert_called_once_with(SOURCE)
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
                    mock.patch.object(launcher, "verify_local_components"),
                    mock.patch.object(launcher, "require_native_interop", return_value="/run/WSL/123_interop"),
                    mock.patch.object(
                        launcher.secrets, "token_hex", return_value="cd" * 16
                    ),
                    mock.patch.object(
                        launcher,
                        "record_missing_supervisor_report",
                        return_value="NO_ATTEMPT",
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


class MissingReportObservationTests(unittest.TestCase):
    def test_exact_attempt_gets_false_launcher_observation_only_when_report_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory)
            run_id = "ab" * 16
            service_name = f"szl-ra3-supervisor-{run_id}"
            attempt = runs_root / run_id
            reports = attempt / "reports"
            reports.mkdir(parents=True)
            real_lstat = os.lstat
            owner_uid = real_lstat(reports).st_uid

            def controlled_lstat(path):
                if pathlib.Path(path) in (runs_root, attempt, reports):
                    real_lstat(path)
                    return types.SimpleNamespace(
                        st_mode=stat.S_IFDIR | 0o700,
                        st_uid=owner_uid,
                    )
                return real_lstat(path)

            arguments = {
                "service_name": service_name,
                "run_id": run_id,
                "attempt_path": str(attempt),
                "source_commit": SOURCE,
                "run_kind": "smoke",
                "systemd_exit_code": 143,
                "launcher_signal": None,
                "outer_stop_error": None,
            }
            with (
                mock.patch.object(launcher.os, "lstat", side_effect=controlled_lstat),
                mock.patch.object(launcher.os, "getuid", return_value=owner_uid, create=True),
                mock.patch.object(
                    launcher,
                    "publish_launcher_observation",
                    return_value=reports / "launcher-stop-observation.json",
                ) as publish,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                observed = launcher.record_missing_supervisor_report(**arguments)
                self.assertEqual("LAUNCHER_OBSERVATION_WRITTEN", observed)
                published_reports, payload = publish.call_args.args
                self.assertEqual(reports, published_reports)
                self.assertEqual(run_id, payload["runId"])
                self.assertEqual(SOURCE, payload["sourceCommit"])
                self.assertEqual(143, payload["systemdRunExitCode"])
                self.assertIs(payload["supervisorReportPresent"], False)
                self.assertIs(payload["trainingCompletionObserved"], False)
                for field in (
                    "receiptEligible",
                    "publicationEligible",
                    "autonomyEligible",
                ):
                    self.assertIs(payload[field], False)

                publish.reset_mock()
                report_path = reports / "supervisor-report.json"
                report_path.write_bytes(supervisor_report(run_id, "smoke", SOURCE))
                self.assertEqual(
                    "SUPERVISOR_REPORT_SUCCESS",
                    launcher.record_missing_supervisor_report(**arguments),
                )
                publish.assert_not_called()

                report_path.write_bytes(
                    supervisor_report(
                        run_id,
                        "smoke",
                        SOURCE,
                        state="SUPERVISOR_TERMINATED_RUN_NO_COMPLETION_CLAIM",
                        cause="SUPERVISOR_STOP_REQUESTED",
                    )
                )
                self.assertEqual(
                    "SUPERVISOR_REPORTED_NON_SUCCESS",
                    launcher.record_missing_supervisor_report(**arguments),
                )
                publish.assert_not_called()

                report_path.unlink()
                reports.rmdir()
                self.assertEqual(
                    "NO_REPORTS_DIRECTORY",
                    launcher.record_missing_supervisor_report(**arguments),
                )
                publish.assert_not_called()

            with self.assertRaisesRegex(launcher.LauncherError, "identity"):
                launcher.record_missing_supervisor_report(
                    **{**arguments, "run_id": "cd" * 16}
                )

    def test_bounded_report_requires_exact_identity_integrity_and_success_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "supervisor-report.json"
            run_id = "ab" * 16
            for kind in ("smoke", "full"):
                with self.subTest(kind=kind):
                    path.write_bytes(supervisor_report(run_id, kind, SOURCE))
                    self.assertEqual(
                        "SUPERVISOR_REPORT_SUCCESS",
                        launcher.classify_supervisor_report(
                            path, run_id=run_id, run_kind=kind, source_commit=SOURCE
                        ),
                    )
            cases = (
                (supervisor_report("cd" * 16, "smoke", SOURCE), "SUPERVISOR_REPORT_IDENTITY_MISMATCH"),
                (supervisor_report(run_id, "smoke", "b" * 40), "SUPERVISOR_REPORT_IDENTITY_MISMATCH"),
                (supervisor_report(run_id, "full", SOURCE), "SUPERVISOR_REPORT_IDENTITY_MISMATCH"),
                (
                    supervisor_report(run_id, "smoke", SOURCE, schema="wrong/v1"),
                    "SUPERVISOR_REPORT_IDENTITY_MISMATCH",
                ),
                (
                    supervisor_report(run_id, "smoke", SOURCE, receipt_eligible=True),
                    "SUPERVISOR_REPORT_CLAIM_BOUNDARY_INVALID",
                ),
                (
                    supervisor_report(
                        run_id,
                        "smoke",
                        SOURCE,
                        state="SUPERVISOR_TERMINATED_RUN_NO_COMPLETION_CLAIM",
                        cause="SUPERVISOR_STOP_REQUESTED",
                    ),
                    "SUPERVISOR_REPORTED_NON_SUCCESS",
                ),
                (
                    supervisor_report(
                        run_id,
                        "smoke",
                        SOURCE,
                        state=launcher.SUPERVISOR_SUCCESS_STATE_BY_KIND["full"],
                    ),
                    "SUPERVISOR_REPORTED_NON_SUCCESS",
                ),
                (b"{}\n", "SUPERVISOR_REPORT_INVALID"),
                (b'{"runId":"a","runId":"b"}\n', "SUPERVISOR_REPORT_INVALID"),
                (b"x" * (launcher.MAX_SUPERVISOR_REPORT_BYTES + 1), "SUPERVISOR_REPORT_INVALID"),
            )
            for data, expected in cases:
                with self.subTest(expected=expected, prefix=data[:30]):
                    path.write_bytes(data)
                    self.assertEqual(
                        expected,
                        launcher.classify_supervisor_report(
                            path, run_id=run_id, run_kind="smoke", source_commit=SOURCE
                        ),
                    )

    @unittest.skipUnless(os.name == "posix", "write-once publisher requires POSIX")
    def test_launcher_observation_is_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = pathlib.Path(directory)
            payload = {"schema": "test", "receiptEligible": False}
            path = launcher.publish_launcher_observation(reports, payload)
            original = path.read_bytes()
            with self.assertRaises(Exception):
                launcher.publish_launcher_observation(
                    reports, {"schema": "replacement", "receiptEligible": True}
                )
            self.assertEqual(original, path.read_bytes())

    def test_success_exit_without_supervisor_report_is_withheld(self):
        run_id = "ab" * 16
        service_name = f"szl-ra3-supervisor-{run_id}"
        for observation_state, expected in (
            ("LAUNCHER_OBSERVATION_WRITTEN", 76),
            ("NO_ATTEMPT", 76),
            ("SUPERVISOR_REPORTED_NON_SUCCESS", 76),
            ("SUPERVISOR_REPORT_INVALID", 76),
            ("SUPERVISOR_REPORT_IDENTITY_MISMATCH", 76),
            ("SUPERVISOR_REPORT_SUCCESS", 0),
        ):
            with self.subTest(observation_state=observation_state):
                with (
                    mock.patch.object(
                        launcher.subprocess,
                        "run",
                        side_effect=lambda command, **_kwargs: subprocess.CompletedProcess(
                            command, 0
                        ),
                    ),
                    mock.patch.object(
                        launcher,
                        "record_missing_supervisor_report",
                        return_value=observation_state,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    observed = launcher.invoke_with_cleanup(
                        [launcher.SYSTEMD_RUN],
                        service_name=service_name,
                        run_id=run_id,
                        attempt_path=f"{RUNS_ROOT}/{run_id}",
                        source_commit=SOURCE,
                        run_kind="smoke",
                    )
                self.assertEqual(expected, observed)

    def test_outer_stop_budget_exceeds_supervisor_worker_cleanup(self):
        self.assertIn("TimeoutStopSec=90s", launcher.SYSTEMD_PROPERTIES)
        self.assertGreater(launcher.CLEANUP_TIMEOUT_SECONDS, 90)


if __name__ == "__main__":
    unittest.main(verbosity=2)
