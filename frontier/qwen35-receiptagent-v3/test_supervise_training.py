from __future__ import annotations

import contextlib
import copy
import errno
import importlib.util
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = "a" * 40
RUN_ID = "b" * 32
UNIT_NAME = f"szl-ra3-supervisor-{RUN_ID}"
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


def load_supervisor():
    here_text = str(HERE)
    if here_text not in sys.path:
        sys.path.insert(0, here_text)
    spec = importlib.util.spec_from_file_location(
        "v3_training_supervisor_under_test", HERE / "supervise_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


supervisor = load_supervisor()


def candidate() -> dict:
    return {
        "supervision_policy": copy.deepcopy(supervisor.EXPECTED_POLICY),
        "training_recipe": {
            "maximum_gpu_temperature_c": 80,
            "minimum_free_gpu_gib": 4.0,
        },
    }


def sample(
    *,
    temperature_c: int = 80,
    free_mib: int = 4096,
    gpu_uuid: str = GPU_UUID,
) -> object:
    return supervisor.TelemetrySample(
        observed_monotonic_ns=1,
        observed_at="2026-08-13T12:00:00+00:00",
        gpu_uuid=gpu_uuid,
        name="NVIDIA Test GPU",
        temperature_c=temperature_c,
        free_mib=free_mib,
        total_mib=8192,
    )


class StrictArgumentTests(unittest.TestCase):
    def test_exact_isolated_python_can_load_supervisor_and_reach_parser(self):
        if not pathlib.Path(supervisor.EXPECTED_POLICY["python_executable"]).exists():
            self.skipTest("exact WSL training Python is unavailable on this platform")
        observed = subprocess.run(
            [
                supervisor.EXPECTED_POLICY["python_executable"],
                "-I",
                "-B",
                str(HERE / "supervise_training.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, observed.returncode, observed.stderr)
        self.assertIn("--source-commit", observed.stdout)

    def assert_rejected(self, argv: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                supervisor.strict_args(argv)

    def test_accepts_exactly_one_value_for_each_public_option(self):
        args = supervisor.strict_args(
            [
                "--source-commit",
                SOURCE,
                "--run-kind",
                "full",
                "--unit-name",
                UNIT_NAME,
            ]
        )
        self.assertEqual(SOURCE, args.source_commit)
        self.assertEqual("full", args.run_kind)
        self.assertEqual(UNIT_NAME, args.unit_name)
        self.assertEqual(RUN_ID, args.run_id)

    def test_rejects_duplicate_and_unknown_options(self):
        valid = [
            "--source-commit",
            SOURCE,
            "--run-kind",
            "smoke",
            "--unit-name",
            UNIT_NAME,
        ]
        duplicate_cases = (
            ["--source-commit", SOURCE, *valid],
            ["--run-kind", "full", *valid],
            ["--unit-name", UNIT_NAME, *valid],
        )
        for argv in duplicate_cases:
            with self.subTest(argv=argv):
                self.assert_rejected(argv)
        self.assert_rejected([*valid, "--unknown", "value"])
        self.assert_rejected(
            [
                "--source",
                SOURCE,
                "--run-kind",
                "smoke",
                "--unit-name",
                UNIT_NAME,
            ]
        )


class FixedPolicyAndTelemetryTests(unittest.TestCase):
    def test_policy_requires_the_exact_mapping_and_fixed_recipe_thresholds(self):
        expected = candidate()
        self.assertEqual(
            supervisor.EXPECTED_POLICY,
            supervisor.validate_policy(expected),
        )

        for field, value in supervisor.EXPECTED_POLICY.items():
            drifted = candidate()
            if isinstance(value, str):
                drifted["supervision_policy"][field] = value + "-drift"
            else:
                drifted["supervision_policy"][field] = value + 1
            with self.subTest(drifted_field=field):
                with self.assertRaises(supervisor.SupervisionError):
                    supervisor.validate_policy(drifted)

        missing = candidate()
        missing["supervision_policy"].pop("maximum_telemetry_gap_seconds")
        with self.assertRaises(supervisor.SupervisionError):
            supervisor.validate_policy(missing)

        extra = candidate()
        extra["supervision_policy"]["caller_override"] = True
        with self.assertRaises(supervisor.SupervisionError):
            supervisor.validate_policy(extra)

        for recipe_field, drifted_value in (
            ("maximum_gpu_temperature_c", 81),
            ("minimum_free_gpu_gib", 3.99),
        ):
            drifted = candidate()
            drifted["training_recipe"][recipe_field] = drifted_value
            with self.subTest(drifted_recipe_field=recipe_field):
                with self.assertRaises(supervisor.SupervisionError):
                    supervisor.validate_policy(drifted)

    def test_temperature_80_passes_and_81_aborts(self):
        recipe = candidate()["training_recipe"]
        supervisor.initial_temperature_gate(sample(temperature_c=80), recipe)
        self.assertIsNone(
            supervisor.sample_trigger(
                sample(temperature_c=80),
                expected_gpu_uuid=GPU_UUID,
                maximum_temperature_c=80,
            )
        )

        with self.assertRaisesRegex(supervisor.SupervisionError, "81 C exceeds 80 C"):
            supervisor.initial_temperature_gate(sample(temperature_c=81), recipe)
        self.assertEqual(
            (
                "THERMAL_POLICY_VIOLATION",
                "GPU temperature 81 C exceeded fixed 80 C",
            ),
            supervisor.sample_trigger(
                sample(temperature_c=81),
                expected_gpu_uuid=GPU_UUID,
                maximum_temperature_c=80,
            ),
        )

    def test_gpu_uuid_drift_is_telemetry_failure(self):
        self.assertEqual(
            ("TELEMETRY_UNAVAILABLE", "GPU UUID changed during the run"),
            supervisor.sample_trigger(
                sample(gpu_uuid="GPU-different"),
                expected_gpu_uuid=GPU_UUID,
                maximum_temperature_c=80,
            ),
        )

    def test_initial_four_gib_gate_is_inclusive_and_one_mib_below_fails(self):
        recipe = candidate()["training_recipe"]
        supervisor.initial_temperature_gate(sample(free_mib=4096), recipe)
        with self.assertRaisesRegex(
            supervisor.SupervisionError, "below the fixed policy floor"
        ):
            supervisor.initial_temperature_gate(sample(free_mib=4095), recipe)

    def test_report_reserve_exceeds_the_derived_full_run_bound(self):
        policy = supervisor.EXPECTED_POLICY
        self.assertEqual(5_402, supervisor.maximum_telemetry_samples(policy))
        self.assertLessEqual(
            supervisor.minimum_evidence_reserve_bytes(policy),
            policy["evidence_reserve_bytes"],
        )
        worst_sample = supervisor.TelemetrySample(
            observed_monotonic_ns=10_800_000_000_000,
            observed_at="9999-12-31T23:59:59.999999+00:00",
            gpu_uuid="GPU-" + "f" * 92,
            name="N" * supervisor.MAX_GPU_NAME_CHARACTERS,
            temperature_c=supervisor.MAX_GPU_TEMPERATURE_C,
            free_mib=supervisor.MAX_GPU_MEMORY_MIB,
            total_mib=supervisor.MAX_GPU_MEMORY_MIB,
        ).public(0)
        self.assertLessEqual(
            len(supervisor.canonical_json(worst_sample).encode("utf-8")),
            supervisor.MAX_TELEMETRY_SAMPLE_JSON_BYTES,
        )
        telemetry_bytes = len(
            supervisor.canonical_json(
                {
                    "samples": [
                        worst_sample
                        for _ in range(supervisor.maximum_telemetry_samples(policy))
                    ]
                }
            ).encode("utf-8")
        )
        self.assertLessEqual(
            telemetry_bytes + supervisor.MAX_NON_TELEMETRY_REPORT_BYTES,
            policy["evidence_reserve_bytes"],
        )

    def test_terminal_observation_preserves_the_first_trigger(self):
        self.assertEqual(
            (
                "WALL_TIMEOUT",
                "fixed deadline; terminal=GPU temperature 81 C exceeded fixed 80 C",
            ),
            supervisor.merge_terminal_observation(
                "WALL_TIMEOUT",
                "fixed deadline",
                (
                    "THERMAL_POLICY_VIOLATION",
                    "GPU temperature 81 C exceeded fixed 80 C",
                ),
            ),
        )
        self.assertEqual(
            ("TELEMETRY_UNAVAILABLE", "terminal sample failed"),
            supervisor.merge_terminal_observation(
                "WORKER_EXIT_FAILURE",
                None,
                ("TELEMETRY_UNAVAILABLE", "terminal sample failed"),
            ),
        )


class EvidenceAndAdmissionTests(unittest.TestCase):
    def test_component_binding_rejects_validator_worktree_drift(self):
        exact = b"exact validator source\n"
        with tempfile.TemporaryDirectory() as directory:
            local_here = pathlib.Path(directory)
            (local_here / "supervisor_validation.py").write_bytes(exact + b"drift")
            with (
                mock.patch.object(supervisor, "HERE", local_here),
                mock.patch.object(
                    supervisor.trainer, "committed_bytes", return_value=exact
                ),
            ):
                with self.assertRaisesRegex(
                    supervisor.SupervisionError, "worktree bytes differ"
                ):
                    supervisor.bind_committed_component(
                        SOURCE, "supervisor_validation.py"
                    )

    def test_strict_json_rejects_duplicate_keys_at_any_depth(self):
        payloads = (
            b'{"state":"first","state":"second"}',
            b'{"outer":{"state":"first","state":"second"}}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            for payload in payloads:
                with self.subTest(payload=payload):
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(
                        supervisor.DuplicateKeyError, "duplicate JSON key: state"
                    ):
                        supervisor.strict_json_file(path)

    def test_existing_empty_attempt_leaf_is_refused_and_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_root = pathlib.Path(directory)
            leaf = runs_root / RUN_ID
            leaf.mkdir(mode=0o700)
            before = leaf.stat()
            before_entries = list(leaf.iterdir())

            root_fd = 123
            collision = FileExistsError(errno.EEXIST, "already exists", RUN_ID)
            with (
                mock.patch.object(supervisor.os, "O_DIRECTORY", 0, create=True),
                mock.patch.object(supervisor.os, "open", return_value=root_fd),
                mock.patch.object(
                    supervisor.os, "mkdir", side_effect=collision
                ) as mkdir,
                mock.patch.object(supervisor.os, "close") as close,
                mock.patch.object(supervisor.os, "fsync") as fsync,
            ):
                with self.assertRaises(FileExistsError):
                    supervisor.admit_attempt(runs_root, RUN_ID, 1024)

            mkdir.assert_called_once_with(RUN_ID, mode=0o700, dir_fd=root_fd)
            close.assert_called_once_with(root_fd)
            fsync.assert_not_called()
            after = leaf.stat()
            self.assertEqual(before_entries, list(leaf.iterdir()))
            self.assertEqual(before.st_ino, after.st_ino)
            self.assertEqual(before.st_mode, after.st_mode)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual([RUN_ID], [entry.name for entry in runs_root.iterdir()])

    def test_publish_once_never_clobbers_an_existing_final(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = pathlib.Path(directory)
            final = reports / "supervisor-report.json"
            original = b"immutable-existing-report"
            final.write_bytes(original)
            before = final.stat()
            with (
                mock.patch.object(supervisor.os, "fchmod", create=True),
                mock.patch.object(supervisor, "fsync_directory"),
            ):
                with self.assertRaises(FileExistsError):
                    supervisor.publish_once(
                        reports, "supervisor-report.json", b"replacement"
                    )
            after = final.stat()
            self.assertEqual(original, final.read_bytes())
            self.assertEqual(before.st_ino, after.st_ino)
            self.assertEqual(before.st_size, after.st_size)
            self.assertEqual([], list(reports.glob(".*.tmp")))

    def test_publish_once_never_clobbers_an_existing_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = pathlib.Path(directory)
            target = reports / "target.json"
            target.write_bytes(b"symlink-target")
            final = reports / "supervisor-report.json"
            try:
                final.symlink_to(target)
            except OSError:
                # Windows without Developer Mode cannot create an unprivileged
                # symlink. Model the kernel's EEXIST response and assert the
                # publisher neither unlinks nor replaces the destination.
                final.write_bytes(b"simulated-symlink-entry")
                with (
                    mock.patch.object(supervisor.os, "fchmod", create=True),
                    mock.patch.object(supervisor, "fsync_directory"),
                    mock.patch.object(
                        supervisor.os,
                        "link",
                        side_effect=FileExistsError(
                            errno.EEXIST, "symlink destination exists", str(final)
                        ),
                    ) as link,
                ):
                    with self.assertRaises(FileExistsError):
                        supervisor.publish_once(
                            reports, "supervisor-report.json", b"replacement"
                        )
                self.assertEqual(b"simulated-symlink-entry", final.read_bytes())
                self.assertEqual(b"symlink-target", target.read_bytes())
                self.assertFalse(final.is_symlink())
                self.assertFalse(link.call_args.kwargs["follow_symlinks"])
            else:
                with (
                    mock.patch.object(supervisor.os, "fchmod", create=True),
                    mock.patch.object(supervisor, "fsync_directory"),
                ):
                    with self.assertRaises(FileExistsError):
                        supervisor.publish_once(
                            reports, "supervisor-report.json", b"replacement"
                        )
                self.assertTrue(final.is_symlink())
                self.assertEqual(target.resolve(), final.resolve())
                self.assertEqual(b"symlink-target", target.read_bytes())
            self.assertEqual([], list(reports.glob(".*.tmp")))


class WorkerEnvironmentTests(unittest.TestCase):
    FORBIDDEN_KEYS = {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
    }

    def test_worker_environment_is_minimal_and_launch_uses_env_i(self):
        root = pathlib.Path("/isolated/run")
        attempt = supervisor.Attempt(
            run_id=RUN_ID,
            root=root,
            payload=root / "payload",
            logs=root / "logs",
            reports=root / "reports",
            runtime_cache=root / "runtime-cache",
            reserve=root / "reserve.bin",
        )
        exact_environment = supervisor.worker_environment(attempt)
        observed_keys = {key.upper() for key in exact_environment}
        self.assertTrue(set(supervisor.WORKER_ENVIRONMENT) <= set(exact_environment))
        self.assertTrue(self.FORBIDDEN_KEYS.isdisjoint(observed_keys))
        self.assertFalse(
            {
                key
                for key in observed_keys
                if key.endswith("_TOKEN") or key.endswith("_SECRET")
            }
        )

        properties = {
            "ActiveState": "active",
            "SubState": "running",
            "KillMode": "control-group",
            "SendSIGKILL": "yes",
            "RemainAfterExit": "yes",
            "NoNewPrivileges": "yes",
            "ProtectControlGroups": "yes",
            "ProtectSystem": "strict",
            "ProtectHome": "read-only",
            "PrivateTmp": "yes",
            "PrivateNetwork": "yes",
            "RestrictSUIDSGID": "yes",
            "RestrictNamespaces": "yes",
            "BindsTo": "outer.service",
            "ControlGroup": "/user.slice/worker.service",
            "MainPID": "123",
            "ExecMainPID": "123",
            "ExecMainStatus": "0",
            "Result": "success",
        }
        worker_argv = ["/trusted/python", "-I", "-B", "/trusted/worker.py"]
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        with (
            mock.patch.object(
                supervisor.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
            mock.patch.object(supervisor, "unit_properties", return_value=properties),
            mock.patch("pathlib.Path.read_text", return_value="123\n"),
        ):
            supervisor.launch_worker_unit(
                policy,
                outer_unit="outer",
                worker_unit="worker",
                worker_argv=worker_argv,
                worker_environment_values=exact_environment,
                stdout_path=pathlib.Path("/tmp/run/logs/stdout.log"),
                stderr_path=pathlib.Path("/tmp/run/logs/stderr.log"),
            )

        command = run.call_args.args[0]
        separator = command.index("--")
        environment_start = separator + 1
        self.assertEqual(
            ["/usr/bin/env", "-i"],
            command[environment_start : environment_start + 2],
        )
        assignments_start = environment_start + 2
        assignments_end = assignments_start + len(exact_environment)
        self.assertEqual(
            [f"{key}={value}" for key, value in sorted(exact_environment.items())],
            command[assignments_start:assignments_end],
        )
        self.assertEqual(worker_argv, command[assignments_end:])
        launched_keys = {
            assignment.split("=", 1)[0].upper()
            for assignment in command[assignments_start:assignments_end]
        }
        self.assertTrue(self.FORBIDDEN_KEYS.isdisjoint(launched_keys))

    def test_fast_exit_is_accepted_with_retained_systemd_identity(self):
        properties = {
            "ActiveState": "active",
            "SubState": "exited",
            "KillMode": "control-group",
            "SendSIGKILL": "yes",
            "RemainAfterExit": "yes",
            "NoNewPrivileges": "yes",
            "ProtectControlGroups": "yes",
            "ProtectSystem": "strict",
            "ProtectHome": "read-only",
            "PrivateTmp": "yes",
            "PrivateNetwork": "yes",
            "RestrictSUIDSGID": "yes",
            "RestrictNamespaces": "yes",
            "BindsTo": "outer.service",
            "ControlGroup": "/user.slice/worker.service",
            "MainPID": "0",
            "ExecMainPID": "456",
            "ExecMainStatus": "70",
            "Result": "exit-code",
        }
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        with (
            mock.patch.object(
                supervisor.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            mock.patch.object(supervisor, "unit_properties", return_value=properties),
        ):
            observed = supervisor.launch_worker_unit(
                policy,
                outer_unit="outer",
                worker_unit="worker",
                worker_argv=["/trusted/python"],
                worker_environment_values={},
                stdout_path=pathlib.Path("/tmp/run/logs/stdout.log"),
                stderr_path=pathlib.Path("/tmp/run/logs/stderr.log"),
            )
        self.assertEqual("70", observed["ExecMainStatus"])
        self.assertEqual("exit-code", observed["Result"])

    def test_retained_status_refresh_records_post_stop_result(self):
        retained = {
            "ExecMainPID": "456",
            "ExecMainStatus": "15",
            "Result": "signal",
        }
        with mock.patch.object(
            supervisor, "unit_properties", return_value=retained
        ) as properties:
            observed = supervisor.retained_worker_properties(
                supervisor.EXPECTED_POLICY, "worker"
            )
        properties.assert_called_once_with(supervisor.EXPECTED_POLICY, "worker.service")
        self.assertEqual("15", observed["ExecMainStatus"])
        self.assertEqual("signal", observed["Result"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
