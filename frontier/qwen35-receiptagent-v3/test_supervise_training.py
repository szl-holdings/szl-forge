from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import inspect
import io
import os
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


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "v3_training_worker_under_test", HERE / "train_candidate.py"
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

    def test_distinct_samplers_use_one_exact_query_and_fixed_timeouts(self):
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                f"{GPU_UUID}, NVIDIA Test GPU, 55, 7000, 8192\n".encode()
            ),
            stderr=b"",
        )
        with mock.patch.object(
            supervisor.subprocess, "run", return_value=completed
        ) as run:
            admission = supervisor.sample_gpu_for_admission(
                supervisor.EXPECTED_POLICY
            )
            confirmation = supervisor.sample_gpu_for_runtime(
                supervisor.EXPECTED_POLICY
            )
        self.assertEqual(GPU_UUID, admission.gpu_uuid)
        self.assertEqual(GPU_UUID, confirmation.gpu_uuid)
        self.assertEqual(2, run.call_count)
        expected_command = [
            supervisor.EXPECTED_POLICY["nvidia_smi_executable"],
            *supervisor.NVIDIA_SMI_QUERY,
        ]
        self.assertEqual(
            [expected_command, expected_command],
            [call.args[0] for call in run.call_args_list],
        )
        self.assertEqual(
            [15.0, 5.0],
            [call.kwargs["timeout"] for call in run.call_args_list],
        )
        for call in run.call_args_list:
            self.assertTrue(call.kwargs["check"])
            self.assertTrue(call.kwargs["capture_output"])

    def test_sampler_timeout_fails_once_without_retry_or_fallback(self):
        for sampler in (
            supervisor.sample_gpu_for_admission,
            supervisor.sample_gpu_for_runtime,
        ):
            with self.subTest(sampler=sampler.__name__):
                with mock.patch.object(
                    supervisor.subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired("nvidia-smi", 1),
                ) as run:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        sampler(supervisor.EXPECTED_POLICY)
                run.assert_called_once()

    def test_readiness_pair_gates_both_samples_and_requires_stable_uuid(self):
        recipe = candidate()["training_recipe"]
        supervisor.readiness_pair_gate(sample(), sample(), recipe)
        for admission, confirmation, error in (
            (
                sample(free_mib=4095),
                sample(),
                "below the fixed policy floor",
            ),
            (
                sample(),
                sample(free_mib=4095),
                "below the fixed policy floor",
            ),
            (
                sample(),
                sample(gpu_uuid="GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                "changed between readiness phases",
            ),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(supervisor.SupervisionError, error):
                    supervisor.readiness_pair_gate(admission, confirmation, recipe)

    def test_readiness_evidence_distinguishes_observed_rejected_and_not_run(self):
        observed = supervisor.telemetry_phase_evidence(
            phase="ADMISSION_READINESS",
            timeout_seconds=15.0,
            started_monotonic_ns=1_000_000_000,
            completed_monotonic_ns=5_250_000_000,
            origin_monotonic_ns=1_000_000_000,
            sample=sample(),
            error=None,
        )
        rejected = supervisor.telemetry_phase_evidence(
            phase="ADMISSION_READINESS",
            timeout_seconds=15.0,
            started_monotonic_ns=1_000_000_000,
            completed_monotonic_ns=2_000_000_000,
            origin_monotonic_ns=1_000_000_000,
            sample=sample(free_mib=4095),
            error="initial free GPU memory is below the fixed policy floor",
        )
        not_run = supervisor.telemetry_phase_not_run_evidence(
            phase="RUNTIME_CONFIRMATION",
            timeout_seconds=5.0,
            reason="ADMISSION_READINESS_NOT_SATISFIED",
        )
        self.assertEqual(
            ("OBSERVED", 4.25, 15.0),
            (
                observed["state"],
                observed["durationSeconds"],
                observed["timeoutSeconds"],
            ),
        )
        self.assertEqual("REJECTED", rejected["state"])
        self.assertEqual("NOT_RUN", not_run["state"])
        self.assertIsNone(not_run["sample"])

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
        self.assertIsNone(
            supervisor.readiness_rejection_cause(sample(free_mib=4096), recipe)
        )
        self.assertEqual(
            "PRECONDITION_DENIED",
            supervisor.readiness_rejection_cause(sample(free_mib=4095), recipe),
        )
        self.assertEqual(
            "THERMAL_POLICY_VIOLATION",
            supervisor.readiness_rejection_cause(sample(temperature_c=81), recipe),
        )
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
            telemetry_bytes
            + supervisor.TELEMETRY_READINESS_SAMPLE_COPIES
            * supervisor.MAX_TELEMETRY_SAMPLE_JSON_BYTES
            + supervisor.MAX_NON_TELEMETRY_REPORT_BYTES,
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

    def test_slow_successful_sample_fails_closed_after_command_completion(self):
        previous_ns = 1_000_000_000
        at_limit = sample()
        at_limit = supervisor.TelemetrySample(
            observed_monotonic_ns=previous_ns + 8_000_000_000,
            observed_at=at_limit.observed_at,
            gpu_uuid=at_limit.gpu_uuid,
            name=at_limit.name,
            temperature_c=at_limit.temperature_c,
            free_mib=at_limit.free_mib,
            total_mib=at_limit.total_mib,
        )
        late = supervisor.TelemetrySample(
            observed_monotonic_ns=at_limit.observed_monotonic_ns + 1,
            observed_at=at_limit.observed_at,
            gpu_uuid=at_limit.gpu_uuid,
            name=at_limit.name,
            temperature_c=at_limit.temperature_c,
            free_mib=at_limit.free_mib,
            total_mib=at_limit.total_mib,
        )
        self.assertIsNone(
            supervisor.successful_telemetry_gap_trigger(previous_ns, at_limit, 8.0)
        )
        self.assertEqual(
            (
                "TELEMETRY_UNAVAILABLE",
                "GPU telemetry exceeded the maximum allowed gap after a successful sample",
            ),
            supervisor.successful_telemetry_gap_trigger(previous_ns, late, 8.0),
        )

    def test_sampled_drain_requires_stop_for_hot_lingering_child(self):
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        hot = sample(temperature_c=81)
        hot = supervisor.TelemetrySample(
            observed_monotonic_ns=3_000_000_000,
            observed_at=hot.observed_at,
            gpu_uuid=hot.gpu_uuid,
            name=hot.name,
            temperature_c=hot.temperature_c,
            free_mib=hot.free_mib,
            total_mib=hot.total_mib,
        )
        with (
            mock.patch.object(supervisor, "cgroup_empty", return_value=False),
            mock.patch.object(
                supervisor, "sample_gpu_for_runtime", return_value=hot
            ),
            mock.patch.object(
                supervisor.time,
                "monotonic_ns",
                side_effect=(2_000_000_000, 3_000_000_000),
            ),
        ):
            result = supervisor.sampled_cgroup_drain(
                policy,
                "/user.slice/worker.service",
                expected_gpu_uuid=GPU_UUID,
                maximum_temperature_c=80,
                last_valid_monotonic_ns=1_000_000_000,
            )
        self.assertEqual("THERMAL_POLICY_VIOLATION", result.trigger[0])
        self.assertTrue(result.stop_required)
        self.assertFalse(result.cgroup_empty_confirmed)
        self.assertEqual((hot,), result.samples)

    def test_sample_gpu_clamps_subprocess_to_remaining_deadline(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{GPU_UUID}, Test GPU, 60, 4096, 8192\n".encode(),
            stderr=b"",
        )
        for remaining, expected in ((0.75, 0.75), (10.0, 5.0)):
            with self.subTest(remaining=remaining):
                with mock.patch.object(
                    supervisor.subprocess, "run", return_value=completed
                ) as run:
                    supervisor.sample_gpu_for_runtime(
                        supervisor.EXPECTED_POLICY,
                        timeout_seconds=remaining,
                    )
                self.assertEqual(expected, run.call_args.kwargs["timeout"])

    def test_sample_gpu_rejects_expired_deadline_before_subprocess(self):
        for remaining in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(remaining=remaining):
                with (
                    mock.patch.object(supervisor.subprocess, "run") as run,
                    self.assertRaisesRegex(
                        supervisor.SupervisionError, "telemetry deadline has expired"
                    ),
                ):
                    supervisor.sample_gpu_for_runtime(
                        supervisor.EXPECTED_POLICY,
                        timeout_seconds=remaining,
                    )
                run.assert_not_called()

    def test_sampled_drain_clamps_to_drain_and_telemetry_gap_deadlines(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{GPU_UUID}, Test GPU, 81, 4096, 8192\n".encode(),
            stderr=b"",
        )
        for drain_seconds, previous_ns, expected in (
            (1.5, 1_000_000_000, 0.5),
            (10.0, -4_250_000_000, 0.75),
        ):
            with self.subTest(drain_seconds=drain_seconds):
                policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
                policy["kill_confirmation_seconds"] = drain_seconds
                with (
                    mock.patch.object(supervisor, "cgroup_empty", return_value=False),
                    mock.patch.object(
                        supervisor.subprocess, "run", return_value=completed
                    ) as run,
                    mock.patch.object(
                        supervisor.time,
                        "monotonic_ns",
                        side_effect=(2_000_000_000, 3_000_000_000, 3_100_000_000),
                    ),
                ):
                    result = supervisor.sampled_cgroup_drain(
                        policy,
                        "/user.slice/worker.service",
                        expected_gpu_uuid=GPU_UUID,
                        maximum_temperature_c=80,
                        last_valid_monotonic_ns=previous_ns,
                    )
                self.assertEqual(expected, run.call_args.kwargs["timeout"])
                self.assertEqual("THERMAL_POLICY_VIOLATION", result.trigger[0])
                self.assertTrue(result.stop_required)

    def test_sampled_drain_requires_stop_after_slow_successful_sample(self):
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        cool = sample(temperature_c=60)
        late = supervisor.TelemetrySample(
            observed_monotonic_ns=21_000_000_001,
            observed_at=cool.observed_at,
            gpu_uuid=cool.gpu_uuid,
            name=cool.name,
            temperature_c=cool.temperature_c,
            free_mib=cool.free_mib,
            total_mib=cool.total_mib,
        )
        with (
            mock.patch.object(supervisor, "cgroup_empty", return_value=False),
            mock.patch.object(
                supervisor, "sample_gpu_for_runtime", return_value=late
            ),
            mock.patch.object(
                supervisor.time,
                "monotonic_ns",
                side_effect=(2_000_000_000, 3_000_000_000),
            ),
        ):
            result = supervisor.sampled_cgroup_drain(
                policy,
                "/user.slice/worker.service",
                expected_gpu_uuid=GPU_UUID,
                maximum_temperature_c=80,
                last_valid_monotonic_ns=1_000_000_000,
            )
        self.assertEqual("TELEMETRY_UNAVAILABLE", result.trigger[0])
        self.assertTrue(result.stop_required)
        self.assertEqual((late,), result.samples)

    def test_sampled_drain_requires_stop_when_child_outlives_deadline(self):
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        policy["kill_confirmation_seconds"] = 1.0
        with (
            mock.patch.object(supervisor, "cgroup_empty", return_value=False),
            mock.patch.object(supervisor, "sample_gpu_for_runtime") as telemetry,
            mock.patch.object(
                supervisor.time,
                "monotonic_ns",
                side_effect=(1_000_000_000, 2_000_000_000),
            ),
        ):
            result = supervisor.sampled_cgroup_drain(
                policy,
                "/user.slice/worker.service",
                expected_gpu_uuid=GPU_UUID,
                maximum_temperature_c=80,
                last_valid_monotonic_ns=1_000_000_000,
            )
        telemetry.assert_not_called()
        self.assertEqual("TERMINATION_UNCONFIRMED", result.trigger[0])
        self.assertTrue(result.stop_required)
        self.assertFalse(result.cgroup_empty_confirmed)
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
        verified_trainer = mock.Mock()
        verified_trainer.committed_bytes.return_value = exact
        with tempfile.TemporaryDirectory() as directory:
            local_here = pathlib.Path(directory)
            (local_here / "supervisor_validation.py").write_bytes(exact + b"drift")
            with (
                mock.patch.object(supervisor, "HERE", local_here),
                mock.patch.object(supervisor, "trainer", verified_trainer),
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

    def test_prepared_atomic_admission_maps_only_exact_bootstrap_paths(self):
        root = pathlib.Path("/isolated") / RUN_ID
        paths = supervisor.bootstrap.AdmissionPaths(
            run_id=RUN_ID,
            root=root,
            payload=root / "payload",
            logs=root / "logs",
            reports=root / "reports",
            runtime_cache=root / "runtime-cache",
            namespace_root=root / "namespace-root",
            reserve=root / ".evidence-reserve",
        )
        result = supervisor.bootstrap.AdmissionResult(
            paths=paths,
            prepared=True,
            failed_stage=None,
            failure_type=None,
            created_entries=("reports", "payload", "logs", "runtime-cache"),
            reserve_allocated=True,
            tombstone=None,
        )
        attempt = supervisor.attempt_from_atomic_admission(result, RUN_ID)
        self.assertEqual(root / "input", attempt.input_bundle)
        self.assertEqual(paths.namespace_root, attempt.namespace_root)
        self.assertEqual(paths.reserve, attempt.reserve)

    def test_partial_atomic_admission_preserves_tombstone_and_false_claims(self):
        root = pathlib.Path("/isolated") / RUN_ID
        paths = supervisor.bootstrap.AdmissionPaths(
            run_id=RUN_ID,
            root=root,
            payload=root / "payload",
            logs=root / "logs",
            reports=root / "reports",
            runtime_cache=root / "runtime-cache",
            namespace_root=root / "namespace-root",
            reserve=root / ".evidence-reserve",
        )
        artifact = supervisor.bootstrap.PublishedArtifact(
            path=paths.reports / "admission-failure.json",
            sha256="c" * 64,
            size=512,
            committed=True,
            commit_point="FINAL_LINK_AND_DIRECTORY_FSYNC",
            cleanup_complete=True,
            cleanup_error=None,
            temporary_path=None,
        )
        result = supervisor.bootstrap.AdmissionResult(
            paths=paths,
            prepared=False,
            failed_stage="CREATE_PAYLOAD_DIRECTORY",
            failure_type="OSError",
            created_entries=("reports", ".evidence-reserve"),
            reserve_allocated=False,
            tombstone=supervisor.bootstrap.TombstoneResult(
                artifact=artifact,
                reserve_released=True,
                indeterminate=False,
                error=None,
            ),
        )
        payload, exit_code = supervisor.partial_admission_outcome(result, "smoke")
        self.assertEqual(
            supervisor.TERMINAL_EXIT_CODES["PRECONDITION_DENIED"], exit_code
        )
        self.assertEqual("COMMITTED", payload["admissionTombstone"]["state"])
        for field in (
            "workerLaunched",
            "supervisorReportPublished",
            "qualificationEligible",
            "receiptEligible",
            "publicationEligible",
            "runtimeWitnessPresent",
            "autonomyEligible",
        ):
            self.assertFalse(payload[field])

    def test_write_once_wrapper_preserves_atomic_commit_metadata(self):
        directory = pathlib.Path("/isolated/reports")
        artifact = supervisor.bootstrap.PublishedArtifact(
            path=directory / "supervisor-report.json",
            sha256="d" * 64,
            size=128,
            committed=True,
            commit_point="FINAL_LINK_AND_DIRECTORY_FSYNC",
            cleanup_complete=True,
            cleanup_error=None,
            temporary_path=None,
        )
        with mock.patch.object(
            supervisor.bootstrap, "publish_write_once", return_value=artifact
        ) as publish:
            observed = supervisor.publish_evidence_write_once(
                directory, "supervisor-report.json", b"payload"
            )
        publish.assert_called_once_with(
            directory, "supervisor-report.json", b"payload"
        )
        self.assertEqual("COMMITTED", observed["publicationState"])
        self.assertEqual("FINAL_LINK_AND_DIRECTORY_FSYNC", observed["commitPoint"])
        self.assertEqual("d" * 64, observed["sha256"])

    @unittest.skipUnless(os.name == "posix", "write-once protocol requires POSIX")
    def test_every_training_bundle_filename_is_atomic_publication_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            for name in (*supervisor.BUNDLE_SOURCE_FILES, "training-bundle.json"):
                with self.subTest(name=name):
                    observed = supervisor.publish_evidence_write_once(
                        root,
                        name,
                        b"committed component",
                    )
                    self.assertEqual("COMMITTED", observed["publicationState"])
                    self.assertEqual(b"committed component", (root / name).read_bytes())

    def test_staged_manifest_retains_exact_trainer_file_identity_schema(self):
        exact_trainer = load_trainer()
        source = {
            "repository": "szl-holdings/szl-forge",
            "revision": SOURCE,
            "branch": "main",
            "originIdentityVerified": True,
            "freshRemoteMainObserved": False,
            "freshRemoteMainObservationDelegatedToSupervisor": True,
            "cachedRemoteTrackingMatches": True,
            "workingTreeClean": True,
            "commitSignatureVerifiedByThisTool": False,
        }

        def committed_bytes(_source_commit: str, relative_path: str) -> bytes:
            return (HERE / pathlib.PurePosixPath(relative_path).name).read_bytes()

        def fake_publish(directory: pathlib.Path, name: str, data: bytes) -> dict:
            path = directory / name
            path.write_bytes(data)
            return {
                "path": str(path),
                "bytes": len(data),
                "sha256": supervisor.sha256_bytes(data),
                "publicationState": "COMMITTED",
                "commitPoint": "FINAL_LINK_AND_DIRECTORY_FSYNC",
                "cleanupComplete": True,
                "cleanupError": None,
                "temporaryPath": None,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory).resolve()
            input_bundle = root / "input"
            input_bundle.mkdir()
            attempt = supervisor.Attempt(
                run_id=RUN_ID,
                root=root,
                payload=root / "payload",
                logs=root / "logs",
                reports=root / "reports",
                runtime_cache=root / "runtime-cache",
                reserve=root / ".evidence-reserve",
                input_bundle=input_bundle,
            )
            staged_trainer = mock.Mock()
            staged_trainer.committed_bytes.side_effect = committed_bytes
            with (
                mock.patch.object(supervisor, "trainer", staged_trainer),
                mock.patch.object(
                    supervisor,
                    "publish_evidence_write_once",
                    side_effect=fake_publish,
                ),
                mock.patch.object(supervisor.os, "chmod"),
                mock.patch.object(supervisor, "fsync_directory"),
            ):
                manifest = supervisor.stage_training_bundle(attempt, SOURCE, source)

            for identity in manifest["files"].values():
                self.assertEqual({"bytes", "sha256"}, set(identity))
            self.assertEqual(
                set(supervisor.BUNDLE_SOURCE_FILES),
                set(manifest["bundlePublication"]["sourceArtifacts"]),
            )
            observed = exact_trainer.validate_training_bundle(input_bundle, SOURCE)
            self.assertEqual(manifest["bundleSha256"], observed["bundleSha256"])

    def test_indeterminate_publication_is_explicit_and_commit_is_unknown(self):
        error = supervisor.bootstrap.PublicationIndeterminate(
            "directory fsync failed",
            final_path=pathlib.Path("/isolated/reports/supervisor-report.json"),
            temporary_path=pathlib.Path("/isolated/reports/.report.tmp"),
            sha256="e" * 64,
            size=256,
        )
        evidence = supervisor.publication_failure_evidence(error)
        self.assertIsNotNone(evidence)
        self.assertEqual("INDETERMINATE", evidence["state"])
        self.assertTrue(evidence["finalLinkExists"])
        self.assertFalse(evidence["directoryCommitConfirmed"])
        self.assertIsNone(evidence["committed"])

    def test_terminal_report_base_contains_exact_prelaunch_provenance(self):
        root = pathlib.Path("/isolated/run")
        attempt = supervisor.Attempt(
            run_id=RUN_ID,
            root=root,
            payload=root / "payload",
            logs=root / "logs",
            reports=root / "reports",
            runtime_cache=root / "runtime-cache",
            reserve=root / ".evidence-reserve",
            input_bundle=root / "input",
        )
        report = supervisor.terminal_report_base(
            attempt=attempt,
            source={"revision": SOURCE},
            run_kind="smoke",
            policy_sha="1" * 64,
            supervisor_sha="2" * 64,
            worker_sha="3" * 64,
            validator_sha="4" * 64,
            candidate_sha="5" * 64,
            interpreter={"path": "/trusted/python"},
            worker_environment_sha="6" * 64,
            outer_containment={"unit": "outer.service"},
            admission_sha="7" * 64,
            training_bundle_sha="8" * 64,
            credential_canary_sha="9" * 64,
        )
        self.assertEqual(
            {
                "trainingBundleSha256": "8" * 64,
                "credentialCanarySha256": "9" * 64,
            },
            report["provenance"],
        )


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

    def test_main_passes_admitted_attempt_to_worker_launcher(self):
        tree = ast.parse(inspect.getsource(supervisor.main))
        launch_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "launch_worker_unit"
        ]
        self.assertEqual(1, len(launch_calls))
        attempt_keywords = [
            keyword
            for keyword in launch_calls[0].keywords
            if keyword.arg == "attempt"
        ]
        self.assertEqual(1, len(attempt_keywords))
        self.assertIsInstance(attempt_keywords[0].value, ast.Name)
        self.assertEqual("attempt", attempt_keywords[0].value.id)

    def test_main_orders_fresh_confirmation_immediately_before_launch(self):
        tree = ast.parse(inspect.getsource(supervisor.main))

        def named_calls(name: str) -> list[ast.Call]:
            return [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
            ]

        admission_calls = named_calls("sample_gpu_for_admission")
        runtime_calls = named_calls("sample_gpu_for_runtime")
        admission_gates = named_calls("initial_temperature_gate")
        pair_gates = named_calls("readiness_pair_gate")
        launch_calls = named_calls("launch_worker_unit")
        self.assertEqual(1, len(admission_calls))
        self.assertEqual(4, len(runtime_calls))
        self.assertEqual(1, len(admission_gates))
        self.assertEqual(2, len(pair_gates))
        self.assertEqual(1, len(launch_calls))
        confirmation = min(runtime_calls, key=lambda call: call.lineno)
        launch = launch_calls[0]
        prelaunch_confirmation = max(
            (call for call in runtime_calls if call.lineno < launch.lineno),
            key=lambda call: call.lineno,
        )
        ordered_pair_gates = sorted(pair_gates, key=lambda call: call.lineno)
        evidence_writes = named_calls("publish_evidence_write_once")
        report_builds = named_calls("terminal_report_base")
        self.assertLess(admission_calls[0].lineno, admission_gates[0].lineno)
        self.assertLess(admission_gates[0].lineno, confirmation.lineno)
        self.assertLess(confirmation.lineno, ordered_pair_gates[0].lineno)
        self.assertLess(
            min(call.lineno for call in evidence_writes),
            prelaunch_confirmation.lineno,
        )
        self.assertLess(
            max(call.lineno for call in report_builds),
            prelaunch_confirmation.lineno,
        )
        self.assertLess(
            prelaunch_confirmation.lineno, ordered_pair_gates[-1].lineno
        )
        self.assertLess(ordered_pair_gates[-1].lineno, launch.lineno)
        self.assertTrue(
            all(
                call.lineno > launch.lineno
                for call in runtime_calls
                if call not in (confirmation, prelaunch_confirmation)
            )
        )
        guarded_confirmation = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(call is confirmation for call in ast.walk(node))
        ]
        self.assertTrue(guarded_confirmation)
        guard = ast.unparse(guarded_confirmation[-1].test)
        self.assertIn("admission_error is None", guard)
        self.assertIn("admission_gate_error is None", guard)

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
            input_bundle=root / "input",
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
            "ProtectHome": "tmpfs",
            "ProtectProc": "invisible",
            "ProcSubset": "pid",
            "PrivateTmp": "yes",
            "PrivateNetwork": "yes",
            "RestrictSUIDSGID": "yes",
            "RestrictNamespaces": "yes",
            "InaccessiblePaths": supervisor.worker_inaccessible_paths(
                supervisor.os.getpid()
            ),
            "LimitFSIZE": "67108864",
            "BindsTo": "outer.service",
            "ControlGroup": "/user.slice/worker.service",
            "MainPID": "123",
            "ExecMainPID": "123",
            "ExecMainStatus": "0",
            "Result": "success",
        }
        worker_argv = ["/trusted/python", "-I", "-B", "/trusted/worker.py"]
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        properties.update(
            supervisor.normalized_worker_mount_contract(
                supervisor.worker_mount_contract(policy, attempt)
            )
        )
        with (
            mock.patch.object(
                supervisor.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
            mock.patch.object(supervisor, "unit_properties", return_value=properties),
            mock.patch.object(supervisor, "validate_namespace_scaffold"),
            mock.patch.object(pathlib.Path, "is_dir", return_value=True),
            mock.patch("pathlib.Path.read_text", return_value="123\n"),
        ):
            supervisor.launch_worker_unit(
                policy,
                attempt=attempt,
                outer_unit="outer",
                worker_unit="worker",
                worker_argv=worker_argv,
                worker_environment_values=exact_environment,
                stdout_path=pathlib.Path("/tmp/run/logs/stdout.log"),
                stderr_path=pathlib.Path("/tmp/run/logs/stderr.log"),
            )

        command = run.call_args.args[0]
        expected_inaccessible = (
            f"--property=InaccessiblePaths="
            f"{supervisor.worker_inaccessible_paths(supervisor.os.getpid())}"
        )
        self.assertIn(expected_inaccessible, command)
        self.assertEqual(
            f"-/run/host /proc/{supervisor.os.getpid()}",
            supervisor.worker_inaccessible_paths(supervisor.os.getpid()),
        )
        mount_contract = supervisor.worker_mount_contract(policy, attempt)
        for name, value in mount_contract.items():
            self.assertIn(f"--property={name}={value}", command)
        read_only_binds = mount_contract["BindReadOnlyPaths"]
        self.assertNotIn("/etc/hostname", read_only_binds)
        self.assertNotIn("/mnt/c", read_only_binds)
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
        root = pathlib.Path("/isolated/run")
        attempt = supervisor.Attempt(
            run_id=RUN_ID,
            root=root,
            payload=root / "payload",
            logs=root / "logs",
            reports=root / "reports",
            runtime_cache=root / "runtime-cache",
            reserve=root / "reserve.bin",
            input_bundle=root / "input",
        )
        properties = {
            "ActiveState": "active",
            "SubState": "exited",
            "KillMode": "control-group",
            "SendSIGKILL": "yes",
            "RemainAfterExit": "yes",
            "NoNewPrivileges": "yes",
            "ProtectControlGroups": "yes",
            "ProtectSystem": "strict",
            "ProtectHome": "tmpfs",
            "ProtectProc": "invisible",
            "ProcSubset": "pid",
            "PrivateTmp": "yes",
            "PrivateNetwork": "yes",
            "RestrictSUIDSGID": "yes",
            "RestrictNamespaces": "yes",
            "InaccessiblePaths": supervisor.worker_inaccessible_paths(
                supervisor.os.getpid()
            ),
            "LimitFSIZE": "67108864",
            "BindsTo": "outer.service",
            "ControlGroup": "/user.slice/worker.service",
            "MainPID": "0",
            "ExecMainPID": "456",
            "ExecMainStatus": "70",
            "Result": "exit-code",
        }
        policy = copy.deepcopy(supervisor.EXPECTED_POLICY)
        properties.update(
            supervisor.normalized_worker_mount_contract(
                supervisor.worker_mount_contract(policy, attempt)
            )
        )
        with (
            mock.patch.object(
                supervisor.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            mock.patch.object(supervisor, "unit_properties", return_value=properties),
            mock.patch.object(supervisor, "validate_namespace_scaffold"),
            mock.patch.object(pathlib.Path, "is_dir", return_value=True),
        ):
            observed = supervisor.launch_worker_unit(
                policy,
                attempt=attempt,
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
