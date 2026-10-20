"""CPU-only fault fixtures. No GPU, framework, network or credentials."""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native_supervisor", HERE / "native_supervisor.py")
supervisor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervisor
spec.loader.exec_module(supervisor)
GPU = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class DurableFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="szl-v4-supervisor-test-")
        self.directory = Path(self.temporary.name)
        os.chmod(self.directory, 0o700)

    def tearDown(self):
        self.temporary.cleanup()

    def test_nonce_is_single_use_and_binds_admission(self):
        with mock.patch.object(supervisor, "fsync_directory") as fsync:
            receipt = supervisor.consume_nonce(self.directory, "a" * 64, "b" * 64)
            self.assertEqual(receipt["durability"], "MEASURED")
            self.assertEqual(fsync.call_count, 1)
            stored = json.loads((self.directory / ("a" * 64 + ".json")).read_bytes())
            self.assertEqual(stored["admission_sha256"], "b" * 64)
            with self.assertRaisesRegex(supervisor.SupervisorError, "NONCE_ALREADY_CONSUMED"):
                supervisor.consume_nonce(self.directory, "a" * 64, "c" * 64)

    def test_fsync_failure_does_not_reuse_nonce(self):
        with mock.patch.object(supervisor.os, "fsync", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                supervisor.consume_nonce(self.directory, "a" * 64, "b" * 64)
        self.assertTrue((self.directory / ("a" * 64 + ".json")).exists())
        with self.assertRaisesRegex(supervisor.SupervisorError, "NONCE_ALREADY_CONSUMED"):
            supervisor.consume_nonce(self.directory, "a" * 64, "b" * 64)

    def test_invalid_nonce_no_file_created(self):
        for nonce in ("../x", "A" * 64, True, "a" * 65):
            with self.assertRaisesRegex(supervisor.SupervisorError, "NONCE_INVALID"):
                supervisor.consume_nonce(self.directory, nonce, "b" * 64)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_chain_tip_is_independently_recomputable_not_signature(self):
        with mock.patch.object(supervisor, "fsync_directory"):
            chain = supervisor.ReceiptChain(self.directory)
            try:
                first = chain.append({"state": "ADMITTED"})
                second = chain.append({"state": "BLOCKED"})
            finally:
                chain.close()
        self.assertFalse(first["authenticated"])
        self.assertEqual(second["previous_sha256"], first["record_sha256"])
        for row in (self.directory / "supervisor-chain.jsonl").read_bytes().splitlines():
            value = json.loads(row)
            digest = value.pop("record_sha256")
            self.assertEqual(digest, hashlib.sha256(supervisor.canonical(value)).hexdigest())

    def test_tree_byte_cap_and_hardlink_rejection(self):
        path = self.directory / "x"
        path.write_bytes(b"1234")
        self.assertEqual(supervisor.tree_bytes(self.directory, cap=4), 4)
        with self.assertRaisesRegex(supervisor.SupervisorError, "TREE_BYTE_CAP"):
            supervisor.tree_bytes(self.directory, cap=3)
        try:
            os.link(path, self.directory / "alias")
        except OSError:
            self.skipTest("hardlinks unavailable on this filesystem")
        with self.assertRaisesRegex(supervisor.SupervisorError, "TREE_SPECIAL_OR_ALIAS"):
            supervisor.tree_bytes(self.directory, cap=100)

    def test_environment_contains_no_ambient_secret(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "fixture", "SSH_AUTH_SOCK": "fixture",
                                          "PYTHONPATH": "fixture"}):
            result = supervisor.minimal_environment(self.directory, gpu_uuid=GPU)
        self.assertNotIn("HF_TOKEN", result)
        self.assertNotIn("SSH_AUTH_SOCK", result)
        self.assertNotIn("PYTHONPATH", result)
        self.assertEqual(result["CUDA_VISIBLE_DEVICES"], GPU)

    def test_worker_argv_uses_exact_isolated_startup(self):
        output = self.directory / "output"
        bounds = {"host_memory_bytes": 1024, "wall_seconds": 10,
                  "kill_seconds": 2}
        run = SimpleNamespace(
            nonce="a" * 64, payload={"bounds": bounds, "runner": {"device_id": GPU}},
            readonly_paths=(), worker_python=Path("/usr/bin/python3"),
            worker_path=Path("/src/native_worker.py"),
            request_path=Path("/run/request.json"), owner_policy_sha256="b" * 64,
            path=lambda name: output,
        )
        argv = supervisor.worker_argv(run, description="szl-ra4-owned-" + "c" * 64)
        marker = (str(run.worker_python), "-I", "-B", str(run.worker_path))
        start = argv.index(str(run.worker_python))
        self.assertEqual(argv[start:start + len(marker)], marker)


class TelemetryFixtures(unittest.TestCase):
    def test_exact_gpu_sample_and_bounds(self):
        sample = supervisor.parse_gpu_sample((GPU + ", 49140, 9000, 40\n").encode(), GPU, 10.0)
        self.assertEqual(sample.used_bytes, 9000 * 1048576)
        supervisor.check_sample(sample, previous=9, now=11, max_gap=3,
                                max_gpu_bytes=10000 * 1048576, max_temperature=80000)

    def test_sample_rejects_wrong_identity_unbounded_and_nan(self):
        for value in ((GPU + ", 49140, 9000, N/A\n").encode(),
                      (GPU + ", 49140, 999999, 40\n").encode(), b"x" * 513):
            with self.assertRaises(supervisor.SupervisorError):
                supervisor.parse_gpu_sample(value, GPU, 10.0)
        with self.assertRaisesRegex(supervisor.SupervisorError, "GPU_IDENTITY_MISMATCH"):
            supervisor.parse_gpu_sample(b"GPU-wrong, 49140, 0, 40\n", GPU, 1)

    def test_gap_memory_and_temperature_fail_closed(self):
        sample = supervisor.parse_gpu_sample((GPU + ", 49140, 9000, 80\n").encode(), GPU, 10.0)
        for kwargs, expected in (({"max_gap": 1}, "TELEMETRY_GAP"),
                                 ({"max_gpu_bytes": 1}, "GPU_MEMORY_CAP"),
                                 ({"max_temperature": 80000}, "GPU_TEMPERATURE_CAP")):
            defaults = {"previous": 9, "now": 11, "max_gap": 3,
                        "max_gpu_bytes": 10000 * 1048576, "max_temperature": 90000}
            defaults.update(kwargs)
            with self.assertRaisesRegex(supervisor.SupervisorError, expected):
                supervisor.check_sample(sample, **defaults)

    def test_bonsai_or_any_compute_process_blocks_no_stop_attempt(self):
        with mock.patch.object(supervisor, "bounded_command", return_value=b"7289, " + GPU.encode()):
            with self.assertRaisesRegex(supervisor.SupervisorError, "GPU_ALREADY_HAS_COMPUTE_PROCESS"):
                supervisor.assert_gpu_exclusive(Path("/usr/bin/nvidia-smi"))
        with mock.patch.object(supervisor, "bounded_command", return_value=b"\n"):
            supervisor.assert_gpu_exclusive(Path("/usr/bin/nvidia-smi"))

    def test_occupied_gpu_denies_before_nonce_launch_writes_or_signals(self):
        # SIMULATED ordering fixture only. Mocked earlier checks are not real
        # admission/qualification; keep the actual GPU denial and supervise flow.
        import native_runtime as runtime
        with tempfile.TemporaryDirectory(prefix="szl-v4-occupied-gpu-") as temporary, ExitStack() as patches:
            run_dir = Path(temporary) / ("a" * 64)
            run = SimpleNamespace(nonce=run_dir.name, payload={"bounds": {}},
                path=lambda name: run_dir if name == "run_dir" else run_dir / "output")
            patches.enter_context(mock.patch.object(supervisor.sys, "platform", "linux"))
            patches.enter_context(mock.patch.object(runtime, "revalidate_bound_run"))
            for name in ("validate_bounds", "qualification_matches"):
                patches.enter_context(mock.patch.object(supervisor, name))
            patches.enter_context(mock.patch.object(supervisor, "private_directory", side_effect=lambda path: path))
            query = patches.enter_context(mock.patch.object(supervisor, "bounded_command",
                return_value=b"7289, " + GPU.encode() + b"\n"))
            untouched = [patches.enter_context(mock.patch.object(supervisor, name)) for name in (
                "sample_gpu", "_check_disk", "consume_nonce", "_allocate_reserve",
                "ReceiptChain", "drain_owned_unit", "_control_command", "fsync_directory")]
            untouched.extend(patches.enter_context(mock.patch.object(target, name, create=True))
                for target, name in ((supervisor.subprocess, "Popen"), (supervisor.os, "mkdir"),
                                     (supervisor.os, "kill"), (supervisor.os, "killpg")))
            models = ("torch", "transformers", "peft", "bitsandbytes", "native_worker")
            before_modules = {name: sys.modules.get(name) for name in models}
            with self.assertRaisesRegex(supervisor.SupervisorError, "GPU_ALREADY_HAS_COMPUTE_PROCESS"):
                supervisor.supervise(run)
            query.assert_called_once_with((str(supervisor.NVIDIA_SMI), "--query-compute-apps=pid,gpu_uuid",
                                           "--format=csv,noheader,nounits"), maximum=4096)
            for effect in untouched:
                effect.assert_not_called()
            self.assertEqual({name: sys.modules.get(name) for name in models}, before_modules)
            self.assertEqual(list(Path(temporary).iterdir()), [])


class ContainmentFixtures(unittest.TestCase):
    def test_pending_unit_is_not_false_containment(self):
        unit = "szl-ra4-" + "a" * 64 + ".service"
        with mock.patch.object(supervisor, "bounded_command", return_value=b"LoadState=not-found\n"):
            self.assertIsNone(supervisor.unit_properties(unit))

    def test_unit_missing_security_property_is_rejected(self):
        unit = "szl-ra4-" + "a" * 64 + ".service"
        with mock.patch.object(supervisor, "bounded_command", return_value=b"MainPID=123\nLoadState=loaded\n"):
            with self.assertRaisesRegex(supervisor.SupervisorError, "UNIT_PROPERTY_INCOMPLETE"):
                supervisor.unit_properties(unit)

    def test_unit_invalid_name_never_queries_manager(self):
        with mock.patch.object(supervisor, "bounded_command") as command:
            with self.assertRaisesRegex(supervisor.SupervisorError, "UNIT_ID_INVALID"):
                supervisor.unit_properties("bonsai.service")
            command.assert_not_called()

    def test_live_property_mismatch_fails_before_reading_cgroup(self):
        props = dict(supervisor.SYSTEMD_REQUIRED_PROPERTIES)
        props["ProtectControlGroups"] = "no"
        with mock.patch.object(supervisor.sys, "platform", "linux"), \
                mock.patch.object(supervisor, "_bounded_file") as read:
            with self.assertRaisesRegex(supervisor.SupervisorError, "CONTAINMENT_PROPERTY_MISMATCH"):
                supervisor.verify_containment("szl-ra4-" + "a" * 64 + ".service", props,
                                              {"host_memory_bytes": 1024})
            read.assert_not_called()

    def test_cgroup_descendant_populated_bit_not_just_leader_exit(self):
        with mock.patch.object(supervisor, "_bounded_file", return_value=b"populated 1\nfrozen 0\n"):
            self.assertFalse(supervisor.cgroup_empty(Path("/sys/fs/cgroup/fixture")))
        with mock.patch.object(supervisor, "_bounded_file", return_value=b"populated 0\nfrozen 0\n"):
            self.assertTrue(supervisor.cgroup_empty(Path("/sys/fs/cgroup/fixture")))

    def test_cgroup_invalid_populated_not_accepted(self):
        with mock.patch.object(supervisor, "_bounded_file", return_value=b"frozen 0\n"):
            with self.assertRaisesRegex(supervisor.SupervisorError, "CGROUP_EVENTS_INVALID"):
                supervisor.cgroup_empty(Path("/sys/fs/cgroup/fixture"))

    def test_kernel_drain_signals_only_exact_owned_unit(self):
        unit = "szl-ra4-" + "a" * 64 + ".service"
        with mock.patch.object(supervisor, "cgroup_empty", side_effect=(False, True)), \
                mock.patch.object(supervisor, "_control_command", return_value=b"") as command:
            self.assertTrue(supervisor.drain_owned_unit(unit, Path("/sys/fs/cgroup/fixture"),
                                                       kill_seconds=1, drain_seconds=1))
        argv = command.call_args.args[0]
        self.assertEqual(argv[-1], unit)
        self.assertIn("--kill-whom=all", argv)
        self.assertIn("--signal=TERM", argv)

    def test_empty_cgroup_does_not_signal(self):
        with mock.patch.object(supervisor, "cgroup_empty", return_value=True), \
                mock.patch.object(supervisor, "_control_command") as command:
            self.assertTrue(supervisor.drain_owned_unit("szl-ra4-" + "a" * 64 + ".service",
                                  Path("/sys/fs/cgroup/fixture"), kill_seconds=1, drain_seconds=1))
            command.assert_not_called()

    def test_invalid_metric_is_not_success_from_child_report(self):
        run = SimpleNamespace(nonce="a" * 64, payload={"curriculum": {"seed": 1,
            "train_sha256": "b" * 64, "manifest_sha256": "c" * 64},
            "base": {"repository": "test", "revision": "d" * 40},
            "source": {"revision": "e" * 40}, "tokens": {"tensor_ledger_sha256": "f" * 64},
            "bounds": {"row_order_sha256": "0" * 64, "total_tokens": 400000},
            "runner": {"device_id": GPU}}, path=lambda _: Path("/fixture/output"))
        report = {"schema": "szl.receiptagent.v4-native-worker-result/v1",
            "state": "SMOKE_COMPLETED_NOT_RELEASED", "evidence_class": "REPORTED",
            "admission_nonce": run.nonce, "rows": 360, "optimizer_updates": 1, "seed": 1,
            "train_sha256": "b" * 64, "manifest_sha256": "c" * 64,
            "base_id": "test", "base_revision": "d" * 40, "source_revision": "e" * 40,
            "token_ledger_sha256": "f" * 64, "row_order_sha256": "0" * 64,
            "device_uuid": GPU, "energy_joules": "UNAVAILABLE", "model_evaluation": "NOT_RUN",
            "independent_replay": "UNAVAILABLE", "training_eligible": False,
            "publication_eligible": False, "execution_authority": False,
            "supervised_tokens": 500, "token_weighted_loss": float("nan"),
            "gradient_norm_before_clip": 1.0, "artifact_sha256": {}}
        runtime = SimpleNamespace(read_worker_result=lambda *a, **k: report,
                                  sha256_file=mock.Mock())
        with self.assertRaisesRegex(supervisor.SupervisorError, "WORKER_REPORTED_METRIC_INVALID"):
            supervisor._worker_observation(run, runtime=runtime)
        runtime.sha256_file.assert_not_called()

    def test_real_runtime_worker_readback_accepts_only_finite_closed_metrics(self):
        # Synthetic byte fixtures, not model weights, admission or a GPU run.
        import native_runtime as runtime
        with tempfile.TemporaryDirectory(prefix="szl-v4-child-readback-test-") as temporary:
            output = Path(temporary)
            os.chmod(output, 0o700)
            (output / "adapter").mkdir(mode=0o700)
            artifacts = {"adapter/adapter_model.safetensors": b"SIMULATED_BYTES_NOT_WEIGHTS",
                         "adapter/adapter_config.json": b'{"fixture":true}'}
            hashes = {}
            for name, raw in artifacts.items():
                (output / name).write_bytes(raw)
                hashes[name] = hashlib.sha256(raw).hexdigest()
            payload = {"curriculum": {"seed": 1, "train_sha256": "b" * 64,
                        "manifest_sha256": "c" * 64},
                       "base": {"repository": "test", "revision": "d" * 40},
                       "source": {"revision": "e" * 40},
                       "tokens": {"tensor_ledger_sha256": "f" * 64},
                       "bounds": {"row_order_sha256": "0" * 64, "total_tokens": 400000,
                                  "artifact_cap_bytes": 512 * 1024**2},
                       "runner": {"device_id": GPU}}
            run = SimpleNamespace(nonce="a" * 64, payload=payload, path=lambda _: output)
            report = {"schema": "szl.receiptagent.v4-native-worker-result/v1",
                "state": "SMOKE_COMPLETED_NOT_RELEASED", "evidence_class": "REPORTED",
                "admission_nonce": run.nonce, "rows": 360, "optimizer_updates": 1, "seed": 1,
                "train_sha256": "b" * 64, "manifest_sha256": "c" * 64,
                "base_id": "test", "base_revision": "d" * 40, "source_revision": "e" * 40,
                "token_ledger_sha256": "f" * 64, "row_order_sha256": "0" * 64,
                "device_uuid": GPU, "energy_joules": "UNAVAILABLE", "model_evaluation": "NOT_RUN",
                "independent_replay": "UNAVAILABLE", "training_eligible": False,
                "publication_eligible": False, "execution_authority": False,
                "supervised_tokens": 500, "token_weighted_loss": 1.25,
                "gradient_norm_before_clip": 0.25, "artifact_sha256": hashes}
            (output / "worker-result.json").write_bytes(runtime.canonical(report))
            result = supervisor._worker_observation(run, runtime=runtime)
            self.assertEqual(result["artifact_sha256"], hashes)
            self.assertFalse(result["optimizer_semantics_independently_replayed"])


@unittest.skipUnless(os.name == "posix", "bounded selector lifecycle uses Linux pipe semantics")
class LinuxControlFixtures(unittest.TestCase):
    def test_fixed_child_output(self):
        self.assertEqual(supervisor.bounded_command((sys.executable, "-I", "-c", "print('fixture')")),
                         b"fixture\n")

    def test_child_output_cap(self):
        with self.assertRaisesRegex(supervisor.SupervisorError, "CONTROL_OUTPUT_CAP"):
            supervisor.bounded_command((sys.executable, "-I", "-c", "print('x'*10000)"), maximum=10)

    def test_child_timeout(self):
        with self.assertRaisesRegex(supervisor.SupervisorError, "CONTROL_TIMEOUT"):
            supervisor.bounded_command((sys.executable, "-I", "-c", "import time; time.sleep(10)"),
                                       timeout=0.1)


if __name__ == "__main__":
    unittest.main()
