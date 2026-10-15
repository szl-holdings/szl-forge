"""Safety controls for the opt-in AirLLM local smoke path."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import local_airllm as run


class LocalAirLLMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.split = self.scratch / "split-fresh"
        self.wheel = self.root / "airllm-4.0.0-py3-none-any.whl"
        self.evidence = self.root / "forge-evidence.json"
        self._write("config.json", {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        self._write("tokenizer.json", {})
        (self.snapshot / "model.safetensors").write_bytes(b"synthetic-not-real-weights")
        self.manifest = self._manifest()
        source_patch = patch.object(run, "SOURCE_EVIDENCE", self.evidence)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        parent_patch = patch.object(run, "SCRATCH_PARENT", self.root)
        parent_patch.start()
        self.addCleanup(parent_patch.stop)

    def _write(self, name, content):
        (self.snapshot / name).write_text(json.dumps(content), encoding="utf-8")

    def _manifest(self):
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in self.snapshot.iterdir() if path.is_file()}
        (self.scratch / run.SCRATCH_MARKER).write_text(json.dumps({"schema": "szl.airllm.scratch-owner/v1",
                                                                "model_id": "example/research-only", "revision": "a" * 40}), encoding="utf-8")
        self.evidence.write_text(json.dumps({"plan": {"modelId": "example/research-only", "modelRevision": "a" * 40},
                                             "artifactSha256": files, "status": "SMOKE_FAIL", "productionDisposition": "HOLD"}), encoding="utf-8")
        return {"schema": run.SCHEMA, "model_id": "example/research-only",
                "revision": "a" * 40, "snapshot_dir": str(self.snapshot),
                "scratch_root": str(self.scratch), "split_dir": str(self.split), "gpu_uuid": "GPU-test-1",
                "files": files, "airllm_wheel": str(self.wheel)}

    def _mock_ready(self):
        return (patch.object(run, "_check_airllm_wheel", return_value={"version": "4.0.0", "wheel_sha256": run.AIRLLM_WHEEL_SHA256}),
                patch.object(run, "_dependency_versions", return_value=({"torch": "test"}, [])),
                patch.object(run, "_available_ram", return_value=20 * run.GIB),
                patch.object(run, "_gpu_free_bytes", return_value=8 * run.GIB),
                patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * run.GIB)))

    def test_local_hashed_builtin_snapshot_is_only_experiment_ready(self):
        mocks = self._mock_ready()
        for item in mocks:
            item.start()
            self.addCleanup(item.stop)
        receipt = run.preflight(self.manifest)
        self.assertEqual(receipt["disposition"], "READY_FOR_LOCAL_EXPERIMENT_ONLY")
        self.assertEqual(receipt["model_origin_verification"], "FORGE_REPORTED_HASH_BINDING_LOCAL_BYTES_RECHECKED")
        self.assertEqual(receipt["forge_source_evidence_status"], "SMOKE_FAIL")
        self.assertEqual(receipt["inference_run"], "NOT_RUN")
        self.assertFalse(receipt["training_eligible"])
        self.assertFalse(receipt["publication_eligible"])
        self.assertFalse(receipt["execution_authority"])
        self.assertFalse(self.split.exists())

    def test_remote_code_in_model_or_tokenizer_is_refused(self):
        for name in ("config.json", "tokenizer_config.json"):
            with self.subTest(name=name):
                self._write(name, {"auto_map": {"AutoModel": "evil.py"}})
                self.manifest = self._manifest()
                with self.assertRaisesRegex(run.PreflightError, "remote code"):
                    run.preflight(self.manifest)
                if name == "config.json":
                    self._write(name, {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})

    def test_split_path_alias_and_existing_path_are_refused(self):
        self.manifest["split_dir"] = str(self.snapshot / "derived")
        with self.assertRaisesRegex(run.PreflightError, "disjoint"):
            run.preflight(self.manifest)
        self.manifest["split_dir"] = str(self.split)
        self.split.mkdir()
        with self.assertRaisesRegex(run.PreflightError, "fresh"):
            run.preflight(self.manifest)

    def test_scratch_ownership_marker_and_source_hash_binding_are_required(self):
        (self.scratch / run.SCRATCH_MARKER).unlink()
        with self.assertRaisesRegex(run.PreflightError, "scratch marker"):
            run.preflight(self.manifest)
        self.manifest = self._manifest()
        self.manifest["files"]["config.json"] = "0" * 64
        with self.assertRaisesRegex(run.PreflightError, "canonical Forge"):
            run.preflight(self.manifest)

    def test_gguf_and_unlisted_files_are_refused(self):
        (self.snapshot / "model.safetensors").unlink()
        (self.snapshot / "model.gguf").write_bytes(b"not-a-transformers-checkpoint")
        self.manifest = self._manifest()
        with self.assertRaisesRegex(run.PreflightError, "unsafe checkpoint format"):
            run.preflight(self.manifest)
        (self.snapshot / "model.gguf").unlink()
        (self.snapshot / "model.safetensors").write_bytes(b"fixture")
        with self.assertRaisesRegex(run.PreflightError, "exactly"):
            run.preflight(self.manifest)

    def test_deletion_request_is_not_a_supported_manifest_field(self):
        self.manifest["delete_original"] = True
        with self.assertRaisesRegex(run.PreflightError, "exactly match"):
            run.preflight(self.manifest)

    def test_known_capacity_shortfall_blocks_even_if_runtime_is_pinned(self):
        with patch.object(run, "_check_airllm_wheel", return_value={"version": "4.0.0"}), \
             patch.object(run, "_dependency_versions", return_value=({}, [])), \
             patch.object(run, "_available_ram", return_value=1), \
             patch.object(run, "_gpu_free_bytes", return_value=1), \
             patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            receipt = run.preflight(self.manifest)
        self.assertEqual(receipt["disposition"], "BLOCKED")
        self.assertEqual(set(receipt["blockers"]), {"DISK_CAPACITY_BELOW_PLANNING_BOUND",
                                                    "RAM_CAPACITY_BELOW_PLANNING_BOUND",
                                                    "GPU_CAPACITY_BELOW_PLANNING_BOUND"})

    def test_library_older_than_checkpoint_declaration_blocks(self):
        self._write("config.json", {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                                    "transformers_version": "99.0.0"})
        self.manifest = self._manifest()
        with patch.object(run, "_check_airllm_wheel", return_value={"version": "4.0.0"}), \
             patch.object(run, "_available_ram", return_value=20 * run.GIB), \
             patch.object(run, "_gpu_free_bytes", return_value=8 * run.GIB), \
             patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * run.GIB)):
            receipt = run.preflight(self.manifest)
        self.assertIn("TRANSFORMERS_BELOW_CHECKPOINT_DECLARATION", receipt["blockers"])

    def test_cli_run_request_never_creates_split_when_wheel_absent(self):
        manifest_path = self.root / "manifest.json"
        receipt_path = self.root / "receipt.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with patch.object(sys, "argv", ["local_airllm.py", "--manifest", str(manifest_path),
                                        "--receipt", str(receipt_path), "--run"]):
            self.assertEqual(run.main(), 2)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["inference_run"], "NOT_RUN")
        self.assertIn("PINNED_AIRLLM_RUNTIME_UNAVAILABLE", receipt["blockers"])
        self.assertFalse(self.split.exists())

    def test_timeout_is_unknown_after_attempt_and_keeps_split_for_inspection(self):
        manifest_path = self.root / "manifest.json"
        receipt_path = self.root / "timeout-receipt.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        mocks = self._mock_ready()
        for item in mocks:
            item.start()
            self.addCleanup(item.stop)
        with patch.object(run.subprocess, "run", side_effect=subprocess.TimeoutExpired(["worker"], 300)), \
             patch.object(sys, "argv", ["local_airllm.py", "--manifest", str(manifest_path),
                                       "--receipt", str(receipt_path), "--run"]):
            self.assertEqual(run.main(), 2)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["inference_run"], "UNKNOWN_AFTER_ATTEMPT")
        self.assertTrue(receipt["worker_started"])
        self.assertIn("TimeoutExpired", receipt["blockers"][0])

    def test_receipt_cannot_be_created_inside_checkpoint(self):
        manifest_path = self.root / "manifest.json"
        receipt_path = self.snapshot / "receipt.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with patch.object(sys, "argv", ["local_airllm.py", "--manifest", str(manifest_path),
                                       "--receipt", str(receipt_path)]):
            self.assertEqual(run.main(), 2)
        self.assertFalse(receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
