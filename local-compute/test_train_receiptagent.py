"""CPU-only admission regressions; no model import, GPU load or training."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import train_receiptagent as target


def options(output):
    return argparse.Namespace(output=output, steps=1, learning_rate=5e-5,
        max_length=2048, max_runtime_seconds=60, max_temperature_c=80,
        min_free_gib=4, source_commit="a" * 40, loss_mode="assistant")


def adapter_config():
    return {"base_model_name_or_path": target.BASE_REPO, "peft_type": "LORA",
        "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32, "lora_dropout": 0,
        "auto_mapping": {"base_model_class": "Qwen3_5ForConditionalGeneration"},
        "revision": None, "target_modules": "language.*proj", "modules_to_save": None,
        "use_dora": False, "use_qalora": False}


class Admissions(unittest.TestCase):
    def test_strict_json_rejects_duplicates_and_nonfinite(self):
        for text in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(text=text), self.assertRaises(target.GateError):
                target.strict_json(text)
        self.assertEqual(target.strict_json(b'{"x":1}'), {"x": 1})

    def test_closed_sequence_loss_modes(self):
        self.assertEqual(target.label_mask([1, 2, 3, 4], [1, 2], "assistant", 8), [-100, -100, 3, 4])
        self.assertEqual(target.label_mask([1, 2, 3], [9], "full-sequence", 8), [1, 2, 3])
        for ids, prefix, mode, limit in (([1, 2], [1, 2], "assistant", 8),
            ([1, 2, 3], [1, 7], "assistant", 8), ([1, 2], [], "assistant", 8),
            ([1, 2, 3], [1], "assistant", 2), ([1, True, 3], [1], "assistant", 8),
            ([1, 2], [1], "automatic", 8)):
            with self.subTest(mode=mode, ids=ids), self.assertRaises(target.GateError):
                target.label_mask(ids, prefix, mode, limit)

    def test_bounded_steps_and_hyperparameters(self):
        for steps in (1, 16, 32):
            args = options(Path("unused")); args.steps = steps
            target.validate_options(args)
        for name, value in (("steps", 64), ("steps", True), ("learning_rate", float("nan")),
            ("learning_rate", 0.01), ("max_length", 4096), ("max_temperature_c", 99),
            ("max_runtime_seconds", 9000), ("source_commit", "main")):
            args = options(Path("unused")); setattr(args, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(target.GateError):
                target.validate_options(args)

    def test_conditional_generation_and_unquantized_base_required(self):
        config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 248320}}
        target.validate_base(config)
        for change in ({"architectures": ["Qwen3_5ForCausalLM"]}, {"quantization_config": {"bits": 4}}, {"model_type": "other"}):
            with self.assertRaises(target.GateError):
                target.validate_base({**config, **change})

    def test_adapter_config_never_drops_unknown_fields(self):
        config = adapter_config()
        target.validate_adapter_config(config, set(config))
        with self.assertRaisesRegex(target.GateError, "PEFT_UNKNOWN_CONFIG_FIELDS"):
            target.validate_adapter_config({**config, "future_field": None}, set(config))
        for change in ({"base_model_name_or_path": "Qwen/other"}, {"r": 8},
            {"auto_mapping": {"base_model_class": "Qwen3_5ForCausalLM"}},
            {"revision": "main"}, {"lora_dropout": 0.1}, {"modules_to_save": ["lm_head"]}):
            with self.assertRaises(target.GateError):
                target.validate_adapter_config({**config, **change})

    def test_only_three_role_owned_text_rows(self):
        row = {"messages": [{"role": role, "content": "synthetic fixture"} for role in ("system", "user", "assistant")]}
        target.validate_row(row)
        for wrong in ({**row, "image": "unwanted"}, {"messages": row["messages"][:-1]},
            {"messages": [{"role": "system", "content": [{"type": "image"}]}] + row["messages"][1:]}):
            with self.assertRaises(target.GateError):
                target.validate_row(wrong)

    def test_curriculum_rejects_drift_before_admission(self):
        with self.assertRaisesRegex(target.GateError, "CURRICULUM_HASH_MISMATCH"):
            target.load_curriculum(Path("unused"), "a" * 40, reader=lambda *args: b"changed")

    def test_only_allowlisted_committed_paths(self):
        with self.assertRaisesRegex(target.GateError, "CURRICULUM_PATH_NOT_ADMITTED"):
            target.committed(Path("unused"), "a" * 40, "frontier/qwen35-receiptagent-v3/test.jsonl")
        with self.assertRaisesRegex(target.GateError, "SOURCE_REVISION_REQUIRED"):
            target.committed(Path("unused"), "main", "receiptagent/train.jsonl")

    def test_path_traversal_absolute_and_windows_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("../outside", "/absolute", "C:/drive", "x\\escape", "./file", "x//y", "x/../y", None):
                with self.subTest(name=name), self.assertRaises(target.GateError):
                    target.artifact_path(root, name)
            self.assertEqual(target.artifact_path(root, "model.safetensors"), root / "model.safetensors")

    def make_artifact(self, root):
        directory = root / "artifact"; directory.mkdir()
        payload = b"synthetic-manifest-test-not-real-weights"
        (directory / "model.safetensors").write_bytes(payload)
        manifest = {"schema": "szl.local-artifact/v1", "repo_id": target.BASE_REPO,
            "revision": target.BASE_REVISION, "verified_from": "PINNED_HUB_METADATA_AND_LOCAL_BYTES",
            "files": [{"path": "model.safetensors", "bytes": len(payload), "sha256": target.sha(payload)}]}
        path = root / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
        self.enterContext(patch.dict(target.TRUSTED_MANIFESTS,
            {(target.BASE_REPO, target.BASE_REVISION): target.sha(path.read_bytes())}))
        return directory, path, manifest

    def test_self_consistent_forged_manifest_cannot_rebind_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, path, manifest = self.make_artifact(Path(temp))
            substituted = b"substitute-arbitrary-weights"
            (directory / "model.safetensors").write_bytes(substituted)
            manifest["files"][0].update(bytes=len(substituted), sha256=target.sha(substituted))
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(target.GateError, "MANIFEST_NOT_AUTHENTICATED"):
                target.verify_artifact(directory, path, target.BASE_REPO, target.BASE_REVISION)

    def test_manifest_rehash_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, path, manifest = self.make_artifact(Path(temp))
            result = target.verify_artifact(directory, path, target.BASE_REPO, target.BASE_REVISION)
            self.assertEqual(result["revision"], target.BASE_REVISION)
            with self.assertRaisesRegex(target.GateError, "ARTIFACT_IDENTITY"):
                target.verify_artifact(directory, path, target.BASE_REPO, "b" * 40)
            (directory / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(target.GateError, "ARTIFACT_BYTES_MISMATCH"):
                target.verify_artifact(directory, path, target.BASE_REPO, target.BASE_REVISION)

    def test_undeclared_files_fail_but_hf_cache_metadata_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, path, _ = self.make_artifact(Path(temp))
            (directory / ".cache").mkdir(); (directory / ".cache" / "local.metadata").write_text("local")
            target.verify_artifact(directory, path, target.BASE_REPO, target.BASE_REVISION)
            (directory / "extra.json").write_text("{}")
            with self.assertRaisesRegex(target.GateError, "ARTIFACT_UNDECLARED_FILES"):
                target.verify_artifact(directory, path, target.BASE_REPO, target.BASE_REVISION)

    def test_no_remote_trackio_or_hidden_space_sync(self):
        for key in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_BUCKET_ID", "SPACE_ID"):
            with patch.dict(os.environ, {key: "remote"}, clear=True), self.assertRaisesRegex(target.GateError, "REMOTE_TRACKING"):
                target.configure_local_environment(Path("unused"))
        with patch.dict(os.environ, {}, clear=True):
            target.configure_local_environment(Path("output"))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRACKIO_DIR"], str(Path("output") / "trackio"))

    def test_unused_gpu_cache_can_be_reclaimed_without_weakening_floor(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(target.subprocess, "check_output", return_value="60\n"):
            args = options(Path(temp))
            for available, accepted in ((512 * 1024**2, True), (80 * 1024**2, False)):
                torch = Mock()
                torch.cuda.mem_get_info.side_effect = [(80 * 1024**2, 8 * 1024**3), (available, 8 * 1024**3)]
                torch.cuda.memory_reserved.return_value = 0
                if accepted:
                    report = target.runtime_guard(torch, args, target.time.monotonic())
                    self.assertTrue(report["unused_cache_reclaim_attempted"])
                    self.assertEqual(report["free_bytes"], available)
                else:
                    with self.assertRaisesRegex(target.GateError, "GPU_MEMORY_LIMIT"):
                        target.runtime_guard(torch, args, target.time.monotonic())
                torch.cuda.empty_cache.assert_called_once()

    def test_failure_emits_evidence_without_promoting_or_leaking_exception(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(target, "HERE", Path(temp)):
            args = options(Path(temp) / "results" / "fresh")
            def fail(_args, _report):
                raise RuntimeError("secret-shaped-example-never-persist")
            code, report = target.run(args, train_function=fail)
            saved = (args.output / "training-report.json").read_text()
            self.assertEqual(code, 1); self.assertEqual(report["state"], "FAILED_CLOSED")
            self.assertNotIn("secret-shaped", saved)
            self.assertFalse(report["publication_eligible"]); self.assertFalse(report["autonomy_eligible"])
            self.assertFalse((Path(temp) / ".native-training.lock").exists())
            without = dict(report); claimed = without.pop("report_sha256")
            self.assertEqual(target.sha(target.canonical(without)), claimed)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(target, "HERE", Path(temp)):
            out = Path(temp) / "results" / "old"; out.mkdir(parents=True)
            marker = out / "keep.txt"; marker.write_text("keep")
            with self.assertRaises(target.GateError):
                target.run(options(out), train_function=lambda *_: self.fail("must not train"))
            self.assertEqual(marker.read_text(), "keep")

    def test_existing_compute_lock_preserved(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(target, "HERE", Path(temp)):
            lock = Path(temp) / ".native-training.lock"; lock.write_text("other owner")
            code, report = target.run(options(Path(temp) / "results" / "fresh"), train_function=lambda *_: self.fail("must not train"))
            self.assertEqual(code, 1); self.assertEqual(report["state"], "FAILED_CLOSED")
            self.assertEqual(lock.read_text(), "other owner")


if __name__ == "__main__":
    unittest.main()
