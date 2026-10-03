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
    def setUp(self):
        self.enterContext(patch.object(target.shutil, "disk_usage", return_value=Mock(free=8 * 1024**3)))

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


class DiskAdmissions(unittest.TestCase):
    FLOOR = 1536 * 1024**2
    RESERVE = 256 * 1024**2

    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.enterContext(patch.object(target, "HERE", self.root))
        self.probe = self.enterContext(patch.object(target.shutil, "disk_usage",
            return_value=Mock(free=8 * 1024**3)))

    def test_low_disk_refuses_before_output_lock_or_training(self):
        for free in (self.FLOOR - 1, self.FLOOR + self.RESERVE - 1):
            with self.subTest(free=free):
                self.probe.return_value = Mock(free=free)
                output = self.root / "results" / "not-created"
                callback = Mock()
                with self.assertRaisesRegex(target.GateError, "DISK_LIMIT"):
                    target.run(options(output), train_function=callback)
                callback.assert_not_called()
                self.assertFalse(output.parent.exists())
                self.assertFalse((self.root / ".native-training.lock").exists())

    def test_exact_headroom_boundary_preserves_complete_receipt_hash(self):
        self.probe.return_value = Mock(free=self.FLOOR + self.RESERVE)
        args = options(self.root / "results" / "boundary")
        def complete(_args, report):
            report["state"] = "MEASURED_LOCAL_CONTINUATION_COMPLETED"
        code, report = target.run(args, train_function=complete)
        self.assertEqual(code, 0)
        self.assertTrue(report["durable_report_written"])
        self.assertEqual(json.loads((args.output / "training-report.json").read_bytes()), report)
        unsigned = dict(report); digest = unsigned.pop("report_sha256")
        self.assertEqual(target.sha(target.canonical(unsigned)), digest)

    def test_probe_nearest_existing_ancestor_without_creating_parents(self):
        missing = self.root / "results" / "not-created"
        target.disk_guard(missing)
        self.probe.assert_called_once_with(self.root.resolve())
        self.assertFalse(missing.parent.exists())

    def test_probe_failure_invalid_free_and_invalid_reserve_fail_closed(self):
        for free in (None, True, -1, "unknown"):
            with self.subTest(free=free):
                self.probe.return_value = Mock(free=free)
                with self.assertRaisesRegex(target.GateError, "DISK_PROBE_UNAVAILABLE"):
                    target.disk_guard(self.root)
        self.probe.side_effect = OSError("private-shaped-probe-error")
        with self.assertRaisesRegex(target.GateError, "DISK_PROBE_UNAVAILABLE"):
            target.disk_guard(self.root)
        self.probe.side_effect = None
        for reserve in (-1, True, 0.5):
            with self.subTest(reserve=reserve), self.assertRaisesRegex(target.GateError, "DISK_RESERVE_INVALID"):
                target.disk_guard(self.root, reserve_bytes=reserve)

    def test_lock_volume_failure_leaves_output_and_other_owner_untouched(self):
        lock = self.root / ".native-training.lock"; lock.write_text("other owner")
        self.probe.side_effect = [Mock(free=8 * 1024**3), OSError("lock volume unavailable")]
        output = self.root / "results" / "fresh"
        callback = Mock()
        with self.assertRaisesRegex(target.GateError, "DISK_PROBE_UNAVAILABLE"):
            target.run(options(output), train_function=callback)
        callback.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(lock.read_text(), "other owner")

    def test_runtime_disk_gate_precedes_any_cuda_probe(self):
        self.probe.return_value = Mock(free=self.FLOOR + self.RESERVE - 1)
        torch = Mock()
        with self.assertRaisesRegex(target.GateError, "DISK_LIMIT"):
            target.runtime_guard(torch, options(self.root), target.time.monotonic())
        torch.cuda.is_available.assert_not_called()

    def test_checkpoint_admission_denies_any_save_when_disk_drops(self):
        self.probe.return_value = Mock(free=self.FLOOR + self.RESERVE - 1)
        model, processor = Mock(), Mock()
        with self.assertRaisesRegex(target.GateError, "DISK_LIMIT"):
            target.save_candidate(model, processor, self.root / "adapter")
        model.save_pretrained.assert_not_called()
        processor.save_pretrained.assert_not_called()

    def test_processor_save_has_an_independent_fresh_disk_probe(self):
        self.probe.side_effect = [Mock(free=8 * 1024**3), Mock(free=self.FLOOR - 1)]
        model, processor = Mock(), Mock()
        candidate = self.root / "adapter"
        with self.assertRaisesRegex(target.GateError, "DISK_LIMIT"):
            target.save_candidate(model, processor, candidate)
        model.save_pretrained.assert_called_once_with(candidate, safe_serialization=True)
        processor.save_pretrained.assert_not_called()

    def test_tracking_write_is_not_called_after_capacity_refusal(self):
        self.probe.return_value = Mock(free=self.FLOOR - 1)
        tracker = Mock()
        with self.assertRaisesRegex(target.GateError, "DISK_LIMIT"):
            target.tracking_write(self.root, tracker.log, {"step": 1})
        tracker.log.assert_not_called()

    def test_secondary_tracking_failure_does_not_mask_execution_failure(self):
        self.probe.return_value = Mock(free=self.FLOOR - 1)
        tracker, report = Mock(), {}
        target.finish_tracking(tracker, self.root, report, primary_error=True)
        tracker.finish.assert_not_called()
        self.assertEqual(report["tracking_finish_error_code"], "DISK_LIMIT")
        with self.assertRaisesRegex(target.GateError, "DISK_LIMIT"):
            target.finish_tracking(tracker, self.root, {}, primary_error=False)

    def test_report_disk_refusal_returns_failure_without_opening_report(self):
        args = options(self.root / "results" / "no-report")
        def complete(_args, report):
            report["state"] = "MEASURED_LOCAL_CONTINUATION_COMPLETED"
            self.probe.return_value = Mock(free=self.FLOOR - 1)
        code, report = target.run(args, train_function=complete)
        self.assertEqual(code, 1)
        self.assertEqual(report["state"], "FAILED_CLOSED")
        self.assertEqual(report["error_code"], "DISK_LIMIT")
        self.assertFalse(report["durable_report_written"])
        self.assertFalse((args.output / "training-report.json").exists())
        self.assertFalse((self.root / ".native-training.lock").exists())
        self.assertFalse(report["publication_eligible"])

    def test_report_failure_preserves_original_error_and_other_owner_lock(self):
        lock = self.root / ".native-training.lock"; lock.write_text("other owner")
        args = options(self.root / "results" / "occupied")
        self.probe.side_effect = [Mock(free=8 * 1024**3), Mock(free=8 * 1024**3),
            Mock(free=8 * 1024**3),
            OSError("capacity unavailable at report")]
        code, report = target.run(args, train_function=Mock())
        self.assertEqual(code, 1)
        self.assertEqual(report["error_code"], "DISK_PROBE_UNAVAILABLE")
        self.assertEqual(report["prior_error_code"], "LOCAL_EXECUTION_FAILED_NO_PROMOTION")
        self.assertFalse(report["durable_report_written"])
        self.assertEqual(lock.read_text(), "other owner")

    def test_report_actual_utf8_bytes_and_explicit_size_limit(self):
        output = self.root / "receipt"; output.mkdir()
        report = {"state": "FAILED_CLOSED", "text": "\u00e9", "durable_report_written": False}
        saved = target.persist_report(output, report)
        raw = (output / "training-report.json").read_bytes()
        self.assertEqual(json.loads(raw), saved)
        self.assertEqual(self.probe.call_count, 2)
        with patch.object(target, "MAX_REPORT_BYTES", len(raw) - 1):
            with self.assertRaisesRegex(target.GateError, "REPORT_TOO_LARGE"):
                target.persist_report(self.root / "not-written", report)
        self.assertFalse((self.root / "not-written").exists())
        boundary = self.root / "boundary"; boundary.mkdir()
        with patch.object(target, "MAX_REPORT_BYTES", len(raw)):
            target.persist_report(boundary, report)

    def test_fsync_failure_cannot_publish_a_successful_final_receipt(self):
        args = options(self.root / "results" / "sync-failed")
        def complete(_args, report):
            report["state"] = "MEASURED_LOCAL_CONTINUATION_COMPLETED"
        with patch.object(target.os, "fsync", side_effect=OSError("private sync error")):
            code, report = target.run(args, train_function=complete)
        self.assertEqual(code, 1)
        self.assertEqual(report["state"], "FAILED_CLOSED")
        self.assertEqual(report["error_code"], "REPORT_PERSISTENCE_FAILED")
        self.assertFalse(report["durable_report_written"])
        self.assertFalse((args.output / "training-report.json").exists())
        self.assertEqual(len(list(args.output.glob(".training-report-*.pending"))), 1)
        self.assertFalse((self.root / ".native-training.lock").exists())

    def test_short_or_unflushed_pending_report_is_never_final(self):
        for failure in ("short", "flush"):
            with self.subTest(failure=failure):
                output = self.root / failure; output.mkdir()
                stream = Mock()
                stream.write.side_effect = lambda raw: len(raw) - (failure == "short")
                if failure == "flush":
                    stream.flush.side_effect = OSError("private flush error")
                manager = Mock()
                manager.__enter__ = Mock(return_value=stream)
                manager.__exit__ = Mock(return_value=False)
                with patch.object(Path, "open", return_value=manager), patch.object(target.os, "link") as publish:
                    with self.assertRaises((target.GateError, OSError)):
                        target.persist_report(output, {"state": "MEASURED_LOCAL_CONTINUATION_COMPLETED"})
                publish.assert_not_called()
                self.assertFalse((output / "training-report.json").exists())

    def test_existing_final_receipt_is_never_overwritten(self):
        output = self.root / "prior"; output.mkdir()
        final = output / "training-report.json"; final.write_bytes(b"preserve older evidence")
        with self.assertRaises(FileExistsError):
            target.persist_report(output, {"state": "FAILED_CLOSED"})
        self.assertEqual(final.read_bytes(), b"preserve older evidence")

    def test_exclusive_lock_write_failure_cleans_only_newly_owned_lock(self):
        original_open = Path.open
        def opening(path, *args, **kwargs):
            if path.name == ".native-training.lock":
                stream = original_open(path, *args, **kwargs)
                manager = Mock()
                manager.__enter__ = Mock(return_value=Mock(write=Mock(side_effect=OSError("private lock write error"))))
                manager.__exit__ = Mock(side_effect=lambda *_: stream.close())
                return manager
            return original_open(path, *args, **kwargs)
        callback = Mock()
        args = options(self.root / "results" / "lock-failed")
        with patch.object(Path, "open", opening):
            code, report = target.run(args, train_function=callback)
        callback.assert_not_called()
        self.assertEqual(code, 1)
        self.assertFalse((self.root / ".native-training.lock").exists())
        self.assertTrue(report["durable_report_written"])
        self.assertEqual(report["state"], "FAILED_CLOSED")

    def test_cleanup_refusal_retains_primary_error_and_attempts_failure_receipt(self):
        original_unlink = Path.unlink
        def unlinking(path, *args, **kwargs):
            if path.name == ".native-training.lock":
                raise OSError("private cleanup error")
            return original_unlink(path, *args, **kwargs)
        def fail(_args, _report):
            raise target.GateError("PRIMARY_EXECUTION_FAILURE")
        args = options(self.root / "results" / "cleanup-failed")
        with patch.object(Path, "unlink", unlinking):
            code, report = target.run(args, train_function=fail)
        self.assertEqual(code, 1)
        self.assertEqual(report["state"], "FAILED_CLOSED")
        self.assertEqual(report["error_code"], "OWNED_LOCK_CLEANUP_FAILED")
        self.assertEqual(report["prior_error_code"], "PRIMARY_EXECUTION_FAILURE")
        self.assertTrue(report["durable_report_written"])
        self.assertTrue((self.root / ".native-training.lock").exists())
        self.assertNotIn("private cleanup", (args.output / "training-report.json").read_text())


if __name__ == "__main__":
    unittest.main()
