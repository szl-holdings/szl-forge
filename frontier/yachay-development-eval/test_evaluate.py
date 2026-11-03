"""CPU-only software tests. No model quality or A6000 qualification evidence."""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as e


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        def snapshot(name, files):
            directory = self.root / name
            directory.mkdir()
            for file, content in files.items():
                (directory / file).write_bytes(content)
            return {"root": str(directory), "files": {n: e.digest(directory / n) for n in files}}
        base = snapshot("base", {"config.json": b'{"model_type":"nemotron_h"}', "model.safetensors": b"fixture-not-model-weights"})
        base.update(id=e.BASE, revision=e.REVISION)
        tok = snapshot("tokenizer", {"tokenizer_config.json": b"{}"})
        tok["revision"] = e.REVISION
        adapter = snapshot("adapter", {"adapter_config.json": json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": e.BASE}).encode(), "adapter_model.safetensors": b"fixture-not-adapter-weights"})
        adapter.update(id="pid13223-step16700", campaign_id="pid13223", step=16700)
        data = self.root / "data.jsonl"
        data.write_text(json.dumps({"id": "case-1", "messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}) + "\n")
        review = self.root / "review.json"
        packages = {n: "1.2.3" for n in ("torch", "transformers", "peft", "safetensors", "accelerate")}
        review.write_bytes(e.canonical({"binding_sha256": hashlib.sha256(e.canonical({"base": base["files"], "tokenizer": tok["files"], "packages": packages})).hexdigest(), "local_custom_code_reviewed": True, "reviewer": "synthetic-test"}))
        self.plan = {"schema": e.SCHEMA, "evaluator_sha256": e.digest(e.__file__), "run_id": "fixture-run", "optimizer_updates": 0,
            "production_eligible": False, "base": base, "tokenizer": tok, "adapters": [adapter],
            "data_role": "development_diagnostic", "data": {"path": str(data), "sha256": e.digest(data)},
            "loader_review": {"path": str(review), "sha256": e.digest(review)}, "seed": 42,
            "runtime": {"python_minor": "3.12", "packages": packages, "hostname": "fixture", "boot_id": "fixture-boot", "gpu_uuid": "GPU-fixture", "compute_capability": [8, 6]},
            "limits": {"wall_seconds": 1, "rows": 1, "sequence_tokens": 512, "cpu_cores": 2, "ram_bytes": 8 * 1024**3, "gpu_allocator_fraction": 0.5}}

    def save(self):
        path = self.root / "plan.json"
        path.write_bytes(e.canonical(self.plan))
        return path

    def test_schema_and_byte_verification_are_distinct(self):
        e.validate(self.plan)
        self.assertEqual(len(e.verify(self.plan)), 1)
        Path(self.plan["base"]["root"], "model.safetensors").unlink()
        e.validate(self.plan)
        with self.assertRaises(e.Blocked):
            e.verify(self.plan)

    def test_unknown_fields_are_enumerated(self):
        self.plan["runtime"]["boot_id"] = None
        self.plan["adapters"][0]["files"]["adapter_config.json"] = "UNKNOWN"
        with self.assertRaisesRegex(e.Blocked, "boot_id"):
            e.validate(self.plan)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(e.Blocked):
                e.parse(raw)

    def test_wrong_base_or_revision_rejected(self):
        for key in ("id", "revision"):
            bad = copy.deepcopy(self.plan)
            bad["base"][key] = "other"
            with self.subTest(key=key), self.assertRaises(e.Blocked):
                e.validate(bad)

    def test_no_held_out_relabeling_or_training(self):
        for key, value in (("data_role", "held_out"), ("optimizer_updates", 1), ("optimizer_updates", False), ("production_eligible", True)):
            bad = copy.deepcopy(self.plan)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(e.Blocked):
                e.validate(bad)

    def test_runtime_and_resource_limits(self):
        for key, value in (("wall_seconds", 901), ("rows", True), ("rows", 0), ("sequence_tokens", 2049), ("cpu_cores", 5), ("ram_bytes", 65 * 1024**3), ("gpu_allocator_fraction", float("nan"))):
            bad = copy.deepcopy(self.plan)
            bad["limits"][key] = value
            with self.subTest(key=key), self.assertRaises(e.Blocked):
                e.validate(bad)

    def test_duplicate_candidate_and_campaign_required(self):
        self.plan["adapters"].append(copy.deepcopy(self.plan["adapters"][0]))
        with self.assertRaises(e.Blocked):
            e.validate(self.plan)
        self.plan["adapters"].pop()
        self.plan["adapters"][0]["campaign_id"] = ""
        with self.assertRaises(e.Blocked):
            e.validate(self.plan)

    def test_snapshot_traversal_and_unbound_file(self):
        bad = copy.deepcopy(self.plan)
        bad["base"]["files"]["../escape"] = "0" * 64
        with self.assertRaises(e.Blocked):
            e.validate(bad)
        Path(self.plan["base"]["root"], "extra.py").write_text("raise RuntimeError")
        with self.assertRaisesRegex(e.Blocked, "inventory"):
            e.verify(self.plan)

    def test_modified_adapter_rejected(self):
        Path(self.plan["adapters"][0]["root"], "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(e.Blocked, "hash mismatch"):
            e.verify(self.plan)

    def test_bound_file_symlinks_rejected_before_hashing(self):
        for key in ("data", "loader_review"):
            with self.subTest(key=key):
                target = Path(self.plan[key]["path"])
                link = self.root / (key + "-link")
                try:
                    link.symlink_to(target)
                except (OSError, NotImplementedError) as exc:
                    self.skipTest("native symlink creation unavailable: " + str(exc))
                spec = {**self.plan[key], "path": str(link)}
                with patch.object(e, "digest", side_effect=AssertionError("must reject before hashing")):
                    with self.assertRaisesRegex(e.Blocked, "symlink|reparse"):
                        e.verify_file(spec)

    def test_bound_file_parent_symlink_and_traversal_rejected(self):
        directory = self.root / "alias"
        try:
            directory.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("native directory symlink creation unavailable: " + str(exc))
        for key in ("data", "loader_review"):
            target = Path(self.plan[key]["path"])
            for path in (directory / target.name, directory / ".." / self.root.name / target.name):
                with self.subTest(key=key, path=path):
                    with self.assertRaisesRegex(e.Blocked, "symlink|reparse|traversal"):
                        e.verify_file({**self.plan[key], "path": str(path)})

    def test_bound_file_dangling_link_rejected(self):
        link = self.root / "dangling"
        try:
            link.symlink_to(self.root / "absent")
        except (OSError, NotImplementedError) as exc:
            self.skipTest("native symlink creation unavailable: " + str(exc))
        with self.assertRaisesRegex(e.Blocked, "symlink|reparse"):
            e.verify_file({"path": str(link), "sha256": "0" * 64})

    def test_bound_file_reparse_attributes_rejected_at_leaf_and_parent(self):
        real_lstat = Path.lstat
        for key in ("data", "loader_review"):
            leaf = Path(self.plan[key]["path"])
            for marked in (leaf, leaf.parent):
                def lstat(path, *args, **kwargs):
                    info = real_lstat(path, *args, **kwargs)
                    return types.SimpleNamespace(
                        st_mode=info.st_mode,
                        st_file_attributes=e.stat.FILE_ATTRIBUTE_REPARSE_POINT if path == marked else 0,
                    )
                with self.subTest(key=key, marked=marked), patch.object(Path, "lstat", lstat):
                    with self.assertRaisesRegex(e.Blocked, "reparse"):
                        e.verify_file(self.plan[key])

    @unittest.skipUnless(os.name == "nt", "native Windows junction test")
    def test_native_windows_parent_junction_rejected(self):
        junction = self.root / "junction"
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(self.root)],
                                 capture_output=True, text=True)
        if created.returncode:
            self.skipTest("native junction creation unavailable")
        try:
            for key in ("data", "loader_review"):
                with self.subTest(key=key), self.assertRaisesRegex(e.Blocked, "reparse"):
                    e.verify_file({**self.plan[key], "path": str(junction / Path(self.plan[key]["path"]).name)})
        finally:
            junction.rmdir()

    def test_review_binds_packages_and_source(self):
        self.plan["runtime"]["packages"]["torch"] = "9.9.9"
        with self.assertRaisesRegex(e.Blocked, "loader review"):
            e.verify(self.plan)

    def test_adapter_wrong_base_fails_even_with_new_hash(self):
        adapter = self.plan["adapters"][0]
        path = Path(adapter["root"], "adapter_config.json")
        config = e.read_json(path)
        config["base_model_name_or_path"] = "other/model"
        path.write_bytes(e.canonical(config))
        adapter["files"][path.name] = e.digest(path)
        with self.assertRaisesRegex(e.Blocked, "base binding"):
            e.verify(self.plan)

    def test_all_rows_required_no_silent_subset(self):
        self.plan["limits"]["rows"] = 2
        with self.assertRaisesRegex(e.Blocked, "row count"):
            e.load_rows(self.plan)

    def test_duplicate_row_ids_rejected(self):
        path = Path(self.plan["data"]["path"])
        path.write_text(path.read_text() * 2)
        self.plan["limits"]["rows"] = 2
        with self.assertRaisesRegex(e.Blocked, "duplicate row"):
            e.load_rows(self.plan)

    def test_template_masking_and_no_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [1, 2] if add_generation_prompt else [1, 2, 3, 4]
        row = e.load_rows(self.plan)[0]
        self.assertEqual(e.encode(Tokenizer(), row, 4), ([1, 2, 3, 4], [-100, -100, 3, 4]))
        with self.assertRaisesRegex(e.Blocked, "no truncation"):
            e.encode(Tokenizer(), row, 3)
        class BadPrefix:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [1, 9] if add_generation_prompt else [1, 2, 3, 4]
        with self.assertRaisesRegex(e.Blocked, "prefix mismatch"):
            e.encode(BadPrefix(), row, 4)

    def test_token_weighted_loss_not_mean_of_means(self):
        result = e.aggregate([{"target_tokens": 1, "nll_sum": 1.0}, {"target_tokens": 3, "nll_sum": 9.0}])
        self.assertEqual(result["token_weighted_nll"], 2.5)
        self.assertAlmostEqual(result["perplexity"], math.exp(2.5))
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(e.Blocked):
                e.aggregate([{"target_tokens": 1, "nll_sum": value}])

    def test_check_and_verify_do_not_launch_workers(self):
        plan_path = self.save()
        with patch.object(e.subprocess, "Popen", side_effect=AssertionError("unexpected launch")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.main(["check", str(plan_path)]), 0)
            self.assertEqual(e.main(["verify", str(plan_path)]), 0)

    def test_wrong_plan_hash_never_launches(self):
        path = self.save()
        with patch.object(e.subprocess, "Popen", side_effect=AssertionError("unexpected launch")), self.assertRaisesRegex(e.Blocked, "plan hash"):
            e.run(self.plan, path, self.root / "out", "0" * 64)

    def test_source_drift_never_launches(self):
        self.plan["evaluator_sha256"] = "0" * 64
        path = self.save()
        with self.assertRaisesRegex(e.Blocked, "source mismatch"):
            e.run(self.plan, path, self.root / "out", e.digest(path))

    def test_worker_requires_supervisor(self):
        with patch.dict(os.environ, {"YACHAY_EVALUATOR_PARENT": "invalid"}), self.assertRaisesRegex(e.Blocked, "supervising parent"):
            e.worker(self.plan, self.root / "out")

    def test_partial_event_retained_as_error(self):
        path = self.root / "events.jsonl"
        path.write_bytes(b'{"candidate":"base","state":"STARTED"}\n{"candidate":')
        events, errors = e.read_events(path, ["base"])
        self.assertEqual(len(events), 1)
        self.assertEqual(errors[0]["line"], 2)

    def test_unplanned_event_rejected(self):
        path = self.root / "events.jsonl"
        path.write_bytes(b'{"candidate":"other","state":"COMPLETED","metrics":{}}\n')
        events, errors = e.read_events(path, ["base"])
        self.assertEqual(events, [])
        self.assertEqual(len(errors), 1)

    @unittest.skipUnless(sys.platform == "linux", "POSIX supervision test")
    def test_spawn_failure_writes_failure_receipt(self):
        path = self.save()
        with patch.object(e, "runtime_guard"), patch.object(e.subprocess, "Popen", side_effect=OSError("synthetic spawn failure")):
            result = e.run(self.plan, path, self.root / "out", e.digest(path))
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["failure"], "synthetic spawn failure")
        self.assertTrue((self.root / "out" / "receipt.json").is_file())

    @unittest.skipUnless(sys.platform == "linux", "Linux cgroup preflight test")
    def test_host_mismatch_fails_before_gpu_access(self):
        with patch.object(e.socket, "gethostname", return_value="unexpected-host"), patch.object(e.subprocess, "run", side_effect=AssertionError("unexpected GPU probe")):
            with self.assertRaisesRegex(e.Blocked, "hostname mismatch"):
                e.runtime_guard(self.plan)

    @unittest.skipUnless(sys.platform == "linux", "POSIX process-session timeout test")
    def test_real_timeout_retains_all_candidates_and_reaps_child(self):
        path = self.save()
        real_popen = subprocess.Popen
        children = []
        def sleeping_child(*args, **kwargs):
            process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(process)
            return process
        with patch.object(e, "runtime_guard"), patch.object(e.subprocess, "Popen", side_effect=sleeping_child):
            result = e.run(self.plan, path, self.root / "out", e.digest(path))
        self.assertEqual(result["state"], "TIMED_OUT")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertTrue(all(c["state"] == "NOT_RUN" for c in result["candidates"]))
        self.assertEqual(children[0].returncode, -9)
        self.assertFalse(result["production_eligible"])
        with self.assertRaises(e.Blocked):
            e.run(self.plan, path, self.root / "out", e.digest(path))

    @unittest.skipUnless(sys.platform == "linux", "POSIX process-session timeout test")
    def test_timeout_with_partial_event_preserves_primary_outcome(self):
        path = self.save()
        real_popen = subprocess.Popen
        children = []
        output = self.root / "out"
        def partial_event_child(*args, **kwargs):
            # Deterministic partial append at termination; no timing-dependent
            # race between a child starting and its first filesystem write.
            (output / "events.jsonl").write_bytes(
                b'{"candidate":"base","state":"STARTED"}\n{"candidate":')
            child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(child)
            return child
        with patch.object(e, "runtime_guard"), patch.object(e.subprocess, "Popen", side_effect=partial_event_child):
            result = e.run(self.plan, path, output, e.digest(path))
        self.assertEqual(result["state"], "TIMED_OUT")
        self.assertEqual(result["event_errors"][0]["line"], 2)
        self.assertEqual([c["state"] for c in result["candidates"]], ["TIMED_OUT", "NOT_RUN"])
        self.assertEqual(children[0].returncode, -9)
        self.assertEqual(e.read_json(output / "receipt.json"), result)
        self.assertIn("events.jsonl", result["files"])
        self.assertFalse(result["production_eligible"])

    @unittest.skipUnless(sys.platform == "linux", "POSIX supervision test")
    def test_partial_event_without_timeout_still_fails(self):
        path = self.save()
        output = self.root / "out"
        real_popen = subprocess.Popen
        def malformed_event_child(*args, **kwargs):
            (output / "events.jsonl").write_bytes(b'{"candidate":')
            return real_popen([sys.executable, "-c", "pass"], **kwargs)
        with patch.object(e, "runtime_guard"), patch.object(e.subprocess, "Popen", side_effect=malformed_event_child):
            result = e.run(self.plan, path, output, e.digest(path))
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["event_errors"][0]["line"], 1)


if __name__ == "__main__":
    unittest.main()
