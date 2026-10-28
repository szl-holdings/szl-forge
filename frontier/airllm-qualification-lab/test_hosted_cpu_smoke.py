"""Byte-identity, remote-code, and dispatch gates for the hosted CPU smoke."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import hosted_cpu_smoke as smoke


class HostedCpuSmokeControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)

    def test_selected_model_is_a_fixed_eight_file_sha256_binding(self):
        self.assertEqual(len(smoke.FILES), 8)
        self.assertEqual(sum(size for size, _ in smoke.FILES.values()), 272437573)
        self.assertTrue(all(len(digest) == 64 for _, digest in smoke.FILES.values()))
        self.assertEqual(smoke.FILES["model.safetensors"][1],
                         "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c")
        self.assertFalse(any(name.endswith((".py", ".bin", ".pkl")) for name in smoke.FILES))

    def test_snapshot_rejects_extra_file_and_nested_auto_map(self):
        snapshot = self.parent / "snapshot"
        snapshot.mkdir()
        fixtures = {
            "config.json": json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}).encode(),
            "tokenizer_config.json": json.dumps({"tokenizer_class": "GPT2Tokenizer"}).encode(),
            "model.safetensors": b"fixture-only-not-real-model",
        }
        for name, content in fixtures.items():
            (snapshot / name).write_bytes(content)
        binding = {name: (len(content), hashlib.sha256(content).hexdigest()) for name, content in fixtures.items()}
        with patch.object(smoke, "FILES", binding):
            self.assertEqual(smoke._verify_snapshot(snapshot)["architecture"], "LlamaForCausalLM")
            (snapshot / "modeling_evil.py").write_text("pass", encoding="utf-8")
            with self.assertRaisesRegex(smoke.SmokeError, "allowlist"):
                smoke._verify_snapshot(snapshot)
            (snapshot / "modeling_evil.py").unlink()
            fixtures["tokenizer_config.json"] = json.dumps({"tokenizer_class": "GPT2Tokenizer",
                                                               "nested": {"auto_map": {}}}).encode()
            (snapshot / "tokenizer_config.json").write_bytes(fixtures["tokenizer_config.json"])
            binding["tokenizer_config.json"] = (len(fixtures["tokenizer_config.json"]),
                                                  hashlib.sha256(fixtures["tokenizer_config.json"]).hexdigest())
            with self.assertRaisesRegex(smoke.SmokeError, "auto_map"):
                smoke._verify_snapshot(snapshot)

    def test_tampered_snapshot_bytes_are_refused(self):
        snapshot = self.parent / "snapshot"
        snapshot.mkdir()
        path = snapshot / "config.json"
        path.write_bytes(b"tampered")
        with patch.object(smoke, "FILES", {"config.json": (len(b"tampered"), "0" * 64)}):
            with self.assertRaisesRegex(smoke.SmokeError, "byte/hash mismatch"):
                smoke._verify_snapshot(snapshot)

    def test_manual_dispatch_requires_exact_main_checkout(self):
        with patch.dict(smoke.os.environ, {"GITHUB_REPOSITORY": "szl-holdings/szl-forge",
                                               "GITHUB_REF": "refs/heads/feature"}):
            with self.assertRaisesRegex(smoke.SmokeError, "main"):
                smoke._source_binding()
        with patch.dict(smoke.os.environ, {"GITHUB_REPOSITORY": "elsewhere/fork",
                                               "GITHUB_REF": "refs/heads/main"}):
            with self.assertRaisesRegex(smoke.SmokeError, "canonical"):
                smoke._source_binding()

    def test_prepare_requires_fresh_direct_runner_child(self):
        with patch.dict(smoke.os.environ, {"RUNNER_TEMP": str(self.parent)}):
            root = self.parent / "one-task"
            receipt = self.parent / "receipts" / "PREPARE.json"
            smoke._runner_paths(root, receipt, prepare=True)
            root.mkdir()
            with self.assertRaisesRegex(smoke.SmokeError, "fresh"):
                smoke._runner_paths(root, receipt, prepare=True)
            with self.assertRaisesRegex(smoke.SmokeError, "outside"):
                smoke._runner_paths(root, root / "receipt.json", prepare=False)

    def _prepare_receipt(self, source_sha: str, **overrides) -> Path:
        binding = smoke._generation_binding()
        document = {
            "airllm_source_commit": smoke.AIRLLM_SOURCE_COMMIT,
            "delete_original": False,
            "disposition": "MEASURED",
            "evidence_class": "MEASURED",
            "generation": smoke._generation_record(),
            "harness_sha256": binding["harness_sha256"],
            "input_sha256": binding["input_sha256"],
            "dataset_sha256": smoke.INPUT_DATASET_SHA256,
            "dataset_evidence_class": smoke.DATASET_EVIDENCE_CLASS,
            "dataset_scope": smoke.DATASET_SCOPE,
            "model_id": smoke.MODEL_ID,
            "model_revision": smoke.MODEL_REVISION,
            "base_revision": smoke.MODEL_REVISION,
            "seed": smoke.GENERATION_SEED,
            "loss": smoke._loss_record(),
            "metrics": smoke._empty_metrics(),
            "hardware": {"runner_os": "ISOLATED"},
            "schema": smoke.SCHEMA,
            "source": {"repository": "szl-holdings/szl-forge", "source_sha": source_sha},
            "stage": "PREPARE",
        }
        document.update(overrides)
        path = self.parent / "PREPARE.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_run_receipt_binds_prepare_bytes_input_seed_and_harness(self):
        binding = smoke._generation_binding()
        self.assertEqual(binding["seed"], 0)
        self.assertFalse(binding["seed_applied"])
        self.assertFalse(binding["do_sample"])
        self.assertEqual(binding["input_sha256"], hashlib.sha256(smoke.PROMPT.encode("utf-8")).hexdigest())
        self.assertEqual(binding["dataset_sha256"], smoke.INPUT_DATASET_SHA256)
        self.assertEqual(binding["harness_sha256"], hashlib.sha256(Path(smoke.__file__).read_bytes()).hexdigest())
        source_sha = "a" * 40
        path = self._prepare_receipt(source_sha)
        digest = smoke._bind_prepare_receipt(path, source_sha)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(smoke.SmokeError, "source"):
            smoke._bind_prepare_receipt(path, "b" * 40)
        broken = self._prepare_receipt(source_sha, harness_sha256="0" * 64)
        with self.assertRaisesRegex(smoke.SmokeError, "harness"):
            smoke._bind_prepare_receipt(broken, source_sha)
        for key, value in (("dataset_sha256", "0" * 64), ("seed", 7),
                           ("loss", {"evidence_class": "MEASURED", "value": 0}),
                           ("metrics", {"evidence_class": "MEASURED"})):
            broken = self._prepare_receipt(source_sha, **{key: value})
            with self.assertRaisesRegex(smoke.SmokeError, "harness or input hash"):
                smoke._bind_prepare_receipt(broken, source_sha)
        broken = self._prepare_receipt(source_sha, base_revision="0" * 40)
        with self.assertRaisesRegex(smoke.SmokeError, "model binding"):
            smoke._bind_prepare_receipt(broken, source_sha)
        for payload in (b"[]", b'"receipt"', b"null", b"false", b"7", b"1.5"):
            path.write_bytes(payload)
            with self.assertRaisesRegex(smoke.SmokeError, "JSON object"):
                smoke._bind_prepare_receipt(path, source_sha)
        with patch.object(smoke, "DO_SAMPLE", True):
            with self.assertRaisesRegex(smoke.SmokeError, "greedy"):
                smoke._generation_binding()
        result = smoke._token_result([7, 9], 1, 1.2345, 12)
        self.assertEqual(result["generated_token_count"], 1)
        self.assertEqual(result["input_sha256"], binding["input_sha256"])
        self.assertEqual(result["harness_sha256"], binding["harness_sha256"])
        self.assertEqual(result["dataset_sha256"], smoke.INPUT_DATASET_SHA256)
        self.assertTrue(result["seed_applied"])
        self.assertFalse(result["execution_authority"])
        workflow = Path(smoke.__file__).resolve().parents[2] / ".github" / "workflows" / "airllm-hosted-cpu-smoke.yml"
        text = workflow.read_text(encoding="utf-8")
        self.assertIn('--prepare-receipt "$RECEIPT_ROOT/PREPARE.json"', text)

    def test_prepare_run_seed_transition_and_receipt_chain(self):
        source = {"repository": "szl-holdings/szl-forge", "source_sha": "a" * 40,
                  "run_id": "123", "run_attempt": "1"}
        root = self.parent / "task"
        prepare = self.parent / "receipts" / "PREPARE.json"
        run = self.parent / "receipts" / "SMOKE.json"
        failed_run = self.parent / "receipts" / "FAILED.json"
        with patch.object(smoke, "_runner_paths"), \
                patch.object(smoke, "_source_binding", return_value=source), \
                patch.object(smoke, "_hardware", return_value={"runner_os": "ISOLATED"}), \
                patch.object(smoke, "_capacity", return_value={"bound_class": "MODELED"}), \
                patch.object(smoke, "_download"), \
                patch.object(smoke, "_verify_snapshot", return_value={"model_type": "llama"}), \
                patch.object(smoke, "_verify_wheel", return_value={"version": smoke.AIRLLM_VERSION}), \
                patch.object(smoke, "_dependency_versions", return_value={}), \
                patch.object(smoke, "_marker", return_value={}):
            with patch.object(sys, "argv", [smoke.__file__, "--prepare", "--root", str(root),
                                            "--receipt", str(prepare)]):
                self.assertEqual(smoke.main(), 0)
            prepared = json.loads(prepare.read_text(encoding="utf-8"))
            self.assertFalse(prepared["generation"]["seed_applied"])
            self.assertFalse(prepared["seed_applied"])
            worker = smoke._token_result([7, 9], 1, 1.25, 123)
            with patch.object(smoke, "_run_worker", return_value=(0, json.dumps(worker))), \
                    patch.object(sys, "argv", [smoke.__file__, "--run", "--root", str(root),
                                               "--receipt", str(run), "--prepare-receipt", str(prepare)]):
                self.assertEqual(smoke.main(), 0)
            with patch.object(smoke, "_run_worker", side_effect=smoke.SmokeError("worker abort")), \
                    patch.object(sys, "argv", [smoke.__file__, "--run", "--root", str(root),
                                               "--receipt", str(failed_run), "--prepare-receipt", str(prepare)]):
                self.assertEqual(smoke.main(), 2)
        result = json.loads(run.read_text(encoding="utf-8"))
        self.assertTrue(result["generation"]["seed_applied"])
        self.assertEqual(result["previous_receipt_sha256"], hashlib.sha256(prepare.read_bytes()).hexdigest())
        self.assertEqual(result["metrics"]["generated_token_count"], 1)
        failed = json.loads(failed_run.read_text(encoding="utf-8"))
        self.assertEqual(failed["generation"]["seed_applied"], "UNKNOWN")
        self.assertEqual(failed["inference_run"], "UNKNOWN")
        self.assertTrue(failed["inference_attempted"])

    def test_non_object_prepare_json_writes_blocked_run_without_snapshot_or_worker(self):
        source_sha = "c" * 40
        cases = {
            "array": b"[]",
            "string": b'"receipt"',
            "null": b"null",
            "false": b"false",
            "integer": b"7",
            "float": b"1.5",
        }
        for name, payload in cases.items():
            with self.subTest(payload=name):
                runner = self.parent / name
                runner.mkdir()
                root = runner / "task"
                root.mkdir()
                receipt = runner / "RUN.json"
                prepare = runner / "PREPARE.json"
                prepare.write_bytes(payload)
                argv = [smoke.__file__, "--run", "--root", str(root), "--receipt", str(receipt),
                        "--prepare-receipt", str(prepare)]
                with patch.object(smoke, "_runner_paths") as paths, \
                        patch.object(smoke, "_source_binding", return_value={
                            "repository": "szl-holdings/szl-forge", "source_sha": source_sha,
                            "run_id": None, "run_attempt": None}) as source, \
                        patch.object(smoke, "_hardware", return_value={"runner_os": "ISOLATED"}) as hardware, \
                        patch.object(smoke, "_capacity", return_value={"bound_class": "ISOLATED"}) as capacity, \
                        patch.object(smoke, "_marker", return_value={}) as marker, \
                        patch.object(smoke, "_verify_snapshot", side_effect=AssertionError("snapshot")) as snapshot, \
                        patch.object(smoke, "_run_worker", side_effect=AssertionError("worker")) as worker, \
                        patch.object(sys, "argv", argv):
                    code = smoke.main()
                self.assertEqual(code, 2)
                paths.assert_called_once()
                source.assert_called_once()
                hardware.assert_called_once()
                capacity.assert_called_once()
                marker.assert_called_once()
                snapshot.assert_not_called()
                worker.assert_not_called()
                document = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(document["stage"], "RUN")
                self.assertEqual(document["disposition"], "BLOCKED")
                self.assertEqual(document["evidence_class"], "BLOCKED")
                self.assertEqual(document["inference_run"], "UNAVAILABLE")
                self.assertIs(document["inference_attempted"], False)
                self.assertIs(document["execution_authority"], False)
                self.assertIs(document["training_eligible"], False)
                self.assertIs(document["publication_eligible"], False)
                self.assertIn("SmokeError", document["reason"])
                self.assertIn("JSON object", document["reason"])
                self.assertEqual(document["prepare_receipt_sha256"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
