"""Byte-identity, remote-code, and dispatch gates for the hosted CPU smoke."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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

    def test_synthetic_input_has_a_stable_non_eval_dataset_digest(self):
        self.assertEqual(smoke.INPUT_DATASET_BYTES, b'{"prompt":"Two plus two equals"}\n')
        self.assertEqual(smoke.INPUT_DATASET_SHA256,
                         hashlib.sha256(smoke.INPUT_DATASET_BYTES).hexdigest())
        self.assertEqual(smoke.SEED, 0)

    def test_executed_harness_must_match_checked_out_blob(self):
        harness = self.parent / "hosted_cpu_smoke.py"
        harness.write_bytes(b"reviewed harness bytes")
        with patch.object(smoke, "__file__", str(harness)), \
                patch.object(smoke.subprocess, "run",
                             return_value=SimpleNamespace(stdout=b"reviewed harness bytes")):
            self.assertEqual(smoke._harness_sha256(), hashlib.sha256(harness.read_bytes()).hexdigest())
        harness.write_bytes(b"changed after checkout")
        with patch.object(smoke, "__file__", str(harness)), \
                patch.object(smoke.subprocess, "run",
                             return_value=SimpleNamespace(stdout=b"reviewed harness bytes")):
            with self.assertRaisesRegex(smoke.SmokeError, "executed harness"):
                smoke._harness_sha256()

    def test_run_links_only_a_matching_successful_prepare_receipt(self):
        source = {"repository": "szl-holdings/szl-forge", "source_sha": "a" * 40,
                  "run_id": "123", "run_attempt": "1"}
        snapshot = {"model_type": "llama", "file_sha256": {"model.safetensors": "b" * 64}}
        prior = {"schema": smoke.SCHEMA, "stage": "PREPARE", "evidence_class": "MEASURED",
                 "disposition": "MEASURED", "source": source, "model_id": smoke.MODEL_ID,
                 "model_revision": smoke.MODEL_REVISION, "base_revision": smoke.MODEL_REVISION,
                 "dataset_sha256": smoke.INPUT_DATASET_SHA256, "dataset_evidence_class": "SAMPLE",
                 "dataset_scope": "SAMPLE_ONE_RECORD_SYNTHETIC_INFERENCE_INPUT_ONLY",
                 "harness_sha256": "c" * 64, "seed": smoke.SEED,
                 "seed_scope": "PYTHON_AND_TORCH_GREEDY_WORKER_ONLY",
                 "loss": {"evidence_class": "UNAVAILABLE", "status": "NOT_RUN", "reason": "INFERENCE_ONLY"},
                 "metrics": smoke._empty_metrics(), "snapshot": snapshot,
                 "wheel_sha256": smoke.WHEEL_SHA256, "training_eligible": False,
                 "publication_eligible": False, "execution_authority": False,
                 "inference_attempted": False, "held_out_gate": "NOT_RUN",
                 "numerical_parity": "NOT_RUN", "signing_status": "UNSIGNED"}
        prepare = self.parent / "PREPARE.json"
        smoke_receipt = self.parent / "SMOKE.json"
        with self.assertRaisesRegex(smoke.SmokeError, "missing"):
            smoke._prepare_receipt_sha256(smoke_receipt, source, "c" * 64, snapshot)
        encoded = (json.dumps(prior, sort_keys=True) + "\n").encode()
        prepare.write_bytes(encoded)
        self.assertEqual(smoke._prepare_receipt_sha256(smoke_receipt, source, "c" * 64, snapshot),
                         hashlib.sha256(encoded).hexdigest())
        for key, value in (("stage", "RUN"), ("disposition", "BLOCKED"),
                           ("source", {**source, "run_id": "other"}),
                           ("dataset_sha256", "d" * 64), ("harness_sha256", "e" * 64),
                           ("loss", {"evidence_class": "MEASURED", "value": 0}),
                           ("metrics", {"evidence_class": "MEASURED"}),
                           ("training_eligible", True)):
            with self.subTest(key=key):
                changed = dict(prior, **{key: value})
                prepare.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(smoke.SmokeError, "PREPARE receipt differs"):
                    smoke._prepare_receipt_sha256(smoke_receipt, source, "c" * 64, snapshot)


if __name__ == "__main__":
    unittest.main()
