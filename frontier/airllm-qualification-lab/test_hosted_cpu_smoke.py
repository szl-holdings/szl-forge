"""Byte-identity, remote-code, and dispatch gates for the hosted CPU smoke."""

import hashlib
import json
from pathlib import Path
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
            "model_id": smoke.MODEL_ID,
            "model_revision": smoke.MODEL_REVISION,
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
        self.assertEqual(binding["seed"], None)
        self.assertFalse(binding["seed_applied"])
        self.assertFalse(binding["do_sample"])
        self.assertEqual(binding["input_sha256"], hashlib.sha256(smoke.PROMPT.encode("utf-8")).hexdigest())
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
        with patch.object(smoke, "DO_SAMPLE", True):
            with self.assertRaisesRegex(smoke.SmokeError, "greedy"):
                smoke._generation_binding()
        result = smoke._token_result([7, 9], 1, 1.2345, 12)
        self.assertEqual(result["generated_token_count"], 1)
        self.assertEqual(result["input_sha256"], binding["input_sha256"])
        self.assertEqual(result["harness_sha256"], binding["harness_sha256"])
        self.assertFalse(result["seed_applied"])
        self.assertFalse(result["execution_authority"])
        workflow = Path(smoke.__file__).resolve().parents[2] / ".github" / "workflows" / "airllm-hosted-cpu-smoke.yml"
        text = workflow.read_text(encoding="utf-8")
        self.assertIn('--prepare-receipt "$RECEIPT_ROOT/PREPARE.json"', text)


if __name__ == "__main__":
    unittest.main()
