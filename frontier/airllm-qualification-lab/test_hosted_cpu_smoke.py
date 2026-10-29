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
        self.assertEqual(smoke.DEFAULT_PROFILE, "smollm2-135m-instruct")
        self.assertEqual(smoke._model_profile(smoke.DEFAULT_PROFILE)[:2],
                         (smoke.MODEL_ID, smoke.MODEL_REVISION))
        self.assertEqual(len(smoke.FILES), 8)
        self.assertEqual(sum(size for size, _ in smoke.FILES.values()), 272437573)
        self.assertTrue(all(len(digest) == 64 for _, digest in smoke.FILES.values()))
        self.assertEqual(smoke.FILES["model.safetensors"][1],
                         "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c")
        self.assertFalse(any(name.endswith((".py", ".bin", ".pkl")) for name in smoke.FILES))

    def test_360m_profile_is_a_distinct_pinned_eight_file_binding(self):
        model_id, revision, license_card, files = smoke._model_profile(smoke.PROFILE_360M)
        self.assertEqual(model_id, "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.assertEqual(revision, "a10cc1512eabd3dde888204e902eca88bddb4951")
        self.assertEqual(license_card, "apache-2.0")
        self.assertEqual(len(files), 8)
        self.assertEqual(sum(size for size, _ in files.values()), 727051918)
        self.assertEqual(files["model.safetensors"],
                         (723674912, "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"))
        self.assertEqual(set(files), set(smoke.FILES))
        self.assertFalse(any(name.endswith((".py", ".bin", ".pkl")) for name in files))
        with self.assertRaisesRegex(smoke.SmokeError, "not pinned"):
            smoke._model_profile("user-supplied-model")

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

    def _prepare_receipt(self, source_sha: str, *, profile_name: str = smoke.DEFAULT_PROFILE,
                         **overrides) -> Path:
        model_id, model_revision, _, _ = smoke._model_profile(profile_name)
        binding = smoke._generation_binding(profile_name=profile_name)
        document = {
            "airllm_source_commit": smoke.AIRLLM_SOURCE_COMMIT,
            "delete_original": False,
            "disposition": "MEASURED",
            "evidence_class": "MEASURED",
            "generation": smoke._generation_record(profile_name=profile_name),
            "harness_sha256": binding["harness_sha256"],
            "input_sha256": binding["input_sha256"],
            "dataset_sha256": smoke.INPUT_DATASET_SHA256,
            "dataset_evidence_class": smoke.DATASET_EVIDENCE_CLASS,
            "dataset_scope": smoke.DATASET_SCOPE,
            "model_profile": profile_name,
            "model_id": model_id,
            "model_revision": model_revision,
            "base_revision": model_revision,
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
        self.assertIn('type: choice', text)
        self.assertIn('default: smollm2-135m-instruct', text)
        self.assertIn('          - smollm2-360m-instruct', text)
        self.assertEqual(text.count('--profile "$MODEL_PROFILE"'), 2)
        self.assertIn('airllm-hosted-cpu-receipts-${{ inputs.model_profile }}-', text)

    def test_360m_prepare_run_worker_and_receipt_bindings(self):
        profile_name = smoke.PROFILE_360M
        source = {"repository": "szl-holdings/szl-forge", "source_sha": "a" * 40,
                  "run_id": "456", "run_attempt": "1"}
        root = self.parent / "task"
        prepare = self.parent / "receipts" / "PREPARE.json"
        run = self.parent / "receipts" / "SMOKE.json"
        wrong_profile_run = self.parent / "receipts" / "WRONG_PROFILE.json"
        with patch.object(smoke, "_runner_paths"), \
                patch.object(smoke, "_source_binding", return_value=source), \
                patch.object(smoke, "_hardware", return_value={"runner_os": "ISOLATED"}), \
                patch.object(smoke, "_capacity", return_value={"bound_class": "MODELED"}), \
                patch.object(smoke, "_download") as download, \
                patch.object(smoke, "_verify_snapshot", return_value={"model_type": "llama"}) as snapshot, \
                patch.object(smoke, "_verify_wheel", return_value={"version": smoke.AIRLLM_VERSION}), \
                patch.object(smoke, "_dependency_versions", return_value={}), \
                patch.object(smoke, "_marker", return_value={}):
            with patch.object(sys, "argv", [smoke.__file__, "--prepare", "--profile", profile_name,
                                            "--root", str(root), "--receipt", str(prepare)]):
                self.assertEqual(smoke.main(), 0)
            prepared = json.loads(prepare.read_text(encoding="utf-8"))
            self.assertEqual(prepared["model_profile"], profile_name)
            self.assertEqual(prepared["generation"]["model_profile"], profile_name)
            self.assertFalse(prepared["generation"]["seed_applied"])
            self.assertEqual(json.loads((root / smoke.MARKER).read_text())["model_profile"], profile_name)
            weight_url = ("https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/"
                          "a10cc1512eabd3dde888204e902eca88bddb4951/model.safetensors")
            self.assertTrue(any(call.args[0] == weight_url and call.args[2:] == smoke.FILES_360M["model.safetensors"]
                                for call in download.call_args_list))
            snapshot.assert_called_with(root / "snapshot", profile_name=profile_name)
            worker = smoke._token_result([7, 9], 1, 1.25, 123, profile_name=profile_name)
            with patch.object(smoke, "_run_worker", return_value=(0, json.dumps(worker))) as worker_call, \
                    patch.object(sys, "argv", [smoke.__file__, "--run", "--profile", profile_name,
                                               "--root", str(root), "--receipt", str(run),
                                               "--prepare-receipt", str(prepare)]):
                self.assertEqual(smoke.main(), 0)
            self.assertEqual(worker_call.call_args.kwargs["profile_name"], profile_name)
            with patch.object(smoke, "_run_worker", side_effect=AssertionError("cross-profile worker")) as blocked_worker, \
                    patch.object(sys, "argv", [smoke.__file__, "--run", "--root", str(root),
                                               "--receipt", str(wrong_profile_run),
                                               "--prepare-receipt", str(prepare)]):
                self.assertEqual(smoke.main(), 2)
            blocked_worker.assert_not_called()
        result = json.loads(run.read_text(encoding="utf-8"))
        self.assertEqual(result["model_profile"], profile_name)
        self.assertEqual(result["worker_result"]["model_profile"], profile_name)
        self.assertTrue(result["generation"]["seed_applied"])
        self.assertEqual(result["previous_receipt_sha256"], hashlib.sha256(prepare.read_bytes()).hexdigest())
        blocked = json.loads(wrong_profile_run.read_text(encoding="utf-8"))
        self.assertEqual(blocked["disposition"], "BLOCKED")
        self.assertIn("model binding differs", blocked["reason"])

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
