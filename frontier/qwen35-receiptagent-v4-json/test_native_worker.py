# SPDX-License-Identifier: Apache-2.0
"""SIMULATED protocol fixtures and bounded real CPU autograd, not Qwen/GPU proof."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_worker as worker
import training_tokens as tokens


def retained_fixture():
    rows = []
    ledgers = []
    for index in range(360):
        ids = (1, 2, 3) if index % 2 else (2, 3, 4, 5, 6)
        labels = (-100, -100, 3) if index % 2 else (-100, -100, -100, 5, 6)
        row = tokens.TrainingTensorRow(ids, (1,) * len(ids), labels)
        rows.append(row)
        supervised = sum(value != -100 for value in labels)
        ledgers.append(tokens.RowLedger(index, len(ids), len(ids), supervised - 1, 1,
                                        "a" * 64, "b" * 64,
                                        worker._sha((row.input_ids, row.attention_mask, row.labels))))
    report = tokens.TokenReport(*("a" * 64 for _ in range(4)), 8,
                                 sum(len(row.input_ids) for row in rows), tuple(ledgers),
                                 worker._sha(tuple((row.index, row.tensors_sha256) for row in ledgers)))
    return tokens.RetainedTrainingTokens(tuple(rows), report)


class PureWorkerTests(unittest.TestCase):
    def test_import_does_not_load_any_model_or_cuda_library(self):
        script = ("import sys; sys.path.insert(0, sys.argv[1]); import native_worker; "
                  "print(any(name in sys.modules for name in "
                  "('torch','peft','transformers','bitsandbytes','huggingface_hub')))")
        result = subprocess.run([sys.executable, "-c", script, str(Path(worker.__file__).parent)],
                                capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual(result.stdout.strip(), "False")

    def test_isolated_cli_without_release_denies_before_runtime_import(self):
        result = subprocess.run([sys.executable, "-I", "-B", worker.__file__,
                                 "--bound-run", "does-not-exist.json", "--owner-policy-sha256", "0" * 64,
                                 "--release-stdin"], input=b"", capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["state"], "BLOCKED")
        self.assertEqual(result.stderr, b"")

    def test_current_source_denies_with_release_and_no_framework_import(self):
        import native_runtime as runtime
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = {name: str(Path(worker.__file__).parent) for name in runtime.PATH_NAMES}
            paths.update(source_root=str(runtime.SOURCE_ROOT),
                         run_dir=str(directory / "unused-run"), output_dir=str(directory / "unused-output"),
                         python_executable=sys.executable)
            request = directory / "request.json"
            request.write_bytes(runtime.canonical({"schema": runtime.REQUEST_SCHEMA, "paths": paths}))
            script = ("import sys; sys.path.insert(0,sys.argv[1]); import native_worker; "
                      "code=native_worker.main(['--bound-run',sys.argv[2],'--owner-policy-sha256','0'*64,"
                      "'--release-stdin']); print('FRAMEWORKS_LOADED='+str(any(name in sys.modules "
                      "for name in ('torch','peft','transformers','bitsandbytes')))); sys.exit(code)")
            result = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                                     str(Path(worker.__file__).parent), str(request)],
                                    input=worker.RELEASE_MARKER, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            lines = result.stdout.decode("ascii").splitlines()
            self.assertEqual(json.loads(lines[0])["state"], "BLOCKED")
            self.assertEqual(lines[1], "FRAMEWORKS_LOADED=False")
            self.assertFalse((directory / "unused-run").exists())
            self.assertFalse((directory / "unused-output").exists())

    def test_loader_rejects_any_missing_unexpected_mismatched_or_error_keys(self):
        import native_runtime as runtime
        calls = []
        run = SimpleNamespace(model_dir=Path("/qualified-model"), recipe=runtime.NATIVE_RECIPE,
                              qualification={"loader_" + name: [] for name in
                                             ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")})
        fake_torch = SimpleNamespace(bfloat16="BFLOAT16_TEST_DOUBLE")
        for fault in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            info = {name: [] for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
            info[fault] = ["SIMULATED_FAULT"]

            class Qwen3_5ForConditionalGeneration:
                def __init__(self):
                    self.is_loaded_in_4bit = True
                    self.config = SimpleNamespace(model_type="qwen3_5",
                                                   architectures=["Qwen3_5ForConditionalGeneration"])

                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    calls.append((args, kwargs))
                    return cls(), info

            transformers = SimpleNamespace(Qwen3_5ForConditionalGeneration=Qwen3_5ForConditionalGeneration,
                                            BitsAndBytesConfig=lambda **kwargs: kwargs)
            with self.subTest(fault=fault), self.assertRaisesRegex(worker.WorkerError, "KEY_COVERAGE"):
                worker.load_native_model(run, fake_torch, transformers, None, None)
        self.assertEqual(len(calls), 4)
        for args, kwargs in calls:
            self.assertEqual(args, (str(run.model_dir),))
            self.assertIs(kwargs["local_files_only"], True)
            self.assertIs(kwargs["trust_remote_code"], False)
            self.assertIs(kwargs["use_safetensors"], True)
            self.assertIs(kwargs["output_loading_info"], True)
            self.assertEqual(kwargs["quantization_config"]["bnb_4bit_quant_type"], "nf4")

    def test_export_extra_or_missing_state_is_rejected_before_artifact_writes(self):
        # Mock only the data-only serializer import. No Torch, PEFT, weights or
        # GPU is needed to test rejection before any file creation.
        fake_serializer = SimpleNamespace(save=lambda *args, **kwargs: b"SIMULATED_SAFETENSORS")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, {"safetensors.torch": fake_serializer}):
            run = SimpleNamespace(output_dir=Path(temporary))
            for state in ({}, {"extra.weight": object()}):
                peft = SimpleNamespace(get_peft_model_state_dict=lambda *args, **kwargs: state)
                with self.subTest(state=state), self.assertRaisesRegex(worker.WorkerError, "EXPORT_PARAMETER"):
                    worker.save_adapter(run, object(), ("target.lora_A.default.weight",), peft, None)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_release_is_exact_sequence_not_a_bool_or_arbitrary_saved_report(self):
        worker.read_supervisor_release(io.BytesIO(worker.RELEASE_MARKER))
        for raw in (b"", b"true", b"{}", worker.RELEASE_MARKER[:-1], worker.RELEASE_MARKER.lower()):
            with self.subTest(raw=raw), self.assertRaises(worker.WorkerError):
                worker.read_supervisor_release(io.BytesIO(raw))

    def test_worker_reads_own_cgroup_v2_and_exact_resource_barriers(self):
        run = SimpleNamespace(nonce="a" * 64, payload={"bounds": {"host_memory_bytes": 1234}})
        with tempfile.TemporaryDirectory() as temporary, patch.object(worker.sys, "platform", "linux"):
            root = Path(temporary)
            proc = root / "proc"
            (proc / "self").mkdir(parents=True)
            location = "/user.slice/szl-ra4-" + run.nonce + ".service"
            (proc / "self" / "cgroup").write_bytes(("0::" + location + "\n").encode("ascii"))
            cgroup_root = root / "cgroups"
            current = cgroup_root / location.lstrip("/")
            current.mkdir(parents=True)
            for name, value in (("memory.max", b"1234\n"), ("memory.swap.max", b"0\n"),
                                 ("pids.max", b"128\n")):
                (current / name).write_bytes(value)
            worker.verify_linux_containment(run, proc=proc, cgroup_root=cgroup_root)
            (current / "memory.max").write_bytes(b"max\n")
            with self.assertRaisesRegex(worker.WorkerError, "RESOURCE_BARRIER"):
                worker.verify_linux_containment(run, proc=proc, cgroup_root=cgroup_root)
            (proc / "self" / "cgroup").write_bytes(b"0::/other.service\n")
            with self.assertRaisesRegex(worker.WorkerError, "OWNED_CGROUP"):
                worker.verify_linux_containment(run, proc=proc, cgroup_root=cgroup_root)

    def test_target_inventory_rejects_expansion_missing_or_nonquantized_targets(self):
        class Linear4bit:
            pass

        name = "model.language_model.layers.0.self_attn.q_proj"
        foreign = "model.visual.layers.0.q_proj"
        model = SimpleNamespace(named_modules=lambda: [(name, Linear4bit()), (foreign, object())])
        self.assertEqual(worker._target_inventory(model, [name], ["q_proj"], Linear4bit), (name,))
        for qualified in ([], [foreign], [name, name]):
            with self.subTest(qualified=qualified), self.assertRaises(worker.WorkerError):
                worker._target_inventory(model, qualified, ["q_proj"], Linear4bit)
        model = SimpleNamespace(named_modules=lambda: [(name, object())])
        with self.assertRaisesRegex(worker.WorkerError, "NOT_NF4"):
            worker._target_inventory(model, [name], ["q_proj"], Linear4bit)

    def test_output_writer_is_exclusive_capped_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker-result.json"
            self.assertEqual(worker._write_exclusive(path, b"{}", 2), worker.hashlib.sha256(b"{}").hexdigest())
            with self.assertRaises(FileExistsError):
                worker._write_exclusive(path, b"[]", 2)
            self.assertEqual(path.read_bytes(), b"{}")
            with self.assertRaisesRegex(worker.WorkerError, "CAP_EXCEEDED"):
                worker._write_exclusive(Path(temporary) / "too-large", b"123", 2)

    def test_adapter_configuration_serializer_rejects_arbitrary_objects(self):
        self.assertEqual(worker._jsonable({"target_modules": {"b", "a"}, "dropout": 0}),
                         {"target_modules": ["a", "b"], "dropout": 0})
        with self.assertRaisesRegex(worker.WorkerError, "NOT_PRIMITIVE"):
            worker._jsonable({"payload": object()})

    def test_local_model_closure_requires_exact_safetensors_only_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {"config.json": b"{}", "model.safetensors": b"SIMULATED_DATA_NOT_WEIGHTS"}
            for name, raw in files.items():
                (root / name).write_bytes(raw)
            closure = [{"id": name, "sha256": worker.hashlib.sha256(raw).hexdigest()}
                       for name, raw in files.items()]
            run = SimpleNamespace(model_dir=root, payload={"base": {"assets_closure": closure}})
            worker.verify_local_model_assets(run)
            (root / "model.safetensors").write_bytes(b"CHANGED_SIMULATED_DATA")
            with self.assertRaisesRegex(worker.WorkerError, "HASH_MISMATCH"):
                worker.verify_local_model_assets(run)
            (root / "model.safetensors").write_bytes(files["model.safetensors"])
            (root / "pytorch_model.bin").write_bytes(b"PICKLE_FORBIDDEN")
            with self.assertRaisesRegex(worker.WorkerError, "SET_MISMATCH"):
                worker.verify_local_model_assets(run)
            closure.append({"id": "pytorch_model.bin", "sha256": "a" * 64})
            with self.assertRaisesRegex(worker.WorkerError, "FORBIDDEN"):
                worker.verify_local_model_assets(run)

    def test_local_sharded_index_must_bind_exact_qualified_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = "model-00001-of-00001.safetensors"
            files = {"config.json": b"{}", shard: b"SIMULATED_SHARD",
                     "model.safetensors.index.json": json.dumps({"weight_map": {"weight": shard}}).encode()}
            for name, raw in files.items():
                (root / name).write_bytes(raw)
            closure = [{"id": name, "sha256": worker.hashlib.sha256(raw).hexdigest()}
                       for name, raw in files.items()]
            run = SimpleNamespace(model_dir=root, payload={"base": {"assets_closure": closure}})
            worker.verify_local_model_assets(run)
            files["model.safetensors.index.json"] = b'{"weight_map":{"weight":"../other.safetensors"}}'
            (root / "model.safetensors.index.json").write_bytes(files["model.safetensors.index.json"])
            closure[-1]["sha256"] = worker.hashlib.sha256(files["model.safetensors.index.json"]).hexdigest()
            with self.assertRaisesRegex(worker.WorkerError, "INDEX_CLOSURE"):
                worker.verify_local_model_assets(run)

    def test_fresh_mask_eos_thinking_template_and_special_tokens_are_signed(self):
        from test_training_tokens import binding
        policy = binding()
        values = {"template_sha256": policy.template_sha256,
                  "special_tokens_sha256": worker._sha(policy.special_tokens),
                  "mask_policy_sha256": policy.masking_sha256, "eos_policy_sha256": policy.eos_sha256,
                  "thinking_policy_sha256": policy.thinking_sha256}
        run = SimpleNamespace(payload={"tokens": dict(values)})
        worker.verify_token_policy(run, policy)
        for name in values:
            run.payload["tokens"] = {**values, name: "0" * 64}
            with self.subTest(name=name), self.assertRaisesRegex(worker.WorkerError, "TOKEN_POLICY"):
                worker.verify_token_policy(run, policy)

    def test_dependency_labels_cannot_replace_unavailable_byte_identity_adapter(self):
        with self.assertRaisesRegex(worker.WorkerError, "DEPENDENCY_IDENTITY_UNAVAILABLE"):
            worker.verify_dependency_identity(SimpleNamespace(qualification={"dependencies_qualified": True}))

    def test_tokenizer_asset_closure_is_exact_and_then_bytes_are_checked(self):
        import probe_base_tokenizer as probe
        closure = [{"id": name, "sha256": digest} for name, digest in probe.ASSET_SHA256.items()]
        run = SimpleNamespace(tokenizer_dir=Path("/SIMULATED_ONLY"), payload={"tokens": {"assets_closure": closure}})
        with patch.object(probe, "_read_assets", return_value={}) as read:
            worker.verify_token_asset_closure(run)
            read.assert_called_once_with(run.tokenizer_dir)
            closure[0] = {**closure[0], "sha256": "0" * 64}
            with self.assertRaisesRegex(worker.WorkerError, "SIGNED_TOKEN_ASSET"):
                worker.verify_token_asset_closure(run)
            self.assertEqual(read.call_count, 1)

    def test_exact_360_rows_and_causal_counts(self):
        retained = retained_fixture()
        self.assertEqual(worker.checked_rows(retained), (2, 1) * 180)
        self.assertEqual(sum(worker.checked_rows(retained)), 540)

    def test_retained_tampering_and_coverage_rejected(self):
        retained = retained_fixture()
        wrong_row = replace(retained.rows[0], labels=(-100, -100, -100, -100, 6))
        wrong_counts = replace(retained.report.rows[0], assistant_tokens=0)
        wrong_order = replace(retained.report.rows[0], index=1)
        cases = (
            replace(retained, rows=retained.rows[:-1]),
            replace(retained, rows=(wrong_row,) + retained.rows[1:]),
            replace(retained, report=replace(retained.report,
                                            rows=(wrong_counts,) + retained.report.rows[1:])),
            replace(retained, report=replace(retained.report,
                                            rows=(wrong_order,) + retained.report.rows[1:])),
            replace(retained, report=replace(retained.report, ordered_tensors_sha256="0" * 64)),
            replace(retained, report=replace(retained.report, total_tokens=1)),
        )
        for value in cases:
            with self.subTest(value=repr(value.report)[:30]), self.assertRaises(worker.WorkerError):
                worker.checked_rows(value)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "Existing CPU Torch unavailable; no dependency is installed by these tests")
class CPUAutogradTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def model(self):
        torch = self.torch

        class TinyCPUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 100)
                self.base = torch.nn.Parameter(torch.ones(8), requires_grad=False)

            def forward(self, input_ids, attention_mask, use_cache=False):
                if use_cache is not False:
                    raise AssertionError("Cache must be disabled")
                return SimpleNamespace(logits=self.adapter[input_ids] + self.base)

        return TinyCPUModel()

    def run_update(self, model=None, optimizer=None, retained=None, **changes):
        model = self.model() if model is None else model
        optimizer = self.torch.optim.AdamW([model.adapter], lr=1e-4, weight_decay=0) if optimizer is None else optimizer
        options = dict(microbatch_rows=3, pad_id=0, max_grad_norm=1.0,
                       expected_names=("adapter",), torch=self.torch, device="cpu",
                       before_update=lambda: None)
        options.update(changes)
        result = worker.one_update(model, optimizer,
                                   retained_fixture() if retained is None else retained, **options)
        return model, optimizer, result

    def test_exact_collation_preserves_ids_labels_mask_and_only_right_pads(self):
        retained = retained_fixture()
        rows = retained.rows[:2]
        result = worker.collate_exact(rows, pad_id=0, torch=self.torch, device="cpu")
        for index, row in enumerate(rows):
            width = len(row.input_ids)
            self.assertEqual(tuple(result["input_ids"][index, :width].tolist()), row.input_ids)
            self.assertEqual(tuple(result["labels"][index, :width].tolist()), row.labels)
            self.assertEqual(tuple(result["attention_mask"][index, :width].tolist()), row.attention_mask)
            self.assertEqual(result["labels"][index, width:].tolist(), [-100] * (5 - width))
        self.assertEqual(rows, retained.rows[:2])

    def test_one_update_equals_full_token_weighted_reference_not_row_average(self):
        torch = self.torch
        model = self.model()
        reference = self.model()
        retained = retained_fixture()
        optimizer = torch.optim.AdamW([model.adapter], lr=1e-4, weight_decay=0)
        reference_optimizer = torch.optim.AdamW([reference.adapter], lr=1e-4, weight_decay=0)
        batch = worker.collate_exact(retained.rows, pad_id=0, torch=torch, device="cpu")
        outputs = reference(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(outputs.logits[:, :-1].reshape(-1, 8),
                                                 batch["labels"][:, 1:].reshape(-1),
                                                 ignore_index=-100, reduction="sum") / 540
        loss.backward()
        torch.nn.utils.clip_grad_norm_([reference.adapter], 1.0, error_if_nonfinite=True)
        reference_optimizer.step()
        callbacks = []
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            _, _, result = self.run_update(model=model, optimizer=optimizer, retained=retained,
                                           after_microbatch=callbacks.append)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(callbacks, list(range(3, 361, 3)))
        self.assertEqual(result["rows"], 360)
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertEqual(result["supervised_tokens"], 540)
        self.assertAlmostEqual(result["token_weighted_loss"], loss.item(), places=5)
        self.assertTrue(torch.allclose(model.adapter, reference.adapter, rtol=1e-5, atol=1e-7))
        self.assertTrue(torch.equal(model.base, torch.ones(8)))
        self.assertEqual(set(result), {"rows", "optimizer_updates", "supervised_tokens",
                                       "token_weighted_loss", "gradient_norm_before_clip",
                                       "ordered_tensors_sha256", "row_order_sha256"})

    def test_optimizer_missing_extra_or_duplicate_parameters_rejected_before_forward(self):
        model = self.model()
        optimizers = (
            self.torch.optim.AdamW([model.base], lr=1e-4),
            self.torch.optim.AdamW([model.adapter, model.base], lr=1e-4),
        )
        for optimizer in optimizers:
            with self.subTest(optimizer=optimizer), self.assertRaisesRegex(worker.WorkerError, "OPTIMIZER"):
                self.run_update(model=model, optimizer=optimizer)
        duplicate = SimpleNamespace(param_groups=[{"params": [model.adapter, model.adapter]}])
        with self.assertRaisesRegex(worker.WorkerError, "OPTIMIZER"):
            worker.verify_optimizer_coverage(model, duplicate, ("adapter",))

    def test_trainable_base_or_wrong_declared_lora_coverage_rejected(self):
        model = self.model()
        model.base.requires_grad_(True)
        with self.assertRaisesRegex(worker.WorkerError, "TRAINABLE"):
            self.run_update(model=model)
        with self.assertRaisesRegex(worker.WorkerError, "TRAINABLE"):
            self.run_update(expected_names=("wrong",))

    def test_nonfinite_loss_rejected_without_update(self):
        model = self.model()
        with self.torch.no_grad():
            model.adapter[2, 0] = float("nan")
        optimizer = self.torch.optim.AdamW([model.adapter], lr=1e-4)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                self.assertRaisesRegex(worker.WorkerError, "NONFINITE_LOSS"):
            self.run_update(model=model, optimizer=optimizer)
        self.assertEqual(step.call_count, 0)

    def test_nonfinite_gradient_rejected_without_update(self):
        model = self.model()
        model.adapter.register_hook(lambda gradient: gradient * float("nan"))
        optimizer = self.torch.optim.AdamW([model.adapter], lr=1e-4)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                self.assertRaisesRegex(worker.WorkerError, "NONFINITE_GRADIENT"):
            self.run_update(model=model, optimizer=optimizer)
        self.assertEqual(step.call_count, 0)

    def test_final_revalidation_failure_prevents_update(self):
        model = self.model()
        before = model.adapter.detach().clone()
        optimizer = self.torch.optim.AdamW([model.adapter], lr=1e-4)

        def deny():
            raise worker.WorkerError("LEASE_OR_ADMISSION_EXPIRED")

        with patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                self.assertRaisesRegex(worker.WorkerError, "LEASE_OR_ADMISSION_EXPIRED"):
            self.run_update(model=model, optimizer=optimizer, before_update=deny)
        self.assertEqual(step.call_count, 0)
        self.assertTrue(self.torch.equal(before, model.adapter))

    def test_invalid_coverage_or_norm_rejected(self):
        for changes in ({"microbatch_rows": 7}, {"microbatch_rows": True},
                        {"max_grad_norm": float("nan")}, {"max_grad_norm": 0},
                        {"before_update": None}):
            with self.subTest(changes=changes), self.assertRaises(worker.WorkerError):
                self.run_update(**changes)


if __name__ == "__main__":
    unittest.main()
