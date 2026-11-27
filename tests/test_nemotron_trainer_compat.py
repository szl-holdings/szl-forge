"""Nemotron-H trainer boundary: LoRA targets, reasoning toggle, guardrails.

Pure-Python checks run without torch.  The tiny Nemotron-H CPU dry-run trains
two steps when torch, TRL, and a Transformers build with the native
``NemotronHForCausalLM`` are installed, and skips otherwise.  No hosted job is
launched and nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANO_4B = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NANO_4B_REV = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
NANO_9B = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
NANO_9B_REV = "6533e8de2c68e4536bf7c411d7a3ce5734111476"


def _load(name: str):
    path = ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"t_{name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"missing {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Leaf:
    def named_children(self):
        return []


def _mixer(class_name: str, children: list[str]):
    cls = type(class_name, (), {})
    obj = cls()
    obj.named_children = lambda: [(c, _Leaf()) for c in children]
    return obj


class _Config:
    def __init__(self, model_type: str, architectures: list[str]):
        self.model_type = model_type
        self.architectures = architectures


class _FakeModel:
    def __init__(self, config, mixers):
        self.config = config
        self._mixers = mixers

    def named_modules(self):
        yield "", self
        for index, mixer in enumerate(self._mixers):
            yield f"model.layers.{index}.mixer", mixer


def _hybrid(mamba_children=("in_proj", "conv1d", "norm", "out_proj")):
    return _FakeModel(
        _Config("nemotron_h", ["NemotronHForCausalLM"]),
        [
            _mixer("NemotronHMamba2Mixer", list(mamba_children)),
            _mixer("NemotronHMLP", ["up_proj", "down_proj", "act_fn"]),
            _mixer("NemotronHAttention", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ],
    )


class _Tokenizer:
    def __init__(self, template):
        self.chat_template = template
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        flag = kwargs.get("enable_thinking")
        return "|".join(m["content"] for m in messages) + f"|think={flag}"


class DenseQwenPathUnchanged(unittest.TestCase):
    def setUp(self) -> None:
        self.compat = _load("trainer_compat")

    def test_dense_targets_are_the_original_seven(self) -> None:
        model = _FakeModel(_Config("qwen2", ["Qwen2ForCausalLM"]), [])
        targets, evidence = self.compat.lora_target_modules(model)
        self.assertEqual(
            targets,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.assertEqual(evidence, {"family": "dense"})

    def test_dense_render_passes_no_reasoning_kwarg(self) -> None:
        tokenizer = _Tokenizer("{{ messages }}")
        row = {"messages": [{"role": "user", "content": "hi"}]}
        out = self.compat.render_chat_example(tokenizer, row)
        self.assertEqual(out["text"], "hi|think=None")
        self.assertEqual(
            tokenizer.calls[0], {"messages": row["messages"], "tokenize": False}
        )

    def test_dense_wave_still_uses_unsloth_and_the_fixed_list(self) -> None:
        source = (ROOT / "training" / "train_wave.py").read_text(encoding="utf-8")
        self.assertIn("FastLanguageModel.get_peft_model(", source)
        self.assertIn('use_gradient_checkpointing="unsloth"', source)
        for name in ("gate_proj", "up_proj", "down_proj"):
            self.assertIn(f'"{name}"', source)

    def test_qwen_is_not_a_nemotron_base(self) -> None:
        self.assertIsNone(self.compat.nemotron_base("Qwen/Qwen3.5-0.8B", "a" * 40))


class NemotronTargets(unittest.TestCase):
    def setUp(self) -> None:
        self.compat = _load("trainer_compat")

    def test_hybrid_targets_from_graph(self) -> None:
        targets, evidence = self.compat.lora_target_modules(
            _hybrid(), fused_mamba=False
        )
        self.assertEqual(
            targets,
            [
                "in_proj",
                "out_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.assertNotIn("conv1d", targets)
        self.assertEqual(evidence["layout"], {"mamba": 1, "attention": 1, "mlp": 1})

    def test_missing_projection_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            self.compat.lora_target_modules(
                _hybrid(("conv1d", "out_proj")), fused_mamba=False
            )

    def test_missing_out_proj_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            self.compat.lora_target_modules(
                _hybrid(("in_proj", "conv1d")), fused_mamba=False
            )

    def test_unknown_architecture_fails_closed(self) -> None:
        model = _FakeModel(_Config("gpt2", ["GPT2LMHeadModel"]), [])
        with self.assertRaises(RuntimeError):
            self.compat.lora_target_modules(model)

    def test_qwen35_targets_come_from_the_graph(self) -> None:
        leaves = list(self.compat.QWEN35_LORA_TARGETS)

        class _Qwen:
            def __init__(self) -> None:
                self.config = _Config("qwen3_5", ["Qwen3_5ForCausalLM"])

            def named_children(self):
                return [("layer", _Leaf())]

            def named_modules(self):
                yield "", self
                for name in leaves:
                    yield f"model.layers.0.{name}", _Leaf()

        targets, evidence = self.compat.lora_target_modules(_Qwen())
        self.assertEqual(targets, leaves)
        self.assertEqual(evidence["family"], "qwen3_5")
        leaves.pop()

        class _Short(_Qwen):
            def named_modules(self):
                yield "", self
                for name in leaves[:-1]:
                    yield f"model.layers.0.{name}", _Leaf()

        with self.assertRaises(RuntimeError):
            self.compat.lora_target_modules(_Short())

    def test_qwen35_recipe_requires_enable_thinking(self) -> None:
        base = self.compat.qwen35_base("Qwen/Qwen3.5-0.8B", "a" * 40)
        self.assertEqual(base["reasoning_control"], "enable_thinking")
        self.assertIsNone(self.compat.qwen35_base("Qwen/Qwen3-8B", "a" * 40))
        with self.assertRaises(RuntimeError):
            self.compat.reasoning_settings(base, {})

    def test_moe_mixer_is_refused(self) -> None:
        model = _hybrid()
        model._mixers.append(_mixer("NemotronHMoE", ["experts"]))
        with self.assertRaises(RuntimeError):
            self.compat.lora_target_modules(model, fused_mamba=False)

    def test_pins(self) -> None:
        self.assertEqual(
            self.compat.nemotron_base(NANO_4B, NANO_4B_REV)["revision"], NANO_4B_REV
        )
        self.assertEqual(
            self.compat.nemotron_base(NANO_9B, NANO_9B_REV)["revision"], NANO_9B_REV
        )
        with self.assertRaises(RuntimeError):
            self.compat.nemotron_base(NANO_4B, "f" * 40)
        self.assertEqual(
            self.compat.nemotron_base(NANO_4B)["chat_template_sha256"],
            "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
        )
        self.assertEqual(
            self.compat.nemotron_base(NANO_9B)["chat_template_sha256"],
            "b7a3a520a4bc1beae6a25d615f9215580df8eea926f488256c2bef74039433c0",
        )

    def test_notice_lines_per_licence_family(self) -> None:
        four = self.compat.nemotron_base(NANO_4B)
        nine = self.compat.nemotron_base(NANO_9B)
        self.assertEqual(
            four["notice"],
            "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.",
        )
        self.assertEqual(four["license_id"], "nvidia-nemotron-open-model-license")
        self.assertEqual(nine["license_id"], "nvidia-open-model-license")
        self.assertIn("NVIDIA Open Model License", nine["notice"])
        self.assertEqual(
            nine["acknowledgement"], "acknowledged: NVIDIA Open Model License"
        )
        self.assertIn("Nemotron Open Model License", four["acknowledgement"])


class ReasoningToggle(unittest.TestCase):
    def setUp(self) -> None:
        self.compat = _load("trainer_compat")

    def test_toggle_must_be_explicit(self) -> None:
        base = self.compat.nemotron_base(NANO_4B)
        for value in (None, "false", 0):
            with self.assertRaises(RuntimeError):
                self.compat.reasoning_settings(base, {"enable_thinking": value})
        self.assertIsNone(self.compat.reasoning_settings(None, {}))

    def test_enable_thinking_kwarg_is_passed(self) -> None:
        tokenizer = _Tokenizer("{{ enable_thinking }}")
        msgs = [{"role": "user", "content": "q"}]
        for flag in (True, False):
            text = self.compat.render_chat_text(
                tokenizer,
                msgs,
                reasoning={"control": "enable_thinking", "enable_thinking": flag},
            )
            self.assertTrue(text.endswith(f"think={flag}"))

    def test_template_without_toggle_fails_closed(self) -> None:
        reasoning = {"control": "enable_thinking", "enable_thinking": True}
        with self.assertRaises(RuntimeError):
            self.compat.render_chat_text(
                _Tokenizer("{{ messages }}"), [], reasoning=reasoning
            )
        with self.assertRaises(RuntimeError):
            self.compat.render_chat_text(_Tokenizer(None), [], reasoning=reasoning)

    def test_system_marker_for_nano_9b(self) -> None:
        tokenizer = _Tokenizer("/think /no_think")
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        text = self.compat.render_chat_text(
            tokenizer,
            msgs,
            reasoning={"control": "system_marker", "enable_thinking": False},
        )
        self.assertTrue(text.startswith("sys /no_think|q"))
        self.assertEqual(msgs[0]["content"], "sys")
        text = self.compat.render_chat_text(
            tokenizer,
            [{"role": "user", "content": "q"}],
            reasoning={"control": "system_marker", "enable_thinking": True},
        )
        self.assertTrue(text.startswith("/think|q"))
        with self.assertRaises(RuntimeError):
            self.compat.render_chat_text(
                tokenizer,
                [{"role": "user", "content": "q /no_think"}],
                reasoning={"control": "system_marker", "enable_thinking": True},
            )


class Guardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.compat = _load("trainer_compat")
        self.template = "pinned template"
        self.base = dict(self.compat.nemotron_base(NANO_4B))
        self.base["chat_template_sha256"] = hashlib.sha256(
            self.template.encode()
        ).hexdigest()
        self._tmp = tempfile.TemporaryDirectory()
        self.notice_path = Path(self._tmp.name) / "NOTICE"
        self.notice_path.write_text(self.base["notice"] + "\n", encoding="utf-8")
        self.guard = {
            "nvidia_guardrails": "kept",
            "license_id": self.base["license_id"],
            "license_acknowledgement": self.base["acknowledgement"],
            "notice": self.base["notice"],
            "notice_file": str(self.notice_path),
            "system_prompt": "preserve",
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_kept_guardrails_pass(self) -> None:
        out = self.compat.require_nemotron_guardrails(
            self.base, _Tokenizer(self.template), self.guard
        )
        self.assertEqual(out["notice"], self.base["notice"])

    def test_stripped_or_altered_is_refused(self) -> None:
        cases = [
            None,
            {**self.guard, "nvidia_guardrails": "removed"},
            {**self.guard, "notice": ""},
            {**self.guard, "notice": self.base["notice"].rstrip(".")},
            {**self.guard, "license_id": "apache-2.0"},
            {**self.guard, "system_prompt": "strip"},
            {**self.guard, "license_acknowledgement": "acknowledged: MIT"},
            {**self.guard, "notice_file": str(self.notice_path.parent / "missing")},
        ]
        for guard in cases:
            with self.assertRaises(RuntimeError):
                self.compat.require_nemotron_guardrails(
                    self.base, _Tokenizer(self.template), guard
                )
        with self.assertRaises(RuntimeError):
            self.compat.require_nemotron_guardrails(
                self.base, _Tokenizer("replaced template"), self.guard
            )

    def test_notice_written_to_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.compat.write_nvidia_notice(directory, self.base)
            text = Path(path).read_text(encoding="utf-8")
        self.assertIn(
            "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.",
            text,
        )
        self.assertIn(NANO_4B_REV, text)

    def test_wave_supports_lora_and_qlora_and_keeps_the_notice_gate(self) -> None:
        source = (ROOT / "training" / "nemotron_wave.py").read_text(encoding="utf-8")
        self.assertIn('args["method"] not in ("lora16", "qlora")', source)
        self.assertIn("NEMOTRON_QLORA", source)
        self.assertIn("keep_mamba_out_proj_trainable", source)
        self.assertIn("allow_nemotron_out_proj_lora", source)
        self.assertIn("require_nemotron_guardrails", source)
        self.assertIn("write_nvidia_notice", source)
        self.assertIn("trust_remote_code=False", source)
        qwen = (ROOT / "training" / "qwen35_wave.py").read_text(encoding="utf-8")
        self.assertIn("enable_thinking", qwen)
        self.assertIn("processing_class=tokenizer", qwen)
        self.assertIn('args["method"] not in ("lora16", "qlora")', qwen)
        wave = (ROOT / "training" / "train_wave.py").read_text(encoding="utf-8")
        self.assertIn("qwen35_base", wave)
        self.assertIn("FastLanguageModel.get_peft_model(", wave)


def _native_nemotron_h() -> bool:
    for name in ("torch", "trl", "peft", "transformers"):
        if importlib.util.find_spec(name) is None:
            return False
    import transformers

    return hasattr(transformers, "NemotronHForCausalLM")


class NemotronCpuDryRun(unittest.TestCase):
    def test_tiny_nemotron_h_trains_two_steps(self) -> None:
        if not _native_nemotron_h():
            if os.environ.get("SZL_NEMOTRON_DRY_RUN_REQUIRED") == "1":
                self.fail("pinned Nemotron-H stack is required and is not installed")
            self.skipTest("torch/TRL/PEFT with native NemotronH is not installed")
        result = _load("cpu_dry_run").run_nemotron_cpu_dry_run(steps=2)
        self.assertEqual(result["global_steps"], 2)
        self.assertTrue(math.isfinite(result["training_loss"]))
        self.assertEqual(sorted(result["lora_targets"]), result["lora_targets_updated"])
        self.assertIn("in_proj", result["lora_targets"])
        self.assertIn("out_proj", result["lora_targets"])
        self.assertTrue(result["notice_in_artifacts"])
        self.assertIn("NOTICE", result["saved_files"])
        self.assertIn("adapter_config.json", result["saved_files"])
        self.assertTrue(
            any(name.startswith("adapter_model") for name in result["saved_files"])
        )
        self.assertFalse(result["gpu_job_launched"])
        self.assertEqual(len(result["refusals"]), 9)
        self.assertIn("unacknowledged_license", result["refusals"])
        self.assertIn("notice_file_altered", result["refusals"])

    def test_fused_kernel_bound_before_patch_cannot_skip_out_proj(self) -> None:
        if not _native_nemotron_h():
            if os.environ.get("SZL_NEMOTRON_DRY_RUN_REQUIRED") == "1":
                self.fail("pinned Nemotron-H stack is required and is not installed")
            self.skipTest("torch/TRL/PEFT with native NemotronH is not installed")
        import torch
        import transformers.models.nemotron_h.modeling_nemotron_h as nemotron_h

        modules = [nemotron_h]
        try:
            import transformers.models.zamba2.modeling_zamba2 as zamba2
        except ImportError:
            zamba2 = None
        else:
            if hasattr(zamba2, "mamba2_split_conv1d_scan_combined"):
                modules.append(zamba2)
        originals = {
            module: module.mamba2_split_conv1d_scan_combined for module in modules
        }
        calls = {"fused": 0}

        def fake(*args: object, **_kwargs: object) -> torch.Tensor:
            calls["fused"] += 1
            projected = args[0]
            if not isinstance(projected, torch.Tensor):
                raise AssertionError("fused kernel was called without a projection")
            return torch.zeros(
                projected.shape[0],
                projected.shape[1],
                32,
                dtype=projected.dtype,
                device=projected.device,
            )

        hits = {"out_proj": 0}

        def count(_module: object, _inputs: object, _output: object) -> None:
            hits["out_proj"] += 1

        try:
            for module in modules:
                module.mamba2_split_conv1d_scan_combined = fake
            model = _load("cpu_dry_run")._tiny_nemotron_h(32)
            model.train()
            hooked = 0
            for name, layer in model.named_modules():
                if name.endswith(".out_proj"):
                    layer.register_forward_hook(count)
                    hooked += 1
            self.assertGreater(hooked, 0)
            self.compat = _load("trainer_compat")
            self.compat.keep_mamba_out_proj_trainable()
            for module in modules:
                current = module.mamba2_split_conv1d_scan_combined
                self.assertTrue(getattr(current, "_szl_reference_mamba", False))
            model(input_ids=torch.randint(0, 32, (1, 8)))
            self.assertEqual(calls["fused"], 0)
            self.assertGreater(hits["out_proj"], 0)
        finally:
            for module, original in originals.items():
                module.mamba2_split_conv1d_scan_combined = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
