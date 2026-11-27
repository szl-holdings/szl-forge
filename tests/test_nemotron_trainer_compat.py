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
    pass


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
            ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        )
        self.assertNotIn("out_proj", targets)
        self.assertNotIn("conv1d", targets)
        self.assertEqual(evidence["layout"], {"mamba": 1, "attention": 1, "mlp": 1})

    def test_missing_in_proj_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            self.compat.lora_target_modules(
                _hybrid(("conv1d", "out_proj")), fused_mamba=False
            )

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
        self.guard = {
            "nvidia_guardrails": "kept",
            "license_id": self.base["license_id"],
            "notice": self.base["notice"],
            "system_prompt": "preserve",
        }

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

    def test_wave_refuses_qlora_on_nemotron(self) -> None:
        source = (ROOT / "training" / "nemotron_wave.py").read_text(encoding="utf-8")
        self.assertIn('if args["method"] != "lora16":', source)
        self.assertIn("require_nemotron_guardrails", source)
        self.assertIn("write_nvidia_notice", source)
        self.assertIn("trust_remote_code=False", source)


def _native_nemotron_h() -> bool:
    for name in ("torch", "trl", "peft", "transformers"):
        if importlib.util.find_spec(name) is None:
            return False
    import transformers

    return hasattr(transformers, "NemotronHForCausalLM")


class NemotronCpuDryRun(unittest.TestCase):
    def test_tiny_nemotron_h_trains_two_steps(self) -> None:
        if not _native_nemotron_h():
            self.skipTest("torch/TRL/PEFT with native NemotronH is not installed")
        result = _load("cpu_dry_run").run_nemotron_cpu_dry_run(steps=2)
        self.assertEqual(result["global_steps"], 2)
        self.assertTrue(math.isfinite(result["training_loss"]))
        self.assertEqual(sorted(result["lora_targets"]), result["lora_targets_updated"])
        self.assertIn("in_proj", result["lora_targets"])
        self.assertTrue(result["notice_in_artifacts"])
        self.assertIn("NOTICE", result["saved_files"])
        self.assertFalse(result["gpu_job_launched"])
        self.assertEqual(len(result["refusals"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
