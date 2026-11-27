"""CPU and source checks for the Forge TRL 0.23.1 trainer boundary.

Signature and dtype policy run without torch.  The integration step trains a
tiny local model when TRL 0.23.1 and Transformers are installed, and skips
when they are not.  No hosted job is launched.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"missing {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls(path: str, name: str) -> list[ast.Call]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


class _Cuda:
    def __init__(self, available: bool, bf16: bool) -> None:
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _Torch:
    def __init__(self, available: bool, bf16: bool) -> None:
        self.cuda = _Cuda(available, bf16)
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"


class TrainerCompatPolicyTests(unittest.TestCase):
    def test_cpu_refuses_four_bit_and_mixed_precision(self) -> None:
        compat = _load("trainer_compat")
        torch = _Torch(False, False)
        with self.assertRaisesRegex(RuntimeError, "4-bit"):
            compat.unsloth_load_kwargs(torch, load_in_4bit=True)
        load = compat.unsloth_load_kwargs(torch, load_in_4bit=False)
        self.assertEqual(load, {"dtype": "float32", "load_in_4bit": False})
        self.assertEqual(compat.precision_kwargs(torch), {"bf16": False, "fp16": False})

    def test_cuda_dtype_matches_mixed_precision(self) -> None:
        compat = _load("trainer_compat")
        ampere = compat.unsloth_load_kwargs(_Torch(True, True), load_in_4bit=True)
        self.assertEqual(ampere, {"dtype": "bfloat16", "load_in_4bit": True})
        turing = compat.precision_kwargs(_Torch(True, False))
        self.assertEqual(turing, {"bf16": False, "fp16": True})

    def test_chat_template_and_padding_fail_closed(self) -> None:
        compat = _load("trainer_compat")

        class Tokenizer:
            chat_template = "template"
            pad_token_id = None
            eos_token = "<eos>"
            pad_token = None

            def apply_chat_template(self, messages, tokenize=False):
                del messages, tokenize
                return "user question assistant one"

        self.assertEqual(compat.require_chat_template(Tokenizer()), "template")
        bare = Tokenizer()
        bare.chat_template = "  "
        with self.assertRaisesRegex(RuntimeError, "chat template"):
            compat.require_chat_template(bare)
        missing_pad = Tokenizer()
        compat.require_padding_token(missing_pad)
        self.assertEqual(missing_pad.pad_token, "<eos>")
        empty = Tokenizer()
        empty.eos_token = None
        with self.assertRaisesRegex(RuntimeError, "padding token"):
            compat.require_padding_token(empty)
        rendered = compat.render_chat_example(
            Tokenizer(),
            {"messages": [{"role": "user", "content": "user question"}]},
        )
        self.assertIn("assistant", rendered["text"])
        blank = compat.render_chat_example(Tokenizer(), {})
        self.assertEqual(blank["text"], "")

        class RecordingTokenizer:
            def __call__(
                self, text: str, return_tensors: str = "pt"
            ) -> dict[str, object]:
                del return_tensors
                return {"input_ids": text, "token_type_ids": [0]}

        batch = compat.generation_batch(RecordingTokenizer(), "user question")
        self.assertEqual(batch["input_ids"], "user question")
        self.assertNotIn("token_type_ids", batch)
        with self.assertRaisesRegex(RuntimeError, "generation prompt"):
            compat.generation_batch(RecordingTokenizer(), "  ")

    def test_wave_sources_use_the_023_boundary(self) -> None:
        sft = _calls("training/train_wave.py", "SFTConfig")
        dpo = _calls("training/dpo_wave.py", "DPOConfig")
        self.assertEqual(len(sft), 1)
        self.assertEqual(len(dpo), 1)
        for call in (sft[0], dpo[0]):
            names = {keyword.arg for keyword in call.keywords}
            self.assertIn("max_length", names)
            self.assertNotIn("max_seq_length", names)
            self.assertNotIn("evaluation_strategy", names)
            self.assertNotIn("bf16", names)
        for path in (
            "training/train_wave.py",
            "training/dpo_wave.py",
            "training/eval_wave.py",
        ):
            source = (ROOT / path).read_text(encoding="utf-8")
            self.assertNotIn("torch_dtype", source)
            self.assertNotIn("evaluation_strategy", source)
            self.assertNotIn("tokenizer=", source)
        eval_loads = [
            node
            for node in ast.walk(
                ast.parse((ROOT / "training/eval_wave.py").read_text(encoding="utf-8"))
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and any(keyword.arg == "device_map" for keyword in node.keywords)
        ]
        self.assertEqual(len(eval_loads), 1)
        names = {keyword.arg for keyword in eval_loads[0].keywords}
        self.assertIn("dtype", names)
        self.assertNotIn("torch_dtype", names)

    def test_dry_run_source_does_not_launch(self) -> None:
        source = (
            (ROOT / "training" / "cpu_dry_run.py").read_text(encoding="utf-8").lower()
        )
        for forbidden in (
            "fastlanguagemodel",
            "keys/",
            "repository_dispatch",
            "torch_dtype",
            "hf_token",
        ):
            self.assertNotIn(forbidden, source)

    def test_cpu_dry_run_trains_a_few_steps(self) -> None:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("trl") is None
        ):
            self.skipTest("pinned trainer stack is not installed")
        result = _load("cpu_dry_run").run_cpu_dry_run()
        self.assertGreaterEqual(result["global_steps"], 1)
        self.assertGreaterEqual(result["dpo_steps"], 1)
        self.assertTrue(result["chat_template_present"])
        self.assertIn("chat template", result["missing_template_refused"])
        self.assertEqual(result["eval_strategy"], "no")
        self.assertEqual(result["max_length"], 32)
        self.assertEqual(result["dpo_max_length"], 32)
        self.assertFalse(result["bf16"])
        self.assertFalse(result["fp16"])
        self.assertEqual(result["cpu_dtype"], "torch.float32")
        self.assertIn("4-bit", result["quantization_refused"])
        self.assertEqual(result["dataset_field"], "messages")
        self.assertGreater(result["generated_tokens"], 1)
        self.assertFalse(result["gpu_job_launched"])
        self.assertTrue(math.isfinite(result["training_loss"]))
        self.assertTrue(math.isfinite(result["dpo_loss"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
