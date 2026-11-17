"""Prevent the SFT trainer from reintroducing unsupported TRL tokenizer= keyword.

The canonical DPO trainer already uses processing_class. This checks source
structure without importing model weights, transformers, torch, or Unsloth.
A runtime GPU smoke remains required before treating a training run as tested.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def trainer_call(path: str, name: str) -> ast.Call:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    found = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]
    if len(found) != 1:
        raise AssertionError(f"expected one {name} call in {path}, got {len(found)}")
    return found[0]


class SFTTrlCompatibilityTests(unittest.TestCase):
    def test_canonical_sft_uses_processing_class(self):
        call = trainer_call("training/train_wave.py", "SFTTrainer")
        keywords = {kw.arg: kw.value for kw in call.keywords}
        self.assertNotIn(None, keywords)
        self.assertNotIn("tokenizer", keywords)
        self.assertIn("processing_class", keywords)
        self.assertIsInstance(keywords["processing_class"], ast.Name)
        self.assertEqual(keywords["processing_class"].id, "tokenizer")
        self.assertIn("train_dataset", keywords)
        self.assertIn("args", keywords)

    def test_dpo_and_sft_use_one_trl_contract(self):
        sft = trainer_call("training/train_wave.py", "SFTTrainer")
        dpo = trainer_call("training/dpo_wave.py", "DPOTrainer")
        for candidate in (sft, dpo):
            keys = {k.arg for k in candidate.keywords}
            self.assertIn("processing_class", keys)
            self.assertNotIn("tokenizer", keys)

    def test_launch_gates_still_precede_gpu(self):
        tree = ast.parse((ROOT / "training/train_wave.py").read_text(encoding="utf-8"))
        funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
        self.assertIn("main", funcs)
        self.assertIn("_train_after_gates", funcs)
        main = funcs["main"]
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        self.assertTrue(any(isinstance(n.func, ast.Attribute) and n.func.attr == "evaluate_path"
                            for n in calls))
        self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == "_refuse_cli_overrides"
                            for n in calls))
        self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == "_train_after_gates"
                            for n in calls))


if __name__ == "__main__":
    unittest.main()
