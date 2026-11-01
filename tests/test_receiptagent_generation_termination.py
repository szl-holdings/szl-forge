"""Exercise actual generation functions with synthetic tensors, no GPU or corpus.

The evaluator's two standalone functions are compiled from its source AST so
this focused stdlib suite needs neither its GPU runtime nor jsonschema imports.
"""
import ast
import contextlib
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "frontier/qwen35-receiptagent-v3"
tree = ast.parse((HERE / "evaluate_candidate.py").read_text(encoding="utf-8"))
selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
selected += [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"generation_termination", "output_text"}]
namespace = {"time": time}
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
             str(HERE / "evaluate_candidate.py"), "exec"), namespace)
diagnose = namespace["generation_termination"]
output_text = namespace["output_text"]


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        row, column = key
        if isinstance(row, int):
            return types.SimpleNamespace(item=lambda: self.rows[row][column])
        return Tensor([value[column] for value in self.rows[row]])


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return "synthetic prompt"

    def __call__(self, **kwargs):
        return Inputs(input_ids=Tensor([[100, 101]]))

    def batch_decode(self, tokens, **kwargs):
        self.decoded_tokens = tokens.rows
        return ["  synthetic answer  "]


class GenerationTerminationTests(unittest.TestCase):
    def test_eos_before_limit(self):
        value = diagnose(3, 4, 9, 9)
        self.assertEqual(value["stopReason"], "EOS")
        self.assertFalse(value["truncationPossible"])
        self.assertFalse(value["tokenBudgetReached"])

    def test_limit_without_eos(self):
        value = diagnose(4, 4, 7, [8, 9])
        self.assertEqual(value["stopReason"], "TOKEN_LIMIT")
        self.assertTrue(value["truncationPossible"])
        self.assertFalse(value["endedWithEos"])

    def test_eos_at_limit_remains_ambiguous(self):
        value = diagnose(4, 4, 9, [8, 9])
        self.assertEqual(value["stopReason"], "EOS_AT_TOKEN_LIMIT")
        self.assertTrue(value["endedWithEos"])
        self.assertTrue(value["truncationPossible"])

    def test_short_output_without_eos_is_unknown(self):
        value = diagnose(2, 4, 7, 9)
        self.assertEqual(value["stopReason"], "UNKNOWN")
        self.assertTrue(value["truncationPossible"])

    def test_missing_eos_configuration_is_not_completion_evidence(self):
        value = diagnose(2, 4, 9, None)
        self.assertFalse(value["eosConfigured"])
        self.assertIsNone(value["endedWithEos"])
        self.assertEqual(value["stopReason"], "UNKNOWN")

    def test_empty_generation_is_unknown(self):
        value = diagnose(0, 4, None, [8, 9])
        self.assertEqual(value["stopReason"], "UNKNOWN")
        self.assertTrue(value["truncationPossible"])

    def test_output_text_keeps_tuple_and_exact_generation_parameters(self):
        processor = Processor()
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[8, 9]))
        model.generate = mock.Mock(return_value=Tensor([[100, 101, 2, 3, 4, 9]]))
        termination = {}
        torch_stub = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            result = output_text(model, processor, [], max_new_tokens=4, termination=termination)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[:2], ("synthetic answer", 4))
        self.assertEqual(processor.decoded_tokens, [[2, 3, 4, 9]])
        self.assertEqual(termination["stopReason"], "EOS_AT_TOKEN_LIMIT")
        self.assertEqual(termination["maxNewTokens"], 4)
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(set(kwargs), {"input_ids", "max_new_tokens", "do_sample", "use_cache"})
        self.assertEqual((kwargs["max_new_tokens"], kwargs["do_sample"], kwargs["use_cache"]), (4, False, True))

    def test_existing_caller_without_diagnostics_still_works(self):
        model = types.SimpleNamespace(generate=mock.Mock(return_value=Tensor([[100, 101, 2]])))
        with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            result = output_text(model, Processor(), [], max_new_tokens=4)
        self.assertEqual(result[:2], ("synthetic answer", 1))

    def test_historical_budget_is_unchanged(self):
        candidate = json.loads((HERE / "candidate.json").read_text(encoding="utf-8"))
        self.assertEqual(candidate["evaluation_protocol"]["structured_max_new_tokens"], 512)
        self.assertEqual(candidate["evaluation_protocol"]["refusal_max_new_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
