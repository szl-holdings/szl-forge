"""Synthetic coverage-control regressions, not model qualification evidence.

Only standard-library fakes execute: no torch, PEFT, Transformers, safetensors,
weights, model generation, network calls, or historical receipts are loaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "chaski_margin_probe.py"
SPEC = importlib.util.spec_from_file_location("_chaski_margin_probe_coverage_under_test", MODULE_PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

MM = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_{ab}.weight"
TXT = "base_model.model.model.layers.0.mlp.down_proj.lora_{ab}.weight"
ML_MODULES = ("torch", "transformers", "peft", "safetensors", "accelerate")


def fake_environment(checkpoint_keys: list[str], live_keys: list[str]):
    """Build modules without importing any real ML dependency or reading files."""
    modules = {name: types.ModuleType(name) for name in ML_MODULES}
    for module in modules.values():
        module.__version__ = "synthetic-test-only"
    modules["torch"].bfloat16 = object()
    modules["torch"].float16 = object()
    modules["torch"].float32 = object()
    modules["torch"].cuda = types.SimpleNamespace(
        is_available=Mock(return_value=False), empty_cache=Mock()
    )
    tokenizer = types.SimpleNamespace(
        chat_template="synthetic-template", init_kwargs={}, pad_token_id=0,
        eos_token="<eos>", apply_chat_template=Mock(return_value="synthetic prompt")
    )
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(architectures=["SyntheticModel"], model_type="synthetic"),
        generation_config=types.SimpleNamespace(eos_token_id=1), active_adapter="default",
        named_parameters=Mock(return_value=[
            (key.replace(".weight", ".default.weight"), object()) for key in live_keys
        ]), eval=Mock()
    )
    model.base_model = types.SimpleNamespace(model=model)
    base_loader = Mock(return_value=model)
    adapter_loader = Mock(return_value=model)
    modules["transformers"].AutoTokenizer = types.SimpleNamespace(
        from_pretrained=Mock(return_value=tokenizer)
    )
    modules["transformers"].AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=base_loader)
    modules["transformers"].AutoModelForImageTextToText = types.SimpleNamespace(from_pretrained=base_loader)
    modules["peft"].PeftModel = types.SimpleNamespace(from_pretrained=adapter_loader)
    header = MagicMock()
    header.__enter__.return_value.keys.return_value = list(checkpoint_keys)
    modules["safetensors"].safe_open = Mock(return_value=header)
    drafts = {"n": 1, "rows": [{"id": "synthetic-draft"}]}
    refusals = {"n": 1, "rows": [{"id": "synthetic-refusal"}]}
    runner = types.SimpleNamespace(
        prompt_messages=Mock(return_value=[{"role": "user", "content": "synthetic"}]),
        score_draft=Mock(return_value=(True, None)), score_refusal=Mock(return_value=(True, None)),
        sha256_adapter=Mock(return_value="b" * 64), DRAFT_KIND="draft", REFUSAL_KIND="refusal",
        CANONICAL_BASE="synthetic-local-base",
        load_named_n_gate=Mock(side_effect=lambda _path, kind: drafts if kind == "draft" else refusals)
    )
    generation = Mock(side_effect=lambda *_args, **_kwargs: {
        "output": "synthetic output", "output_sha256": "c" * 64, "new_tokens": 1,
        "seconds": 0.0, "prompt_tokens": 1, "ended_on_eos": True, "margin_min": None,
        "margin_argmin": None, "margin_mean": None, "structural_decisions": [], "steps": []
    })
    return types.SimpleNamespace(
        modules=modules, model=model, tokenizer=tokenizer, runner=runner, generation=generation,
        base_loader=base_loader, adapter_loader=adapter_loader, drafts=drafts, refusals=refusals
    )


class ChaskiDiagnosticCoverageTests(unittest.TestCase):
    def fake_context(self, fixture):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, fixture.modules))
        stack.enter_context(patch.object(probe, "generate_with_margins", fixture.generation))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def run_candidate(self, fixture, *, adapter=Path("synthetic-adapter")):
        return probe.probe_candidate(
            fixture.runner, base="synthetic-local-base", adapter=adapter,
            drafts=fixture.drafts, refusals=fixture.refusals, device="cpu", dtype_name="float32",
            draft_max_new_tokens=1, refusal_max_new_tokens=1, allow_missing_template=False,
            threshold=1.0, model_class_name="AutoModelForImageTextToText"
        )

    def assert_blocked_before_execution(self, checkpoint_keys, live_keys):
        fixture = fake_environment(checkpoint_keys, live_keys)
        with self.fake_context(fixture):
            with self.assertRaisesRegex(SystemExit, "ADAPTER_NOT_APPLIED"):
                self.run_candidate(fixture)
        fixture.model.eval.assert_not_called()
        fixture.generation.assert_not_called()
        fixture.runner.prompt_messages.assert_not_called()
        fixture.runner.score_draft.assert_not_called()
        fixture.runner.score_refusal.assert_not_called()

    def test_empty_checkpoint_blocks_before_eval_generation_and_scoring(self):
        self.assert_blocked_before_execution([], [])

    def test_coverage_only_rejects_empty_without_generation_or_receipt(self):
        fixture = fake_environment([], [])
        with self.fake_context(fixture), ExitStack() as stack:
            stack.enter_context(patch.object(probe, "load_runner", return_value=(fixture.runner, "a" * 64)))
            stack.enter_context(patch.object(probe, "sha256_file", return_value="a" * 64))
            writer = stack.enter_context(patch.object(Path, "write_text"))
            stack.enter_context(patch.object(sys, "argv", [
                "chaski_margin_probe.py", "--device", "cpu", "--adapter", "synthetic-adapter",
                "--coverage-only"
            ]))
            self.assertEqual(probe.main(), 4)
        fixture.model.eval.assert_not_called()
        fixture.generation.assert_not_called()
        fixture.runner.score_draft.assert_not_called()
        fixture.runner.score_refusal.assert_not_called()
        writer.assert_not_called()

    def test_partial_checkpoint_coverage_blocks_before_execution(self):
        keys = [MM.format(ab=ab) for ab in ("A", "B")]
        self.assert_blocked_before_execution(keys, keys[:1])

    def test_text_vs_multimodal_layout_blocks_before_execution(self):
        self.assert_blocked_before_execution(
            [MM.format(ab=ab) for ab in ("A", "B")],
            [TXT.format(ab=ab) for ab in ("A", "B")]
        )

    def test_complete_nonempty_coverage_accepts_two_synthetic_cases_only(self):
        keys = [MM.format(ab=ab) for ab in ("A", "B")]
        fixture = fake_environment(keys, keys)
        with self.fake_context(fixture):
            result = self.run_candidate(fixture)
        coverage = result["environment"]["adapter_keys"]
        self.assertEqual((coverage["checkpoint_tensors"], coverage["applied"], coverage["unapplied"]), (2, 2, 0))
        self.assertEqual(coverage["checkpoint_layout"], "language_model")
        self.assertEqual((result["json_draft_valid"], result["adversarial_refused"]), (1, 1))
        self.assertEqual(len(result["cases"]), 2)
        fixture.model.eval.assert_called_once_with()
        self.assertEqual(fixture.generation.call_count, 2)
        fixture.runner.score_draft.assert_called_once_with("synthetic output")
        fixture.runner.score_refusal.assert_called_once_with("synthetic output")

    def test_base_only_path_does_not_read_an_adapter(self):
        fixture = fake_environment([], [])
        with self.fake_context(fixture):
            result = self.run_candidate(fixture, adapter=None)
        self.assertIsNone(result["adapter"])
        self.assertIsNone(result["adapter_sha256"])
        self.assertIsNone(result["environment"]["adapter_keys"])
        fixture.adapter_loader.assert_not_called()
        fixture.modules["safetensors"].safe_open.assert_not_called()
        self.assertEqual(fixture.generation.call_count, 2)

    def test_fake_ml_modules_restore_the_prior_module_cache(self):
        missing = object()
        before = {name: sys.modules.get(name, missing) for name in ML_MODULES}
        fixture = fake_environment([], [])
        with self.fake_context(fixture):
            self.run_candidate(fixture, adapter=None)
            for name in ML_MODULES:
                self.assertIs(sys.modules[name], fixture.modules[name])
        for name in ML_MODULES:
            self.assertIs(sys.modules.get(name, missing), before[name])

    def test_synthetic_diagnostic_report_never_claims_qualification_or_publication(self):
        keys = [MM.format(ab="A")]
        fixture = fake_environment(keys, keys)
        with self.fake_context(fixture), ExitStack() as stack:
            stack.enter_context(patch.object(probe, "load_runner", return_value=(fixture.runner, "a" * 64)))
            stack.enter_context(patch.object(probe, "sha256_file", return_value="a" * 64))
            stack.enter_context(patch.object(Path, "mkdir"))
            writer = stack.enter_context(patch.object(Path, "write_text"))
            stack.enter_context(patch.object(sys, "argv", [
                "chaski_margin_probe.py", "--device", "cpu", "--adapter", "synthetic-adapter",
                "--out", "synthetic-diagnostic.json"
            ]))
            self.assertEqual(probe.main(), 0)
        writer.assert_called_once()
        report = json.loads(writer.call_args.args[0])
        self.assertEqual(report["kind"], "szl-chaski-margin-probe")
        self.assertEqual(report["schema"], "szl.chaski-margin-probe/v1")
        self.assertEqual(report["label"], "DIAGNOSTIC")
        for field in ("is_canonical_receipt", "publication_eligible", "autonomy_eligible", "hub_put"):
            self.assertIs(report[field], False)
        self.assertIn("Not the canonical gate; counts here qualify nothing", report["claim_boundary"])
        digest = report.pop("report_sha256")
        canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(digest, hashlib.sha256(canonical.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    unittest.main()
