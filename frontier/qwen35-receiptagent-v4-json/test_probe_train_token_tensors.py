# SPDX-License-Identifier: Apache-2.0
"""SIMULATED adapter-shape tests; the optional CLI checks real pinned assets."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_base_tokenizer as tokenizer_probe
import probe_train_token_tensors as probe
import training_tokens as tokens


class FixtureTokenizer:
    pad_token = "<|pad|>"

    def convert_tokens_to_ids(self, value):
        return 248045 if value == "<|im_start|>" else None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": False,
                          "enable_thinking": False}
        return "".join(fragment + message["content"] for fragment, message
                       in zip(probe.FRAGMENTS, messages)) + "<|im_end|>\n"

    def __call__(self, rendered, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_offsets_mapping": True}
        return {"input_ids": list(map(ord, rendered)),
                "offset_mapping": [(i, i + 1) for i in range(len(rendered))]}

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False,
                          "clean_up_tokenization_spaces": False}
        return "".join(map(chr, ids))


class TrainTensorProbeTests(unittest.TestCase):
    def test_binding_is_pinned_but_never_authorizes_training(self):
        binding = probe._binding(FixtureTokenizer())
        self.assertEqual(binding.base_revision, tokenizer_probe.BASE_REVISION)
        self.assertEqual(binding.termination_ids, tokenizer_probe.ENDING_IDS)
        self.assertEqual(binding.admitted_limit, 2048)
        self.assertEqual(binding.total_token_limit, 400_000)
        self.assertEqual(binding.template_sha256,
                         tokenizer_probe.ASSET_SHA256["chat_template.jinja"])
        self.assertEqual(binding.render_fragments[2],
                         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        class WrongStart(FixtureTokenizer):
            def convert_tokens_to_ids(self, value):
                return 0
        with self.assertRaisesRegex(tokenizer_probe.ProbeError, "START_ID"):
            probe._binding(WrongStart())

    def test_reference_labels_mask_prompt_and_supervise_complete_json_and_ending(self):
        messages = (("system", "SIMULATED system"), ("user", "SIMULATED request"),
                    ("assistant", '{"decision":"DRAFT"}'))
        tokenizer = FixtureTokenizer()
        row = probe._encode(tokenizer, messages)
        start, end = row.message_spans[-1]
        self.assertEqual(row.rendered[start:end], messages[-1][1])
        self.assertEqual(row.rendered[end:], "<|im_end|>\n")
        self.assertTrue(all(label == tokens.IGNORE_INDEX for label in row.labels[:start]))
        self.assertEqual(row.labels[start:], row.input_ids[start:])
        self.assertEqual(row.attention_mask, (1,) * len(row.input_ids))
        self.assertEqual(probe._decode(tokenizer, row.input_ids), row.rendered)

    def test_wrong_message_shape_and_missing_assets_fail_closed(self):
        with self.assertRaisesRegex(tokenizer_probe.ProbeError, "MESSAGE_SHAPE"):
            probe._encode(FixtureTokenizer(), (("user", "x"),))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), patch.object(
                probe.tokenizer_probe, "_read_assets", side_effect=tokenizer_probe.ProbeError(
                    "TOKENIZER_ASSET_HASH_MISMATCH")):
            code = probe.main(["--tokenizer-dir", "unused", "--model-config", "unused",
                               "--train", "train.jsonl", "--manifest", "curriculum-manifest.json",
                               "--nemo-source-root", "unused"])
        self.assertEqual(code, 1)
        self.assertIn('"state": "BLOCKED"', out.getvalue())
        self.assertIn('"training_eligible": false', out.getvalue())
        self.assertNotIn("SIMULATED request", out.getvalue())


if __name__ == "__main__":
    unittest.main()
