# SPDX-License-Identifier: Apache-2.0
"""SIMULATED config fixtures; they do not load weights or qualify a runtime."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_base_config as probe
import probe_base_tokenizer as tokenizer_probe


def fixture_config() -> dict:
    return {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text", "vocab_size": probe.VOCAB_SIZE,
                            "max_position_embeddings": probe.MODEL_CONTEXT_LIMIT,
                            "eos_token_id": probe.CONFIG_EOS_ID}}


class Qwen2Tokenizer:
    is_fast = True
    vocab_size = probe.TOKENIZER_VOCAB_SIZE
    eos_token_id = probe.CHAT_EOS_ID
    pad_token_id = tokenizer_probe.PAD_ID

    def __len__(self):
        return probe.TOKENIZER_LENGTH

    def convert_ids_to_tokens(self, token):
        return {probe.CONFIG_EOS_ID: "<|endoftext|>",
                probe.CHAT_EOS_ID: "<|im_end|>",
                tokenizer_probe.ENDING_IDS[-1]: "Ċ"}.get(token)


class ProbeBaseConfigTests(unittest.TestCase):
    def read(self, raw: bytes) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(raw)
            with patch.object(probe, "CONFIG_SHA256", tokenizer_probe._sha(raw)):
                return probe._read_config(path)

    def test_exact_config_and_distinct_chat_termination(self):
        self.assertEqual(self.read(json.dumps(fixture_config()).encode()), fixture_config())
        probe._qualify_tokenizer(Qwen2Tokenizer())
        self.assertNotEqual(probe.CONFIG_EOS_ID, probe.CHAT_EOS_ID)

    def test_hash_and_shape_fail_closed(self):
        raw = json.dumps(fixture_config()).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(raw)
            with self.assertRaisesRegex(tokenizer_probe.ProbeError, "HASH_MISMATCH"):
                probe._read_config(path)
        for field, value in (("vocab_size", True), ("vocab_size", probe.VOCAB_SIZE - 1),
                             ("max_position_embeddings", 1024),
                             ("eos_token_id", probe.CHAT_EOS_ID)):
            with self.subTest(field=field, value=value):
                changed = fixture_config()
                changed["text_config"][field] = value
                with self.assertRaisesRegex(tokenizer_probe.ProbeError, "TEXT_MISMATCH"):
                    self.read(json.dumps(changed).encode())

    def test_duplicate_members_and_oversize_rejected(self):
        raw = json.dumps(fixture_config()).encode()
        duplicate = raw[:-1] + b',"model_type":"qwen3_5"}'
        with self.assertRaisesRegex(tokenizer_probe.ProbeError, "DUPLICATE_KEY"):
            self.read(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(raw)
            with patch.object(probe, "MAX_CONFIG_BYTES", 8):
                with self.assertRaisesRegex(tokenizer_probe.ProbeError, "TOO_LARGE"):
                    probe._read_config(path)

    def test_tokenizer_drift_rejected(self):
        for attribute, value in (("is_fast", False), ("vocab_size", 2),
                                 ("eos_token_id", probe.CONFIG_EOS_ID),
                                 ("pad_token_id", 0)):
            with self.subTest(attribute=attribute):
                class Changed(Qwen2Tokenizer):
                    pass
                Changed.__name__ = "Qwen2Tokenizer"
                setattr(Changed, attribute, value)
                with self.assertRaises(tokenizer_probe.ProbeError):
                    probe._qualify_tokenizer(Changed())


if __name__ == "__main__":
    unittest.main()
