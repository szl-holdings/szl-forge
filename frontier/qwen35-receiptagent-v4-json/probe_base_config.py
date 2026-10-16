# SPDX-License-Identifier: Apache-2.0
"""Offline, pinned base-config/tokenizer boundary check; never a training permit.

The text-model config uses <|endoftext|> while the chat tokenizer and committed
train template end assistant turns with <|im_end|>. This probe makes that
distinction explicit before a future runner chooses a generation stop policy.
It loads neither weights nor any train, dev, or held-out row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_base_tokenizer as tokenizer_probe

SCHEMA = "szl.receiptagent.v4-base-config-precheck/v1"
CONFIG_SHA256 = "f8745ebb3cf9d980cb746a4b09f700954975357e468e8321c8fa33cdd57acb55"
MAX_CONFIG_BYTES = 1024 * 1024
CONFIG_EOS_ID = 248044
CHAT_EOS_ID = tokenizer_probe.EOS_ID
VOCAB_SIZE = 248320
TOKENIZER_VOCAB_SIZE = 248044
TOKENIZER_LENGTH = 248077
MODEL_CONTEXT_LIMIT = 262144


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        tokenizer_probe._require(key not in result, "MODEL_CONFIG_DUPLICATE_KEY")
        result[key] = value
    return result


def _read_config(path: Path) -> dict:
    raw = tokenizer_probe._bounded_read(path, MAX_CONFIG_BYTES,
                                        "MODEL_CONFIG_TOO_LARGE", local_asset=True)
    tokenizer_probe._require(tokenizer_probe._sha(raw) == CONFIG_SHA256,
                             "MODEL_CONFIG_HASH_MISMATCH")
    try:
        config = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeError, ValueError, TypeError):
        raise tokenizer_probe.ProbeError("MODEL_CONFIG_INVALID_JSON") from None
    tokenizer_probe._require(type(config) is dict and config.get("model_type") == "qwen3_5"
                             and config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
                             "MODEL_CONFIG_ARCHITECTURE_MISMATCH")
    text = config.get("text_config")
    tokenizer_probe._require(type(text) is dict and text.get("model_type") == "qwen3_5_text"
                             and type(text.get("vocab_size")) is int
                             and text["vocab_size"] == VOCAB_SIZE
                             and type(text.get("max_position_embeddings")) is int
                             and text["max_position_embeddings"] == MODEL_CONTEXT_LIMIT
                             and type(text.get("eos_token_id")) is int
                             and text["eos_token_id"] == CONFIG_EOS_ID,
                             "MODEL_CONFIG_TEXT_MISMATCH")
    return config


def _qualify_tokenizer(tokenizer) -> None:
    tokenizer_probe._require(tokenizer.is_fast and type(tokenizer).__name__ == "Qwen2Tokenizer"
                             and tokenizer.vocab_size == TOKENIZER_VOCAB_SIZE
                             and len(tokenizer) == TOKENIZER_LENGTH
                             and len(tokenizer) <= VOCAB_SIZE,
                             "TOKENIZER_CONFIG_VOCAB_MISMATCH")
    tokenizer_probe._require(tokenizer.eos_token_id == CHAT_EOS_ID
                             and tokenizer.pad_token_id == tokenizer_probe.PAD_ID
                             and tokenizer.convert_ids_to_tokens(CONFIG_EOS_ID) == "<|endoftext|>"
                             and tokenizer.convert_ids_to_tokens(CHAT_EOS_ID) == "<|im_end|>"
                             and tokenizer.convert_ids_to_tokens(tokenizer_probe.ENDING_IDS[-1]) == "Ċ",
                             "TOKENIZER_CONFIG_TERMINATION_MISMATCH")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        assets = tokenizer_probe._read_assets(args.tokenizer_dir)
        _read_config(args.model_config)
        for name, expected in tokenizer_probe.DEPENDENCIES.items():
            tokenizer_probe._require(version(name) == expected, "DEPENDENCY_VERSION_MISMATCH")
        from transformers import AutoTokenizer
        with tempfile.TemporaryDirectory(prefix="szl-v4-config-tokenizer-") as private_dir:
            for name, raw in assets.items():
                (Path(private_dir) / name).write_bytes(raw)
            tokenizer = AutoTokenizer.from_pretrained(private_dir, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
            _qualify_tokenizer(tokenizer)
    except Exception as exc:
        code = str(exc) if type(exc) is tokenizer_probe.ProbeError else "RUNTIME_OR_INPUT_UNAVAILABLE"
        print(json.dumps({"state": "BLOCKED", "code": code, "training_eligible": False,
                          "publication_eligible": False, "execution_authority": False}, sort_keys=True))
        return 1
    record = {
        "schema": SCHEMA, "evidence_class": "MEASURED", "state": "CONFIG_PRECHECK_ONLY",
        "source_binding_evidence_class": "DECLARED", "signature": "UNAVAILABLE",
        "independent_replay": "UNAVAILABLE", "base_id": tokenizer_probe.BASE_ID,
        "base_revision": tokenizer_probe.BASE_REVISION, "model_config_sha256": CONFIG_SHA256,
        "tokenizer_asset_sha256": tokenizer_probe.ASSET_SHA256,
        "dependency_versions": tokenizer_probe.DEPENDENCIES,
        "text_model_context_limit": MODEL_CONTEXT_LIMIT, "text_model_vocab_size": VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE, "tokenizer_length": TOKENIZER_LENGTH,
        "model_config_eos_id": CONFIG_EOS_ID, "chat_tokenizer_eos_id": CHAT_EOS_ID,
        "chat_termination_ids": tokenizer_probe.ENDING_IDS,
        "config_eos_matches_chat_eos": False,
        "runtime_stop_policy": "UNQUALIFIED_REQUIRES_EXPLICIT_BINDING",
        "hardware": "CPU_ONLY_UNQUALIFIED", "loss": "NOT_RUN", "energy_joules": "UNAVAILABLE",
        "training_eligible": False, "publication_eligible": False, "execution_authority": False,
    }
    record["record_sha256"] = tokenizer_probe._sha(tokenizer_probe._canonical(record))
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
