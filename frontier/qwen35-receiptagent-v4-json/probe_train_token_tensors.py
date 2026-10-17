# SPDX-License-Identifier: Apache-2.0
"""Fresh train-only tensor precheck with the pinned Qwen tokenizer.

This is a reference collator, not the future trainer's collator or a training
permit. No weights, optimizer, GPU, dev cases, or held-out cases are opened.
Installed Python dependencies and the supplied local source/assets are trusted
inputs; hashes do not authenticate their origin or create a sandbox.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import probe_base_config as config_probe
import probe_base_tokenizer as tokenizer_probe
import training_tokens as tokens

SCHEMA = "szl.receiptagent.v4-train-token-tensor-precheck/v1"
FRAGMENTS = ("<|im_start|>system\n",
             "<|im_end|>\n<|im_start|>user\n",
             "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
PRECHECK_SEQUENCE_LIMIT = 2048
PRECHECK_TOTAL_TOKEN_LIMIT = 400_000
MASKING_POLICY = "unshifted-assistant-json-and-termination-only/v1"
THINKING_POLICY = {"add_generation_prompt": False, "enable_thinking": False}


def _binding(tokenizer) -> tokens.TokenBinding:
    """Bind declared local policy to verified asset commitments, not authority."""
    tokenizer_probe._require(tokenizer.convert_tokens_to_ids("<|im_start|>") == 248045,
                             "TOKENIZER_START_ID_MISMATCH")
    tokenizer_probe._require(type(tokenizer.pad_token) is str and tokenizer.pad_token,
                             "TOKENIZER_PAD_TEXT_MISMATCH")
    special = ((248045, "<|im_start|>"),
               (tokenizer_probe.EOS_ID, "<|im_end|>"),
               (tokenizer_probe.ENDING_IDS[-1], "\n"),
               (tokenizer_probe.PAD_ID, tokenizer.pad_token))
    return tokens.TokenBinding(
        base_id=tokenizer_probe.BASE_ID,
        base_revision=tokenizer_probe.BASE_REVISION,
        tokenizer_sha256=tokenizer_probe._sha(tokenizer_probe._canonical(tokenizer_probe.ASSET_SHA256)),
        template_sha256=tokenizer_probe.ASSET_SHA256["chat_template.jinja"],
        dependency_sha256=tokenizer_probe._sha(tokenizer_probe._canonical(tokenizer_probe.DEPENDENCIES)),
        masking_sha256=tokenizer_probe._sha(MASKING_POLICY.encode("ascii")),
        eos_sha256=tokenizer_probe._sha(tokenizer_probe._canonical({
            "model_config_eos_id": config_probe.CONFIG_EOS_ID,
            "chat_tokenizer_eos_id": tokenizer_probe.EOS_ID,
            "termination_ids": tokenizer_probe.ENDING_IDS,
            "pad_token_id": tokenizer_probe.PAD_ID,
        })),
        thinking_sha256=tokenizer_probe._sha(tokenizer_probe._canonical(THINKING_POLICY)),
        tokenizer_limit=tokenizer_probe.SEQUENCE_LIMIT,
        model_limit=tokens.MAX_SEQUENCE,  # safety cap, not the model's 262144-token context
        admitted_limit=PRECHECK_SEQUENCE_LIMIT,  # local precheck cap, never signed admission
        total_token_limit=PRECHECK_TOTAL_TOKEN_LIMIT,
        vocab_size=config_probe.VOCAB_SIZE,
        pad_token_id=tokenizer_probe.PAD_ID,
        render_fragments=FRAGMENTS,
        special_tokens=special,
        termination_ids=tokenizer_probe.ENDING_IDS,
    )


def _encode(tokenizer, messages: tuple[tuple[str, str], ...]) -> tokens.EncodedRow:
    tokenizer_probe._require(type(messages) is tuple and len(messages) == 3,
                             "TRAIN_MESSAGE_SHAPE_MISMATCH")
    rendered = tokenizer.apply_chat_template(
        [{"role": role, "content": content} for role, content in messages],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = tuple(encoded["input_ids"])
    offsets = tuple(tuple(pair) for pair in encoded["offset_mapping"])
    spans = []
    cursor = 0
    for fragment, (_, content) in zip(FRAGMENTS, messages):
        cursor += len(fragment)
        spans.append((cursor, cursor + len(content)))
        cursor += len(content)
    assistant_start, assistant_end = spans[-1]
    labels = tuple(token if (start >= assistant_start and end <= assistant_end)
                   or start >= assistant_end else tokens.IGNORE_INDEX
                   for token, (start, end) in zip(ids, offsets))
    return tokens.EncodedRow(rendered, ids, (1,) * len(ids), offsets, labels,
                             tuple(spans), (assistant_end, len(rendered)))


def _decode(tokenizer, ids: tuple[int, ...]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def _run(args: argparse.Namespace) -> dict:
    harness_raw = tokenizer_probe._bounded_read(Path(__file__), tokenizer_probe.MAX_HARNESS_BYTES,
                                                "HARNESS_SOURCE_TOO_LARGE")
    harness_sha = tokenizer_probe._sha(harness_raw)
    assets = tokenizer_probe._read_assets(args.tokenizer_dir)
    config_probe._read_config(args.model_config)
    for name, expected in tokenizer_probe.DEPENDENCIES.items():
        tokenizer_probe._require(version(name) == expected, "DEPENDENCY_VERSION_MISMATCH")
    snapshot = admission.capture_train_only(args.train, args.manifest,
                                            nemo_source_root=args.nemo_source_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory(prefix="szl-v4-tensor-tokenizer-") as private_dir:
        for name, raw in assets.items():
            (Path(private_dir) / name).write_bytes(raw)
        tokenizer = AutoTokenizer.from_pretrained(private_dir, local_files_only=True,
                                                   trust_remote_code=False, use_fast=True)
        config_probe._qualify_tokenizer(tokenizer)
        binding = _binding(tokenizer)
        report = tokens.check_training_tokens(
            snapshot, binding, lambda messages: _encode(tokenizer, messages),
            lambda ids: _decode(tokenizer, ids))
    tokenizer_probe._require(tokenizer_probe._sha(tokenizer_probe._bounded_read(
        Path(__file__), tokenizer_probe.MAX_HARNESS_BYTES, "HARNESS_SOURCE_TOO_LARGE")) == harness_sha,
        "HARNESS_CHANGED_DURING_RUN")
    record = {
        "schema": SCHEMA, "evidence_class": "MEASURED", "state": "PRECHECK_ONLY",
        "source_binding_evidence_class": "DECLARED", "signature": "UNAVAILABLE",
        "independent_replay": "UNAVAILABLE", "base_id": tokenizer_probe.BASE_ID,
        "base_revision": tokenizer_probe.BASE_REVISION,
        "model_config_sha256": config_probe.CONFIG_SHA256,
        "tokenizer_asset_sha256": tokenizer_probe.ASSET_SHA256,
        "dependency_versions": tokenizer_probe.DEPENDENCIES,
        "harness_sha256": harness_sha,
        "manifest_sha256": report.manifest_sha256,
        "train_sha256": report.train_sha256,
        "conformance_report_sha256": report.conformance_report_sha256,
        "binding_sha256": report.binding_sha256,
        "ordered_tensors_sha256": report.ordered_tensors_sha256,
        "rows": len(report.rows), "total_tokens": report.total_tokens,
        "max_sequence_tokens": max(row.sequence_tokens for row in report.rows),
        "min_sequence_tokens": min(row.sequence_tokens for row in report.rows),
        "local_sequence_cap": PRECHECK_SEQUENCE_LIMIT,
        "local_total_token_cap": PRECHECK_TOTAL_TOKEN_LIMIT,
        "model_config_eos_id": config_probe.CONFIG_EOS_ID,
        "chat_tokenizer_eos_id": tokenizer_probe.EOS_ID,
        "termination_ids": tokenizer_probe.ENDING_IDS,
        "runtime_collator_binding": "UNAVAILABLE",
        "runtime_stop_policy": "UNQUALIFIED_REQUIRES_EXPLICIT_BINDING",
        "model_training": "NOT_RUN", "model_evaluation": "NOT_RUN",
        "hardware": "CPU_ONLY_UNQUALIFIED", "loss": "NOT_RUN",
        "energy_joules": "UNAVAILABLE", "training_eligible": False,
        "publication_eligible": False, "execution_authority": False,
    }
    record["record_sha256"] = tokenizer_probe._sha(tokenizer_probe._canonical(record))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nemo-source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = _run(args)
    except Exception as exc:
        if type(exc) is tokenizer_probe.ProbeError:
            code = str(exc)
        elif isinstance(exc, admission.GateNotReady):
            code = "NEMO_SOURCE_NOT_READY"
        elif isinstance(exc, admission.AdmissionError):
            code = "TRAIN_TOKEN_CONFORMANCE_FAILED"
        else:
            code = "RUNTIME_OR_INPUT_UNAVAILABLE"
        print(json.dumps({"state": "BLOCKED", "code": code,
                          "training_eligible": False, "publication_eligible": False,
                          "execution_authority": False}, sort_keys=True))
        return 1
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
