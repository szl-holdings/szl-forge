# SPDX-License-Identifier: Apache-2.0
"""Offline, train-only compatibility precheck for the pinned v4 base tokenizer.

This checks rendering and offsets, not model weights, a trainer collator, signed
curriculum admission, GPU resources, held-out quality, or release eligibility.
The seven local tokenizer assets must match the reviewed byte commitments.
Its explicit dataset input is the committed train split; it does not request
model weights. Supplied asset paths and installed dependencies remain trusted
inputs, not a sandbox against hostile filesystem or Python code.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

BASE_ID = "unsloth/Qwen3.5-0.8B"
BASE_REVISION = "23c69c53358a07516b5827588b3fdb12ae78fd65"
TRAIN_SHA256 = "26c3b11c64d0d6201e29716656c1898893fb449249f6a67611a68f477a64699f"
MANIFEST_SHA256 = "7ec31cfddf31228164a7a67e7edd8152c611c2f04a4860d2ad85c168339337ae"
ROWS = 360
SEQUENCE_LIMIT = 65_536
MAX_ASSET_BYTES = 32 * 1024 * 1024
MAX_TRAIN_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_HARNESS_BYTES = 1024 * 1024
EOS_ID = 248046
PAD_ID = 248055
ENDING_IDS = (248046, 198)
ASSET_SHA256 = {
    "added_tokens.json": "e5f9bfb644ead13bd36dd6cfc41553a93b7c264c0a2e1ac67d033aa8250c8538",
    "chat_template.jinja": "ea210e5b4997b7b1d3af5ee3ad34e7677ffe38df7469e8d88089789b5a27fae5",
    "merges.txt": "3bd640ba6d8da8f5844f3548b7e2184fc664dd670e1447e425a48dbaaad1ef43",
    "special_tokens_map.json": "ce5cbe737f4aacc7696180aac743091101b59e4dc5a338d9fc8a6f716aa513f7",
    "tokenizer_config.json": "50470511708577d43f7c18c2d8e2ab02fb8abf3add1c40f60b74579d6d16e27e",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    "vocab.json": "4ab0d5c096294054b66444a116a20baf6e29998fc1fae410ffe4b6fadbc56b5c",
}
DEPENDENCIES = {
    "transformers": "5.5.0",
    "huggingface-hub": "1.5.0",
    "tokenizers": "0.22.2",
    "jinja2": "3.1.6",
}


class ProbeError(Exception):
    """Bounded diagnostic that must never include row content."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ProbeError(code)


def _plain_local_stat(path: Path, *, directory: bool = False) -> os.stat_result:
    _require(path.is_absolute() and not str(path).startswith("\\\\"),
             "TOKENIZER_LOCAL_PATH_REQUIRED")
    if os.name == "nt":
        import ctypes
        _require(ctypes.windll.kernel32.GetDriveTypeW(path.anchor) == 3,
                 "TOKENIZER_FIXED_DRIVE_REQUIRED")
    current = Path(path.anchor)
    info = os.lstat(current)
    for part in (None, *path.parts[1:]):
        if part is not None:
            current /= part
            info = os.lstat(current)
        _require(not stat.S_ISLNK(info.st_mode)
                 and not (getattr(info, "st_file_attributes", 0)
                          & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
                 "TOKENIZER_PATH_REDIRECTED")
    _require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
             "TOKENIZER_PATH_TYPE_MISMATCH")
    return info


def _bounded_read(path: Path, limit: int, code: str, *, local_asset: bool = False) -> bytes:
    before = _plain_local_stat(path) if local_asset else None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if local_asset:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if before is not None:
            opened = os.fstat(stream.fileno())
            _require(stat.S_ISREG(opened.st_mode)
                     and (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
                     "TOKENIZER_PATH_CHANGED")
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, code)
    return raw


def _read_assets(directory: Path) -> dict[str, bytes]:
    _plain_local_stat(directory, directory=True)
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            _require(len(names) < len(ASSET_SHA256), "TOKENIZER_ASSET_SET_MISMATCH")
            names.append(entry.name)
    _require(set(names) == set(ASSET_SHA256), "TOKENIZER_ASSET_SET_MISMATCH")
    assets = {}
    for name, expected in ASSET_SHA256.items():
        path = directory / name
        raw = _bounded_read(path, MAX_ASSET_BYTES, "TOKENIZER_ASSET_TOO_LARGE",
                            local_asset=True)
        _require(_sha(raw) == expected, "TOKENIZER_ASSET_HASH_MISMATCH")
        assets[name] = raw
    return assets


def _read_train(root: Path) -> tuple[dict, ...]:
    manifest_raw = _bounded_read(root / "curriculum-manifest.json",
                                 MAX_MANIFEST_BYTES, "TRAIN_MANIFEST_TOO_LARGE")
    _require(_sha(manifest_raw) == MANIFEST_SHA256, "TRAIN_MANIFEST_MISMATCH")
    manifest = json.loads(manifest_raw)
    _require(manifest["splits"]["train"]["sha256"] == TRAIN_SHA256
             and manifest["seed"] == 2601003,
             "TRAIN_MANIFEST_MISMATCH")
    raw = _bounded_read(root / "train.jsonl", MAX_TRAIN_BYTES, "TRAIN_BYTES_TOO_LARGE")
    _require(_sha(raw) == TRAIN_SHA256 and b"\r" not in raw and raw.endswith(b"\n"),
             "TRAIN_BYTES_MISMATCH")
    lines = raw.splitlines()
    _require(len(lines) == ROWS, "TRAIN_ROW_COUNT_MISMATCH")
    rows = tuple(json.loads(line) for line in lines)
    for row in rows:
        messages = row.get("messages")
        _require(row.get("split") == "train" and isinstance(messages, list)
                 and [m.get("role") for m in messages] == ["system", "user", "assistant"]
                 and all(isinstance(m.get("content"), str) and m["content"] for m in messages),
                 "TRAIN_MESSAGE_SHAPE_MISMATCH")
    return rows


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def _measure(rows: tuple[dict, ...], tokenizer, chain_seed: bytes) -> dict:
    _require(tokenizer.is_fast and type(tokenizer).__name__ == "Qwen2Tokenizer",
             "TOKENIZER_TYPE_MISMATCH")
    _require(tokenizer.eos_token_id == EOS_ID and tokenizer.pad_token_id == PAD_ID,
             "TOKENIZER_TERMINATION_MISMATCH")
    total = 0
    minimum = SEQUENCE_LIMIT
    maximum = 0
    fragments = None
    ending = None
    chain = chain_seed
    for index, row in enumerate(rows):
        messages = row["messages"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=False)
        encoded = tokenizer(rendered, add_special_tokens=False,
                            return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        _require(2 <= len(ids) <= SEQUENCE_LIMIT and len(ids) == len(offsets),
                 f"ROW_{index}_CAPACITY_OR_OFFSETS")
        _require(_decode(tokenizer, ids) == rendered, f"ROW_{index}_FULL_DECODE")

        cursor = 0
        current_fragments = []
        spans = []
        for message in messages:
            content = message["content"]
            start = rendered.find(content, cursor)
            _require(start >= cursor, f"ROW_{index}_MESSAGE_MISSING")
            current_fragments.append(rendered[cursor:start])
            spans.append((start, start + len(content)))
            cursor = start + len(content)
        current_ending = rendered[cursor:]
        if fragments is None:
            fragments, ending = current_fragments, current_ending
        _require(current_fragments == fragments and current_ending == ending,
                 f"ROW_{index}_VARIABLE_FRAMING")
        assistant_start, assistant_end = spans[-1]
        target_ids: list[int] = []
        ending_ids: list[int] = []
        cursor = 0
        for token, (start, end) in zip(ids, offsets):
            _require(start == cursor and start < end <= len(rendered),
                     f"ROW_{index}_OFFSET_GAP_OR_OVERLAP")
            if end <= assistant_start:
                pass
            elif start >= assistant_start and end <= assistant_end:
                _require(token not in tokenizer.all_special_ids,
                         f"ROW_{index}_SPECIAL_TARGET_TOKEN")
                target_ids.append(token)
            elif start >= assistant_end:
                ending_ids.append(token)
            else:
                raise ProbeError(f"ROW_{index}_SUPERVISION_BOUNDARY")
            cursor = end
        _require(cursor == len(rendered) and target_ids and tuple(ending_ids) == ENDING_IDS,
                 f"ROW_{index}_INCOMPLETE_OR_WRONG_ENDING")
        _require(_decode(tokenizer, target_ids) == messages[-1]["content"]
                 and _decode(tokenizer, ending_ids) == ending,
                 f"ROW_{index}_TARGET_OR_ENDING_DECODE")
        row_hash = hashlib.sha256(_canonical({
            "index": index, "rendered_sha256": _sha(rendered.encode("utf-8")),
            "input_ids": ids, "offsets": offsets,
        })).digest()
        chain = hashlib.sha256(chain + row_hash).digest()
        total += len(ids)
        minimum = min(minimum, len(ids))
        maximum = max(maximum, len(ids))
    return {"rows": len(rows), "total_tokens": total, "min_tokens": minimum,
            "max_tokens": maximum, "termination_ids": ENDING_IDS,
            "ordered_row_chain_sha256": chain.hex(),
            "ending_sha256": _sha(ending.encode("utf-8"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        harness_sha = _sha(_bounded_read(Path(__file__), MAX_HARNESS_BYTES,
                                         "HARNESS_SOURCE_TOO_LARGE"))
        assets = _read_assets(args.tokenizer_dir)
        for name, expected in DEPENDENCIES.items():
            _require(version(name) == expected, "DEPENDENCY_VERSION_MISMATCH")
        rows = _read_train(root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        binding_sha = _sha(_canonical({
            "base_id": BASE_ID, "base_revision": BASE_REVISION,
            "dataset_sha256": TRAIN_SHA256, "manifest_sha256": MANIFEST_SHA256,
            "harness_sha256": harness_sha, "asset_sha256": ASSET_SHA256,
            "dependency_versions": DEPENDENCIES,
        }))
        from transformers import AutoTokenizer
        # Load exactly the verified bytes, not a mutable source path after check.
        with tempfile.TemporaryDirectory(prefix="szl-v4-tokenizer-") as private_dir:
            for name, raw in assets.items():
                (Path(private_dir) / name).write_bytes(raw)
            tokenizer = AutoTokenizer.from_pretrained(private_dir,
                                                       local_files_only=True,
                                                       trust_remote_code=False,
                                                       use_fast=True)
            metrics = _measure(rows, tokenizer, bytes.fromhex(binding_sha))
        _require(_sha(_bounded_read(Path(__file__), MAX_HARNESS_BYTES,
                                   "HARNESS_SOURCE_TOO_LARGE")) == harness_sha,
                 "HARNESS_CHANGED_DURING_RUN")
    except Exception as exc:
        code = str(exc) if type(exc) is ProbeError else "RUNTIME_OR_INPUT_UNAVAILABLE"
        print(json.dumps({"state": "BLOCKED", "code": code,
                          "training_eligible": False, "publication_eligible": False,
                          "execution_authority": False}, sort_keys=True))
        return 1
    record = {"schema": "szl.receiptagent.v4-tokenizer-precheck/v1",
                      "evidence_class": "MEASURED", "source_binding_evidence_class": "DECLARED",
                      "signature": "UNAVAILABLE", "independent_replay": "UNAVAILABLE",
                      "state": "PRECHECK_ONLY",
                      "base_id": BASE_ID, "base_revision": BASE_REVISION,
                      "dataset_sha256": TRAIN_SHA256, "manifest_sha256": MANIFEST_SHA256,
                      "curriculum_seed": 2601003,
                      "hardware": "CPU_ONLY_UNQUALIFIED", "loss": "NOT_RUN",
                      "energy_joules": "UNAVAILABLE",
                      "harness_sha256": harness_sha, "binding_sha256": binding_sha,
                      "asset_sha256": ASSET_SHA256, "dependency_versions": DEPENDENCIES,
                      "metrics": metrics, "training_eligible": False,
                      "publication_eligible": False, "execution_authority": False}
    # Unsigned self-checksum binds fields; it is not origin authentication.
    record["record_sha256"] = _sha(_canonical(record))
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
