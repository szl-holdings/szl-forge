"""Bounded local extractive reader; raw margins are NOT calibrated confidence.

The caller must verify the pinned model/tokenizer files before construction and
keep calibration and evaluation data separate. This module does not download,
train, calibrate, generate text, or decide whether an answer should be accepted.
Model artifacts retain their own licenses: deepset/roberta-base-squad2 is CC BY
4.0, with MIT-licensed RoBERTa ancestry; native Transformers code is Apache-2.0.
Keep the upstream model card and applicable notices with redistributed artifacts.

Window aggregation follows the null/span rule described in the official
Transformers QA example (independently implemented here):
https://github.com/huggingface/transformers/blob/
3a275d3581c0ecf962f7412aa764c2047331fd6b/examples/pytorch/question-answering/utils_qa.py
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
import threading


MAX_QUESTION_CHARS = 512
MAX_QUESTION_TOKENS = 128
MAX_CONTEXT_CHARS = 16_000
MAX_SEQUENCE_TOKENS = 384
WINDOW_STRIDE = 128
MAX_WINDOWS = 8
BATCH_SIZE = 8
N_BEST = 20
MAX_ANSWER_TOKENS = 30


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain finite numeric values")
    try:
        converted = float(value)
    except (OverflowError, ValueError, TypeError):
        raise ValueError(f"{name} must contain finite numeric values") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must contain finite numeric values")
    return converted


def decode_windows(context: str, windows: list[dict]) -> dict:
    """Decode windows belonging to ONE unchanged context, never multiple docs.

    Each window contains equally sized lists: ``offsets`` (character pairs),
    ``context_mask`` (bools), ``start_logits``, ``end_logits``, and ``cls_index``.
    Non-context offsets are ignored. Zero-width context tokens are not span
    endpoints. Top-20 start/end selection happens after context/offset masking.
    Ties prefer the earliest character start, then the earliest character end.

    Returns ``prediction, margin, start, end, windows``. Offsets are Python string
    indices with exclusive end. Margin is best_span - min(window_null), so the
    external caller accepts only when margin > its calibrated threshold. No
    valid span, malformed inputs, or nonfinite arithmetic raises ValueError.
    """
    _text(context, "context", MAX_CONTEXT_CHARS)
    _sequence(windows, "windows")
    if not 1 <= len(windows) <= MAX_WINDOWS:
        raise ValueError(f"windows must contain between 1 and {MAX_WINDOWS} items")
    best = None
    minimum_null = None
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("each window must be a mapping")
        try:
            offsets = _sequence(window["offsets"], "offsets")
            mask = _sequence(window["context_mask"], "context_mask")
            starts = _sequence(window["start_logits"], "start_logits")
            ends = _sequence(window["end_logits"], "end_logits")
            cls = window["cls_index"]
        except KeyError as error:
            raise ValueError(f"window missing {error.args[0]}") from None
        length = len(offsets)
        if not 1 <= length <= MAX_SEQUENCE_TOKENS:
            raise ValueError("window token count is out of bounds")
        if any(len(values) != length for values in (mask, starts, ends)):
            raise ValueError("window lists must have identical lengths")
        if any(type(item) is not bool for item in mask):
            raise ValueError("context_mask must contain booleans")
        if type(cls) is not int or not 0 <= cls < length or mask[cls]:
            raise ValueError("cls_index must identify a non-context token")
        starts = [_finite(value, "start_logits") for value in starts]
        ends = [_finite(value, "end_logits") for value in ends]
        null = _finite(starts[cls] + ends[cls], "null score")
        minimum_null = null if minimum_null is None else min(minimum_null, null)

        valid = []
        previous = (0, 0)
        for index, is_context in enumerate(mask):
            if not is_context:
                continue
            offset = _sequence(offsets[index], "context offset")
            if (len(offset) != 2 or any(type(item) is not int for item in offset)
                    or not 0 <= offset[0] <= offset[1] <= len(context)):
                raise ValueError("invalid context character offset")
            if offset[0] < previous[0] or offset[1] < previous[1]:
                raise ValueError("context offsets must be monotonically ordered")
            previous = offset
            if offset[0] < offset[1]:
                valid.append(index)
        start_candidates = sorted(valid, key=lambda index: (-starts[index], index))[:N_BEST]
        end_candidates = sorted(valid, key=lambda index: (-ends[index], index))[:N_BEST]
        for start_token in start_candidates:
            for end_token in end_candidates:
                if not start_token <= end_token < start_token + MAX_ANSWER_TOKENS:
                    continue
                if not all(mask[start_token:end_token + 1]):
                    continue
                start, end = offsets[start_token][0], offsets[end_token][1]
                if not start < end or not context[start:end].strip():
                    continue
                score = _finite(starts[start_token] + ends[end_token], "span score")
                candidate = (score, start, end)
                if (best is None or score > best[0]
                        or (score == best[0] and (start, end) < best[1:])):
                    best = candidate
    if best is None:
        raise ValueError("no valid context span")
    score, start, end = best
    margin = _finite(score - minimum_null, "span-minus-null margin")
    return {"prediction": context[start:end], "margin": margin,
            "start": start, "end": end, "windows": len(windows)}


class ExtractiveReader:
    """Native float32 QA over a caller-verified, local-only model directory."""

    def __init__(self, path: Path, device="cuda"):
        snapshot = Path(path).resolve(strict=True)
        if not snapshot.is_dir():
            raise ValueError("reader path must be a verified local snapshot directory")
        # Lazy imports keep the pure decoder and its tests dependency-free.
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("reader device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA reader requested but CUDA is unavailable")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise ValueError("reader requires a native fast tokenizer with offsets")
        if self.tokenizer.cls_token_id is None:
            raise ValueError("reader tokenizer has no CLS token")
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32,
        )
        if self.model.config.model_type != "roberta":
            raise ValueError("reader requires the pinned native RoBERTa QA architecture")
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self._lock = threading.Lock()

    def predict(self, question: str, context: str) -> dict:
        """Return a candidate and raw margin; acceptance belongs to the caller."""
        _text(question, "question", MAX_QUESTION_CHARS)
        _text(context, "context", MAX_CONTEXT_CHARS)
        with self._lock:
            return self._predict(question, context)

    def _predict(self, question, context):
        question_ids = self.tokenizer(
            question, add_special_tokens=False, truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if not 1 <= len(question_ids) <= MAX_QUESTION_TOKENS:
            raise ValueError(f"question must contain 1 to {MAX_QUESTION_TOKENS} tokens")
        encoded = self.tokenizer(
            question, context, truncation="only_second", max_length=MAX_SEQUENCE_TOKENS,
            stride=WINDOW_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, return_token_type_ids=False, padding=False,
        )
        count = len(encoded["input_ids"])
        if not 1 <= count <= MAX_WINDOWS:
            raise ValueError(f"context requires more than {MAX_WINDOWS} reader windows")
        features = []
        for index in range(count):
            ids = encoded["input_ids"][index]
            attention = encoded["attention_mask"][index]
            offsets = encoded["offset_mapping"][index]
            sequence_ids = encoded.sequence_ids(index)
            if (not 1 <= len(ids) <= MAX_SEQUENCE_TOKENS
                    or any(len(values) != len(ids) for values in (attention, offsets, sequence_ids))
                    or any(type(value) is not int or value not in (0, 1) for value in attention)
                    or any(value is not None and (type(value) is not int or value not in (0, 1))
                           for value in sequence_ids)):
                raise ValueError("invalid tokenizer window metadata")
            cls_positions = [position for position, token in enumerate(ids)
                             if token == self.tokenizer.cls_token_id
                             and sequence_ids[position] is None and attention[position] == 1]
            if len(cls_positions) != 1:
                raise ValueError("tokenizer window must contain one active special CLS token")
            features.append({"offsets": offsets,
                             "context_mask": [seq == 1 and bool(active)
                                              for seq, active in zip(sequence_ids, attention)],
                             "cls_index": cls_positions[0]})
        torch = self._torch
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            for begin in range(0, count, BATCH_SIZE):
                stop = min(begin + BATCH_SIZE, count)
                batch = self.tokenizer.pad(
                    {"input_ids": encoded["input_ids"][begin:stop],
                     "attention_mask": encoded["attention_mask"][begin:stop]},
                    padding=True, return_tensors="pt",
                )
                batch = {name: value.to(self.device) for name, value in batch.items()}
                output = self.model(**batch)
                for name in ("start_logits", "end_logits"):
                    logits = getattr(output, name)
                    if (tuple(logits.shape) != tuple(batch["input_ids"].shape)
                            or not torch.isfinite(logits).all().item()):
                        raise ValueError("reader returned invalid or nonfinite logits")
                    values = logits.detach().to(device="cpu", dtype=torch.float32).tolist()
                    for row, feature_index in enumerate(range(begin, stop)):
                        features[feature_index][name] = values[row][:len(encoded["input_ids"][feature_index])]
        return decode_windows(context, features)
