# SPDX-License-Identifier: Apache-2.0
"""Pure train-only token/label conformance, not runtime qualification or authority.

The caller must freshly capture a TrainOnlySnapshot and supply a trusted adapter
for its actual renderer, tokenizer and collator. This module imports no model,
reads no paths and downloads nothing. Digests and limits are DECLARED bindings,
not authenticated assets or signed admission. A saved report cannot replace a
fresh check of the exact tensors consumed by a future separately admitted runner.

Offsets use Python character indices, not UTF-8 byte offsets. Ordinary tokens
must provide exact nonoverlapping coverage. Zero-offset special tokens are
accepted only through the explicit ID/text mapping. Tokenizers with ambiguous
content offsets or non-exact target decoding fail closed. Only right padding is
supported; sequence packing, target truncation and extra assistant text are not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Callable

import curriculum_admission as admission
import json_contract as contract
import training_data

BASE_ID = "unsloth/Qwen3.5-0.8B"
BASE_REVISION = "23c69c53358a07516b5827588b3fdb12ae78fd65"
MAX_SEQUENCE = 65_536
MAX_RENDERED_BYTES = 4 * contract.MAX_BYTES
IGNORE_INDEX = -100
SCHEMA = "szl.receiptagent.v4-training-tokens/v1"


@dataclass(frozen=True, slots=True)
class TokenBinding:
    """Declared assets/policies and resource ceilings; no permission flags."""

    base_id: str
    base_revision: str
    tokenizer_sha256: str
    template_sha256: str
    dependency_sha256: str
    masking_sha256: str
    eos_sha256: str
    thinking_sha256: str
    tokenizer_limit: int
    model_limit: int
    admitted_limit: int
    total_token_limit: int
    vocab_size: int
    pad_token_id: int
    render_fragments: tuple[str, str, str]
    special_tokens: tuple[tuple[int, str], ...]
    termination_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EncodedRow:
    """Exact output of a trusted renderer/tokenizer/collator, before shifting."""

    rendered: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    labels: tuple[int, ...]
    message_spans: tuple[tuple[int, int], ...]
    termination_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RowLedger:
    index: int
    sequence_tokens: int
    active_tokens: int
    assistant_tokens: int
    termination_tokens: int
    rendered_sha256: str
    target_sha256: str
    tensors_sha256: str


@dataclass(frozen=True, slots=True)
class TokenReport:
    binding_sha256: str
    manifest_sha256: str
    train_sha256: str
    conformance_report_sha256: str
    sequence_limit: int
    total_tokens: int
    rows: tuple[RowLedger, ...]
    ordered_tensors_sha256: str
    schema: str = SCHEMA
    evidence_class: str = "REPORTED"
    assets_authenticated: bool = False
    training_eligible: bool = False
    publication_eligible: bool = False
    execution_authority: bool = False


@dataclass(frozen=True, slots=True)
class TrainingTensorRow:
    """Checked immutable tensors; deliberately omitted from object representations."""

    input_ids: tuple[int, ...] = field(repr=False)
    attention_mask: tuple[int, ...] = field(repr=False)
    labels: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class RetainedTrainingTokens:
    """All checked train tensors and their redacted ledger, never a launch permit.

    The tensors contain private training content. Do not serialize or publish
    this object; publish only its separately redacted ``report`` if appropriate.
    Frozen containers are a trusted-process interface, not a security boundary.
    """

    rows: tuple[TrainingTensorRow, ...] = field(repr=False)
    report: TokenReport


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise admission.AdmissionError(message)


def _integer(value: object, minimum: int, maximum: int, label: str) -> None:
    _require(type(value) is int and minimum <= value <= maximum,
             f"{label} must be a bounded exact integer")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_digest(value: object) -> str:
    return _digest(contract.canonical_json(value).encode("utf-8"))


def _binding(binding: TokenBinding) -> tuple[int, dict[int, str]]:
    _require(type(binding) is TokenBinding, "an exact TokenBinding is required")
    _require(binding.base_id == BASE_ID and binding.base_revision == BASE_REVISION,
             "base identity must match the immutable v4 preregistration")
    for name in ("tokenizer", "template", "dependency", "masking", "eos", "thinking"):
        value = getattr(binding, name + "_sha256")
        _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                 "asset/policy digests must be exact lowercase SHA256 strings")
    limits = (binding.tokenizer_limit, binding.model_limit, binding.admitted_limit)
    for value in limits:
        _integer(value, 2, MAX_SEQUENCE, "sequence limit")
    _integer(binding.total_token_limit, 1, admission.MAX_ROWS * min(limits), "total limit")
    _integer(binding.vocab_size, 2, 2**31 - 1, "vocabulary size")
    _integer(binding.pad_token_id, 0, binding.vocab_size - 1, "padding ID")
    _require(type(binding.render_fragments) is tuple and len(binding.render_fragments) == 3,
             "three exact static message-prefix fragments are required")
    for text in binding.render_fragments:
        _require(type(text) is str and len(text) <= 1024, "render fragment must be bounded text")
        try:
            _require(len(text.encode("utf-8")) <= 1024, "render fragment byte limit exceeded")
        except UnicodeError:
            raise admission.AdmissionError("invalid render-fragment Unicode") from None
    _require(type(binding.special_tokens) is tuple and 1 <= len(binding.special_tokens) <= 64,
             "special-token mapping must be a bounded immutable tuple")
    special: dict[int, str] = {}
    for item in binding.special_tokens:
        _require(type(item) is tuple and len(item) == 2, "invalid special-token mapping")
        token, text = item
        _integer(token, 0, binding.vocab_size - 1, "special token ID")
        _require(type(text) is str and 0 < len(text) <= 128 and token not in special,
                 "special tokens require unique IDs and bounded nonempty strings")
        try:
            text.encode("utf-8")
        except UnicodeError as exc:
            raise admission.AdmissionError("invalid special-token Unicode") from exc
        special[token] = text
    _require(type(binding.termination_ids) is tuple and 1 <= len(binding.termination_ids) <= 8,
             "termination IDs must be an explicit bounded immutable tuple")
    for token in binding.termination_ids:
        _integer(token, 0, binding.vocab_size - 1, "termination ID")
        _require(token in special, "every termination ID needs an exact special-token string")
    return min(limits), special


def _span(value: object, size: int) -> tuple[int, int]:
    _require(type(value) is tuple and len(value) == 2, "span must be an exact pair")
    start, end = value
    _integer(start, 0, size, "span start")
    _integer(end, start + 1, size, "span end")
    return start, end


def _row(index: int, messages: tuple[tuple[str, str], ...], row: EncodedRow,
         binding: TokenBinding, special: dict[int, str], limit: int,
         decode: Callable[[tuple[int, ...]], str]) -> RowLedger:
    _require(type(row) is EncodedRow, "adapter must return an exact EncodedRow")
    _require(type(row.rendered) is str and 0 < len(row.rendered) <= MAX_RENDERED_BYTES,
             "rendered text must be a bounded nonempty string")
    try:
        rendered_raw = row.rendered.encode("utf-8")
    except UnicodeError as exc:
        raise admission.AdmissionError("invalid rendered Unicode") from exc
    _require(len(rendered_raw) <= MAX_RENDERED_BYTES, "rendered UTF-8 byte limit exceeded")
    n = len(row.input_ids) if type(row.input_ids) is tuple else 0
    _require(2 <= n <= limit, "sequence exceeds the minimum declared capacity or is empty")
    for values in (row.input_ids, row.attention_mask, row.labels, row.offsets):
        _require(type(values) is tuple and len(values) == n, "tensor shape/type mismatch")
    _require(type(row.message_spans) is tuple and len(row.message_spans) == 3,
             "all three exact message spans are required")
    spans = tuple(_span(span, len(row.rendered)) for span in row.message_spans)
    previous_end = 0
    for (_, content), (start, end) in zip(messages, spans):
        _require(start >= previous_end and row.rendered[start:end] == content,
                 "message was rewritten, reordered or truncated")
        previous_end = end
    assistant_start, assistant_end = spans[-1]
    end_start, end_end = _span(row.termination_span, len(row.rendered))
    ending = "".join(special[token] for token in binding.termination_ids)
    expected_render = ""
    expected_spans = []
    for fragment, (_, content) in zip(binding.render_fragments, messages):
        expected_render += fragment
        expected_spans.append((len(expected_render), len(expected_render) + len(content)))
        expected_render += content
    _require(row.rendered == expected_render + ending and spans == tuple(expected_spans),
             "rendering differs from bound framing; packing or extra text is forbidden")
    _require(end_start == assistant_end and end_end == len(row.rendered)
             and row.rendered[end_start:end_end] == ending,
             "only the exact declared termination may follow the assistant")
    target = messages[-1][1]
    _require(contract.canonical_json(contract.strict_object(target)) == target,
             "assistant target must remain one canonical JSON object")
    active = 0
    cursor = 0
    content_indices: list[int] = []
    ending_indices: list[int] = []
    padding_started = False
    for position, (token, attention, label, offset) in enumerate(zip(
            row.input_ids, row.attention_mask, row.labels, row.offsets)):
        _integer(token, 0, binding.vocab_size - 1, "input ID")
        _integer(attention, 0, 1, "attention mask")
        _require(type(label) is int and (label == IGNORE_INDEX or 0 <= label < binding.vocab_size),
                 "label must be an exact vocabulary ID or ignore index")
        _require(type(offset) is tuple and len(offset) == 2
                 and all(type(value) is int for value in offset), "invalid character offset")
        if attention == 0:
            padding_started = True
            _require(token == binding.pad_token_id and label == IGNORE_INDEX and offset == (0, 0),
                     "padding must be masked, use its bound ID, and have no content offset")
            continue
        _require(not padding_started, "only right padding is supported")
        start, end = offset
        if offset == (0, 0):
            _require(token in special, "zero-offset ordinary content is unreliable")
            start, end = cursor, cursor + len(special[token])
        _require(start == cursor and start < end <= len(row.rendered),
                 "character mapping has a gap, overlap, truncation or invalid bound")
        if token in special:
            _require(row.rendered[start:end] == special[token], "special token text differs")
        if end <= assistant_start:
            expected_label = IGNORE_INDEX
        elif start >= assistant_start and end <= assistant_end:
            _require(token not in special, "assistant JSON cannot contain special tokens")
            content_indices.append(position)
            expected_label = token
        elif start >= end_start:
            ending_indices.append(position)
            expected_label = token
        else:
            raise admission.AdmissionError("token straddles an assistant supervision boundary")
        _require(label == expected_label, "collator labels differ from complete assistant-only supervision")
        cursor = end
        active += 1
    _require(cursor == len(row.rendered), "token sequence does not cover the complete rendered text")
    _require(content_indices and content_indices[0] > 0,
             "first assistant target requires a preceding causal prediction position")
    _require(ending_indices and tuple(row.input_ids[i] for i in ending_indices) == binding.termination_ids,
             "termination IDs differ or EOS target is missing")
    # Labels are intentionally unshifted. A causal-LM loss pairs logits[:-1]
    # with labels[1:]; both the first JSON token and final EOS must survive it.
    for position in (content_indices[0], ending_indices[-1]):
        _require(position > 0 and row.attention_mask[position - 1] == 1
                 and row.labels[position] == row.input_ids[position], "causal target is not supervised")
    for ids, expected in ((row.input_ids[:active], row.rendered),
                          (tuple(row.input_ids[i] for i in content_indices), target),
                          (binding.termination_ids, ending)):
        try:
            decoded = decode(ids)
        except Exception:
            raise admission.AdmissionError("trusted decoder failed; details redacted") from None
        _require(type(decoded) is str and decoded == expected, "exact tokenizer decode roundtrip failed")
    tensors = (row.input_ids, row.attention_mask, row.labels)
    return RowLedger(index, n, active, len(content_indices), len(ending_indices),
                     _digest(rendered_raw), _digest(target.encode("utf-8")), _json_digest(tensors))


def _check_training_tokens(snapshot: admission.TrainOnlySnapshot, binding: TokenBinding,
                           encode: Callable[[tuple[tuple[str, str], ...]], EncodedRow],
                           decode: Callable[[tuple[int, ...]], str], *,
                           retain: bool) -> tuple[TokenReport, tuple[TrainingTensorRow, ...]]:
    limit, special = _binding(binding)
    _require(callable(encode) and callable(decode), "trusted encode/decode adapters are required")
    messages = training_data.training_messages(snapshot)
    _require(len(messages) == admission.MAX_ROWS, "all 360 training rows are required")
    rows: list[RowLedger] = []
    retained: list[TrainingTensorRow] = []
    total = 0
    for index, message in enumerate(messages):
        try:
            encoded = encode(message)
        except Exception:
            raise admission.AdmissionError("trusted encoder/collator failed; details redacted") from None
        _require(type(encoded) is EncodedRow, "adapter must return an exact EncodedRow")
        # Detach the wrapper before invoking the decoder. Every nested value is
        # checked below for exact immutable types before any tensor is retained;
        # a callback cannot swap fields on its original wrapper after validation.
        checked = EncodedRow(encoded.rendered, encoded.input_ids, encoded.attention_mask,
                             encoded.offsets, encoded.labels, encoded.message_spans,
                             encoded.termination_span)
        ledger = _row(index, message, checked, binding, special, limit, decode)
        total += ledger.sequence_tokens
        _require(total <= binding.total_token_limit, "aggregate token capacity exceeded")
        rows.append(ledger)
        if retain:
            retained.append(TrainingTensorRow(checked.input_ids, checked.attention_mask,
                                              checked.labels))
    report = TokenReport(_json_digest(asdict(binding)), _digest(snapshot.manifest_bytes),
                         _digest(snapshot.train_bytes), _digest(snapshot.conformance_report_bytes),
                         limit, total, tuple(rows),
                         _json_digest(tuple((row.index, row.tensors_sha256) for row in rows)))
    return report, tuple(retained)


def check_training_tokens(snapshot: admission.TrainOnlySnapshot, binding: TokenBinding,
                          encode: Callable[[tuple[tuple[str, str], ...]], EncodedRow],
                          decode: Callable[[tuple[int, ...]], str]) -> TokenReport:
    """Check all train rows and return only the compatible redacted hash ledger.

    ``encode`` must return actual collator labels and reliable offsets, not a
    self-reported pass flag. Both callbacks are trusted Python code, not a sandbox.
    Reports contain no target text or token IDs; hashes are integrity bindings,
    not confidentiality guarantees. No report authenticates the declared assets.
    A future worker must retain/use these exact immutable tensors in the trusted
    adapter process or requalify them. This function alone cannot prevent a later
    retokenization. The fixed three-message framing is deliberately restrictive.
    """
    report, _ = _check_training_tokens(snapshot, binding, encode, decode, retain=False)
    return report


def retain_training_tokens(snapshot: admission.TrainOnlySnapshot, binding: TokenBinding,
                           encode: Callable[[tuple[tuple[str, str], ...]], EncodedRow],
                           decode: Callable[[tuple[int, ...]], str]) -> RetainedTrainingTokens:
    """Return all exact checked tensors only after every train row passes.

    Each row is encoded once, checked by the same conformance/capacity path as
    ``check_training_tokens``, and retained as immutable integer tuples. No
    partial result is returned on failure. The caller must consume these tensors
    without changing or retokenizing them in its separately admitted runtime.
    This function adds no loader, optimizer, device, signing or launch interface.
    Its redacted report preserves all false eligibility/authority fields.
    """
    report, rows = _check_training_tokens(snapshot, binding, encode, decode, retain=True)
    return RetainedTrainingTokens(rows, report)
