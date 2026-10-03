"""Fail-closed guard: every tensor of a LoRA checkpoint must land in the loaded model.

Why this exists (2026-09-30, szl-forge PR #444 evidence): the published chaski adapters are
keyed for the multimodal module layout (``base_model.model.model.language_model.layers.*``).
When the base is instantiated through a class whose modules live at ``model.layers.*``
(transformers 5.18 ``AutoModelForCausalLM`` -> ``Qwen3_5ForCausalLM``), PEFT applies **0 of
192** adapter tensors and only emits a ``UserWarning``. The "adapter" candidate then reproduces
the bare base model byte-for-byte and would be receipted as a MEASURED 0/5 with no error.

This module makes that condition a hard failure and a receipt field. The key logic is pure
Python (the safetensors header is 8 bytes of little-endian length + JSON), so it is unit-tested
without torch; the model-facing helpers only need ``named_parameters()``.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ADAPTER_WEIGHTS = "adapter_model.safetensors"
# Local LoRA inspection profile, deliberately below SafeTensors' 100 MB ceiling.
MAX_ADAPTER_HEADER_BYTES = 8 * 1024 * 1024
MAX_UNSIGNED_64 = (1 << 64) - 1
# SafeTensors v0.8.0 bit widths, including packed and complex types.
DTYPE_BITS = {
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
    **dict.fromkeys(("BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0",
                     "F8_E4M3FNUZ", "F8_E5M2FNUZ"), 8),
    **dict.fromkeys(("I16", "U16", "F16", "BF16"), 16),
    **dict.fromkeys(("I32", "U32", "F32"), 32),
    **dict.fromkeys(("I64", "U64", "F64", "C64"), 64),
}


class AdapterNotApplied(RuntimeError):
    """Raised when at least one checkpoint tensor has no target in the loaded model."""


class InvalidAdapterCheckpoint(AdapterNotApplied):
    """Malformed or unsupported checkpoint; never eligible for key coverage."""


def _invalid(reason: str) -> InvalidAdapterCheckpoint:
    # Reasons are fixed strings: never echo untrusted header values into a receipt.
    return InvalidAdapterCheckpoint(f"INVALID_ADAPTER_CHECKPOINT: {reason}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _invalid("nonstandard JSON numeric constant")


def _valid_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


def _validate_header(header: Any, payload_bytes: int) -> list[str]:
    if not isinstance(header, dict):
        raise _invalid("header must be an object")
    intervals: list[tuple[int, int]] = []
    names: list[str] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or not all(
                _valid_string(k) and _valid_string(v) for k, v in descriptor.items()
            ):
                raise _invalid("metadata must contain string pairs")
            continue
        if not _valid_string(name) or not name:
            raise _invalid("invalid tensor name")
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise _invalid("invalid tensor descriptor")
        dtype, shape, offsets = (descriptor[k] for k in ("dtype", "shape", "data_offsets"))
        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            raise _invalid("unsupported tensor dtype")
        if not isinstance(shape, list) or any(
            type(dim) is not int or not 0 <= dim <= MAX_UNSIGNED_64 for dim in shape
        ):
            raise _invalid("invalid tensor shape")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)):
            raise _invalid("invalid tensor offsets")
        start, end = offsets
        if not start <= end <= payload_bytes:
            raise _invalid("tensor offsets exceed payload")
        bits = DTYPE_BITS[dtype]
        elements = 1
        for dim in shape:
            # Preserve ordered checked multiplication, even before a zero dimension.
            if elements and dim > MAX_UNSIGNED_64 // elements:
                raise _invalid("tensor shape exceeds integer capacity")
            elements *= dim
        if elements > MAX_UNSIGNED_64 // bits:
            raise _invalid("tensor bit size exceeds integer capacity")
        total_bits = elements * bits
        if total_bits % 8 or total_bits // 8 != end - start:
            raise _invalid("tensor byte size disagrees with shape and dtype")
        intervals.append((start, end))
        names.append(name)
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise _invalid("tensor payload has a hole or overlap")
        cursor = end
    if cursor != payload_bytes:
        raise _invalid("tensor payload has an uncovered tail")
    return sorted(names)


def checkpoint_keys(adapter_dir: Path) -> list[str]:
    """Bounded strict header/layout inspection, without reading weights or importing torch.

    This local profile accepts known v0.8.0 dtypes and exact descriptor fields. It is
    not a replacement for the loader, artifact hashes, or numerical adapter evaluation.
    """
    path = Path(adapter_dir) / ADAPTER_WEIGHTS
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise _invalid("truncated length prefix")
        (header_len,) = struct.unpack("<Q", prefix)
        file_size = os.fstat(handle.fileno()).st_size
        if not 2 <= header_len <= MAX_ADAPTER_HEADER_BYTES or header_len > file_size - 8:
            raise _invalid("header length exceeds inspection bounds")
        body = handle.read(header_len)
        if len(body) != header_len:
            raise _invalid("truncated header")
    if not body.startswith(b"{"):
        raise _invalid("header must start with an object")
    try:
        header = json.loads(body.decode("utf-8", errors="strict"),
                            object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _invalid("invalid UTF-8 or JSON header") from None
    return _validate_header(header, file_size - 8 - header_len)


def live_lora_keys(model: Any) -> set[str]:
    """LoRA parameter names of a loaded PeftModel, with the adapter-name segment removed
    (``lora_A.default.weight`` -> ``lora_A.weight``) so they compare against checkpoint keys."""
    names: set[str] = set()
    for name, _ in model.named_parameters():
        if "lora_" not in name:
            continue
        parts = name.split(".")
        # drop the adapter-name segment that PEFT inserts after lora_A / lora_B / lora_embedding_*
        cleaned = [
            p for i, p in enumerate(parts)
            if not (i > 0 and parts[i - 1].startswith("lora_") and p not in {"weight", "bias"})
        ]
        names.add(".".join(cleaned))
    return names


def coverage_report(ckpt_keys: Iterable[str], live_keys: Iterable[str]) -> dict[str, Any]:
    ckpt = sorted(set(ckpt_keys))
    live = set(live_keys)
    unapplied = [k for k in ckpt if k not in live]
    layout = "language_model" if any(".language_model." in k for k in ckpt) else "text"
    return {
        "checkpoint_tensors": len(ckpt),
        "applied": len(ckpt) - len(unapplied),
        "unapplied": len(unapplied),
        "unapplied_sample": unapplied[:3],
        "checkpoint_layout": layout,
        "fully_applied": not unapplied and bool(ckpt),
    }


def assert_adapter_applied(model: Any, adapter_dir: Path) -> dict[str, Any]:
    """Return the coverage report, or raise ``AdapterNotApplied`` (fail closed)."""
    report = coverage_report(checkpoint_keys(adapter_dir), live_lora_keys(model))
    if not report["fully_applied"]:
        base = getattr(getattr(model, "base_model", None), "model", model)
        raise AdapterNotApplied(
            f"ADAPTER_NOT_APPLIED: {report['unapplied']}/{report['checkpoint_tensors']} adapter tensors "
            f"have no target in {type(base).__name__} (checkpoint layout: {report['checkpoint_layout']}; "
            f"sample {report['unapplied_sample']}). Refusing to score the base model as an adapter."
        )
    return report


def describe_loader(model: Any, report: dict[str, Any] | None) -> dict[str, Any]:
    """Receipt field: which classes actually ran, with library versions and key coverage."""
    import transformers

    base = getattr(getattr(model, "base_model", None), "model", model)
    from importlib.metadata import PackageNotFoundError, version

    try:
        peft_version: str | None = version("peft")
    except PackageNotFoundError:  # base-only runs may not have peft installed
        peft_version = None
    return {
        "wrapper_class": type(model).__name__,
        "model_class": type(base).__name__,
        "model_type": getattr(getattr(base, "config", None), "model_type", None),
        "transformers": transformers.__version__,
        "peft": peft_version,
        "adapter_keys": report,
    }
