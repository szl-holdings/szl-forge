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
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ADAPTER_WEIGHTS = "adapter_model.safetensors"


class AdapterNotApplied(RuntimeError):
    """Raised when at least one checkpoint tensor has no target in the loaded model."""


def checkpoint_keys(adapter_dir: Path) -> list[str]:
    """Tensor names stored in ``adapter_model.safetensors`` (header parse only, no torch)."""
    path = Path(adapter_dir) / ADAPTER_WEIGHTS
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return sorted(k for k in header if k != "__metadata__")


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
