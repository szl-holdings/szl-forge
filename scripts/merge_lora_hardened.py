#!/usr/bin/env python3
"""Fail-closed LoRA merge for the 2026-09-13 empty-merge incident.

The critical invariant is source-to-target binding: an adapter module such as
``model.layers.0.self_attn.q_proj.lora_A.weight`` updates the base tensor
``model.layers.0.self_attn.q_proj.weight``.  A missing, ambiguous, duplicated,
shape-incompatible, or zero-delta target aborts before any output is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_LORA_SUFFIXES = (
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
)
_LEADING_WRAPPERS = ("base_model.model.", "base_model.")


def normalize_module_stem(key: str) -> tuple[str, str] | None:
    """Return (base-module stem, side) for supported PEFT/Unsloth LoRA keys."""
    suffix = next(((s, side) for s, side in _LORA_SUFFIXES if key.endswith(s)), None)
    if suffix is None:
        return None
    suffix_text, side = suffix
    stem = key[: -len(suffix_text)]
    for prefix in _LEADING_WRAPPERS:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    if not stem or stem.endswith("."):
        raise ValueError(f"invalid normalized LoRA stem from {key!r}")
    return stem, side


def base_weight_key(stem: str, base_keys: set[str]) -> str:
    """Resolve exactly one base weight tensor for a normalized LoRA module stem."""
    candidates = [f"{stem}.weight"]
    # Some checkpoints retain an outer model. wrapper after PEFT removes its own.
    if stem.startswith("model."):
        candidates.append(f"{stem[len('model.'):]}.weight")
    else:
        candidates.append(f"model.{stem}.weight")
    matches = [candidate for candidate in dict.fromkeys(candidates) if candidate in base_keys]
    if len(matches) != 1:
        raise KeyError(
            f"LoRA module {stem!r} must resolve to exactly one base weight tensor; "
            f"matches={matches!r} candidates={candidates!r}"
        )
    return matches[0]


def delta_checksum(delta: torch.Tensor) -> str:
    return hashlib.sha256(delta.float().cpu().numpy().tobytes()).hexdigest()


def collect_targets(adapter: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    sides: dict[str, dict[str, torch.Tensor]] = {}
    source_keys: dict[tuple[str, str], str] = {}
    for key, tensor in adapter.items():
        normalized = normalize_module_stem(key)
        if normalized is None:
            continue
        stem, side = normalized
        slot = (stem, side)
        if slot in source_keys:
            raise ValueError(
                f"duplicate normalized LoRA {side} target {stem!r}: "
                f"{source_keys[slot]!r} and {key!r}"
            )
        source_keys[slot] = key
        sides.setdefault(stem, {})[side] = tensor

    if not sides:
        raise ValueError("adapter contains no supported LoRA A/B tensors")

    targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    base_keys = set(base)
    for stem, pair in sides.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA target {stem!r} is unpaired; sides={sorted(pair)}")
        key = base_weight_key(stem, base_keys)
        a, b = pair["A"], pair["B"]
        weight = base[key]
        if a.ndim != 2 or b.ndim != 2 or weight.ndim != 2:
            raise ValueError(f"LoRA/base tensors for {key!r} must all be rank-2")
        if b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != tuple(weight.shape):
            raise ValueError(
                f"shape mismatch for {key!r}: A={tuple(a.shape)} B={tuple(b.shape)} "
                f"weight={tuple(weight.shape)}"
            )
        if key in targets:
            raise ValueError(f"multiple LoRA modules resolved to base tensor {key!r}")
        targets[key] = (a, b)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Local base-model safetensors directory")
    parser.add_argument("--adapter", required=True, help="Local adapter directory")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--expect-modules", type=int, default=96)
    args = parser.parse_args()

    if args.rank <= 0 or args.alpha <= 0 or args.expect_modules <= 0:
        print("FAIL: alpha, rank, and expect-modules must be positive", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    if out_dir.exists():
        print("FAIL: output directory already exists; refusing partial/overwrite semantics", file=sys.stderr)
        return 1

    base_files = sorted(Path(args.base).glob("*.safetensors"))
    adapter_file = Path(args.adapter) / "adapter_model.safetensors"
    if not base_files or not adapter_file.is_file():
        print("FAIL: base safetensors or adapter_model.safetensors not found", file=sys.stderr)
        return 1

    try:
        base: dict[str, torch.Tensor] = {}
        for file in base_files:
            for key, tensor in load_file(str(file)).items():
                if key in base:
                    raise ValueError(f"duplicate base tensor across shards: {key}")
                base[key] = tensor
        adapter = dict(load_file(str(adapter_file)))
        targets = collect_targets(adapter, base)
    except (KeyError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if len(targets) != args.expect_modules:
        print(
            f"FAIL: expected {args.expect_modules} target modules, resolved {len(targets)}",
            file=sys.stderr,
        )
        return 1

    scale = args.alpha / args.rank
    merged = dict(base)
    receipt: dict[str, object] = {
        "schema": "szl.forge.lora-merge-receipt.v1",
        "method": "W' = W + (alpha/r) * B @ A",
        "alpha": args.alpha,
        "rank": args.rank,
        "scale": scale,
        "expectedModules": args.expect_modules,
        "modulesApplied": 0,
        "baseFiles": [file.name for file in base_files],
        "adapterFile": adapter_file.name,
        "deltas": {},
    }

    deltas: dict[str, str] = {}
    for key, (a, b) in sorted(targets.items()):
        weight = base[key]
        delta = (scale * (b.float() @ a.float())).to(weight.dtype)
        if not torch.isfinite(delta).all().item():
            print(f"FAIL: non-finite delta for {key}", file=sys.stderr)
            return 1
        if torch.count_nonzero(delta).item() == 0:
            print(f"FAIL: zero delta for {key}", file=sys.stderr)
            return 1
        merged_weight = weight + delta
        if torch.equal(merged_weight, weight):
            print(f"FAIL: dtype rounding produced unchanged target {key}", file=sys.stderr)
            return 1
        merged[key] = merged_weight
        deltas[key] = delta_checksum(delta)

    receipt["modulesApplied"] = len(targets)
    receipt["deltas"] = deltas

    out_dir.mkdir(parents=False)
    save_file(merged, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    (out_dir / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"OK: {len(targets)}/{len(targets)} modules merged with non-zero deltas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
