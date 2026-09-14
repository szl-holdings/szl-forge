#!/usr/bin/env python3
"""
Hardened LoRA merge — the fix for the 2026-09-13 empty-merge incident.

Policy (frontier/HF_FRONTIER_DELTA_2026-09-14.md):
  1. STRICT key matching: every adapter key must map to a base tensor. Unmapped keys hard-fail.
  2. Zero-delta assert: every target module must show a non-zero weight delta. Zero delta hard-fails.
  3. Merge receipt: modules applied, per-module delta checksums, method, written next to the output.

Usage:
  python merge_lora_hardened.py --base Qwen/Qwen3.5-0.8B --adapter ./chaski_r2_adapter \
      --out ./chaski_r2_merged --alpha 16 --rank 32

Exit codes: 0 = clean merge, 1 = hard failure (no output written).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Key-normalization rules learned from the incident: Unsloth on the Qwen3.5 mm build
# saves LoRA keys with a different module path and no adapter-name segment.
def normalize_key(key: str) -> str:
    k = key
    for marker in (".adapter.", ".default.", ".base_model.model."):
        if marker in k:
            k = k.replace(marker, ".")
    k = k.replace(".self_attn.", ".self_attn.")
    if k.startswith("base_model."):
        k = k[len("base_model."):]
    return k


def delta_checksum(base_t: torch.Tensor, merged_t: torch.Tensor) -> str:
    d = (merged_t.float() - base_t.float()).abs()
    return hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to base model safetensors dir")
    ap.add_argument("--adapter", required=True, help="Path to adapter dir (adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="Output dir for merged checkpoint")
    ap.add_argument("--alpha", type=float, required=True, help="LoRA alpha")
    ap.add_argument("--rank", type=int, required=True, help="LoRA rank r")
    ap.add_argument("--expect-modules", type=int, default=96, help="Expected target module count")
    args = ap.parse_args()

    base_files = sorted(Path(args.base).glob("*.safetensors"))
    adapter_file = Path(args.adapter) / "adapter_model.safetensors"
    if not base_files or not adapter_file.exists():
        print("FAIL: base safetensors or adapter_model.safetensors not found", file=sys.stderr)
        return 1

    base = {}
    for f in base_files:
        base.update(load_file(str(f)))
    adapter = load_file(str(adapter_file))

    lora_A = {normalize_key(k): v for k, v in adapter.items() if k.endswith(".lora_A.weight")}
    lora_B = {normalize_key(k): v for k, v in adapter.items() if k.endswith(".lora_B.weight")}

    targets = {}
    for k, a in lora_A.items():
        stem = k[: -len(".lora_A.weight")]
        b = lora_B.get(stem + ".lora_B.weight")
        if b is None:
            print(f"FAIL: unpaired lora_A without lora_B: {stem}", file=sys.stderr)
            return 1
        targets[stem] = (a, b)

    # STRICT MATCHING: every adapter target must exist in the base tensors.
    missing = [t for t in targets if t not in base]
    if missing:
        print(f"FAIL: {len(missing)} adapter keys matched no base tensor (silent no-op risk):", file=sys.stderr)
        for m in missing[:10]:
            print(f"  {m}", file=sys.stderr)
        return 1
    if len(targets) != args.expect_modules:
        print(f"FAIL: expected {args.expect_modules} target modules, found {len(targets)}", file=sys.stderr)
        return 1

    scale = args.alpha / args.rank
    merged = dict(base)
    receipt = {
        "method": "manual LoRA merge: W' = W + (alpha/r) * B @ A",
        "alpha": args.alpha,
        "rank": args.rank,
        "scale": scale,
        "modules_applied": 0,
        "zero_delta_failures": [],
        "deltas": {},
    }

    for stem, (a, b) in targets.items():
        w = base[stem]
        delta = (scale * (b @ a)).to(w.dtype)
        if delta.abs().max().item() == 0.0:
            receipt["zero_delta_failures"].append(stem)
            continue
        merged[stem] = w + delta
        receipt["modules_applied"] += 1
        receipt["deltas"][stem] = delta_checksum(w, merged[stem])

    if receipt["zero_delta_failures"]:
        print(f"FAIL: {len(receipt['zero_delta_failures'])} modules had zero delta — merge aborted", file=sys.stderr)
        for m in receipt["zero_delta_failures"][:10]:
            print(f"  {m}", file=sys.stderr)
        return 1
    if receipt["modules_applied"] != len(targets):
        print(f"FAIL: applied {receipt['modules_applied']}/{len(targets)} modules", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    receipt_path = out_dir / "merge_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"OK: {receipt['modules_applied']}/{len(targets)} modules merged, zero-delta assert passed")
    print(f"receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
