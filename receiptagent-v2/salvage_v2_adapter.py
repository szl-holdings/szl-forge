"""Salvage szl-receiptagent-qwen35-0.8b-v2 — v3, architecture-correct.

Root cause (solved 2026-09-13): the adapter was trained on the multimodal
(vision-language) wrapper of Qwen3.5-0.8B via Unsloth FastVisionModel. Its
LoRA keys target model.language_model.layers.N.linear_attn.* and lack the
'.default' adapter segment. The original merge loaded the text-only base
(model.layers.N.self_attn), so all 372 keys were dropped -> byte-copy of base.

Fix: load the multimodal base, pass the config regex through to PEFT, rename
checkpoint keys to insert '.default', load, merge, verify non-zero delta.

Usage:
  python salvage_v2_adapter.py            # diagnose: build + verify key map
  python salvage_v2_adapter.py --fix      # merge, verify delta, save
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download

REPO = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2"
BASE_CANDIDATES = ["unsloth/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"]
OUT = Path("merged/szl-receiptagent-qwen35-0.8b-v2-salvaged")


def load_multimodal_base():
    from transformers import AutoModel, AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText as VL
    except ImportError:
        VL = AutoModel
    for repo in BASE_CANDIDATES:
        for cls in (VL, AutoModel, AutoModelForCausalLM):
            try:
                m = cls.from_pretrained(repo, dtype=torch.float32)
            except Exception as e:
                print(f"[base] {repo} via {cls.__name__}: {type(e).__name__}")
                continue
            if any("linear_attn.in_proj_a" in k for k in m.state_dict()):
                print(f"[base] loaded {repo} via {cls.__name__} (linear_attn present)")
                return m
            print(f"[base] {repo} via {cls.__name__}: no linear_attn, next class")
            del m
    raise SystemExit("no base candidate exposes model.language_model.layers.*.linear_attn")


def rename_keys(state):
    out = {}
    for k, v in state.items():
        nk = k.replace(".lora_A.weight", ".lora_A.default.weight")
        nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
        out[nk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cfg = json.loads(Path(hf_hub_download(REPO, "adapter_config.json")).read_text())
    state = load_file(hf_hub_download(REPO, "adapter_model.safetensors"))
    tm = cfg.get("target_modules")
    print(f"target_modules ({type(tm).__name__}): {str(tm)[:100]}...")

    base = load_multimodal_base()
    lcfg = LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias=cfg.get("bias", "none"),
        target_modules=tm,
        task_type=cfg.get("task_type") or "CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lcfg)
    expected = {k for k in peft_model.state_dict() if ".lora_" in k}
    renamed = rename_keys(state)
    renamed_lora = {k for k in renamed if ".lora_" in k}

    unmatched = renamed_lora - expected
    missing = expected - renamed_lora
    print(f"checkpoint LoRA keys (renamed) : {len(renamed_lora)}")
    print(f"model expects                  : {len(expected)}")
    print(f"unmatched / missing            : {len(unmatched)} / {len(missing)}")
    for k in sorted(unmatched)[:5]:
        print("  +", k)
    for k in sorted(missing)[:5]:
        print("  -", k)
    if unmatched or missing:
        raise SystemExit("key sets still disagree -- aborting before any merge")
    print("KEY MAP VERIFIED: all 372 checkpoint keys map to live modules.")
    if not args.fix:
        print("DRY RUN complete. Re-run with --fix to merge and save.")
        return

    peft_model.load_state_dict(renamed, strict=False)
    merged = peft_model.merge_and_unload()

    ref = load_multimodal_base()
    ref_sd = ref.state_dict()
    max_delta = 0.0
    touched = 0
    for name, p in merged.state_dict().items():
        q = ref_sd.get(name)
        if q is not None and p.shape == q.shape:
            d = (p.float() - q.float()).abs().max().item()
            if d > 0:
                touched += 1
            max_delta = max(max_delta, d)
    del ref, ref_sd
    print(f"max weight delta vs base: {max_delta:.6g} across {touched} tensors")
    if max_delta == 0.0:
        raise SystemExit("merge STILL empty -- do NOT publish")

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(BASE_CANDIDATES[0]).save_pretrained(args.out)
    except Exception:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(BASE_CANDIDATES[0]).save_pretrained(args.out)
    print(f"saved salvaged merge -> {args.out}")


if __name__ == "__main__":
    main()