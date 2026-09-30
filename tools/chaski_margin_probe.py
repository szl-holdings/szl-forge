#!/usr/bin/env python3
"""chaski_margin_probe.py — DIAGNOSTIC receipt upgrade for the chaski named-N bake-off.

Why this exists (thread audit, Part A.3): the committed receipts show two adapters flipping
between 0/5 and 5/5 on the same held-out gate across two torch/CUDA builds while the r2 control
stayed at 5/5. A greedy 0.8B model can sit on a knife edge at a single token (`}` closes the
three-key object, `,` continues to seven keys). The canonical runner records the output and the
verdict but not *how close* each decision was, nor the environment that produced it. This probe
records both, per case, so an environment-sensitive verdict is visible instead of inferred.

What it does
  * imports the PRISTINE canonical runner as a module (sha256 verified against --runner-sha256)
    and reuses ITS prompt construction (`prompt_messages`, `apply_chat_template(...,
    enable_thinking=False)`, `add_special_tokens=False`, greedy, `use_cache=True`) and ITS scorers
    (`score_draft`, `score_refusal`) — no re-implementation of the contract;
  * re-generates every held-out case with `output_scores=True` and records the top-2 logit
    margin at every generated token, the minimum margin and where it occurred, and the margins at
    structural JSON decision points (`}` vs `,`);
  * records the environment the canonical receipt omits: torch / transformers / peft / accelerate
    versions, model class, config.architectures, dtype, device, chat-template sha256, rendered
    prompt sha256 + token count, eos/pad ids, whether generation ended on EOS;
  * writes ONE diagnostic receipt under chaski_r4/evidence/ (never the canonical receipt path).

What it is not
  * not the gate. `label` is DIAGNOSTIC, `is_canonical_receipt` is false, counts here do not
    qualify anything, and `publication_eligible` / `autonomy_eligible` / `hub_put` stay false.
  * no training, adapter modification, merge, upload, publication, deployment, or Git write.

Usage (owner metal):
  python tools/chaski_margin_probe.py --root C:\\Users\\steph\\szl-forge \
      --adapter chaski_r4/chaski-r4-adapter --adapter chaski_r2/chaski-r2-adapter
  (CPU smoke test with a tiny model: --base <tiny-model> --device cpu --allow-missing-template)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNNER_PRISTINE_SHA256 = "f4ca282a29e2fa2f0246ebcb6e3298b7b3f023d686ba883d948459bcaf56dadb"
SCHEMA = "szl.chaski-margin-probe/v1"
KIND = "szl-chaski-margin-probe"
CHATML_FALLBACK = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_runner(path: Path, expected_sha: str | None):
    actual = sha256_file(path)
    if expected_sha and actual != expected_sha:
        raise SystemExit(f"FAIL-CLOSED: runner sha256 {actual} != expected {expected_sha}; refusing to reuse a non-pristine contract")
    argv = list(sys.argv)
    sys.argv = [str(path)]
    try:
        spec = importlib.util.spec_from_file_location("chaski_canonical_runner", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    for name in ("prompt_messages", "score_draft", "score_refusal", "load_named_n_gate", "DRAFT_KIND", "REFUSAL_KIND", "CANONICAL_BASE"):
        if not hasattr(mod, name):
            raise SystemExit(f"FAIL-CLOSED: runner lacks {name}; contract drifted")
    return mod, actual


def render_prompt(tok, messages):
    """Byte-for-byte the canonical runner's prompt path (enable_thinking=False, TypeError fallback)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False), "enable_thinking=False"
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), "default"


def generate_with_margins(model, tok, prompt: str, *, max_new_tokens: int, device: str) -> dict[str, Any]:
    import torch

    enc = tok(text=prompt, add_special_tokens=False, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_tokens = int(enc["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                             output_scores=True, return_dict_in_generate=True)
    seconds = time.perf_counter() - started
    new_ids = out.sequences[0, prompt_tokens:]
    steps = []
    for t, score in enumerate(out.scores):
        row = score[0].float()
        top = torch.topk(row, 2)
        chosen = int(new_ids[t])
        best, second = int(top.indices[0]), int(top.indices[1])
        margin = float(top.values[0] - top.values[1])
        steps.append({
            "t": t, "token_id": chosen, "token": tok.decode([chosen]),
            "runner_up_id": second if best == chosen else best,
            "runner_up": tok.decode([second if best == chosen else best]),
            "margin": round(margin, 4),
        })
    eos_ids = model.generation_config.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]) if eos_ids is not None else set()
    ended_on_eos = bool(len(new_ids) and int(new_ids[-1]) in eos_set)
    text = tok.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    def _kind(tok_text: str) -> str | None:
        x = tok_text.strip()
        if x.startswith("}"):
            return "close"
        if x.startswith(","):
            return "continue"
        return None
    struct = [s for s in steps if {_kind(s["token"]), _kind(s["runner_up"])} == {"close", "continue"}]
    argmin = min(steps, key=lambda s: s["margin"]) if steps else None
    return {
        "output": text, "output_sha256": sha256_text(text), "new_tokens": int(len(new_ids)), "seconds": round(seconds, 6),
        "prompt_tokens": prompt_tokens, "ended_on_eos": ended_on_eos,
        "margin_min": argmin["margin"] if argmin else None, "margin_argmin": argmin,
        "margin_mean": round(sum(s["margin"] for s in steps) / len(steps), 4) if steps else None,
        "structural_decisions": struct[:8], "margins": [s["margin"] for s in steps],
    }


def probe_candidate(mod, *, base: str, adapter: Path | None, drafts, refusals, device: str, dtype_name: str,
                    draft_max_new_tokens: int, refusal_max_new_tokens: int, allow_missing_template: bool, threshold: float,
                    model_class_name: str = "AutoModelForCausalLM") -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoTokenizer
    AutoModelForCausalLM = getattr(transformers, model_class_name)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(base, local_files_only=("/" in base and not Path(base).exists() and device != "cpu"))
    if tok.chat_template is None:
        if not allow_missing_template:
            raise SystemExit("FAIL-CLOSED: tokenizer has no chat template; the canonical prompt path cannot be reproduced")
        tok.chat_template = CHATML_FALLBACK
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device, local_files_only=tok.init_kwargs.get("local_files_only", False))
    model_class = type(model).__name__
    adapter_keys = None
    if adapter is not None:
        from peft import PeftModel
        from safetensors import safe_open
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        # FAIL-CLOSED: PEFT only *warns* when checkpoint keys do not exist in the model (e.g. an
        # adapter trained on the multimodal layout `model.language_model.layers.*` loaded into the
        # text-only class whose modules are `model.layers.*`). A silently unapplied adapter makes
        # the "adapter" candidate the bare base model. Verify every checkpoint tensor landed.
        with safe_open(str(Path(adapter) / "adapter_model.safetensors"), "pt") as f:
            ckpt_keys = list(f.keys())
        live = {n.replace(".default.", ".") for n, _ in model.named_parameters() if "lora_" in n}
        unapplied = [k for k in ckpt_keys if k not in live]
        adapter_keys = {"checkpoint_tensors": len(ckpt_keys), "applied": len(ckpt_keys) - len(unapplied),
                        "unapplied": len(unapplied), "unapplied_sample": unapplied[:3],
                        "checkpoint_layout": "language_model" if any(".language_model." in k for k in ckpt_keys) else "text"}
        if unapplied:
            raise SystemExit(f"FAIL-CLOSED: ADAPTER_NOT_APPLIED — {len(unapplied)}/{len(ckpt_keys)} adapter tensors have no target in "
                             f"{model_class} (checkpoint layout: {adapter_keys['checkpoint_layout']}; sample {unapplied[:2]}). "
                             f"Load the class the adapter was trained against (e.g. AutoModelForImageTextToText for "
                             f"`model.language_model.layers.*` keys) or re-key the adapter. Refusing to score the base model as an adapter.")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    env = {
        "model_class": model_class, "wrapped_class": type(model).__name__,
        "config_architectures": list(getattr(model.config, "architectures", None) or []),
        "model_type": getattr(model.config, "model_type", None), "dtype": dtype_name, "device": device,
        "chat_template_sha256": sha256_text(tok.chat_template or ""), "tokenizer_class": type(tok).__name__,
        "eos_token_id": model.generation_config.eos_token_id, "pad_token_id": tok.pad_token_id,
        "active_adapter": getattr(model, "active_adapter", None) if adapter is not None else None,
        "adapter_keys": adapter_keys,
    }
    cases, draft_valid, refused, knife = [], 0, 0, 0
    for kind, rows, budget, scorer in (("draft", drafts["rows"], draft_max_new_tokens, mod.score_draft),
                                       ("adversarial", refusals["rows"], refusal_max_new_tokens, mod.score_refusal)):
        for index, item in enumerate(rows, 1):
            messages = mod.prompt_messages(item)
            prompt, mode = render_prompt(tok, messages)
            g = generate_with_margins(model, tok, prompt, max_new_tokens=budget, device=device)
            ok, error = scorer(g["output"])
            if kind == "draft":
                draft_valid += int(ok)
            else:
                refused += int(ok)
            # knife_edge = a STRUCTURAL decision (`}` vs `,`: close the object or add a key) below the
            # threshold; free-text word choices inside `claim` are naturally close and are reported
            # separately as margin_min / margin_argmin.
            struct_min = min((s_["margin"] for s_ in g["structural_decisions"]), default=None)
            g["structural_margin_min"] = struct_min
            flag = struct_min is not None and struct_min < threshold
            knife += int(flag)
            cases.append({"kind": kind, "id": item.get("id"), "index": index, "prompt_mode": mode,
                          "prompt_sha256": sha256_text(prompt), **g, "contract_valid" if kind == "draft" else "refused": ok,
                          "error": error, "knife_edge": flag})
            print(f"  {kind:11} {item.get('id')}: ok={ok} tokens={g['new_tokens']} eos={g['ended_on_eos']} "
                  f"min_margin={g['margin_min']} at t={g['margin_argmin']['t'] if g['margin_argmin'] else None} "
                  f"structural_min={struct_min} knife_edge={flag}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"adapter": str(adapter) if adapter else None, "adapter_sha256": mod.sha256_adapter(adapter) if adapter else None,
            "environment": env, "json_draft_valid": draft_valid, "json_draft_total": drafts["n"],
            "adversarial_refused": refused, "adversarial_total": refusals["n"], "knife_edge_cases": knife, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser(description="chaski named-N margin probe (diagnostic, never the gate)")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--runner", type=Path, default=None, help="default ROOT/chaski_r4/bakeoff_canonical_four_way_r4.py")
    ap.add_argument("--runner-sha256", default=RUNNER_PRISTINE_SHA256, help="'' to skip the pristine check (tests only)")
    ap.add_argument("--gate-dir", type=Path, default=None, help="default ROOT/chaski/gate")
    ap.add_argument("--base", default=None, help="default: runner CANONICAL_BASE")
    ap.add_argument("--adapter", type=Path, action="append", default=[], help="adapter dir (repeatable); none = base only")
    ap.add_argument("--include-base", action="store_true", help="also probe the bare base model when adapters are given")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-class", default="AutoModelForCausalLM", choices=["AutoModelForCausalLM", "AutoModelForImageTextToText"],
                    help="loader class; the canonical runner uses AutoModelForCausalLM, the archived 2026-09-16 runner used AutoModelForImageTextToText")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--draft-max-new-tokens", type=int, default=256)
    ap.add_argument("--refusal-max-new-tokens", type=int, default=96)
    ap.add_argument("--margin-threshold", type=float, default=1.0, help="logit gap below which a token decision is flagged knife-edge")
    ap.add_argument("--allow-missing-template", action="store_true", help="tests only: inject a ChatML template if the tokenizer has none")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    root = a.root.resolve()
    runner = a.runner or (root / "chaski_r4" / "bakeoff_canonical_four_way_r4.py")
    gate_dir = a.gate_dir or (root / "chaski" / "gate")
    mod, runner_sha = load_runner(runner, a.runner_sha256 or None)
    drafts = mod.load_named_n_gate(gate_dir / "json_drafts.n5.jsonl", mod.DRAFT_KIND)
    refusals = mod.load_named_n_gate(gate_dir / "adversarial_refusals.n6.jsonl", mod.REFUSAL_KIND)
    base = a.base or mod.CANONICAL_BASE

    import torch
    import transformers
    try:
        import peft
        peft_v = peft.__version__
    except Exception:
        peft_v = None
    try:
        import accelerate
        acc_v = accelerate.__version__
    except Exception:
        acc_v = None
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("FAIL-CLOSED: --device cuda but CUDA unavailable")

    candidates = []
    targets = list(a.adapter) if a.adapter else [None]
    if a.adapter and a.include_base:
        targets = [None] + targets
    for adapter in targets:
        label = "base" if adapter is None else adapter.name
        print(f"[margin-probe] {label}")
        candidates.append({"id": label, **probe_candidate(
            mod, base=base, adapter=adapter, drafts=drafts, refusals=refusals, device=a.device, dtype_name=a.dtype,
            draft_max_new_tokens=a.draft_max_new_tokens, refusal_max_new_tokens=a.refusal_max_new_tokens,
            allow_missing_template=a.allow_missing_template, threshold=a.margin_threshold, model_class_name=a.model_class)})

    payload = {
        "kind": KIND, "schema": SCHEMA, "label": "DIAGNOSTIC", "is_canonical_receipt": False,
        "publication_eligible": False, "autonomy_eligible": False, "hub_put": False,
        "base_model": base, "loader_class": a.model_class, "runner": str(runner), "runner_sha256": runner_sha, "runner_pristine": runner_sha == RUNNER_PRISTINE_SHA256,
        "gate_hashes": {"json_drafts.n5.jsonl": sha256_file(gate_dir / "json_drafts.n5.jsonl"),
                        "adversarial_refusals.n6.jsonl": sha256_file(gate_dir / "adversarial_refusals.n6.jsonl")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "peft": peft_v, "accelerate": acc_v,
                        "cuda": torch.cuda.is_available(), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "margin_threshold": a.margin_threshold, "candidates": candidates,
        "claim_boundary": ("Diagnostic replay of the held-out named-N prompts through the pristine runner's own prompt path and "
                           "scorers, with per-token top-2 logit margins. Not the canonical gate; counts here qualify nothing. "
                           "knife_edge marks decisions whose logit gap is below the threshold — the candidates for "
                           "environment-sensitive verdicts across torch/CUDA builds."),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["report_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out = a.out or (root / "chaski_r4" / "evidence" / f"margin_probe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for c in candidates:
        print(f"[margin-probe] {c['id']:14} drafts {c['json_draft_valid']}/{c['json_draft_total']}  refusals {c['adversarial_refused']}/{c['adversarial_total']}"
              f"  knife-edge cases {c['knife_edge_cases']}  class={c['environment']['model_class']}")
    print(f"RECEIPT={out}\nRECEIPT_SHA256={sha256_file(out)}")
    print("DIAGNOSTIC ONLY. No training, adapter modification, merge, upload, publication, deployment, or Git write occurred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
