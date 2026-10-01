"""Qualify the salvaged v2 merge: merged model vs adapter-on-base, greedy equivalence."""
import json, sys
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from salvage_v2_adapter import load_multimodal_base, rename_keys, REPO

MERGED = Path("merged/szl-receiptagent-qwen35-0.8b-v2-salvaged")
EVAL = Path(r"C:\Users\steph\szl-forge\receiptagent\eval.jsonl")
N = 8

def load_adapter_stack():
    cfg = json.loads(Path(hf_hub_download(REPO, "adapter_config.json")).read_text())
    state = load_file(hf_hub_download(REPO, "adapter_model.safetensors"))
    base = load_multimodal_base()
    lcfg = LoraConfig(r=cfg.get("r",16), lora_alpha=cfg.get("lora_alpha",32),
                      lora_dropout=cfg.get("lora_dropout",0.0), bias=cfg.get("bias","none"),
                      target_modules=cfg.get("target_modules"), task_type=cfg.get("task_type") or "CAUSAL_LM")
    pm = get_peft_model(base, lcfg)
    pm.load_state_dict(rename_keys(state), strict=False)
    return pm

prompts, users = [], []
for line in EVAL.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line: continue
    try: obj = json.loads(line)
    except json.JSONDecodeError: continue
    msgs = [m for m in obj.get("messages", []) if m.get("role") in ("system", "user")]
    if len(msgs) < 2: continue
    users.append(next((m["content"] for m in msgs if m["role"] == "user"), "?"))
    prompts.append(msgs)
    if len(prompts) >= N: break
if len(prompts) < 4:
    raise SystemExit(f"only {len(prompts)} chat-format prompts from {EVAL} -- inspect")

tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-0.8B")
from transformers import AutoModelForImageTextToText as VL
merged = VL.from_pretrained(MERGED, dtype=torch.float32); merged.eval()
adapter_stack = load_adapter_stack(); adapter_stack.eval()

@torch.inference_mode()
def gen(model, msgs):
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

matches, results = 0, []
for i, (msgs, u) in enumerate(zip(prompts, users)):
    a = gen(merged, msgs); b = gen(adapter_stack, msgs)
    m = a.strip() == b.strip()
    matches += int(m)
    results.append({"i": i, "user": u[:80], "match": m, "merged": a[:120], "adapter": b[:120]})
    print(f"[{i+1}/{len(prompts)}] {'MATCH' if m else 'DIVERGE'} :: {u[:60]}")

rate = matches / len(prompts)
print(f"EQUIVALENCE: {matches}/{len(prompts)} ({rate:.0%})")
receipt = {"label": "MEASURED", "test": "merged-vs-adapter greedy equivalence (chat-template prompts)",
           "prompts": len(prompts), "exact_matches": matches, "rate": rate,
           "eval_source": str(EVAL), "merged_model": str(MERGED),
           "publication_eligible": rate >= 0.75, "hub_write": False,
           "note": "adapter behavior was previously qualified; this tests merge fidelity"}
Path("qual_v2_salvage_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("receipt -> qual_v2_salvage_receipt.json")
sys.exit(0 if rate >= 0.75 else 1)