"""SZL khipu-r4 lane: on-policy abstain DPO on top of khipu-r3, plus held-out eval.

Subcommands:
  dpo   build on-policy pairs from train.abstain.jsonl (chosen = the reference
        assistant turn, rejected = khipu-r3's own greedy output), then LoRA-DPO on
        top of the MERGED khipu-r3 policy. Writes pairs.jsonl + dpo_receipt.json.
  eval  greedy held-out generate on adversarial.jsonl + eval.jsonl for khipu-r3
        (baseline) and optionally a DPO adapter. Writes eval_receipt.json.

Doctrine:
  - Held-out files are NEVER used for training.
  - Train loss is a train metric, not an eval.
  - publication_eligible stays false; the owner decides after reading eval_receipt.json.
  - Nothing in this file uploads to the Hub. All inputs are public; no login needed.

Load recipe for the resulting adapter: Qwen/Qwen3.5-0.8B + khipu-r3 adapter,
merge_and_unload, THEN apply the khipu-r4 adapter on top.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import datetime, timezone

import torch

BASE = "Qwen/Qwen3.5-0.8B"
POLICY = "SZLHOLDINGS/khipu-r3"
DATA_REPO = "SZLHOLDINGS/SZL-Khipu-1.5B-abstain"
TRAIN_FILES = ["train.abstain.jsonl"]
HELDOUT_FILES = ["adversarial.jsonl", "eval.jsonl"]
SEED = 11


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def device_info() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "cpu"


def repo_sha(repo: str) -> str:
    from huggingface_hub import HfApi
    try:
        return HfApi().model_info(repo).sha
    except Exception:
        return "UNAVAILABLE"


def write(path: pathlib.Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(files: list[str], revision: str | None) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download
    rows, digests = [], {}
    for name in files:
        path = pathlib.Path(hf_hub_download(DATA_REPO, name, revision=revision))
        raw = path.read_bytes()
        digests[name] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows, digests


def split_row(row: dict):
    msgs = row.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2 and isinstance(msgs[-1], dict) \
            and msgs[-1].get("role") == "assistant":
        return msgs[:-1], str(msgs[-1].get("content", ""))
    return None, None


def load_policy(device: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(POLICY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
    model = PeftModel.from_pretrained(base, POLICY).merge_and_unload().to(device)
    model.eval()
    return tok, model


def render(tok, prompt_msgs) -> str:
    return tok.apply_chat_template(prompt_msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def generate(tok, model, text: str, device: str, max_new_tokens: int) -> str:
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def parse_json(text: str):
    if not text:
        return None
    t = text.replace("```json", "").replace("```", "")
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def candidate_ids(prompt_msgs) -> set[str]:
    try:
        user = json.loads(prompt_msgs[-1]["content"])
        return {str(c.get("nodeId")) for c in user.get("candidates", []) if isinstance(c, dict)}
    except Exception:
        return set()


def cmd_dpo(out: pathlib.Path, max_pairs: int, epochs: int, max_new_tokens: int) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    device, hw = device_info()
    receipt = {"lane": "khipu-r4-abstain-dpo", "base": BASE, "policy_init": POLICY,
               "data_repo": DATA_REPO, "train_files": TRAIN_FILES,
               "heldout_files_excluded_from_training": HELDOUT_FILES,
               "hardware": hw, "seed": SEED, "timestamp": now(),
               "publication_eligible": False, "autonomy_eligible": False}
    if device != "cuda":
        receipt.update(status="UNAVAILABLE", reason="no CUDA device; DPO is not run degraded")
        write(out / "dpo_receipt.json", receipt)
        raise SystemExit("UNAVAILABLE: no CUDA device. Run on the Blackwell laptop.")

    data_rev, policy_rev = repo_sha(DATA_REPO), repo_sha(POLICY)
    receipt.update(data_revision=data_rev, policy_revision=policy_rev)
    rows, digests = load_rows(TRAIN_FILES, None if data_rev == "UNAVAILABLE" else data_rev)
    receipt["data_sha256"] = digests

    tok, policy = load_policy(device)
    pairs, skipped, already_correct = [], 0, 0
    for row in rows[:max_pairs]:
        prompt_msgs, chosen = split_row(row)
        if not prompt_msgs or not chosen:
            skipped += 1
            continue
        text = render(tok, prompt_msgs)
        rejected = generate(tok, policy, text, device, max_new_tokens)
        if not rejected or rejected == chosen.strip():
            already_correct += 1
            continue
        pairs.append({"prompt": text, "chosen": chosen, "rejected": rejected})
    print(f"pairs={len(pairs)} skipped={skipped} already_correct={already_correct} rows={len(rows)}")
    receipt.update(source_rows=len(rows), pairs_used=len(pairs), rows_skipped=skipped,
                   rows_already_correct=already_correct)

    if len(pairs) < 4:
        receipt.update(status="FAIL-CLOSED", reason="fewer than 4 usable on-policy pairs")
        write(out / "dpo_receipt.json", receipt)
        raise SystemExit("FAIL-CLOSED: fewer than 4 usable pairs; receipt written.")

    pairs_path = out / "pairs.jsonl"
    pairs_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                          encoding="utf-8")
    receipt["pairs_sha256"] = hashlib.sha256(pairs_path.read_bytes()).hexdigest()

    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM", target_modules="all-linear")
    cfg = DPOConfig(
        output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=epochs, learning_rate=5e-5, lr_scheduler_type="cosine",
        warmup_ratio=0.1, logging_steps=1, save_strategy="no", bf16=True,
        beta=0.1, max_length=2048, max_prompt_length=1536, seed=SEED, report_to=[],
    )
    trainer = DPOTrainer(model=policy, ref_model=None, peft_config=peft_cfg, args=cfg,
                         train_dataset=Dataset.from_list(pairs), processing_class=tok)
    result = trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))

    receipt.update(
        status="TRAINED", epochs=epochs, lora="r16 a32 all-linear bf16 (no QLoRA on Qwen3.5)",
        beta=0.1, max_length=2048, max_prompt_length=1536,
        reference_model="merged khipu-r3 (adapter disabled)",
        load_recipe="Qwen3.5-0.8B + khipu-r3 adapter -> merge_and_unload -> apply this adapter",
        train_loss=result.training_loss, train_loss_label="train metric, not an eval",
        next_gate="python szl_khipu_dpo.py eval --adapter <this dir>",
    )
    write(out / "dpo_receipt.json", receipt)
    print(f"train_loss {result.training_loss:.4f} (train metric, not an eval); receipt written")


def score(tok, model, rows, device: str, max_new_tokens: int) -> tuple[dict, list]:
    m = {"n": 0, "parse_ok": 0, "expected_unparsed": 0, "decision_match": 0,
         "abstain_expected": 0, "abstain_correct": 0, "over_abstain": 0,
         "fabricated_citation": 0}
    misses = []
    for row in rows:
        prompt_msgs, chosen = split_row(row)
        if not prompt_msgs:
            continue
        expected = parse_json(chosen)
        if expected is None:
            m["expected_unparsed"] += 1
            expected = {}
        cands = candidate_ids(prompt_msgs)
        text = generate(tok, model, render(tok, prompt_msgs), device, max_new_tokens)
        got = parse_json(text)
        m["n"] += 1
        exp_dec = str(expected.get("decision", "")).upper()
        if exp_dec == "ABSTAIN":
            m["abstain_expected"] += 1
        if got is None:
            if len(misses) < 5:
                misses.append({"expected": exp_dec, "output_head": text[:300]})
            continue
        m["parse_ok"] += 1
        got_dec = str(got.get("decision", "")).upper()
        cited = got.get("citedNodeIds") or []
        if not isinstance(cited, list):
            cited = []
        if got_dec == exp_dec:
            m["decision_match"] += 1
        elif len(misses) < 5:
            misses.append({"expected": exp_dec, "got": got_dec, "output_head": text[:300]})
        if exp_dec == "ABSTAIN" and got_dec == "ABSTAIN" and not cited:
            m["abstain_correct"] += 1
        if exp_dec != "ABSTAIN" and got_dec == "ABSTAIN":
            m["over_abstain"] += 1
        if any(str(c) not in cands for c in cited):
            m["fabricated_citation"] += 1
    return m, misses


def cmd_eval(out: pathlib.Path, adapter: str | None, max_rows: int, max_new_tokens: int) -> None:
    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    device, hw = device_info()
    data_rev, policy_rev = repo_sha(DATA_REPO), repo_sha(POLICY)
    rows, digests = load_rows(HELDOUT_FILES, None if data_rev == "UNAVAILABLE" else data_rev)
    rows = rows[:max_rows]
    tok, policy = load_policy(device)

    baseline, base_misses = score(tok, policy, rows, device, max_new_tokens)
    print("baseline khipu-r3:", json.dumps(baseline))
    receipt = {"lane": "khipu-r4-heldout-eval", "label": "held-out generate (greedy) - EVAL",
               "files": HELDOUT_FILES, "data_sha256": digests, "data_revision": data_rev,
               "policy_revision": policy_rev, "hardware": hw, "device": device,
               "max_new_tokens": max_new_tokens, "seed": SEED, "timestamp": now(),
               "baseline_khipu_r3": baseline, "baseline_misses_sample": base_misses,
               "dpo_adapter": "NOT_RUN", "publication_eligible": False,
               "owner_decision_required": True}

    if adapter:
        from peft import PeftModel
        dpo = PeftModel.from_pretrained(policy, adapter)
        dpo.eval()
        dpo_m, dpo_misses = score(tok, dpo, rows, device, max_new_tokens)
        print("khipu-r4 dpo:     ", json.dumps(dpo_m))
        receipt.update(dpo_adapter=adapter, dpo_khipu_r4=dpo_m, dpo_misses_sample=dpo_misses,
                       delta={k: dpo_m[k] - baseline[k] for k in baseline})
    write(out / "eval_receipt.json", receipt)
    print(f"eval receipt written to {out / 'eval_receipt.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dpo")
    d.add_argument("--out", default="dpo-khipu-r4")
    d.add_argument("--max-pairs", type=int, default=256)
    d.add_argument("--epochs", type=int, default=2)
    d.add_argument("--max-new-tokens", type=int, default=384)
    e = sub.add_parser("eval")
    e.add_argument("--out", default="eval-khipu-r4")
    e.add_argument("--adapter", default=None)
    e.add_argument("--max-rows", type=int, default=200)
    e.add_argument("--max-new-tokens", type=int, default=384)
    args = ap.parse_args()
    if args.cmd == "dpo":
        cmd_dpo(pathlib.Path(args.out), args.max_pairs, args.epochs, args.max_new_tokens)
    else:
        cmd_eval(pathlib.Path(args.out), args.adapter, args.max_rows, args.max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
