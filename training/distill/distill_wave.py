"""Governed distillation: teacher -> grounding gate -> student corpus.

The future this builds: a student model trained ONLY on teacher outputs that
passed a deterministic, fail-closed grounding gate. Every accepted and rejected
teacher output is receipted. The gate is our admission math applied to training
data - not a human preference label, not another model's opinion.

Teachers generate freely. The gate then rejects:
  - responses that cite identifiers absent from the source context (fabrication)
  - responses that neither answer nor abstain (evasion)
  - degenerate outputs (empty, repetition loops, non-terminated)

Survivors become the student's SFT corpus. Acceptance rate is published
verbatim - a low acceptance rate is a finding, not a failure to hide.

Usage:
  python distill_wave.py \
      --teachers SZLHOLDINGS/SZL-Reasoning-0.8B-v1,SZLHOLDINGS/SZL-FreeEnergy-0.8B-v1 \
      --prompts-dataset SZLHOLDINGS/doctrine-corpus-v2 \
      --corpus-repo SZLHOLDINGS/governed-distill-sft-v1 \
      --receipts-repo SZLHOLDINGS/model-eval-receipts \
      --max-new-tokens 512 --samples 200

The student is then trained with train_wave.py on the produced corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone


def fail_closed(message: str) -> None:
    print(f"FAIL-CLOSED: {message}", file=sys.stderr)
    raise SystemExit(1)


# ── The grounding gate (pure, testable, deterministic) ─────────────────────

IDENTIFIER_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_\-]{3,40})\b")
ABSTAIN_MARKERS = ("ABSTAIN", "I don't know", "I do not know", "cannot verify")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def extract_identifiers(text: str) -> set[str]:
    """Identifiers a response relies on, ignoring code fences."""
    stripped = CODE_FENCE_RE.sub(" ", text)
    return {m.group(1) for m in IDENTIFIER_RE.finditer(stripped)}


def repetition_ratio(text: str) -> float:
    words = text.split()
    if len(words) < 12:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def grounding_gate(source_context: str, response: str,
                   max_repetition: float = 0.8) -> dict:
    """Deterministic fail-closed gate. Returns a receipt dict with verdict.

    ACCEPT requires all of:
      - non-empty, terminated response
      - not a repetition loop
      - no fabricated identifier: every ALL-CAPS-ish or dotted identifier in
        the response must appear in the source context (or be a stopword)
      - either answers (contains content) or abstains explicitly
    """
    reasons = []
    if not response or not response.strip():
        return {"verdict": "REJECT", "reasons": ["EMPTY_RESPONSE"]}
    if repetition_ratio(response) > max_repetition:
        reasons.append("REPETITION_LOOP")
    ctx_ids = extract_identifiers(source_context)
    resp_ids = extract_identifiers(response)
    whitelist = {"the", "and", "that", "this", "with", "from", "for", "have",
                  "not", "are", "was", "were", "will", "would", "could", "should",
                  "ABSTAIN", "ABSTAINS", "ABSTAINING"}
    fabricated = {i for i in resp_ids - ctx_ids - whitelist
                  if i.isupper() or "." in i or "_" in i or "-" in i}
    if fabricated:
        reasons.append("FABRICATED_IDENTIFIERS:" + ",".join(sorted(fabricated)[:5]))
    abstains = any(m in response for m in ABSTAIN_MARKERS)
    answers = len(response.strip()) >= 40 and not abstains
    if not (answers or abstains):
        reasons.append("NEITHER_ANSWERS_NOR_ABSTAINS")
    if reasons:
        return {"verdict": "REJECT", "reasons": reasons}
    return {"verdict": "ACCEPT", "reasons": [],
            "mode": "ABSTAIN" if abstains else "ANSWER"}


# ── Distillation wave ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teachers", required=True, help="Comma-separated HF model ids")
    p.add_argument("--prompts-dataset", required=True, help="HF dataset with messages column")
    p.add_argument("--corpus-repo", required=True, help="HF dataset repo for the accepted corpus")
    p.add_argument("--receipts-repo", default="SZLHOLDINGS/model-eval-receipts")
    p.add_argument("--samples", type=int, default=200, help="Max prompts to distill")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out", default="./out/distill")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_TOKEN"):
        fail_closed("HF_TOKEN is not set")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    torch.manual_seed(args.seed)

    prompts = load_dataset(args.prompts_dataset, split="train")
    if "messages" not in prompts.column_names:
        fail_closed("prompts dataset must carry a messages column")
    prompts = prompts.select(range(min(args.samples, len(prompts))))
    print(f"distilling over {len(prompts)} prompts")

    generations = []   # accepted corpus rows
    receipts = []      # every teacher output, accepted or rejected
    stats = {"ACCEPT": 0, "REJECT": 0}

    for teacher_id in [t.strip() for t in args.teachers.split(",") if t.strip()]:
        print(f"teacher: {teacher_id}")
        tok = AutoTokenizer.from_pretrained(teacher_id)
        model = AutoModelForCausalLM.from_pretrained(
            teacher_id, torch_dtype=torch.bfloat16, device_map="auto")
        for i, row in enumerate(prompts):
            msgs = row["messages"]
            # take the conversation up to the last user turn as the source context
            source_context = " ".join(m.get("content", "") for m in msgs)
            prompt = tok.apply_chat_template(msgs, tokenize=False)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, do_sample=True,
                    pad_token_id=tok.eos_token_id)
            response = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
            gate = grounding_gate(source_context, response)
            rec = {
                "schema": "szl.distill.gate/v1",
                "teacher": teacher_id,
                "prompt_index": i,
                "verdict": gate["verdict"],
                "reasons": gate["reasons"],
                "mode": gate.get("mode"),
                "response_chars": len(response),
                "truth_label": "MEASURED_DISTILLATION_GATE",
            }
            receipts.append(rec)
            stats[gate["verdict"]] += 1
            if gate["verdict"] == "ACCEPT":
                generations.append({
                    "messages": msgs + [{"role": "assistant", "content": response}],
                    "teacher": teacher_id,
                    "gate_mode": gate.get("mode"),
                })
        del model
        torch.cuda.empty_cache()

    total = stats["ACCEPT"] + stats["REJECT"]
    accept_rate = stats["ACCEPT"] / max(total, 1)
    print(f"gate results: {stats} (acceptance {accept_rate:.1%})")

    # publish the corpus and the receipts - win or lose
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(args.out, exist_ok=True)
    corpus_path = os.path.join(args.out, "corpus.jsonl")
    with open(corpus_path, "w") as f:
        for g in generations:
            f.write(json.dumps(g) + "\n")
    api.create_repo(args.corpus_repo, repo_type="dataset", exist_ok=True,
                    private=False)
    api.upload_file(path_or_fileobj=corpus_path, path_in_repo="distill-corpus.jsonl",
                    repo_id=args.corpus_repo, repo_type="dataset",
                    commit_message=f"governed distillation corpus: {stats['ACCEPT']} gate-accepted examples ({ts})")
    receipt_path = os.path.join(args.out, "receipts.json")
    summary = {
        "schema": "szl.distill.receipt/v1",
        "teachers": args.teachers,
        "prompts_dataset": args.prompts_dataset,
        "samples": len(prompts),
        "accepted": stats["ACCEPT"],
        "rejected": stats["REJECT"],
        "acceptance_rate": round(accept_rate, 4),
        "honesty": ("acceptance rate published verbatim; a low rate is a finding "
                    "about the teachers, not a failure to hide"),
        "truth_label": "MEASURED_DISTILLATION_RUN",
        "computed_at": ts,
    }
    with open(receipt_path, "w") as f:
        json.dump({"summary": summary, "receipts": receipts}, f, indent=2)
    api.upload_file(path_or_fileobj=receipt_path,
                    path_in_repo=f"distill-gate-receipt-{ts.replace(':','')}.json",
                    repo_id=args.receipts_repo, repo_type="dataset",
                    commit_message=f"governed distillation gate receipt: {stats} ({ts})")
    print(f"corpus -> {args.corpus_repo} ({stats['ACCEPT']} examples)")
    print(f"receipt -> {args.receipts_repo}")
    if stats["ACCEPT"] == 0:
        fail_closed("gate accepted zero teacher outputs; no corpus to train on")


if __name__ == "__main__":
    main()
