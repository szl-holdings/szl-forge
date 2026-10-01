"""SZL one-file pipeline v2 for the OMEN and the 5050 laptop (repaired 2026-09-30).

Repairs the four Frontier-1 execution blockers recorded against v1
(szl-forge huggingface@86046685, blob 3e2751fa):

  1. CUDA is enforced for training/eval. No silent CPU fallback: a missing GPU
     writes a FAIL-CLOSED preflight receipt and exits 2.
  2. Held-out separation is enforced. DPO pairs come ONLY from train.abstain.jsonl
     and train.jsonl; adversarial.jsonl (6 ABSTAIN) and eval.jsonl (5 NAVIGATE)
     are sealed, hashed, and touched only by `eval-abstain`. Prompt disjointness
     is asserted and recorded.
  3. The challenger starts from the loaded policy (base + khipu-r3 adapter merged),
     not from the bare base id. A fresh LoRA (r16/alpha32, bf16) trains on top;
     the reference model is the same policy with the new adapter disabled.
  4. Publication is owner-gated. `publish` refuses any folder carrying a receipt
     with publication_eligible=false (every challenger), and otherwise requires
     --owner-confirmed-wo6 plus HF_TOKEN in the environment. Nothing prompts for
     or prints a token.

Subcommands
  merge         merge the eight Tier-3 adapters into standalone checkpoints (unchanged)
  dpo           on-policy DPO challenger for the khipu abstain lane (CUDA only)
  eval-abstain  held-out MEASURED counts, baseline vs challenger (CUDA only)
  cards         WO1 artifact-identity note for repos that hold a merged root checkpoint
                (additive by default; --relabel flips frontmatter; dry-run unless --apply;
                publisher-managed cards are refused Hub-side and route through szl-forge)
  publish       gated upload of a merged/ folder (never a challenger)

Every stage writes a *_receipt.json with pinned revisions and byte digests.
Train loss is a TRAIN METRIC, NOT AN EVAL. publication_eligible stays false until
an owner decision follows a held-out receipt. Nothing here merges PRs, deletes,
or changes visibility.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import hashlib
import json
import os
import pathlib
import platform
import re

# Set before torch is imported anywhere: expandable segments cut allocator fragmentation on 8 GB laptop GPUs.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCHEMA = "szl.omen-pipeline/v2"
SEED = 11

# ---------------- shared helpers ----------------


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def self_digest() -> str:
    return sha256_of(pathlib.Path(__file__).resolve())


def env_block() -> dict:
    block = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pipeline_sha256": self_digest(),
        "schema": SCHEMA,
        "generated_at": utcnow(),
    }
    for mod in ("torch", "transformers", "peft", "trl", "datasets", "huggingface_hub"):
        try:
            block[mod] = __import__(mod).__version__
        except Exception:
            block[mod] = "NOT_INSTALLED"
    return block


def fail_closed(receipt_path: pathlib.Path, stage: str, reason: str, extra: dict | None = None) -> int:
    payload = {"stage": stage, "status": "FAIL-CLOSED", "reason": reason, **env_block()}
    if extra:
        payload.update(extra)
    write_json(receipt_path, payload)
    print(f"FAIL-CLOSED [{stage}] {reason}\n    receipt -> {receipt_path}")
    return 2


def require_cuda(stage: str, receipt_path: pathlib.Path) -> tuple[int | None, dict]:
    """Return (exit_code_or_None, hardware_block). Never falls back to CPU."""
    import torch

    if not torch.cuda.is_available():
        return fail_closed(receipt_path, stage, "CUDA_UNAVAILABLE: training/eval lanes are GPU-only; CPU fallback is forbidden"), {}
    if not torch.cuda.is_bf16_supported():
        return fail_closed(receipt_path, stage, "BF16_UNSUPPORTED: bf16 LoRA is required (QLoRA forbidden on Qwen3.5)"), {}
    props = torch.cuda.get_device_properties(0)
    hw = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": int(props.total_memory),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    return None, hw


def hub_token_present() -> bool:
    """True if huggingface_hub can find a token (HF_TOKEN env or the local token file). Never returns the value."""
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return bool(os.environ.get("HF_TOKEN"))


def pin_revision(repo_id: str, repo_type: str = "model") -> str:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type=repo_type)
    if not info.sha or len(info.sha) != 40:
        raise RuntimeError(f"could not pin a 40-char revision for {repo_type}:{repo_id}")
    return info.sha


# ---------------- merge (unchanged from the audit wave) ----------------

MERGE_JOBS = [
    ("SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3", "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/khipu-r3", "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/chaski-r2", "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/chaski-5050", "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/brain-navigator-r2", "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/KHIPU-R2", "Qwen/Qwen2.5-1.5B-Instruct", None),
    ("SZLHOLDINGS/WILLAY", "Qwen/Qwen2.5-0.5B-Instruct", None),
]


def cmd_merge(out: pathlib.Path) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out.mkdir(exist_ok=True)
    for repo, base, rev in MERGE_JOBS:
        name = repo.split("/")[-1]
        dest = out / name
        if (dest / "merge_receipt.json").exists():
            print(f"--- {repo}: merge receipt present, skipping (idempotent)")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        adapter_rev = pin_revision(repo)
        print(f"--- merging {repo}@{adapter_rev[:10]}")
        model = AutoModelForCausalLM.from_pretrained(base, revision=rev, torch_dtype=torch.float32, device_map="cpu")
        model = PeftModel.from_pretrained(model, repo, revision=adapter_rev).merge_and_unload()
        model.save_pretrained(dest, safe_serialization=True)
        AutoTokenizer.from_pretrained(repo, revision=adapter_rev).save_pretrained(dest)
        receipt = {
            "stage": "merge", "status": "DONE",
            "repo": repo, "adapter_revision": adapter_rev, "base_model": base,
            "base_revision": rev or "main@merge-time",
            "merged_files": {p.name: sha256_of(p) for p in sorted(dest.glob("*.safetensors"))},
            "merge_dtype": "float32", "publication_eligible": False, **env_block(),
        }
        write_json(dest / "merge_receipt.json", receipt)
        print("    -> done")
    return 0


# ---------------- khipu abstain lane: shared data contract ----------------

DPO_ADAPTER = "SZLHOLDINGS/khipu-r3"           # the policy whose 0/6 abstain this lane exists to fix
DPO_BASE = "Qwen/Qwen3.5-0.8B"
DATA_REPO = "SZLHOLDINGS/SZL-Khipu-1.5B-abstain"  # NOTE: a MODEL-type repo that carries the curriculum files
DATA_REPO_TYPE = "model"
TRAIN_FILES = ("train.abstain.jsonl", "train.jsonl")          # 8 ABSTAIN + 15 NAVIGATE gold rows
HELDOUT_FILES = ("adversarial.jsonl", "eval.jsonl")           # 6 ABSTAIN + 5 NAVIGATE, sealed
GEN_MAX_NEW_TOKENS = 768
CHALLENGER_ID = "dpo-khipu-r4-challenger"


def _download_rows(fname: str, revision: str) -> tuple[list[dict], str]:
    from huggingface_hub import hf_hub_download

    local = pathlib.Path(hf_hub_download(DATA_REPO, fname, repo_type=DATA_REPO_TYPE, revision=revision))
    rows = [json.loads(line) for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, sha256_of(local)


def _user_prompt(row: dict) -> str:
    return row["messages"][1]["content"]


def _gold_plan(row: dict) -> dict:
    return json.loads(row["messages"][-1]["content"])


def _offered_ids(row: dict) -> set[str]:
    return {c["nodeId"] for c in json.loads(_user_prompt(row))["candidates"]}


def load_split(revision: str) -> dict:
    """Load train + held-out rows, assert disjointness, return a bound data block."""
    train_rows: list[dict] = []
    digests: dict[str, str] = {}
    for f in TRAIN_FILES:
        rows, d = _download_rows(f, revision)
        digests[f] = d
        for r in rows:
            r["_source"] = f
        train_rows += rows
    heldout: dict[str, list[dict]] = {}
    for f in HELDOUT_FILES:
        rows, d = _download_rows(f, revision)
        digests[f] = d
        heldout[f] = rows
    train_prompts = {_user_prompt(r) for r in train_rows}
    for f, rows in heldout.items():
        overlap = train_prompts & {_user_prompt(r) for r in rows}
        if overlap:
            raise RuntimeError(f"HELD-OUT CONTAMINATION: {len(overlap)} prompt(s) shared between train and {f}")
    return {
        "train_rows": train_rows, "heldout": heldout, "digests": digests,
        "counts": {"train": len(train_rows), **{f: len(r) for f, r in heldout.items()}},
        "disjoint_prompts_verified": True,
    }


def render_prompt(tok, row: dict) -> str:
    msgs = row["messages"][:-1]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompt_text: str, device: str) -> str:
    import torch

    ids = tok(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(gen[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)


def load_policy(device: str, adapter_rev: str):
    """base + khipu-r3 adapter, merged: this IS the policy under repair."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DPO_ADAPTER, revision=adapter_rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DPO_BASE, torch_dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, DPO_ADAPTER, revision=adapter_rev).merge_and_unload()
    policy.to(device)
    policy.config.use_cache = True
    return tok, policy


# ---------------- grader (task-compatible, integer counts) ----------------


def parse_plan(text: str) -> dict | None:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def grade(row: dict, output: str) -> dict:
    """Grade one held-out generation against the gold decision.

    ABSTAIN rows: correct iff decision == ABSTAIN and citedNodeIds == [].
    NAVIGATE rows: correct iff decision == NAVIGATE and citedNodeIds is a non-empty
    subset of the offered handles. Invented handles are a hard failure. Gold-set
    equality is reported separately as a secondary count.
    """
    gold = _gold_plan(row)
    expected = gold.get("decision")
    offered = _offered_ids(row)
    plan = parse_plan(output)
    if plan is None:
        return {"expected": expected, "parsed": False, "decision": None, "correct": False, "invented_handles": 0, "gold_citation_match": False}
    decision = plan.get("decision")
    cited = plan.get("citedNodeIds") if isinstance(plan.get("citedNodeIds"), list) else []
    cited_set = {c for c in cited if isinstance(c, str)}
    invented = len(cited_set - offered)
    if expected == "ABSTAIN":
        correct = decision == "ABSTAIN" and len(cited_set) == 0
    else:
        correct = decision == "NAVIGATE" and len(cited_set) > 0 and invented == 0
    return {
        "expected": expected, "parsed": True, "decision": decision, "correct": bool(correct),
        "invented_handles": invented, "gold_citation_match": cited_set == set(gold.get("citedNodeIds") or []),
    }


def summarize(grades: list[dict]) -> dict:
    return {
        "total": len(grades),
        "correct": sum(1 for g in grades if g["correct"]),
        "parse_failures": sum(1 for g in grades if not g["parsed"]),
        "invented_handle_rows": sum(1 for g in grades if g["invented_handles"] > 0),
        "gold_citation_matches": sum(1 for g in grades if g["gold_citation_match"]),
    }


# ---------------- dpo ----------------


def cmd_dpo(out: pathlib.Path, epochs: int, max_pairs: int) -> int:
    receipt_path = out / "dpo_receipt.json"
    code, hw = require_cuda("dpo", receipt_path)
    if code is not None:
        return code
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    torch.manual_seed(SEED)
    device = "cuda"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "adapter_config.json").exists():
        return fail_closed(receipt_path, "dpo", f"OUTPUT_EXISTS: {out} already holds an adapter; challengers are immutable, choose a new --out")

    adapter_rev = pin_revision(DPO_ADAPTER)
    data_rev = pin_revision(DATA_REPO, DATA_REPO_TYPE)
    data = load_split(data_rev)
    print(f"--- dpo on {hw['gpu']} | policy {DPO_ADAPTER}@{adapter_rev[:10]} | data @{data_rev[:10]} | "
          f"train rows {data['counts']['train']} | held-out sealed {data['counts']}")

    tok, policy = load_policy(device, adapter_rev)
    policy.eval()

    pairs, classes = [], {"failure_class": 0, "format_drift": 0, "already_correct": 0}
    for row in data["train_rows"][:max_pairs]:
        prompt_text = render_prompt(tok, row)
        chosen = row["messages"][-1]["content"]
        rejected = generate(policy, tok, prompt_text, device)
        g = grade(row, rejected)
        if rejected.strip() == chosen.strip():
            classes["already_correct"] += 1
            continue
        classes["failure_class" if not g["correct"] else "format_drift"] += 1
        pairs.append({"prompt": prompt_text, "chosen": chosen, "rejected": rejected,
                      "source": row["_source"], "expected": g["expected"], "rejected_decision": g["decision"]})
    print(f"    {len(pairs)} on-policy pairs ({classes})")
    if len(pairs) < 4:
        return fail_closed(receipt_path, "dpo", "TOO_FEW_PAIRS: fewer than 4 usable on-policy pairs", {"pair_classes": classes})

    pairs_path = out / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs), encoding="utf-8")

    policy.train()
    policy.config.use_cache = False
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # Memory profile for an 8 GB laptop GPU (RTX 5050 run 2026-09-30 went OOM at step 2/12 on the reference
    # forward: a 248k-vocab logits tensor over the full 1024-token sequence was materialised in fp32).
    # use_logits_to_keep restricts logits to completion positions; precompute_ref_log_probs takes the
    # reference pass out of the training step entirely.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    cfg_kwargs = dict(
        output_dir=str(out / "trainer"), per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=epochs, learning_rate=5e-5, lr_scheduler_type="cosine", logging_steps=1,
        save_strategy="no", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, beta=0.1, max_length=1024,
        max_prompt_length=512, seed=SEED, report_to=[],
        use_logits_to_keep=True, precompute_ref_log_probs=True, precompute_ref_batch_size=1,
    )
    memory_profile = {"use_logits_to_keep": True, "precompute_ref_log_probs": True, "per_device_train_batch_size": 1,
                      "gradient_accumulation_steps": 4, "gradient_checkpointing": True,
                      "alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF", "")}
    for _attempt in range(4):
        try:
            cfg = DPOConfig(**cfg_kwargs)
            break
        except TypeError as exc:
            # older TRL: drop the knob it does not know, record the downgrade honestly
            dropped = next((k for k in ("precompute_ref_batch_size", "use_logits_to_keep", "precompute_ref_log_probs", "max_prompt_length") if k in str(exc) and k in cfg_kwargs), None)
            if dropped is None:
                raise
            cfg_kwargs.pop(dropped)
            memory_profile[dropped] = "UNAVAILABLE_IN_INSTALLED_TRL"
    else:
        raise SystemExit("DPOConfig could not be constructed with the installed TRL")
    trainer = DPOTrainer(
        model=policy,            # the loaded khipu-r3 policy (merged), NOT the bare base id
        ref_model=None,          # reference = same policy with the new LoRA disabled
        peft_config=peft_cfg,    # fresh isolated challenger adapter
        args=cfg,
        train_dataset=Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs]),
        processing_class=tok,
    )
    oom_type = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    try:
        result = trainer.train()
    except oom_type as exc:
        props = torch.cuda.get_device_properties(0)
        return fail_closed(receipt_path, "dpo", "CUDA_OOM: training step exceeded device memory; nothing was saved",
                           {"oom_message": str(exc)[:400], "memory_profile": memory_profile,
                            "device_total_bytes": int(props.total_memory),
                            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                            "pair_classes": classes, "pairs_used": len(pairs)})
    trainer.model.save_pretrained(out)   # adapter only (challenger identity)
    tok.save_pretrained(out)

    receipt = {
        "stage": "dpo", "status": "TRAINED_CHALLENGER", "artifact": CHALLENGER_ID,
        "lane": "khipu-abstain-dpo", "seed": SEED, "epochs": epochs,
        "policy": {"base": DPO_BASE, "adapter": DPO_ADAPTER, "adapter_revision": adapter_rev, "init": "base+adapter merged"},
        "data": {"repo": f"{DATA_REPO_TYPE}:{DATA_REPO}", "revision": data_rev, "digests_sha256": data["digests"],
                 "train_files": list(TRAIN_FILES), "heldout_files_sealed": list(HELDOUT_FILES),
                 "counts": data["counts"], "disjoint_prompts_verified": data["disjoint_prompts_verified"]},
        "pairs_used": len(pairs), "pair_classes": classes, "pairs_sha256": sha256_of(pairs_path),
        "lora": {"r": 16, "alpha": 32, "dtype": "bf16", "qlora": "FORBIDDEN_ON_QWEN3.5"},
        "dpo": {"beta": 0.1, "max_length": 1024, "max_prompt_length": 512, "reference": "policy with challenger adapter disabled",
                "memory_profile": memory_profile},
        "train_loss": float(result.training_loss), "train_loss_label": "TRAIN METRIC, NOT AN EVAL",
        "hardware": hw, "adapter_sha256": {p.name: sha256_of(p) for p in sorted(out.glob("adapter_model.*"))},
        "evaluation": "NOT_RUN", "publication_eligible": False, "autonomy_eligible": False, "promotion": "NOT_PROMOTABLE",
        "next_gate": f"python szl_omen_pipeline.py eval-abstain --challenger {out}", **env_block(),
    }
    write_json(receipt_path, receipt)
    print(f"    train_loss {result.training_loss:.4f} (TRAIN METRIC, NOT AN EVAL); receipt -> {receipt_path}")
    return 0


# ---------------- eval-abstain (held-out, baseline vs challenger) ----------------


def cmd_eval(challenger: pathlib.Path, out: pathlib.Path) -> int:
    receipt_path = out / "eval_receipt.json"
    code, hw = require_cuda("eval-abstain", receipt_path)
    if code is not None:
        return code
    import torch
    from peft import PeftModel

    torch.manual_seed(SEED)
    device = "cuda"
    dpo_receipt_path = challenger / "dpo_receipt.json"
    if not (challenger / "adapter_config.json").exists() or not dpo_receipt_path.exists():
        return fail_closed(receipt_path, "eval-abstain", f"CHALLENGER_INCOMPLETE: {challenger} lacks adapter_config.json or dpo_receipt.json")
    dpo_receipt = json.loads(dpo_receipt_path.read_text(encoding="utf-8"))
    adapter_rev = dpo_receipt["policy"]["adapter_revision"]
    data_rev = dpo_receipt["data"]["revision"]
    data = load_split(data_rev)
    for f, d in data["digests"].items():
        if dpo_receipt["data"]["digests_sha256"].get(f) != d:
            return fail_closed(receipt_path, "eval-abstain", f"DATA_DRIFT: {f} digest differs from the training receipt")

    tok, policy = load_policy(device, adapter_rev)
    model = PeftModel.from_pretrained(policy, str(challenger))
    model.eval()

    results: dict[str, dict] = {}
    transcripts = []
    for f, rows in data["heldout"].items():
        base_grades, chal_grades = [], []
        for i, row in enumerate(rows):
            prompt_text = render_prompt(tok, row)
            with model.disable_adapter():
                base_out = generate(model, tok, prompt_text, device)
            chal_out = generate(model, tok, prompt_text, device)
            bg, cg = grade(row, base_out), grade(row, chal_out)
            base_grades.append(bg)
            chal_grades.append(cg)
            transcripts.append({"file": f, "index": i, "baseline": {"output": base_out, **bg}, "challenger": {"output": chal_out, **cg}})
        results[f] = {"baseline": summarize(base_grades), "challenger": summarize(chal_grades)}
        print(f"    {f}: baseline {results[f]['baseline']['correct']}/{len(rows)} | challenger {results[f]['challenger']['correct']}/{len(rows)}")

    out.mkdir(parents=True, exist_ok=True)
    tpath = out / "eval_transcripts.jsonl"
    tpath.write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in transcripts), encoding="utf-8")
    abst, nav = results["adversarial.jsonl"], results["eval.jsonl"]
    advisory = (abst["challenger"]["correct"] > abst["baseline"]["correct"]
                and nav["challenger"]["correct"] >= nav["baseline"]["correct"]
                and nav["challenger"]["invented_handle_rows"] <= nav["baseline"]["invented_handle_rows"])
    receipt = {
        "stage": "eval-abstain", "status": "MEASURED", "artifact": CHALLENGER_ID,
        "challenger_dir": str(challenger.resolve()), "challenger_adapter_sha256": dpo_receipt.get("adapter_sha256"),
        "policy": dpo_receipt["policy"], "data": {"repo": dpo_receipt["data"]["repo"], "revision": data_rev,
                                                  "heldout_digests_sha256": {f: data["digests"][f] for f in HELDOUT_FILES},
                                                  "counts": data["counts"], "disjoint_prompts_verified": True},
        "decode": {"greedy": True, "max_new_tokens": GEN_MAX_NEW_TOKENS, "seed": SEED},
        "grader": "decision + citedNodeIds subset-of-offered; integer counts; see grade()",
        "results": results, "transcripts_sha256": sha256_of(tpath),
        "advisory_challenger_improves": bool(advisory),
        "advisory_note": "Advisory only. Promotion is derived by the canonical Forge gate and an owner decision, never by this tool.",
        "hardware": hw, "publication_eligible": False, "autonomy_eligible": False, "promotion": "NOT_PROMOTABLE", **env_block(),
    }
    write_json(receipt_path, receipt)
    print(f"    MEASURED receipt -> {receipt_path}")
    return 0


# ---------------- cards (WO1 reconciliation) ----------------

# Where each card is actually managed. Hub-side edits to publisher-managed cards are overwritten by the
# next szl-forge main-push fanout, so those route through a szl-forge PR on the source card instead.
CARD_MANAGEMENT = {
    "SZLHOLDINGS/khipu-r3": "hub-pr",
    "SZLHOLDINGS/brain-navigator-r2": "hub-pr",
    "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3": "hub-pr",
    "SZLHOLDINGS/chaski-r2": "szl-forge:chaski-r2/card/README.md (publish-chaski-card.yml)",
    "SZLHOLDINGS/chaski-5050": "szl-forge:chaski-5050/card/README.md (publish-chaski-card.yml)",
    "SZLHOLDINGS/KHIPU-R2": "szl-forge:khipu-r2/card/README.md (publish-khipu-card.yml)",
    "SZLHOLDINGS/WILLAY": "szl-forge:willay/card/README.md (closed curated pipeline, pinned source sha)",
    "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2": "adapter-only root; out of scope",
}
CARD_REPOS_DEFAULT = [r for r, m in CARD_MANAGEMENT.items() if m == "hub-pr"]
CARD_REPOS_ALL = [job[0] for job in MERGE_JOBS]
NOTE_MARKER = "<!-- szl:artifact-identity-reconciled -->"


def reconcile_card(text: str, files: set[str], today: str, relabel: bool = False) -> tuple[str, list[str]]:
    """Return (new_text, changes). Default is ADDITIVE (identity note only). relabel=True also flips the
    frontmatter to transformers/finetune, which the publisher contracts for chaski-*/KHIPU-R2 forbid."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return text, ["SKIP: no YAML frontmatter"]
    fm, body = m.group(1), text[m.end():]
    changes: list[str] = []
    if not ({"config.json", "model.safetensors"} <= files):
        return text, ["SKIP: root is adapter-only (no config.json + model.safetensors at root)"]
    if relabel and re.search(r"^library_name: peft$", fm, re.M):
        fm = re.sub(r"^library_name: peft$", "library_name: transformers", fm, flags=re.M)
        changes.append("library_name: peft -> transformers")
    if relabel and re.search(r"^base_model_relation: adapter$", fm, re.M):
        fm = re.sub(r"^base_model_relation: adapter$", "base_model_relation: finetune", fm, flags=re.M)
        changes.append("base_model_relation: adapter -> finetune")
    for tag in (("peft", "lora") if relabel else ()):
        if re.search(rf"^- {tag}$", fm, re.M):
            fm = re.sub(rf"^- {tag}\n", "", fm, flags=re.M)
            changes.append(f"tags: removed '{tag}'")
    if relabel and re.search(r"^- base_model:adapter:", fm, re.M):
        fm = re.sub(r"^- base_model:adapter:", "- base_model:finetune:", fm, flags=re.M)
        changes.append("tags: base_model:adapter:* -> base_model:finetune:*")
    if NOTE_MARKER not in body:
        merge_ref = " (`merge_receipt.json`)" if "merge_receipt.json" in files else ""
        note = (f"{NOTE_MARKER}\n> **Artifact identity (noted {today}).** Besides the LoRA adapter, this repository's root carries a "
                "merged full-precision checkpoint (`model.safetensors` + `config.json`, loadable with `transformers`) produced by "
                f"the receipted CPU merge of this adapter into its declared base{merge_ref}. The adapter remains the artifact of "
                "record for every figure on this card; the merged bytes carry no separate held-out receipt and add no claim. "
                "Metadata-only note.\n\n")
        body = note + body.lstrip("\n")
        changes.append("note: artifact-identity blockquote inserted after frontmatter")
    if not changes:
        return text, ["NOOP: card already reconciled"]
    return f"---\n{fm}\n---\n{body}", changes


def cmd_cards(apply: bool, direct: bool, owner_confirmed: bool, out: pathlib.Path, relabel: bool = False, all_repos: bool = False,
              owner_authorized: bool = False) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    receipt_path = out / "cards_receipt.json"
    if apply and not (owner_confirmed or owner_authorized):
        return fail_closed(receipt_path, "cards", "OWNER_GATE_CLOSED: --apply requires --owner-confirmed-wo6 or --owner-authorized")
    if apply and direct and not owner_confirmed:
        return fail_closed(receipt_path, "cards", "OWNER_GATE_CLOSED: --direct commits require --owner-confirmed-wo6; --owner-authorized only opens Hub PRs")
    wo6_status = ("OWNER_CONFIRMED_REVOKED" if owner_confirmed
                  else "OWNER_AUTHORIZED_PR_MODE_REVOCATION_NOT_VERIFIED_HERE" if owner_authorized else "CLOSED")
    if apply and not hub_token_present():
        return fail_closed(receipt_path, "cards", "HF_TOKEN_MISSING: set HF_TOKEN in this shell or save it via the Notepad token-file route; never paste it in chat")
    api = HfApi()
    today = utcnow()[:10]
    items = []
    out.mkdir(parents=True, exist_ok=True)
    for repo in (CARD_REPOS_ALL if all_repos else CARD_REPOS_DEFAULT):
        managed = CARD_MANAGEMENT.get(repo, "unknown")
        info = api.repo_info(repo)
        rev = info.sha
        files = {s.rfilename for s in info.siblings}
        local = pathlib.Path(hf_hub_download(repo, "README.md", revision=rev))
        before = local.read_text(encoding="utf-8")
        after, changes = reconcile_card(before, files, today, relabel=relabel)
        if managed != "hub-pr" and after != before:
            changes = changes + [f"WARNING: card is managed by {managed}; a Hub-side edit is overwritten on the next fanout"]
        item = {"repo": repo, "revision": rev, "managed_by": managed, "before_sha256": sha256_text(before), "after_sha256": sha256_text(after),
                "changes": changes, "mode": "dry-run"}
        if after != before:
            diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "README.md@" + rev[:10], "README.md (reconciled)"))
            (out / f"{repo.split('/')[1]}.README.diff").write_text(diff, encoding="utf-8")
            (out / f"{repo.split('/')[1]}.README.md").write_text(after, encoding="utf-8")
            if apply and managed != "hub-pr":
                item["mode"] = "refused: publisher-managed; route through szl-forge"
            elif apply:
                from huggingface_hub import CommitOperationAdd

                res = api.create_commit(
                    repo_id=repo, operations=[CommitOperationAdd("README.md", after.encode("utf-8"))],
                    commit_message="card: note the merged root checkpoint alongside the adapter (metadata only)",
                    commit_description="Additive artifact-identity note per SZL-HF-FRONTIER-1 WO1. No evaluation claim added; frontmatter unchanged unless --relabel was used.",
                    parent_commit=rev, create_pr=not direct,
                )
                item["mode"] = "commit" if direct else "pull-request"
                item["result_url"] = getattr(res, "pr_url", None) or getattr(res, "commit_url", None)
        print(f"--- {repo}@{rev[:10]}: {'; '.join(changes)}" + (f" -> {item.get('result_url')}" if item.get("result_url") else ""))
        items.append(item)
    receipt = {"stage": "cards", "status": "DONE" if apply else "DRY-RUN", "relabel": relabel, "wo6_status": wo6_status, "items": items,
               "publication_eligible": False, "note": "Additive identity note by default; publisher-managed cards are refused Hub-side and route through szl-forge; v2 (adapter-only root) is skipped by rule.", **env_block()}
    write_json(receipt_path, receipt)
    print(f"    receipt -> {receipt_path}")
    return 0


# ---------------- publish (gated) ----------------

CONFIG_FIXES = {
    "chaski-5050": "Qwen/Qwen3.5-0.8B",
    "szl-receiptagent-qwen35-0.8b-v2": "Qwen/Qwen3.5-0.8B",
    "KHIPU-R2": "Qwen/Qwen2.5-1.5B-Instruct",
}


def cmd_publish(folder: pathlib.Path, only: str | None, owner_confirmed: bool) -> int:
    receipt_path = folder / "publish_receipt.json"
    for rp in list(folder.rglob("dpo_receipt.json")) + list(folder.rglob("eval_receipt.json")):
        data = json.loads(rp.read_text(encoding="utf-8"))
        if not data.get("publication_eligible", False):
            return fail_closed(receipt_path, "publish", f"CHALLENGER_NOT_PUBLISHABLE: {rp} carries publication_eligible=false; this tool never publishes challengers")
    if not owner_confirmed:
        return fail_closed(receipt_path, "publish", "OWNER_GATE_CLOSED: publish requires --owner-confirmed-wo6 (exposed tokens revoked)")
    if not hub_token_present():
        return fail_closed(receipt_path, "publish", "HF_TOKEN_MISSING: set HF_TOKEN in this shell or save it via the Notepad token-file route; never paste it in chat")
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()
    me = api.whoami()
    done = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if only and sub.name != only:
            continue
        if not (sub / "merge_receipt.json").exists():
            print(f"--- {sub.name}: no merge_receipt.json; skipped (only receipted merges publish)")
            continue
        repo = f"SZLHOLDINGS/{sub.name}"
        cfg = sub / "adapter_config.json"
        if sub.name in CONFIG_FIXES and cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8-sig"))
            data["base_model_name_or_path"] = CONFIG_FIXES[sub.name]
            cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        readme = sub / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            new = text.replace("base_model_relation: adapter", "base_model_relation: finetune").replace("library_name: peft", "library_name: transformers")
            if new != text:
                readme.write_text(new, encoding="utf-8")
        parent = pin_revision(repo)
        info = upload_folder(repo_id=repo, folder_path=str(sub), parent_commit=parent,
                             commit_message="Merged full-precision checkpoint (receipted) + config fixes")
        url = getattr(info, "commit_url", str(info))
        print(f"--- {repo}: OK -> {url}")
        done.append({"repo": repo, "parent_commit": parent, "commit_url": url})
    write_json(receipt_path, {"stage": "publish", "status": "DONE", "actor": me.get("name"), "published": done,
                              "publication_eligible": False, **env_block()})
    return 0


# ---------------- main ----------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("--out", default="merged")
    d = sub.add_parser("dpo"); d.add_argument("--out", default=CHALLENGER_ID); d.add_argument("--epochs", type=int, default=2); d.add_argument("--max-pairs", type=int, default=64)
    e = sub.add_parser("eval-abstain"); e.add_argument("--challenger", default=CHALLENGER_ID); e.add_argument("--out", default=None)
    c = sub.add_parser("cards"); c.add_argument("--apply", action="store_true"); c.add_argument("--direct", action="store_true", help="commit directly instead of opening Hub PRs"); c.add_argument("--owner-confirmed-wo6", action="store_true"); c.add_argument("--out", default="cards-reconcile"); c.add_argument("--relabel", action="store_true", help="also flip frontmatter to transformers/finetune (publisher contracts forbid this for chaski-*/KHIPU-R2)"); c.add_argument("--all", action="store_true", help="include publisher-managed cards in the dry-run"); c.add_argument("--owner-authorized", action="store_true", help="owner-directed run with the estate publisher credential: opens Hub PRs only, never direct commits; WO6 revocation is recorded as not verified here")
    p = sub.add_parser("publish"); p.add_argument("--folder", default="merged"); p.add_argument("--only", default=None); p.add_argument("--owner-confirmed-wo6", action="store_true")
    args = ap.parse_args()
    if args.cmd == "merge":
        return cmd_merge(pathlib.Path(args.out))
    if args.cmd == "dpo":
        return cmd_dpo(pathlib.Path(args.out), args.epochs, args.max_pairs)
    if args.cmd == "eval-abstain":
        ch = pathlib.Path(args.challenger)
        return cmd_eval(ch, pathlib.Path(args.out) if args.out else ch / "eval")
    if args.cmd == "cards":
        return cmd_cards(args.apply, args.direct, args.owner_confirmed_wo6, pathlib.Path(args.out), args.relabel, args.all, args.owner_authorized)
    return cmd_publish(pathlib.Path(args.folder), args.only, args.owner_confirmed_wo6)


if __name__ == "__main__":
    raise SystemExit(main())
