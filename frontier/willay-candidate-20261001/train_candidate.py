#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 3 - train an isolated challenger adapter. Writes out/adapter/ and out/training_report.json.

Preconditions (fail closed): runtime_qualification PASS, leakage_receipt PASS for the bound digests.
--smoke runs `smoke_optimizer_steps` (default 1) to prove the path on this metal; it is NOT a challenger.
--full runs `full_optimizer_steps`; it needs the curriculum binding confirmed (candidate.json
training_data.confirmed_by_owner=true or --confirm-binding) because a heuristically bound corpus must be
checked by a human before GPU hours and a receipt are spent on it.
The report is UNSIGNED. Sign it with the owner's existing signer (receiptagent/sign_receipt.py canonical JSON);
an unsigned report is not receipt-eligible and never publication-eligible.
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_lib as lib  # noqa: E402


def response_only_labels(full, prompt, max_length):
    """Require the exact rendered prompt prefix before assigning assistant loss.

    Inputs are plain token-ID lists from the existing text-only encoder. A chat
    template that renders a different generation prefix is unsupported here;
    never guess a mask from its length or silently drop a row with no target.
    """
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("MASK_MAX_LENGTH_INVALID")
    for ids in (full, prompt):
        if type(ids) is not list or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("MASK_TOKEN_IDS_INVALID")
    if len(prompt) >= len(full) or full[:len(prompt)] != prompt:
        raise ValueError("MASK_PROMPT_PREFIX_MISMATCH")
    if max_length <= len(prompt):
        raise ValueError("MASK_NO_ASSISTANT_TOKENS_AFTER_TRUNCATION")
    clipped = full[:max_length]
    labels = [-100] * len(prompt) + clipped[len(prompt):]
    return clipped, labels, int(len(full) > max_length)


def build_examples(rows, tok, max_length):
    """Response-only loss on a verified prompt prefix; malformed rows fail closed."""
    import torch
    examples, truncated = [], 0
    for r in rows:
        _, full = lib.text_only_encode(tok, r["messages"], add_generation_prompt=False)
        _, prompt = lib.text_only_encode(tok, r["messages"][:-1], add_generation_prompt=True)
        full, labels, clipped = response_only_labels(full, prompt, max_length)
        truncated += clipped
        examples.append({"input_ids": torch.tensor(full, dtype=torch.long),
                         "labels": torch.tensor(labels, dtype=torch.long),
                         "attention_mask": torch.ones(len(full), dtype=torch.long)})
    return examples, truncated


def collate(pad_id):
    import torch

    def _c(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = n - len(b["input_ids"])
            out["input_ids"].append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
            out["labels"].append(torch.cat([b["labels"], torch.full((pad,), -100)]))
            out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in out.items()}
    return _c


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--confirm-binding", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    cand = lib.load_candidate()
    recipe = cand["training_recipe"]
    td = cand["training_data"]
    lib.require_report("runtime_qualification.json")
    leak = lib.require_report("leakage_receipt.json")
    bound = {k: v.get("sha256") for k, v in (td.get("files") or {}).items()}
    if leak.get("curriculum_digests") != bound:
        raise SystemExit("[train] leakage receipt covers different curriculum digests; rerun curriculum.py")
    if args.full and not (td.get("confirmed_by_owner") is True or args.confirm_binding):
        raise SystemExit("[train] --full needs training_data.confirmed_by_owner=true or --confirm-binding "
                         f"(binding_mode={td.get('binding_mode')}). Smoke runs do not need confirmation.")

    rows, _, _ = lib.load_splits(cand)
    train_rows = rows["train"]
    steps = int(recipe["smoke_optimizer_steps"] if args.smoke else recipe["full_optimizer_steps"])
    seed = int(recipe["seed"])
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(seed)

    t0 = time.time()
    model, tok, backend = lib.load_base_model(cand, for_training=True, allow_cpu=args.allow_cpu)
    examples, truncated = build_examples(train_rows, tok, int(recipe["max_length"]))
    if not examples:
        raise SystemExit("[train] no trainable examples after masking")
    print(f"[train] backend={backend} rows={len(train_rows)} examples={len(examples)} truncated={truncated} steps={steps}")
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    optim = recipe.get("optimizer", "adamw_8bit")
    if not cuda or lib.installed_versions().get("bitsandbytes") is None:
        optim = "adamw_torch"
    targs = dict(output_dir=str(lib.OUT / "trainer"), per_device_train_batch_size=int(recipe["per_device_batch_size"]),
                 gradient_accumulation_steps=int(recipe["gradient_accumulation_steps"]), max_steps=steps,
                 learning_rate=float(recipe["learning_rate"]), warmup_steps=min(int(recipe["warmup_steps"]), max(0, steps - 1)),
                 lr_scheduler_type=recipe.get("lr_scheduler", "constant_with_warmup"), weight_decay=float(recipe["weight_decay"]),
                 optim=optim, logging_steps=1, save_strategy="no", seed=seed, report_to="none",
                 bf16=cuda and torch.cuda.is_bf16_supported(), fp16=False, remove_unused_columns=False,
                 gradient_checkpointing=bool(cuda and backend != "unsloth"), dataloader_pin_memory=cuda)
    try:
        training_args = TrainingArguments(**targs)
    except TypeError:
        targs.pop("dataloader_pin_memory", None)
        training_args = TrainingArguments(**targs)
    trainer = Trainer(model=model, args=training_args, train_dataset=examples, data_collator=collate(tok.pad_token_id))
    stats = trainer.train()
    final_loss = f"{stats.training_loss:.6f}"
    adapter_dir = lib.OUT / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    adapter_sha = lib.sha256_dir(adapter_dir)
    wall = time.time() - t0
    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.3f}" if cuda else None
    report = {"kind": "szl.frontier-training-report/v1", "signed": False,
              "state": "SMOKE_ONLY_NOT_A_CHALLENGER" if args.smoke else "TRAINED_CHALLENGER_UNSIGNED",
              "candidate_id": cand["candidate_id"], "target_repo_id": cand["target_repo_id"],
              "predecessor": cand.get("predecessor"), "base": cand["actual_training_base"],
              "curriculum_digests": bound, "train_rows": len(train_rows), "train_examples": len(examples),
              "truncated_examples": truncated, "recipe": recipe, "optimizer_steps": steps, "optimizer_used": optim,
              "backend": backend, "seed": seed, "finalTrainLoss": final_loss, "wall_seconds": f"{wall:.1f}",
              "peak_vram_gib": peak, "gpu": lib.gpu_snapshot(), "versions": lib.installed_versions(),
              "runtime_lock": cand.get("runtime_lock"), "host": lib.host_info(), "trainedAt": lib.now_utc(),
              "adapterSha256": adapter_sha, "adapter_dir": str(adapter_dir), "promotion": "NOT_PROMOTABLE",
              "next": "evaluate_candidate.py --split dev, then sign with the owner signer; never overwrite the predecessor"}
    lib.write_json(lib.OUT / "training_report.json", lib.stringify_floats(report))
    print(f"[train] {report['state']} loss={final_loss} adapter_sha256={adapter_sha} wall={wall:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
