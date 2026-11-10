"""Single-GPU DPO preference pass for SZL Forge models (Unsloth backend).

Runs on one A6000-class GPU after an SFT wave. Loads a preference dataset
with `prompt` / `chosen` / `rejected` fields and applies Direct Preference
Optimization on top of the SFT checkpoint. Fails closed on any missing
prerequisite; writes a train receipt like the SFT wave.

Usage:
  python dpo_wave.py --model SZLHOLDINGS/SZL-Khipu-1.5B-sft-v2 \
      --dataset SZLHOLDINGS/governed-sft-mix-v2 \
      --out ./out/khipu-dpo --beta 0.1 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def fail_closed(message: str) -> None:
    print(f"FAIL-CLOSED: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SFT checkpoint HF id or local path")
    parser.add_argument("--dataset", required=True, help="HF dataset id with prompt/chosen/rejected")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default=None, help="If set, push the merged model to this HF repo")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("HF_TOKEN"):
        fail_closed("HF_TOKEN is not set; dataset access and hub push require it.")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        fail_closed("unsloth is not installed. pip install unsloth")

    import torch

    if not torch.cuda.is_available():
        fail_closed("CUDA is not available; this trainer requires one GPU.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seqlen,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    data = load_dataset(args.dataset, split="train")
    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(data.column_names)
    if missing:
        fail_closed(f"dataset lacks preference fields {sorted(missing)}; DPO needs prompt/chosen/rejected.")
    print(f"preference pairs: {len(data)}")

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=DPOConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            beta=args.beta,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=True,
            max_length=args.seqlen,
            seed=args.seed,
            report_to=[],
        ),
    )

    print("DPO start")
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    receipt = {
        "model": args.model,
        "dataset": args.dataset,
        "stage": "DPO",
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "steps": int(trainer.state.global_step),
        "final_loss": float(trainer.state.log_history[-1].get("train_loss", float("nan")))
        if trainer.state.log_history else None,
    }
    receipt_path = Path(args.out) / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt written: {receipt_path}")

    if args.repo_id:
        fail_closed(
            "DIRECT_HUB_PUBLISH_DISABLED: export artifacts and use "
            ".github/workflows/model-publish-gate.yml"
        )

    print("DPO wave complete")


if __name__ == "__main__":
    main()
