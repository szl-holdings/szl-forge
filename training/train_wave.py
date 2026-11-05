#!/usr/bin/env python3
"""Single-GPU SFT training wave for SZL Forge models (Unsloth backend).

Runs on one A6000-class GPU. Loads one or more HuggingFace chat SFT datasets,
applies the model's chat template, and fine-tunes with QLoRA (4-bit) or LoRA
(bf16). Fails closed on any missing prerequisite; never silently continues.

Usage:
  python train_wave.py --model SZLHOLDINGS/SZL-Khipu-1.5B \
      --datasets SZLHOLDINGS/governed-sft-mix-v2,SZLHOLDINGS/szl-1-doctrine-sft \
      --method qlora --epochs 3 --lr 2e-4 --seqlen 4096 --pack --out ./out/khipu-sft
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
    parser.add_argument("--model", required=True, help="HF model id, e.g. SZLHOLDINGS/SZL-Khipu-1.5B")
    parser.add_argument("--datasets", required=True, help="Comma-separated HF dataset ids")
    parser.add_argument("--method", choices=("qlora", "lora16"), default="qlora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--pack", action="store_true", help="Pack variable-length examples")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--repo-id", default=None, help="If set, push the merged model to this HF repo")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset (smoke runs)")
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

    dtype = None
    load_in_4bit = args.method == "qlora"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seqlen,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=target_modules,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    from datasets import load_dataset, concatenate_datasets

    frames = []
    for dataset_id in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"loading dataset: {dataset_id}")
        frame = load_dataset(dataset_id, split="train")
        if args.max_samples:
            frame = frame.select(range(min(args.max_samples, len(frame))))
        frames.append(frame)
    data = concatenate_datasets(frames)
    print(f"merged corpus: {len(data)} examples")

    def formatting(example):
        messages = example.get("messages") or example.get("conversations")
        if not messages:
            fail_closed(f"example lacks 'messages'/'conversations': {list(example.keys())}")
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def add_text(example):
        example["text"] = formatting(example)
        return example

    data = data.map(add_text, desc="applying chat template")

    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            bf16=True,
            packing=args.pack,
            max_length=args.seqlen,
            seed=args.seed,
            dataset_text_field="text",
            report_to=[],
        ),
    )

    print("training start")
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    receipt = {
        "model": args.model,
        "datasets": args.datasets.split(","),
        "method": args.method,
        "epochs": args.epochs,
        "lr": args.lr,
        "seqlen": args.seqlen,
        "pack": args.pack,
        "seed": args.seed,
        "steps": int(trainer.state.global_step),
        "final_loss": float(trainer.state.log_history[-1].get("train_loss", float("nan")))
        if trainer.state.log_history else None,
    }
    receipt_path = Path(args.out) / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt written: {receipt_path}")

    if args.repo_id:
        model.push_to_hub(args.repo_id, token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(args.repo_id, token=os.environ["HF_TOKEN"])
        print(f"pushed to hub: {args.repo_id}")

    print("wave complete")


if __name__ == "__main__":
    main()
