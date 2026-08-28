---
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
library_name: peft
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
tags:
  - cutting
  - adapters
  - lora
  - szl-holdings
  - doctrine-sft
  - proposal-only
szl:
  doctrine: v11-LOCKED
  lambda: "Conjecture 1 — advisory, never a theorem"
  evidence_ceiling: 0.97
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  recut: local-rtx-5050-bf16-lora
  evals: none-this-run
  publication_eligible: false
  qlora: banned-on-qwen3.5
---

# Chaski 5050

Adapters. Evals none-this-run. Not MEASURED.

**One line.** Local RTX 5050 bf16 LoRA recut of SZL doctrine SFT. QLoRA is banned on this family.

| | |
|---|---|
| **Artifact** | LoRA adapters. Not a merge claim until adapter files exist on this repo. |
| **Originality** | SZL doctrine SFT of a disclosed Apache Qwen instruct base. Not a Qwen rehost. |
| **Silhouette** | `Qwen/Qwen3.5-0.8B` (Apache-2.0) |
| **Method** | Unsloth bf16 LoRA. `load_in_4bit=False`, `load_in_16bit=True`. transformers v5. |
| **Why not QLoRA** | Unsloth 2026 Qwen3.5 docs: do not QLoRA this family (quantization differences). |
| **Target GPU** | NVIDIA GeForce RTX 5050 Laptop, 8GB GDDR7 (MEASURED 8151 MiB). |
| **VRAM (REPORTED)** | Unsloth table: 0.8B bf16 LoRA ~3GB. Fit: batch 1, seq 2048, Unsloth gradient checkpointing. |
| **Data** | `SZLHOLDINGS/szl-1-doctrine-sft` — 41 chat JSONL examples, Apache-2.0. Admitted `szl_dataset.jsonl` only. |
| **Status** | CUTTING. `publication_eligible: false`. |
| **Sibling (do not overwrite)** | [`SZLHOLDINGS/chaski`](https://huggingface.co/SZLHOLDINGS/chaski) |

> **Fashion rule.** Silhouette from Qwen3.5 instruct. Cut is original SZL doctrine SFT. We do not republish someone else's tensors. This card is house copy, not an Unsloth default card.

## Intended use

- **Who:** Alloy controller, not an end-user chatbot.
- **What:** doctrine-faithful drafts and honest UNKNOWN.
- **Where:** behind a validating controller. The weights propose. The controller gates.

Product host: [a-11-oy.com](https://a-11-oy.com).

## What it is NOT

- Not an autonomous agent, executor, factual oracle, or weapon.
- Not a Qwen rehost.
- Not a recut of the Unsloth default model card.
- Not a QLoRA run.

## Evaluation

**Status: none-this-run.** Not MEASURED. Train loss is a train metric, not an eval. Do not invent 5/5.

## Training

Pinned: seed 11, 3 epochs, LoRA r=16 alpha=16, lr 2e-4, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `optim=adamw_8bit`, `warmup_steps=10`.

Script: `train_chaski_bf16_5050.py`. Optional Hub push is `SZLHOLDINGS/chaski-5050` only. Never overwrite `SZLHOLDINGS/chaski`.

## Limitations

Narrow curriculum. Controller required. Λ = Conjecture 1. Trust ceiling 0.97. `publication_eligible: false`. CUTTING.
