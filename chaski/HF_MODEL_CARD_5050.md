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
  sku: SZLHOLDINGS/chaski-5050
  not_live_chaski: SZLHOLDINGS/chaski
  jobs: not-an-hf-job
  training_label: REPORTED owner-metal
  evals: none-this-run
  publication_eligible: false
  qlora: banned-on-qwen3.5
  a11oy_mini: live-chaski-gguf
---

# Chaski 5050

Separate SKU `SZLHOLDINGS/chaski-5050`. Not live `SZLHOLDINGS/chaski`. Adapters. Evals none-this-run. Not MEASURED. Training label: **REPORTED owner-metal** until a signed receipt exists.

**One line.** Local RTX 5050 bf16 LoRA recut of SZL doctrine SFT. QLoRA is banned on this family.

| | |
|---|---|
| **Hub id** | `SZLHOLDINGS/chaski-5050` |
| **Live Chaski** | `SZLHOLDINGS/chaski` — a different artifact. Never overwrite. |
| **Silhouette** | `Qwen/Qwen3.5-0.8B` (Apache-2.0). Do not recut onto another instruct family. |
| **Method** | Unsloth bf16 LoRA. `load_in_4bit=False`, `load_in_16bit=True`. r=16, **alpha=16** (not live Chaski's 16/32 pair). |
| **Why not QLoRA** | Unsloth 2026 Qwen3.5 docs: do not QLoRA this family (quantization differences). |
| **Hardware** | owner GPU / local RTX 5050. Not an HF Job. |
| **Data** | jsonl-only `szl_dataset.jsonl`. Refuses `SZL_ESTATE_MANAGED.json`. |
| **Status** | CUTTING. `publication_eligible: false`. |
| **Lab** | Do not load into the Khipu lab. No tok/s claims. |
| **A11OY-MINI** | GGUF of **live** Chaski, not this 5050 kit. |

> **Fashion rule.** Silhouette from Qwen3.5 instruct. Cut is original SZL doctrine SFT. We do not republish someone else's tensors. This card is house copy, not an Unsloth default card.

## Intended use

- **Who:** Alloy controller, not an end-user chatbot.
- **What:** doctrine-faithful drafts and honest UNKNOWN.
- **Where:** behind a validating controller. The weights propose. The controller gates.

Product host: [a-11-oy.com](https://a-11-oy.com).

## What it is NOT

- Not an autonomous agent, executor, factual oracle, or weapon.
- Not a Qwen rehost.
- Not live `SZLHOLDINGS/chaski`.
- Not a QLoRA run.
- Not A11OY-MINI.

## Evaluation

**Status: none-this-run.** Not MEASURED. Train loss is a train metric, not an eval. Do not invent 5/5.

## Training

Pinned: seed 11, 3 epochs, LoRA r=16 alpha=16, lr 2e-4, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `optim=adamw_8bit`, `warmup_steps=6`.

Script: `train_chaski_bf16_5050.py`. No Hub PUT from this checkout. Never overwrite `SZLHOLDINGS/chaski`.

## Limitations

Narrow curriculum. Controller required. Λ = Conjecture 1. Trust ceiling 0.97. `publication_eligible: false`. Lab load forbidden.
