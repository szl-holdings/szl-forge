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
  job: local-5050
  hub_commit: c907ebe6e1fa900021be7b6fec19b38ec45be574
  hub_readme_commit: 3734d562bb1e06927c736ac293b4a482a142c4a6
  hub_adapter: adapter_model.safetensors
  adapter_sha256: 620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5
  weights: AVAILABLE
  train_loss: 2.228136855544466
  train_loss_label: MEASURED
  train_loss_is_eval: false
  train_runtime_s: 883.2224
  training_rows: 41
  seed: 11
  lora_r: 16
  lora_alpha: 16
  dataset_sha256: ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243
  training_label: REPORTED owner-metal
  sku_status: NOT MEASURED
  live_shelf: false
  live_portfolio: false
  chawpi: separate-sku-off-receipted-unsloth-live
  evals: none-this-run
  publication_eligible: false
  qlora: banned-on-qwen3.5
  a11oy_mini: live-chaski-gguf
---

# Chaski 5050

Separate SKU `SZLHOLDINGS/chaski-5050`. Not live `SZLHOLDINGS/chaski`. Adapters. Evals none-this-run. The SKU is **NOT MEASURED**. Not MEASURED. Do not stamp this model as MEASURED. Training label: **REPORTED owner-metal** until a signed receipt exists.

**One line.** Local RTX 5050 bf16 LoRA recut of SZL doctrine SFT. QLoRA is banned on this family.

| | |
|---|---|
| **Hub id** | `SZLHOLDINGS/chaski-5050` |
| **Hub commit** | `c907ebe6e1fa900021be7b6fec19b38ec45be574` |
| **Hub README** | surgical commit `3734d562bb1e06927c736ac293b4a482a142c4a6`. GitHub stamp only. No Hub PUT. |
| **Live Chaski** | `SZLHOLDINGS/chaski` — a different artifact. Never overwrite. |
| **Silhouette** | `Qwen/Qwen3.5-0.8B` (Apache-2.0). Do not recut onto another instruct family. |
| **Method** | Unsloth bf16 LoRA. `load_in_4bit=False`, `load_in_16bit=True`. r=16, **alpha=16** (not live Chaski's 16/32 pair). |
| **Why not QLoRA** | Unsloth 2026 Qwen3.5 docs: do not QLoRA this family (quantization differences). |
| **Hardware** | owner GPU / local RTX 5050. Not an HF Job. job `local-5050`. |
| **Data** | jsonl-only `szl_dataset.jsonl`. Refuses `SZL_ESTATE_MANAGED.json`. `dataset_sha256` `ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243`. |
| **Weights** | AVAILABLE. `adapter_model.safetensors` present. `adapter_sha256` `620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5`. |
| **Train loss** | MEASURED `2.228136855544466` (`train_runtime` 883.2224s, 3 epochs, 41 rows). train metric, not an eval. |
| **SKU** | **NOT MEASURED.** Do not stamp this model as MEASURED. Separate SKU. Off Receipted Unsloth LIVE shelf. Off the LIVE portfolio table. |
| **Status** | CUTTING. `publication_eligible: false`. |
| **Lab** | No lab load. Do not load into the Khipu lab. No tok/s claims. |
| **A11OY-MINI** | GGUF of **live** Chaski, not this 5050 kit. |
| **CHAWPI** | Do not add to Receipted Unsloth LIVE shelf. That shelf stays live Chaski / Khipu / ReceiptAgent. Do not merge #59. |

> **Fashion rule.** Silhouette from Qwen3.5 instruct. Cut is original SZL doctrine SFT. We do not republish someone else's tensors. This card is house copy, not an Unsloth default card.

> **CHAWPI fashion.** `chaski-5050` is a separate SKU. Do not add it to the Receipted Unsloth LIVE shelf. That shelf stays live Chaski / Khipu / ReceiptAgent. Keep it off the LIVE portfolio table. No lab load. Do not merge #59. Train loss is not an eval.

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

**Status: none-this-run.** The SKU is NOT MEASURED. Not MEASURED. Train loss MEASURED `2.228136855544466` is a train metric, not an eval. Do not stamp this model as MEASURED. Do not invent 5/5. Not 5/5.

## Training

Pinned: seed 11, 3 epochs, LoRA r=16 alpha=16, lr 2e-4, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `optim=adamw_8bit`, `warmup_steps=6`. QLoRA false. job `local-5050`.

Verified Hub receipt (GitHub stamp only): commit `c907ebe6e1fa900021be7b6fec19b38ec45be574`. Hub README surgical commit `3734d562bb1e06927c736ac293b4a482a142c4a6`. `adapter_model.safetensors` present. `adapter_sha256` `620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5`. weights AVAILABLE. `training_receipt.json` train_loss MEASURED `2.228136855544466` is a train metric, not an eval. `train_runtime` 883.2224s, 41 rows. `dataset_sha256` `ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243`. The SKU is **NOT MEASURED**. Do not stamp this model as MEASURED. Training label stays **REPORTED owner-metal** until a signed receipt exists.

Script: `train_chaski_bf16_5050.py`. GitHub stamp only. No Hub PUT from this checkout. Never overwrite `SZLHOLDINGS/chaski`.

## Limitations

Narrow curriculum. Controller required. Λ = Conjecture 1. Trust ceiling 0.97. `publication_eligible: false`. Lab load forbidden.
