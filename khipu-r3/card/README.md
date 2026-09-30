---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - unsloth
  - lora
  - szl-holdings
  - khipu
---

> **SEPARATE RESEARCH ADAPTER · TRAINING RECORD · CANONICAL HELD-OUT GATE FAIL · NOT PROMOTABLE**

# SZLHOLDINGS/khipu-r3

## What this is

A laptop Blackwell research adapter, with a separate derived merged checkpoint also distributed, based on `Qwen/Qwen3.5-0.8B`. The [training_receipt.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/khipu-r3/blob/89807a2680e2400bbbc7523772aa6ff4b86a7988/training_receipt.laptop-blackwell.json) records bf16 LoRA with r=16 and alpha=32, **23 training rows** from `train.jsonl` and `train.abstain.jsonl`, and **train loss 0.4397**. That loss is a training metric, not an evaluation.

## Evidence and artifact identity

Card evidence reviewed on 2026-09-30 UTC at Hub revision [`90af1d3c2d3e79e60d8e1c82fa99a723e6888b90`](https://huggingface.co/SZLHOLDINGS/khipu-r3/tree/90af1d3c2d3e79e60d8e1c82fa99a723e6888b90). The historical artifact/evaluation snapshot below is `89807a2680e2400bbbc7523772aa6ff4b86a7988`. The repository contains [adapter_model.safetensors](https://huggingface.co/SZLHOLDINGS/khipu-r3/blob/89807a2680e2400bbbc7523772aa6ff4b86a7988/adapter_model.safetensors) and [model.safetensors](https://huggingface.co/SZLHOLDINGS/khipu-r3/blob/89807a2680e2400bbbc7523772aa6ff4b86a7988/model.safetensors); the historical training receipt's `weights=LOCAL` and `push_to_hub=false` describe that training step, not current file absence.

There is **no `eval_report.json` in this Hub snapshot**, and no committed Hub held-out receipt supporting the former README claim “grounding 4/5, abstain 0/6.” That unsupported claim is withdrawn.

The separate [post-publication gate receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/evaluation/gate_runs/khipu-r3_89807a2680e2_20260915T040737Z_receipt.json) names Hub revision `89807a2680e2400bbbc7523772aa6ff4b86a7988` and reports **1/5** on the canonical held-out suite in `transformers-cpu` mode, with overall **FAIL**, `heldout_passed=false`, `productionAuthorization=false`, and `publicationAuthorization=false`. This recorded canonical result does not substantiate the former 4/5 + 0/6 claim. It records `refusalPassed=false` and a refusal regression: rate 0.2 against baseline 0.4. The receipt retains runtime warnings, and earlier execution failures remain in the [canonical gate archive](https://github.com/szl-holdings/szl-forge/tree/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/evaluation/gate_runs). Therefore “no held-out evaluation exists anywhere” would also be inaccurate: the canonical source contains this later failed evaluation.

The [merge_receipt.json](https://huggingface.co/SZLHOLDINGS/khipu-r3/blob/89807a2680e2400bbbc7523772aa6ff4b86a7988/merge_receipt.json) reports a 2026-09-14 repair of a prior empty base-copy merge: 96/96 adapter modules applied, maximum weight delta `0.0013806568458676338`. A nonzero merge delta does not establish behavioral quality.

## What the evidence does and does not establish

The [training_receipt.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/khipu-r3/blob/89807a2680e2400bbbc7523772aa6ff4b86a7988/training_receipt.laptop-blackwell.json) retains `evals=none-this-run`, `publication_eligible=false`, and `autonomy_eligible=false`. Its training-run adapter digest is receipt-reported and is not asserted as a direct hash of the currently published adapter file. Exact current artifact qualification does not follow from a training receipt, a merge receipt, or file presence. The later canonical gate names the reviewed Hub revision and returns FAIL. The [provenance.json](https://huggingface.co/SZLHOLDINGS/khipu-r3/blob/89807a2680e2400bbbc7523772aa6ff4b86a7988/provenance.json) also retains blocked artifact-lineage, consent, privacy-review, deployment, and served-revision boundaries.

## Release, autonomy, and deployment boundary

**Not promotable.** `publication_eligible=false` and `autonomy_eligible=false` remain unchanged. No production, deployment authorization, autonomous action, approval authority, or house-lab pin follows from these records. Lambda uniqueness remains **Conjecture 1: open and advisory, never a theorem**.

## Relationship and source of truth

This SKU does not overwrite [SZL-Khipu-1.5B](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B), [SZL-Khipu-1.5B-GGUF](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF), [KHIPU-R2](https://huggingface.co/SZLHOLDINGS/KHIPU-R2), [brain-navigator-r2](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2), or [SZL-Khipu-1.5B-BrainNavigator](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator).

Use the immutable Hub training and merge receipts and the [canonical failed evaluation receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/evaluation/gate_runs/khipu-r3_89807a2680e2_20260915T040737Z_receipt.json) above. The [historical source card](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/khipu-r3/card/README.md) and [Khipu curriculum source](https://github.com/szl-holdings/szl-forge/tree/a6b58f623185820eddfb0b279d21b7d81b87137c/khipu) provide context; their prose is not a substitute for an artifact-scoped evaluation receipt.

Canonical card authoring path: `khipu-r3/card/README.md` in `szl-holdings/szl-forge`. This source correction preserves the stronger dated published evidence; it changes no receipt, model file, release gate or deployed service.
