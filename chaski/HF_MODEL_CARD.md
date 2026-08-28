---
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
tags:
  - szl-holdings
  - series-a
  - doctrine-v11
  - governed-ai
  - proposal-only
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  evidence_ceiling: 0.97
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  collection: "SZL Fall 2026 — Original Cuts"
  jobs: COMPLETED
  job_id: 6a91bf1045686a1580c12105
  job_completed_no_weights: 6a91ba00984507d9db4ea07f
  job_error_no_weights: 6a91bb7c984507d9db4ea0a4
  job_namespace: SZLHOLDINGS
  weights: AVAILABLE
  hub_tensors_observed_at: "2026-08-28T17:08Z"
  evals: none-this-run
  publication_eligible: false
  job_prior_failed:
    - 6a91b8ba984507d9db4ea071
    - 6a91b990984507d9db4ea077
---

# Chaski

Messenger LLM. Proposal-only drafts and honest refusals for the SZL controller.

| | |
|---|---|
| **Artifact** | Hub files as of 2026-08-28T17:08Z: `adapter_model.safetensors` (25.6MB), merged 1.7GB shard `model.safetensors-00001-of-00001.safetensors`, `adapter_config.json`, `training_receipt.json` |
| **Originality** | SZL fine-tune of a disclosed Apache Qwen instruct base |
| **Base** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, 0.6B–2B lock) |
| **License** | `apache-2.0` |
| **HF Jobs** | Attempt 5 [`6a91bf1045686a1580c12105`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf1045686a1580c12105) COMPLETED, weights AVAILABLE. `6a91bb7c` stays ERROR. |
| **Status** | CUTTING. Weights AVAILABLE. Evals none-this-run. MEASURED train_loss ~1.7839 is not an eval. Not ROADMAP. |
| **Later SKU** | `A11OY-MINI` is a later quantized GGUF of this adapter. ROADMAP. Not `base_model_relation: quantized` of an empty parent. Hub adapters exist; a GGUF is not cut. |
| **Sibling** | [`szl-receiptagent-qwen35-0.8b-v2`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2) |

> **Fashion rule.** Silhouette from Qwen3 / Qwen3.5 instruct. Cut is original SZL. We do not republish someone else's tensors.

The house CPU lab serves **Khipu GGUF**, not Chaski. Lab load forbidden.

Forge trainer: [`chaski/train_chaski.py`](./train_chaski.py). One recipe. Not `train_szl.py`.

## Intended use

- **Who:** a11oy / Alloy controller, not an end-user chatbot
- **What:** JSON drafts (`decision=DRAFT`, `approvalRequired=true`, `executed=false`) and doctrine-faithful UNKNOWN
- **Where:** behind a validating controller. The weights propose. The controller gates.

## What it is NOT

- Not an autonomous agent, executor, factual oracle, or weapon.
- Not a Qwen rehost.
- Not the live lab model. Not a tokens/s claim.

## Evaluation

**Status: none-this-run.** Quality is UNKNOWN. MEASURED train_loss ~1.7839 is a train metric, not an eval. Adapter files on Hub are not an eval. Λ is Conjecture 1. Hub README is the job’s card; this GitHub copy is not recut onto Hub.

## Training

- Recipe: Unsloth QLoRA SFT. Script: train_chaski.py. Loads only szl_dataset.jsonl.
- Hub files: `adapter_model.safetensors` (25.6MB), merged 1.7GB shard, `training_receipt.json`. Evals none-this-run.
- Attempt 5 6a91bf1045686a1580c12105 COMPLETED, weights AVAILABLE. Evals none-this-run.
- Attempt 4 ERROR 6a91bb7c after upload (after 64/64, train_loss MEASURED 1.7844666938763112, merge ran, upload_folder Trackio 404).
- Attempt 3 COMPLETED. Receipt-only. Train loss MEASURED 1.782708187121898 (64/64, 45 rows, seed 11). Safetensors UNAVAILABLE on that job. Job `6a91ba00984507d9db4ea07f`.
- Attempts 1–2 FAILED: 6a91b8ba CastError; 6a91b990 pyyaml 30s timeout.
- Trackio: 404. No dashboard URL.
- GitHub stamp only. Hub README is not recut from this checkout.

## Limitations

- Narrow curriculum. Controller required. Λ = Conjecture 1. Trust ceiling 0.97. `publication_eligible: false`. CUTTING. Hub has `adapter_model.safetensors` (25.6MB) and a merged 1.7GB shard as of 2026-08-28T17:08Z. Evals remain none-this-run.
