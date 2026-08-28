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
  jobs: RUNNING
  job_id: 6a91bf1045686a1580c12105
  job_completed_no_weights: 6a91ba00984507d9db4ea07f
  job_error_no_weights: 6a91bb7c984507d9db4ea0a4
  job_namespace: SZLHOLDINGS
  weights: present-on-hub-as-of-2026-08-28T17:08Z
  hub_tensors_observed_at: "2026-08-28T17:08Z"
  evals: none-this-run
  publication_eligible: false
  job_prior_failed:
    - 6a91b8ba984507d9db4ea071
    - 6a91b990984507d9db4ea077
---

# Chaski

**One line.** Messenger LLM. Proposal-only drafts and honest refusals for the SZL controller.

| | |
|---|---|
| **Artifact** | adapter files on Hub as of 2026-08-28T17:08Z (`adapter_model.safetensors`, `adapter_config.json`, `model.safetensors-00001-of-00001.safetensors`) |
| **Originality** | SZL fine-tune of a disclosed Apache Qwen instruct base |
| **Base** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, 0.6B–2B lock) |
| **License** | `apache-2.0` |
| **HF Jobs** | Attempt 5 RUNNING [`6a91bf1045686a1580c12105`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf1045686a1580c12105) (`report_to=none`; likely the upload). Attempt 4 ERROR (no safetensors at that job). |
| **Status** | CUTTING. Hub adapter files exist as of 2026-08-28T17:08Z. Evals none-this-run. Train loss is not an eval. Not 5/5. |
| **Later SKU** | `A11OY-MINI` GGUF of this model. ROADMAP. Hub adapters exist; a GGUF is not cut. |
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

**Status: none-this-run.** Not 5/5. Not 6/6. Quality is UNKNOWN. Train loss is a train metric, not an eval. Adapter files on Hub are not an eval.

## Training

- Recipe: Unsloth QLoRA SFT. Script: train_chaski.py. Loads only szl_dataset.jsonl.
- Attempt 5 RUNNING 6a91bf1045686a1580c12105 report_to=none. Likely the upload that landed Hub tensors. Files on repo as of 2026-08-28T17:08Z. Not restamped COMPLETED.
- Attempt 4 ERROR 6a91bb7c after 64/64, train_loss MEASURED 1.7844666938763112, merge ran, upload_folder Trackio 404, no safetensors.
- Attempt 3 COMPLETED. Receipt-only. Train loss MEASURED 1.782708187121898 (64/64, 45 rows, seed 11). Safetensors UNAVAILABLE on that job. Job `6a91ba00984507d9db4ea07f`.
- Attempts 1–2 FAILED: 6a91b8ba CastError; 6a91b990 pyyaml 30s timeout.
- Trackio: 404. No dashboard URL.
- GitHub stamp only. Hub README is not recut from this checkout.

## Limitations

- Narrow curriculum. Controller required. Λ = Conjecture 1. Trust ceiling 0.97. `publication_eligible: false`. CUTTING. Hub adapter files exist as of 2026-08-28T17:08Z. Evals remain none-this-run.
