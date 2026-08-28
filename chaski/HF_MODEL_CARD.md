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
  job_id: 6a91bb7c984507d9db4ea0a4
  job_retry: 6a91bb7c984507d9db4ea0a4
  job_completed_no_weights: 6a91ba00984507d9db4ea07f
  job_namespace: SZLHOLDINGS
  weights: UNAVAILABLE
  evals: none-this-run
  publication_eligible: false
  job_prior_failed:
    - 6a91b8ba984507d9db4ea071
    - 6a91b990984507d9db4ea077
---

# Chaski

**One line.** Messenger LLM. Proposal-only drafts and honest refusals for the SZL controller.

Copied from the Hub card (`SZLHOLDINGS/chaski`). Forge trainer path: `chaski/train_chaski.py`. One recipe. Not `train_szl.py`.

| | |
|---|---|
| **Artifact** | adapter (UNAVAILABLE on Hub) |
| **Originality** | SZL fine-tune of a disclosed Apache Qwen instruct base |
| **Base** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, 0.6B–2B lock) |
| **License** | `apache-2.0` |
| **HF Jobs** | Attempt 4 RUNNING [`6a91bb7c984507d9db4ea0a4`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bb7c984507d9db4ea0a4) (`upload_folder` adapter + merged 16-bit). |
| **Status** | CUTTING until files exist on the repo |
| **Later SKU** | `A11OY-MINI` GGUF of this model after adapters land. ROADMAP. A receipt is not a GGUF parent. |
| **Sibling** | [`szl-receiptagent-qwen35-0.8b-v2`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2) |

> **Fashion rule.** Silhouette from Qwen3 / Qwen3.5 instruct. Cut is original SZL. We do not republish someone else's tensors.

The house CPU lab serves **Khipu GGUF**, not Chaski. Lab load forbidden.

## Intended use

- **Who:** a11oy / Alloy controller, not an end-user chatbot
- **What:** JSON drafts (`decision=DRAFT`, `approvalRequired=true`, `executed=false`) and doctrine-faithful UNKNOWN
- **Where:** behind a validating controller. The weights propose. The controller gates.

## What it is NOT

- Not an autonomous agent, executor, factual oracle, or weapon.
- Not a Qwen rehost.
- Not the live lab model. Not a tokens/s claim.

## Evaluation

**Status: none-this-run.** Not 5/5. Not 6/6. Quality is UNKNOWN. Train loss is a train metric, not an eval.

## Training

- **Recipe:** Unsloth QLoRA SFT. Script: `train_chaski.py`. Loads **only** `szl_dataset.jsonl`.
- **Attempt 4 RUNNING:** [`6a91bb7c984507d9db4ea0a4`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bb7c984507d9db4ea0a4) explicit `upload_folder` adapter + merged 16-bit.
- **Attempt 3 COMPLETED:** [`6a91ba00984507d9db4ea07f`](https://huggingface.co/jobs/SZLHOLDINGS/6a91ba00984507d9db4ea07f). Train loss MEASURED `1.782708187121898` (64/64, 45 rows, seed 11). Safetensors UNAVAILABLE. Receipt only.
- **Attempts 1–2 FAILED:** `6a91b8ba984507d9db4ea071` CastError; `6a91b990984507d9db4ea077` pyyaml 30s timeout.
- **Trackio:** 404 `betterwithage/trackio-bucket`. No dashboard URL.

## Limitations

- Narrow curriculum. Controller required. Λ = Conjecture 1. Trust ceiling 0.97. `publication_eligible: false`. CUTTING until files exist on the repo.
