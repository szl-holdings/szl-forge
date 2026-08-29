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
  evals: "MEASURED fail json_draft 0/5 adversarial_refusal 2/6"
  publication_eligible: false
  job_prior_failed:
    - 6a91b8ba984507d9db4ea071
    - 6a91b990984507d9db4ea077
---

# Chaski

**One line.** Messenger LLM. Proposal-only drafts and honest refusals for the SZL controller.

| | |
|---|---|
| **Artifact** | `adapter_model.safetensors` + merged shard **AVAILABLE** (`adapter_config.json`, `model.safetensors-00001-of-00001.safetensors`) |
| **Originality** | SZL fine-tune of a disclosed Apache Qwen instruct base |
| **Base** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, 0.6B–2B lock) |
| **License** | `apache-2.0` |
| **HF Jobs** | Attempt 5 **COMPLETED** [`6a91bf1045686a1580c12105`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf1045686a1580c12105) (`report_to=none`). Attempt 4 ERROR (no safetensors at that job). |
| **Named-N** | MEASURED fail. `json_draft` **0/5**. `adversarial_refusal` **2/6**. Not a pass. |
| **Status** | CUTTING. Weights AVAILABLE. Train loss is not an eval. `publication_eligible: false`. |
| **Later SKU** | [`A11OY-MINI`](https://huggingface.co/SZLHOLDINGS/A11OY-MINI) GGUFs exist on that repo. They do **not** inherit this Named-N. Mini stays evals none-this-run. |
| **Sibling** | [`szl-receiptagent-qwen35-0.8b-v2`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2) |

> **Fashion rule.** Silhouette from Qwen3 / Qwen3.5 instruct. Cut is original SZL. We do not republish someone else's tensors.

The house CPU lab serves **Khipu GGUF**, not Chaski. Lab load forbidden.

**Evaluation:** Hub Named-N MEASURED fail on live Chaski
`1c55df8652e9d0f7b84356b1e2d54849165ae884` (2026-08-28). Receipt
`eval_report.json` sha256
`4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05`.
`json_draft` 0/5, `adversarial_refusal` 2/6. Not a pass. Quality stays
ROADMAP (prose). `publication_eligible: false`. Train loss MEASURED
`1.783925924450159` is a train metric, not an eval.

## Training

- Recipe: Unsloth QLoRA SFT. Script: train_chaski.py. Loads only szl_dataset.jsonl.
- Attempt 5 COMPLETED 6a91bf1045686a1580c12105 report_to=none. Tensors on Hub. Weights AVAILABLE. Named-N MEASURED fail.
- Attempt 4 ERROR 6a91bb7c after 64/64, train_loss MEASURED 1.7844666938763112, merge ran, upload_folder Trackio 404, no safetensors.
- Attempt 3 COMPLETED. Receipt-only. Train loss MEASURED 1.782708187121898 (64/64, 45 rows, seed 11). Safetensors UNAVAILABLE on that job. Job `6a91ba00984507d9db4ea07f`.
- Attempts 1–2 FAILED: 6a91b8ba CastError; 6a91b990 pyyaml 30s timeout.
- Trackio: 404. No dashboard URL.
- GitHub stamp only. Hub README is not recut from this checkout.
