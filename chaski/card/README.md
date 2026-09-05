---
thumbnail: https://huggingface.co/SZLHOLDINGS/chaski/resolve/main/og-card.png
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: finetune
tags:
- szl-holdings
- series-a
- doctrine-v11
- governed-ai
- proposal-only
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  evidence_ceiling: 0.97
  artifact_class: MERGED_FINETUNE
  originality: FINETUNE_DISCLOSED_BASE
  collection: SZL Fall 2026 — Original Cuts
  jobs: COMPLETED
  job_id: 6a91bf1045686a1580c12105
  job_namespace: SZLHOLDINGS
  weights: AVAILABLE
  evals: MEASURED
  named_n:
    revision: 1c55df8652e9d0f7b84356b1e2d54849165ae884
    date: '2026-08-28'
    json_draft: 0/5
    adversarial_refusal: 2/6
    label: MEASURED
    gate: fail
    report: eval_report.json
    report_sha256: 4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05
    report_bytes: 3996
    report_commit: db71c243d0176bccff1ff087cd4dd57663bd6502
    method: in-process greedy generate, messages[:-1], transformers 5.16.1 bf16 CPU,
      load_in_4bit=False; live Chaski merged shard 1c55df8; PR 63 gates
  publication_eligible: false
  gpu: UNAVAILABLE
  job_prior_error: 6a91bb7c984507d9db4ea0a4
  job_completed_no_weights: 6a91ba00984507d9db4ea07f
  job_prior_failed:
  - 6a91b8ba984507d9db4ea071
  - 6a91b990984507d9db4ea077
  - 6a91bb7c984507d9db4ea0a4
---

<p align="center">
  <img src="holo-banner.svg" alt="Chaski — SZL holographic banner" width="100%"/>
</p>

<h1 align="center">C H A S K I</h1>

<p align="center"><em>The courier that cannot invent the dispatch.</em></p>

<p align="center">
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-8b5cf6?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/chaski?style=flat-square&color=22d3ee&label=downloads"/>
  <img alt="Base: Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen3.5--0.8B-334155?style=flat-square"/>
  <img alt="Doctrine v11-LOCKED" src="https://img.shields.io/badge/doctrine-v11--LOCKED-0f172a?style=flat-square"/>
  <img alt="Named-N: MEASURED FAIL" src="https://img.shields.io/badge/Named--N-MEASURED%20FAIL-dc2626?style=flat-square"/>
</p>

<p align="center">
  <code>KANCHAY</code> · Lean <code>749/14/163</code> · Λ = Conjecture 1 (advisory, never a theorem) · <a href="https://a-11-oy.com">a-11-oy.com</a>
</p>

---

> **RESEARCH / NEGATIVE EVIDENCE.** Failed qualification. Not flagship.
> Later SKU `A11OY-MINI` inherits this failed parent. Not a product claim.

> **Metadata correction 2026-08-30.** This repo declared
> `base_model_relation: adapter` and `artifact_class: ADAPTER` while also declaring
> `weights: AVAILABLE` and shipping a 1.75 GB merged shard. Relative to
> `Qwen/Qwen3.5-0.8B` this is a **finetune**, not an adapter — corrected to
> `base_model_relation: finetune` / `artifact_class: MERGED_FINETUNE`. The card's own
> `originality: FINETUNE_DISCLOSED_BASE` already said so. No eval, weight or receipt
> claim is touched; the MEASURED failing gate below stands exactly as it was.

## One line

Messenger LLM. Proposal-only drafts and honest refusals for the SZL controller. Adapters are on this repo. Named-N evals MEASURED `json_draft` 0/5, `adversarial_refusal` 2/6 — not a pass. Not publication-eligible.

## The cut

Multimodal models narrate what they see. Chaski may only carry a payload the controller already signed. Vision without authorship.

A courier that cannot invent the dispatch. The oldest job in the Andes, as a LoRA.

| Leader | Take, then tweak |
|---|---|
| Anthropic | Claude vision, minus the right to conclude. |
| NVIDIA | NVLM / NeMo multimodal, minus the right to act. |
| Unsloth | Adapter on Qwen3.5-0.8B. |

Nobody else ships this combination. That is the point of a one-of-one.

Canonical GitHub: [`szl-holdings/szl-forge`](https://github.com/szl-holdings/szl-forge/blob/main/chaski/)

## Load it right

Three loadable artifacts sit in this repository at once — a bare `from_pretrained` on the repo id is ambiguous. Be explicit about which one you want.

| Want | Load |
|---|---|
| the merged finetune | `AutoModelForCausalLM.from_pretrained("SZLHOLDINGS/chaski")` — resolves the merged shard via `model.safetensors.index.json` |
| the LoRA adapter on the stock base | base `Qwen/Qwen3.5-0.8B` + `PeftModel.from_pretrained(base, "SZLHOLDINGS/chaski")` |
| the Unsloth-base adapter | base `unsloth/...-bnb-4bit` + PEFT pointed at the `adapter-unsloth` subfolder |

Three sharp edges worth knowing before you debug them:

- **Both `config.json` and `adapter_config.json` live at the repo root.** With `peft` installed, some loader paths follow the adapter rather than the merged weights. Pass the path you mean rather than relying on precedence.
- **The merged shard has a non-standard filename**, `model.safetensors-00001-of-00001.safetensors` instead of the conventional `model-00001-of-00001.safetensors`. It resolves through the index file, but tooling that globs the conventional pattern will not find it.
- **`config.json` is `Qwen3_5ForConditionalGeneration` and carries a `vision_config`;** visual weights are present in the index. If you only want text, that tower still loads.

## Specification

| | |
|---|---|
| **Artifact** | `adapter_model.safetensors` + merged ~1.7GB shard **AVAILABLE** |
| **Parent / eval revision** | `1c55df8652e9d0f7b84356b1e2d54849165ae884` |
| **Originality** | SZL fine-tune of a disclosed Apache Qwen instruct base |
| **Base** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, 0.6B–2B lock). Not an Unsloth-default card. |
| **License** | `apache-2.0` |
| **HF Jobs** | Attempt 5 **COMPLETED** [`6a91bf1045686a1580c12105`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf1045686a1580c12105) (`report_to=none`). Tensors on Hub. |
| **Named-N** | MEASURED fail. `json_draft` **0/5**. `adversarial_refusal` **2/6**. Not a pass. |
| **Receipt** | `eval_report.json` 3996 bytes sha256 `4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05` (commit `db71c24`) |
| **Method** | in-process greedy generate, `messages[:-1]`, transformers 5.16.1 bf16 CPU, `load_in_4bit=False`. Date 2026-08-28. File `eval_report.json`. PR 63. |
| **Quality** | ROADMAP (prose). Failed gate is not a publish. |
| **Status** | CUTTING |
| **Lab** | House CPU lab stays **Khipu GGUF**. Lab load forbidden for Chaski. |
| **Later SKU** | [`A11OY-MINI`](https://huggingface.co/SZLHOLDINGS/A11OY-MINI) GGUFs exist on that repo. They do **not** inherit this Named-N. Mini stays evals none-this-run. |
| **Sibling** | [`szl-receiptagent-qwen35-0.8b-v2`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2) |

## Evaluation

**Named-N MEASURED fail** on live Chaski `1c55df8` (2026-08-28). Receipt: [`eval_report.json`](https://huggingface.co/SZLHOLDINGS/chaski/blob/main/eval_report.json) sha256 `4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05`.

| Probe | N | Score | Label |
|---|---|---|---|
| `json_draft` | 5 | **0/5** | MEASURED fail |
| `adversarial_refusal` | 6 | **2/6** | MEASURED fail |

Method: in-process greedy generate, `messages[:-1]`, transformers 5.16.1 bf16 CPU, `load_in_4bit=False` on live Chaski `1c55df8` (2026-08-28). File: `eval_report.json`. PR 63. What this is NOT: a passing eval gate, a published score, an A11OY-MINI eval, or a 5/5. `publication_eligible: false`.

`train_loss` MEASURED `1.783925924450159` is a train metric, not an eval.

## Training

- **Recipe:** Unsloth QLoRA SFT. Script: `train_chaski.py`. Loads **only** `szl_dataset.jsonl`.
- **1–2 FAILED:** `6a91b8ba984507d9db4ea071` CastError; `6a91b990984507d9db4ea077` pyyaml 30s timeout.
- **3 COMPLETED receipt-only:** [`6a91ba00984507d9db4ea07f`](https://huggingface.co/jobs/SZLHOLDINGS/6a91ba00984507d9db4ea07f).
- **4 ERROR, not RUNNING:** [`6a91bb7c984507d9db4ea0a4`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bb7c984507d9db4ea0a4) Trackio 404. Files persisted.
- **5 COMPLETED `report_to=none`:** [`6a91bf1045686a1580c12105`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf1045686a1580c12105). Tensors now on Hub.
- **Trackio:** 404 `betterwithage/trackio-bucket`. No dashboard URL.
- **publication_eligible:** false

## Intended use

- **Who:** a11oy / Alloy controller, not an end-user chatbot
- **What:** JSON drafts (`decision=DRAFT`, `approvalRequired=true`, `executed=false`) and doctrine-faithful UNKNOWN — carry signed payloads across organs
- **Where:** behind a validating controller. The weights propose. The controller gates.

## What it is NOT

- Not an autonomous agent, executor, factual oracle, or weapon.
- Not a Qwen rehost.
- Not the live lab model. Not a tokens/s claim.
- Not publication-eligible on this MEASURED run.
- Not a 5/5 or 6/6. Do not read 2/6 as a pass.
- Not a VLM product card without a signed vision eval.

## Limitations

- Proposal-only. Narrow curriculum. Controller required.
- Λ = Conjecture 1. Trust ceiling 0.97. CUTTING.
- This MEASURED run failed Named-N. Do not ship as a pass.

---

<p align="center">
  <sub><strong>Fashion rule.</strong> Silhouette from Qwen3 / Qwen3.5 instruct. Cut is original SZL. We do not republish someone else's tensors.</sub>
</p>
