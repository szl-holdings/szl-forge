---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- governed-agent
- retrieval
- brain-navigator
- grounded-only
- proposal-only
- research-only
- szl-holdings
- khipu
- abstain-retrain
- no-weights
- curriculum-only
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  weights: UNAVAILABLE
  jobs: UNAVAILABLE
  evals: none-this-run
  publication_eligible: false
  autonomy_eligible: false
  original_signed_weights: SZLHOLDINGS/SZL-Khipu-1.5B
  successor_with_weights: SZLHOLDINGS/KHIPU-R2
---

> **EXPERIMENT. Adapter bytes missing or unverified.**
> Evaluators use `SZLHOLDINGS/SZL-Khipu-1.5B` until a receipted adapter exists.

> **NO WEIGHTS IN THIS REPO — metadata corrected.** The card already said
> "WEIGHTS UNAVAILABLE", but the front matter simultaneously declared
> `library_name: peft`, `base_model_relation: adapter` and
> `pipeline_tag: text-generation`, plus `peft`/`qlora` tags. Together those tell
> the Hub this is a loadable PEFT adapter. There is no
> `adapter_model.safetensors` here, so it is not. Those four declarations have
> been removed; the prose was already honest and is unchanged.
>
> What *is* here is a complete, runnable training curriculum: `train.jsonl`,
> `train.abstain.jsonl`, `adversarial.jsonl`, `eval.jsonl`, a 23 KB training
> script, and a manifest. Everything needed to produce the adapter is present —
> it has simply not been run. The trained successor is
> [KHIPU-R2](https://huggingface.co/SZLHOLDINGS/KHIPU-R2) (abstain 3/6 MEASURED,
> declared not a pass).

<p align="center">
  <img src="holo-banner.svg" alt="SZL-Khipu-1.5B-abstain — holographic closed-gate banner" width="100%"/>
</p>

<h1 align="center">K H I P U &nbsp;A B S T A I N</h1>

<p align="center"><em>A specialist in silence. Capability is someone else's LoRA.</em></p>

<p align="center">
  <img alt="Base: Qwen2.5-1.5B-Instruct" src="https://img.shields.io/badge/base-Qwen2.5--1.5B--Instruct-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/SZL-Khipu-1.5B-abstain?style=flat-square&color=fb7185&label=downloads"/>
  <img alt="Weights: UNAVAILABLE — curriculum only" src="https://img.shields.io/badge/weights-UNAVAILABLE%20%C2%B7%20curriculum%20only-991b1b?style=flat-square"/>
  <img alt="Eval: not yet run — no fabricated k/n" src="https://img.shields.io/badge/eval-not%20yet%20run%20%C2%B7%20no%20fabricated%20k%2Fn-b45309?style=flat-square"/>
  <img alt="Prior abstain: 2 of 6 blocker" src="https://img.shields.io/badge/prior%20abstain-2%20of%206%20blocker-9f1239?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

**WEIGHTS UNAVAILABLE.** No `adapter_model.safetensors` on this ID. Successor adapter with weights is [`SZLHOLDINGS/KHIPU-R2`](https://huggingface.co/SZLHOLDINGS/KHIPU-R2).

QLoRA **adapter** retrain recipe of the existing Khipu line. Raises in-memory
`ABSTAIN_OVERSAMPLE` from 2 to 4 (8×4=32 abstain vs 15 navigate). Proposal-only.
Λ = Conjecture 1. Doctrine v11 LOCKED 749/14/163.

This ID currently holds curriculum + script only. It is **not** a loadable PEFT adapter.

| | |
|---|---|
| **Weights** | **UNAVAILABLE** |

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Most 'safety LoRAs' teach tone. This one teaches a binary: the handles are not enough. Research-only until abstain beats 2/6.

A specialist in silence. Capability is someone else's LoRA.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Constitutional fine-tune, but only the refuse clause. |
| NVIDIA | A guardrail as weights, not as Colang. |
| Unsloth | QLoRA adapter, proposal-only, research-only tag. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Stack on the navigator. Measure abstain. Do not ship on hope.

## Limitations

- research-only
- proposal-only
- Does not magically fix 2/6 until a signed eval says so.

Canonical GitHub: [`szl-holdings/szl-forge`](https://github.com/szl-holdings/szl-forge/blob/main/khipu/)
<!-- SZL-ATELIER-CUT:v1:END -->

| **Jobs** | **UNAVAILABLE** |
| **Base (canonical)** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **Runtime train** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (same Qwen2.5-1.5B weights, 4-bit) |
| **Relation** | `adapter` (declared; files not present) |
| **License** | Apache-2.0 |
| **Does NOT overwrite** | [`SZLHOLDINGS/SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) signed weights |
| **Successor with weights** | [`SZLHOLDINGS/KHIPU-R2`](https://huggingface.co/SZLHOLDINGS/KHIPU-R2) (MEASURED abstain 3/6, not a pass) |
| **This is NOT** | the Chaski Qwen3.5 lock |

## Evaluation

**Status: NOT YET RUN.** No fabricated k/n. `publication_eligible` is false until
the held-out eval in `train_khipu_abstain.py` actually executes after training.

Prior original MEASURED abstain on `SZLHOLDINGS/SZL-Khipu-1.5B` is **2/6** (blocker).
Eval protocol: `eval.jsonl` 5 navigate + `adversarial.jsonl` 6 abstain. Report k/n only.

## Training (this job)

- Unsloth QLoRA, seed 11, lr 2e-4, adamw_8bit, `train_on_responses_only`, Trackio
- LoRA r=32 α=64, 45 epochs, ga=2, batch=1, constant_with_warmup (from `train_khipu.py`)
- Train: `train.jsonl` 15 navigate + `train.abstain.jsonl` 8 rows × 4
- Held-out never in gradients
- Script: [`train_khipu_abstain.py`](train_khipu_abstain.py)

## Intended use

Proposal-only JSON retrieval plans (`NAVIGATE` / `ABSTAIN`) over synthetic Brain
node handles. A controller outside the weights validates and resolves content.
Not autonomous. Not a replacement for the signed original weights.

---

<p align="center">
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-abstain">SZLHOLDINGS/SZL-Khipu-1.5B-abstain</a>
</p>
