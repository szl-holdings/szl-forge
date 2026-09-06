---
thumbnail: https://huggingface.co/SZLHOLDINGS/KHIPU-R2/resolve/main/og-card.png
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - qlora
  - peft
  - governed-agent
  - proposal-only
  - research-only
  - szl-holdings
  - khipu
  - abstain-retrain
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: ADAPTER
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/SZL-Khipu-1.5B
  jobs: COMPLETED
  job_id: "6a91bf11984507d9db4ea104"
  job_prior_error: "6a91ba2c45686a1580c12020"
  weights: AVAILABLE
  evals: MEASURED
  gpu: UNAVAILABLE
---

<p align="center">
  <img src="holo-banner.svg" alt="KHIPU-R2 — holographic step-up banner" width="100%"/>
</p>

<h1 align="center">K H I P U — R 2</h1>

<p align="center"><em>Failure is an artifact, not a footnote.</em></p>

<p align="center">
  <img alt="Base: Qwen2.5-1.5B-Instruct" src="https://img.shields.io/badge/base-Qwen2.5--1.5B--Instruct-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/KHIPU-R2?style=flat-square&color=3af4c8&label=downloads"/>
  <img alt="Artifact: QLoRA adapter r=32 a=64" src="https://img.shields.io/badge/artifact-QLoRA%20adapter%20r32%20a64-0d9488?style=flat-square"/>
  <img alt="Abstain: MEASURED 3/6, was 2/6 — not a pass" src="https://img.shields.io/badge/abstain-3%2F6%20was%202%2F6%20%C2%B7%20not%20a%20pass-b45309?style=flat-square"/>
  <img alt="Plan-valid 11/11" src="https://img.shields.io/badge/plan--valid-11%2F11-16a34a?style=flat-square"/>
  <img alt="Hallucinated citations: 0" src="https://img.shields.io/badge/hallucinated%20citations-0-16a34a?style=flat-square"/>
  <img alt="Lifecycle: research-only" src="https://img.shields.io/badge/lifecycle-research--only-7e8aa3?style=flat-square"/>
</p>

<p align="center">
  <code>KANCHAY</code> · Doctrine v11 · Lean <code>749/14/163</code> · Λ = Conjecture 1 (advisory) · <a href="https://a-11-oy.com">a-11-oy.com</a>
</p>

Adapter files are present in this Hub model repository. This card-only release does not modify them. Abstain is MEASURED 3/6, not a pass. Not publication-eligible.

QLoRA adapter on disclosed [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (runtime `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`). Proposal-only brain navigator / abstain retrain. Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory, never a theorem).

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Leaders ship v1 and changelog the rest. We name the retraining of silence as its own model. Failure is an artifact, not a footnote.

A public retrain whose only job is to improve one metric: honest abstain under adversarial handles.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Red-team → constitution update. We red-team → adapter. |
| NVIDIA | Recipe re-run with a new seed and a bounded delta. |
| Unsloth | Same FastLanguageModel loop, new curriculum, new evidence record. |

This card makes the combination inspectable: a separate abstain-retraining SKU, bounded small-n results, and an exact-source documentation publisher that cannot modify weights or promotion state.

## Intended use

Continue the abstain-retrain loop. Keep publication and autonomy gates closed until an independently verifiable evaluation receipt satisfies them.

## Limitations

- research-only
- No signed R2 eval receipt in this atelier.
- Publishing this card is a documentation update, not model promotion or autonomy approval.

Canonical GitHub: [`szl-holdings/szl-forge/khipu_r2`](https://github.com/szl-holdings/szl-forge/tree/main/khipu_r2)
<!-- SZL-ATELIER-CUT:v1:END -->

## Specification

| | |
|---|---|
| **Artifact** | `adapter_model.safetensors` (147.8M) + `adapter_config.json` **AVAILABLE** |
| **Job** | [`6a91bf11984507d9db4ea104`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf11984507d9db4ea104) **COMPLETED** |
| **Does NOT overwrite** | signed [`SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) |
| **Prior job** | [`6a91ba2c`](https://huggingface.co/jobs/SZLHOLDINGS/6a91ba2c45686a1580c12020) **ERROR** Trackio 404 |
| **Card publication** | Exact-source README + local SVG only; weights, adapter, configs, evals, visibility, hardware, and runtime unchanged |
| **Lab** | Forbidden. Pin stays Khipu GGUF. GPU **UNAVAILABLE**. |
| **License** | Apache-2.0 |
| **Autonomy** | false |

## Evaluation (MEASURED this job)

Method: in-process Unsloth generate, scoring ported from `eval_khipu.py`, temperature 0, held-out never in gradients. Host job worker. Date 2026-08-28 17:20 UTC. File: `eval_measured.json`.

| split | k/n | what-NOT |
|---|---|---|
| plan-valid | **11 / 11** | not a public leaderboard |
| grounding (`eval.jsonl` navigate) | **5 / 5** | n=5 |
| abstain (`adversarial.jsonl`) | **3 / 6** | not 5/5, not 6/6 |
| hallucinated citations | **0** | this job only |

Prior published original `SZL-Khipu-1.5B` MEASURED abstain was **2/6**. This run is **3/6**. Small n. Do not derive a world-rank score from k/n on n=11.

## Training (MEASURED / REPORTED)

- Unsloth QLoRA, seed 11, lr 2e-4, LoRA r=32 α=64, 45 epochs
- Train: 15 navigate + 8 abstain rows × oversample 4 (in-memory 32)
- Held-out: 5 + 6, `held_out_in_gradients: false`
- `training_loss` MEASURED `0.017188…` is a train metric, not an eval
- adapter sha256 `e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349`

## What this is NOT

- Not a replacement for `SZL-Khipu-1.5B`
- Not Chaski (Qwen3.5 lock)
- Not an autonomous agent
- Not a GGUF. Mini GGUFs exist on [`A11OY-MINI`](https://huggingface.co/SZLHOLDINGS/A11OY-MINI); Mini evals none-this-run; Mini does **not** inherit this 3/6

## Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, "SZLHOLDINGS/KHIPU-R2")
```

Owner: Stephen Lutar / SZL Holdings.
