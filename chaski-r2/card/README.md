---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: peft
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
tags:
- base_model:adapter:Qwen/Qwen3.5-0.8B
- lora
- sft
- transformers
- trl
- unsloth
- proposal-only
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  sku: CHASKI-R2
  quant: bf16-lora
  qlora: false
  weights: AVAILABLE
  evals: none-this-run
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
---

<p align="center">
  <img src="holo-banner.svg" alt="Chaski-R2 — holographic knot banner" width="100%"/>
</p>

<h1 align="center">C H A S K I · R 2</h1>

<p align="center"><em>A lineage you can walk. R1 stays up. R2 is the next knot.</em></p>

<p align="center">
  <img alt="Base: Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen3.5--0.8B-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/chaski-r2?style=flat-square&color=f472b6&label=downloads"/>
  <img alt="Artifact: bf16 LoRA r=16 a=32" src="https://img.shields.io/badge/artifact-bf16%20LoRA%20r16%20a32-8b5cf6?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
  <img alt="Evals: none this run" src="https://img.shields.io/badge/evals-none%20this%20run-b45309?style=flat-square"/>
  <a href="https://huggingface.co/SZLHOLDINGS/chaski"><img alt="Lineage: never overwrites SZLHOLDINGS/chaski" src="https://img.shields.io/badge/lineage-never%20overwrites%20R1-fda4af?style=flat-square"/></a>
</p>

Owner-GPU recut on NVIDIA GeForce RTX 5050 Laptop (8GB). Original SZL cut of
disclosed Apache `Qwen/Qwen3.5-0.8B`. **Not QLoRA.** Unsloth 2026-08 does not
recommend QLoRA on Qwen3.5 (dense or MoE) because of higher-than-normal
quantization differences.

This is a **separate SKU**. It does **not** overwrite live `SZLHOLDINGS/chaski`
and is **not** `SZLHOLDINGS/chaski-5050` (that kit is r=16 α=16 on doctrine
SFT). This SKU is r=16 α=32 on `chaski_r2/train.jsonl` only.

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Round-2 is a first-class citizen in this estate. We do not overwrite R1. We add a sibling.

A lineage you can walk. R1 stays up. R2 is the next knot.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Versioned constitutions. |
| NVIDIA | Recipe rerun. |
| Unsloth | Another FastLanguageModel job. |

This adapter remains a research proposal artifact; no qualification pass is claimed.

## Intended use

Lineage walk. Compare, do not silently replace.

## Limitations

- proposal-only

Canonical GitHub: [`chaski-r2/card/README.md`](https://github.com/szl-holdings/szl-forge/blob/main/chaski-r2/card/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

## Honest status

| | |
|---|---|
| **Base** | `Qwen/Qwen3.5-0.8B` |
| **Method** | Unsloth bf16 LoRA (`load_in_4bit=False`, `load_in_16bit=True`) |
| **LoRA** | r=16, α=32, seed 11, response-only CE |
| **Dataset** | `chaski_r2/train.jsonl` (32 rows). Named-N gates held out of gradients. |
| **Epochs / steps** | 3 epochs, batch 1, grad accum 2 |
| **Train loss** | MEASURED `0.7656` — train metric, **not an eval** |
| **Train runtime** | MEASURED on RTX 5050 Laptop 8GB |
| **Adapter sha256** | `440340ce29e19344c0625d0adfe820b277cdb0e24099d4e612f88ad6b3cf49c6` |
| **Evals** | none-this-run; train loss is not a JSON-draft/refusal evaluation |
| **publication_eligible** | false; published adapter bytes do not establish qualification |
| **Jobs** | local-5050 owner metal; HF Jobs not fired from the GitHub kit |
| **Ollama / llama-server** | `llama-server` is missing. No tok/s claimed. |

Train loss is not a JSON-draft or refusal gate. Not 5/5 or 6/6. Lab load
forbidden. House CPU lab stays signed Khipu GGUF.

### Framework versions

- PEFT 0.19.1
