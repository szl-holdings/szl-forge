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

<p align="center">
  <img src="holo-banner.svg" alt="khipu-r3 — holographic cord banner, four knots held and one frayed" width="100%"/>
</p>

<h1 align="center">K H I P U &nbsp;R 3</h1>

<p align="center"><em>The knot that reports its own slipping — abstain 0/6, on the card.</em></p>

<p align="center">
  <img alt="Base: Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen3.5--0.8B-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/khipu-r3?style=flat-square&color=fb7185&label=downloads"/>
  <img alt="Recipe: bf16 LoRA r16 a32" src="https://img.shields.io/badge/recipe-bf16%20LoRA%20r16%20%CE%B132-9f1239?style=flat-square"/>
  <img alt="Abstain gate: 0 of 6 MEASURED" src="https://img.shields.io/badge/abstain%20gate-0%20of%206%20MEASURED-991b1b?style=flat-square"/>
  <img alt="Publication: false" src="https://img.shields.io/badge/publication-false-b45309?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

Laptop Blackwell Unsloth **bf16 LoRA** (r=16 α=32) of the forge `khipu/` curriculum
(`train.jsonl` + `train.abstain.jsonl`, 23 rows) on `Qwen/Qwen3.5-0.8B`.

| Claim | Status |
|---|---|
| Train loss 0.4397 | MEASURED (train metric, not eval) |
| Held-out generate | MEASURED grounding 4/5, abstain 0/6 (NAVIGATE instead of ABSTAIN) |
| publication_eligible | false |
| Overwrites 1.5B / KHIPU-R2 / brain-navigator-r2 | never |
| QLoRA | forbidden on Qwen3.5; this SKU is bf16 LoRA |
| Λ uniqueness | Conjecture 1, never a theorem |

---

<p align="center">
  GitHub source: <a href="https://github.com/szl-holdings/szl-forge/tree/main/khipu">szl-holdings/szl-forge · khipu/</a><br/>
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/khipu-r3">SZLHOLDINGS/khipu-r3</a>
</p>
