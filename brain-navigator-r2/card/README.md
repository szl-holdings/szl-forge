---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- unsloth
- governed-agent
- retrieval
- brain-navigator
- szl-holdings
---

<p align="center">
  <img src="holo-banner.svg" alt="BrainNavigator-R2 — holographic starfield banner" width="100%"/>
</p>

<h1 align="center">B R A I N N A V I G A T O R &nbsp;R2</h1>

<p align="center"><em>575 public handles on the route. 9,464 private nodes never touched.</em></p>

<p align="center">
  <img alt="Base: Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen3.5--0.8B-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/brain-navigator-r2?style=flat-square&color=a78bfa&label=downloads"/>
  <img alt="Recipe: bf16 LoRA r16 a32" src="https://img.shields.io/badge/recipe-bf16%20LoRA%20r16%20%CE%B132-6366f1?style=flat-square"/>
  <img alt="Private graph in gradients: 0 of 9464" src="https://img.shields.io/badge/private%20graph%20in%20gradients-0%20of%209464-22d3ee?style=flat-square"/>
  <img alt="Publication: false until MEASURED generate" src="https://img.shields.io/badge/publication-false%20until%20MEASURED%20generate-b45309?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

Separate 0.8B LoRA SKU. **Does not overwrite**
[`SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator)
or [`SZLHOLDINGS/SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B).

| | |
|---|---|
| Base | `Qwen/Qwen3.5-0.8B` Apache-2.0 |
| Quant | **bf16 LoRA** r=16 α=32. QLoRA forbidden on Qwen3.5. |
| GPU | RTX 5050 Laptop 8GB **Blackwell** |
| Curriculum | synthetic NAVIGATE/ABSTAIN over **575 public handles** |
| Private graph | 9464 nodes admitted to gradients = **0** |
| publication_eligible | **false** until MEASURED generate |
| Λ | Conjecture 1 — never a theorem |

Train loss is not eval. Named-N generate lives in `eval_report.json`.

---

<p align="center">
  Software retrieval hologram: <a href="https://huggingface.co/spaces/SZLHOLDINGS/second-brain">SZLHOLDINGS/second-brain</a><br/>
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/brain-navigator-r2">SZLHOLDINGS/brain-navigator-r2</a>
</p>
