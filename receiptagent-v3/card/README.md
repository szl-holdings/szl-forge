---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - qwen3.5
  - peft
  - lora
  - unsloth
  - governed-ai
  - receipt-agent
  - szl-holdings
---

> **NON-RELEASE / PLACEHOLDER.** Not a ReceiptAgent tournament winner.
> `publication_eligible=false`. `autonomy_eligible=false`.
> Do not treat this ID as flagship.

<p align="center">
  <img src="holo-banner.svg" alt="receiptagent-v3 — holographic stamp-seal banner" width="100%"/>
</p>

<h1 align="center">R E C E I P T A G E N T &nbsp;v3</h1>

<p align="center"><em>Proposal-only receipt drafts. The stamp that never mints.</em></p>

<p align="center">
  <img alt="Base: Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen3.5--0.8B-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3?style=flat-square&color=7e8aa3&label=downloads"/>
  <img alt="Recipe: bf16 LoRA r16 a32" src="https://img.shields.io/badge/recipe-bf16%20LoRA%20r16%20%CE%B132-cbd5e1?style=flat-square"/>
  <img alt="Status: non-release placeholder" src="https://img.shields.io/badge/status-non--release%20placeholder-991b1b?style=flat-square"/>
  <img alt="Never overwrites v2 or ReceiptAgent" src="https://img.shields.io/badge/never%20overwrites-v2%20%C2%B7%20ReceiptAgent-be123c?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

Laptop Blackwell Unsloth **bf16 LoRA** (r=16 α=32) on `Qwen/Qwen3.5-0.8B`.
Separate SKU from v2. **Does not overwrite**
`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2` or
`SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent`.

| Claim | Status |
|---|---|
| Train loss 0.0537 | MEASURED (train metric, not eval) |
| Training rows | 180 (committed v3 `train.jsonl`) |
| Dataset SHA256 | `ad30ce9b478eff50d78064f387143b750be3f746b459f98f7961784b9ce1081f` |
| Adapter SHA256 | `339efa6ea0ed0719db0dede26ee836f290048deb1b6ce68403ab07cb255b28d0` |
| Held-out generate / 1024 eval | DEV n=12 MEASURED 12/12 (4 draft / 4 recovery / 4 refuse). `test.jsonl` never opened |
| QLoRA | forbidden on Qwen3.5; this SKU is bf16 LoRA (`load_in_4bit=false`) |
| Rosie WSL 4bit supervised launch | not this card |
| publication_eligible | false |
| autonomy_eligible | false |
| Λ uniqueness | Conjecture 1, never a theorem |

## Artifact truth card

| Field | Classification |
|---|---|
| Artifact | Trained LoRA adapter weights for `Qwen/Qwen3.5-0.8B`, not a from-scratch model. |
| Value | Proposal-only receipt drafts. Validation, approval, execution, and minting stay outside the weights. |
| Evidence | Train loss MEASURED on the 180-row committed split. No held-out eval on this card. |
| Limits | Small synthetic curriculum. No broad capability, factuality, safety, or third-party benchmark claim. |
| Runtime | Drafts only. Not an autonomous agent. |

---

<p align="center">
  GitHub source: <a href="https://github.com/szl-holdings/szl-forge/tree/main/frontier/qwen35-receiptagent-v3">frontier/qwen35-receiptagent-v3</a><br/>
  Hardware: NVIDIA GeForce RTX 5050 Laptop GPU 8GB, torch 2.10.0+cu128, Unsloth 2026.7.2, Windows.<br/>
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3">SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3</a>
</p>
