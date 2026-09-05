---
license: apache-2.0
language:
  - en
base_model: SZLHOLDINGS/SZL-Khipu-1.5B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: llama.cpp
tags:
  - gguf
  - ollama
  - llama.cpp
  - qwen2.5
  - governed-agent
  - brain-navigator
  - szl-holdings
  - receipts
---

<!-- SZL-ESTATE-CARD:v2:START -->
<p align="center"><a href="https://a-11-oy.com/"><img src="https://huggingface.co/spaces/SZLHOLDINGS/README/resolve/main/assets/estate-banner-v2.svg" alt="SZL Holdings — governed, receipted, verifiable" width="100%"></a></p>
<p align="center">
  <a href="https://github.com/szl-holdings/.github/tree/main/doctrine"><img src="https://img.shields.io/badge/doctrine-v11%20LOCKED-0B1F3A?style=flat-square" alt="doctrine v11"></a>
  <a href="https://a-11-oy.com/"><img src="https://img.shields.io/badge/evidence%20wall-LIVE%20%C2%B7%20verify%20in%20browser-3AF4C8?style=flat-square" alt="live evidence wall"></a>
  <a href="https://huggingface.co/datasets/SZLHOLDINGS/szl-lake"><img src="https://img.shields.io/badge/szl--lake-offline%20verifiable-C9B787?style=flat-square" alt="szl-lake offline verifiable"></a>
  <a href="https://huggingface.co/spaces/SZLHOLDINGS/holographic"><img src="https://img.shields.io/badge/estate%20map-holographic-5B8DEE?style=flat-square" alt="holographic estate map"></a>
</p>
<p align="center"><sub>Part of the <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> governed estate — claims are designed to carry checkable receipts. Verification proves integrity &amp; origin, never accuracy or performance.</sub></p>
<!-- SZL-ESTATE-CARD:v2:END -->

<p align="center">
  <img src="holo-banner.svg" alt="SZL-Khipu-1.5B-GGUF — holographic house banner" width="100%"/>
</p>

<h1 align="center">K H I P U · G G U F</h1>

<p align="center"><em>Small enough for a laptop, honest enough for an audit.</em></p>

<p align="center">
  <img alt="Format: GGUF" src="https://img.shields.io/badge/format-GGUF-C9B787?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF?style=flat-square&color=fbbf24&label=downloads"/>
  <a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"><img alt="Base: Qwen2.5-1.5B-Instruct" src="https://img.shields.io/badge/base-Qwen2.5--1.5B--Instruct-334155?style=flat-square"/></a>
  <a href="https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B"><img alt="Receipts: signed, in base repo" src="https://img.shields.io/badge/receipts-signed%20in%20base%20repo-3af4c8?style=flat-square"/></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-7e8aa3?style=flat-square"/></a>
  <img alt="Derived artifact — signed hash does not cover GGUF" src="https://img.shields.io/badge/derived-signed%20hash%20does%20not%20cover%20GGUF-b45309?style=flat-square"/>
</p>

<p align="center">
  A compact 1.5B model for governed agent navigation — quantized for everywhere.
</p>

GGUF quantizations of [**SZL-Khipu-1.5B**](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) — a QLoRA fine-tune of Qwen2.5-1.5B-Instruct for governed, grounded-only navigation of the SZL receipt lake.

**Published provenance:** the base model ships with **owner-signed training and
eval receipts**. They are Ed25519 signatures over canonical JSON and chain the
evaluation receipt to the training receipt. This repo carries those receipts
plus the repo-declared public key so you can verify repository-key continuity
and receipt integrity *before* you load a tensor. They are not DSSE envelopes;
the repository does not establish external key provenance, and an owner
signature is not an independent evaluation.

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Leaders treat GGUF as the model. We print on the card: signed hash does not cover these files. CPU-honest, receipt-honest.

Edge navigation that still fails closed, with a card that refuses to launder a quant as a weight.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | No GGUF. We keep the honesty they apply to API vs weights. |
| NVIDIA | TensorRT-LLM is their derived path. GGUF is ours. Same idea, smaller church. |
| Unsloth | Unsloth's GGUF export, labeled derived. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

llama.cpp / Ollama / LM Studio. Still proposal-only.

## Limitations

- Derived. Numerics drift vs BF16.
- Do not cite GGUF as the signed checkpoint.

Canonical GitHub: [`szl-holdings/szl-serve`](https://github.com/szl-holdings/szl-serve/blob/main/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

## Specification

| | |
|---|---|
| **Base model** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **License** | `apache-2.0` |
| **Parameters** | 1.5B |
| **Hardware** | Runs CPU-only via GGUF Q4_K_M (~0.99 GB); GPU optional |
| **One command** | `ollama run hf.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF:Q4_K_M` |

## Quants

| File | Bits | Size | Uploaded-byte SHA-256 (Hub LFS OID) | Use when |
|---|---:|---:|---|---|
| [SZL-Khipu-1.5B-Q4_K_M.gguf](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/main/SZL-Khipu-1.5B-Q4_K_M.gguf) | 4-bit | 0.99 GB | `13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a` | Default - best size/quality balance |
| [SZL-Khipu-1.5B-Q5_K_M.gguf](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/main/SZL-Khipu-1.5B-Q5_K_M.gguf) | 5-bit | 1.13 GB | `3bf460ac163c5dc952c273999c38a41349e3e6d666e4b713aed22c996860fd4c` | More quality headroom |
| [SZL-Khipu-1.5B-Q8_0.gguf](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/main/SZL-Khipu-1.5B-Q8_0.gguf) | 8-bit | 1.65 GB | `6aff1087f64631679f4cdf032613aee6911dbde38cd3bac6b81bf63741a56f0d` | Near-lossless CPU inference |
| [SZL-Khipu-1.5B-F16.gguf](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/main/SZL-Khipu-1.5B-F16.gguf) | 16-bit | 3.09 GB | `2348ee342efe639e100f3fb31a3dc11b8c12d8c43ecfe45e18041b9f94c71a12` | Reference / requantizing |

Chat template (Qwen2.5 ChatML) is embedded in every file.

## Run it

**Ollama**

```bash
ollama run hf.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF:Q4_K_M
```

**llama.cpp**

```bash
llama-cli -hf SZLHOLDINGS/SZL-Khipu-1.5B-GGUF:Q4_K_M -p "Navigate: which receipt signed decision d-42?"
```

**LM Studio** — search `SZLHOLDINGS/SZL-Khipu-1.5B-GGUF`, pick Q4_K_M.

## Prompt contract

The user turn is a single JSON object `{query, candidates:[{nodeId, nodeKind, label, note}]}`
— handles only, never node content — and the model returns a single JSON **plan**
(`decision=NAVIGATE` citing offered handles, or `decision=ABSTAIN` with an
`abstainReason`) per `khipu.schema.json`. The full contract and expected output shape
live on the [BrainNavigator card](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B#quick-start).

## Verify before you trust

```bash
# The owner-signed receipts travel with the weights:
#   training_receipt.signed.json; eval_receipt.signed.json; owner_pubkey.json
# They are Ed25519 signatures over canonical JSON, not DSSE envelopes.
# Verify them offline against the repo-declared public key before use.
```

Quantization: llama.cpp `convert_hf_to_gguf.py` -> `llama-quantize` (F16 -> Q4_K_M / Q5_K_M / Q8_0), 2026-07-15. Quantization changes numerics; the signed evaluation receipt covers the pre-quantized BrainNavigator evaluation artifact described by that receipt, **not** any GGUF. The LFS hashes above bind the exact uploaded GGUF bytes. No post-quantization quality evaluation or independent benchmark is claimed.

---

<p align="center">
  <strong>Governed AI you can prove.</strong><br/>
  <a href="https://a-11-oy.com">a-11-oy.com</a> ·
  <a href="https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B">base model + full card</a> ·
  <a href="https://github.com/szl-holdings/szl-forge">source/harness</a> ·
  <a href="https://huggingface.co/SZLHOLDINGS">SZLHOLDINGS on Hugging Face</a> ·
  <a href="https://szlholdings-szl-estate-live.static.hf.space">Estate hub — live</a>
</p>

<p align="center"><sub>Lambda = Conjecture 1, never green; owner-signed receipts verified against a repo-declared key; no independent benchmark or post-quant evaluation claimed.<br/>SLSA: L1 honest · L2 attested · L3 roadmap. Λ = Conjecture 1 (advisory, never a theorem). Trust ceiling 0.97 — never 100%. Labels honest by default: MEASURED / REPORTED / MODELED / HEURISTIC / UNKNOWN / UNAVAILABLE. locked-proven = exactly 8 {F1,F4,F7,F11,F12,F18,F19,F22}.</sub></p>
