---
license: apache-2.0
library_name: numpy
tags:
- governed-ai
- khipu
- szl-holdings
- embedding
- silhouette
- needs-loader
- test-fixture
---

> ### How to actually load this — the weights alone are not enough
>
> `mini_embed.npz` is a real, honest hash + table, V=64 d=12 produced by a real training run, and the card
> below does not overstate it. But it is a bare NumPy archive with **no
> `config.json` and no loader in this repo**, so `from_pretrained` and the Hub
> inference widget cannot touch it. Nothing here tells you the array names or the
> forward pass.
>
> ```python
> import numpy as np
> from huggingface_hub import hf_hub_download
>
> path = hf_hub_download("SZLHOLDINGS/MiniEmbed-Nano", "mini_embed.npz")
> w = np.load(path)
> print(sorted(w.files))   # array names are the de-facto interface
> ```
>
> The forward pass this was trained against lives in the `szl_khipu` package, in
> [SZLHOLDINGS/szl-khipu-kernels](https://huggingface.co/SZLHOLDINGS/szl-khipu-kernels)
> — a **different repository**. Until the loader ships alongside the weights (or a
> `custom_code` handler is added), treat this repo as a **test fixture**, not a
> deployable model. Evidence status: no retrieval/analogy score — UNAVAILABLE by design.

<p align="center">
  <img src="holo-banner.svg" alt="MiniEmbed-Nano — holographic hash-table banner" width="100%"/>
</p>

<h1 align="center">M I N I E M B E D &nbsp;N A N O</h1>

<p align="center"><em>Tiny hash+table embed: V=64, d=12, L2-normalized rows. Not a foundation embed. Not neural. Not MiniEmbed 3290×128.</em></p>

<p align="center">
  <img alt="Table: V=64 d=12 hash+table" src="https://img.shields.io/badge/table-V%3D64%20d%3D12%20hash%2Btable-4c1d95?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/MiniEmbed-Nano?style=flat-square&color=a78bfa&label=downloads"/>
  <img alt="Retrieval / analogy score: UNAVAILABLE by design" src="https://img.shields.io/badge/retrieval%20%C2%B7%20analogy-UNAVAILABLE%20by%20design-b45309?style=flat-square"/>
  <img alt="Needs loader: test fixture" src="https://img.shields.io/badge/needs%20loader-test%20fixture-991b1b?style=flat-square"/>
  <img alt="Not the 3290×128 table" src="https://img.shields.io/badge/not%20the%203290%C3%97128%20table-a%20different%20artifact-334155?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

Canonical source: [szl-holdings/szl-khipu](https://github.com/szl-holdings/szl-khipu)  
Sibling card: [SZLHOLDINGS/szl-khipu](https://huggingface.co/SZLHOLDINGS/szl-khipu)  
The larger statistical MiniEmbed (3290 × 128) lives on [SZLHOLDINGS/szl-kernels](https://huggingface.co/SZLHOLDINGS/szl-kernels) — a different artifact. Do not mix them.

```python
from szl_khipu.train import mini_embed

emb = mini_embed.build(seed=20260721)
vec = emb.embed("knot the run")
print(emb.V, emb.D, vec.shape)
# 64 12 (12,)
emb.save_npz("mini_embed.npz")
```

## What it does

- SHA-256 token id modulo 64. Mean-pool then L2. Deterministic given seed.
- Built here on CPU NumPy. Honesty **REPORTED**. Energy **UNAVAILABLE**.
- No analogy score. No retrieval score. No SVD variance claim (that belongs to the 3290×128 table).

## Bench (this tree)

`TRAINING_RECEIPT.json` seed `20260721` · honesty **REPORTED**

| Metric | Value |
|---|---|
| V×d | 64 × 12 |
| method | hash+table L2 |
| weights | `mini_embed.npz` sha256 `ae31a3a7214d1f142d8ea3f4f86c35bdedd7c108bc5d04ea00c87e7b674e6e3b` |

Infers on `POST /api/infer {"kind":"mini_embed","token":"F18"}`. **Not neural. Not 3290×128.**

## What it is NOT

- **Not** the [SZLHOLDINGS/szl-kernels](https://huggingface.co/SZLHOLDINGS/szl-kernels) MiniEmbed (3290 × 128, SVD var 0.3146).
- **Not neural. Not word2vec. Not a foundation embed.**
- **Not 1.5B. Not Qwen.**
- **Not proven trust.** Λ uniqueness remains Conjecture 1 OPEN.
- Energy **UNAVAILABLE**. CUDA **UNAVAILABLE**. Never a fabricated joule.

## Honesty

| Claim | Label | What-NOT |
|---|---|---|
| Table built in this package | REPORTED | V=64 d=12, not 3290×128 |
| Neural / trained embed | FALSE | hash+table, not SGD |
| Analogy / retrieval score | UNAVAILABLE | not measured |
| Energy | UNAVAILABLE | never a fabricated joule |
| CUDA | UNAVAILABLE | CPU numpy LIVE |

Doctrine v11 LOCKED · 749/14/163 · locked-proven 8. Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr. · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).

---

<p align="center">
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/MiniEmbed-Nano">SZLHOLDINGS/MiniEmbed-Nano</a>
</p>
