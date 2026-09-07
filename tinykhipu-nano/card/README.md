---
license: apache-2.0
library_name: numpy
tags:
- governed-ai
- szl-holdings
- doctrine-v11
- nano
- synthetic
- needs-loader
- test-fixture
---

> ### How to actually load this — the weights alone are not enough
>
> `tiny_khipu.npz` is a real, honest 4-6-2 MLP produced by a real training run, and the card
> below does not overstate it. But it is a bare NumPy archive with **no
> `config.json` and no loader in this repo**, so `from_pretrained` and the Hub
> inference widget cannot touch it. Nothing here tells you the array names or the
> forward pass.
>
> ```python
> import numpy as np
> from huggingface_hub import hf_hub_download
>
> path = hf_hub_download("SZLHOLDINGS/TinyKhipu-Nano", "tiny_khipu.npz")
> w = np.load(path)
> print(sorted(w.files))   # array names are the de-facto interface
> ```
>
> The forward pass this was trained against lives in the `szl_khipu` package, in
> [SZLHOLDINGS/szl-khipu-kernels](https://huggingface.co/SZLHOLDINGS/szl-khipu-kernels)
> — a **different repository**. Until the loader ships alongside the weights (or a
> `custom_code` handler is added), treat this repo as a **test fixture**, not a
> deployable model. Evidence status: plan_valid 1.00 / abstain 1.00 (SYNTHETIC).

<p align="center">
  <img src="holo-banner.svg" alt="TinyKhipu-Nano — holographic 4-6-2 MLP banner" width="100%"/>
</p>

<h1 align="center">T I N Y K H I P U &nbsp;N A N O</h1>

<p align="center"><em>Four features in. NAVIGATE or ABSTAIN out. Abstain is the default class, not a post-hoc filter.</em></p>

<p align="center">
  <img alt="Params: 4-6-2 MLP on numpy" src="https://img.shields.io/badge/params-4--6--2%20MLP%20%C2%B7%20numpy-9f1239?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/TinyKhipu-Nano?style=flat-square&color=fb7185&label=downloads"/>
  <img alt="Evidence: SYNTHETIC — design fact, not field claim" src="https://img.shields.io/badge/evidence-SYNTHETIC%20%C2%B7%20design%20fact-b45309?style=flat-square"/>
  <img alt="Needs loader: test fixture" src="https://img.shields.io/badge/needs%20loader-test%20fixture-991b1b?style=flat-square"/>
  <img alt="Not 1.5B, not Qwen" src="https://img.shields.io/badge/not%201.5B-not%20Qwen-334155?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

<p align="center">
  <strong>Family.</strong> nano · <strong>Evidence.</strong> SYNTHETIC · <strong>Weights.</strong> numpy · <strong>Params.</strong> 4-6-2 · <strong>Not 1.5B.</strong>
</p>

## The cut

Leaders train models to answer. We train a silhouette to shut up when overlap is thin or the lure is adversarial. This nano is the 4-6-2 silhouette of that cut. Not 1.5B.

A navigator that has never seen document text — only handles — and still knows when to refuse the walk.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Refusal as a typed output, not a polite paragraph. |
| NVIDIA | Guardrail inside the head, not a sidecar. |
| Unsloth | The 1.5B QLoRA is the grown form of this MLP. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Unit-test the NAVIGATE|ABSTAIN schema before GPU spend.

## Bench (this tree)

`TRAINING_RECEIPT.json` seed `20260721` · steps 280 · honesty **REPORTED**

| Metric | Value |
|---|---|
| plan_valid | 1.00 |
| abstain | 1.00 |
| hallucinated | 0 |
| weights | `tiny_khipu.npz` sha256 `cc8d0385b2c75079669df809d7e4823f1ad8d9d535aec511446347490b11dff9` |

Infers on `POST /api/infer {"kind":"tiny_khipu"}`. Hard ID filter. **Not Qwen. Not 1.5B.**

## Limitations

- Synthetic features. Perfect holdout is a design fact, not a field claim.
- The 1.5B abstain rate is 2/6 — this nano does not wash that.

## Honesty

| Claim | Label |
|---|---|
| This card's numbers | SYNTHETIC |
| Energy / joules | UNAVAILABLE unless a signed meter says MEASURED |
| Λ uniqueness | Conjecture 1 OPEN — not a theorem |
| GGUF as the signed object | FALSE |

Doctrine v11 LOCKED · 749 declarations · 14 axioms · 163 sorries · locked-proven 8.

Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr. · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).

---

<p align="center">
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/TinyKhipu-Nano">SZLHOLDINGS/TinyKhipu-Nano</a>
</p>
