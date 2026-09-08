---
license: apache-2.0
library_name: numpy
tags:
- governed-ai
- khipu
- szl-holdings
- moons
- mlp
- silhouette
- needs-loader
- test-fixture
---

> ### How to actually load this — the weights alone are not enough
>
> `moons.npz` is a real, honest 2-8-2 MLP produced by a real training run, and the card
> below does not overstate it. But it is a bare NumPy archive with **no
> `config.json` and no loader in this repo**, so `from_pretrained` and the Hub
> inference widget cannot touch it. Nothing here tells you the array names or the
> forward pass.
>
> ```python
> import numpy as np
> from huggingface_hub import hf_hub_download
>
> path = hf_hub_download("SZLHOLDINGS/Moons-Nano", "moons.npz")
> w = np.load(path)
> print(sorted(w.files))   # array names are the de-facto interface
> ```
>
> The forward pass this was trained against lives in the `szl_khipu` package, in
> [SZLHOLDINGS/szl-khipu-kernels](https://huggingface.co/SZLHOLDINGS/szl-khipu-kernels)
> — a **different repository**. Until the loader ships alongside the weights (or a
> `custom_code` handler is added), treat this repo as a **test fixture**, not a
> deployable model. Evidence status: acc 0.93 / loss 0.13 REPORTED on the TRAIN set.

<p align="center">
  <img src="holo-banner.svg" alt="Moons-Nano — holographic two-moons banner" width="100%"/>
</p>

<h1 align="center">M O O N S &nbsp;N A N O</h1>

<p align="center"><em>Two-moons 2→8→2 tanh-softmax SGD. A few hundred floats. Not 1.5B. Not Qwen. Not a foundation model.</em></p>

<p align="center">
  <img alt="Params: 2-8-2 MLP on numpy" src="https://img.shields.io/badge/params-2--8--2%20MLP%20%C2%B7%20numpy-0e7490?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/Moons-Nano?style=flat-square&color=22d3ee&label=downloads"/>
  <img alt="Accuracy: 0.93 REPORTED on train" src="https://img.shields.io/badge/accuracy-0.93%20REPORTED%20on%20train-b45309?style=flat-square"/>
  <img alt="Needs loader: test fixture" src="https://img.shields.io/badge/needs%20loader-test%20fixture-991b1b?style=flat-square"/>
  <img alt="Energy: UNAVAILABLE — never a fabricated joule" src="https://img.shields.io/badge/energy-UNAVAILABLE-334155?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

Canonical source: [szl-holdings/szl-khipu](https://github.com/szl-holdings/szl-khipu)  
Sibling card: [SZLHOLDINGS/szl-khipu](https://huggingface.co/SZLHOLDINGS/szl-khipu)

```python
from szl_khipu.train import moons

weights, ev = moons.train(seed=20260721, steps=400)
print(ev["acc"], ev["loss"])
# REPORTED: acc 0.93 · loss ~0.13 on the training moons
moons.save_npz("moons.npz", weights)
```

## What it does

- Classic two-moons toy classification. Hidden width 8. Softmax over 2.
- Trained here on CPU NumPy. Honesty **REPORTED**. Energy **UNAVAILABLE**.

## Bench (this tree)

`TRAINING_RECEIPT.json` seed `20260721` · steps 400 · honesty **REPORTED**

| Metric | Value |
|---|---|
| acc | 0.93 |
| loss | ~0.13 |
| weights | `moons.npz` sha256 `dda50e3b293534de3f5aec01ebf9f8d6688e06069931618dfd35f01369904104` |

Infers on `POST /api/infer {"kind":"moons","x":0.2,"y":0.3}`. **Not 1.5B. Not a published benchmark.**

## What it is NOT

- **Not SZL-Khipu-1.5B.** Not QLoRA. Not a chat model.
- **Not sklearn moons as a product claim.** A live silhouette so the estate has a TRAINED tiny MLP that actually ran.
- **Not proven trust.** Λ uniqueness remains Conjecture 1 OPEN.
- Energy **UNAVAILABLE**. CUDA **UNAVAILABLE**. Never a fabricated joule.

## Honesty

| Claim | Label | What-NOT |
|---|---|---|
| Weights trained in this package | REPORTED | silhouette, Not 1.5B |
| acc 0.93 on the training moons | REPORTED | not a published benchmark |
| Λ | ADVISORY · Conjecture 1 OPEN | never a theorem |
| Energy | UNAVAILABLE | never a fabricated joule |
| CUDA | UNAVAILABLE | CPU numpy LIVE |

Doctrine v11 LOCKED · 749/14/163 · locked-proven 8. Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr. · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).

---

<p align="center">
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/Moons-Nano">SZLHOLDINGS/Moons-Nano</a>
</p>
