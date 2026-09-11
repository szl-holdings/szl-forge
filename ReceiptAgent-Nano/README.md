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
> `receipt_agent.npz` is a real, honest 4-10-4 MLP produced by a real training run, and the card
> below does not overstate it. But it is a bare NumPy archive with **no
> `config.json` and no loader in this repo**, so `from_pretrained` and the Hub
> inference widget cannot touch it. Nothing here tells you the array names or the
> forward pass.
>
> ```python
> import numpy as np
> from huggingface_hub import hf_hub_download
>
> path = hf_hub_download("SZLHOLDINGS/ReceiptAgent-Nano", "receipt_agent.npz")
> w = np.load(path)
> print(sorted(w.files))   # array names are the de-facto interface
> ```
>
> The forward pass this was trained against lives in the `szl_khipu` package, in
> [SZLHOLDINGS/szl-khipu-kernels](https://huggingface.co/SZLHOLDINGS/szl-khipu-kernels)
> — a **different repository**. Until the loader ships alongside the weights (or a
> `custom_code` handler is added), treat this repo as a **test fixture**, not a
> deployable model. Evidence status: see TRAINING_RECEIPT.json / BENCH.*.json (SYNTHETIC).

> **Now loadable:** [`load.py`](https://github.com/szl-holdings/szl-forge/blob/main/ReceiptAgent-Nano/load.py) ships the forward pass — tanh hidden layer, softmax over the four gates, and any read under 0.5 confidence returns ESCALATE. `python load.py 0.1,0.2,0.3,0.4` prints a gate.

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/receiptagent-nano/card/holo-banner.svg" alt="ReceiptAgent-Nano — holographic 4-10-4 four-gate banner" width="100%"/>
</p>

<h1 align="center">R E C E I P T A G E N T &nbsp;N A N O</h1>

<p align="center"><em>ALLOW · DENY · ABSTAIN · ESCALATE. Escalation is a class, not a retry loop.</em></p>

<p align="center">
  <img alt="Params: 4-10-4 MLP on numpy" src="https://img.shields.io/badge/params-4--10--4%20MLP%20%C2%B7%20numpy-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/ReceiptAgent-Nano?style=flat-square&color=94a3b8&label=downloads"/>
  <img alt="Held-out agree vs rule_check: 0.905 REPORTED" src="https://img.shields.io/badge/held--out%20agree%20vs%20rule__check-0.905%20REPORTED-7e8aa3?style=flat-square"/>
  <img alt="Needs loader: test fixture" src="https://img.shields.io/badge/needs%20loader-test%20fixture-991b1b?style=flat-square"/>
  <img alt="Kernel is truth" src="https://img.shields.io/badge/kernel%20is%20truth-four%20gates-e2e8f0?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

<p align="center">
  <strong>Family.</strong> nano · <strong>Evidence.</strong> SYNTHETIC · <strong>Weights.</strong> numpy · <strong>Params.</strong> 4-10-4
</p>

## The cut

Anthropic refuses. NVIDIA rails. Unsloth trains. We add a fourth way: hand the decision to a human with a receipt. Loop-tax lives here.

A policy head that cannot silently succeed. Every output is one of four named gates.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Constitutional refuse → typed DENY/ABSTAIN. |
| NVIDIA | NeMo Guardrails flow → four-way head. |
| Unsloth | Grown form is SZL-Forge-1.5B-ReceiptAgent. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Fail-closed unit tests for the 4-way gate.

## Bench (this tree)

`TRAINING_RECEIPT.json` seed `20260721` · honesty **REPORTED** · kernel is truth

| Metric | Value |
|---|---|
| held-out agree vs rule_check | 0.905 |
| weights | `receipt_agent.npz` sha256 `8aca4d24c90d6159cbb2bb885c7a94822d715899437f58abe69fb5c9664a1381` |

Infers on `POST /api/infer {"kind":"receipt_agent"}`. Surrogate may disagree. Kernel wins. **Not 1.5B.**

## Limitations

- Synthetic 4-D features. Not a substitute for the 1.5B agent.
- Hub kernel labels are ALLOW/WARN/BLOCKED/ESCALATE — kernel is truth. This atelier MLP is a 4-class silhouette, not rule_check.

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
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/ReceiptAgent-Nano">SZLHOLDINGS/ReceiptAgent-Nano</a>
</p>
