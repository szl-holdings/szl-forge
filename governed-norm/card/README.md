---
tags:
- kernel
- software
- normalization
- rmsnorm
- layernorm
- provenance
- governance
- compatibility
- surrogate
- deprecated
- doi:10.5281/zenodo.19944926
library_name: kernels
license: apache-2.0
---

> ### CORRECTION 2026-08-30 — the surrogate weights are NOT in this repo
>
> A callout below stated that this repository "also ships" `model.joblib` with MEASURED
> fidelity **0.8632**. It is not here — every one returns **HTTP 404** on
> `resolve/main`. `MODEL_PROVENANCE.json` also asserted `trained_weights_present: true`
> with a sha256 for the missing file; that attestation has been corrected in the same
> commit and now reads `false`, with the digest retained as the *expected* value for
> when the weights are pushed.
>
> What remains true: the kernel is real, `get_kernel` is import-LIVE, and
> `TRAINING_RECEIPT.json` documents a genuine training run, so the fidelity figures
> keep their provenance. What was false: the claim that the resulting artifact is
> downloadable from this repo. **Do not build against the surrogate here — there is
> nothing to load.** The kernel was always the declared ground truth; that part of the
> card was correct and is unchanged.

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **FAIL**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **FAIL** | MEASURED 2026-08-29T15:53:59Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `6 failed, 170 passed, 14 warnings in 9.18s`. Failed nodes: `tests/test_hardening.py::test_rms_norm_fullgraph_compile[False]; tests/test_hardening.py::test_rms_norm_fullgraph_compile[True]; tests/test_hardening.py::test_layer_norm_fullgraph_compile; tests/test_hardening.py::test_fused_fullgraph_compile; tests/test_hardening.py::test_governed_compile_does_not_record_but_numerics_match; tests/test_kernel_core.py::test_ops_torch_compile`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-governed-norm", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `0.316227766016838`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-governed-norm`](https://github.com/szl-holdings/szl-governed-norm) @ `c68d06d35058542ea77dc6bad4c8bde2361cc16a`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-governed-norm", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->


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

<!-- SZL-ARTIFACT-NOTICE:v1:START — honesty plate: repo semantics, no fake model tags. -->
> **🟥 Kernel real; surrogate weights NOT PUBLISHED — see the correction at the top.** The governed-normalization kernel (pure-torch, correctness-verified RMSNorm/LayerNorm + SHA3-256 receipt chain) is UNCHANGED and remains the sole ground truth. Since **surrogate v1** this repo was described as shipping (IT DOES NOT — 404) `model.joblib` — a real trained sklearn classifier that triages which norm-violation class a full kernel replay would assign, with **MEASURED** fidelity **0.8632** (agreement vs the kernel on a held-out split). Receipt/chain/eps violations are caught at **100%** recall; numeric correctness is a measured blind spot (see the table). The surrogate never replaces the kernel's reference recompute or `ReceiptChain.verify()`. **Λ = Conjecture 1 · ADVISORY.**
<!-- SZL-ARTIFACT-NOTICE:v1:END -->

<p align="center">
  <img src="holo-banner.svg" alt="szl-governed-norm — holographic house banner" width="100%"/>
</p>

<h1 align="center">G O V E R N E D · N O R M</h1>

<p align="center"><em>A norm layer whose version cannot drift silently.</em></p>

<p align="center">
  <a href="https://huggingface.co/SZLHOLDINGS/szl-governed-norm"><img alt="Lifecycle: compatibility-only" src="https://img.shields.io/badge/lifecycle-compatibility--only-7e8aa3?style=flat-square"/></a>
  <a href="https://huggingface.co/SZLHOLDINGS/szl-governed-norm/blob/main/MODEL_PROVENANCE.json"><img alt="Provenance: MODEL_PROVENANCE.json" src="https://img.shields.io/badge/provenance-MODEL__PROVENANCE.json-3af4c8?style=flat-square"/></a>
  <a href="https://huggingface.co/SZLHOLDINGS/szl-lambda-gate"><img alt="Successor: szl-lambda-gate" src="https://img.shields.io/badge/successor-szl--lambda--gate-d7b96b?style=flat-square"/></a>
  <img alt="Kernel Hub: import-LIVE" src="https://img.shields.io/badge/Kernel%20Hub-import--LIVE-3af4c8?style=flat-square"/>
  <img alt="tests: FAIL (6 failed, 170 passed, Windows Blackwell)" src="https://img.shields.io/badge/tests-FAIL%20%C2%B7%20import--LIVE-dc2626?style=flat-square"/>
  <a href="https://doi.org/10.5281/zenodo.19944926"><img alt="DOI: 10.5281/zenodo.19944926" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19944926-5B8DEE?style=flat-square"/></a>
</p>

**SOFTWARE/KERNEL.** Not a trained model. pipeline_tag is not tabular-classification. Any sklearn surrogate is optional compatibility, not this kernel.

> **Kernel Hub migration (verified 2026-07-15):** `get_kernel(...)` now resolves
> the matching first-class [Kernel Hub repository](https://huggingface.co/kernels/SZLHOLDINGS/szl-governed-norm).
> Its `main` and stable `v1` refs both pin verified revision
> `fe16433d44be03177167e8355c43a4bfdc63e03e`. This model-type repository is
> retained as the legacy source/card mirror; the compatibility-only lifecycle
> documented below is unchanged.

> **Lifecycle: DEPRECATED FOR NEW ADOPTION · COMPATIBILITY-ONLY · ARTIFACT RETAINED**
>

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Leaders delete the old kernel. We stamp deprecated and leave the receipt. History is part of governance.

A norm layer whose version cannot drift silently.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Honest deprecation. |
| NVIDIA | RMSNorm kernel discipline. |
| Unsloth | No. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Do not start new work here. Read the stamp.

## Limitations

- Deprecated.
- Compatibility surrogate.

Canonical GitHub: [`szl-holdings/szl-khipu`](https://github.com/szl-holdings/szl-khipu/blob/main/szl_khipu/governed_norm.py)
<!-- SZL-ATELIER-CUT:v1:END -->

> Canonical development has moved to [`SZLHOLDINGS/szl-lambda-gate`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate). The normalization implementation and receipt chain were folded into the successor's GitHub source under [`szl_lambda_gate.governed_norm`](https://github.com/szl-holdings/szl-lambda-gate/tree/d3a91edbe2595bac1ead1007963b4b7b8857eb19/torch-ext/szl_lambda_gate/governed_norm). Nothing in this repository has been deleted, renamed, or hidden; it remains a reversible compatibility artifact for existing callers.
>
> **Migration gate:** the current immutable Hub successor revision [`fb0481cfc4046c52898aa83a96b1c118a6372f1a`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate/tree/fb0481cfc4046c52898aa83a96b1c118a6372f1a) does **not** yet publish the `governed_norm` compatibility subtree. It is therefore **not a drop-in Hub replacement yet**. Existing production callers should pin this legacy artifact at [`27faddd262c6ee36d08aad9ae234595d75a999f1`](https://huggingface.co/SZLHOLDINGS/szl-governed-norm/tree/27faddd262c6ee36d08aad9ae234595d75a999f1) until a successor revision containing the folded package is published and independently verified.

**Compatibility kernel retained on the Hugging Face Kernel Hub.** Correctness-verified RMSNorm & LayerNorm with optional governance receipts that make calls auditable at the kernel layer. (v0.2.0)

### Immutable evidence

| Evidence | Immutable reference | What it proves |
|---|---|---|
| Legacy Hub artifact | [`27faddd262c6ee36d08aad9ae234595d75a999f1`](https://huggingface.co/SZLHOLDINGS/szl-governed-norm/tree/27faddd262c6ee36d08aad9ae234595d75a999f1) | Retained compatibility package and exact pre-lifecycle-card bytes. |
| Legacy GitHub source | [`3ef27eb7ebf491b0a6ce69be170ecef4c37885a2`](https://github.com/szl-holdings/szl-governed-norm/tree/3ef27eb7ebf491b0a6ce69be170ecef4c37885a2) | Source-side deprecation notice, migration map, tests, and kernel implementation. |
| Successor GitHub source | [`d3a91edbe2595bac1ead1007963b4b7b8857eb19`](https://github.com/szl-holdings/szl-lambda-gate/tree/d3a91edbe2595bac1ead1007963b4b7b8857eb19) | Folded `szl_lambda_gate.governed_norm` source and successor tests. |
| Current successor Hub artifact | [`fb0481cfc4046c52898aa83a96b1c118a6372f1a`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate/tree/fb0481cfc4046c52898aa83a96b1c118a6372f1a) | Current published successor contents; compatibility subtree is not present yet. |

Lifecycle labels describe support and consolidation, not model quality: this repository contains a pure-PyTorch kernel and **no trained weights**.

> Most Kernel Hub kernels compete on raw speed. `szl-governed-norm` opens a different axis: **verifiable provenance**. Same clean `get_kernel` one-liner, plus a SHA3-256 hash-chained audit trail no other kernel ships.

A universal (pure-PyTorch) normalization kernel from [SZL Holdings](https://huggingface.co/SZLHOLDINGS). It gives you a trustworthy reference implementation of RMSNorm and LayerNorm that runs on CPU and CUDA and plays nicely with `torch.compile` — plus an opt-in *governed* mode that emits content-addressed, SHA3-256 hash-chained receipts of each normalization call.

---

## What it is

`szl-governed-norm` is a [Kernel Hub](https://huggingface.co/docs/kernels) kernel built for two things people actually need from a normalization layer:

1. **A correctness reference you can trust.** RMSNorm and LayerNorm are implemented in pure PyTorch, computed in float32 for numerical stability and cast back to the input dtype (the standard Llama-style convention). They are verified against PyTorch's own references in the test suite.
2. **Provenance you can verify.** Run any call with `governed=True` and the kernel records a small, deterministic receipt — input shape/dtype, `eps`, and a SHA3-256 digest of the (rounded) output — hash-chained to the previous receipt. The result is an independently re-walkable audit trail for a sequence of kernel calls.

This is a **universal kernel**: it ships no hand-tuned CUDA/Triton binary. Its differentiator is verifiable governance, not raw FLOPs.

---

## Quickstart

```bash
pip install kernels torch
```

```python
import torch
from kernels import get_kernel

# Current `kernels` (>=0.15) requires an explicit revision/version + trust flag for org kernels:
# Compatibility use only: pin the immutable retained artifact.
gn = get_kernel(
    "SZLHOLDINGS/szl-governed-norm",
    revision="27faddd262c6ee36d08aad9ae234595d75a999f1",
    trust_remote_code=True,
)

print(gn.__version__)        # "0.2.0"
print(gn.selfcheck())        # one-shot correctness + receipt verification

x = torch.randn(4, 1024, dtype=torch.float16, device="cuda")
w = torch.ones(1024, dtype=torch.float16, device="cuda")

# Plain path — drop-in normalization.
y = gn.rms_norm(x, weight=w, eps=1e-6)
z = gn.layer_norm(x, weight=w, eps=1e-5)
```

### Governed mode + receipts

```python
# Same math, plus an audit receipt.
y = gn.rms_norm(x, weight=w, eps=1e-6, governed=True)

print(gn.receipt_head())     # SHA3-256 head over all governed calls
print(gn.receipt_verify())   # {'ok': True, 'depth': 1, 'first_break_seq': -1, 'head': '...'}

# Per-call chain (no global state — ideal for concurrent threads/requests):
chain = gn.ReceiptChain()
y = gn.rms_norm(x, weight=w, eps=1e-6, chain=chain)
print(chain.verify())        # (ok, depth, first_break_seq)
```

Governance is strictly opt-in: with `governed=False` (the default) nothing is recorded, and the kernel never writes to disk or the network.

---

## API reference

### Functional API

| Function | Signature | Notes |
|---|---|---|
| `rms_norm` | `rms_norm(x, weight=None, eps=1e-6, governed=False, chain=None)` | RMSNorm over the last dim. Emits a receipt when `governed=True` or a `chain` is passed. |
| `layer_norm` | `layer_norm(x, weight=None, bias=None, eps=1e-5, governed=False, chain=None)` | LayerNorm over the last dim. |
| `fused_add_rms_norm` | `fused_add_rms_norm(x, residual, weight=None, eps=1e-6, governed=False, chain=None)` | Residual-add + RMSNorm (pre-norm transformer block). Returns `(y, new_residual)`. |
| `selfcheck` | `selfcheck()` | One-shot correctness + governance check; returns a JSON-able dict, never raises. |

All compute in float32 and cast back to the input dtype. `rms_norm` matches a Llama-style RMSNorm reference; `layer_norm` matches `torch.nn.functional.layer_norm` for the last-dim case (verified in `tests/`, 165 passing).

### Governance receipt API

| Function | Returns | Description |
|---|---|---|
| `receipt_head()` | `str` | SHA3-256 head of the default receipt chain (`"0"*64` if empty). |
| `receipt_count()` | `int` | Number of governed calls recorded on the default chain. |
| `receipt_tail(n=10)` | `list[dict]` | The last `n` receipts. |
| `receipt_verify()` | `dict` | Re-walks the chain; returns `{ok, depth, first_break_seq, head}`. |
| `ReceiptChain` | class | Construct your own isolated chain (`emit`, `head`, `count`, `tail`, `verify`). |

### `nn.Module` layers (for the `kernels` layer-mapping mechanism)

Pure `torch.nn.Module` subclasses (only `forward`, no custom `__init__`, no class variables) so they drop in over an existing module:

| Layer | Reads from host module |
|---|---|
| `RMSNorm` | `self.weight` (optional), `self.variance_epsilon` or `self.eps` |
| `LayerNorm` | `self.weight`/`self.bias` (optional), `self.eps` |
| `FusedAddRMSNorm` | `self.weight` (optional), `self.variance_epsilon` or `self.eps` |

---

## Governed mode — provenance at the kernel layer

When a call runs in governed mode, the kernel builds a receipt body, takes a **SHA3-256 digest over its canonical JSON**, and links each receipt to the previous one via a `prev` field — a classic hash chain:

```json
{
  "seq": 0, "op": "rms_norm", "in_shape": [4, 1024], "in_dtype": "float16",
  "eps": 1e-06, "out_digest": "<sha3-256 of the rounded output>", "prev": "<prev digest or 64 zeros>"
}
```

`receipt_verify()` re-walks the chain and reports the first break, so tampering with any receipt invalidates everything downstream. This is the same **provenance doctrine** SZL Holdings applies across its [a11oy governed-AI platform](https://a-11-oy.com) — applied here at the lowest layer of the stack, the kernel itself.

---

## Correctness & honesty

- **Universal, pure-Python kernel — a correctness reference**, verified against PyTorch's own references (165 passing tests).
- **Runs on CPU and CUDA**, `torch.compile(fullgraph=True)`-compatible. Under compile, governed numerics are unchanged but receipt emission (an eager byte-hashing side effect) is skipped — govern at the eager audit boundary.
- **No fabricated benchmarks.** This is not a hand-tuned CUDA/Triton binary; we make **no speedup claims**.
- **The receipt digest is an integrity fingerprint, NOT a cryptographic signature.** It proves a receipt sequence is internally consistent and untampered — not authorship. DSSE signing is a separate, out-of-band concern.
- **Governance is opt-in and side-effect-free by default.**

---

## Compatibility

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | `torch>=2.5` |
| Dependencies | Python standard library + `torch` only |

---

## Interactive demo

> **Live demos (in-browser, nothing to install)** — [`governed-norm-holo`](https://szlholdings-governed-norm-holo.static.hf.space) (this kernel's holographic receipt-chain demo) · [`receipt-chain-live`](https://szlholdings-receipt-chain-live.static.hf.space) (receipt-chain walk-through) · [`szl-kernels-live`](https://szlholdings-szl-kernels-live.static.hf.space) (unified suite demo).
>
> In the meantime, the quickstart above runs fully locally in any Python environment. See the [szl-kernels model card](https://huggingface.co/SZLHOLDINGS/szl-kernels) for the full kernel suite.

---

## Trained norm-violation surrogate v1 (MEASURED — see `TRAINING_RECEIPT.json`)

A real sklearn `HistGradientBoostingClassifier` trained on **19,000 governed-norm records**
synthesized and **labeled by this kernel itself** (its own `rms_norm`/`layer_norm` reference
recompute + receipt-digest recompute + chain-link check; seed 20260721; 800 samples re-audited
by full kernel replay during generation — all agreed). Each non-clean record corrupts EXACTLY
one aspect (single-aspect rule): `wrong-output`, `wrong-eps-record`, `digest-tamper`, `chain-break`.
Features are cheap structural observables + digest/chain recompute — never the numeric reference recompute.

| metric | value |
|---|---|
| fidelity vs kernel (held-out agreement) | **0.8632** |
| test accuracy (structural classes only, excl. numeric) | **0.8323** |

| per-class recall | value |
|---|---|
| `clean` | 0.7610 |
| `wrong-output` (numeric) | 0.5986 |
| `wrong-eps-record` | 1.0000 |
| `digest-tamper` | 1.0000 |
| `chain-break` | 1.0000 |

**The blind spot is the point:** receipt/chain/eps tampering is caught at **1.0000** recall because
the surrogate can cheaply recompute digests and chain links. But `wrong-output` (recall
0.5986) is confused with `clean` (recall 0.7610) — **verifying that a
normalization is numerically correct requires the kernel's reference recompute**, which the surrogate
deliberately does NOT run (cheap output statistics cannot see it). Fast triage belongs to the
surrogate; numeric verdicts belong to the kernel. Class counts: {'clean': 5000, 'wrong-output': 3500, 'wrong-eps-record': 3500, 'digest-tamper': 3500, 'chain-break': 3500}. Λ untouched = Conjecture 1.

```python
import joblib
clf = joblib.load("model.joblib")   # feature spec: TRAINING_RECEIPT.json data.features
```

Re-verify everything: `python scripts/eval.py` (sha256-checks the shipped model against the
receipt, regenerates the seeded kernel-labeled dataset, retrains, and compares fidelity within ±0.02).

## SZL Kernels Suite

Part of the [`szl-kernels`](https://huggingface.co/SZLHOLDINGS/szl-kernels) governed-kernel suite — the hub links every member, and each member links back to the hub so no leaf is orphaned:

| Kernel | Lane |
|---|---|
| [`szl-kernels`](https://huggingface.co/SZLHOLDINGS/szl-kernels) | **hub** — unified suite, cross-kernel `UnifiedReceiptChain` |
| **`szl-governed-norm`** (this repo) | **RMSNorm/LayerNorm + SHA3-256 receipts** |
| [`szl-lambda-gate`](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate) | advisory Λ gate (Conjecture 1, OPEN) |
| [`governed-inference-meter`](https://huggingface.co/SZLHOLDINGS/governed-inference-meter) | MEASURED-joule energy accounting (NVML) |
| [`szl-govsign`](https://huggingface.co/SZLHOLDINGS/szl-govsign) | signed governance attestation (DSSE / in-toto) |
| [`szl-blocked`](https://huggingface.co/SZLHOLDINGS/szl-blocked) | honest-BLOCKED state + EU AI Act Annex IV DRAFT |
| [`szl-provctl`](https://huggingface.co/SZLHOLDINGS/szl-provctl) | provenance-DAG verify + in-toto/SLSA interop |

**Live Spaces:** [a11oy](https://huggingface.co/spaces/SZLHOLDINGS/a11oy) · [hatun-mcp](https://huggingface.co/spaces/SZLHOLDINGS/hatun-mcp).

**Related — Governed Kernels collection:** [Governed Kernels & Verifiers](https://huggingface.co/collections/SZLHOLDINGS/governed-kernels-and-verifiers-6a542ad83a4b75151bf5eae3) groups the whole family in one page. **Live console:** [a11oy](https://szlholdings-a11oy.hf.space) · [a-11-oy.com](https://a-11-oy.com) · [llm-router](https://szlholdings-llm-router-live.hf.space) · [receipt verifier](https://szlholdings-governed-receipt-verifier.static.hf.space) · [receipt spec (hub)](https://github.com/szl-holdings/governed-receipt-spec).

---

## About SZL Holdings

SZL Holdings, founded by **Stephen Lutar**, builds governed-AI infrastructure — provenance, observability, and security tooling for AI systems. Its work includes the **[a11oy governed-AI platform](https://a-11-oy.com)** and **[killinchu](https://huggingface.co/spaces/SZLHOLDINGS/killinchu)**, and a large public dataset corpus on the [SZL Holdings Hugging Face org](https://huggingface.co/SZLHOLDINGS).

## License

Apache-2.0. Copyright 2026 SZL Holdings.

---

*SZL Holdings · Doctrine v11 LOCKED 749/14/163 @ c7c0ba17 · Λ = Conjecture 1 · SLSA L1 honest*  
*Signed-off-by: Stephen Lutar <stephenlutar2@gmail.com>*

<sub><b>SZL Holdings</b> · governed normalization · provenance at the kernel layer · <a href="https://a-11-oy.com">a-11-oy.com</a> · <a href="https://github.com/szl-holdings">github.com/szl-holdings</a> · <a href="https://huggingface.co/SZLHOLDINGS">huggingface.co/SZLHOLDINGS</a></sub>

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19944926.svg)](https://doi.org/10.5281/zenodo.19944926)

## Citation

**Cite this.** Part of the SZL Holdings *Ouroboros Thesis* (Governed Post-Determinism).  
Concept DOI (always-latest): [10.5281/zenodo.19944926](https://doi.org/10.5281/zenodo.19944926).  
Author: Stephen P. Lutar Jr. · [ORCID 0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173) · License CC-BY-4.0.  
Full DOI-pinned lineage (v1→v26) + the 8 papers: [szl-papers PAPERS_INDEX](https://github.com/szl-holdings/szl-papers/blob/main/PAPERS_INDEX.md).  
No artifact-specific DOI is minted for this model; the concept DOI above covers the program.

Honesty (Doctrine v11): Λ unconditional uniqueness is **Conjecture 1** (machine-checked FALSE as stated) — never a theorem; conditional uniqueness is **Theorem U** (axiom-free). Locked-proven formulas = **exactly 8** {F1,F4,F7,F11,F12,F18,F19,F22}; ~185 experimental theorems are a separate CI-green tier; Khipu BFT safety = Conjecture 2. Trust never 100%.

```bibtex
@misc{lutar_szl_ouroboros,
  author    = {Lutar, Stephen P., Jr.},
  title     = {SZL Holdings --- The Ouroboros Thesis (Governed Post-Determinism)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19944926},
  url       = {https://doi.org/10.5281/zenodo.19944926},
  note      = {Concept DOI --- always resolves to the latest version. ORCID 0009-0001-0110-4173. CC-BY-4.0.}
}
```

*Signed-off-by: Stephen Lutar <stephenlutar2@gmail.com>*

## Files in this repo

| Path | What it is |
|---|---|
| `build/torch-universal/szl_governed_norm/__init__.py` | public API — `rms_norm`, `layer_norm`, receipts, `selfcheck()` |
| `build/torch-universal/szl_governed_norm/_norm.py` | the normalization math (pure PyTorch, float32 internal) |
| `build/torch-universal/szl_governed_norm/_receipt.py` | SHA3-256 hash-chained receipt engine |
| `build/torch-universal/szl_governed_norm/layers.py` | `nn.Module` wrappers |
| `build.toml` · `metadata.json` | Kernel Hub build/metadata manifests |
| `LICENSE` · `SECURITY.md` | Apache-2.0 · security policy |

---

<p align="center">
  <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> ·
  <a href="https://a-11-oy.com">a-11-oy.com</a> ·
  <a href="https://huggingface.co/SZLHOLDINGS/szl-lambda-gate">szl-lambda-gate</a>
</p>

<p align="center"><sub>SLSA: L1 honest · L2 attested · L3 roadmap. Λ = Conjecture 1 (advisory, never a theorem). Trust ceiling 0.97 — never 100%. Labels honest by default: MEASURED / REPORTED / MODELED / HEURISTIC / UNKNOWN / UNAVAILABLE. locked-proven = exactly 8 {F1,F4,F7,F11,F12,F18,F19,F22}.</sub></p>
