---
thumbnail: https://huggingface.co/SZLHOLDINGS/YARQA-ATTN/resolve/main/og-card.png
language:
- code
license: apache-2.0
library_name: kernels
tags:
  - kernel
  - attention
  - yarqa
  - governed-ai
  - szl-holdings
  - kernel-lane
szl:
  owner: KERNEL
  not_a_weight: true
  not_an_alias: true
  collection: none
  python: present
  import_live: true
  gpu: UNAVAILABLE
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/YARQA-ATTN/card/holo-banner.svg" alt="YARQA-ATTN — compartment attention as a hologram: a channel gated into canals, each lock signed" width="100%"/>
</p>

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:54:08Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `25 passed, 1 skipped in 0.33s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/YARQA-ATTN", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/YARQA-ATTN`](https://github.com/szl-holdings/YARQA-ATTN) @ `160640bd8ed138e0170838c5a0de470ba8539367`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/YARQA-ATTN", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->


# YARQA-ATTN

**The cut, in one line:** attention in canals — every compartment signed, nothing over the lock.

<p align="center">
  <img src="og-card.png" alt="YARQA-ATTN" width="100%"/>
</p>

`KANCHAY` · Doctrine v11 · Lean `749/14/163` · Λ = Conjecture 1 (advisory) · [a-11-oy.com](https://a-11-oy.com)


Python kernel is on this repo. CPU `get_kernel` import-LIVE MEASURED (`7e533ce`). GPU cubins **UNAVAILABLE** (not ROADMAP). Not an alias of `szl-receipt-attn`. Not a fourth Flash / Flex / paged stack.

<!-- SZL-KERNEL-STATUS:import-LIVE:START -->
## Status


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

FlashAttention is faster. YARQA is accountable. We steal the kernel discipline from NVIDIA and spend it on provenance, not FLOPs.

An attention op whose softmax support is reconstructable from a signed log.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Interpretability as a runtime artifact. |
| NVIDIA | cuDNN / FlashAttention silhouette — then we add the receipt. |
| Unsloth | Unrelated. Don't wrap this in FastLanguageModel. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Drop-in attention with an audit tape.

## Limitations

- Not a checkpoint.
- Performance vs FlashAttention is not claimed.

Canonical GitHub: [`szl-holdings/szl-khipu`](https://github.com/szl-holdings/szl-khipu/blob/main/szl_khipu/yarqa.py)
<!-- SZL-ATELIER-CUT:v1:END -->


> **STATUS: import-LIVE** on CPU Kernel Hub `get_kernel` (kernels `0.16.1`). GPU cubins **UNAVAILABLE** this session (not ROADMAP).

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| Kernel Hub `get_kernel` | **import-LIVE** | MEASURED 2026-08-28 3:08pm ET on kernels `0.16.1`. Package HEAD [`7e533ce`](https://huggingface.co/kernels/SZLHOLDINGS/YARQA-ATTN/commit/7e533ce702029061bc68f9f9cafe88efdd7f5f00) (`7e533ce702029061bc68f9f9cafe88efdd7f5f00`). README at MEASURE [`2871b3c`](https://huggingface.co/kernels/SZLHOLDINGS/YARQA-ATTN/commit/2871b3cbd73ee05e9f1aa010b75b190683f0ecd3). Legal name `yarqa-attn` (Python module `yarqa_attn`). Variants: `build/torch-universal` (default `get_kernel`) and `build/torch-cpu` (`backend="cpu"`). Working calls: `get_kernel("SZLHOLDINGS/YARQA-ATTN", revision="main", trust_remote_code=True)` and the same with `backend="cpu"`. `selfcheck` **ok**. `max_abs_vs_compartment_ref=3.58e-07` (full `3.5762786865234375e-07`), `path=torch_compartment`. What-NOT: no tokens/s; no joules; not a fourth Flash / Flex / paged stack. Lambda = Conjecture 1 (advisory). |
| GPU cubins | **UNAVAILABLE** | MEASURED 2026-08-28 7:01pm ET this session. Host `cursor` (Linux 6.12.94+ x86_64, Intel Xeon 8-core). `torch` `2.13.0+cu130` compiled CUDA 13.0. `torch.cuda.is_available()=false`. `nvidia-smi` UNAVAILABLE. `device_count=0`. Triton `3.7.1` present with no CUDA device. No cubin shipped. No tokens/s. No joules. CPU import-LIVE unchanged. Not a fourth Flash / Flex / paged stack. Lab stays Khipu. |

<!-- SZL-KERNEL-STATUS:import-LIVE:END -->

KERNEL kernel card. Original SZL **compartment / plug-flow** attention cut. Receipt-aware. Honesty-labeled.

**Not a Fall 2026 ATELIER weight.** No tensors in this repo. Not an alias of [`szl-receipt-attn`](https://github.com/szl-holdings/szl-receipt-attn). Not a pointer at the Triton trio (`szl-receipt-attn`, `szl-maskmod`, `szl-block-kv`). Those three stay separate. a11oy-net does not list this as a fourth Flash / Flex / paged stack.

GitHub is source of truth: [`szl-holdings/YARQA-ATTN`](https://github.com/szl-holdings/YARQA-ATTN). KERNEL binds Hub bytes from that tree. Do not PUT an empty card.

| | |
|---|---|
| **Owner** | KERNEL |
| **Artifact** | kernel (Python present; no weights; GPU cubins not claimed) |
| **Status** | **import-LIVE** CPU · GPU cubins **UNAVAILABLE** |
| **License** | Apache-2.0 |
| **Λ** | Conjecture 1 (advisory, never a theorem) |
| **Path** | `torch_compartment` (CPU) |
| **Serve studio** | not this repo. Live CPU lab is [`szl-model-inference-lab`](https://huggingface.co/spaces/SZLHOLDINGS/szl-model-inference-lab) (Khipu GGUF only) |

Silhouette: partition a sequence into canals (contiguous compartments), attend within a canal, emit SHA3-256 of the partition and of the attention output. We do not copy Dao hopper, Sage `csrc`, vLLM paged `.cu`, cuDNN FMHA, TRT cubins, CuTeDSL, or `flex_attention.py`. Metaphor only vs [`szl-holdings/yarqa`](https://github.com/szl-holdings/yarqa) (CFD; different product). Throughput is MEASURED only from a timed run on named hardware. Until then every speed claim is unstamped. No tokens/s. No joules.

Do not list this next to Chaski, Qantu, Waman, Chakana, or Tinku.

## Load

```python
from kernels import get_kernel
attn = get_kernel("SZLHOLDINGS/YARQA-ATTN", revision="main", trust_remote_code=True)
```

Fashion GO 2026-08-28 3:10pm ET. import-LIVE CPU stays. GPU cubins stamped **UNAVAILABLE** 2026-08-28 7:01pm ET (no CUDA device this session). Not a fourth Flash / Flex / paged stack.

Apache-2.0. Copyright 2026 SZL Holdings.
