---
thumbnail: https://huggingface.co/SZLHOLDINGS/szl-block-kv/resolve/main/og-card.png
library_name: kernels
license: apache-2.0
tags:
  - kernel
  - szl-holdings
---

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:54:24Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `10 passed in 2.52s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-block-kv`](https://github.com/szl-holdings/szl-block-kv) @ `7cf99cca9d2749482f5f57572c3affc2e7a5ebc7`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->

<p align="center">
  <img src="holo-banner.svg" alt="szl-block-kv — holographic paged-grid banner, lit pages beside forgotten cells" width="100%"/>
</p>

<h1 align="center">S Z L &nbsp;B L O C K &nbsp;K V</h1>

<p align="center"><em>A cache that cannot be subpoenaed for what it was never allowed to hold.</em></p>

<p align="center">
  <img alt="Artifact: receipted kernel" src="https://img.shields.io/badge/artifact-receipted%20kernel-0e7490?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/szl-block-kv?style=flat-square&color=22d3ee&label=downloads"/>
  <img alt="get_kernel: import-LIVE MEASURED" src="https://img.shields.io/badge/get__kernel-import--LIVE%20MEASURED-a5f3fc?style=flat-square"/>
  <img alt="Triton GPU kernel: UNAVAILABLE" src="https://img.shields.io/badge/triton%20gpu-UNAVAILABLE-b45309?style=flat-square"/>
  <img alt="No speedup claim" src="https://img.shields.io/badge/speedup%20claim-none-334155?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
</p>

`KANCHAY` · Doctrine v11 · Lean `749/14/163` · Λ = Conjecture 1 (advisory) · [a-11-oy.com](https://a-11-oy.com)

Kernel sources are on this repo. CPU `get_kernel` import-LIVE is MEASURED. GPU **UNAVAILABLE** (no cubin + timed run). Not a model. Not listed next to Chaski or Qantu.

Canonical GitHub source for `SZLHOLDINGS/szl-block-kv`.

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Blocked KV was invented for speed. We use the blocking to forget on purpose. Memory is a privilege.

A cache that cannot be subpoenaed for what it was never allowed to hold.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | No persistent unauthorized memory. |
| NVIDIA | PagedAttention / blocked KV — then we spend it on governance. |
| Unsloth | No. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Governed decode memory.

## Limitations

- Kernel. Not a drop-in vLLM replacement.

Canonical GitHub: [`szl-holdings/szl-khipu`](https://github.com/szl-holdings/szl-khipu/blob/main/szl_khipu/block_kv.py)
<!-- SZL-ATELIER-CUT:v1:END -->

**Original SZL construction in the paged-KV category.** Inspired by Kwon et al. PagedAttention SOSP 2023 https://arxiv.org/abs/2309.06180. **NOT a rehost of vLLM or kernels-community/paged-attention.** **Distinct from a11oy MODELED H2O eviction.**

Doctrine v11. **Λ = Conjecture 1 OPEN** (advisory; uniqueness unproven).

<!-- SZL-KERNEL-STATUS:import-LIVE:START -->
## Status

> **STATUS: import-LIVE** on CPU Kernel Hub `get_kernel` (kernels `0.16.1`). Triton page kernel is **UNAVAILABLE**.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| Kernel Hub `get_kernel` | **import-LIVE** | MEASURED 2026-08-28 2:29pm ET on kernels `0.16.1`. HEAD [`d3ede3e`](https://huggingface.co/kernels/SZLHOLDINGS/szl-block-kv/commit/d3ede3e471b51080492b1c69306283507dcf507e) (`d3ede3e471b51080492b1c69306283507dcf507e`). Legal name `szl-block-kv` (Python module `szl_block_kv`). Variants: `build/torch-universal` (default `get_kernel`) and `build/torch-cpu` (`backend="cpu"`). Working calls: `get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)` and the same with `backend="cpu"`. `selfcheck` **ok**. `max_abs_vs_contiguous=2.38e-07` (full `2.384185791015625e-07`), `path=torch_gather`, `chain_ok=true`. What-NOT: no tokens/s; no joules. |
| Triton page kernel | **UNAVAILABLE** | MEASURED 2026-08-28 7:01pm ET this session. Host `cursor` (Linux 6.12.94+ x86_64, Intel Xeon 8-core). `torch` `2.13.0+cu130` compiled CUDA 13.0. `torch.cuda.is_available()=false`. `nvidia-smi` UNAVAILABLE. `device_count=0`. Triton `3.7.1` present with no CUDA device. No cubin. No timed GPU run. No tokens/s. No joules. |

<!-- SZL-KERNEL-STATUS:import-LIVE:END -->

v0 is a **labeled torch gather** over a block table. GPU paged Triton is **UNAVAILABLE**. **No speedup claim.**

## Load

```python
from kernels import get_kernel

kv = get_kernel("SZLHOLDINGS/szl-block-kv", revision="main", trust_remote_code=True)
print(kv.selfcheck())
```

Local checkout:

```python
from szl_block_kv import PagedCache, paged_attn, reshape_and_cache, selfcheck
print(selfcheck())
```

Correctness (documented): paged gather matches contiguous KV SDPA within atol/rtol `1e-5` on float32. Skip CUDA Triton — that kernel is not in v0.

Apache-2.0. Copyright 2026 SZL Holdings.

---

<p align="center">
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/szl-block-kv">SZLHOLDINGS/szl-block-kv</a>
</p>
