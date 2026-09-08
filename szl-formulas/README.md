---
library_name: kernels
license: apache-2.0
tags:
- kernel
- governance
- formulas
- proof-status
- lean4
- lambda-aggregate
- doi:10.5281/zenodo.19944926
szl-governance:
  verdict: ADVISORY
  lambda: Conjecture 1 (open) — uniqueness unproven; composer Λ roll-up is ADVISORY only, never proven trust
  provenance: offline replay of SZLHOLDINGS/canonical-formulas-v1 — 21 pure formulas + the governed-loop composer
  honest_blocked: PROOF_STATUS is mirrored verbatim; locked-proven canonical set is EXACTLY 8 (machine-enforced) — never inflated
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-formulas/card/holo-banner.svg" alt="szl-formulas — the formula registry as a hologram: twenty-one bars, exactly eight lit" width="100%"/>
</p>

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:52:52Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `19 passed in 0.04s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-formulas", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-formulas`](https://github.com/szl-holdings/szl-formulas) @ `977f344ccf248b55299a623d91db162f266eda29`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-formulas", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->


# szl-formulas

**The cut, in one line:** twenty-one formulas walk in; exactly eight walk out proven.

Software kernel for SZL formula composition. **Not a model. No weights.**

**IS:** a pure-Python, stdlib-only governance kernel — offline replay of the 21 canonical formulas plus the governed-loop composer. Public surface includes `registry_count()`, `lambda_aggregate`, `LOCKED_PROVEN_FORMULA_IDS`, `run_governed_loop`, `PROOF_STATUS` (mirrored verbatim).

**IS NOT:** trained weights, LoRA, GGUF, `.pt`, or a classifier. Hub `model.joblib` is **QUARANTINED** executable serialization (and is not on this tree) — do not `joblib.load` it. Not a CUDA/Triton speedup. Not `lutar-lean` and not the TypeScript `ouroboros` product.

Canonical GitHub source: https://github.com/szl-holdings/szl-formulas  
Hub package: https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas  
This model-type repo is the publish / card mirror. Apache-2.0.

```python


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Papers have appendices. We have a kernel of formulas whose sorry-count is public (Doctrine v11: 749 / 14 / 163).

A formula registry that cannot drift from the Lean tree.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Written principles, but ours compile. |
| NVIDIA | Recipe as code. |
| Unsloth | No. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Import formulas. Check proof-status before citing.

## Limitations

- Λ uniqueness is Conjecture 1 — NOT a theorem.

Canonical GitHub: [`szl-holdings/szl-formulas`](https://github.com/szl-holdings/szl-formulas/blob/main/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

from kernels import get_kernel
fx = get_kernel("SZLHOLDINGS/szl-formulas", revision="main", trust_remote_code=True)
print(fx.registry_count())                   # 21
print(sorted(fx.LOCKED_PROVEN_FORMULA_IDS))  # exactly 8 F-ids
chain = fx.run_governed_loop([
    {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
    {"formula_name": "reed_solomon_singleton", "args": [255, 223]},
])
print(chain["replay_ok"], chain["lambda_label"])  # True, ADVISORY (Conjecture 1)
```

Proof honesty: per-formula `PROOF_STATUS` is `PROVEN / AXIOM / SORRY / CONJECTURE` copied verbatim — that is not membership in the locked-proven set. Locked-proven is **exactly 8**: `{F1, F4, F7, F11, F12, F18, F19, F22}`. The F-number mapping onto these 21 registry names is **UNKNOWN — never fabricated.**

| Claim | Label |
|---|---|
| Source on GitHub | REACHABLE |
| CUDA benches | UNAVAILABLE |
| Weights | not applicable |
| Λ | Conjecture 1 (advisory, never a theorem) |

Doctrine v11. Copyright 2026 SZL Holdings · Stephen P. Lutar · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
