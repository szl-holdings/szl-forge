---
library_name: kernels
license: apache-2.0
tags:
- kernel
- governance
- ouroboros
- loop-tax
- provenance
- doi:10.5281/zenodo.19944926
szl-governance:
  verdict: ADVISORY
  lambda: Conjecture 1 (open) — uniqueness unproven; untouched by loop-tax accounting; advisory only
  provenance: reconstructs the a11oy bounded-loop trace + MEASURED/DERIVED loop-tax split offline
  honest_blocked: a measurement gap stays a gap — overheadMs is UNAVAILABLE when wall is unmeasured, never fabricated
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-ouroboros/card/holo-banner.svg" alt="szl-ouroboros — the bounded loop as a hologram: budget ticks along the arc, tail stopping short" width="100%"/>
</p>

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:54:01Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `15 passed in 0.04s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-ouroboros", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-ouroboros`](https://github.com/szl-holdings/szl-ouroboros) @ `cdd2e8f619672f4b8835f5697d84c0850d389a33`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-ouroboros", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->


# szl-ouroboros

**The cut, in one line:** a loop with a budget, a gap, and a receipt.

Bounded-loop trace + loop-tax accounting. **Not a model. No weights.** **Not** the TypeScript product [`szl-holdings/ouroboros`](https://github.com/szl-holdings/ouroboros).

**IS:** a pure-Python, stdlib-only governance kernel. It rebuilds a run's attempt windows into labeled fields (`modelMs` MEASURED, `overheadMs` DERIVED, `serializationTaxMs` a counterfactual, never a realized saving). Public API: `build_loop_trace`, `loop_tax`, `selfcheck`.

**IS NOT:** trained weights. Not a CUDA bench. Hub `model.joblib` is **QUARANTINED** executable serialization — do not `joblib.load` it. When `wall_ms` is missing, `overheadMs` is **UNAVAILABLE**, never fabricated.

Canonical GitHub source: https://github.com/szl-holdings/szl-ouroboros  
Hub package: https://huggingface.co/kernels/SZLHOLDINGS/szl-ouroboros  
This model-type repo is the publish / card mirror. Apache-2.0.

```python


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Agent frameworks retry until success. We tax the loop. Success that costs all remaining λ is a failure.

A runtime that cannot grind a gate into dust.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Don't jailbreak yourself by persistence. |
| NVIDIA | Scheduler quota, but for authority. |
| Unsloth | No. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Wrap agent retries.

## Limitations

- Kernel. Pair with governed-agent-bench.

Canonical GitHub: [`szl-holdings/szl-khipu`](https://github.com/szl-holdings/szl-khipu/blob/main/szl_khipu/ouroboros.py)
<!-- SZL-ATELIER-CUT:v1:END -->

from kernels import get_kernel
ou = get_kernel("SZLHOLDINGS/szl-ouroboros", revision="main", trust_remote_code=True)
attempts = [
    {"provider": "sovereign", "model": "own-metal", "ok": False, "latency_ms": 220, "node": "tower"},
    {"provider": "sovereign", "model": "own-metal", "ok": True,  "latency_ms": 900, "node": "laptop"},
]
trace = ou.build_loop_trace(attempts, wall_ms=1300, exit="converged", max_budget=4)
print(trace["modelMs"], trace["overheadMs"], trace["deadHopMs"])  # 1120 180 220
print(ou.selfcheck())
```

`LOOP_DOCTRINE = "bounded, terminating, receipt-closed"`. A budget violation is reported, never clamped. Λ is not touched here and stays Conjecture 1.

Doctrine v11. Copyright 2026 SZL Holdings · Stephen P. Lutar · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
