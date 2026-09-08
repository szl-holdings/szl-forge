---
library_name: kernels
license: apache-2.0
tags:
- kernel
- governance
- invariants
- provenance
- receipts
- ed25519
- doi:10.5281/zenodo.19944926
szl-governance:
  verdict: ADVISORY
  lambda: Conjecture 1 (open) — uniqueness unproven; untouched by these invariants; advisory only
  provenance: recomputes the a11oy receipt/ledger self-consistency invariants offline
  honest_blocked: a VIOLATED invariant stays VIOLATED — statuses are never coerced to a pass
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-invariants/card/holo-banner.svg" alt="szl-invariants — the eight invariants as a hologram: eight interlocked links" width="100%"/>
</p>

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:54:00Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `16 passed in 0.07s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-invariants", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-invariants`](https://github.com/szl-holdings/szl-invariants) @ `0a1e6bdeaf4d7515bbbf4ca37c3c838a4f293542`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-invariants", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->


# szl-invariants

**The cut, in one line:** a violated check stays violated — the card you cannot sweet-talk.

Eight falsifiable receipt/ledger invariants, offline. **Not a model. No weights.**

**IS:** a pure-Python, stdlib-only governance kernel that replays the same eight checks the a11oy backbone runs at `/api/invariants` over a ledger JSONL you already hold. Public API: `run_invariants`, `load_jsonl`, `verify_ed25519`, `selfcheck`. Statuses `HOLDS / VIOLATED / KEY_ROTATED / NO_DATA / UNAVAILABLE` are first-class.

**IS NOT:** trained weights or a LoRA. Not a CUDA bench. Passing `selfcheck` is not an eval leaderboard. Hub `model.joblib` is **QUARANTINED** executable serialization — do not `joblib.load` it. These checks do not prove the export is complete.

Canonical GitHub source: https://github.com/szl-holdings/szl-invariants  
Hub package: https://huggingface.co/kernels/SZLHOLDINGS/szl-invariants  
This model-type repo is the publish / card mirror. Apache-2.0.

```python


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Checksums are optional. Ours are the product. If the signature fails, the model is not the model.

Weights that are invalid if unsigned — at load time, not at audit time.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Trust, but here we verify. |
| NVIDIA | Supply-chain (signed containers) applied to tensors. |
| Unsloth | After merge, sign. Not during QLoRA. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Load-time invariant check.

## Limitations

- Proves integrity and origin, never accuracy.

Canonical GitHub: [`szl-holdings/szl-invariants`](https://github.com/szl-holdings/szl-invariants/blob/main/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

from kernels import get_kernel
inv = get_kernel("SZLHOLDINGS/szl-invariants", revision="main", trust_remote_code=True)
report = inv.run_invariants(inv.load_jsonl("runs_export.jsonl"),
    samples=inv.load_jsonl("training_samples.jsonl"), pubkey="<SPKI base64>")
print(report["summary"])
print(inv.selfcheck())
```

The eight ids: `receipt-chain-continuity`, `ledger-failure-shape`, `served-run-has-model`, `signed-columns-atomic`, `loop-steps-positive`, `receipt-ed25519-verify`, `receipt-columns-consistent`, `flywheel-lineage`. Missing pubkey → #6 UNAVAILABLE; missing samples → #8 UNAVAILABLE.

Λ is not touched here and stays Conjecture 1. Doctrine v11. Copyright 2026 SZL Holdings · Stephen P. Lutar · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
