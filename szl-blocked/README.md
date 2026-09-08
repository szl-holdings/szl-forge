---
library_name: kernels
license: apache-2.0
tags:
- kernel
- eu-ai-act
- compliance
- annex-iv
- governance
- doi:10.5281/zenodo.19944926
szl-governance:
  verdict: HONEST-BLOCKED
  doctrine: hard DENY dominates; advisory Λ can only tighten, never override; a BLOCKED op never executes — no fake-green
  lambda: Conjecture 1 (open) — a recorded ALLOW is never proven trust
  energy: MEASURED-only
  honest_blocked: BLOCKED is a first-class governed state; fn is NEVER called on block
  annex_iv: DRAFT skeleton auto-derived from provenance — NOT legal advice, NOT a conformity guarantee
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-blocked/card/holo-banner.svg" alt="szl-blocked — honest-BLOCKED as a hologram: a call path meeting a hard wall, receipt stub dropping below" width="100%"/>
</p>

<!-- SZL-KERNEL-OPERATIONAL:START -->
## Operational (MEASURED laptop-Blackwell)

> **STATUS:** tests **PASS**. `get_kernel` **import-LIVE**. Unsloth/LoRA is the wrong tool. Receipted kernels, not silent CUDA.

| Thing | Label | Method / N / date / what-NOT |
|---|---|---|
| tests (`PYTHONPATH=torch-ext`) | **PASS** | MEASURED 2026-08-29T15:54:04Z host `betterwithage` Windows-10-10.0.26200-SP0. torch `2.10.0+cu128`. GPU `NVIDIA GeForce RTX 5050 Laptop GPU` arch `Blackwell`. pytest `3 passed in 1.57s`. Failed nodes: `none`. What-NOT: not a leaderboard. torch.compile fullgraph failures on Windows Blackwell (`cl is not found`) are MEASURED, not hidden. |
| Kernel Hub `get_kernel` | **import-LIVE** | kernels `0.16.1`. Default: `get_kernel("SZLHOLDINGS/szl-blocked", revision="main", trust_remote_code=True)` → `True`. `backend="cpu"` → `True`. trust_remote_code=False → `ValueError` (SZLHOLDINGS is not a trusted publisher). repo_type=kernel required (kernels 0.16). What-NOT: not a weight load; do not pickle/joblib.load. |
| formula-tax | **ADVISORY** | locked-8 `F1 F4 F7 F11 F12 F18 F19 F22`. registry_count=21. Λ geomean `1.0`. uniqueness **Conjecture 1** (never a theorem). |
| I1–I8 | **catalog** | `I1 receipt-chain-continuity; I2 ledger-failure-shape; I3 served-run-has-model; I4 signed-columns-atomic; I5 loop-steps-positive; I6 receipt-ed25519-verify; I7 receipt-columns-consistent; I8 flywheel-lineage`. Executed by `SZLHOLDINGS/szl-invariants`. Statuses never coerced. Λ untouched. |
| CUDA speedup / tokens/s / joules | **UNAVAILABLE** | Not claimed. Receipted kernels, not silent CUDA. |

GitHub source: [`szl-holdings/szl-blocked`](https://github.com/szl-holdings/szl-blocked) @ `79f950a60debb7f1f0c6f5e2f3baa5e8b7ab6739`. Artifacts: [`BENCH.laptop-blackwell.json`](./BENCH.laptop-blackwell.json), [`OPERATIONAL.json`](./OPERATIONAL.json).

```python
from kernels import get_kernel
k = get_kernel("SZLHOLDINGS/szl-blocked", revision="main", trust_remote_code=True)
```

<!-- SZL-KERNEL-OPERATIONAL:END -->


# szl-blocked

**The cut, in one line:** the deny is the deliverable.

Honest-BLOCKED as a first-class governed state. **Not a model. No weights.**

**IS:** a fail-closed Python governance kernel. On deny the guarded `fn` never runs (`BlockedResult.output is None`) and a BLOCK receipt is written. Public API: `governed_call`, `GovernedGate`, `deny_by_default` / `deny_if_flag` / `deny_if_action_in`, `UnifiedReceiptChain`. Sibling package `szl_euaiact` can emit an Annex IV-style **DRAFT skeleton**.

**IS NOT:** trained weights. Not FlashAttention. Not a drop-in blocker library. Not legal advice and **not** a declaration of conformity. No MEASURED latency or CUDA benches here. Hub `model.joblib` is **QUARANTINED** executable serialization — do not `joblib.load` it.

Canonical GitHub source: https://github.com/szl-holdings/szl-blocked  
Hub package: https://huggingface.co/kernels/SZLHOLDINGS/szl-blocked  
This model-type repo is the publish / card mirror. Apache-2.0.

```python


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

People write Annex IV in Word. We keep a kernel that refuses to publish if the annex fields are empty.

A blocked publish — the name is the feature.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Policy docs. We want a gate. |
| NVIDIA | Enterprise compliance packs. |
| Unsloth | No. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Block publish when annex fields are missing.

## Limitations

- Not legal advice. A documentation gate.

Canonical GitHub: [`szl-holdings/szl-blocked`](https://github.com/szl-holdings/szl-blocked/blob/main/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

from kernels import get_kernel
blk = get_kernel("SZLHOLDINGS/szl-blocked", revision="main", trust_remote_code=True)

chain = blk.UnifiedReceiptChain()
policy = blk.deny_if_action_in({"exfiltrate", "delete_all"})
def do_work(x): return x * 2
ok = blk.governed_call(do_work, policy, chain, request={"action": "summarize"}, args=(21,))
no = blk.governed_call(do_work, policy, chain, request={"action": "exfiltrate"}, args=(21,))
assert ok.blocked is False and ok.output == 42
assert no.blocked is True and no.output is None
```

Hard DENY dominates. Advisory Λ can only tighten, never loosen, and never manufactures trust. Annex IV output is a draft skeleton with explicit TODOs.

Λ = Conjecture 1 (advisory, never a theorem). Doctrine v11. Copyright 2026 SZL Holdings · Stephen P. Lutar · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
