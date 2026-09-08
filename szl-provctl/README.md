---
library_name: kernels
license: apache-2.0
tags:
- kernel
- provenance
- supply-chain
- in-toto
- slsa
- governance
- doi:10.5281/zenodo.19944926
szl-governance:
  verdict: INTEROP-PROVENANCE
  interop: in-toto Statement v1 + SLSA provenance v1 shapes — repository-selfchecked; external compatibility must be tested
  lambda: Conjecture 1 (open) — proven_trust structurally locked False
  energy: MEASURED-only — real NVML per-kernel delta; None when no GPU, never fabricated
  honest_blocked: BLOCKED nodes are surfaced in the provenance DAG, never silently dropped
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-provctl/card/holo-banner.svg" alt="szl-provctl — the provenance DAG as a hologram: six nodes walked edge-by-edge, one ringed BLOCKED" width="100%"/>
</p>

# szl-provctl

**The cut, in one line:** the graph keeps its blocked nodes — especially its blocked nodes.

Provenance-DAG control + in-toto / SLSA interop. **Not a model. No weights.**

**IS:** a software kernel that (1) emits declared in-toto Statement v1 and SLSA provenance v1 shapes from a `UnifiedReceiptChain`, (2) walks a provenance DAG edge-by-edge, (3) can bind a per-kernel NVML energy delta when a GPU is present. Public API: `statement_from_chain`, `slsa_statement`, `ProvenanceDAG` / `verify_dag`, `measure_kernel_energy`, `selfcheck`.

**IS NOT:** trained weights. Not a complete signing product (signing is `szl-govsign`). Not a CUDA speedup. A hash-chain digest is an integrity fingerprint, not a signature. External verifier compatibility must be tested. Hub `model.joblib` is **QUARANTINED** executable serialization — do not `joblib.load` it.

Canonical GitHub source: https://github.com/szl-holdings/szl-provctl  
Hub package: https://huggingface.co/kernels/SZLHOLDINGS/szl-provctl  
This model-type repo is the publish / card mirror. Apache-2.0.

```python


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Model cards skip the builder. We treat the builder as part of the model.

A weight that is invalid without its provenance predicate.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | No public SLSA for Claude weights. |
| NVIDIA | NGC signed images — take, then apply to LoRAs. |
| Unsloth | After the job, attach SLSA. Unsloth does not. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

CI predicate for every publish.

## Limitations

- Kernel.

Canonical GitHub: [`szl-holdings/szl-provctl`](https://github.com/szl-holdings/szl-provctl/blob/main/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

from kernels import get_kernel
pc = get_kernel("SZLHOLDINGS/szl-provctl", revision="main", trust_remote_code=True)
print(pc.selfcheck()["ok"])
chain = pc.UnifiedReceiptChain()
chain.emit("governed_norm", "rms_norm", {"in_shape": [4, 64], "eps": 1e-6})
stmt = pc.statement_from_chain(chain, lambda_score=0.9, decision="ALLOWED")
```

Energy is MEASURED-only: no GPU/NVML → `joules=None` / `UNAVAILABLE_NO_NVML`, never fabricated. BLOCKED nodes stay in the DAG. `proven_trust` is structurally locked False.

Λ = Conjecture 1 (advisory, never a theorem). Doctrine v11. Copyright 2026 SZL Holdings · Stephen P. Lutar · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
