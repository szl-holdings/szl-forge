---
library_name: kernels
license: apache-2.0
tags:
- kernel
- signing
- attestation
- dsse
- in-toto
- governance
- doi:10.5281/zenodo.19944926
szl-governance:
  verdict: SIGNED-ATTESTATION
  predicate_type: https://szl.holdings/governance/v1
  lambda: Conjecture 1 (open) — signature never upgrades advisory to proven trust
  energy: MEASURED-only (label constrained structurally)
  honest_blocked: a BLOCKED verdict is signed as BLOCKED — never flipped
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-govsign/card/holo-banner.svg" alt="szl-govsign — the signed envelope as a hologram: flap drawn, seal ringed, check inside" width="100%"/>
</p>

# szl-govsign

**The cut, in one line:** the seal proves the envelope, never the truth inside.

Signed governance provenance. **Not a model. No weights.**

**IS:** a software kernel that builds in-toto / DSSE envelopes over an SZL governance predicate (Λ advisory, MEASURED energy, honest-BLOCKED, allow/block) and signs them with ECDSA P-256. Public API: `attest`, `verify`, `build_governance_predicate`, `generate_ephemeral_keypair`, `selfcheck`.

**IS NOT:** trained weights. Not a CUDA bench. A signature proves integrity and key possession only — it does **not** prove Λ uniqueness and does **not** upgrade advisory to proven trust (`proven_trust` stays locked False). Hub `model.joblib` is **QUARANTINED** executable serialization — do not `joblib.load` it. GitHub is the approved source.

Canonical GitHub source: https://github.com/szl-holdings/szl-govsign  
Hub package: https://huggingface.co/kernels/SZLHOLDINGS/szl-govsign  
This model-type repo is the publish / card mirror. Apache-2.0.

```python


<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Attestation as a library, not a PDF.

Every organ speaks DSSE.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | No equivalent public kernel. |
| NVIDIA | Signed containers / in-toto in supply chain. |
| Unsloth | No. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Sign receipts. Verify offline.

## Limitations

- Kernel.

Canonical GitHub: [`szl-holdings/szl-govsign`](https://github.com/szl-holdings/szl-govsign/blob/main/README.md)
<!-- SZL-ATELIER-CUT:v1:END -->

from kernels import get_kernel
gs = get_kernel("SZLHOLDINGS/szl-govsign", revision="main", trust_remote_code=True)
print(gs.selfcheck())
```

Energy labels are MEASURED-only (modeled/fabricated values are rejected). A BLOCKED verdict is signed as BLOCKED — never flipped. Imports: stdlib + `cryptography`. Prior art: in-toto, DSSE, Sigstore model-transparency. The `https://szl.holdings/governance/v1` predicate is SZL's own.

Λ = Conjecture 1 (advisory, never a theorem). Doctrine v11. Copyright 2026 SZL Holdings · Stephen P. Lutar · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
