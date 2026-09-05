# SZLHOLDINGS family index

**Decision (2026-09-04):** Hugging Face model repo `SZLHOLDINGS/SZLHOLDINGS` is an **org family index**, not a model.

- Role: `org-index`
- Weights: none
- Receipts / benches: not applicable
- Canonical product: https://a-11-oy.com
- Canonical proof: https://a11oy.net
- Canonical source org: https://github.com/szl-holdings
- Hub org: https://huggingface.co/SZLHOLDINGS
- Λ: Conjecture 1 (advisory, never a theorem)
- Trust ceiling: 0.97

This file is the GitHub-side close of [szl-forge#92](https://github.com/szl-holdings/szl-forge/issues/92).
Publishing the same wording onto the Hub card still requires an owner HF write token.

## Honest family labels

| Artifact | Kind | Status |
|---|---|---|
| `SZLHOLDINGS/SZLHOLDINGS` | index card | PLACEHOLDER / org-index |
| `SZL-Khipu-1.5B` | weights | CANDIDATE, not production |
| `SZL-Khipu-1.5B-abstain` | adapter | QUARANTINE C2 — empty / experiment |
| `SZL-Forge-1.5B-ReceiptAgent` | weights | CANDIDATE, proposal-only |
| `szl-receiptagent-qwen35-0.8b-v3` | adapter | QUARANTINE C1 — placeholder card |
| `chaski` | weights | FAILED qualification — research-only (C5/L3) |
| `chaski-5050` | weights | QUARANTINE C5 — path leak |
| `A11OY-MINI` | GGUF | QUARANTINE C5 — derived from failed parent |
| `khipu-r3` | weights | CANDIDATE |
| `szl-blocked` | kernel + surrogate | SOFTWARE + MEASURED fidelity on held-out split |
| `szl-provctl` | kernel + surrogate | SOFTWARE |
| `szl-energy-attest` | measurement software | GitHub canonical; Hub mirror EMPTY until republish |
| `governed-inference-meter` | deprecated meter | DEPRECATED; successor is `szl-energy-attest` |

No half-state: an index is an index. Do not treat reachability as readiness.

Collection membership deny list: `publishing/collection-quarantine.json`.
That file does not write the Hub.
