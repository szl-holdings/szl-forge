# SZL Evidence Remediation - PR Queue

Generated: 2026-09-24 17:10:49
Branch: evidence/identity-bindings-20260924-171023

## Completed and live (verified on Hub)
- Research-card rewrites: brain-navigator-r2, chaski-r2, khipu-r3, A11OY-MINI,
  receiptagent v2-merged, receiptagent v3, triage study5
- Kernel cards: szl-governed-norm, szl-lambda-gate, governed-inference-meter
- szl-energy-attest source-pointer card
- SZL-Khipu-1.5B: signed hashes bound to safetensors, not GGUF
- szl-frontier-evaluation-receipts: summary-only Viewer config

## This branch: identity bindings (byte identity only)
| Repository | Pinned revision | Manifest | Still required |
|---|---|---|---|
| System.Object[] | System.Object[] | frontier/evaluation/identity_bindings/System.Object[] | Evaluation receipt referencing these hashes |

## Boundary
Identity manifests prove which bytes exist at a pinned revision. They do not
evaluate, qualify, or promote anything. MISSING_BINDING closes only when a new
evaluation receipt references these exact hashes. All artifacts remain
publication_eligible=false, autonomy_eligible=false, HOLD, promotion_effect=NONE.