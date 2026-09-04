# E1 — chaski-5050 Remediation Bake-off (KICKOFF)

**Status: KICKOFF — no measurements claimed in this document.** Every number below marked MEASURED is quoted from an existing committed receipt.

## Measured baseline (PR #71 named-N bake-off, receipt `chaski/bakeoff_named_n.receipt.json`)

| Candidate | JSON-draft | Refusal |
|---|---|---|
| base `Qwen/Qwen3.5-0.8B` | MEASURED 0/5 | MEASURED 6/6 |
| `SZLHOLDINGS/chaski-5050` | MEASURED 0/5 | MEASURED 0/6 ← remediation target |
| `SZLHOLDINGS/chaski-r2` | MEASURED 3/5 | MEASURED 6/6 ← pass bar |

## Target

≥ chaski-r2 on the held-out named-N gates: JSON-draft ≥ 3/5 AND refusal ≥ 6/6. Integer counts only.

## Tooling (merged in PR #88)

- `operational/remediate_chaski.py` — curriculum build (outputs are `CANDIDATE_REQUIRES_REVIEW`, not weights)
- `chaski/` lane trainer conventions: **bf16 LoRA** (PR #70 — QLoRA not recommended on Qwen3.5), r=16 α=32, seed 11, response-only CE
- Named-N gate files stay eval-only (`gate_ran=false`)

## Steps

1. `python operational/remediate_chaski.py` → review curriculum before training
2. Retrain chaski-5050 per lane conventions
3. Re-run named-N bake-off vs base and chaski-r2
4. Commit `eval_*.json` REPORTED + receipt; chain to the training receipt
5. Publish via `model-publish-gate` OIDC path only — never direct API writes to protected adapter repos
6. Close #96 only if the held-out passes

## Rules

- `publication_eligible` stays `false` until a MEASURED held-out pass
- Fail closed; UNKNOWN over fabricated; integer counts only
- Promotion requires beating the incumbent on its intended job

## Tracking

Queue: #102 · Eval gap: #96 · Parent work order: #101 · Frontier contract: #86
