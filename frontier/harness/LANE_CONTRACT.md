# The Lane Contract

Every evaluation lane under `frontier/harness/` obeys the same five invariants. A runner that violates any one of them is not a lane — it is a demo.

## The five invariants

1. **Receipts or it didn't happen.** Every lane run emits a signed receipt: inputs hashed, verdict, timestamp, host, key id. Counts travel; raw hidden content never does.
2. **Fail closed.** Ties, unreadable evidence, self-authored evidence, and unparseable receipts all resolve to FAIL — never to "probably fine."
3. **Hidden evaluation sets.** The artifacts that decide promotion are not in this repository. Public smoke fixtures exist only to prove the runner runs; they are never evidence.
4. **Declared baselines, strictly beaten.** Each lane carries a disclosed line (e.g. `KHIPU_BASELINE` in `lane_l2.py`) anchored to a signed receipt under `frontier/evidence/`. A candidate must strictly beat the line on the hidden set before anything freezes.
5. **Byte-identical anchors.** Receipts, manifests, and baselines are compared as exact bytes or exact counts. "Close enough" is a finding, not a pass.

## Frozen lanes

| Lane | Runner | What it grades |
|---|---|---|
| L1 | `lane_l1.py` | Harness self-check: verify path, receipt shape, fail-closed exits |
| L2 | `lane_l2.py` | Khipu abstention: strictly beat the declared 3/6 line on the hidden handle set |
| L3 | `lane_l3.py` | Controller operating point: freeze only from a bench run, never from prose |

Evidence anchoring each lane's declared line lives in `../evidence/` as `szl-frontier-baseline-evidence` JSON — receipt counts and identifiers only, per the content policy recorded in each file.

## Definition of done for a new lane

- Runner exits 0/1 with a signed receipt; no third state
- Public smoke fixtures + a hidden-set path that refuses to run without explicit owner invocation
- Declared baseline anchored to a signed receipt filed under `frontier/evidence/`
- Tracker entry in szl-hf-frontier#9 updated in the same change

## Unfrozen lanes

L4 and beyond are deliberately unfrozen. Their charters come from szl-hf-frontier#9 and owner decisions, not from this file. Anyone building L4 starts there — not by copying an existing lane's shape and hoping.
