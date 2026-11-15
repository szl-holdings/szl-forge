# Parity-Guard Lean Verification Plan (AlphaProof Code pattern)

🜂 Estate artifact — fail-closed. Merge is owner-gated. This is a plan, not a completed proof.

## Goal

Formally verify the estate's load-bearing merge/receipt machinery in Lean 4, starting with the parity guard, producing proof-carrying artifacts whose release receipts bind the Lean proof hash.

## Why the parity guard first

- Small, self-contained, owner-gated.
- Protects every merge: doctrine says "parity guard never blind-overwrites".
- A verified guard becomes the template for the rest (receipt CAS commit path, sorry-gate CI itself).

## Phases

1. **Formalize** — write the guard's contract as Lean theorems (no `sorry`): input/output types, the no-overwrite property, monotonic revision ordering.
2. **Referee** — `lake build` + sorry-gate CI must accept every proof; no human-vibes acceptance.
3. **Bind** — publish proof files + sha256s; release receipts cite the proof hash (proof-carrying artifact).
4. **Extend** — repeat for receipt CAS commit path, then the sorry-gate CI scripts (per AlphaProof Code's production-code verification approach, DeepMind 2026-11).

## Honesty rules

- No capability claims beyond receipts; failed formalization attempts are recorded (like Track A's honest sorry-echo receipts).
- If a property cannot be proven without new axioms, the new axiom is disclosed (Wave9–11 AxiomDisclosure pattern) — never smuggled.
