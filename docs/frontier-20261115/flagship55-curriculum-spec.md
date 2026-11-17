# Flagship55 Curriculum Spec — formal-math lane (owner review draft)

🜂 Estate artifact — fail-closed. Merge is owner-gated. Nothing here authorizes a launch by itself.

## Status inputs (2026-11-15)

- flagship54 (salad-flagship54-v1) in progress: checkpoint-11000/37000 at last check, ETA ~2026-11-17T13:40Z. **Do not modify the running job.** This spec targets the NEXT run only.
- Measured estate batteries (mass-battery receipts, 2026-11-13/14): 3/6–4/6; disclosed weakness = abstention failures on unanswerable probes.
- Frontier pattern (2026): verified-trace loop — generate → Lean kernel referees → train only on accepted traces (AlphaProof/Nexus; DeepSeek-Prover-V2 subgoal decomposition + binary kernel reward).

## Curriculum (proposal, to be pinned by the owner before launch)

1. Retain the flagship54 base mix (owner-pinned).
2. Add [SZLHOLDINGS/lean-math-sft-v1](https://huggingface.co/datasets/SZLHOLDINGS/lean-math-sft-v1) (2,126 sorry-free blocks; train_file_sha256 3e02f8f9…) — sorry-free = kernel-checked ground truth at corpus revision e80aebb2.
3. Add [SZLHOLDINGS/axiom-invariant-corpus-v2](https://huggingface.co/datasets/SZLHOLDINGS/axiom-invariant-corpus-v2) (94 verbatim axiom/invariant declarations from lutar-lean @ c2ca813; jsonl sha256 c34c9f24…) as statement-context data — axioms/invariants appear as statements, not as proven claims. sorry-bearing invariant decls excluded from SFT tier; axiom tier verbatim/unconditional
4. Keep the receipt-of-doubt unanswerables paired with the Lean data so abstention trains against the same referee (battery failure mode).
5. Receipts: every run publishes a training_report + eval battery receipt to frontier-54-runs (or successor), same as wave-4.

## Kernel-referee RL lane (Track C scaled)

- Corpus v2 admission rule: **only lake-build-accepted proofs** (sorry-gate CI green) enter training data. No exceptions, no auto-merge; PhD/owner gate per CAMPAIGN.md.
- Reward signal: binary kernel accept/reject per attempt receipt (PASS/FAIL/ABSTAIN honest).
- External yardstick (honest benchmark): miniF2F + Lean-Workbook sample; publish the real number next to DeepSeek-Prover-V2's 88.9% miniF2F, whatever ours is. A receipt-bound low score beats a fabricated high score (doctrine).
- Candidate models attempt the 143 sorry_open targets each round; discharged sorries → corpus v2 → next round.

## Gates before launch (owner checklist)

- [ ] flagship54 finished + evaluated (receipts bound)
- [ ] Curriculum mix pinned by owner (exact ratios)
- [ ] Salad lane selected; cost check (0-replica groups cost decision open)
- [ ] Dataset revisions pinned by sha256 in the run manifest
