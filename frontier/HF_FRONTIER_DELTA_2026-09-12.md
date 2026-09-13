# HF Frontier Delta — 2026-09-12

Local Omen pipeline night (RTX 5050 laptop, CPU merge). Delta since `HF_FRONTIER_INTAKE_2026-09-06.md`.

## Evidence boundary

This is a dated operator-reported activity log, not an independently verified
publication, benchmark, or production-qualification receipt. The original report
below is retained verbatim, including its `MEASURED` and `Verified` labels; those
labels describe the operator's reported observations, not a new verification
performed by the documentation reviewer or by merging this file.

The abbreviated commit IDs and approximate sizes are investigation pointers,
not immutable source/weight bindings. Before using these results for promotion,
resolve full provider revisions, inspect and validate each `merge_receipt.json`,
bind base/adapter/merged-weight hashes and evaluation configuration, and reconcile
the stated abstention discrepancy. A non-zero weight delta alone does not prove
correct adapter mapping, reload parity, model quality, or lawful data use.

The v3 recommendation below is part of the historical operator report, not a
change to a canonical candidate, model route, or approval gate. The reported token
exposure remains a blocker for reuse of that credential until rotation and scope
are verified; this document neither contains a credential nor certifies rotation.
Source CI validates source changes, not the historical training or publication
claims below. No training, salvage, provider mutation, or promotion is requested
or executed by this note.

## Original operator report

## Merged + published (additive)

Full standalone `model.safetensors` merged and published into all seven live adapter
repos. Adapter files preserved; `merge_receipt.json` added alongside each merge.

| Repo | Merge commit | Merged size |
|---|---|---|
| SZLHOLDINGS/brain-navigator-r2 | b635d815 | 2.8 GB |
| SZLHOLDINGS/chaski-5050 | 32337ead | 2.8 GB |
| SZLHOLDINGS/chaski-r2 | 9e37ff05 | 2.8 GB |
| SZLHOLDINGS/KHIPU-R2 | 825ec2f0 | 5.75 GB |
| SZLHOLDINGS/khipu-r3 | 1d6228ff | 2.8 GB |
| SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3 | ed7a3f69 | 2.8 GB |
| SZLHOLDINGS/WILLAY | 375a007e | 1.84 GB |

## KHIPU-R2 — local train + MEASURED eval

- Trained locally on RTX 5050: 45 epochs, final loss 0.0175, training receipt in repo.
- Held-out MEASURED: plan 11/11, navigate 5/5, abstain 2/6, hallucinations 0.
- Discrepancy: a prior hub eval reported abstain 3/6 — under investigation
  (decode params, quantization, or revision drift).
- Weak gate: abstention. 4-bit NF4 apples-to-apples eval queued.

## v2 defect — inert adapter under standard PEFT

- `szl-receiptagent-qwen35-0.8b-v2` adapter keys do not match PEFT target-module
  naming for the base; `PeftModel` load silently drops every key.
- Verified: merged output byte-equal to base (empty merge). v2 excluded from the
  merge/publish wave.
- Card advisory written; push pending token rotation (session token exposed).
- Salvage drafted (`salvage_v2_adapter.py`): rebuilds `target_modules` from the
  checkpoint's own keys, re-merges, and requires a non-zero weight delta vs base
  before anything is published.
- v3 is the recommended clean adapter (25.6 MB, merges cleanly).

## Hygiene

- HF token exposed in session logs during publish — must be rotated before the
  next run.
- Disk recovered 0.6 → 8.3 GB free; byte-copy v2 merge deleted twice; KHIPU-R2
  verified intact (5.75 GB) after a disk-full crash mid-rewrite.
- venv rebuilt; router-level DNS block against huggingface.co resolved.

## Open queue

1. v2 card advisory push (after token rotation)
2. v2 key-rename salvage + re-merge + verification
3. KHIPU-R2 4-bit NF4 apples-to-apples abstain eval (`szl_heldout_runner.py --compare-4bit`)
4. chaski-r2 held-out hardening eval
5. Merge-receipt review for the remaining five repos
