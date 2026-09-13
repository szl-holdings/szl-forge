# HF Frontier Delta — 2026-09-12

Local Omen pipeline night (RTX 5050 laptop, CPU merge). Delta since `HF_FRONTIER_INTAKE_2026-09-06.md`.

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
