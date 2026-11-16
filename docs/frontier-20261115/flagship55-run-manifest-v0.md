# Flagship55 Run Manifest v0 — PROPOSED (owner pin pending)

🜂 Estate artifact — fail-closed. This is a draft pinned-input manifest. The owner must pin the ratios before any launch. Nothing here authorizes a launch by itself.

## Pinned inputs (sha256-bound)

| Input | Revision / sha256 | Proposed share (v0 draft) |
|---|---|---|
| flagship54 base mix | owner-pinned at flagship54 launch (unchanged) | 70% |
| [lean-math-sft-v1](https://huggingface.co/datasets/SZLHOLDINGS/lean-math-sft-v1) | corpus e80aebb2…; train_file sha256 3e02f8f9e695bb5885fda1ea2c73495cc4926c5ec14a9356ceb59e96665fbcad | 20% |
| [axiom-invariant-corpus-v1](https://huggingface.co/datasets/SZLHOLDINGS/axiom-invariant-corpus-v1) | lutar-lean bdf8255…; jsonl sha256 325ba8d63537d4eacce5673bac8bfffe98c5545b7985652ace2fed0063f269a9 | 5% (statement-context, not completions) |
| receipt-of-doubt-math-unanswerables-v1 | dataset revision pinned at launch (private) | 5% (paired abstention) |

## Admission rule (binding)

Only lake-build-accepted (sorry-gate green) proof traces enter any corpus v2 feeding this lane. No auto-merge; PhD/owner gate per CAMPAIGN.md.

## Launch gates (from the merged curriculum spec, PR #746)

- [ ] flagship54 finished + evaluated, receipts bound
- [ ] Owner pins the ratios (this v0 draft is not a pin)
- [ ] Salad lane selected + cost check
- [ ] This manifest committed with final digests at launch time

## Benchmark commitment (honest)

External yardstick: miniF2F + Lean-Workbook sample; publish the measured number next to DeepSeek-Prover-V2's reported 88.9% miniF2F pass rate, whatever ours is. A receipt-bound low score beats a fabricated high score (doctrine).
