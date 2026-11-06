# Wave 29 — Untrained Estate Model Candidates

One config per genuine untrained model candidate (audit of 2026-11-06: 122 models,
82 with weights; the no-weight repos split into these candidates plus 20 code/spec
mirrors now classified CODE_SPEC_MIRROR on the Hub).

## Run
On an A6000 node with `HF_TOKEN` set: `bash run_w29.sh`. Each QLoRA run uses the
proven train_wave.py pipeline (merged-16bit push, heartbeat log to wave-logs).

## Special trainers (configs present, NOT runnable by train_wave.py)
- `miniembed-nano` (sentence-similarity): requires a contrastive embedding trainer
  (sentence-transformers class), not SFT. Config kept for the dataset mapping.
- `khipu-abstain-r3` (abstention): two DPO rounds failed to break the measured 3/6
  confabulation floor; this round is reserved for rejection sampling — generate,
  filter by grounding check, SFT on survivors. Do NOT run as plain SFT/DPO.

## Honesty contract
- Each run writes an eval receipt to SZLHOLDINGS/model-eval-receipts — win or lose.
- No promotion, no publication_eligible, no LIVE label from a training run alone.
- Dataset choices are PROPOSALS pending collaborator agreement where applicable.
