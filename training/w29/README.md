# Wave 29 — Untrained Estate Model Candidates

One config per genuine untrained model candidate (audit of 2026-11-06: 122 models,
82 with weights; the no-weight repos split into these candidates plus 20 code/spec
mirrors now classified CODE_SPEC_MIRROR on the Hub).

## Run
On an A6000 node with `HF_TOKEN` set: `bash run_w29.sh`. Only `qlora`-method
configs run — the six causal-LM SFT candidates (receiptagent-nano, tinykhipu-nano,
moons-nano, willay-r1, willay-r2, forge-v3), each on the proven train_wave.py
pipeline (merged-16bit push, heartbeat log to wave-logs).

## Special trainers (configs present, run_w29.sh SKIPS them — fail-closed)
These repos hold NON-causal-LM artifacts. Running train_wave.py against them would
push a merged 0.8B causal-LM checkpoint INTO a live kernel/detector/VLM/logistic
repo, overwriting the artifact. Each config carries `method: special-trainer` and
the required trainer in `notes`:
- `miniembed-nano`, `chakana`, `tinku` — embedding/reranker kits (contrastive trainer)
- `khipu-abstain-r3` — rejection sampling then SFT (two DPO rounds failed the 3/6 floor)
- `qantu` — VLM (vision-language trainer)
- `yarqa-attn` — attention kernel (not a model checkpoint)
- `waman`, `killinchu-eye` — RF-DETR detectors (detection trainer)
- `oac-*-health` — logistic health models (classification trainer)
- `triage-linear-v1`, `triage-retrain` — linear text classifiers (classification trainer)

## Honesty contract
- Each run writes an eval receipt to SZLHOLDINGS/model-eval-receipts — win or lose.
- No promotion, no publication_eligible, no LIVE label from a training run alone.
- Dataset choices are PROPOSALS pending collaborator agreement where applicable.
