# Wave 29 — Untrained Estate Model Candidates

One config per genuine untrained model candidate (audit of 2026-11-06: 122 models,
82 with weights; the no-weight repos split into these candidates plus 20 code/spec
mirrors now classified CODE_SPEC_MIRROR on the Hub).

## Run

On an A6000 node with `HF_TOKEN` set: `bash run_w29.sh`.
`W29_PLAN=1 bash run_w29.sh` prints the same decisions and does not train or push.

`safety.py` choosing RUN is not launch authorization. `train_wave.py` rejects
the explicit arguments this shell passes, and rejects a config that has not
cleared `training/launch_contract.py`, before it loads a model.

Only three configs are runnable. Each uses `train_wave.py` QLoRA (merged-16bit
push, heartbeat log to wave-logs):

- `willay-r1` → `SZLHOLDINGS/willay-r1-candidate`
- `willay-r2` → `SZLHOLDINGS/willay-r2-candidate`
- `forge-v3` → `SZLHOLDINGS/forge-v3`

`training/w29/safety.py` allowlists those three target repos. `run_w29.sh` calls
it before any trainer process. Every other config is skipped. A `method: qlora`
tag on any other target refuses the wave, so retagging a protected repo cannot
SFT and push into it.

A keyless Hub API read of `willay-r1-candidate`, `willay-r2-candidate`, and
`forge-v3` returned HTTP 401. The same 401 was returned for a repo id that does
not exist, so this checkout could not tell whether those three repos exist or
what they contain. That is UNKNOWN, not a verification that they are empty.
They are allowlisted only as this wave's causal-LM candidate ids. They are not
the live numpy fixtures, the YARQA-ATTN kernel, the detector kits, the VLM, or
the logistic models below. Confirm the target before a non-plan run.

## Skipped — live non-causal-LM artifacts

These configs stay in the catalog with `method: special-trainer`. Running
`train_wave.py` against them would push a merged 0.8B causal-LM checkpoint into
a live non-LLM repo.

- `yarqa-attn` — attention kernel (`yarqa_attn.npz` plus the kernels-repo pointer), not a model checkpoint
- `miniembed-nano`, `chakana`, `tinku` — embedding or reranker kits (contrastive trainer)
- `receiptagent-nano`, `tinykhipu-nano`, `moons-nano` — numpy weight fixtures (`receipt_agent.npz`, `tiny_khipu.npz`, `moons.npz`), not causal LMs
- `qantu` — VLM (vision-language trainer)
- `waman`, `killinchu-eye` — RF-DETR detectors (`KILLINCHU-EYE` is an alias of waman, not a second kit)
- `oac-clinical-health`, `oac-ops-health`, `oac-system-health` — logistic health models
- `triage-linear-v1`, `triage-retrain` — linear text classifiers

## Skipped — causal-LM candidate, wrong trainer

- `khipu-abstain-r3` → `SZLHOLDINGS/khipu-abstain-r3-candidate` is a report-only
  abstention candidate (public README and `training-eval-report.json`, no weights;
  base recorded as Qwen2.5-1.5B-Instruct). It is a causal-LM candidate, not a
  numpy or kernel artifact. Two DPO rounds failed the recorded 3/6 confabulation
  floor, so this round stays reserved for rejection sampling. The runner does
  not plain-SFT it and does not push a merged checkpoint over that report.

## Honesty contract

- Each runnable SFT writes an eval receipt to SZLHOLDINGS/model-eval-receipts — win or lose.
- No promotion, no publication_eligible, no LIVE label from a training run alone.
- Dataset choices are PROPOSALS pending collaborator agreement where applicable.
- Skip and run decisions are the `safety.py` policy above. The README names the same repos.
