# frontier/ — evaluation lanes, contracts, and the evidence shelf

This directory is where the estate decides what is allowed to matter. Nothing here promotes a model, authorizes production, or writes to the Hub — everything here grades evidence and fails closed.

## Read this directory in 60 seconds

1. `harness/LANE_CONTRACT.md` — the five invariants every runner obeys. Read it before reading any lane.
2. `evidence/` — declared baselines anchored to signed receipts. Numbers here are MEASURED; anything not here is not a line.
3. Everything else is either a **contract** (`*_contract.py` — validates receipts, authorizes nothing) or an **evaluation record** (`*_evaluation.json` — what a lane actually observed).

## The map (as of 2026-09-11)

### Lane contracts (validate receipts; authorize nothing)

- `repo2rlenv_eval_contract.py` — Repo2RLEnv task-synthesis lane: exact upstream pin (huggingface/Repo2RLEnv v0.8.7), oracle-leak fail-closed, provider/Hub/training hard-denied, repeatability as byte-stable digests
- `harbor_eval_contract.py` — Harbor reproducibility lane
- `harbor_private_datasets_contract.py` — Harbor private-dataset safety boundary
- `harbor_provider_refs_contract.py` — Harbor provider-reference safety contract
- `asyncgrpo_lora_contract.py` — TRL AsyncGRPO + LoRA + vLLM sync lane

### Evaluation records (what lanes observed)

- `tau_042_evaluation.json` — Tau agent-harness sandbox, rebound to stable 0.4.2
- `ghlore_030_evaluation.json` — Ghlore project-memory lane, rebound to 0.3.0
- `receiptagent_tournament.json`, `model_bakeoff_manifest.json` — earlier tournament and bake-off records

### Evidence shelf

- `evidence/khipu-abstain-baseline-2026-09-08.json` — the L2 declared line (abstain 3/6) anchored to its signed owner-metal receipt
- `evidence/` convention: `kind: szl-frontier-baseline-evidence`, receipt counts and identifiers only, never hidden-set content

### Harness

- `harness/` — L1 (self-check), L2 (khipu abstention bench), L3 (controller operating point), the shared verify path, and `LANE_CONTRACT.md`

### Intake and history

- `HF_FRONTIER_INTAKE_2026-09-06.md`, `K2_HORIZON_*`, `PINNED_MODEL_WAVE_2026-09-07.json` — the intake wave that seeded this directory; keep for lineage

## Conventions

- New lane = one `*_contract.py` + tests that mirror every gate + an `*_evaluation.json` when the lane runs + a baseline-anchor JSON in `evidence/` when a line freezes + a tracker update in szl-hf-frontier#9 in the same change
- Exact-source always: upstream pins are 40-hex revisions and named releases; "latest" is never an input
- Fail closed always: missing capability is HOLD, unreadable evidence is FAIL, self-authored evidence is FAIL
- Contracts never clone repos, call providers, run containers, publish, or promote — they grade receipts and stop

## What does not live here

Hidden evaluation sets (by design), model weights, runner secrets, and anything that can authorize. If a file in this directory can spend, publish, or promote, it is misfiled — treat that as a finding.
