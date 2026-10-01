# frontier/ — evaluation lanes, contracts, and the evidence shelf

This directory is where the estate decides what is allowed to matter. Nothing here promotes a model, authorizes production, or writes to the Hub — everything here grades evidence and fails closed.

## Read this directory in 60 seconds

1. `harness/LANE_CONTRACT.md` — the five invariants every runner obeys. Read it before reading any lane.
2. `evidence/` — declared baselines anchored to signed receipts. Numbers here are MEASURED; anything not here is not a line.
3. `evaluation/` — the hosted GLM/Khipu runner, the sealed-run counter, and the exact-string fixture contract.
4. Everything else is either a **contract** (`*_contract.py` — validates receipts, authorizes nothing) or an **evaluation record** (`*_evaluation.json` — what a lane actually observed).

## The map (as of 2026-09-11, head after #232)

### Hosted evaluation plane (`evaluation/`)

- `evaluation/runner.py` — bounded hosted candidate-versus-Khipu runner. Provider-unavailable seals stay local; `--publish` on that path updates only the public counter.
- `evaluation/sealed_count.py` — W1. Public `runs/SEALED_COUNT.json` schema: UTC dates and integer counts, hash-chained, fail-closed on extra keys. Genesis `total_sealed=0` means no counted seals since the counter existed.
- `evaluation/FIXTURES.md` — W2. Documents each public `answer_equals` string. Paraphrases are rejected. `fixtures.v1.json` stays byte-pinned at `588299a6d926bb979a7a42316d781f3469220062399aacd0c22d859922ef2ee6`.
- `evaluation/core.py`, `receipt.py`, `provider.py`, `failure.py`, `terminal.py`, `source.py` — runner internals. They do not promote.

Public Hub path: `SZLHOLDINGS/szl-frontier-evaluation-receipts` / `runs/SEALED_COUNT.json`. As of 2026-09-11 19:01Z that file is **absent** (`EntryNotFound`). The honest observation is **UNAVAILABLE**, not a filled zero. Pre-counter silence after the 2026-09-08 GLM payloads is the same state.

Watcher (outside this directory, do not rebuild): `inference/sealed_count_watch.py` (#232). Missing file → `observationStatus=UNAVAILABLE`. Valid genesis → `WATCH` with honest `0`. Disposition HOLD / NONE.

### Lane contracts (validate receipts; authorize nothing)

- `repo2rlenv_eval_contract.py` — Repo2RLEnv task-synthesis lane: exact upstream pin (huggingface/Repo2RLEnv v0.8.7), oracle-leak fail-closed, provider/Hub/training hard-denied, repeatability as byte-stable digests
- `harbor_eval_contract.py` — Harbor reproducibility lane
- `harbor_private_datasets_contract.py` — Harbor private-dataset safety boundary
- `harbor_provider_refs_contract.py` — Harbor provider-reference safety contract
- `asyncgrpo_lora_contract.py` — TRL AsyncGRPO + LoRA + vLLM sync lane
- `asyncgrpo_lora_runner.py` — job-local protocol + recorded PEFT/vLLM closure; GPU path talks to TRL `VLLMClient` endpoints and stays UNAVAILABLE without CUDA
- `asyncgrpo_lora_gpu.py` — adapter-only `load_lora_adapter` vs merged process-reload; env weights are staged into the job-local `.vllm_lora` cache; negatives include no-LoRA, restart, eviction, symlink-at-use, partial adapter

### Evaluation records (what lanes observed)

- `tau_042_evaluation.json` — Tau agent-harness sandbox, rebound to stable 0.4.2
- `ghlore_030_evaluation.json` — Ghlore project-memory lane, rebound to 0.3.0
- `receiptagent_tournament.json`, `model_bakeoff_manifest.json` — earlier tournament and bake-off records
- `chaski_research_only.json` — Chaski stays research-only until a held-out gate exists

A 2026-09-11 hosted GLM-5.3-Flash full-suite receipt exists on the Hub dataset (`runs/2026/09/11/github-34621084074-1-a0a6e5d246e7`). Decision `EVIDENCE_COMPLETE_REVIEW_REQUIRED`. `production_disposition` **HOLD**. That is not promotion and is not a sealed-unavailable count.

### Evidence shelf

- `evidence/khipu-abstain-baseline-2026-09-08.json` — the L2 declared line (abstain 3/6) anchored to its signed owner-metal receipt
- `evidence/glm-5-3-flash-provider-evaluation-2026-09-08.json` — GLM provider-evaluation anchor from the six 2026-09-08 published runs
- `evidence/sealed-run-counter-genesis-2026-09-11.json` — W1 genesis pin (`total_sealed: 0`). That zero is not a fill-in for pre-counter silence.
- `evidence/` convention: `kind: szl-frontier-baseline-evidence`, receipt counts and identifiers only, never hidden-set content

### Harness

- `harness/` — L1 (self-check), L2 (khipu abstention bench), L3 (controller operating point), the shared verify path, and `LANE_CONTRACT.md`
- L2 must **strictly beat** the signed 3/6 hidden-set line. Public fixtures are not evidence. Ties fail. Even a beat stays `MODEL_PROPOSED`.
- L4+ is deliberately unfrozen. Charters come from `szl-hf-frontier#9` and the owner, not from this directory.

### Intake and history

- `HF_FRONTIER_INTAKE_2026-09-06.md`, `K2_HORIZON_*`, `PINNED_MODEL_WAVE_2026-09-07.json` — the intake wave that seeded this directory; keep for lineage
- `khipu-abstain-20260908/`, `minicpm5/`, `qwen35-receiptagent-v2/`, `qwen35-receiptagent-v3/` — family-specific curricula and qualification code. Presence is not production admission.

## Conventions

- New lane = one `*_contract.py` + tests that mirror every gate + an `*_evaluation.json` when the lane runs + a baseline-anchor JSON in `evidence/` when a line freezes + a tracker update in szl-hf-frontier#9 in the same change
- Exact-source always: upstream pins are 40-hex revisions and named releases; "latest" is never an input
- Fail closed always: missing capability is HOLD, unreadable evidence is FAIL, self-authored evidence is FAIL, missing public counter is UNAVAILABLE not zero
- Contracts never clone repos, call providers, run containers, publish, or promote — they grade receipts and stop

## What does not live here

Hidden evaluation sets (by design), model weights, runner secrets, Cloudflare keys, Hub write-repos tokens, and anything that can authorize. If a file in this directory can spend, publish, or promote, it is misfiled — treat that as a finding.
