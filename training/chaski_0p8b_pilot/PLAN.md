# Chaski 0.8B training pilot

Status: **prepared, not launched**. This directory does not start a GPU job,
download weights, or write to the Hugging Face Hub. `launch_authorized` is
false on every manifest. `--execute` on the teacher and student entry points
exits before a model load.

The Q3 approval this plan spends against is **16 GPU-h** for Chaski 0.8B
training, plus a Nemotron teacher for distillation data. The hard boxes below
sum to **6 GPU-h**. The expected use is about **1.5 GPU-h**. The rest of the
16 GPU-h is not allocated.

## Models

| Role | Repo | Revision (Hub API `sha`, 2026-11-27) | License check |
| --- | --- | --- | --- |
| Student | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Hub `cardData.license` is `apache-2.0`. The `LICENSE` file at that revision is the Apache License, Version 2.0. Architecture on the card is `Qwen3_5ForConditionalGeneration`. |
| Teacher | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` | Hub `license_name` is `nvidia-nemotron-open-model-license`. That is the NVIDIA Nemotron Open Model License (last modified 15 December 2025), not the separate `nvidia-open-model-license` used by Nano-9B-v2. |

The forge Unsloth wave is pinned to TRL 0.23.1 and cannot train Qwen3.5.
Training uses the image

`ghcr.io/szl-holdings/szl-train-pinned@sha256:8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235`

(transformers 5.22.2, trl 1.17.1, peft 0.21.2). The transformers-5 lane for
Nemotron-H is `kallpa/nemotron-h-trainer` (PR 804). This pilot is based on
that branch and calls `training/trainer_compat.py` for the teacher pin,
guardrails, reasoning toggle, and NOTICE writer. It does not copy those
helpers. The student path is new because Qwen3.5 is not a Nemotron-H graph.

## Teacher attribution and NOTICE

License: [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).

Obligations that apply to distilled data and to a student trained on it:

1. NVIDIA does not claim ownership of outputs generated with the model.
2. Redistribution of the model or of a derivative must give recipients the
   license, and must keep the copyright, patent, trademark, and attribution
   notices that belong to the reused part (license sections 3(a) and 3(b)).
3. A distributed derivative must include a Notice file with this sentence,
   which is the license's section 3(c) text and the lane's pinned NOTICE:
   `Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.`
4. NVIDIA trademarks are not licensed, except for the attribution needed to
   name the teacher and to reproduce that NOTICE (section 4).
5. The model card's guardrails stay in place: `nvidia_guardrails: kept`,
   `system_prompt: preserve`, and the pinned chat template, byte for byte.
   Phase 1 sets `enable_thinking` to false so the courier completions stay
   short. The toggle is explicit. An omitted toggle is refused by the lane.

Phase 1 writes `NOTICE` next to `teacher.jsonl`. Phase 2 writes the same
NOTICE into the student artifact directory. The student repo is not named as
a Nemotron model. Neither phase is an operator admission receipt, and neither
sets `commercial_use: ADMITTED`.

## Output repos

Both names were absent on the Hub on 2026-11-27.

| Artifact | Repo |
| --- | --- |
| Teacher JSONL (phase 1 output, phase 2 input) | `SZLHOLDINGS/chaski-0p8b-pilot-sft` |
| Student LoRA (phase 2 output) | `SZLHOLDINGS/chaski-0p8b-pilot` |

`SZLHOLDINGS/chaski` and `SZLHOLDINGS/chaski-r4` are refused, including case
variants. This pilot does not push to either repo.

## Prompt sources

Assistant turns are dropped. The teacher writes the completion. Exact user
text from the held-out gates is removed. A paraphrase check is **not** done.
That check stays UNKNOWN. It is not a pass.

Held out, never training (sha256 matches `chaski_r4/heldout_manifest.json`):

| File | sha256 | User prompts |
| --- | --- | --- |
| `chaski/gate/json_drafts.n5.jsonl` | `dd093403e4aeeb906c169d8b6d8229115b1d170a0226afa939c1beaa6fa5e9be` | 5 |
| `chaski/gate/adversarial_refusals.n6.jsonl` | `1e58dda9cf07e22406f0f953fddc2c26397fb5e784c27516578d133035cb006d` | 6 |

Copies of those gates under the r2 evidence tree are held out as well.

Admitted, prompts only. The license basis is this repository's Apache-2.0
license plus first-party authorship. There is no second dataset-license file.
`operator_admission` is false.

| File | Prompts kept | Why it is in |
| --- | --- | --- |
| `szl_dataset.jsonl` | 41 | The only file `chaski/train_chaski.py` loads. |
| `chaski_r2/train.jsonl` | 10 | The r2 train turns named by `chaski_r2/README.md`. |
| `chaski_r4/train.jsonl` | 40 | The r4 curriculum named by `chaski_r4/training_plan.json`. |

91 prompts. Zero exact overlaps with the 11 held-out user prompts.

UNKNOWN, excluded:

| File | Why |
| --- | --- |
| `szl_dataset_5050_merged.jsonl` | Merge. Rows beyond `szl_dataset.jsonl` have no separate license record. |
| `chaski_r4/train.pre-fix-backup.jsonl` | Superseded backup. The fix is not re-audited here. |
| `chaski_r4/evidence/geh/*.jsonl` | Evidence logs, not a licensed prompt source. |
| Any other Chaski JSONL without a recorded basis | Same rule. |

## Phases

Caps on every run: **$100 / run**, **$500 / day**, **$5,000 / month**.
Providers: **salad** or **massed** only. The pool's projected cost is
`usd_per_hour * max_hours`. This plan does not invent a provider rate. A
device whose rate makes that product exceed $100 is refused by the pool.
Phase 2's 4 hour box needs a rate at or below $25 per GPU-hour to clear the
run cap. Phase 1's 2 hour box needs a rate at or below $50.

| Phase | Work | GPU | Hard box | Expected | VRAM request |
| --- | --- | --- | --- | --- | --- |
| 1 | Teacher completions, reasoning off, `max_new_tokens` 256, 91 prompts | 1× Nemotron-3 Nano-4B BF16 | 2 GPU-h | 0.5 GPU-h | 24 GB |
| 2 | Student LoRA r=16, alpha=32, 3 epochs, seq 1024, batch 1, grad accum 4, seed 11, assistant-only loss | 1× Qwen3.5-0.8B | 4 GPU-h | 1.0 GPU-h | 16 GB |

Hard total 6 GPU-h of the 16 GPU-h approval. Isolation is one device, not a
shared GPU. Checkpoint interval for a real pool node must stay at or below
4 hours. Plateau settings are on the manifests so the queue schema is
complete. The pool runbook says current trainers do not yet write its metrics
file, so the hard time box is the stop that is actually specified here.

Phase 2 cannot start until phase 1 has produced a corpus, a sha256, and an
immutable dataset revision. `dataset_revision` is null in this checkout.
`training/launch_contract.py` rejects both wave documents (missing dataset
pin, missing admission receipt, missing train file). A rejected contract is
the intended state.

## Go / no-go against the live Chaski baseline

Baseline is the published card for `SZLHOLDINGS/chaski`, not a train loss.

| Field | Value |
| --- | --- |
| Named-N revision | `1c55df8652e9d0f7b84356b1e2d54849165ae884` |
| Date | 2026-08-28 |
| JSON draft | **0/5** |
| Adversarial refusal | **2/6** |
| Label | MEASURED fail |
| Report sha256 | `4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05` |
| `publication_eligible` | false |

Eval files are only the two held-out gates above. They are not in the 91
prompts. A score that does not record the loader class and 100% of the saved
adapter keys applied is not a score. The historical Chaski card required
192/192 for that adapter. This pilot's key count is whatever the new LoRA
saves. Coverage is against that saved set.

**GO** (pilot checkpoint is worth a separate eval review, and still not a
publication):

- JSON draft at least **3/5** (above the live 0/5).
- Adversarial refusal at least **2/6** (no regression versus the live card).
- Exact user-text overlap with the held-out gates is 0.
- Adapter key coverage is complete, and the loader class is recorded.
- Output repo is `SZLHOLDINGS/chaski-0p8b-pilot`.

**NO-GO** when any of those fail. Train loss is not a gate.
`publication_eligible` stays false on a GO. Promotion is a different decision.

`chaski/family_leaderboard.json` records chaski-r2 and chaski-r4-local at
5/5 and 6/6 with `promote: false`. That is a research comparison. It is not
this pilot's go threshold, and it is not a claim that those scores transfer
to the new repo.

## What is still refused

- Launch, image push, Hub write, merge, and force-push.
- Training on held-out, eval, UNKNOWN, or unlicensed rows.
- Overwriting `SZLHOLDINGS/chaski` or `SZLHOLDINGS/chaski-r4`.
- Treating this plan, the CPU dry-run, or a train loss as a named-N result.
