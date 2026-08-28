# CHASKI-R2

**Separate SKU. CHAWPI silhouette. Not an overwrite of live `SZLHOLDINGS/chaski`.**

This kit is the JSON-draft / refuse-abstain successor for the Chaski line on
`Qwen/Qwen3.5-0.8B` (Apache-2.0). It ships one Unsloth QLoRA trainer, an honest
eval stamp that **reuses** the parent named-N gates, a Python serve path, and a
Jobs launcher that **refuses to fire**. This checkout does not PUT Hub.

| | |
|---|---|
| **SKU** | `CHASKI-R2` — Hub [`SZLHOLDINGS/chaski-r2`](https://huggingface.co/SZLHOLDINGS/chaski-r2) |
| **CANONICAL_BASE** | `Qwen/Qwen3.5-0.8B` (Apache-2.0). Disclosed. |
| **Runtime train** | `unsloth/Qwen3.5-0.8B` QLoRA |
| **LoRA** | r=16 α=32, seed 11, response-only CE |
| **Not** | live `SZLHOLDINGS/chaski`, `SZLHOLDINGS/chaski-5050`, bf16, owner-metal sixteen-alpha |
| **Status** | GPU honesty **MEASURED** or **UNAVAILABLE**. **No ROADMAP parking.** |
| **HF Jobs** | **UNAVAILABLE** — this PR does not fire a job |
| **Weights** | **UNAVAILABLE** this checkout |
| **Evals** | none-this-run / **UNAVAILABLE** (named-N wired, generate not run) |
| **publication_eligible** | **false** until MEASURED generate |
| **Hub PUT** | **false** |
| **Lab** | House CPU lab stays **Khipu GGUF** |
| **A11OY-MINI** | PR #61 stays **scripts-only** of live Chaski. Not this SKU. |
| **Doctrine** | v11 LOCKED (749 / 14 / 163). Λ = Conjecture 1 |

## What this is

A governed courier silhouette: **new** train turns that emit JSON with
`decision`, `approvalRequired`, `executed`, `artifact`, `base_model`, `claim`,
`label`, plus a REFUSE/ABSTAIN line. Held-out eval **reuses**
`chaski/gate/json_drafts.n5.jsonl` and `chaski/gate/adversarial_refusals.n6.jsonl`
(eval-only named-N files from PR 63). Those files never enter gradients.

`train_chaski_r2.py` **refuses** `--hub SZLHOLDINGS/chaski` and
`--hub SZLHOLDINGS/chaski-5050`. It **refuses** `--dataset-file` pointing at
`chaski/gate/*.jsonl`.

## What this is NOT

- Not a replacement for live `SZLHOLDINGS/chaski`
- Not `SZLHOLDINGS/chaski-5050` and not a bf16 / sixteen-alpha owner-metal kit
- Not a publication of a passing JSON-draft or refusal gate
- Not a Hub overwrite, merge, or Jobs launch from this checkout
- Not a Khipu lab pin and not an A11OY-MINI GGUF
- Not an autonomous agent

## Commands (local, no Hub write)

```bash
python chaski_r2/train_chaski_r2.py
python chaski_r2/eval_chaski_r2.py
python chaski_r2/serve_chaski_r2.py --check
python chaski_r2/jobs/launch_chaski_r2_job.py
```

`--train` runs Unsloth on the owner's metal and writes a **local** adapter
under `chaski_r2/chaski-r2-adapter/`. Train loss may be MEASURED as a train
metric, not an eval. It still does not PUT Hub and still leaves
`publication_eligible` false.

`--run-job` on the launcher exits 2 (`refusing`).
