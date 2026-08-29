# CHASKI-R2

**Separate SKU. CHAWPI silhouette. ATELIER lock.**

This is a GitHub recipe, not a Hub card. The reserved Hub id is
`SZLHOLDINGS/chaski-r2`. Do **not** costume a README-only Hub ID: this checkout
does not PUT Hub and does not claim a live Hub page.

Base in prose: **Qwen/Qwen3.5-0.8B** (Apache-2.0). Disclose that silhouette.
Never overwrite live `SZLHOLDINGS/chaski`. Not `SZLHOLDINGS/chaski-5050`.
Not bf16. Not the owner-metal sixteen-alpha kit.

It ships one Unsloth QLoRA trainer, an honest eval stamp that **reuses** the
PR 63 named-N gates **after train**, a Python serve path, and a Jobs launcher
that **refuses to fire**.

| | |
|---|---|
| **SKU** | `CHASKI-R2` — reserved Hub id `SZLHOLDINGS/chaski-r2` (declared, not a Hub page) |
| **Base** | Qwen/Qwen3.5-0.8B (Apache-2.0). Disclosed in prose. |
| **Runtime train** | `unsloth/Qwen3.5-0.8B` QLoRA |
| **LoRA** | r=16 α=32, seed 11, response-only CE |
| **Not** | live `SZLHOLDINGS/chaski`, `SZLHOLDINGS/chaski-5050`, bf16, owner-metal sixteen-alpha |
| **ATELIER lock** | GitHub recipe only. No Hub YAML card. No README-only Hub costume. |
| **Status** | GPU honesty **MEASURED** or **UNAVAILABLE**. **No ROADMAP parking.** |
| **HF Jobs** | **UNAVAILABLE** — this PR does not fire a job |
| **Weights** | **UNAVAILABLE** this checkout |
| **Evals** | PR 63 named-N after train; **none-this-run** until that generate |
| **publication_eligible** | **false** until MEASURED generate |
| **Hub PUT** | **false** |
| **Lab** | House CPU lab stays **Khipu GGUF** |
| **A11OY-MINI** | PR #61 stays **scripts-only** of live Chaski. Not this SKU. |
| **Doctrine** | v11 LOCKED (749 / 14 / 163). Λ = Conjecture 1 |

## What this is

A governed courier silhouette: **new** train turns that emit JSON with
`decision`, `approvalRequired`, `executed`, `artifact`, `base_model`, `claim`,
`label`, plus a REFUSE/ABSTAIN line. Eval **after train** reuses
`chaski/gate/json_drafts.n5.jsonl` and `chaski/gate/adversarial_refusals.n6.jsonl`
(eval-only named-N files from PR 63). Those files never enter gradients.
Evals stay **none-this-run** until that generate actually runs.

`train_chaski_r2.py` **refuses** `--hub SZLHOLDINGS/chaski` and
`--hub SZLHOLDINGS/chaski-5050`. It **refuses** `--dataset-file` pointing at
`chaski/gate/*.jsonl`.

## What this is NOT

- Not a replacement for live `SZLHOLDINGS/chaski`
- Not `SZLHOLDINGS/chaski-5050` and not a bf16 / sixteen-alpha owner-metal kit
- Not a publication of a passing JSON-draft or refusal gate
- Not a Hub overwrite, merge, Hub card, or Jobs launch from this checkout
- Not a costumed README-only Hub ID
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
