# KHIPU-R2

**Separate SKU. ROADMAP. Not an overwrite of signed `SZL-Khipu-1.5B`.**

This kit is the 2/6 abstain-blocker successor for the Khipu line. It ships
one Unsloth QLoRA trainer, an honest eval stamp, a Python serve path, and a
Jobs launcher that **refuses to fire**. This checkout does not PUT Hub.

| | |
|---|---|
| **SKU** | `KHIPU-R2` — separate from signed [`SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) |
| **Base (canonical)** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Runtime train** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| **Status** | **ROADMAP** |
| **HF Jobs** | **UNKNOWN** — this PR does not fire a job |
| **Signed original abstain** | **MEASURED 2/6** (blocker, unchanged on signed 1.5B) |
| **This-SKU evals** | not-this-run |
| **publication_eligible** | **false** |
| **Hub PUT** | **false** |
| **Doctrine** | v11 LOCKED (749 / 14 / 163). Λ = Conjecture 1 |

## What this is

A governed retrieval-plan abstain retrain: same synthetic Khipu curriculum
as `khipu/`, with in-memory `ABSTAIN_OVERSAMPLE=4` (8×4 abstain vs 15
navigate). Held-out `eval.jsonl` (5) and `adversarial.jsonl` (6) never enter
gradients. Seed 11. LoRA r=32 α=64, 45 epochs, response-only CE.

Hub currently also hosts a leftover doctrine-SFT file named
`train_khipu_r2.py`. This repository ships **one** trainer:
`khipu_r2/train_khipu_r2.py` — the abstain-retrain recipe, not a second
doctrine-SFT recipe.

`train_khipu_r2.py` **refuses** `--hub SZLHOLDINGS/SZL-Khipu-1.5B`.

## What this is NOT

- Not a replacement for signed `SZLHOLDINGS/SZL-Khipu-1.5B`
- Not a publication of a passing abstain gate (2/6 remains the signed blocker)
- Not a Hub overwrite, merge, or Jobs launch from this checkout
- Not an autonomous agent

## Commands (local, no Hub write)

```bash
python khipu_r2/train_khipu_r2.py
python khipu_r2/eval_khipu_r2.py
python khipu_r2/serve_khipu_r2.py --check
python khipu_r2/jobs/launch_khipu_r2_job.py
```

`--train` runs Unsloth on the owner's metal and writes a **local** adapter
under `khipu_r2/khipu-r2-adapter/`. It still does not PUT Hub and still
leaves `publication_eligible` false.

`--run-job` on the launcher exits 2 (`refusing`).
