# KHIPU-R2

Forge kit for a **separate SKU**. Live Hub [`SZLHOLDINGS/KHIPU-R2`](https://huggingface.co/SZLHOLDINGS/KHIPU-R2) is **not empty**: job [`6a91bf11984507d9db4ea104`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf11984507d9db4ea104) **COMPLETED**, adapter **147.8MB AVAILABLE**, abstain **MEASURED 3/6** (not a pass).

This checkout does not fire a job (`jobs` UNKNOWN for this kit) and does not re-run held-out generate (this-SKU evals not-this-run). No Hub PUT. Does not overwrite signed [`SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B). House CPU lab stays signed Khipu GGUF.

| | |
|---|---|
| **SKU** | `KHIPU-R2` — separate from signed [`SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) |
| **Base (canonical)** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Runtime train** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| **Hub job** | [`6a91bf11`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bf11984507d9db4ea104) **COMPLETED** |
| **Hub adapter** | **AVAILABLE** (147.8MB `adapter_model.safetensors`) |
| **Hub abstain** | **MEASURED 3/6** (not a pass) |
| **This-kit jobs** | **UNKNOWN** — this PR does not fire a job |
| **This-SKU evals** | not-this-run |
| **publication_eligible** | **false** |
| **Hub PUT** | **false** |
| **Lab** | signed Khipu GGUF (no KHIPU-R2 pin) |
| **Doctrine** | v11 LOCKED (749 / 14 / 163). Λ = Conjecture 1 |

## What this is

A governed retrieval-plan abstain retrain: same synthetic Khipu curriculum
as `khipu/`, with in-memory `ABSTAIN_OVERSAMPLE=4` (8×4 abstain vs 15
navigate). Held-out `eval.jsonl` (5) and `adversarial.jsonl` (6) never enter
gradients. Seed 11. LoRA r=32 α=64, 45 epochs, response-only CE.

This repository ships **one** trainer: `khipu_r2/train_khipu_r2.py`. Hub also
hosts a leftover doctrine-SFT file of the same name; that is not a second
forge recipe.

`train_khipu_r2.py` **refuses** `--hub SZLHOLDINGS/SZL-Khipu-1.5B`.

## What this is NOT

- Not a replacement for signed `SZLHOLDINGS/SZL-Khipu-1.5B`
- Not a passing abstain gate (Hub MEASURED 3/6 is not a pass)
- Not a Hub overwrite, merge, or Jobs launch from this checkout
- Not a house-lab pin (lab stays signed Khipu GGUF)
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
