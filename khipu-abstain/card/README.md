---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- governed-agent
- retrieval
- brain-navigator
- grounded-only
- proposal-only
- research-only
- szl-holdings
- khipu
- abstain-retrain
- weights-unreceipted
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  weights: PRESENT_UNRECEIPTED
  jobs: UNAVAILABLE
  evals: none-this-run
  publication_eligible: false
  autonomy_eligible: false
  original_signed_weights: SZLHOLDINGS/SZL-Khipu-1.5B
  successor_with_weights: SZLHOLDINGS/KHIPU-R2
---

> **EXPERIMENT. Adapter bytes present but unreceipted.**
> Evaluators use `SZLHOLDINGS/SZL-Khipu-1.5B` until a receipt binds the bytes on this ID.

> **WEIGHTS ARE PRESENT ON THIS ID — card corrected 2026-09-25.** Earlier versions of
> this card said "NO WEIGHTS IN THIS REPO". That is no longer true. Observed on
> 2026-09-25 (authenticated Hub tree read):
>
> | File | Bytes | SHA-256 (LFS) |
> |---|---:|---|
> | `khipu-abstain-adapter/adapter_model.safetensors` | 147,770,496 | `da0f948b7a6b555cbd50026eafcbb38ae8187b66be450fd1c3ed27f7eff9bef1` |
> | `khipu-f16.gguf` | 3,093,668,832 | `0df16da8dc6d5865370b90880080dfefda028a5e4c0f758825bf40cdcfb93bd2` |
>
> `training_receipt.signed.json` on this ID (Ed25519 keyId `89540347a69b789e`,
> `trainedAt` 2026-09-08T18:44:18Z, finalTrainLoss 0.0250) binds
> `adapterSha256 = bd5a1a92…9f1db` and `weightsArtifactSha256 = 86c33222…3476`.
> **Neither hash matches the bytes above.** The receipt therefore does not
> attest these files. Until a receipt binds them, the weights are
> **PRESENT_UNRECEIPTED**: loadable, not evidence. `khipu-abstain-adapter/README.md`
> is an untouched PEFT template and carries no provenance.
>
> What *is* receipted is the curriculum: `train.jsonl`, `train.abstain.jsonl`,
> `adversarial.jsonl`, `eval.jsonl`, `khipu.schema.json` (hashes in
> `manifest.json` and in the receipt match) and the 23 KB training script.

# SZL-Khipu-1.5B-abstain

**WEIGHTS: PRESENT_UNRECEIPTED.** Bytes exist on this ID; no receipt binds them. Successor adapter with a receipted evaluation is [`SZLHOLDINGS/KHIPU-R2`](https://huggingface.co/SZLHOLDINGS/KHIPU-R2).

QLoRA **adapter** retrain recipe of the existing Khipu line. Raises in-memory
`ABSTAIN_OVERSAMPLE` from 2 to 4 (8×4=32 abstain vs 15 navigate). Proposal-only.
Λ = Conjecture 1. Doctrine v11 LOCKED 749/14/163.

| | |
|---|---|
| **Weights** | **PRESENT_UNRECEIPTED** (adapter 147.8 MB + f16 GGUF 3.09 GB; hashes above; no binding receipt) |
| **Jobs** | **UNAVAILABLE** |
| **Base (canonical)** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **Runtime train** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (same Qwen2.5-1.5B weights, 4-bit) — matches `adapter_config.json` |
| **Relation** | `adapter` (LoRA r=32, α=64, q/k/v/o/gate/up/down; PEFT 0.20.0) |
| **License** | Apache-2.0 |
| **Does NOT overwrite** | [`SZLHOLDINGS/SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) signed weights |
| **Successor with weights** | [`SZLHOLDINGS/KHIPU-R2`](https://huggingface.co/SZLHOLDINGS/KHIPU-R2) (MEASURED abstain 3/6, not a pass) |
| **This is NOT** | the Chaski Qwen3.5 lock |

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Most 'safety LoRAs' teach tone. This one teaches a binary: the handles are not enough. Research-only until abstain beats 2/6.

A specialist in silence. Capability is someone else's LoRA.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Constitutional fine-tune, but only the refuse clause. |
| NVIDIA | A guardrail as weights, not as Colang. |
| Unsloth | QLoRA adapter, proposal-only, research-only tag. |

No equivalent public combination was found among the leaders surveyed above (a snapshot, not an ecosystem-wide novelty claim).

## Intended use

Stack on the navigator. Measure abstain. Do not ship on hope.

## Limitations

- research-only
- proposal-only
- Does not magically fix 2/6 until a signed eval says so.
- The adapter bytes on this ID are not bound by any receipt; treat them as unverified.

Canonical GitHub: [`szl-holdings/szl-forge`](https://github.com/szl-holdings/szl-forge/blob/main/khipu-abstain/)
<!-- SZL-ATELIER-CUT:v1:END -->

## Evaluation

**Status: NOT YET RUN.** No fabricated k/n. `publication_eligible` is false until
the held-out eval in `train_khipu_abstain.py` actually executes after training
and its receipt binds the adapter hash above.

Prior original MEASURED abstain on `SZLHOLDINGS/SZL-Khipu-1.5B` is **2/6** (blocker).
Eval protocol: `eval.jsonl` 5 navigate + `adversarial.jsonl` 6 abstain. Report k/n only.

## Training (this job)

- Unsloth QLoRA, seed 11, lr 2e-4, adamw_8bit, `train_on_responses_only`, Trackio
- LoRA r=32 α=64, 45 epochs, ga=2, batch=1, constant_with_warmup (from `train_khipu.py`)
- Train: `train.jsonl` 15 navigate + `train.abstain.jsonl` 8 rows × 4
- Held-out never in gradients
- Script: [`train_khipu_abstain.py`](train_khipu_abstain.py)

## Intended use

Proposal-only JSON retrieval plans (`NAVIGATE` / `ABSTAIN`) over synthetic Brain
node handles. A controller outside the weights validates and resolves content.
Not autonomous. Not a replacement for the signed original weights.
