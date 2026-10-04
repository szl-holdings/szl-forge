---
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - governed-ai
  - proposal-only
  - research-only
  - szl-holdings
  - chaski
  - cutting
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  jobs: local-5050
  job_id: local-5050
  weights: AVAILABLE
  evals: HISTORICAL_TRAINING_NONE_SEPARATE_GATE_FAIL
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
  copied_live_chaski_weights: false
  train_loss: 2.228136855544466
  train_loss_label: MEASURED
  adapter_sha256: 620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5
---

<!-- SZL-CARD-PRESENTATION:v1 -->
<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112" /></a></p>

# Chaski-5050

A separate bf16 LoRA research experiment with its training and failed gate records retained.

**Artifact:** LoRA adapter; separate merged checkpoint · **Stage:** QUARANTINE · research residue

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-forge/tree/main/chaski-5050) · [Evidence](https://github.com/szl-holdings/szl-forge/blob/2b4d7a0f69e68d60cb0f35f09c9bac837d66acdc/chaski-5050/card/README.md)

## Before you use it

- The later held-out gate recorded 1/5 and overall FAIL. Publication and autonomy eligibility remain false.
- Lab loading remains forbidden; root adapter metadata contains a nonportable owner-local base path and runtime loading is unverified.
- Do not substitute Chaski or R2 evidence. Adapter results require explicit loader-class and tensor-coverage evidence.

<details>
<summary>Technical details and evidence</summary>

<!-- SZL-CARD-TECHNICAL:v1:START -->
<!-- szl:artifact-identity-reconciled -->
> **Artifact identity (noted 2026-09-30).** Besides the LoRA adapter, this repository's root carries a merged full-precision checkpoint (`model.safetensors` + `config.json`, loadable with `transformers`) produced by the receipted CPU merge of this adapter into its declared base (`merge_receipt.json`). The adapter remains the artifact of record for every figure on this card; the merged bytes carry no separate held-out receipt and add no claim. Metadata-only note.

<!-- szl:loader-class-boundary -->
> **Loader-class boundary (noted 2026-09-30).** The adapter tensors are keyed for the multimodal module layout (`base_model.model.model.language_model.layers.*`, the class `Qwen3_5ForConditionalGeneration` / `AutoModelForImageTextToText`). Under transformers 5.18, `AutoModelForCausalLM` instantiates `Qwen3_5ForCausalLM` (`model.layers.*`); PEFT then applies 0 of the 192 adapter tensors and emits only a warning, so the result is the bare base model (observed and receipted on the sibling `SZLHOLDINGS/chaski-r2` adapter, which shares this exact key layout: byte-identical base outputs on the held-out prompts). Any run that reports a score for this adapter must show adapter key coverage (192/192) and the loader class it used; a record without those fields does not establish that the adapter was applied. Evidence: szl-holdings/szl-forge `tools/geh_v8/evidence_sandbox/chaski_probe/` (CPU replay receipts, #444). Metadata-only note; it authorizes no load, changes no artifact, and adds no claim.

# Chaski-5050

> **QUARANTINE.** Research residue. Root adapter metadata contains a
> nonportable owner-local base path. Runtime loading has not been verified.

This bf16 LoRA experiment is separate from `SZLHOLDINGS/chaski` and
Chaski-R2. Publication and autonomy eligibility remain **false**.
Lab load is forbidden. The card does not authorize artifact changes or
model promotion.

Card and metadata review snapshot: [8c9d4782b76e55377e00fba249c6938000dca423](https://huggingface.co/SZLHOLDINGS/chaski-5050/tree/8c9d4782b76e55377e00fba249c6938000dca423).
Small source and receipt files were read; no model weights were downloaded,
no inference or tensor comparison was performed, and no current artifact
was qualified.

## Distributed forms

| Form | Observed metadata | Qualification limit |
|---|---|---|
| Root LoRA adapter | `adapter_model.safetensors` and `adapter_config.json`; historical card reports 25,587,104 bytes and raw-file SHA-256 `620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5` | The root config embeds an owner-local absolute base path and names `Qwen3_5ForConditionalGeneration` in `auto_mapping`. Loading is unverified. |
| `adapter-unsloth/` | Additional adapter and processor metadata are listed | Lineage and evaluation applicability were not independently established here; this is not interchangeable with the root adapter by assumption. |
| Root merged checkpoint | `model.safetensors`, `config.json`, and `merge_receipt.json` are listed | A separate runtime form; file presence or a merge receipt does not establish behavioral qualification. |

The [2026-09-14 merge repair record](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/8c9d4782b76e55377e00fba249c6938000dca423/merge_receipt.json)
reports replacing an earlier base-copy merge, touching 96/96 modules with
maximum weight delta `0.0012716073542833328`. This owner-reported repair
does not establish the quality, loading, or exact current identity of the
merged checkpoint.

## Dated training and evaluation

The [2026-08-28 training receipt](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/8c9d4782b76e55377e00fba249c6938000dca423/training_receipt.json)
records `evals=none-this-run`: no JSON/refusal gate ran in that training step.
Its train loss **2.228136855544466** and runtime **883.2224 seconds** are
training measurements, not held-out evaluation scores.

A [separate 2026-09-15 canonical gate receipt](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/evaluation/gate_runs/chaski-5050_63ea80915ccb_20260915T041028Z_receipt.json)
at historical Hub revision `63ea80915ccb663b4aa589cbf2ce52a1dec574fe`
reports **1/5 held-out and overall FAIL**, with
`heldout.refusal_no_regression=false`. Its separate refusal command passed;
that does not change the overall failure. Production and publication
authorization are **false**. This receipt is a distinct owner-run suite;
its applicability to the current distributed forms was not independently
established in this review. No 5/5 or 6/6 qualification is claimed.

## Historical recipe

The training receipt describes an owner-local RTX 5050 Laptop run, rather
than an HF Job: bf16 LoRA r=16, α=16, seed 11, three epochs, batch size 1,
accumulation 4, maximum sequence length 2048, learning rate 2e-4, and
`adamw_8bit`. The dataset is `SZLHOLDINGS/szl-1-doctrine-sft`, 41 rows,
JSONL SHA-256 `ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243`.
The recorded stack is Unsloth 2026.7.2, Transformers 5.5.0, and Torch
2.10.0+cu128. `copied_live_chaski_weights=false` is the receipt's declaration.
These records are preserved; this card review reran none of them.

## Research loading boundary

The former generic `AutoModelForCausalLM` example was not validated against
the root adapter's multimodal wrapper metadata or the coexisting merged
weights. Treat this repository as research residue rather than a runnable
quickstart. Before loading, a source owner must select a particular form,
verify its compatible wrapper and immutable base revision, and resolve the
nonportable path under a separately reviewed artifact change. A card edit
does not sanitize the configuration or qualify a replacement loader.

[A11OY-MINI](https://huggingface.co/SZLHOLDINGS/A11OY-MINI) has its own
artifact-specific legacy and R2 evidence. It inherits no training score,
gate result, or qualification from this card.

Apache-2.0 is declared in metadata. No standalone `LICENSE` is listed at
the reviewed revision; artifact license coverage was not independently
verified. A license addition requires source-owner review of that coverage.

Canonical card authoring source: [chaski-5050/card/README.md](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/chaski-5050/card/README.md).
The older `chaski/README_5050.md` reference is a separate recipe document,
not the card file selected by the canonical card publisher.
Doctrine v11 LOCKED 749/14/163. Λ = Conjecture 1, advisory, never a theorem.

<!-- SZL-CARD-TECHNICAL:v1:END -->
</details>
