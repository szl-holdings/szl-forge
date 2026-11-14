---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: peft
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
tags:
- base_model:adapter:Qwen/Qwen3.5-0.8B
- lora
- sft
- transformers
- trl
- unsloth
- proposal-only
- research-only
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  sku: CHASKI-R2
  quant: bf16-lora
  qlora: false
  weights: AVAILABLE
  evals: HISTORICAL_OWNER_RECORDS_UNQUALIFIED
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
---

<!-- SZL-CARD-PRESENTATION:v1 -->
<p><a href="https://huggingface.co/SZLHOLDINGS"><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="48" height="48" /></a></p>

# Chaski-R2

A bf16 LoRA research recut of Qwen3.5-0.8B, with separate adapter and merged artifact identities.

**Artifact:** LoRA adapter; separate merged checkpoint · **Stage:** HOLD · not promotable

[Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Canonical source](https://github.com/szl-holdings/szl-forge/tree/main/chaski-r2) · [Reviewed evidence](https://github.com/szl-holdings/szl-forge/blob/2b4d7a0f69e68d60cb0f35f09c9bac837d66acdc/chaski-r2/card/README.md)

Provider compatibility support [requested on 4 October 2026](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/12785). No provider was listed in the dated Hub observation; adoption and use qualification remain unchanged.

## Before you use it

- The historical 1/5 held-out gate and overall FAIL remain. Later reused-fixture adapter results do not clear that gate.
- Publication and autonomy eligibility remain false; lab loading remains forbidden.
- Record the loader class and 192/192 applied adapter tensors. The merged runtime needs its own byte-bound evaluation.

<!-- SZL-EVIDENCE-PASSPORT:v1:START -->
## Evidence passport

**Dated input observation: 2026-11-13T22:17:00.237505+00:00. Existing research limits remain in force.**

[Inspect the immutable repository snapshot](https://huggingface.co/SZLHOLDINGS/chaski-r2/tree/523b733e20095fda3b1426649348e9fb179098ae).

| Observation | Scope |
|---|---|
| Repository revision | `523b733e20095fda3b1426649348e9fb179098ae` |
| Historical README retrieval | HTTP 200; not a readback of this proposed panel |
| Recognized artifact filenames | 3 filename hints; not tensor verification |
| Structured records retrieved | 5; bounded selection, not an exhaustive evidence census |

Historical input README SHA-256: `935571a2129071d5343c4c280799890e5ff506863f82178deb5bc36a6c1ee8c4`.

[Reviewed authoring baseline](https://github.com/szl-holdings/szl-forge/blob/2f1f7636acf4e77d87de34319c34507f9441dc71/chaski-r2/card/README.md). Existing publisher: `publish-chaski-card.yml`. Source changes still require normal review and exact deployed-byte readback.

### Inspect the records

| Record | Retrieved-byte SHA-256 |
|---|---|
| [training\_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/training_receipt.json) | `77b8b850dcb8c76e139b17c42e198c0ad91ec6b17c43649aa49444729f7a9b49` |
| [training\_receipt.local.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/training_receipt.local.json) | `9a69e1c501db01ba5d50c01fd23f7bc098d7af1ea237a3c84a50cd8fb8242a0c` |
| [adapter-unsloth/training\_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/adapter-unsloth/training_receipt.json) | `29981c3b1f8771e2bcc6b1f325222f7782b279500f1dcd3f22ce47cc8debce84` |
| [eval\_merged\_measured.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/eval_merged_measured.json) | `4351df066e5078ebf9a496a148173df7f3e6b1fa7d9bf7db7ace1afeea5eb866` |
| [eval\_named\_n\_generate.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/eval_named_n_generate.json) | `c688bbcdfa40508c148233e93e4676028d559d3492d3649d5435a154401f272e` |

### Interpretation

Documentation and retrieved-record integrity only. No new training, model evaluation, weight-byte verification, independent signer identity, runtime qualification or promotion. Missing evidence is not a zero score.
Training loss is not held-out performance. A copied receipt does not qualify a quantization or merged checkpoint. Existing negative results, license and technical disclosures remain authoritative.

<!-- SZL-EVIDENCE-PASSPORT:v1:END -->

<details>
<summary>Technical details and evidence</summary>

<!-- SZL-CARD-TECHNICAL:v1:START -->
<!-- szl:artifact-identity-reconciled -->
> **Artifact identity (noted 2026-09-30).** Besides the LoRA adapter, this repository's root carries a merged full-precision checkpoint (`model.safetensors` + `config.json`, loadable with `transformers`) produced by the receipted CPU merge of this adapter into its declared base (`merge_receipt.json`). The adapter remains the artifact of record for every figure on this card; the merged bytes carry no separate held-out receipt and add no claim. Metadata-only note.

<!-- szl:loader-class-boundary -->
> **Loader-class boundary (noted 2026-09-30).** The adapter tensors are keyed for the multimodal module layout (`base_model.model.model.language_model.layers.*`, the class `Qwen3_5ForConditionalGeneration` / `AutoModelForImageTextToText`). Under transformers 5.18, `AutoModelForCausalLM` instantiates `Qwen3_5ForCausalLM` (`model.layers.*`); PEFT then applies 0 of 192 adapter tensors, emits only a warning, and the result reproduces the bare base model byte-for-byte on the held-out prompts. Any run that reports a score for this adapter must show adapter key coverage (192/192) and the loader class it used; a record without those fields does not establish that the adapter was applied. Evidence: szl-holdings/szl-forge `tools/geh_v8/evidence_sandbox/chaski_probe/` (CPU replay receipts, #444). Metadata-only note; it authorizes no load, changes no artifact, and adds no claim.

# Chaski-R2

**Research adapter · dated owner-run evidence · HOLD**

This research candidate is **not promotable**. Publication and autonomy
eligibility remain false; the dated observations authorize no deployment.

Chaski-R2 is a bf16 LoRA research recut of `Qwen/Qwen3.5-0.8B`.
It is separate from `SZLHOLDINGS/chaski` and the r=16, α=16
`SZLHOLDINGS/chaski-5050` experiment. Publication and autonomy eligibility
remain **false**. Lab load remains forbidden.

Card and evidence review snapshot: [a5fbffcccabcc77ae87bf3740358a02f91a7c27b](https://huggingface.co/SZLHOLDINGS/chaski-r2/tree/a5fbffcccabcc77ae87bf3740358a02f91a7c27b).
The review read small records and metadata; it did not download weights,
rerun inference, verify signatures, or qualify the current repository head.

## Dated evaluation records

These are separate owner-run records. Their scores and identities must not
be combined or transferred between adapter and merged artifacts.

| Record | Reported result | Scope and limit |
|---|---|---|
| [2026-08-29 named-N generation](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/a5fbffcccabcc77ae87bf3740358a02f91a7c27b/eval_named_n_generate.json) | 5/5 JSON drafts; 6/6 adversarial refusals | Historical owner record; current-byte applicability was not independently established here. |
| [2026-09-14 merged-checkpoint record](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/a5fbffcccabcc77ae87bf3740358a02f91a7c27b/eval_merged_measured.json) | 5/5 JSON drafts; 6/6 adversarial refusals | Owner-reported merged checkpoint, plain Transformers fp16. Separate from the later adapter-only binding. |
| [2026-09-15 canonical gate](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/evaluation/gate_runs/chaski-r2_4ef29684c56d_20260915T093042Z_receipt.json) | **1/5 held-out; overall FAIL** | Separate suite at Hub revision `4ef29684c56de07626b1cefb4b93a1b486de2aa1`; `heldout.refusal_no_regression=false`. A separate refusal command passed. Production and publication authorization are false. |
| [2026-09-24 adapter bake-off](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/a5fbffcccabcc77ae87bf3740358a02f91a7c27b/evidence/2026-09-25-adapter-binding/receipt.json), with [2026-09-25 additive binding](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/a5fbffcccabcc77ae87bf3740358a02f91a7c27b/evidence/2026-09-25-adapter-binding/source_publication_binding.json) | 5/5 draft contracts; 6/6 refusal prefixes | Adapter at historical Hub revision `661f8ee9ff6ab8b11dda6e7a9d42c14d3124c6dd`; reused named fixtures. The unsigned owner binding performs no new inference and preserves the earlier gate failure. |

The later adapter result does not erase the earlier gate **FAIL**, qualify
the merged checkpoint, or establish semantic truth or general safety.
The original bake-off contains stale `keep gate_ran=false` boilerplate;
the additive binding explicitly explains its conflict with the measured
generated cases and the top-level `gate_ran=true`. Historical records are
preserved rather than rewritten.

## Artifact identities

| Identity | Value | Interpretation |
|---|---|---|
| Historical training-era adapter digest | `440340ce29e19344c0625d0adfe820b277cdb0e24099d4e612f88ad6b3cf49c6` | Value retained from the earlier card; it is not the later adapter-file identity. |
| Adapter raw-file SHA-256 in the additive binding | `6f12981ea5df5e22d3493eefb20d75db75c1c88961ba6621a35af09edd0cdde6` | `adapter_model.safetensors` at historical Hub revision `661f8ee9ff6ab8b11dda6e7a9d42c14d3124c6dd`; reported by the binding, not rehashed in this review. |
| Adapter directory digest in the additive binding | `078ec09fb3e8e215d1b6a98bb7164a94c7bb332d847c8e1d66f565e767171107` | SHA-256 of sorted safetensor filenames encoded UTF-8, a NUL delimiter, then file bytes; distinct from a raw-file digest. |
| Merged checkpoint | Separately distributed runtime form | Its historical report is separate; the adapter-only binding does not establish merged-model evaluation identity. |
| `adapter-unsloth/` | Additional adapter residue | Presence does not transfer any other artifact's evaluation or qualification. |

The 2026-09-25 binding also records the older sidecar's Windows CRLF hashes
and the published Git LF bytes separately. Those differing receipt-byte
domains are explained in the additive record; neither original is replaced.
Base weights and equality of current-head payloads to historically evaluated
bytes were not independently rehashed in this review.

## Historical training

The earlier owner-local RTX 5050 Laptop recipe used bf16 LoRA, r=16,
α=32, seed 11, response-only cross entropy, 32 training rows in
`chaski_r2/train.jsonl`, three epochs, batch size 1 and accumulation 2.
The reported train loss **0.7656** is a training metric, not an evaluation.
The training step itself reported no evaluation; the dated records above
were added later. PEFT 0.19.1 is the version recorded by the earlier card.

## Use and release boundary

Use only for controlled research on proposal outputs. Validate outputs
outside the model. No deployment, production route, autonomy, or promotion
is authorized by these records. The later binding states **HOLD** and
`promotion_effect=NONE`. No throughput or currently working loader is
claimed; selecting a runtime form requires a compatible loader and verified
base and artifact revisions.

Apache-2.0 is declared, and a standalone `LICENSE` is listed in the reviewed
Hub tree. This observation does not independently establish ownership or
downstream artifact license coverage.

Canonical authoring source: [chaski-r2/card/README.md](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/chaski-r2/card/README.md).
An observed source revision identifies the reviewed file; it is not release
approval. Doctrine v11 LOCKED. Λ = Conjecture 1, advisory, never a theorem.

<!-- SZL-CARD-TECHNICAL:v1:END -->
</details>

<!-- SZL-NATIVE-ARTWORK:v1 -->
SZL Orbit artwork: [source](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Software and artifact licensing remain declared above.
<!-- /SZL-NATIVE-ARTWORK:v1 -->

<!-- SZL-CARD-RESTORATION:2026-11-14:v1 -->
## Publication restoration note (2026-11-14)

On 2026-11-14, an off-route automated job stamped the public SZLHOLDINGS Hub
model cards with a marketing template that replaced curated card bytes,
including per-model claims this artifact has not earned. This source card in
szl-holdings/szl-forge is the canonical reviewed card; that job did not alter
it. This note marks the card so the closed-profile publisher republishes the
reviewed bytes, banner, and source binding to the Hub with exact readback.
The incident and the per-model evidence manifest are documented in szl-forge
under `docs/model-card-overhaul-20261114/`. No qualification, evaluation,
weight, adapter, config, visibility, hardware, or runtime state changed by
this note.
