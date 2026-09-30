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

# Chaski-R2

**Research adapter · dated owner-run evidence · HOLD**

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
