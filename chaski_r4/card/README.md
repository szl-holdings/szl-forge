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
- peft
- experimental
- research-only
szl:
  doctrine: v11-LOCKED
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  artifact_state: PUBLIC_EXPERIMENTAL_ARTIFACT
  originality: FINETUNE_DISCLOSED_BASE
  sku: CHASKI-R4
  quant: bf16-lora
  qlora: false
  weights: AVAILABLE
  evals: MEASURED_NAMED_N_RECEIPT_C
  promotion: NOT_PROMOTABLE
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
---
<!-- szl:loader-class-boundary -->
> **Loader-class boundary.** The adapter tensors are keyed for the multimodal module layout (`base_model.model.model.language_model.layers.*`, class `Qwen3_5ForConditionalGeneration` / `AutoModelForImageTextToText`). Under transformers 5.4–5.18, `AutoModelForCausalLM` instantiates `Qwen3_5ForCausalLM` (`model.layers.*`); PEFT 0.18–0.21 then applies 0 of 192 adapter tensors, emits only a warning, and the result reproduces the bare base model byte-for-byte on the held-out prompts. Any run that reports a score for this adapter must show adapter key coverage (192/192) and the loader class it used; a record without those fields does not establish that the adapter was applied. Evidence: szl-holdings/szl-forge `tools/geh_v8/THREAD_AUDIT.md` and `tools/geh_v8/evidence_sandbox/chaski_probe/`. Metadata note; it authorizes no load and adds no claim.

# Chaski-R4

**Public experimental artifact · measured named-N record (receipt C) · NOT PROMOTABLE · NOT AUTONOMOUS**

Chaski-R4 is a bf16 LoRA research recut of `Qwen/Qwen3.5-0.8B`, trained on owner metal on
2026-09-17, evaluated on 2026-10-01 beside its `SZLHOLDINGS/chaski-r2` control in one run of
the canonical named-N bake-off, and published here the same day with digest-verified bytes. It is separate from `SZLHOLDINGS/chaski`, `SZLHOLDINGS/chaski-5050`
and `SZLHOLDINGS/chaski-r2`, and it does not inherit any of their records. Publication and autonomy
eligibility remain **false**; `promotion: NOT_PROMOTABLE` is the artifact state, not a verdict on
the counts below. Publishing this artifact publishes evidence; it does not promote anything.

## What was measured

Receipt C: szl-holdings/szl-forge [`chaski_r4/evidence/canonical_rerun_20261001_140615.receipt.json`](https://github.com/szl-holdings/szl-forge/blob/06804ce4/chaski_r4/evidence/canonical_rerun_20261001_140615.receipt.json),
SHA-256 `5ae3de970014726f190dfe8e4d5b35e667fd737b1be29e27205c25d143bdf403`, `label=MEASURED`,
`gate_ran=true`, `publication_eligible=false` (fixed by the evaluator), `held_out_in_gradients=false`.
Evaluator `chaski/bakeoff_named_n.py` at szl-forge `55c3027b` with the adapter-coverage guard.
Owner metal: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), torch 2.11.0+cu128, transformers 5.16.1,
peft 0.20.0, Python 3.11.9, Windows 11. Base `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`. Prompt renderer `AutoProcessor`; every case carries
`prompt_sha256`.

| Candidate (same run) | JSON drafts | Adversarial refusals | Adapter directory digest | Adapter tensors applied |
|---|---|---|---|---|
| base-qwen35-0.8b | 0/5 | 6/6 | — | base, no adapter |
| chaski-5050 | 5/5 | 6/6 | `fc7da61d9e30d9bc…` | 192/192 |
| chaski-r2 (control) | 5/5 | 6/6 | `e35df3bea9b9d260…` | 192/192 |
| **chaski-r4 (this artifact)** | **5/5 JSON drafts; 6/6 adversarial refusals** | | `e1abc37a5c41a82b…` | 192/192 |

The gate is small and synthetic: n=5 JSON drafts, n=6 adversarial refusals, integer counts only.
It is not a broad quality or safety benchmark. Not SOTA. The counts show that this adapter, when
actually applied, produces schema-valid drafts and refusal prefixes on these eleven held-out prompts;
they establish nothing about semantic truth, general safety, or behaviour outside the gate.

## Earlier records and what they do not settle

| Record | Evaluator | Reported | Scope and limit |
|---|---|---|---|
| Receipt A, 2026-09-16 ([`chaski_r4/evidence/r4_receipt_20260916_091416.json`](https://github.com/szl-holdings/szl-forge/blob/06804ce4/chaski_r4/evidence/r4_receipt_20260916_091416.json)) | `chaski_r4/bakeoff_canonical_four_way_r4.py` (`f4ca282a…`, `AutoModelForCausalLM`) | 0/5 JSON drafts; 3/6 refusals | **Original** r4 adapter (`b116832a…`), not this artifact. torch 2.11.0+cu128. |
| Receipt B, 2026-09-17 ([`chaski_r4/bakeoff_four_way_receipt.json`](https://github.com/szl-holdings/szl-forge/blob/06804ce4/chaski_r4/bakeoff_four_way_receipt.json)) | same copy | 5/5 JSON drafts; 6/6 refusals | **Retrained** r4 adapter (`e1abc37a…`, this artifact). torch 2.10.0+cu130. |
| Receipt C, 2026-10-01 | `chaski/bakeoff_named_n.py` with coverage guard | 5/5; 6/6 | This artifact beside the r2 control; adapters 192/192 applied. |

The `f4ca282a…` evaluator copy cannot apply these adapters under any transformers/peft pairing
probed (it loads the text-only class and applies 0/192 tensors), so the adapter effects recorded in
receipts A and B cannot be reproduced with the current adapter files through that path. The
provenance of receipts A and B remains unresolved; it is recorded as such in
[`tools/geh_v8/THREAD_AUDIT.md`](https://github.com/szl-holdings/szl-forge/blob/06804ce4/tools/geh_v8/THREAD_AUDIT.md)
rather than explained away. Receipt C neither explains nor depends on them. Receipt C reproduces
receipt B's integer counts; output bytes differ case by case across torch builds (r4 8/11 identical),
the counts do not.

## Artifact identities

| Identity | Value | Interpretation |
|---|---|---|
| `adapter_model.safetensors` SHA-256 | `f1a2cdc313795775966280bc8648367005700327dd28010da8d2f88d2a5e2a02` | Raw file digest; matches `adapter_weights_sha256` in the training receipt. 25,587,104 bytes. |
| `adapter_config.json` SHA-256 | `d36472a3100231dfdf0c4aa16d3a80cb8ab53c86af33a15172b06f9d9a313193` | Matches `adapter_config_sha256` in the training receipt. |
| Adapter directory digest | `e1abc37a5c41a82b0fc2cd98ccd6edbb2a08fceb8ca9e883bcfb76b861c221cf` | SHA-256 of sorted safetensor filenames encoded UTF-8, a NUL delimiter, then file bytes; the `adapter_sha256` field of receipts B and C. |
| Training dataset | `chaski_r4/train.jsonl`, 40 rows, SHA-256 `0fea0d85f2ca4cd55d7ce51b8399a409d7046d2cd77ffc131945ee20c706535a` | Held-out gate files were not in gradients. The committed `training_plan` hash differs from the receipt's dataset hash; the receipt is the binding record. |
| Hub bytes revision | `f662e24aa9e878dc6c2df50151d4fd500ea8a16c` | The commit (2026-10-01T21:10Z) that added `adapter_config.json`, `adapter_model.safetensors`, `training_receipt.json`, `evidence/` and `LICENSE` after the card (`13c9b10fee50542426b105520b7f32969d595ed8`). Every file was read back byte-identical and the directory digest above was recomputed from the Hub files. |
| Publication receipt | szl-forge [`chaski_r4/evidence/publication_receipt_20261001_171009.json`](https://github.com/szl-holdings/szl-forge/blob/main/chaski_r4/evidence/publication_receipt_20261001_171009.json), SHA-256 `d8eea67be2062b0741c4d52dedb511ac1e87e99c43d54b6e1f34809b1de49af2` | Written by `tools/publish_chaski_r4_bytes.py` only after the read-back passed; records card and bytes revisions and per-file digests. Publication of bytes and evidence; not promotion. |

Loading verification is part of the artifact: after download, recompute the directory digest and
require `192/192` applied tensors under `AutoModelForImageTextToText` before reporting any number.

## Training

`chaski_r4/training_receipt.json` (2026-09-17T02:36Z): bf16 LoRA, r=16, α=32, seed 11, response-only
loss, 40 rows, three epochs, batch size 1, accumulation 4, learning rate 2e-4 constant with 6 warmup
steps, adamw_8bit, max sequence 2048, RTX 5050 Laptop GPU. The reported final train loss **0.9820** is
a training metric, not an evaluation; the training step itself reported `evals: none-this-run`.

## Use and release boundary

Use only for controlled research on proposal outputs. Validate outputs outside the model. No
deployment, production route, autonomy, or promotion is authorized by this card or by receipt C.
`publication_eligible` is false by construction of the evaluator and stays false here; promotion is an
owner decision taken outside the gate, and it has not been taken. `SZLHOLDINGS/chaski` is never
overwritten by this artifact. No throughput claim is made. No uniqueness claim is made.

Apache-2.0 is declared for the adapter; the base model carries its own license.

Qualification state of record: [`chaski_r4/QUALIFICATION_STATE.md`](https://github.com/szl-holdings/szl-forge/blob/57e6e6c3/chaski_r4/QUALIFICATION_STATE.md).
Canonical card authoring source: [chaski_r4/card/README.md](https://github.com/szl-holdings/szl-forge/tree/main/chaski_r4/card).
Doctrine v11 LOCKED. Λ = Conjecture 1, advisory, never a theorem.
