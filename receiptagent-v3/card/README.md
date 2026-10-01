---
license: apache-2.0
language: [en]
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags: [lora, qwen3.5, receipt-agent, research, proposal-only]
szl:
  artifact_class: ADAPTER_PLUS_DERIVED_MERGE
  publication_eligible: false
  autonomy_eligible: false
  evaluation_scope: TWO_DISTINCT_OWNER_RUN_DEV_RECORDS
---

# ReceiptAgent v3

**Structured receipt drafts for external review**

**Research artifact · proposal-only · not promotable · not autonomous**

A separate LoRA research adapter based on `Qwen/Qwen3.5-0.8B`, with a derived merged checkpoint also present. Its intended output is a proposed receipt draft. Validation, approval, execution and receipt issuance belong to the surrounding controller.

## Identity and files

Reviewed Hub revision: [`910343a2293f6bee8cc6e23f6cd16a981b92a155`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/tree/910343a2293f6bee8cc6e23f6cd16a981b92a155), inspected 2026-09-30 UTC.

- `adapter_model.safetensors` and `adapter_config.json`: root LoRA adapter, declared rank 16 and alpha 32
- `model.safetensors` and `config.json`: derived merged checkpoint, a separate artifact
- Tokenizer, chat-template and processor configuration files are present
- Training, development-evaluation, merge and license-provenance records are present

File presence was checked against the repository tree. This review did not download, hash or load weights. A file's presence is distinct from release approval and from evaluation of that exact file.

## Two development records, kept separate

| Record | Stated result | Identity and scope |
|---|---|---|
| [2026-08-29 development check](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/eval_dev_named_n.json) | 11/12: draft 4/4, recovery 3/4, refuse 4/4 | Historical owner-run DEV check; one recovery output did not parse |
| [2026-09-29 additive bound development check](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/eval_dev_named_n.bound.json) | 12/12: draft 4/4, recovery 4/4, refuse 4/4 | Owner-run DEV record naming adapter digest and Hub revision; base model is explicitly `UNRECORDED` |

The newer record explicitly preserves the 11/12 record and reports `promotion_effect=NONE`. Both use DEV n=12 and state that the test split was not opened. These are small development checks, not an independent benchmark, general safety result, held-out qualification or production release gate. This review did not rerun either evaluation or independently verify its weight bindings.

The newer record names adapter SHA-256 `90b82ade0b2c84eca92e90cdeb56137d61d7da4b4863ff4d29fa75e9dc037870` and Hub revision `e10f21923b32f9c0840a4fdb7c9027571ad1ccec`. It also records the development-file digest and harness digests. The unrecorded base identity prevents claiming a complete reproducible inference lineage. Do not transfer an adapter result to the merged checkpoint.

## Training and merge evidence

The [training receipt](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/training_receipt.laptop-blackwell.json) describes a 180-row laptop bf16 LoRA run and training loss 0.0537. Training loss is not evaluation. Its local-adapter digest differs from the later bound evaluation's digest; each belongs to its own record and must not be silently conflated.

The [merge receipt](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/merge_receipt.json) describes a 2026-09-14 repair that applied the adapter to 96/96 modules after an earlier merge was found to be a base copy. It does not contain a digest of the repaired merged `model.safetensors`; it is construction evidence, not a behavioral evaluation or a complete current-byte binding.

## Usage boundary

Use only for isolated, reviewed research with explicitly pinned model, base and runtime identities. The root contains both adapter and merged artifacts, so select the load path explicitly. Before publishing an executable load recipe, establish the matching immutable base revision and qualify the exact Transformers/PEFT versions and architecture class. The adapter configuration references a conditional-generation base class while the merged configuration names `Qwen3_5ForCausalLM`; do not hide this distinction behind a generic loader example.

Keep schema validation and human/controller authority outside the model. A generated approval flag, signature-shaped string or receipt identifier does not authorize an action or establish a real signature. Do not use the model to make consequential decisions about people.

## Source of truth

The declared canonical training source directory is [`szl-forge/frontier/qwen35-receiptagent-v3`](https://github.com/szl-holdings/szl-forge/tree/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/qwen35-receiptagent-v3), with card authoring at `receiptagent-v3/card/README.md`. These are declared relationships, not a claim that this review established source-to-weight reproducibility. Preserve separate identities for v2, v3, merged derivatives, and SZL-Forge-1.5B-ReceiptAgent.

## License and release status

The card declares Apache-2.0 and a standalone [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/LICENSE) is present. [provenance.json](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/provenance.json) and [status.json](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3/blob/910343a2293f6bee8cc6e23f6cd16a981b92a155/status.json) retain unknown ownership/relicensing authority and blocked lineage, consent, privacy, training-suitability and deployment checks. License-file publication does not clear those checks.

`publication_eligible=false`; `autonomy_eligible=false`. No flagship selection, production promotion, independent certification or deployment readiness is claimed. Lambda uniqueness remains an open advisory conjecture.
