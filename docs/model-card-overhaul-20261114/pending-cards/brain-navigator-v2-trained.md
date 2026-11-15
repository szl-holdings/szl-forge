---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- adapter
- peft
- eval-battery-bound
- szl-estate
license: apache-2.0
---

# brain-navigator-v2-trained

LoRA adapter in the SZLHOLDINGS Hugging Face estate. This repository was created by the
2026-11-13/14 wave-4 automated train/publish run.

## What this repository contains

- .gitattributes
- adapter_config.json
- adapter_model.safetensors
- chat_template.jinja
- tokenizer.json
- tokenizer_config.json
- receipts/eval_battery.json
- LoRA adapter over base model Qwen/Qwen3-1.7B (rank 16, alpha 32, as recorded in adapter_config.json).

## Measured status: EVALUATION BATTERY BOUND (2026-11-14)

- An estate evaluation battery receipt (kind `szl.estate-battery/mass-training`) naming this
  repository was measured on NVIDIA RTX A6000 (node `0056-kci3-sma6-prxmx200114`) at `2026-11-13T22:37:20.482213+00:00` and published in the estate
  evidence dataset `SZLHOLDINGS/frontier-54-runs` at `benchmark/mass-battery-brain_navigator_v2_trained.json` (receipt verified present on 2026-11-14).
  It is mirrored in this repository at `receipts/eval_battery.json` (same byte size as the dataset
  receipt, 3641 bytes; dataset receipt sha256 `3ae2dadc72b192cac0e4ec4a380edb2a7a634ca405b0dc6c0904139a865bda80`).
- Battery result (state `MEASURED`): 3/6 — 3/3 answerable probes passed; 0/3 unanswerable
  probes were correctly abstained or refused.
- Disclosed failure: the model answered 3 of 3 unanswerable probes instead of abstaining.
  This is a measured abstention failure, not a capability claim.
- Binding scope: the receipt names this repository and its training-lane run window; it does not
  embed this adapter's byte-level digest, so the binding is by repository identity, not a
  weight hash.
- No training receipt (step count, loss curve, hardware, duration) was published with this
  artifact. Training provenance remains not independently verifiable from this repository.
- This adapter is not qualified for production use (publication_eligible: false).

## Card history

- At the 2026-11-14 estate model-card overhaul this repository was listed in the overhaul
  manifest (szl-holdings/szl-forge docs/model-card-overhaul-20261114/manifest-20261114.json)
  as reads: 0, pub_map: UNMAPPED, and its card remained the default auto-generated template.
- This honest minimal card was written on 2026-11-14 following the overhaul rules, from this
  repository's own battery receipt and adapter_config.json. No claim in this card goes beyond
  those artifacts.
- Estate record: szl-holdings/szl-forge docs/model-card-overhaul-20261114
  (schema, manifest, and rules for this overhaul)

## License

This repository's card metadata already declares `license: apache-2.0` (present in the
previous auto-generated card front matter at the 2026-11-14 audit; not invented by this card).

## License (derived 2026-11-14)

- The adapter is derived from `Qwen/Qwen3-1.7B` (recorded in
  `adapter_config.json: base_model_name_or_path`), whose model card declares
  `license: apache-2.0` (estate-verified 2026-11-14 at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`).
- This repository contains adapter weights produced by the estate's own training runs;
  no upstream base-model weights are redistributed here.
