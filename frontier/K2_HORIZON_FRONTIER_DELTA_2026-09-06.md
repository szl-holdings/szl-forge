# K2 Horizon Frontier Delta — 2026-09-06

Status: **REPORTED_UPSTREAM / QUEUED_UNQUALIFIED**

This record supplements `HF_FRONTIER_INTAKE_2026-09-06.md`. It does not promote a model, dataset, runtime, or benchmark claim.

## Why this is a material delta

The K2 Horizon collection adds a coherent open-weight family spanning small dense models, a 7B-core model, a 36B MoVA model with 4B active parameters per token, and a 375B model with 23B active parameters per token. The collection also exposes training-data repositories and official GGUF/FP8 derivatives. That combination is relevant to SZL Forge across bounded agent evaluation, long-context workloads, post-training research, and sovereign inference.

Primary collection: https://huggingface.co/collections/IFM/k2-horizon

## Evidence correction: openness is variant-specific

Do **not** describe the whole family as having all training assets available.

- `IFM/K2-Horizon-7B` reports a 524,288-token context, public training data/recipe/code/evaluation resources, and released intermediate checkpoints.
- `IFM/K2-Horizon-MoVA-36B-A4B` and `IFM/K2-Horizon-375B-A23B` publish final checkpoints, but their cards describe intermediate checkpoints, data, and training code as future releases.
- Therefore each artifact keeps its own observed/announced state. Family-level marketing language is not inherited as SZL evidence.

Primary models:

- https://huggingface.co/IFM/K2-Horizon-7B
- https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B
- https://huggingface.co/IFM/K2-Horizon-375B-A23B

## Qualification order

1. **K2-Horizon-7B — FIRST_BENCHMARK.** Best practical starting point for governed navigation, receipt drafting, coding, tool-schema, and long-context evaluation.
2. **K2-Horizon-MoVA-36B-A4B — EFFICIENCY_CHALLENGER.** Evaluate only after runtime fit is verified; compare active-compute efficiency with quality and governance behavior.
3. **K2-Horizon-375B-A23B — FRONTIER_REFERENCE.** Track as a capability ceiling unless suitable multi-GPU infrastructure and an explicit cost envelope exist.

GGUF variants may enter the quantized sovereign-inference lane, but no quantized derivative inherits quality or reproducibility claims from its source checkpoint.

## Dataset posture

Observed collection datasets include:

- `IFM/TxT360-v2` — CC-BY-4.0.
- `IFM/Code-Reasoning` — Apache-2.0.
- `IFM/Math-Reasoning` — Apache-2.0.
- `IFM/SFT-Reasoning` — Apache-2.0.

All remain **RESEARCH_ONLY_PENDING_PROVENANCE_REVIEW**. Availability and a permissive repository license do not establish suitability for SZL training, data-subject consent, contamination safety, or downstream redistribution of every record.

## Required SZL evidence

Before any promotion:

1. Pin exact model, tokenizer, template, dataset, runtime, and quantization revisions.
2. Review every `trust_remote_code` path and bind the reviewed code digest.
3. Run fixed held-out cases for schema validity, grounding, citation binding, abstention, authority boundaries, tool-call correctness, coding, and long-context retrieval.
4. Measure time-to-first-token, throughput, peak memory, failure rate, and total evaluation cost on identified hardware.
5. Preserve Nemo PRE/POST inference witnessing and keep A11oy as the only consequential-action admission layer.
6. Emit signed or honestly unsigned receipts with limitations; upstream benchmark tables remain `REPORTED_UPSTREAM`.

## Watch suppression

Do not alert for wording edits, README formatting, popularity changes, routine metadata edits, or another quantization alone. Alert only when an announced training asset becomes publicly inspectable, a materially new model/runtime appears, licensing changes, serving requirements change substantially, or reproducible evidence changes the qualification decision.
