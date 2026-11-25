# szl-forge — Domain Leader Study Card r1 (propose-only)

OUROBOROS pass 1 of 10 - 2026-11-25 - CTO web lane. Propose-only: never merge without owner GO.

## Domain leaders
- QLoRA: Efficient Finetuning of Quantized LLMs - Dettmers, Pagnoni, Holtzman, Zettlemoyer (arXiv:2305.14314; NeurIPS 2023): 4-bit NormalFloat, double quantization, paged optimizers; the recipe szl-forge descends from.

## Moat to absorb
- Report the bits: every forged artifact receipt carries quantization architecture (NF4, double-quant, compute dtype) as a model-card field.
- Train-stability telemetry per step bucket flagged MEASURED (compute-dtype instability checks after QLoRA).
- Paged-optimizer OOM survival: on SZL-owned metal, pinned configs include OOM-fail receipts rather than silent retries.
- Model Lab requirements policy: forbid pins unresolvable across the supported python matrix (structural fix vs PR #787-class breakage; pair with dependabot ignore-rule proposal).

## Recommended upgrades (owner-gated)
1. README fine-tune-lineage section citing arXiv:2305.14314 + recipe-parameter disclosure table (bits, compute dtype, pager).
2. Fine-tune receipts: quantization_arch, oom_events[], honest train-loss telemetry with step timestamps.
3. Model Lab pin-policy gate (matrix-resolvable pins only).