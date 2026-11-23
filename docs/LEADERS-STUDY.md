# Domain Leader Study — szl-forge (QLoRA kit → SZL-1)

OUROBOROS phase-3, cohort-02 tranche-1, Kimi ruling W26-C2T1-KIMI-20261123-ASK (GO). PROPOSE-ONLY study card: documentation only, no implementation in this PR.

## Current state (repo inspection 2026-11-23)
Python repo; description: QLoRA kit that fine-tunes an open base model into SZL-1 on SZL-owned hardware; dataset, Unsloth training recipe. Topics: fine-tuning, ollama, python, qlora, sovereign-ai, szl-holdings.

## Domain leaders studied
| Leader | What they have | Key sources |
| --- | --- | --- |
| QLoRA (Dettmers et al.) | The foundational recipe this repo builds on: 4-bit NF4 quantization + double quantization + paged optimizers, with published reproducibility detail per configuration | arxiv.org/abs/2305.14314; huggingface.co/papers/2305.14314 (accessed 2026-11-23) |
| AQLoRA (2026) | A zero-search method that selects protected layers from NF4 reconstruction error; it does not remove adapter-configuration search, and QLoRA does not inherently require that search | arxiv.org/abs/2608.23816 (accessed 2026-11-23) |
| Unsloth | The upstream training library the repo already depends on: maintained fine-tuning recipes, benchmarked speed/memory claims, active issue-driven development | github.com/unslothai/unsloth (upstream of repo's stated Unsloth training) (accessed 2026-11-23) |

## What the leaders do better
- Recipes publish their search/budget assumptions (AQLoRA's zero-search framing makes reproducibility a first-class property rather than an artifact of trial-and-error).
- Leader fine-tuning repos pin versions + publish benchmark configs; the leader version of receipt-bound training is one where another operator can re-derive the same adapter quality.

## Gaps
- Adapter and seed settings are already recorded in the training paths. Remaining provenance gaps include the base-model digest, Unsloth version, quantization details, and any configuration-search policy.

## Ranked #1 absorb proposal (NOT implemented in this PR)
Record full recipe provenance in every training receipt: base model digest, Unsloth version, quant config (NF4 class), seed, and config-search policy (with AQLoRA cited as the zero-search upgrade path). Expected benefit: SZL-1 adapters become re-derivable evidence, matching the estate receipts doctrine. Cost/risk: low; receipt schema addition, training code untouched by this card. Validation: two receipt-emitting runs from identical config produce byte-identical descriptor fields. Rollback: fields optional; receipts remain valid without them.

## Sources
- QLoRA paper: https://arxiv.org/abs/2305.14314 (accessed 2026-11-23)
- AQLoRA paper: https://arxiv.org/abs/2608.23816 (accessed 2026-11-23)
- Unsloth: https://github.com/unslothai/unsloth (accessed 2026-11-23)
