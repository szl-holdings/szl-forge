---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- unsloth
- governed-agent
- retrieval
- brain-navigator
- szl-holdings
---

> **SEPARATE RESEARCH ADAPTER · MEASURED NAMED-N RECORD · LATER GATE FAIL · NOT PROMOTABLE · NOT AUTONOMOUS**

# BrainNavigator-R2

## What this is

A research LoRA adapter, with a separate derived merged checkpoint also distributed, based on `Qwen/Qwen3.5-0.8B`. The [training_receipt.json](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2/blob/ad510c4b05429acb862b216718f098ace6d05af3/training_receipt.json) records bf16 LoRA with r=16 and alpha=32, a synthetic NAVIGATE/ABSTAIN curriculum over **575 public handles**, and **0 raw private-graph nodes admitted to gradients**. It is not a model of the private 9,464-node graph and confers no private-graph access or content resolution in weights.

## Evidence and artifact identity

Card evidence reviewed on 2026-09-30 UTC at Hub revision [`7fe3872fcb78ec8f5cc2f0c7464538b50b34e4b2`](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2/tree/7fe3872fcb78ec8f5cc2f0c7464538b50b34e4b2). The historical artifact/evaluation snapshot below is `ad510c4b05429acb862b216718f098ace6d05af3`. The [eval_report.json](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2/blob/ad510c4b05429acb862b216718f098ace6d05af3/eval_report.json) records `maturity=MEASURED_RESEARCH_ONLY` and `publication_eligible=false`.

| Owner-run named-N generate measure | Receipt-reported result |
|---|---|
| Retrieval hit | 5/5 |
| Abstention | 6/6 |
| Parse failures | 0 |
| Hallucinated citations | 0 |

These values are from the `generate` section of [eval_report.json](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2/blob/ad510c4b05429acb862b216718f098ace6d05af3/eval_report.json), computed on 2026-08-29 on a local LoRA. Its separate `software` section concerns lexical retrieval over the public projection. Neither section is a public leaderboard or an independent third-party benchmark. The generate receipt does not identify an exact public adapter-file SHA-256 or a fully specified runtime build.

The separate [post-publication gate receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/evaluation/gate_runs/brain-navigator-r2_ad510c4b0542_20260915T040842Z_receipt.json) names Hub revision `ad510c4b05429acb862b216718f098ace6d05af3` and reports **2/5** on the canonical held-out suite in `transformers-cpu` mode, with overall **FAIL**, `heldout_passed=false`, `productionAuthorization=false`, and `publicationAuthorization=false`. This is a different suite from the earlier named-N record; its results cannot be exchanged or combined with those earlier scores. It also records `refusalPassed=false` and a refusal regression: rate 0.2 against baseline 0.4. The linked receipt retains the execution warnings; earlier execution failures remain in the [canonical gate archive](https://github.com/szl-holdings/szl-forge/tree/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/evaluation/gate_runs).

The [merge_receipt.json](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2/blob/ad510c4b05429acb862b216718f098ace6d05af3/merge_receipt.json) separately records a 2026-09-14 repair after the prior merged checkpoint was an empty base copy: 96/96 adapter modules applied, maximum weight delta `0.0025102924555540085`. That merge record establishes the reported repair operation, not a new behavioral evaluation. The earlier named-N result does not automatically transfer to the repaired merged checkpoint.

## What the evidence does and does not establish

Published files establish that scoped measured records and a later failed gate are available for inspection. Train loss is not evaluation; file integrity is not quality. The training receipt retains `evals=none-this-run` for its training step. That historical field does not erase the later generate record. Neither the named-N record nor the merge receipt qualifies this artifact for release. The [provenance.json](https://huggingface.co/SZLHOLDINGS/brain-navigator-r2/blob/ad510c4b05429acb862b216718f098ace6d05af3/provenance.json) also retains blocked artifact-lineage, consent, privacy-review, deployment, and served-revision boundaries.

## Release, autonomy, and deployment boundary

`publication_eligible=false` and `autonomy_eligible=false` remain in force. Proposal only: no qualification, promotion, deployment authorization, autonomy, or house-lab pin follows from these records or from public file availability. Lambda uniqueness remains **Conjecture 1: open and advisory, never a theorem**.

## Relationship to other artifacts

This SKU does not overwrite [SZL-Khipu-1.5B-BrainNavigator](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator), [SZL-Khipu-1.5B](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B), or [SZL-Khipu-1.5B-GGUF](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF). The [second-brain Space](https://huggingface.co/spaces/SZLHOLDINGS/second-brain) is a separate software retrieval surface, not proof of model or private-graph access.

## Source of truth

The immutable Hub evidence linked above and the [canonical post-publication receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/frontier/evaluation/gate_runs/brain-navigator-r2_ad510c4b0542_20260915T040842Z_receipt.json) govern these claims. The [historical source card](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/brain-navigator-r2/card/README.md) is retained for provenance; its wording alone is not qualification evidence.

Canonical card authoring path: `brain-navigator-r2/card/README.md` in `szl-holdings/szl-forge`. This source correction preserves the stronger dated published evidence; it changes no receipt, model file, release gate or deployed service.
