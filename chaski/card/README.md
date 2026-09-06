---
thumbnail: https://huggingface.co/SZLHOLDINGS/chaski/resolve/main/holo-banner.svg
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: finetune
tags:
- szl-holdings
- doctrine-v11
- governed-ai
- proposal-only
- research
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  evidence_ceiling: 0.97
  artifact_class: MERGED_FINETUNE
  originality: FINETUNE_DISCLOSED_BASE
  weights: AVAILABLE
  evals: MEASURED
  named_n:
    revision: 1c55df8652e9d0f7b84356b1e2d54849165ae884
    date: '2026-08-28'
    json_draft: 0/5
    adversarial_refusal: 2/6
    label: MEASURED
    gate: fail
    report: eval_report.json
    report_sha256: 4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05
    report_bytes: 3996
    report_commit: db71c243d0176bccff1ff087cd4dd57663bd6502
  publication_eligible: false
  autonomy_eligible: false
  gpu: UNAVAILABLE
---

<p align="center">
  <img src="holo-banner.svg" alt="Chaski — SZL holographic banner" width="100%"/>
</p>

<h1 align="center">C H A S K I</h1>

<p align="center"><em>The courier that cannot invent the dispatch.</em></p>

> **RESEARCH / NEGATIVE EVIDENCE. Failed qualification. Not flagship.**
> The current Named-N run failed both release gates. This card does not promote
> the model, its later SKUs, or any live endpoint.

## Contract

Chaski is an SZL fine-tune of the disclosed Apache-2.0 base
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). It produces
proposal-only drafts behind a validating controller. The model does not execute
actions, authorize mutations, or convert uncertainty into a fact.

**No uniqueness claim is made.** The value of this artifact is its explicit
controller boundary, disclosed lineage, and retained negative evidence.

| Field | Verified statement |
|---|---|
| Artifact | Merged fine-tune plus LoRA artifacts are present on the Hub repository |
| Parent/eval revision | `1c55df8652e9d0f7b84356b1e2d54849165ae884` |
| Base relation | `finetune`; `artifact_class: MERGED_FINETUNE` |
| Originality | `FINETUNE_DISCLOSED_BASE` |
| Publication eligibility | `false` |
| Autonomy eligibility | `false` |
| Lambda | Conjecture 1, advisory and never a theorem |
| Trust ceiling | `0.97` |

Canonical source:
[`szl-holdings/szl-forge/chaski/card`](https://github.com/szl-holdings/szl-forge/tree/main/chaski/card).
Exact publication source and asset hashes are recorded in
`szl-source-binding.json` on this model repository.

## Evaluation

**Named-N: MEASURED FAIL.** The measured run on 2026-08-28 is a failed gate, not a passing benchmark.

| Probe | Result | Interpretation |
|---|---:|---|
| JSON draft | **0/5** | failed |
| Adversarial refusal | **2/6** | failed |
| Hallucinated citations | **0 observed in this bounded run** | not a general guarantee |

Receipt: `eval_report.json`, 3996 bytes, SHA-256
`4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05`,
source commit `db71c243d0176bccff1ff087cd4dd57663bd6502`.

Method: in-process greedy generation over `messages[:-1]`, Transformers 5.16.1,
bf16 CPU, `load_in_4bit=False`. This evidence does not transfer to
`A11OY-MINI`, another revision, another runtime, or another prompt set.

## Loading

The repository contains multiple loadable forms. Select the intended artifact
explicitly rather than relying on loader precedence.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "SZLHOLDINGS/chaski",
    trust_remote_code=False,
)
```

For the LoRA path, load `Qwen/Qwen3.5-0.8B` first and attach the adapter with
PEFT. The root contains both `config.json` and `adapter_config.json`; callers
must verify which path their library selects. The merged shard uses a
non-standard filename resolved through its index.

## Intended use

- Proposal-only JSON drafts with `approvalRequired=true` and `executed=false`.
- Doctrine-faithful abstention and routing experiments.
- Research behind an independent validator and human approval boundary.

## Prohibited interpretation

- Not an autonomous agent, executor, factual oracle, or weapons system.
- Not a passing 5/5 or 6/6 release.
- Not evidence of a deployed Alloy endpoint.
- Not evidence that a later SKU inherits this evaluation.
- **Lab load forbidden for Chaski.**

## Limitations

Narrow curriculum, small bounded evaluation, CPU-only measured run, and failed
release gates. A new model or runtime revision requires a new immutable
evaluation receipt; this card cannot confer approval.

Doctrine v11 LOCKED. Λ = Conjecture 1. Owner: Stephen Lutar / SZL Holdings.
