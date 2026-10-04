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

<!-- SZL-CARD-PRESENTATION:v1 -->
<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112" /></a></p>

# Chaski

A controller-bound drafting experiment that retains its failed qualification as research evidence.

**Artifact:** Fine-tuned checkpoint; separate LoRA adapter · **Stage:** HOLD · failed qualification

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-forge/tree/main/chaski) · [Evidence](https://github.com/szl-holdings/szl-forge/blob/2b4d7a0f69e68d60cb0f35f09c9bac837d66acdc/chaski/card/README.md)

Provider compatibility support [requested on 4 October 2026](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/12783). No provider was listed in the dated Hub observation; adoption and use qualification remain unchanged.

## Before you use it

- The historical release gates failed: 0/5 JSON drafts and 2/6 adversarial refusals. Publication and autonomy eligibility remain false.
- Choose the exact checkpoint or adapter form. Adapter evaluation must record the loader class and all 192/192 applied tensors.
- Generated drafts remain untrusted and require an independent validating controller. This artifact is not a promoted flagship.

<details>
<summary>Technical details and evidence</summary>

<!-- SZL-CARD-TECHNICAL:v1:START -->
<!-- szl:loader-class-boundary -->
> **Loader-class boundary (noted 2026-09-30).** The adapter tensors are keyed for the multimodal module layout (`base_model.model.model.language_model.layers.*`, the class `Qwen3_5ForConditionalGeneration` / `AutoModelForImageTextToText`). Under transformers 5.18, `AutoModelForCausalLM` instantiates `Qwen3_5ForCausalLM` (`model.layers.*`); PEFT then applies 0 of the 192 adapter tensors and emits only a warning, so the result is the bare base model (observed and receipted on the sibling `SZLHOLDINGS/chaski-r2` adapter, which shares this exact key layout: byte-identical base outputs on the held-out prompts). Any run that reports a score for this adapter must show adapter key coverage (192/192) and the loader class it used; a record without those fields does not establish that the adapter was applied. Evidence: szl-holdings/szl-forge `tools/geh_v8/evidence_sandbox/chaski_probe/` (CPU replay receipts, #444). Metadata-only note; it authorizes no load, changes no artifact, and adds no claim.


<p align="center">
  <img src="holo-banner.svg" alt="Chaski — SZL holographic banner" width="100%"/>
</p>

<h1 align="center">C H A S K I</h1>

<p align="center"><em>A controller-bound courier experiment with its failed qualification retained.</em></p>

> **RESEARCH / NEGATIVE EVIDENCE. Failed qualification. Not flagship.**
> The historical Named-N run failed both release gates. This card does not promote
> the model, its later SKUs, or any live endpoint.

## Contract

Chaski is an SZL fine-tune of the disclosed Apache-2.0 base
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). It produces
proposal-only drafts intended for an independent validating controller. Model
outputs remain untrusted: this architecture does not establish that the model
cannot invent content. Execution and mutation approval belong to the separate
controller and human approval boundary, not to generated text.

**No uniqueness claim is made.** The value of this artifact is its explicit
controller boundary, disclosed lineage, and retained negative evidence.

| Field | Recorded statement |
|---|---|
| Artifact | Merged fine-tune plus LoRA artifacts are present on the Hub repository |
| Parent/eval revision | `1c55df8652e9d0f7b84356b1e2d54849165ae884` |
| Base relation | `finetune`; `artifact_class: MERGED_FINETUNE` |
| Originality | `FINETUNE_DISCLOSED_BASE` |
| Publication eligibility | `false` |
| Autonomy eligibility | `false` |
| Lambda | Conjecture 1, advisory and never a theorem |
| Declared policy ceiling | `0.97`; not a measured accuracy, confidence calibration, or safety score |

Canonical source:
[`szl-holdings/szl-forge/chaski/card`](https://github.com/szl-holdings/szl-forge/tree/main/chaski/card).
Exact publication source and asset hashes are recorded in
`szl-source-binding.json` on this model repository. A source-binding receipt
identifies card assets; it does not qualify model behavior.

## Artifact and metadata scope

The artifact inventory and small-config review below refer to Hub revision
`90b99063731a8c9e7d4567bfa44231b081ba2932`. This is a documentation review
snapshot, not a newly evaluated model revision.

| Form or metadata | Observed scope and unresolved boundary |
|---|---|
| Root merged checkpoint | `config.json`, `model.safetensors.index.json`, and the nonstandard indexed shard `model.safetensors-00001-of-00001.safetensors` are listed; tensor contents and runtime selection were not tested |
| Root LoRA adapter | `adapter_config.json` and `adapter_model.safetensors` coexist with merged metadata; the adapter must be selected separately from the merged checkpoint |
| `adapter-unsloth/` residue | Separately listed adapter files; their configuration, tensor equality, and evaluation applicability were not established in this review |
| Declared base | Card metadata names `Qwen/Qwen3.5-0.8B`; the inspected root adapter config names `unsloth/Qwen3.5-0.8B` with `revision: null`. An immutable base revision is not recorded there, and equivalence between those distributions is unverified |
| Runtime wrapper | Root adapter metadata names `Qwen3_5ForConditionalGeneration` from `transformers.models.qwen3_5.modeling_qwen3_5`; merged config declares the same architecture. Compatible client versions and successful loading remain unverified |

Immutable small-config references:
[root adapter config](https://huggingface.co/SZLHOLDINGS/chaski/blob/90b99063731a8c9e7d4567bfa44231b081ba2932/adapter_config.json)
and [merged config](https://huggingface.co/SZLHOLDINGS/chaski/blob/90b99063731a8c9e7d4567bfa44231b081ba2932/config.json).
The merged config contains vision settings; the recorded text-only run does
not establish vision qualification. No weights were downloaded or rehashed
for this card correction.

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
`A11OY-MINI`, another revision, another runtime, or another prompt set. The
historical evaluated revision and the later review snapshot are distinct;
current packaged tensor equality to the evaluated artifact was not verified.

## Research loading review

**Lab load forbidden for Chaski.** The review preflight below imports only Python
standard-library modules. It does not fetch artifacts, import a model client,
or load weights. Runtime loading has not been validated by this card correction;
the review preflight does not authorize a runtime. No validated runtime loader
is supplied by this card.

Before a separately authorized offline experiment, a reviewer must select the
form, an immutable Chaski artifact revision, a compatible wrapper/client build,
and, for the adapter form, the actual base distribution and its immutable
revision. The inspected adapter config leaves the base revision unknown; the
example deliberately supplies no fallback to `main` or to the differently named
Qwen distribution. Supplying syntactically valid values does not qualify them.

```python
import os
import re

full_revision = re.compile(r"[0-9a-f]{40}")
form = os.environ.get("CHASKI_REVIEWED_FORM", "")
artifact_revision = os.environ.get("CHASKI_REVIEWED_ARTIFACT_REVISION", "")
wrapper = os.environ.get("CHASKI_REVIEWED_WRAPPER", "")
client_build = os.environ.get("CHASKI_REVIEWED_CLIENT_BUILD", "")

if form not in {"merged", "root-adapter"}:
    raise RuntimeError("Select the separately reviewed merged or root-adapter form")
if not full_revision.fullmatch(artifact_revision):
    raise RuntimeError("An independently reviewed immutable Chaski revision is required")
if wrapper != "Qwen3_5ForConditionalGeneration" or not client_build.strip():
    raise RuntimeError("Record the reviewed compatible wrapper and client build first")

review_inputs = {
    "form": form,
    "artifact_revision": artifact_revision,
    "wrapper": wrapper,
    "client_build": client_build,
}
if form == "root-adapter":
    base_id = os.environ.get("CHASKI_REVIEWED_BASE_ID", "")
    base_revision = os.environ.get("CHASKI_REVIEWED_BASE_REVISION", "")
    if base_id != "unsloth/Qwen3.5-0.8B":
        raise RuntimeError("Resolve the configured Unsloth base before adapter loading")
    if not full_revision.fullmatch(base_revision):
        raise RuntimeError("The adapter config does not record an immutable base revision")
    review_inputs.update(base_id=base_id, base_revision=base_revision)

# Review inputs only. No from_pretrained(), adapter attachment, or inference.
```

For a merged-form experiment, prepare a separately verified local merged
snapshot that excludes adapter metadata, preserves the selected index/shards,
and binds every copied file to the reviewed artifact revision. Do not rewrite
the public repository to achieve that separation. For an adapter-form
experiment, use the reviewed compatible base wrapper and explicitly attach only
the selected root adapter with PEFT. A generic call on the mixed repository
does not establish which form the library selects. Neither form is granted
qualification by this loading note.

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

## License scope

Apache-2.0 is declared in card metadata. A standalone `LICENSE` file is
absent from the catalogued Hub revision
`90b99063731a8c9e7d4567bfa44231b081ba2932`. Artifact license coverage and
ownership have not been independently verified. This card correction does
not add a license file or assert relicensing authority.

## Limitations

Narrow curriculum, small bounded evaluation, CPU-only measured run, and failed
release gates. A new model or runtime revision requires a new immutable
evaluation receipt; this card cannot confer approval.

Doctrine v11 LOCKED. Λ = Conjecture 1. Owner: Stephen Lutar / SZL Holdings.

<!-- SZL-CARD-TECHNICAL:v1:END -->
</details>
