---
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - governed-ai
  - proposal-only
  - research-only
  - szl-holdings
  - chaski
  - cutting
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  jobs: local-5050
  job_id: local-5050
  weights: AVAILABLE
  evals: HISTORICAL_TRAINING_NONE_SEPARATE_GATE_FAIL
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
  copied_live_chaski_weights: false
  train_loss: 2.228136855544466
  train_loss_label: MEASURED
  adapter_sha256: 620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5
track_downloads: true
datasets:
  - SZLHOLDINGS/verticals-sft-v1
szl_card:
  system: szl-card/v1
  display_title: chaski-5050
  quechua_name: Hamut'aq 5050
  quechua_name_status: CHAKANA_REGISTRY root (proposed), Amaru holo-v2 title
  quechua_name_source: Chakana model-names registry (display only)
  hub_repo_unchanged: SZLHOLDINGS/chaski-5050
  draft_from_hub_revision: c1f9882e556550bafc3716f84902d0272b992ddb
  github_source: szl-forge
  github_binding: PINNED_COMMIT
  evidence_lattice:
    LICENSE: PRESENT
    BASE: PRESENT
    TRAIN: PRESENT
    EVAL: UNKNOWN
    SIGNED: GAP
    HELD-OUT: FAIL
    STD BENCH: NOT_RUN
    GITHUB: PRESENT
  standard_benchmarks: NOT_RUN
  draft_state: HUB_PR_UNMERGED
  title_banner: assets/banner.svg (holo-v2 style, repo-name title)
---
<!-- szl:artifact-identity-reconciled -->
> **Artifact identity (noted 2026-11-27).** Besides the LoRA adapter, this repository's root carries a merged full-precision checkpoint (`model.safetensors` + `config.json`, loadable with `transformers`) produced by the receipted CPU merge of this adapter into its declared base (`merge_receipt.json`). The adapter remains the artifact of record for every figure on this card; the merged bytes carry no separate held-out receipt and add no claim. Metadata-only note.

<!-- szl-card-style: holo-v2 2026-11-26 -->
<p align="center">
  <img src="https://huggingface.co/SZLHOLDINGS/chaski-5050/resolve/main/assets/banner.svg" alt="chaski-5050 holographic title" width="100%">
</p>

<h1 align="center">chaski-5050</h1>
<p align="center"><sub>Quechua name: <b>Hamut&#x27;aq 5050</b> · display only, from Chakana's registry · the model name is the repo id</sub></p>
<p align="center"><code>text-generation</code> · <code>peft</code> on <code>Qwen/Qwen3.5-0.8B</code> · Hamut&#x27;aq · line A<br><sub>Card heading: chaski-5050</sub><br><sub>Quechua display name <b>Hamut&#x27;aq</b> — one who cogitates / reasons carefully (agentive of hamut&#x27;ay &#x27;to cogitate&#x27;). Status: proposed in Chakana's naming registry (2026-11-26); display only, repo id unchanged.</sub></p>

<p align="center">
  <img alt="license: apache-2.0" src="https://img.shields.io/badge/license-apache--2.0-7c3aed?style=for-the-badge">
  <img alt="base: Qwen/Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen%2FQwen3.5--0.8B-0ea5e9?style=for-the-badge">
  <img alt="scale (per repo name): UNKNOWN" src="https://img.shields.io/badge/scale%20%28per%20repo%20name%29-UNKNOWN-14b8a6?style=for-the-badge">
  <img alt="publication_eligible: False" src="https://img.shields.io/badge/publication__eligible-False-dc2626?style=for-the-badge">
  <img alt="evals: HISTORICAL_TRAINING_NONE_SEPARATE_GATE_FAIL" src="https://img.shields.io/badge/evals-HISTORICAL__TRAINING__NONE__SEPARATE__GATE__FAIL-f59e0b?style=for-the-badge">
  <img alt="weights: AVAILABLE" src="https://img.shields.io/badge/weights-AVAILABLE-64748b?style=for-the-badge">
</p>

<details>
<summary><b>◈ At a glance</b> — facts copied from this card's YAML; anything absent is UNKNOWN</summary>

| Field | Value |
|:--|:--|
| Repo | `SZLHOLDINGS/chaski-5050` |
| Display title | chaski-5050 |
| Quechua name (secondary, display only) | Hamut'aq 5050 |
| Family | Hamut'aq · line A |
| License (YAML) | `apache-2.0` |
| Base model (YAML) | `Qwen/Qwen3.5-0.8B` |
| Relation (YAML) | `adapter` |
| Library / pipeline (YAML) | `peft` / `text-generation` |
| szl.artifact_class (YAML) | `ADAPTER` |
| szl.weights (YAML) | `AVAILABLE` |
| szl.evals (YAML) | `HISTORICAL_TRAINING_NONE_SEPARATE_GATE_FAIL` |
| szl.publication_eligible (YAML) | `False` |
| szl.autonomy_eligible (YAML) | `False` |

</details>

---

<!-- szl-card: v1 begin -->
![github](assets/szl-card-v1/badge-github.svg) ![receipts](assets/szl-card-v1/badge-receipts.svg) ![std bench](assets/szl-card-v1/badge-stdbench.svg)

![evidence lattice](assets/szl-card-v1/hero-lattice.gif)

> **Evidence lattice.** Each column is one kind of evidence. Solid = present, green = signature verifies, red = FAIL on record, amber = partial or caveat, wireframe = UNKNOWN or NOT RUN. Height shows which evidence exists. It is not a quality score.

## ◈ Evidence at a glance (szl-card/v1)

| Field | Value |
|---|---|
| Hub repo | [`SZLHOLDINGS/chaski-5050`](https://huggingface.co/SZLHOLDINGS/chaski-5050/tree/c1f9882e556550bafc3716f84902d0272b992ddb) @ `c1f9882e5565` |
| Visibility (live read) | public |
| Weights | adapter+full |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base license | apache-2.0 |
| Card license | apache-2.0 |
| NOTICE / license duty | none found |
| GitHub source | `szl-holdings/szl-forge` (PINNED_COMMIT) |
| Commit pins | szl-forge@2eb236117f5a:EXISTS(szl-source-binding.json) |
| Standard benchmarks | NOT RUN (UNKNOWN) |
| Why this card | A. Chaski / frontier abstention · sibling / older iteration |

## Evidence lattice

| Evidence | Status |
|---|---|
| LICENSE | **PRESENT** |
| BASE | **PRESENT** |
| TRAIN | **PRESENT** |
| EVAL | **UNKNOWN** |
| SIGNED | **GAP** |
| HELD-OUT | **FAIL** |
| STD BENCH | **NOT_RUN** |
| GITHUB | **PRESENT** |

## Receipts


| Unsigned receipt / eval file | sha256 at this revision |
|---|---|
| [`adapter-unsloth/training_receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/c1f9882e556550bafc3716f84902d0272b992ddb/adapter-unsloth/training_receipt.json) | `236a443d7c339983…` |
| [`merge_receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/c1f9882e556550bafc3716f84902d0272b992ddb/merge_receipt.json) | `04c7b5835a8a5511…` |
| [`training_receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/c1f9882e556550bafc3716f84902d0272b992ddb/training_receipt.json) | `9420fda23d9ef4db…` |

## Benchmarks

![benchmark board](assets/szl-card-v1/benchmarks.png)

| Benchmark / suite | Result | 95% CI (Wilson) | Receipt provenance |
|---|---|---|---|
| MMLU-Pro | **NOT RUN** | n/a | none |
| GPQA-Diamond | **NOT RUN** | n/a | none |
| GSM8K | **NOT RUN** | n/a | none |
| MATH-500 | **NOT RUN** | n/a | none |
| IFEval | **NOT RUN** | n/a | none |
| BBH | **NOT RUN** | n/a | none |
| ARC-C | **NOT RUN** | n/a | none |
| TruthfulQA | **NOT RUN** | n/a | none |
| SimpleQA | **NOT RUN** | n/a | none |
| AbstentionBench | **NOT RUN** | n/a | none |

Small-n SZL suites are behaviour checks, not capability benchmarks; the confidence interval shows how little n=5 or n=6 can prove.


## Evaluation

No evaluation receipt was found in this Hub revision, so EVAL is **UNKNOWN**.
The separate canonical held-out result is disclosed in the FAIL / QUARANTINE section below.

Standard benchmarks (MMLU-Pro, GSM8K, IFEval, BBH, ARC-C, TruthfulQA, SimpleQA, AbstentionBench): **NOT RUN**. They are planned as baseline item Y1 in the SZL training plan and need Stephen's yes for that run. 

model-index: no receipted results; model-index omitted (nothing to report).

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "SZLHOLDINGS/chaski-5050", revision="c1f9882e556550bafc3716f84902d0272b992ddb")  # pin the revision you verified
```

Interactive 3D view: proposed binding to the public Space [`SZLHOLDINGS/szl-model-inference-lab`](https://huggingface.co/spaces/SZLHOLDINGS/szl-model-inference-lab). Not bound yet; binding needs Stephen's yes.

## Limitations and boundaries

- Research artifact. Not promoted, not autonomous, not a production authorization.
- Every number on this card comes from a receipt file linked above. Anything without a receipt is shown as UNKNOWN or NOT RUN.
- Λ = Conjecture 1 (advisory, never a theorem).
- Card quality gaps found in the audit: no model-index; no standard benchmark; held-out FAIL on record; no limitations / intended-use section; no citation.

## Citation

```bibtex
@misc{szl_chaski_5050,
  title  = {SZLHOLDINGS/chaski-5050},
  author = {{SZL Holdings}},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/SZLHOLDINGS/chaski-5050}},
  note   = {revision c1f9882e556550bafc3716f84902d0272b992ddb}
}
```

<sub>Card system szl-card/v1. Rendered from Hub revision c1f9882e556550bafc3716f84902d0272b992ddb and receipts read on 2026-11-27 UTC. Opened as an unmerged Hub PR for review.</sub>
<!-- szl-card: v1 end -->

---

## Canonical held-out gate: FAIL / QUARANTINE

> **QUARANTINE.** Research residue.

The [2026-08-28 training receipt](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/c1f9882e556550bafc3716f84902d0272b992ddb/training_receipt.json) records `evals=none-this-run`: no JSON/refusal gate ran in that training step.

A [separate 2026-09-15 canonical gate receipt](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/evaluation/gate_runs/chaski-5050_63ea80915ccb_20260915T041028Z_receipt.json) at historical Hub revision `63ea80915ccb663b4aa589cbf2ce52a1dec574fe` reports **1/5 held-out and overall FAIL**, with `heldout.refusal_no_regression=false`. Its separate refusal command passed; that does not change the overall failure. Production and publication authorization are **false**. This receipt is a distinct owner-run suite; its applicability to the current distributed forms was not independently established in this review. No 5/5 or 6/6 qualification is claimed.

Lab load is forbidden. The card does not authorize artifact changes or model promotion. Runtime loading has not been verified. Root adapter metadata contains a nonportable owner-local base path.

The training receipt describes an owner-local recipe of bf16 LoRA r=16, α=16, seed 11, and the dataset `SZLHOLDINGS/szl-1-doctrine-sft`, 41 rows.

Lineage and evaluation applicability were not independently established here for `adapter-unsloth/`; it is not interchangeable with the root adapter by assumption.

Canonical card authoring source: [chaski-5050/card/README.md](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/chaski-5050/card/README.md).


<!-- HOLO3D:v1 -->
<details>
<summary><b>▸ SZL Holo-Stats 3D</b></summary>

## SZL Holo-Stats 3D

![SZL Holo-Stats 3D](assets/holo_stats_3d.svg)

*Live hub metrics, isometric holo render, geometrically non-overlapping, 2026-11-26T02:41Z. Receipts: SZLHOLDINGS/wave-logs mesh/2026-11-26/ouroboros/holo3d-cards/*



<p align="center"><img src="https://huggingface.co/SZLHOLDINGS/chaski-5050/resolve/main/holo/banner.svg" alt="SZL Holo Banner"></p>

[![org](https://img.shields.io/badge/org-SZLHOLDINGS-FFD21E?logo=huggingface)](https://huggingface.co/SZLHOLDINGS)
[![downloads](https://img.shields.io/badge/downloads-721-green)](https://huggingface.co/SZLHOLDINGS)
[![state](https://img.shields.io/badge/state-MEASURED-blue)](https://github.com/szl-holdings)
[![code](https://img.shields.io/badge/GitHub-szl--holdings-24292f?logo=github)](https://github.com/szl-holdings)
[![license](https://img.shields.io/badge/license-apache-2.0-blue)](https://huggingface.co/SZLHOLDINGS/chaski-5050)

> **MEASURED** - chaski 50/50 merge base, training + merge receipts

**Estate position:** #3 of 57 org models by downloads (721 in the last 30 days). Keepers: [SZLHOLDINGS org](https://huggingface.co/SZLHOLDINGS) | [szl-holdings on GitHub](https://github.com/szl-holdings)

</details>


<details>
<summary><b>▸ Org leaderboard</b></summary>

## Org leaderboard

![org downloads top-10](./assets/szl-card/org-leaderboard-2026-11-25.svg)

</details>


<details>
<summary><b>▸ Model Overview</b></summary>

## Model Overview

| Attribute | Value |
| --- | --- |
| Developer | SZL Holdings |
| Repo class | MERGED |
| Base model | Qwen/Qwen3.5-0.8B |
| State | MEASURED (strict done-bar: measured receipts required; see ledger) |
| Role | chaski 50/50 merge base, training + merge receipts |
| Notes | - |

</details>


## Measured Results (receipts-first)

| Receipt | Score | State | Model/lane | Curriculum |
| --- | --- | --- | --- | --- |
| [training_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/main/adapter-unsloth/training_receipt.json) | - | - | - | SZLHOLDINGS/szl-1-doctrine-sft |
| [training_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/main/training_receipt.json) | - | - | - | SZLHOLDINGS/szl-1-doctrine-sft |
| [merge_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/main/merge_receipt.json) | - | - | - | - |
| [mass-battery-chaski_5050_v2_trained.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/benchmark/mass-battery-chaski_5050_v2_trained.json) | 4/6 | MEASURED | SZLHOLDINGS/chaski-5050-v2-trained | SZLHOLDINGS/verticals-sft-v1 |
| [estate-battery-chaski_5050_laptop_lora.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_5050_laptop_lora.json) | - | - | SZLHOLDINGS/chaski-5050-laptop-lora | - |
| [estate-battery-chaski_5050_estate_sft_v1.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_5050_estate_sft_v1.json) | - | - | SZLHOLDINGS/chaski-5050-estate-sft-v1 | - |

## Measured Graphs & Metrics

<!-- SZL-CARD-GRAPHS-20261125:BEGIN -->

## SZL Card Graphs - 2026-11-25

Cutting-edge estate visualization rendered directly from receipt artifacts. No fabricated numbers. 

![chaski-5050](assets/szl-card/training-2026-11-25.png)

<details>
<summary><b>▸ Training</b></summary>

## Training
Fully trained - verified by training receipt. Final train loss n/a | base Qwen/Qwen3.5-0.8B | receipt file: adapter-unsloth/training_receipt.json

Generated by the SZL card-graphs rail on 2026-11-25 from in-repo receipt artifacts (anti-overclaim doctrine).
<!-- SZL-CARD-GRAPHS-20261125:END -->

</details>


<details>
<summary><b>▸ Quickstart</b></summary>

## Quickstart

- Merged weights, load directly:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("SZLHOLDINGS/chaski-5050", device_map="auto")
t = AutoTokenizer.from_pretrained("SZLHOLDINGS/chaski-5050")
```

</details>


<details>
<summary><b>▸ Corpus &amp; Dataset-side Receipts</b></summary>

## Corpus & Dataset-side Receipts

Primary corpus: [SZLHOLDINGS/verticals-sft-v1](https://huggingface.co/datasets/SZLHOLDINGS/verticals-sft-v1)

- Dataset [SZLHOLDINGS/frontier-54-runs/mass-battery-chaski_5050_v2_trained.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/benchmark/mass-battery-chaski_5050_v2_trained.json)
- Dataset [SZLHOLDINGS/frontier-54-runs/estate-battery-chaski_5050_laptop_lora.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_5050_laptop_lora.json)
- Dataset [SZLHOLDINGS/frontier-54-runs/estate-battery-chaski_5050_estate_sft_v1.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_5050_estate_sft_v1.json)

</details>


<details>
<summary><b>▸ Provenance &amp; Doctrine</b></summary>

## Provenance & Doctrine

- Receipts-first, fail-closed: every score above links a measured receipt file. No receipt, no claim.
- Pre-overhaul card captured verbatim in wave-logs mesh/2026-11-25/ouroboros/w28t7-card-overhaul/evidence/cards_pre_overhaul.json before this rewrite.
- Governance ledger: knowledge topic szl-estate-decisions (decisions-2.md).
- Compute doctrine: Salad / Massed Computing only for estate model work.

</details>


<details>
<summary><b>▸ Changelog</b></summary>

## Changelog

- 2026-11-25: receipts-first audit + showcase (W28T1/T4); graph blocks appended by estate graphs rail; leader-style overhaul with org leaderboard chart (W28T7, this revision).

<!-- SZL-CARD-TEMPLATE-W28T7 -->


<!-- SZL-HOLO-v1 -->

</details>


<details>
<summary><b>▸ Estate Telemetry (3D Holo, real data only)</b></summary>

## Estate Telemetry (3D Holo, real data only)

<p align="center"><img src="https://huggingface.co/SZLHOLDINGS/chaski-5050/resolve/main/holo/metrics.svg" alt="Measured Telemetry"></p>

<p align="center"><img src="https://huggingface.co/SZLHOLDINGS/chaski-5050/resolve/main/holo/chain.svg" alt="Provenance Chain"></p>

Measured live from the HF API at card-build time (2026-11-26T00:47:16Z): downloads 721, likes 0, files 36, trained-weight artifacts 3, checkpoints 0, status TRAINED.

</details>

