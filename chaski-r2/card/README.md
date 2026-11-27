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
- trl
- unsloth
- proposal-only
- research-only
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  originality: FINETUNE_DISCLOSED_BASE
  sku: CHASKI-R2
  quant: bf16-lora
  qlora: false
  weights: AVAILABLE
  evals: HISTORICAL_OWNER_RECORDS_UNQUALIFIED
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
track_downloads: true
datasets:
  - SZLHOLDINGS/verticals-sft-v1
szl_card:
  system: szl-card/v1
  display_title: chaski-r2
  quechua_name: Hamut'aq r2
  quechua_name_status: CHAKANA_REGISTRY root (proposed), Amaru holo-v2 title
  quechua_name_source: Chakana model-names registry (display only)
  hub_repo_unchanged: SZLHOLDINGS/chaski-r2
  draft_from_hub_revision: 25451cf8298962cea18413dbbf38dc88574f01b1
  github_source: szl-forge
  github_binding: PINNED_COMMIT
  evidence_lattice:
    LICENSE: PRESENT
    BASE: PRESENT
    TRAIN: PRESENT
    EVAL: PRESENT
    SIGNED: GAP
    HELD-OUT: FAIL
    STD BENCH: NOT_RUN
    GITHUB: PRESENT
  standard_benchmarks: NOT_RUN
  draft_state: HUB_PR_UNMERGED
  title_banner: assets/banner.svg (holo-v2 style, repo-name title)
model-index:
- name: chaski-r2
  results:
  - task:
      type: text-generation
      name: canonical held-out gate
    dataset:
      type: szl-canonical-held-out-gate
      name: SZL canonical held-out gate (n=5)
    metrics:
    - type: pass_rate
      name: canonical held-out gate pass rate (1/5)
      value: 0.2
    source:
      name: SZL receipt
      url: https://huggingface.co/SZLHOLDINGS/chaski-r2/tree/25451cf8298962cea18413dbbf38dc88574f01b1
---
<!-- szl:artifact-identity-reconciled -->
> **Artifact identity (noted 2026-11-27).** Besides the LoRA adapter, this repository's root carries a merged full-precision checkpoint (`model.safetensors` + `config.json`, loadable with `transformers`) produced by the receipted CPU merge of this adapter into its declared base (`merge_receipt.json`). The adapter remains the artifact of record for every figure on this card; the merged bytes carry no separate held-out receipt and add no claim. Metadata-only note.

<!-- szl-card-style: holo-v2 2026-11-26 -->
<p align="center">
  <img src="https://huggingface.co/SZLHOLDINGS/chaski-r2/resolve/main/assets/banner.svg" alt="chaski-r2 holographic title" width="100%">
</p>

<h1 align="center">chaski-r2</h1>
<p align="center"><sub>Quechua name: <b>Hamut&#x27;aq r2</b> · display only, from Chakana's registry · the model name is the repo id</sub></p>
<p align="center"><code>text-generation</code> · <code>peft</code> on <code>Qwen/Qwen3.5-0.8B</code> · Hamut&#x27;aq · line A<br><sub>Card heading: chaski-r2</sub><br><sub>Quechua display name <b>Hamut&#x27;aq</b> — one who cogitates / reasons carefully (agentive of hamut&#x27;ay &#x27;to cogitate&#x27;). Status: proposed in Chakana's naming registry (2026-11-26); display only, repo id unchanged.</sub></p>

<p align="center">
  <img alt="license: apache-2.0" src="https://img.shields.io/badge/license-apache--2.0-7c3aed?style=for-the-badge">
  <img alt="base: Qwen/Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen%2FQwen3.5--0.8B-0ea5e9?style=for-the-badge">
  <img alt="scale (per repo name): UNKNOWN" src="https://img.shields.io/badge/scale%20%28per%20repo%20name%29-UNKNOWN-14b8a6?style=for-the-badge">
  <img alt="publication_eligible: False" src="https://img.shields.io/badge/publication__eligible-False-dc2626?style=for-the-badge">
  <img alt="evals: HISTORICAL_OWNER_RECORDS_UNQUALIFIED" src="https://img.shields.io/badge/evals-HISTORICAL__OWNER__RECORDS__UNQUALIFIED-f59e0b?style=for-the-badge">
  <img alt="weights: AVAILABLE" src="https://img.shields.io/badge/weights-AVAILABLE-64748b?style=for-the-badge">
</p>

<details>
<summary><b>◈ At a glance</b> — facts copied from this card's YAML; anything absent is UNKNOWN</summary>

| Field | Value |
|:--|:--|
| Repo | `SZLHOLDINGS/chaski-r2` |
| Display title | chaski-r2 |
| Quechua name (secondary, display only) | Hamut'aq r2 |
| Family | Hamut'aq · line A |
| License (YAML) | `apache-2.0` |
| Base model (YAML) | `Qwen/Qwen3.5-0.8B` |
| Relation (YAML) | `adapter` |
| Library / pipeline (YAML) | `peft` / `text-generation` |
| szl.artifact_class (YAML) | `ADAPTER` |
| szl.weights (YAML) | `AVAILABLE` |
| szl.evals (YAML) | `HISTORICAL_OWNER_RECORDS_UNQUALIFIED` |
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
| Hub repo | [`SZLHOLDINGS/chaski-r2`](https://huggingface.co/SZLHOLDINGS/chaski-r2/tree/25451cf8298962cea18413dbbf38dc88574f01b1) @ `25451cf82989` |
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
| EVAL | **PRESENT** |
| SIGNED | **GAP** |
| HELD-OUT | **FAIL** |
| STD BENCH | **NOT_RUN** |
| GITHUB | **PRESENT** |

## Receipts


| Unsigned receipt / eval file | sha256 at this revision |
|---|---|
| [`adapter-unsloth/training_receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/adapter-unsloth/training_receipt.json) | `055ef442768821a8…` |
| [`eval_merged_measured.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/eval_merged_measured.json) | `4351df066e5078eb…` |
| [`eval_named_n_generate.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/eval_named_n_generate.json) | `c688bbcdfa40508c…` |
| [`evidence/2026-09-25-adapter-binding/receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/evidence/2026-09-25-adapter-binding/receipt.json) | `5e1cbc555ce61ecb…` |
| [`merge_receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/merge_receipt.json) | `a1799878b3010ba5…` |
| [`training_receipt.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/training_receipt.json) | `77b8b850dcb8c76e…` |
| [`training_receipt.local.json`](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/25451cf8298962cea18413dbbf38dc88574f01b1/training_receipt.local.json) | `9a69e1c501db01ba…` |

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
| SZL canonical held-out gate | 1/5 | 0.04-0.62 | GATE_RUN |

Small-n SZL suites are behaviour checks, not capability benchmarks; the confidence interval shows how little n=5 or n=6 can prove.


## Evaluation

![receipted outcomes](assets/szl-card-v1/outcomes-3d.png)

| Source | What it shows |
|---|---|
| `szl-forge frontier/evaluation/gate_runs/chaski-r2_4ef29684c56d_20260915T093042Z_heldout.json` | held-out gate 1/5, heldout_passed=False |

Standard benchmarks (MMLU-Pro, GSM8K, IFEval, BBH, ARC-C, TruthfulQA, SimpleQA, AbstentionBench): **NOT RUN**. They are planned as baseline item Y1 in the SZL training plan and need Stephen's yes for that run. The SZL suites above are small (largest n = 5) and are not capability benchmarks.

model-index: generated from receipts captured this phase.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "SZLHOLDINGS/chaski-r2", revision="25451cf8298962cea18413dbbf38dc88574f01b1")  # pin the revision you verified
```

Interactive 3D view: proposed binding to the public Space [`SZLHOLDINGS/szl-model-inference-lab`](https://huggingface.co/spaces/SZLHOLDINGS/szl-model-inference-lab). Not bound yet; binding needs Stephen's yes.

## Limitations and boundaries

- Research artifact. Not promoted, not autonomous, not a production authorization.
- Every number on this card comes from a receipt file linked above. Anything without a receipt is shown as UNKNOWN or NOT RUN.
- Λ = Conjecture 1 (advisory, never a theorem).
- Card quality gaps found in the audit: no model-index; no standard benchmark; held-out FAIL on record; no limitations / intended-use section; no citation.

## Citation

```bibtex
@misc{szl_chaski_r2,
  title  = {SZLHOLDINGS/chaski-r2},
  author = {{SZL Holdings}},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/SZLHOLDINGS/chaski-r2}},
  note   = {revision 25451cf8298962cea18413dbbf38dc88574f01b1}
}
```

<sub>Card system szl-card/v1. Rendered from Hub revision 25451cf8298962cea18413dbbf38dc88574f01b1 and receipts read on 2026-11-27 UTC. Opened as an unmerged Hub PR for review.</sub>
<!-- szl-card: v1 end -->

---

<!-- SZL-EVIDENCE-PASSPORT:v1:START -->
## Evidence passport

**Dated input observation: 2026-11-13T22:17:00.237505+00:00. Existing research limits remain in force.**

[Inspect the immutable repository snapshot](https://huggingface.co/SZLHOLDINGS/chaski-r2/tree/523b733e20095fda3b1426649348e9fb179098ae).

| Observation | Scope |
|---|---|
| Repository revision | `523b733e20095fda3b1426649348e9fb179098ae` |
| Historical README retrieval | HTTP 200; not a readback of this proposed panel |
| Recognized artifact filenames | 3 filename hints; not tensor verification |
| Structured records retrieved | 5; bounded selection, not an exhaustive evidence census |

Historical input README SHA-256: `935571a2129071d5343c4c280799890e5ff506863f82178deb5bc36a6c1ee8c4`.

[Reviewed authoring baseline](https://github.com/szl-holdings/szl-forge/blob/2f1f7636acf4e77d87de34319c34507f9441dc71/chaski-r2/card/README.md). Existing publisher: `publish-chaski-card.yml`. Source changes still require normal review and exact deployed-byte readback.

### Inspect the records

| Record | Retrieved-byte SHA-256 |
|---|---|
| [training\_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/training_receipt.json) | `77b8b850dcb8c76e139b17c42e198c0ad91ec6b17c43649aa49444729f7a9b49` |
| [training\_receipt.local.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/training_receipt.local.json) | `9a69e1c501db01ba5d50c01fd23f7bc098d7af1ea237a3c84a50cd8fb8242a0c` |
| [adapter-unsloth/training\_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/adapter-unsloth/training_receipt.json) | `29981c3b1f8771e2bcc6b1f325222f7782b279500f1dcd3f22ce47cc8debce84` |
| [eval\_merged\_measured.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/eval_merged_measured.json) | `4351df066e5078ebf9a496a148173df7f3e6b1fa7d9bf7db7ace1afeea5eb866` |
| [eval\_named\_n\_generate.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/raw/523b733e20095fda3b1426649348e9fb179098ae/eval_named_n_generate.json) | `c688bbcdfa40508c148233e93e4676028d559d3492d3649d5435a154401f272e` |

### Interpretation

Documentation and retrieved-record integrity only. No new training, model evaluation, weight-byte verification, independent signer identity, runtime qualification or promotion. Missing evidence is not a zero score.
Training loss is not held-out performance. A copied receipt does not qualify a quantization or merged checkpoint. Existing negative results, license and technical disclosures remain authoritative.

<!-- SZL-EVIDENCE-PASSPORT:v1:END -->

<details>
<summary>Technical details and evidence</summary>

## Canonical post-publication gate: FAIL

The canonical post-publication held-out gate is FAIL (`status: FAIL`).
Receipt: `canonical/frontier/evaluation/gate_runs/chaski-r2_4ef29684c56d_20260915T093042Z_receipt.json` (gate revision `4ef29684c56de07626b1cefb4b93a1b486de2aa1`, observed 2026-09-15T09:35:25Z).
Held-out: 1/5 passed (pass_rate 0.2); refusalPassed true; publicationAuthorization false; productionAuthorization false.

This model is not promotable. publication_eligible: false. no autonomy. Not authorized for production deployment. Research-only.
The 2026-09-25 evaluation binding is adapter-only (adapter-scoped); it does not authorize the merged weights.

The training step itself reported no evaluation.
The later adapter result does not erase the earlier gate **FAIL**, qualify the merged checkpoint, or establish semantic truth or general safety.
The 2026-09-25 adapter bake-off reported 5/5 draft contracts; 6/6 refusal prefixes; that adapter-only binding does not establish merged-model evaluation identity.

</details>

<!-- HOLO3D:v1 -->
<details>
<summary><b>▸ SZL Holo-Stats 3D</b></summary>

## SZL Holo-Stats 3D

![SZL Holo-Stats 3D](assets/holo_stats_3d.svg)

*Live hub metrics, isometric holo render, geometrically non-overlapping, 2026-11-26T02:41Z. Receipts: SZLHOLDINGS/wave-logs mesh/2026-11-26/ouroboros/holo3d-cards/*



<p align="center"><img src="https://huggingface.co/SZLHOLDINGS/chaski-r2/resolve/main/holo/banner.svg" alt="SZL Holo Banner"></p>

[![org](https://img.shields.io/badge/org-SZLHOLDINGS-FFD21E?logo=huggingface)](https://huggingface.co/SZLHOLDINGS)
[![downloads](https://img.shields.io/badge/downloads-401-green)](https://huggingface.co/SZLHOLDINGS)
[![state](https://img.shields.io/badge/state-MEASURED-blue)](https://github.com/szl-holdings)
[![code](https://img.shields.io/badge/GitHub-szl--holdings-24292f?logo=github)](https://github.com/szl-holdings)
[![license](https://img.shields.io/badge/license-apache-2.0-blue)](https://huggingface.co/SZLHOLDINGS/chaski-r2)

> **MEASURED** - chaski r2 merged model, measured merged eval

**Estate position:** #6 of 57 org models by downloads (401 in the last 30 days). Keepers: [SZLHOLDINGS org](https://huggingface.co/SZLHOLDINGS) | [szl-holdings on GitHub](https://github.com/szl-holdings)

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
| Role | chaski r2 merged model, measured merged eval |
| Notes | - |

</details>


## Measured Results (receipts-first)

| Receipt | Score | State | Model/lane | Curriculum |
| --- | --- | --- | --- | --- |
| [eval_merged_measured.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/main/eval_merged_measured.json) | - | - | - | - |
| [eval_named_n_generate.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/main/eval_named_n_generate.json) | PASS | - | - | - |
| [frontier_gpu_summary.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/main/frontier_gpu_summary.json) | 11/12 | - | - | - |
| [training_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/main/training_receipt.json) | - | - | - | - |
| [receipt.json](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/main/evidence/2026-09-25-adapter-binding/receipt.json) | - | MEASURED | - | - |
| [mass-battery-chaski_r2_v2_trained.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/benchmark/mass-battery-chaski_r2_v2_trained.json) | 3/6 | MEASURED | SZLHOLDINGS/chaski-r2-v2-trained | SZLHOLDINGS/verticals-sft-v1 |
| [estate-battery-chaski_r2_laptop_lora.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_r2_laptop_lora.json) | - | - | SZLHOLDINGS/chaski-r2-laptop-lora | - |
| [estate-battery-chaski_r2_estate_sft_v1.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_r2_estate_sft_v1.json) | - | - | SZLHOLDINGS/chaski-r2-estate-sft-v1 | - |

## Measured Graphs & Metrics

<!-- SZL-CARD-GRAPHS-20261125:BEGIN -->

## SZL Card Graphs - 2026-11-25

Cutting-edge estate visualization rendered directly from receipt artifacts. No fabricated numbers. 

![chaski-r2](assets/szl-card/training-2026-11-25.png)

<details>
<summary><b>▸ Training</b></summary>

## Training
Fully trained - verified by training receipt. Final train loss 1.4571 | base Qwen/Qwen3.5-0.8B | receipt file: adapter-unsloth/training_receipt.json

Generated by the SZL card-graphs rail on 2026-11-25 from in-repo receipt artifacts (anti-overclaim doctrine).
<!-- SZL-CARD-GRAPHS-20261125:END -->

</details>


<details>
<summary><b>▸ Quickstart</b></summary>

## Quickstart

- Merged weights, load directly:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("SZLHOLDINGS/chaski-r2", device_map="auto")
t = AutoTokenizer.from_pretrained("SZLHOLDINGS/chaski-r2")
```

</details>


<details>
<summary><b>▸ Corpus &amp; Dataset-side Receipts</b></summary>

## Corpus & Dataset-side Receipts

Primary corpus: [SZLHOLDINGS/verticals-sft-v1](https://huggingface.co/datasets/SZLHOLDINGS/verticals-sft-v1)

- Dataset [SZLHOLDINGS/frontier-54-runs/mass-battery-chaski_r2_v2_trained.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/benchmark/mass-battery-chaski_r2_v2_trained.json)
- Dataset [SZLHOLDINGS/frontier-54-runs/estate-battery-chaski_r2_laptop_lora.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_r2_laptop_lora.json)
- Dataset [SZLHOLDINGS/frontier-54-runs/estate-battery-chaski_r2_estate_sft_v1.json](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/blob/main/receipts/estate-battery-chaski_r2_estate_sft_v1.json)

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

<p align="center"><img src="https://huggingface.co/SZLHOLDINGS/chaski-r2/resolve/main/holo/metrics.svg" alt="Measured Telemetry"></p>

<p align="center"><img src="https://huggingface.co/SZLHOLDINGS/chaski-r2/resolve/main/holo/chain.svg" alt="Provenance Chain"></p>

Measured live from the HF API at card-build time (2026-11-26T00:47:28Z): downloads 401, likes 0, files 48, trained-weight artifacts 3, checkpoints 0, status TRAINED.

</details>

