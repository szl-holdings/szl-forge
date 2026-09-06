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
  evals: none-this-run
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/chaski
  copied_live_chaski_weights: false
  train_loss: 2.228136855544466
  train_loss_label: MEASURED
  adapter_sha256: 620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5
---

> **QUARANTINE.** Research residue. Strip owner-machine absolute paths.
> Not flagship. Not a production checkpoint.

<p align="center">
  <img src="holo-banner.svg" alt="Chaski-5050 — holographic 50/50 banner" width="100%"/>
</p>

<h1 align="center">C H A S K I · 5 0 5 0</h1>

<p align="center"><em>Curriculum as identity. The filename is the experiment.</em></p>

<p align="center">
  <img alt="Base: Qwen3.5-0.8B" src="https://img.shields.io/badge/base-Qwen3.5--0.8B-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/chaski-5050?style=flat-square&color=e879a9&label=downloads"/>
  <img alt="Artifact: bf16 LoRA r=16 a=16" src="https://img.shields.io/badge/artifact-bf16%20LoRA%20r16%20a16-22d3ee?style=flat-square"/>
  <img alt="QUARANTINE — research residue" src="https://img.shields.io/badge/QUARANTINE-research%20residue-dc2626?style=flat-square"/>
  <img alt="Evals: none this run" src="https://img.shields.io/badge/evals-none%20this%20run-b45309?style=flat-square"/>
  <a href="https://huggingface.co/SZLHOLDINGS/chaski"><img alt="Lineage: never overwrites chaski" src="https://img.shields.io/badge/lineage-never%20overwrites%20chaski-fda4af?style=flat-square"/></a>
</p>

<p align="center">
  <code>KANCHAY</code> · Doctrine v11 · Lean <code>749/14/163</code> · Λ = Conjecture 1 (advisory) · <a href="https://a-11-oy.com">a-11-oy.com</a>
</p>

Adapters are present in the target model repository. Evaluation state: none-this-run; no evaluation score is claimed by this card. This card-only release updates documentation and an exact-source binding; it does not modify model artifacts or provider settings.

Owner-GPU recut on an RTX 5050 Laptop (bf16 LoRA, not QLoRA). Original SZL cut of disclosed Apache [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). Not a republish of Qwen tensors. Not an Unsloth-default card. CUTTING.

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Mix-ratio is usually a blog footnote. We named the model after the mix. 50/50 is the cut.

Curriculum as identity. The filename is the experiment.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Balanced helpful/harmless mix, as a named checkpoint. |
| NVIDIA | Recipe variant, published. |
| Unsloth | LoRA on Qwen3.5-0.8B, cutting tag. |

The differentiator is inspectability: the mix is named, the adapter digest is recorded, the evaluation gap stays visible, and the publication controller cannot promote the model.

## Intended use

Ablation sibling of chaski.

## Limitations

- proposal-only and research-only
- No signed held-out evaluation receipt for this 5050 adapter is present in this source tree.
- Publishing this card is a documentation update, not model promotion or autonomy approval.

Canonical GitHub: [`chaski/README_5050.md`](https://github.com/szl-holdings/szl-forge/blob/main/chaski/README_5050.md)
<!-- SZL-ATELIER-CUT:v1:END -->

## Specification

| | |
|---|---|
| **Artifact** | `adapter_model.safetensors` 25,587,104 bytes **AVAILABLE** (sha256 `620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5`) |
| **Job** | `local-5050` (owner metal, **not** an HF Job) |
| **Does NOT overwrite** | [`SZLHOLDINGS/chaski`](https://huggingface.co/SZLHOLDINGS/chaski) |
| **Dataset** | [`SZLHOLDINGS/szl-1-doctrine-sft`](https://huggingface.co/datasets/SZLHOLDINGS/szl-1-doctrine-sft) · 41 rows · jsonl sha256 `ddc5594b…0611a243` |
| **Publication / autonomy** | false / false |
| **Card publication** | Card, banner, and source binding only; weights, adapter, configs, evals, visibility, hardware, collection, and runtime state unchanged |
| **License** | Apache-2.0 |

## Evaluation

**Status: none-this-run.** No JSON/refusal gate ran. Not 5/5. Not 6/6. Do not load this ID into the Khipu lab.

`train_loss` MEASURED `2.228136855544466` is a **train metric**, not an eval (method: Unsloth trainer log, N=41 rows, 3 epochs, 33 steps, `train_runtime` 883.2224s, 2026-08-28 17:56 UTC). File: `training_receipt.json`.

## Training (MEASURED this run)

- Recipe: `train_chaski_bf16_5050.py` · Unsloth 2026.7.2 · transformers 5.5.0 · torch 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 5050 Laptop, 7.96 GB
- LoRA r=16 α=16, bf16, batch 1, ga 4, lr 2e-4, adamw_8bit, seed 11, max_seq 2048
- `copied_live_chaski_weights: false`

## What this is NOT

- Not live Chaski
- Not a Qwen rehost
- Not a GGUF of this adapter. Mini GGUFs are LIVE on [`A11OY-MINI`](https://huggingface.co/SZLHOLDINGS/A11OY-MINI) (evals none-this-run; they do not inherit this card)
- Not production. Lab load forbidden.

## Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, "SZLHOLDINGS/chaski-5050")
```

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory, never a theorem). Owner: Stephen Lutar / SZL Holdings.
