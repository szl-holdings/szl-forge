---
thumbnail: https://huggingface.co/SZLHOLDINGS/KHIPU-R2/resolve/9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52/og-card.png
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
tags:
  - qlora
  - peft
  - governed-agent
  - proposal-only
  - research-only
  - szl-holdings
  - khipu
  - abstain-retrain
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: ADAPTER
  publication_eligible: false
  autonomy_eligible: false
  never_overwrite: SZLHOLDINGS/SZL-Khipu-1.5B
  jobs: COMPLETED
  job_id: "6a91bf11984507d9db4ea104"
  job_prior_error: "6a91ba2c45686a1580c12020"
  weights: AVAILABLE
  evals: MEASURED
  gpu: UNAVAILABLE
---

# KHIPU-R2

**Research-only · historical abstention 3/6 · publication and autonomy gates closed**

Adapters are on this repo. Abstain is MEASURED 3/6, not a pass.
That statement describes the dated owner-run records below, rather than
a new evaluation or current model qualification. This is a separate
abstention-retraining SKU. Not a replacement for `SZL-Khipu-1.5B`.

Card and metadata review snapshot: [9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52](https://huggingface.co/SZLHOLDINGS/KHIPU-R2/tree/9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52).
No weights were downloaded, no model was loaded, and no evaluation or
signature verification was rerun in this review.

## Distributed forms

| Form | Files | Scope |
|---|---|---|
| QLoRA adapter | `adapter_model.safetensors`, `adapter_config.json` | The PEFT example below explicitly selects this adapter form. |
| Separate fp32 merged checkpoint | Root `model.safetensors`, `config.json`, `merge_receipt.json` | A distinct runtime form; selecting the adapter does not load this merged checkpoint. |

Current artifact bytes and their equality to historical evaluated bytes
were not independently rehashed. The merge receipt's base revision is
`main@merge-time`, which is not an immutable base identity.

## Dated evaluation records

The [2026-08-28 owner-run job record](https://huggingface.co/SZLHOLDINGS/KHIPU-R2/blob/9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52/eval_measured.json)
reports in-process Unsloth generation at temperature 0 with scoring ported
from `eval_khipu.py`. It reports `held_out_in_gradients: false`.

| Metric | Historical result | Limit |
|---|---|---|
| plan-valid | **11 / 11** | Small owner-run synthetic protocol |
| grounding (`eval.jsonl` navigate) | **5 / 5** | Five navigation cases |
| abstain (`adversarial.jsonl`) | **3 / 6** | Visible release blocker; not a pass |
| hallucinated citations | **0** | These cases only |

The [2026-09-13 corrected-harness CPU re-verification](https://huggingface.co/SZLHOLDINGS/KHIPU-R2/blob/9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52/eval_verified_2026-09-13.json)
reports **11/11 plan-valid, 5/5 grounding, and 3/6 abstention** on rebuilt
fp16 merges and the published fp32 merge. The owner fixed a decoding
harness that returned prompt characters instead of generated output;
**all prior fp16 scores produced by that buggy harness are void**.
The corrected runs are separate owner records and do not promote the model.

No signed R2 eval receipt in this atelier. The linked owner-run records
are not an independent evaluation or public leaderboard. GPU **UNAVAILABLE**
describes the CPU-only re-verification record and the earlier card state;
no current hardware availability was tested. Publication and autonomy
eligibility remain **false**.

The original Khipu **2/6** abstention record and R2's **3/6** record remain
separate. No broad quality score or general safety guarantee follows.

## Historical training

The prior card and [training receipt](https://huggingface.co/SZLHOLDINGS/KHIPU-R2/blob/9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52/training_receipt.json)
describe QLoRA r=32, α=64, seed 11, learning rate 2e-4 and 45 epochs,
on 15 navigation rows plus 8 abstention rows oversampled four times.
The reported train loss **0.017188…** is a training metric, not evaluation.
The historical adapter digest is
`e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349`;
it was not recomputed here. The earlier card records successful job
`6a91bf11984507d9db4ea104` and prior error `6a91ba2c45686a1580c12020`
(Trackio 404). This review did not query those jobs or rerun training.

## Adapter usage

This example is **untested in this review**. An owner-verified immutable
base revision is required; none is invented here. The blank base revision
fails before imports or remote loading. The adapter repository is pinned
to the reviewed documentation and metadata snapshot; this is not an
evaluation binding or release approval.

```python
import re

BASE_REVISION = ""  # Supply an owner-verified immutable base commit.
if not re.fullmatch(r"[0-9a-f]{40}", BASE_REVISION):
    raise ValueError("An owner-verified immutable base revision is required")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "SZLHOLDINGS/KHIPU-R2"
adapter_revision = "9fc2f7bebeaf4b3d6d26f7a2a1788a27f8c9ed52"
tok = AutoTokenizer.from_pretrained(base_id, revision=BASE_REVISION)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=BASE_REVISION)
model = PeftModel.from_pretrained(base, adapter_id, revision=adapter_revision)
```

Outputs are proposals requiring external validation and approval. Lab load
remains forbidden. [A11OY-MINI](https://huggingface.co/SZLHOLDINGS/A11OY-MINI)
contains separately scoped legacy and R2 evidence and inherits no KHIPU-R2
score or qualification.

Apache-2.0 is declared in metadata. No standalone `LICENSE` is listed at
the reviewed revision; artifact license coverage was not independently
verified. A license addition requires source-owner review of that coverage.

Canonical card authoring source: [khipu-r2/card/README.md](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/khipu-r2/card/README.md).
Doctrine v11 LOCKED 749/14/163. Λ = Conjecture 1, advisory, never a theorem.
