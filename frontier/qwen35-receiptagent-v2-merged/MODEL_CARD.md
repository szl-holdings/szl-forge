---
library_name: transformers
license: apache-2.0
base_model: unsloth/Qwen3.5-0.8B
pipeline_tag: text-generation
language:
  - en
tags:
  - szl-holdings
  - doctrine-v11
  - governed-ai
  - proposal-only
  - receipt-agent
  - qwen3.5
  - merged-lora
  - research
szl:
  doctrine: v11-LOCKED
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: MERGED_FULL_MODEL
  originality: FINETUNE_DISCLOSED_BASE
  predecessor: SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2
  predecessor_role: FROZEN_COMPARATOR_NEVER_OVERWRITTEN
  merge_base: unsloth/Qwen3.5-0.8B
  merge_base_revision: 23c69c53358a07516b5827588b3fdb12ae78fd65
  canonical_base_of_adapter: Qwen/Qwen3.5-0.8B
  base_repos_byte_identical: UNKNOWN
  weights: AVAILABLE
  evals: MERGE_FIDELITY_5_OF_5_OWNER_RUN
  publication_eligible: false
  autonomy_eligible: false
  production_disposition: HOLD
  license_file: UNAVAILABLE
---

# ReceiptAgent v2 merged

**Published derived full-model artifact · bounded merge fidelity · proposal only**

Hub publication of the admitted merge is recorded. The earlier
“QUALIFIED, NOT YET PUBLISHED” admission banner is stale. This full-model
runtime form remains separate from the frozen
[v2 PEFT adapter](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2)
and the 1.5B ReceiptAgent family. The historical publication concerns this
derived artifact; it does not authorize production deployment, a new
default, autonomous operation, or promotion of another model.

## Publication and artifact identities

The canonical [publication record](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/qwen35-receiptagent-v2-merged/publication.json)
records an owner-authorized Hub write following lane admission and the
state `MEASURED_HF_READBACK_VERIFIED`. Its readback compared file presence
and byte sizes. That is not a cryptographic binding of a Hub revision to
the historical equivalence evaluation.

The [2026-09-24 publication follow-up](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/qwen35-receiptagent-v2-merged/publication-followup.20260924.json)
preserves these distinct historical identities:

| Identity | Revision | Scope |
|---|---|---|
| First model-file publication | `138a1c78eaa8553f12abf2c86387571be6861882` | Historical model-file commit |
| Admission-card publication | `6f7dc8e2215b689aa2b7d2a28d794d29f5ba3079` | Historical card commit |
| Follow-up observation | `b770792751f00069fb042e04ee60f28f9faf5ebf` | Documentation and inventory readback recorded on 2026-09-24 |
| Current card and metadata review snapshot | `0ef9ed208b66c258e1d11e9112f0721c3b5c7565` | Current immutable documentation snapshot; not the revision recorded by the historical equivalence run |

The follow-up reports unchanged identities of the six model files across
its observed revisions. It does not retroactively bind or rerun the
equivalence evaluation. This review downloaded no large weights,
performed no inference, and did not inspect endpoint state.

## Historical inventory

The published card's 2026-09-25 inventory reports:

| Path | Bytes | Role |
|---|---:|---|
| `model.safetensors` | 3,412,002,208 | Merged fp32 weights; LFS SHA-256 `3435ba19eafc0bc0781067f31e8192ff84a5b21d307aca89573af880a30feb68` from Hub metadata, not rehashed here |
| `config.json` | 2,874 | `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`, text and vision configuration, `transformers_version=5.17.0` |
| `generation_config.json` | 122 | Generation defaults |
| `tokenizer.json` / `tokenizer_config.json` | 19,989,325 / 1,227 | Tokenizer assets |
| `chat_template.jinja` | 8,149 | Chat template |
| `README.md` / `.gitattributes` | Card / 1,570 | Documentation and platform metadata |

The reviewed repository lists no standalone `LICENSE`, image
`preprocessor_config.json` or `processor_config.json`, or evaluation
receipt file. Merge and equivalence records live in Forge. Image input
loading and image capability are not qualified by this repository's
assets or the text-only evidence linked here.

## Base-model naming

The [candidate record](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/qwen35-receiptagent-v2-merged/candidate.json)
records the merge base as `unsloth/Qwen3.5-0.8B` at immutable revision
`23c69c53358a07516b5827588b3fdb12ae78fd65`, using the multimodal wrapper
`AutoModelForImageTextToText`. The frozen v2 adapter declares canonical
base `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`.
These are distinct repository names. Whether they contain byte-identical
base weights is **UNKNOWN** and was not tested in this card review.

## What the evidence establishes

The [2026-09-13 merge repair receipt](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/receiptagent-v2/salvage_receipt.20260913.json)
reports **372/372 LoRA keys matched**, none unmatched or missing, maximum
weight delta **0.00422266** across **186 tensors**. The earlier empty
base-copy merge was excluded before publication.

The [equivalence receipt](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/receiptagent-v2/qual_v2_salvage_receipt.20260913.json)
reports **5/5 exact matches** between adapter and repaired merged build
under greedy decoding on named prompts. This is a small owner-run
merge-fidelity check. It is not new capability qualification, a broad
benchmark, independent evaluation, or a guarantee that the model never
fabricates citations. The historical equivalence record's limited
publication eligibility and the later recorded publication remain
historical facts; they do not open current production or autonomy gates.

## Text-only usage

This example is **untested in this review**. It follows the reviewed
configuration and the candidate's loader class. It pins the repository
to the card and metadata snapshot, without claiming that this snapshot
is cryptographically bound to the equivalence evaluation. It requires a
compatible Transformers installation and about 3.4 GB for the fp32 weights.

```python
from transformers import AutoTokenizer, AutoModelForImageTextToText

repo = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2-merged"
revision = "0ef9ed208b66c258e1d11e9112f0721c3b5c7565"
tok = AutoTokenizer.from_pretrained(repo, revision=revision)
model = AutoModelForImageTextToText.from_pretrained(repo, revision=revision)
messages = [{"role": "user", "content": "Draft a receipt decision for the attached evidence."}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

The required external output contract is `decision=DRAFT`,
`approvalRequired=true`, `executed=false`. The weights do not enforce that
contract; validate outputs against the ReceiptAgent schema. Validation,
approval, execution, and receipt minting remain outside the model.
No served endpoint, production route, or lab pin is established by the
evidence linked here.

Production disposition in the 2026-09-24 follow-up is **HOLD**;
`promotion_effect=NONE`. Autonomy eligibility remains **false**.

Apache-2.0 is declared in metadata. No standalone `LICENSE` is listed at
the reviewed revision; artifact license coverage was not independently
verified. Any license-file addition requires source-owner review of that
coverage.

Canonical card authoring source: [frontier/qwen35-receiptagent-v2-merged/MODEL_CARD.md](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/frontier/qwen35-receiptagent-v2-merged/MODEL_CARD.md).
Keep the publication, merge, equivalence, and later inventory records at
their declared scopes. Λ = Conjecture 1, advisory, never a theorem.
