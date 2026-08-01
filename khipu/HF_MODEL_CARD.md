---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - qlora
  - governed-agent
  - retrieval
  - brain-navigator
  - grounded-only
  - proposal-only
  - research-only
  - synthetic-evaluation
  - szl-holdings
  - alloy
---

# SZL-Khipu-1.5B

> **STATUS: MEASURED_RESEARCH_ONLY / NOT_PROMOTED.** This is a real QLoRA
> fine-tune of `Qwen/Qwen2.5-1.5B-Instruct`, with public merged and adapter
> weights bound by expected hashes and owner-signed training/evaluation
> receipts. The held-out abstention result is **2/6**, so autonomous or
> high-stakes use is prohibited.

## Artifact truth card

| Field | Classification |
|---|---|
| Artifact | **Trained fine-tuned weights**, not a recipe, software-only card, or formal verifier. |
| Intended role | Proposal-only governed retrieval plans over synthetic Brain node handles. |
| Evidence | Repository-declared Ed25519 key continuity, signed training/evaluation receipts, source and dataset bindings, and expected public weight hashes. |
| Evaluation | **MEASURED_RESEARCH_ONLY** on a synthetic held-out curriculum; abstention 2/6 is a visible failing limitation, not a green gate. |
| Promotion | `NOT_PROMOTED_RESEARCH_ONLY`; a controller must validate plans and retain all content resolution and execution authority. |
| Unavailable claims | Independent certification, live-Brain navigation quality, autonomous execution, broad capability, and safety are **UNAVAILABLE / NOT ESTABLISHED**. |

## Investor value

Khipu demonstrates a narrow architecture in which learned weights propose
routes while evidence resolution and authority remain outside the model. The
public evidence also preserves a poor held-out abstention result instead of
hiding it behind an aggregate score.

## Intended use

Supply a query and candidate handles containing IDs plus synthetic metadata.
The model proposes JSON conforming to `khipu.schema.json`: either `NAVIGATE`
with citations limited to offered handles, or `ABSTAIN` with no citations.
A validating controller must reject malformed or unsupported output and must
resolve any content outside the weights.

## Evaluator quickstart

From the canonical [`szl-forge`](https://github.com/szl-holdings/szl-forge)
source checkout:

```bash
python khipu/sanity_gate.py
python khipu/eval_khipu.py --help
python tools/publish_model_source_bindings.py \
  --source-revision "$(git rev-parse HEAD)"
```

The binding plan is read-only unless an authorized protected publication flow
is explicitly invoked. Compare the expected weight hashes, receipt files, key
scope, and exact Hub revision before loading remote artifacts.

## Model use after verification

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "SZLHOLDINGS/SZL-Khipu-1.5B"
revision = "<reviewed-immutable-hub-commit>"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype="auto",
    device_map="auto",
)
```

Validate every generated plan against `khipu.schema.json`; never execute model
output directly.

## Evidence and limitations

- The curriculum measures synthetic routing-policy conformance, not navigation
  of live Brain content.
- Repository-declared key continuity is not independent ownership or authorship
  certification.
- Public weight bytes are observed and hash-bound separately from the signed
  training receipt.
- The quantized `SZL-Khipu-1.5B-GGUF` repository is a derivative distribution;
  reproducible quantization is not claimed.
- The bounded Model Inference Lab emits unsigned runtime records. Training and
  evaluation signatures do not cover those generated outputs.
- Outputs may be wrong. The controller and human review boundary is mandatory.

## Citation

Part of the **SZL Forge** family by **SZL Holdings**. Canonical source and
release policy:
[`publishing/model-source-bindings.json`](https://github.com/szl-holdings/szl-forge/blob/main/publishing/model-source-bindings.json).
