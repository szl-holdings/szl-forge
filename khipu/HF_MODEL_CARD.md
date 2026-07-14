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
  - szl-holdings
  - alloy
---

<!--
  Model card for SZL-Khipu-1.5B-BrainNavigator.

  STATUS: this card describes the model the forge kit PRODUCES; the weights are
  NOT yet trained or evaluated. There are deliberately NO metrics below. Numbers
  appear here ONLY after the owner runs the forge and commits the signed
  receipts — and even then they must be DERIVED from
  training_receipt.signed.json + eval_receipt.signed.json (which the Alloy
  backbone independently re-verifies at /api/forge/family), NEVER hand-typed.
  License Apache-2.0 matches the Qwen2.5-1.5B-Instruct base; the operator may
  change it at any time.
-->

# SZL-Khipu-1.5B-BrainNavigator

> ⚠️ **STATUS: UNTRAINED / NOT_EVALUATED.** No weights have been forged yet. This
> card is the honest contract for what the model *will* be once the owner runs
> the forge kit on metal. Until `training_receipt.signed.json` and
> `eval_receipt.signed.json` exist and **verify in-app**, every capability below
> is a *goal*, not a claim — and no metric is stated.

A **governed retrieval navigator** fine-tune of `Qwen/Qwen2.5-1.5B-Instruct`.
Given a query and a set of candidate Brain node **handles** (ids + synthetic
metadata only — never node content), it **proposes** a retrieval **plan** as
JSON: route over the handles, cite only the handles whose metadata supports the
query, and **abstain** when none do. It holds no node content and never answers
from memory — a controller resolves handles *outside* the weights.

> **Provenance, not vibes.** When this model is forged, every capability claim on
> this card will be backed by an ed25519 owner-signed receipt committed alongside
> the weights and **independently re-verified** by the Alloy backbone. Nothing
> will be asserted that a signature does not already prove. Today: nothing is
> asserted.

## What it does

- Emits a single JSON **plan** conforming to the Khipu output schema
  (`khipu.schema.json`): `contentAccess=HANDLES_ONLY`,
  `brainBinding.status=NOT_RESOLVED`, a `decision` of `NAVIGATE` (≥1 citation, no
  `abstainReason`) or `ABSTAIN` (zero citations, an `abstainReason`), and
  `citedNodeIds` that are a **subset of the offered candidates**.
- The model is a **navigator inside a controller boundary**: Alloy validates the
  plan, resolves handles, and applies governance *outside the weights*. The
  model never resolves content and never acts.

## Training (planned — owner-metal, not server-measured)

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** QLoRA SFT with **response-only loss masking** and **abstain
  oversampling** (the adversarial abstain set is held out from training). Full,
  reproducible recipe: `train_khipu.py` in the forge kit.
- **Curriculum:** deterministic, schema-validated **synthetic** navigate +
  abstain plans; every dataset file is sha256-pinned in `manifest.json` and
  byte-reproducible. Synthetic scenarios train the routing **policy**, not real
  Brain content.
- **Final train loss / trained-at / host:** *reported by the signed training
  receipt once forged — absent until then.*

## Evaluation (planned — held-out, owner-signed)

Once forged, measured on a **held-out** curriculum, signed into
`eval_receipt.signed.json` and chained to the training receipt. The eval records
**raw integer counts only** — plan-valid, routing-correct, held-out-abstain, and
`hallucinatedCitationCount` — from which the backbone DERIVES percentages. The
held-out **abstain** rate (not the memorizable routing rate) is the meaningful
honesty score, and a nonzero hallucinated-citation count is a true, visible
failure. **No metric is stated on this card until those receipts verify.**

## Verify this model (don't trust — check)

When the receipts are committed:

1. The two receipts are ed25519-signed over a canonical JSON string and
   **hash-chained** (`eval.trainingReceiptSha256` == `sha256(trainingCanonical)`).
2. The signing key is committed as `owner_pubkey.json`; its `keyId` re-derives
   from the SPKI.
3. Every `datasets[*]` sha256 in the receipts equals the committed curriculum
   files, and the output-schema sha equals `khipu.schema.json`.
4. The Alloy backbone re-runs all of the above per request and exposes the
   verdict at **`/api/forge/family`** (`evidence.trainingStatus` /
   `evidence.evalStatus`). Anyone can reproduce it against these files.

**Honesty stance:** results (when they exist) are `REPORTED` (produced on owner
metal), not `MEASURED` by a third party. Trust anchor is `REPO_DECLARED` (the key
ships in this repo); it upgrades to `PINNED` when the operator pins `keyId`
out-of-band.

## Files & provenance bindings

- **Merged model weights** (`*.safetensors`) — the receipts' `weightsArtifactSha256`
  is a deterministic digest over the sorted `*.safetensors` of the merge
  (basename + bytes), reproducible with `sha256_safetensors_dir` in the forge kit.
  This — **not** any GGUF — is the artifact the signed weights hash covers.
- **LoRA adapter** (`*.safetensors`) — bound by `adapterSha256` the same way.
- `owner_pubkey.json`, `training_receipt.signed.json`, `eval_receipt.signed.json`,
  `khipu.schema.json` — the verifiable provenance bundle (committed post-forge).
- Any `*.gguf` is a **derived** convenience for llama.cpp / Ollama and is **not**
  covered by the signed weights hash.

## Run it (after forging)

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# The user turn is a JSON object: {"query": ..., "candidates": [{nodeId, nodeKind, label, note}, ...]}
user = {
    "query": "Which handle records the rolling 24h spend-cap policy?",
    "candidates": [
        {"nodeId": "node://khipu-synthetic/0000000000000000", "nodeKind": "CLAIM",
         "label": "DECLARED", "note": "synthetic handle - topic tag policy-spend-cap; no node content."}
    ],
}
messages = [{"role": "user", "content": json.dumps(user)}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

The model returns a single JSON **plan** (`decision=NAVIGATE` citing offered
handles, or `decision=ABSTAIN` with an `abstainReason`) — never resolved node
content. Validate the output against `khipu.schema.json` before acting on it.

### Adapter (PEFT) alternative

The LoRA adapter ships under `adapter/` for stacking on the stock base:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(
    base, "SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator", subfolder="adapter"
)
```

## Intended use & limits

- **Use:** proposing governed, grounded-only retrieval plans over Brain node
  handles for a human-/controller-in-the-loop system (e.g. Alloy).
- **Not for:** resolving node content, autonomous retrieval/execution, or being
  treated as a source of ground-truth navigation. It is a 1.5B **proposer** over
  synthetic routing scenarios, not an oracle, and (until forged + evaluated) not
  a demonstrated navigator of the real Brain.

## Citation

Part of the **SZL-Forge** family by **SZL Holdings**. Provenance verifiable via
the Alloy governed-inference backbone.
