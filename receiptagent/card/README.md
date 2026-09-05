---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - qlora
  - governed-agent
  - proposal-only
  - receipt-verified
  - szl-holdings
  - alloy
---

<p align="center">
  <a href="https://huggingface.co/SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent/blob/main/training_receipt.signed.json"><img alt="receipts: training + eval signed" src="https://img.shields.io/badge/receipts-training%20%2B%20eval%20signed-3af4c8?style=flat-square&labelColor=0b0f1a"></a>
  <a href="https://huggingface.co/SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent/blob/main/eval_receipt.signed.json"><img alt="adversarial refusal 6 of 6" src="https://img.shields.io/badge/adversarial%20refusal-6%2F6-b96bff?style=flat-square&labelColor=0b0f1a"></a>
  <a href="https://huggingface.co/SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent/tree/main"><img alt="weights: 1.5B safetensors + LoRA" src="https://img.shields.io/badge/weights-1.5B%20safetensors%20%2B%20LoRA-5b8dee?style=flat-square&labelColor=0b0f1a"></a>
  <a href="https://huggingface.co/SZLHOLDINGS"><img alt="family: SZL-Forge" src="https://img.shields.io/badge/family-SZL--Forge-d7b96b?style=flat-square&labelColor=0b0f1a"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"><img alt="base: Qwen2.5-1.5B-Instruct" src="https://img.shields.io/badge/base-Qwen2.5--1.5B--Instruct-3af4c8?style=flat-square&labelColor=0b0f1a"></a>
  <a href="https://github.com/szl-holdings/szl-forge"><img alt="authority: outside the tensor" src="https://img.shields.io/badge/authority-outside%20the%20tensor-b96bff?style=flat-square&labelColor=0b0f1a"></a>
</p>

<h1 align="center">SZL-Forge-1.5B-ReceiptAgent</h1>

<p align="center"><strong>The agent that cannot act.</strong><br>
<sub>It proposes. The controller signs. The weights never hold the keys.</sub></p>

<!--
  Model card for SZL-Forge-1.5B-ReceiptAgent. Every number below is DERIVED from
  the committed owner-signed receipts (training_receipt.signed.json +
  eval_receipt.signed.json), which the Alloy backbone independently re-verifies
  at /api/forge/family. Do NOT hand-edit a number here — regenerate it from the
  receipts, or it becomes a fabrication. License Apache-2.0 matches the
  Qwen2.5-1.5B-Instruct base; the operator may change it at any time.
-->

A **governed, proposal-only** fine-tune of `Qwen/Qwen2.5-1.5B-Instruct`. It
emits evidence-bound, approval-gated decision **drafts** as JSON — it never
finalizes, never executes, and never fabricates a number, citation, or receipt.
Asked to overstep that boundary, it **refuses** — and the refusal rate is the
metric we signed.

> **Provenance, not vibes.** Every capability claim on this card is backed by an
> ed25519 owner-signed receipt committed alongside the weights and
> **independently re-verified** by the Alloy backbone. Verify it yourself below.
> Nothing here is asserted that a signature does not already prove.

<p align="center">✦ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ✦ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ✦</p>

<!-- SZL-ATELIER-CUT:v1:START -->
## The cut

Agents that 'use tools' skip the envelope. This model cannot act. It proposes. The controller signs. That split is the product.

An agent whose weights are physically incapable of being the actor. Authority lives outside the tensor.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Claude tool-use, but the tool is always 'emit a proposal'. |
| NVIDIA | NIM agent runtime, minus the runtime — we refuse to let the weights call. |
| Unsloth | QLoRA on Qwen2.5-1.5B-Instruct, receipt-verified tag, signed train+eval. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Alloy controller inbound. Never a naked chatbot.

## Limitations

- Proposal-only. Ungoverned decode is misuse.
- Owner eval, not a public leaderboard.

Canonical GitHub: [`szl-holdings/szl-forge`](https://github.com/szl-holdings/szl-forge/blob/main/receiptagent/)
<!-- SZL-ATELIER-CUT:v1:END -->

## The architecture of a refusal

```
        ┌──────────────────────────────────────────────┐
        │                 human operator               │
        └───────────────────┬──────────────────────────┘
                            │ approves / denies
        ┌───────────────────▼──────────────────────────┐
        │            Alloy controller boundary         │
        │   validates schema · gates approval · acts   │
        └───────────────────┬──────────────────────────┘
                            │ draft in · receipt out
        ┌───────────────────▼──────────────────────────┐
        │      SZL-Forge-1.5B-ReceiptAgent (1.5B)      │
        │   proposes JSON drafts — or refuses, signed  │
        └──────────────────────────────────────────────┘
```

The arrow of authority never points into the tensor. Every draft leaves the
weights stamped `decision=DRAFT`, `approvalRequired=true`, `executed=false`,
`provenance=MODEL_PROPOSED`, `receiptBinding.status=NOT_BOUND` — with at least
one evidence citation carrying an honest label
(`MEASURED / REPORTED / DECLARED / SIMULATED / UNKNOWN / UNAVAILABLE`).

## Training (REPORTED — owner-metal, not server-measured)

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** QLoRA SFT with **response-only loss masking** and **refusal
  oversampling** (the adversarial refusal set is held out from training). Full,
  reproducible recipe: `train_receiptagent.py` in the forge kit.
- **Final train loss:** `0.1038` (REPORTED by the owner's signed training receipt)
- **Trained at:** `2026-07-13T21:33:44Z` on host `betterwithage`
- **Curriculum:** deterministic, schema-validated synthetic drafts + refusals;
  every dataset file is sha256-pinned in the receipt and byte-reproducible.

## Evaluation (REPORTED — held-out, owner-signed)

Measured on a **held-out** curriculum, signed into `eval_receipt.signed.json`
and chained to the training receipt:

| Metric | Result |
| --- | --- |
| Draft-conformance (schema-valid drafts) | **5 / 5 (100%)** |
| Adversarial-refusal (correctly refused overstep) | **6 / 6 (100%)** |
| Sanity gate (train-set reproduction, pre-eval) | drafts 15/15 · refusals 8/8 |

The adversarial-refusal rate — not the memorizable conformance rate — is the
meaningful honesty score.

## Verify this model (don't trust — check)

1. The two receipts are ed25519-signed over a canonical JSON string and
   **hash-chained** (`eval.trainingReceiptSha256` == `sha256(trainingCanonical)`).
2. The signing key is committed as `owner_pubkey.json`
   (`keyId e7f01810aaa97394`); its `keyId` re-derives from the SPKI.
3. Every `datasets[*]` sha256 in the receipts equals the committed curriculum
   files, and the output-schema sha equals `receiptagent.schema.json`.
4. The Alloy backbone re-runs all of the above per request and exposes the
   verdict at **`/api/forge/family`** (`evidence.trainingStatus` /
   `evidence.evalStatus`). Anyone can reproduce it against these files.

**Honesty stance:** results are `REPORTED` (produced on owner metal), not
`MEASURED` by a third party. Trust anchor is `REPO_DECLARED` (the key ships in
this repo); it upgrades to `PINNED` when the operator pins `keyId` out-of-band.

## Files & provenance bindings

- **Merged model weights** (`*.safetensors`) — the receipts' `weightsArtifactSha256`
  is a deterministic digest over the sorted `*.safetensors` of the merge
  (basename + bytes), reproducible with `sha256_safetensors_dir` in the forge kit.
  This — **not** any GGUF — is the artifact the signed weights hash covers.
- **LoRA adapter** (`*.safetensors`) — bound by `adapterSha256` the same way.
- `owner_pubkey.json`, `training_receipt.signed.json`, `eval_receipt.signed.json`,
  `receiptagent.schema.json` — the verifiable provenance bundle.
- Any `*.gguf` is a **derived** convenience for llama.cpp / Ollama and is **not**
  covered by the signed weights hash.

## Run it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Draft a decision on raising the rolling-24h spend cap."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

The model returns a single JSON **draft** (`decision=DRAFT`, `approvalRequired=true`,
`executed=false`) or a **refusal** — never a finalized action. Validate the output
against `receiptagent.schema.json` before acting on it.

### Adapter (PEFT) alternative

The LoRA adapter ships under `adapter/` for stacking on the stock base:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(
    base, "SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent", subfolder="adapter"
)
```

## Intended use & limits

- **Use:** proposing governed, evidence-cited decision drafts for a
  human-in-the-loop controller (e.g. Alloy).
- **Not for:** autonomous execution, finalizing actions, or being treated as a
  source of ground-truth numbers. It is a 1.5B proposer, not an oracle.

## Citation

Part of the **SZL-Forge** family by **SZL Holdings**. Provenance verifiable via
the Alloy governed-inference backbone.

---

<p align="center">
  <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> ·
  <a href="https://a-11-oy.com">a-11-oy.com</a> ·
  <a href="https://github.com/szl-holdings/szl-forge">szl-forge (GitHub source · forge kit)</a> ·
  <a href="https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF">Khipu GGUF</a>
</p>

<p align="center"><sub>SLSA: L1 honest · L2 attested · L3 roadmap. Λ = Conjecture 1. Trust ceiling 0.97.</sub></p>
