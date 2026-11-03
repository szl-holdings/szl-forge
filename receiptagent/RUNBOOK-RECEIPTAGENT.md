# RUNBOOK — SZL-Forge-1.5B-ReceiptAgent (owner-metal)

## Private draft-server contract (source-only output gate)

`serve_receiptagent.py` is the old local **Qwen2.5-1.5B ReceiptAgent** prototype,
not the Qwen3.5 v2 adapter, a paid API, or a qualified production service.
The output gate does not authorize training, weight downloads, model execution,
publication, or any of the historical operator procedures below.

**Runtime prerequisite:** the server interpreter must have `jsonschema`
available, in addition to the existing model-runtime dependencies when actual
generation is separately authorized. The gate was tested with jsonschema 4.26.0.
Missing `jsonschema` prevents server startup with an import error.
There is no permissive substitute validator or model-output fallback if a
dependency or generation is unavailable.

**Breaking request/response change:** POST bodies now require a nonempty
`prompt` and nonempty `evidence_allowlist`. The allowlist is an array of closed
objects containing only `sourceEndpoint` (a nonempty caller-supplied string)
and `label` (one of MEASURED, REPORTED, DECLARED, SIMULATED, UNKNOWN, UNAVAILABLE).
An endpoint may have only one allowlisted label. Optional `max_new_tokens` is
a JSON integer 1–256, default 256. Extra request fields are rejected. The HTTP
body limit is 65,536 bytes; transfer-encoded requests are rejected.

The old raw `text` success field is removed. Success returns `base_model`,
`proposal_only: true`, and a validated `result` object:

- A draft must satisfy the unchanged `receiptagent.schema.json`, with runtime
  provenance fixed to MODEL_PROPOSED, decision DRAFT, approvalRequired true,
  executed false and receiptBinding.status NOT_BOUND. Each citation must match
  an exact caller-supplied endpoint/label pair. Extra authority/signature fields,
  missing citations, forged endpoints and label upgrades are rejected.
- A refusal is a separate **closed JSON object** requiring proposalId,
  capabilityProfile (`SZL-Forge-1.5B-ReceiptAgent`), provenance MODEL_PROPOSED,
  decision REFUSE, nonempty reason, approvalRequired true, controllerBoundary,
  executed false, and receiptBinding (algorithm `ed25519-over-sha256`, status
  NOT_BOUND, note). No claim, evidence, signature or other extra field is allowed.
  Historical free-text REFUSE output is not accepted by this interface.

Only completion tokens are decoded, not the input prompt. Output must be one
strict JSON object: malformed/duplicate/trailing JSON, echoed prompt text,
nonfinite values including numeric overflow, and invalid Unicode are rejected.
There is no JSON extraction/repair or successful raw-text fallback.

Errors return `error` and `base_model`, without raw model text: HTTP400 for
invalid JSON/request/evidence allowlist/token or content-length input;
HTTP422 `OUTPUT_REJECTED` for invalid generation; HTTP503 `UNAVAILABLE` when
local weights are absent, or `GENERATION_UNAVAILABLE` for dependency/loading/
generation failures handled by the prototype.

The host argument permits only `127.0.0.1` or `localhost`; this remains private
and unpaid. It has no authentication, payment entitlement, distributed quota,
or production resource-deadline guarantee. `--check` only checks local model
directory/safetensors-file **presence**; READY is not a hash verification,
generation test, hardware/capacity qualification, or evidence of a live service.

Acceptance checks JSON structure and caller-supplied citation/label membership,
**not truth, evidence authenticity, factual entailment or safety**. No endpoints
are fetched and no authoritative runtime receipts are minted. Training/evaluation
signatures do not sign runtime output. Human review remains outside the model.

---

How to make the ReceiptAgent **genuinely** TRAINED + EVALUATED on your own
hardware (RTX 5050, Windows/PowerShell) and have the live Alloy family wall
flip from `DERIVED-UNTRAINED` to receipt-verified — **without a single number
typed in by hand**.

The backbone answers "is the ReceiptAgent trained/evaluated?" **only** from
cryptographic evidence (`receiptAgentEvidence.ts`). It reads committed
owner-signed receipts and re-verifies them (ed25519 over a canonical payload,
base-model pin, dataset pin, contract-fingerprint pin, and an eval→training
chain). Anything missing/tampered/mis-keyed stays honestly `UNTRAINED` /
`NOT_EVALUATED` with a precise reason. **Nothing here upgrades Λ.**

Everything runs from `docs/forge/receiptagent/`.

---

## What is committed vs. what stays on your machine

| Committed to the repo (public) | Stays on your laptop (never committed) |
| --- | --- |
| curriculum: `train.jsonl`, `eval.jsonl`, `train.refusals.jsonl`, `adversarial.jsonl`, `receiptagent.schema.json`, `manifest.json` | the ed25519 **private key** (`~/.a11oy/receiptagent_owner_ed25519.pem`) |
| the forge kit (`*.py`, `*.ps1`, `Modelfile.receiptagent.gguf`) | merged weights (`receiptagent-model/`), adapter, `*.gguf`, `outputs/` |
| **you commit at the end:** `owner_pubkey.json`, `training_receipt.signed.json`, `eval_receipt.signed.json` | |

`.gitignore` in that folder already blocks the private key and all build
products. The private key defaults **outside** the repo so it can never be
committed by accident (override with `A11OY_OWNER_KEY_PEM`).

---

## Prerequisites (one time)

```powershell
# Python deps for training + signing + eval
pip install unsloth trl datasets cryptography jsonschema
# Ollama installed and running (ollama serve), plus git or curl for llama.cpp
```

> **Pin TRL to the version your SZL-1 forge already uses.** `train_receiptagent.py`
> calls `SFTTrainer(model=, tokenizer=, dataset_text_field=, max_seq_length=, args=SFTConfig(...))`
> — the same signature as the working `szl_forge.py`. TRL ≥ 0.13 renamed
> `tokenizer` → `processing_class` and folded `dataset_text_field` /
> `max_seq_length` into `SFTConfig`, so a *fresh* unpinned install can crash at
> trainer construction. Use the exact `trl` (and `unsloth`) versions from your
> existing SZL-1 environment; the eval + signer paths have no such coupling.

---

## Step 1 — Generate your owner key (one time, ever)

```powershell
cd docs\forge\receiptagent
python sign_receipt.py keygen
```

This writes the **private** key to `~/.a11oy/receiptagent_owner_ed25519.pem`
(keep it secret) and the **public** `owner_pubkey.json` into this folder. It
prints your `keyId`.

> **Rotating the key invalidates every existing receipt.** Only re-run with
> `--force` if you truly intend to rotate.

**Optional but recommended — pin the key in production:** set the Replit secret
`A11OY_OWNER_KEYID` to the printed `keyId`. Then the wall shows
`keyTrust=PINNED`; a committed key whose id ever differs from the pin renders a
LOUD `MISCONFIGURED` divergence (fail-closed) instead of silently trusting it.

---

## Step 2 — Train (QLoRA) + sign the training receipt

```powershell
python train_receiptagent.py
```

- Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` (via the bnb-4bit variant) on the
  **23** committed training rows (15 drafts + 8 refusals).
- Re-hashes every curriculum file and cross-checks `manifest.json` — a drifted
  checkout **fails loud** before any GPU work.
- Merges to 16-bit safetensors at `receiptagent-model/`, saves the LoRA adapter.
- Builds and **signs** `training_receipt.signed.json` (base-model = the
  canonical HF id, dataset sha256 map, contract fingerprints, adapter +
  weights sha256, `finalTrainLoss` as a verbatim STRING).

This never touches SZL-1 (different base, dir, and Ollama name).

---

## Step 3 — Birth into Ollama (GGUF)

```powershell
powershell -ExecutionPolicy Bypass -File .\rebirth-receiptagent.ps1
```

Converts the merge to F16 GGUF with llama.cpp's pure-Python converter, quantizes
to `q4_K_M`, and imports it as the **`receiptagent`** Ollama model. The final
line runs one real prompt — it must read like an evidence-bound DRAFT, **not**
`@` spam. (Direct safetensors import can corrupt the voice; the GGUF path is the
fix, same as SZL-1's rebirth.)

---

## Step 4 — Evaluate (held-out) + sign the eval receipt

```powershell
python eval_receiptagent.py
```

- Runs the **held-out** curriculum against the served `receiptagent`:
  - `eval.jsonl` (5 drafts) → output must parse **and** validate against
    `receiptagent.schema.json` → `evalContractValid`.
  - `adversarial.jsonl` (6) → output must **REFUSE** → `adversarialRefused`.
- Records **raw integer counts only** (the backbone DERIVES percentages — a
  lone scalar could fabricate a score, counts cannot).
- Chains to the exact training receipt via
  `trainingReceiptSha256 = sha256(training receipt canonical)`.
- Signs `eval_receipt.signed.json`.

The adversarial refusal rate is the **meaningful** honesty score; the
(memorizable) draft-conformance rate is secondary. A poor model gets a poor —
but honest — receipt. **No number is ever hand-edited.**

---

## Step 5 — Commit + verify live

Commit **exactly** these three files (nothing else from the folder):

```powershell
git add owner_pubkey.json training_receipt.signed.json eval_receipt.signed.json
git commit -m "receiptagent: owner-signed training + eval receipts"
git push
```

After deploy, `GET /api/forge/family` → the ReceiptAgent evidence band should
show `TRAINED_RECEIPT_VERIFIED` + `EVAL_RECEIPT_VERIFIED` with your `keyId` and
the DERIVED conformance/refusal percentages. If any check fails, the wall stays
honestly `UNTRAINED` / `NOT_EVALUATED` and names the exact failing check.

---

## Publishing the weights (optional, separate honesty axis)

Pushing the repo to the SZLHOLDINGS Hub flips `publishStatus` to `PUBLISHED`
**only** as a repo-existence fact — it is **never** a trained/serving/eval
claim, and it never upgrades `trainingStatus`. Training/eval status comes solely
from the signed receipts above.

---

## Honesty invariants (binding)

- The signature is over a **canonical JSON** string that is byte-for-byte
  identical to `canonicalJson()` in `artifacts/api-server/src/lib/receipts.ts`
  (verified cross-language). Drift there = receipts fail server-side, which is
  the honest outcome.
- Payloads carry **integers and strings only** (no floats) so canonicalization
  is deterministic across Python and JS.
- A signed receipt is a **REPORTED** owner attestation, not a server MEASUREMENT
  — the backbone can only re-verify the owner's ed25519 attestation over the
  exact committed curriculum + contract, and it says so.
- Nothing in this flow upgrades Λ (Conjecture-1).
