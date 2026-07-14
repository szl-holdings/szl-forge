# RUNBOOK — SZL-Khipu-1.5B-BrainNavigator (owner-metal)

How to make the Khipu **BrainNavigator** genuinely TRAINED + EVALUATED on your
own hardware (RTX 5050, Windows/PowerShell) and have the live Alloy family wall
flip from `DERIVED-UNTRAINED` to receipt-verified — **without a single number
typed in by hand**.

The backbone answers "is Khipu trained/evaluated?" **only** from cryptographic
evidence (`khipuEvidence.ts`). It reads committed owner-signed receipts and
re-verifies them (ed25519 over a canonical payload, base-model pin, dataset pin,
contract-fingerprint pin, and an eval→training chain). Anything
missing/tampered/mis-keyed stays honestly `UNTRAINED` / `NOT_EVALUATED` with a
precise reason. **Nothing here upgrades Λ.**

Everything runs from `docs/forge/khipu/`.

> **The one-liner.** From ANY PowerShell window:
> `irm https://raw.githubusercontent.com/szl-holdings/szl-forge/main/khipu/forge-khipu.ps1 | iex`
> That does every step below and prints the three files to send back. The manual
> steps are here for when you want to run (or debug) one stage at a time.

---

## What Khipu is

A **governed retrieval navigator**. Given a QUERY and a set of candidate Brain
node **HANDLES** (ids + synthetic metadata only — never node content), it
proposes a retrieval **PLAN** as JSON: route over the handles, CITE only the
handles whose metadata supports the query, and **ABSTAIN** (zero citations, an
`abstainReason`) when no offered handle supports it. It holds no node content and
never answers from memory — the A11oy controller resolves handles *outside* the
weights. The curriculum is fully **synthetic**, so this measures routing-POLICY
conformance over synthetic scenarios, **not** real-Brain navigation skill.

This is a SEPARATE model from SZL-1 (the 3B sovereign) and the ReceiptAgent —
different base dir (`khipu-model/`), different Ollama name (`khipu`), different
receipt kinds (`szl-khipu-training-receipt` / `szl-khipu-eval-receipt`).

---

## What is committed vs. what stays on your machine

| Committed to the repo (public) | Stays on your laptop (never committed) |
| --- | --- |
| curriculum: `train.jsonl`, `eval.jsonl`, `train.abstain.jsonl`, `adversarial.jsonl`, `khipu.schema.json`, `manifest.json` | the ed25519 **private key** (`~/.a11oy/khipu_owner_ed25519.pem`) |
| the forge kit (`*.py`, `*.ps1`, `RUN_ME.bat`, `Modelfile.khipu.gguf`) | merged weights (`khipu-model/`), adapter, `*.gguf`, `outputs/` |
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

> **Pin TRL to the version your SZL-1 forge already uses.** `train_khipu.py`
> calls `SFTTrainer(model=, tokenizer=, dataset_text_field=, max_seq_length=, args=SFTConfig(...))`
> — the same signature as the working `szl_forge.py`. TRL ≥ 0.13 renamed
> `tokenizer` → `processing_class` and folded `dataset_text_field` /
> `max_seq_length` into `SFTConfig`, so a *fresh* unpinned install can crash at
> trainer construction. Use the exact `trl` (and `unsloth`) versions from your
> existing SZL-1 environment; the eval + signer paths have no such coupling.

---

## Step 1 — Generate your owner key (one time, ever)

```powershell
cd docs\forge\khipu
python sign_receipt.py keygen
```

This writes the **private** key to `~/.a11oy/khipu_owner_ed25519.pem` (keep it
secret) and the **public** `owner_pubkey.json` into this folder. It prints your
`keyId`.

> **Rotating the key invalidates every existing receipt.** Only re-run with
> `--force` if you truly intend to rotate.

**Optional but recommended — pin the key in production:** set the Replit secret
`A11OY_KHIPU_OWNER_KEYID` to the printed `keyId`. Then the wall shows
`keyTrust=PINNED`; a committed key whose id ever differs from the pin renders a
LOUD `MISCONFIGURED` divergence (fail-closed) instead of silently trusting it.

---

## Step 2 — Train (QLoRA) + sign the training receipt

```powershell
python train_khipu.py
```

- Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` (via the bnb-4bit variant) on the
  committed training rows (navigate plans + abstain plans; the abstain set is
  oversampled in memory to ~1:1, never on disk).
- Re-hashes every curriculum file and cross-checks `manifest.json` — a drifted
  checkout **fails loud** before any GPU work.
- Merges to 16-bit safetensors at `khipu-model/`, saves the LoRA adapter.
- Builds and **signs** `training_receipt.signed.json` (base-model = the
  canonical HF id, dataset sha256 map, contract fingerprints, adapter +
  weights sha256, `finalTrainLoss` as a verbatim STRING).

This never touches SZL-1 or the ReceiptAgent (different base dir, Ollama name).

---

## Step 3 — Birth into Ollama (GGUF)

```powershell
powershell -ExecutionPolicy Bypass -File .\rebirth-khipu.ps1
```

Converts the merge to F16 GGUF with llama.cpp's pure-Python converter, quantizes
to `q4_K_M`, and imports it as the **`khipu`** Ollama model. The final line runs
one real prompt — it must read like a JSON retrieval **PLAN** (a NAVIGATE citing
the offered handle, or an ABSTAIN), **not** `@` spam. (Direct safetensors import
can corrupt the voice; the GGUF path is the fix, same as SZL-1's rebirth.)

---

## Step 4 — Sanity gate (train-set reproduction)

```powershell
python sanity_gate.py
```

Runs the freshly-rebirthed model against its **own training prompts** and
requires it to reproduce them (navigate rows route to the same handle; abstain
rows abstain cleanly). If it can't even reproduce its training set it is
undertrained — this **aborts** so eval never scores (or signs) a bad model. If it
fails, copy the whole `[sanity]` block to Alloy so the levers (epochs /
`ABSTAIN_OVERSAMPLE`) can be retuned. It reads **only** the training curriculum,
never the held-out set, and writes no receipt.

---

## Step 5 — Evaluate (held-out) + sign the eval receipt

```powershell
python eval_khipu.py
```

- Re-hashes the held-out files against `manifest.json` first (a softened
  adversarial set can never be scored while the receipt still carries the
  committed hashes).
- Runs the **held-out** curriculum against the served `khipu`:
  - `eval.jsonl` (navigate) → output must parse **and** validate against
    `khipu.schema.json`, pass the cross-field contract, and route to the same
    handle the reference cited → `groundingCorrect`.
  - `adversarial.jsonl` (no handle supports the query) → output must **ABSTAIN**
    → `abstainCorrect`.
- Records **raw integer counts only** — `planTotal/planValid`,
  `groundingTotal/groundingCorrect`, `abstainTotal/abstainCorrect`,
  `hallucinatedCitationCount` (the backbone DERIVES percentages; a lone scalar
  could fabricate a score, counts cannot; a nonzero hallucination count is a
  true, visible failure, never rounded to zero).
- Chains to the exact training receipt via
  `trainingReceiptSha256 = sha256(training receipt canonical)`.
- Signs `eval_receipt.signed.json`.

The held-out **abstain** rate — not the memorizable routing rate — is the
meaningful honesty score. A poor model gets a poor — but honest — receipt.
**No number is ever hand-edited.**

---

## Step 6 — Commit + verify live

Commit **exactly** these three files (nothing else from the folder):

```powershell
git add owner_pubkey.json training_receipt.signed.json eval_receipt.signed.json
git commit -m "khipu: owner-signed training + eval receipts"
git push
```

After deploy, `GET /api/forge/family` → the Khipu evidence band should show
`TRAINED_RECEIPT_VERIFIED` + `EVAL_RECEIPT_VERIFIED` with your `keyId` and the
DERIVED conformance/abstain percentages. If any check fails, the wall stays
honestly `UNTRAINED` / `NOT_EVALUATED` and names the exact failing check.

---

## Publishing the weights (optional, separate honesty axis)

Pushing the repo to the SZLHOLDINGS Hub flips `publishStatus` to `PUBLISHED`
**only** as a repo-existence fact — it is **never** a trained/serving/eval claim,
and it never upgrades `trainingStatus`. Training/eval status comes solely from
the signed receipts above.

After a PASS eval, one paste from this folder publishes everything — receipts
to the szl-forge trust root, receipts + weights + adapter to the Hub repo
(`SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator`, ReceiptAgent layout):

```powershell
powershell -ExecutionPolicy Bypass -File publish-khipu.ps1
```

The script refuses to run if `owner_pubkey.json`'s `keyId` differs from the
pinned owner keyId (`89540347a69b789e` — rotate deliberately or not at all),
and it never reads or uploads the private key. It is a *courier*, not a judge:
receipt verification stays server-side.

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
- The curriculum is fully synthetic: a good eval means Khipu learned the
  **routing policy** (ground-only, abstain-when-unsupported), **not** that it can
  navigate the real Brain. That remains an explicit roadmap gap.
- Nothing in this flow upgrades Λ (Conjecture-1).
