---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- governed-agent
- retrieval
- brain-navigator
- grounded-only
- proposal-only
- research-only
- szl-holdings
- khipu
- abstain-retrain
- weights-unreceipted
szl:
  doctrine: v11-LOCKED
  lean: 749/14/163
  lambda: Conjecture 1 — advisory, never a theorem
  artifact_class: ADAPTER
  weights: PRESENT_UNRECEIPTED
  jobs: UNAVAILABLE
  evals: HISTORICAL_OWNER_RECORD_CURRENT_BINDING_UNVERIFIED
  publication_eligible: false
  autonomy_eligible: false
  original_signed_weights: SZLHOLDINGS/SZL-Khipu-1.5B
  successor_with_weights: SZLHOLDINGS/KHIPU-R2
---

# SZL-Khipu-1.5B-abstain

**Research adapter and derived GGUF · historical receipts present · current binding unverified**

This repository contains adapter and GGUF bytes. A binding from the
historical receipts to the currently published adapter and GGUF has **not
been established in this review**. `PRESENT_UNRECEIPTED` retains the card's
conservative disposition; it does not assert a proven hash mismatch or
that no historical receipts exist. Publication and autonomy eligibility
remain **false**.

Card and evidence review snapshot: [942c02dd4e94f2cb7cfb0432a051c2db2219ee79](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-abstain/tree/942c02dd4e94f2cb7cfb0432a051c2db2219ee79).
Small cards, receipts, and the hash implementation were read. No weights
were downloaded, no directory digest was recomputed, no runtime was tested,
and signatures were not independently verified.

## Historical inventory and digest domains

The earlier card records this 2026-09-25 Hub metadata observation:

| Artifact | Bytes | Raw-file SHA-256 from LFS metadata |
|---|---:|---|
| `khipu-abstain-adapter/adapter_model.safetensors` | 147,770,496 | `da0f948b7a6b555cbd50026eafcbb38ae8187b66be450fd1c3ed27f7eff9bef1` |
| `khipu-f16.gguf` | 3,093,668,832 | `0df16da8dc6d5865370b90880080dfefda028a5e4c0f758825bf40cdcfb93bd2` |

The [historical training receipt](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-abstain/blob/942c02dd4e94f2cb7cfb0432a051c2db2219ee79/training_receipt.signed.json)
reports `adapterSha256=bd5a1a92b24d85bac19a8df203287e1d4173b6d59d4bc7212f862eef60d9f1db`
and `weightsArtifactSha256=86c33222c44349c31c3db9a42429a4bfe7b79acc1b68b26645645e881e553476`.
These must not be compared directly with the raw-file LFS hashes above.

The committed [training script](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-abstain/blob/942c02dd4e94f2cb7cfb0432a051c2db2219ee79/train_khipu_abstain.py)
computes a safetensor directory digest by sorting `*.safetensors` paths,
then hashing each UTF-8 basename followed by its file bytes. This algorithm
does **not** insert a delimiter. A directory digest is a different hash
domain from an individual-file digest. A GGUF is also a different artifact
form from merged safetensors and needs its own conversion binding.

The earlier direct hash comparison therefore did not establish that the
receipt fails to attest the current files. Correctly scoped adapter
directory verification and GGUF conversion provenance remain unresolved.
Byte presence establishes neither loadability nor evaluation applicability.

## Historical evaluation

An [owner-signed evaluation record](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-abstain/blob/942c02dd4e94f2cb7cfb0432a051c2db2219ee79/eval_receipt.signed.json)
dated **2026-09-08T19:00:17.946003+00:00** is present:

| Metric | Historical result |
|---|---|
| Plan-valid | 11/11 |
| Grounding | 4/5 |
| Abstention | **3/6** |
| Hallucinated citations | 0 in that run |

The file carries an owner signature under key ID `89540347a69b789e`; this
review did not establish independent key trust or reverify that signature.
Its binding to the currently published adapter and GGUF remains unverified.
The record is a small owner-reported evaluation, not artifact-scoped
qualification of the current Hub revision. **3/6 abstention is not a pass.**
Historical `SZLHOLDINGS/SZL-Khipu-1.5B` abstention **2/6** and the separate
`SZLHOLDINGS/KHIPU-R2` **3/6** record remain separate observations; neither
result transfers to this repository's current bytes.

## Declared training recipe

The recipe describes Unsloth QLoRA on disclosed
`Qwen/Qwen2.5-1.5B-Instruct`, with runtime base
`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`. Byte equality of those base
repositories was not verified here. LoRA r=32, α=64, seed 11, learning
rate 2e-4, 45 epochs, batch size 1, accumulation 2, response-only loss,
and `adamw_8bit` are recipe declarations. Increasing in-memory abstention
oversampling from 2 to 4 yields 15 navigation rows plus 8×4 abstention
rows. The held-out recipe uses five navigation and six adversarial rows.

The training receipt reports training at
`2026-09-08T18:44:18.461205+00:00` and final train loss **0.0250**.
Train loss is not evaluation. `jobs=UNAVAILABLE` is retained from the
card; it does not erase those historical owner records or assert a new job.

## Research use and authority

The intended outputs are proposal-only JSON retrieval plans (`NAVIGATE`
or `ABSTAIN`) over synthetic Brain node handles. An external controller
validates the plan and resolves content. Stacking adapters is a gated
research procedure requiring its own byte binding and evaluation; no
performance improvement, production use, or autonomous operation is
authorized by this card. The original signed Khipu line remains separate.

Apache-2.0 is declared, and a standalone `LICENSE` is listed in the reviewed
Hub tree. This observation does not independently establish ownership or
downstream artifact license coverage.

Canonical card authoring source: [khipu-abstain/card/README.md](https://github.com/szl-holdings/szl-forge/blob/6f4ac90ff503265cec9c0436587f086e52d6124e/khipu-abstain/card/README.md).
Doctrine v11 LOCKED 749/14/163. Λ = Conjecture 1, advisory, never a theorem.
