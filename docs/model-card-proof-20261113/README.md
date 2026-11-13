# Public model-card evidence audit — 13 November 2026

## Observation and scope

The bounded public audit ran on BETTERWITHAGE from **2026-11-13T22:15:51.896601Z to 22:18:51.403181Z**. It exhausted public model-list pagination at **99 model-type repositories**, retrieved **97 README files** at immutable Hub revisions, and inspected **121 bounded JSON evidence files** across **44 repositories**. Two README requests returned HTTP 401. Private inventory completeness is not qualified. No weights were downloaded, model code loaded, model evaluation rerun or training started.

Full row-level audit, original cards, selected evidence bytes and HTTP log are retained at `C:\Users\steph\szl\runs\model-card-proof-20261113T2215Z`. This source report is an aggregate and selected-identity record, not a copy of the entire audit. Each repository's evidence selection was capped at five JSON files; unselected evidence may exist elsewhere.

| Observation | Count | Limit |
|---|---:|---|
| Public model-type repositories listed | 99 | Registry membership, not a trained-neural-model count |
| README bodies read | 97 | Two HTTP 401 responses remain unobserved |
| Selected JSON evidence files read | 121 | Not all logs, datasets or external evidence |
| Repositories listing adapter filenames | 29 | Can also include a merged checkpoint; bytes not downloaded |
| Other repositories listing recognized checkpoint filenames | 42 | Filename observation, not training or runtime qualification |
| No recognized checkpoint filename listed | 28 | May include kernels, code/specifications or alternative formats |
| Cards with a canonical-org GitHub link | 70 | Link detection, not scientific verification |
| Cards with a full-commit GitHub link | 33 | Exact link syntax, not complete lineage |
| Missing license metadata | 5 | Not a legal conclusion about underlying rights |

Unread cards: `SZLHOLDINGS/szl-lambda-gate` and `SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent`. Missing license metadata: `reasoning-master-v1-candidate`, `quant-master-4b-v1-candidate`, `quant-master-4b-500step-candidate`, `quant-master-1.7b-500step-candidate`, and `SZL-Khipu-1.5B-v2-trained` under SZLHOLDINGS. Do not invent licenses or replace an unread card.

## Receipt verification and measurement boundaries

Ten signed receipt occurrences passed offline signature verification and tampered-message rejection, representing **eight distinct byte payloads**. Khipu and the GGUF repository contain copies of the same receipts. Verification used each receipt's embedded public key and checked canonical JSON against the payload. It does not establish independently anchored signer identity, the truth of reported training, current tensor equality, model quality, rights or quantization equivalence.

Khipu's historical aggregate safetensor hash and current per-file LFS hash use different domains. Their numerical difference must not be mislabeled as demonstrated corruption. Current-to-historical weight equivalence remains unverified.

| Dated owner-reported record | Preserved result | Scope |
|---|---|---|
| Khipu, 14 July 2026 | Plan-valid 11/11; grounding 4/5; abstention 2/6 | Small synthetic protocol; release blocker retained |
| KHIPU-R2, 28 August 2026 | Plan-valid 11/11; grounding 5/5; abstention 3/6 | Separate artifact/run; not a controlled comparative improvement |
| Chaski, 28 August 2026 | JSON drafts 0/5; refusals 2/6 | Failure retained; no inheritance by a later adapter |

R2's corrected-harness records do not rehabilitate the voided scores from its earlier buggy harness. Chaski-R4's historical A/B provenance remains unresolved; a favorable result must not be selected as its sole headline. Nemotron Estate's complete-knowledge wording needs an evidence-bounded scope review, and its existing contamination disclosure must remain prominent.

Trainer-state histories were found in triage study5, abstention-v2, physics-master-v1 and quant-master-v1. They support checkpoint-scoped training-loss visualization only. Training loss is not held-out performance; losses from different objectives are not ranked. A lone final-loss number must not become a fabricated convergence curve.

## Exact featured identities

| Model | Observed Hub revision | Training record SHA-256 |
|---|---|---|
| [SZL-Khipu-1.5B](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B/blob/7328262402e05b10bffa16cf6446da504b05f666/training_receipt.signed.json) | `7328262402e05b10bffa16cf6446da504b05f666` | `7af76dd4f26dcd122012bfd1e47a0f55481a952b86aee28956cf7cfaaf59bd04` |
| [KHIPU-R2](https://huggingface.co/SZLHOLDINGS/KHIPU-R2/blob/3a698a0f9a96eb80c38b11a77e114f318f6e4820/training_receipt.json) | `3a698a0f9a96eb80c38b11a77e114f318f6e4820` | `cd3b5f8b8f083314abd3bc3eee46aa2a6be8491571aa867860e5b4f85c4ed6ce` |
| [chaski](https://huggingface.co/SZLHOLDINGS/chaski/blob/d6306bb06fc4b5d4258ddd694fb493cf82332feb/training_receipt.json) | `d6306bb06fc4b5d4258ddd694fb493cf82332feb` | `7845609c9e540a2ce71743f152237d896c0e0dd9df13d7f2c814232ce61fe3eb` |
| [chaski-5050](https://huggingface.co/SZLHOLDINGS/chaski-5050/blob/6bd12f97b3d19415c018473f9ff0427ad9343b1a/training_receipt.json) | `6bd12f97b3d19415c018473f9ff0427ad9343b1a` | `9420fda23d9ef4db6620a2802c1920df8ac6252f560567ee8e0590b7912a68ed` |
| [chaski-r2](https://huggingface.co/SZLHOLDINGS/chaski-r2/blob/523b733e20095fda3b1426649348e9fb179098ae/training_receipt.json) | `523b733e20095fda3b1426649348e9fb179098ae` | `77b8b850dcb8c76e139b17c42e198c0ad91ec6b17c43649aa49444729f7a9b49` |
| [chaski-r4](https://huggingface.co/SZLHOLDINGS/chaski-r4/blob/0f97176df71cdb8a617590e717683c6a4f59d8fb/training_receipt.json) | `0f97176df71cdb8a617590e717683c6a4f59d8fb` | `fea82085c6bcc2c8f35ee805169207b03e2c0347ef902c85f505cff3168c1af2` |

## Presentation references studied

- [Kimi](https://huggingface.co/moonshotai/Kimi-K3.1-Pro): grouped evaluation tables, generation settings, explicit unavailable entries and deployment instructions.
- [NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4): precision-specific identities, hardware distinctions, training disclosures and reproducible evaluation recipes.
- [Google Gemma](https://ai.google.dev/gemma/docs/core/model_card): intended use, data preprocessing and evaluation limitations.
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards) and [organization cards](https://huggingface.co/docs/hub/organizations-cards): metadata and supported publication surfaces.

These are design/documentation references, not copied scores, artwork, endorsements or licensing claims.

## Changes and non-changes

This proposal adds this audit report, machine-readable aggregate and an additive org-card section only. It does not change any existing card, model, dataset, workflow, publisher manifest, permission, key, gate, source pin or runtime. Six card-passport drafts and a static portable evidence-atlas design were prepared separately; application and publication are not claimed. The local offline renderer was written but its execution/tests were not confirmed after the Desktop Commander transport stopped returning device results.

The existing org-card owner is `.github#1008` at last checked head `2f19ad40a3a5ab27fba82bee03478c2969803e7a`. Its branch, existing integration, 32-file publisher manifest and eleven retained receipts are preserved. Its original November 1 inventory stays explicitly historical. The new section is an integration proposal, not a competing Hub writer.

Admission requires exact-head checks, signing requirements, independent review, protected merge and canonical publisher readback. Missing training evidence requires a real source record or a visible gap, not stronger graphics. Rollback: close this documentation-only proposal, or revert its added files after normal admission.
