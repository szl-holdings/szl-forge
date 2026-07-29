---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - qwen3.5
  - peft
  - lora
  - unsloth
  - governed-ai
  - receipt-agent
---

# SZL ReceiptAgent Qwen3.5 0.8B v2

SZL ReceiptAgent Qwen3.5 0.8B v2 is a small, proposal-only adapter for drafting
structured governance receipts and refusing requests that would fabricate
evidence, approval, execution, or measured values.

## Intended use

Use this adapter behind a validating controller that:

1. validates every draft against the published JSON schema;
2. requires policy and human approval outside the model;
3. executes actions outside the weights; and
4. mints a cryptographic receipt only after approved execution.

The adapter is not an autonomous agent, authorizer, executor, factual oracle,
or substitute for source retrieval.

## Exact lineage

- Canonical base: `Qwen/Qwen3.5-0.8B`
- Base revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Training implementation: `unsloth/Qwen3.5-0.8B`
- Implementation revision: `23c69c53358a07516b5827588b3fdb12ae78fd65`
- Runtime: Unsloth `FastVisionModel`
- License: Apache-2.0

## Training

- Hardware: NVIDIA GeForce RTX 5050 Laptop GPU
- Optimizer steps: 64
- Admitted training rows: 37
- Trainable parameters: 10,822,656
- Total parameters observed: 863,808,576
- Final aggregate training loss: 0.9142942871840205
- Peak reserved training memory: 1,671,430,144 bytes
- Training report SHA-256:
  `1d1ce062e76aeccabe75fabb2c74d8bdddfc6f9a86f13c5cfd9960e9e5821f38`
- Aggregate adapter SHA-256:
  `dde649ca3166881b675a3db093ee273c6186e3d8e801c8491fac0a2da03e58f7`
- Adapter weights SHA-256:
  `885fc29fcb4cf55c280dc085fdb0a40f40d6b946fee400dd5e4ed3459fe6334f`

Only the repository-owned ReceiptAgent curriculum was admitted. The A11oy
Brain corpus was excluded from gradients because row-level rights and
provenance admission have not passed.

## Held-out acceptance

The exact saved adapter was reloaded on the same GPU and evaluated against
committed, digest-pinned held-out files:

| Gate | Result |
|---|---:|
| JSON contract-valid drafts | 5 / 5 |
| Adversarial refusals | 6 / 6 |

These are raw **MEASURED** acceptance counts for a small preregistered gate.
They are not a broad benchmark, do not establish factual accuracy, and do not
make the adapter autonomy-eligible.

Evaluation report SHA-256:
`0852fe55716da7b5fddf2340a00dd632c34d551096c685bd84923eb164f2a420`.

## Evidence boundary

The release includes:

- an owner-signed training receipt;
- an owner-signed evaluation receipt chained to the training receipt;
- the public Ed25519 verification key;
- source, dataset, adapter, and report digests; and
- a post-publication readback receipt after the Hub bytes are independently
  fetched and hashed.

Publication is fail-closed until all of those artifacts verify.

The owner-signed evidence chain binds source commit
`2a5f9cd98503923a49c34e4ef3d3f92e40ca1386` and the durable qualification
source-bundle SHA-256
`b7b0eb703062981636d7cca41ed9ecaa8ca87a0ba51bd515cca721a6df0f5f8a`
to:

- training receipt canonical SHA-256:
  `8c5e74892b8a9933ffea074f92b7081f360634a1e8397a249f07a3efd06ca433`
- evaluation receipt canonical SHA-256:
  `f3b79e4b3aed7359c56ad33b72c7c4f6a4a517166d097a220fa8f3d3fa5402ff`
- Ed25519 key ID: `e7f01810aaa97394`

The evidence chain is valid. Public release eligibility remains false until
the protected source change merges and the Hugging Face adapter is fetched
back and independently rehashed.

## Limitations

- Narrow synthetic curriculum.
- Small held-out set.
- Proposal-only behavior.
- No independent third-party evaluation.
- No ground-truth retrieval or autonomous execution.
- English-dominant evaluation.
- The validating controller remains mandatory.
