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
- Admitted training rows: 31
- Trainable parameters: 10,822,656
- Total parameters observed: 863,808,576
- Peak reserved training memory: 2,204,106,752 bytes
- Aggregate adapter SHA-256:
  `286336b01858f598c62e47ee6b71902863aa739ddb42b1036f326b8867370ee4`
- Adapter weights SHA-256:
  `f839d83a982c9768cd519604e92bd2be81fcaaa89d4375727e2d81dbf3c70bbb`

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
`505daf3e8d9570b1dfba33bb832bd571d2c97b5b` and the durable qualification
source-bundle SHA-256
`ea4f5717edc1abbe16fe622fb8a9f34fc931784fae7fd9e9b3b81ceeef7e5998`
to:

- training receipt canonical SHA-256:
  `b158cff6d81401b991f20d9f1226747c62a4e75526f88d484cd6800f102fd67f`
- evaluation receipt canonical SHA-256:
  `fd172c2a6198658c576757232e60aaeac573386fe8e1ab4f73eb26cf814fb4aa`
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
