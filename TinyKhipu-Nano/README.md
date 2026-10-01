---
license: apache-2.0
library_name: numpy
tags:
- governed-ai
- szl-holdings
- nano
- synthetic
- test-fixture
---

# TinyKhipu-Nano

**Reference/test fixture, not a production navigator or a foundation model.**

The receipted `tiny_khipu.npz` is a query-and-handle embedding model, **not a
4-6-2 MLP and not a model taking four numeric features**. Its actual arrays are:

| Tensor | Shape | Purpose |
|---|---|---|
| E | 24 x 12 | Token embeddings |
| W | 2 x 12 | ABSTAIN/NAVIGATE head |
| b | 2 | Class bias |
| Wc | 12 | Offered-handle citation scoring |

The standalone [load.py](./load.py) implements the canonical tokenizer,
mean embeddings, two-class softmax and offered-ID citation filter from
[szl-khipu](https://github.com/szl-holdings/szl-khipu/blob/7d7ead17e509d11dd9d715510e0bf8f0839e2930/szl_khipu/train/tiny_khipu.py).
Class order is **ABSTAIN=0, NAVIGATE=1**. The trained tokenizer uses ordered
substring matching; changing that behavior would change the model contract.

## Local use

Requires Python 3.11+ and NumPy 2.4.6. Obtain the public weights from the
[immutable Hub revision](https://huggingface.co/SZLHOLDINGS/TinyKhipu-Nano/tree/e67149f9c583cd159a4e95ee9e49746b57dd055d)
and verify the digest below. No Hugging Face credential is needed for these
public bytes. The GitHub loader and Hub weights are separate artifacts; this
card does not assert that a loader has been uploaded to the Hub.

```sh
python load.py '{"query":"ask about F18","handles":[{"id":"h.a","note":"F18 knot"}]}' --weights tiny_khipu.npz
```

`load()` returns the named tensors; `forward()` returns probabilities, the
numeric decision, citation scores and cited IDs; `infer()` returns the label.
Malformed inputs/tensors, non-finite values, unsupported layouts, and arithmetic
overflow raise `ValueError`. NPZ loading never enables pickle.

## Safety and evidence boundary

The model's raw output is **advisory, not authorization or a valid navigation
receipt**. It can predict NAVIGATE on unrelated or empty queries, including with
no offered handles. The loader preserves the trained computation; it does not
hide that limitation behind an invented confidence margin. Do not use this
fixture as the decision authority for navigation, policy, or production safety.

The reported training metrics below are preserved, not independently reproduced
by loader tests or local inference. Those tests establish executable-contract
behavior only. Not Qwen. Not 1.5B. The signed 1.5B abstain line is a separate
model and separate evidence; this fixture cannot improve that result by proxy.

## Bench (reported receipt)

`TRAINING_RECEIPT.json` seed `20260721` · steps 280 · honesty **REPORTED**

| Metric | Value |
|---|---|
| plan_valid | 1.00 |
| abstain | 1.00 |
| hallucinated | 0 |
| weights | `tiny_khipu.npz` sha256 `cc8d0385b2c75079669df809d7e4823f1ad8d9d535aec511446347490b11dff9` |

## Honesty

| Claim | Label |
|---|---|
| This card's numbers | SYNTHETIC |
| Energy / joules | UNAVAILABLE unless a signed meter says MEASURED |
| Λ uniqueness | Conjecture 1 OPEN — not a theorem |
| GGUF as the signed object | FALSE |

Apache-2.0. Copyright 2026 SZL Holdings.
