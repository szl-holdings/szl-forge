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

# ReceiptAgent-Nano

**Advisory reference/test fixture, not a production policy authority.**

The receipted `receipt_agent.npz` is a **24-16-8-4 ReLU MLP**, not a 4-10-4
network. Input is a finite numeric vector with exactly 24 features.

| Tensor | Shape |
|---|---|
| W1 / b1 | 16 x 24 / 16 |
| W2 / b2 | 8 x 16 / 8 |
| W3 / b3 | 4 x 8 / 4 |

[load.py](./load.py) implements the canonical computation and class order from
[szl-khipu](https://github.com/szl-holdings/szl-khipu/blob/7d7ead17e509d11dd9d715510e0bf8f0839e2930/szl_khipu/train/receipt_agent.py):
**ALLOW, WARN, BLOCKED, ESCALATE**. Two ReLU hidden layers feed a four-class
softmax. Prediction is the argmax, without an invented 0.5 confidence override
or a relabeling of WARN/BLOCKED to DENY/ABSTAIN.

## Local use

Requires Python 3.11+ and NumPy 2.4.6. Obtain public weights from the
[immutable Hub revision](https://huggingface.co/SZLHOLDINGS/ReceiptAgent-Nano/tree/525149f4b5132f1b9ca09e5dc35a21e4598e240e)
and verify the digest below. No Hugging Face credential is needed. The GitHub
loader and Hub weights are separate artifacts; this card does not assert that
a loader has been uploaded to the Hub.

```sh
python load.py '[1,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0]' --weights receipt_agent.npz
```

`load()` returns the six named tensors; `forward()` returns four probabilities;
`infer()` returns the advisory class. Missing/extra/transposed tensors, non-real
or non-finite values, invalid inputs and arithmetic overflow raise `ValueError`.
Caller-supplied weights undergo the same validation. Pickle remains disabled.

## Safety and evidence boundary

The canonical **rule_check kernel remains authoritative**. An MLP prediction
of ALLOW grants no authorization. The surrogate may disagree with the kernel;
it is not a signed receipt, a delivery witness, or the 1.5B ReceiptAgent model.

Reported metrics below are preserved, not independently reproduced by loader
contract tests or local inference. Those tests establish executable behavior,
not production fitness, policy correctness, or model quality.

## Bench (reported receipt)

`TRAINING_RECEIPT.json` seed `20260721` · honesty **REPORTED** · kernel is truth

| Metric | Value |
|---|---|
| held-out agree vs rule_check | 0.905 |
| weights | `receipt_agent.npz` sha256 `8aca4d24c90d6159cbb2bb885c7a94822d715899437f58abe69fb5c9664a1381` |

## Honesty

| Claim | Label |
|---|---|
| This card's numbers | SYNTHETIC |
| Energy / joules | UNAVAILABLE unless a signed meter says MEASURED |
| Λ uniqueness | Conjecture 1 OPEN — not a theorem |
| GGUF as the signed object | FALSE |

Apache-2.0. Copyright 2026 SZL Holdings.
