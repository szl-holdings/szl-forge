"""ReceiptAgent-Nano loader.

4-10-4 MLP: (4->10 tanh) then (10->4 gate logits). Every output is one of the
four named gates. ESCALATE is a class, not a retry loop; a low-confidence read
hands the decision to a human with a receipt rather than silently succeeding.
Layout resolved by shape, like the other nano loaders.
"""
from __future__ import annotations

import pathlib

import numpy as np

GATES = ("ALLOW", "DENY", "ABSTAIN", "ESCALATE")
_CONFIDENCE = 0.5  # below this, the read escalates: a human decides


def _tree(path: str | pathlib.Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if not all(np.isfinite(v).all() for v in d.values()):
        raise ValueError("ReceiptAgent-Nano: non-finite entry in archive")
    return d


def load(path: str | pathlib.Path = "receipt_agent.npz"):
    d = _tree(path)
    w1 = next(v for k, v in d.items() if v.ndim == 2 and set(v.shape) == {4, 10})
    b1 = next(v for k, v in d.items() if v.ndim == 1 and v.shape == (10,))
    w2 = next(v for k, v in d.items() if v.ndim == 2 and set(v.shape) == {10, 4})
    b2 = next(v for k, v in d.items() if v.ndim == 1 and v.shape == (4,))
    if id(w1) == id(w2):
        raise ValueError("ReceiptAgent-Nano: W1/W2 collision")
    if w1.shape[0] == 4:  # in x out -> transpose to torch layout
        w1 = w1.T
    if w2.shape[0] == 10:
        w2 = w2.T
    if not (w1.shape == (10, 4) and w2.shape == (4, 10)):
        raise ValueError(f"ReceiptAgent-Nano: unrecognised layout {sorted((k, v.shape) for k, v in d.items())}")
    return w1, b1, w2, b2


def infer(x, weights=None, path="receipt_agent.npz"):
    """4-D feature vector -> gate label. Low confidence escalates, never guesses."""
    w1, b1, w2, b2 = weights if weights is not None else load(path)
    h = np.tanh(w1 @ np.asarray(x, dtype=np.float64) + b1)
    logits = w2 @ h + b2
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = int(p.argmax())
    if GATES[top] != "ESCALATE" and p[top] < _CONFIDENCE:
        return "ESCALATE"
    return GATES[top]


if __name__ == "__main__":
    import sys

    x = np.zeros(4) if len(sys.argv) < 2 else np.fromstring(sys.argv[1], sep=",")
    print(infer(x))
