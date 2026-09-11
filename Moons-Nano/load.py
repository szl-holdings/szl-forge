"""Moons-Nano loader.

2->8->2 tanh-softmax MLP over classic two-moons. The archive's documented
arrays are W1[8,2] b1[8] W2[2,8] b2[2] — torch layout, load directly. The
docstring value is the contract; nothing clever is inferred.
"""
from __future__ import annotations

import pathlib

import numpy as np

_NAMES = ("W1", "b1", "W2", "b2")
_SHAPES = ((8, 2), (8,), (2, 8), (2,))


def load(path: str | pathlib.Path = "moons.npz"):
    z = np.load(path, allow_pickle=False)
    missing = [n for n in _NAMES if n not in z.files]
    if missing:
        raise ValueError(f"Moons-Nano: missing arrays {missing}; have {sorted(z.files)}")
    w1, b1, w2, b2 = (z[n].astype(np.float64) for n in _NAMES)
    if (w1.shape, b1.shape, w2.shape, b2.shape) != _SHAPES:
        raise ValueError(
            "Moons-Nano: shape drift "
            f"{(w1.shape, b1.shape, w2.shape, b2.shape)} != {_SHAPES}"
        )
    if not all(np.isfinite(v).all() for v in (w1, b1, w2, b2)):
        raise ValueError("Moons-Nano: non-finite entry in archive")
    return w1, b1, w2, b2


def infer(x, y, weights=None, path="moons.npz"):
    """(x, y) point -> moon index (0 or 1) with the softmax confidence."""
    w1, b1, w2, b2 = weights if weights is not None else load(path)
    h = np.tanh(w1 @ np.asarray([x, y], dtype=np.float64) + b1)
    logits = w2 @ h + b2
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = int(p.argmax())
    return top, float(p[top])


if __name__ == "__main__":
    import sys

    x, y = (0.0, 0.0) if len(sys.argv) < 3 else (float(sys.argv[1]), float(sys.argv[2]))
    print(infer(x, y))
