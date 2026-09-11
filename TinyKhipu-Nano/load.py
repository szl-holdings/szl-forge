"""TinyKhipu-Nano loader.

The repo's stated tensor names are the interface; we resolve them by signature.
4-6-2 MLP: (4->6 tanh) then (6->2 logits). Two conventions fit: a torch-layout
archive (W: out x in) loads directly; a numpy-layout archive (W: in x out) is
transposed. We detect (hidden x input == 6 x 4) and behave accordingly.

Profile: abstain- unless another class clears the plank.
"""
from __future__ import annotations

import pathlib

import numpy as np

CLASSES = ("NAVIGATE", "ABSTAIN")
_MARGIN = 0.25  # plank a NAVIGATE call must clear


def _tree(path: str | pathlib.Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if not all(np.isfinite(v).all() for v in d.values()):
        raise ValueError("TinyKhipu-Nano: non-finite entry in archive")
    return d


def load(path: str | pathlib.Path = "tiny_khipu.npz"):
    d = _tree(path)
    w1 = next(v for k, v in d.items() if v.ndim == 2 and (v.shape == (6, 4) or v.shape == (4, 6)))
    b1 = next(v for k, v in d.items() if v.ndim == 1 and v.shape == (6,))
    w2 = next(v for k, v in d.items() if v.ndim == 2 and (v.shape == (2, 6) or v.shape == (6, 2)))
    b2 = next(v for k, v in d.items() if v.ndim == 1 and v.shape == (2,))
    if w1.shape == (4, 6):
        w1 = w1.T
    if w2.shape == (6, 2):
        w2 = w2.T
    if not (w1.shape == (6, 4) and w2.shape == (2, 6)):
        raise ValueError(f"TinyKhipu-Nano: unrecognised layout {sorted((k, v.shape) for k, v in d.items())}")
    return w1, b1, w2, b2


def infer(x, weights=None, path="tiny_khipu.npz"):
    """4-D feature vector -> class label. ABSTAIN is the default, not a filter."""
    w1, b1, w2, b2 = weights if weights is not None else load(path)
    h = np.tanh(w1 @ np.asarray(x, dtype=np.float64) + b1)
    logits = w2 @ h + b2
    p = np.exp(logits - logits.max())
    p /= p.sum()
    if p[0] - p[1] > _MARGIN:
        return CLASSES[0]
    return CLASSES[1]


if __name__ == "__main__":
    import sys

    x = np.zeros(4) if len(sys.argv) < 2 else np.fromstring(sys.argv[1], sep=",")
    print(infer(x))
