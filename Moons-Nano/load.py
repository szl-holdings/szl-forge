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


def _weights(values):
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("Moons-Nano: expected four named tensors")
    result = []
    for value, name, shape in zip(values, _NAMES, _SHAPES):
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise ValueError(f"Moons-Nano: invalid real tensor {name}; expected {shape}")
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"Moons-Nano: non-finite tensor {name}")
        result.append(array)
    return tuple(result)


def load(path: str | pathlib.Path = "moons.npz"):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != set(_NAMES):
            raise ValueError("Moons-Nano: expected exactly W1/b1/W2/b2")
        return _weights(tuple(archive[name] for name in _NAMES))


def infer(x, y, weights=None, path="moons.npz"):
    """(x, y) point -> moon index (0 or 1) with the softmax confidence."""
    point = np.asarray([x, y])
    if point.shape != (2,) or point.dtype.kind not in "fiu":
        raise ValueError("Moons-Nano: expected two real scalar coordinates")
    point = point.astype(np.float64)
    if not np.isfinite(point).all():
        raise ValueError("Moons-Nano: non-finite coordinates")
    w1, b1, w2, b2 = _weights(weights) if weights is not None else load(path)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            h = np.tanh(w1 @ point + b1)
            logits = w2 @ h + b2
            p = np.exp(logits - logits.max())
            p /= p.sum()
    except FloatingPointError as error:
        raise ValueError("Moons-Nano: non-finite forward pass") from error
    top = int(p.argmax())
    return top, float(p[top])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, nargs="?", default=0.0)
    parser.add_argument("y", type=float, nargs="?", default=0.0)
    parser.add_argument("--weights", default="moons.npz")
    args = parser.parse_args()
    print(infer(args.x, args.y, path=args.weights))
