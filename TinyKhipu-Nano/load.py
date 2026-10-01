# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Load the receipted TinyKhipu query/handle model, not a fictitious 4-D MLP.

Contract: szl-holdings/szl-khipu@7d7ead17e509d11dd9d715510e0bf8f0839e2930,
szl_khipu/train/tiny_khipu.py. Predictions are advisory fixture outputs, not
authorization or evidence of reproduced training. No pickle or remote code.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping

import numpy as np

CLASSES = ("ABSTAIN", "NAVIGATE")
FORMULA_TOKS = (
    "F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22",
    "LAMBDA", "YUYAY", "NAVIGATE", "ABSTAIN", "MEASURED", "REPORTED",
    "UNKNOWN", "BLOCKED",
)
_SHAPES = {"E": (24, 12), "W": (2, 12), "b": (2,), "Wc": (12,)}


def _weights(values):
    if not isinstance(values, Mapping) or set(values) != set(_SHAPES):
        raise ValueError("TinyKhipu-Nano: expected exactly E, W, b, Wc")
    result = {}
    for name, shape in _SHAPES.items():
        value = np.asarray(values[name])
        if value.shape != shape or value.dtype.kind not in "fiu":
            raise ValueError(f"TinyKhipu-Nano: invalid real tensor {name}; expected {shape}")
        value = value.astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"TinyKhipu-Nano: non-finite tensor {name}")
        result[name] = value
    return result


def load(path: str | pathlib.Path = "tiny_khipu.npz"):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(_SHAPES):
            raise ValueError("TinyKhipu-Nano: invalid archive tensor count")
        return _weights({name: archive[name] for name in archive.files})


def tok_id(token: str) -> int:
    # Preserve the trained tokenizer, including ordered substring matching.
    upper = token.upper()
    for index, formula in enumerate(FORMULA_TOKS):
        if formula in upper:
            return index
    value = 0
    for char in token:
        value = (value * 31 + ord(char)) & 0xFFFFFFFF
        if value >= 0x80000000:
            value -= 0x100000000
    return len(FORMULA_TOKS) + abs(value) % 8


def _embed(weights, text):
    tokens = text.split()
    total = np.zeros(12, dtype=np.float64)
    for token in tokens:
        total += weights["E"][tok_id(token)]
    return total / max(len(tokens), 1)


def forward(example, weights=None, path="tiny_khipu.npz"):
    """Return canonical probabilities, decision index, and offered citations.

    Input is {"query": str, "handles": [{"id": str, "note": str}, ...]}.
    A model prediction does not execute a navigation or grant authorization.
    """
    if not isinstance(example, Mapping) or not isinstance(example.get("query"), str):
        raise ValueError("TinyKhipu-Nano: expected a query/handles object, not 4-D features")
    handles = example.get("handles")
    if not isinstance(handles, list):
        raise ValueError("TinyKhipu-Nano: handles must be a list")
    offered = set()
    for handle in handles:
        if (not isinstance(handle, Mapping) or not isinstance(handle.get("id"), str)
                or not handle["id"] or not isinstance(handle.get("note"), str)
                or handle["id"] in offered):
            raise ValueError("TinyKhipu-Nano: handles require unique nonempty IDs and string notes")
        offered.add(handle["id"])
    w = load(path) if weights is None else _weights(weights)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            logits = w["b"] + w["W"] @ _embed(w, example["query"])
            p = np.exp(logits - logits.max())
            p /= p.sum()
            cites = np.array([float(w["Wc"] @ _embed(w, h["note"])) for h in handles])
    except FloatingPointError as error:
        raise ValueError("TinyKhipu-Nano: non-finite forward pass") from error
    decision = 1 if p[1] >= p[0] else 0
    cited = [handles[int(np.argmax(cites))]["id"]] if decision and handles else []
    return {"p": p, "cites": cites, "decision": decision,
            "cited": [cid for cid in cited if cid in offered], "hallucinated": 0}


def infer(example, weights=None, path="tiny_khipu.npz"):
    """Return the advisory label with the trained class order (ABSTAIN, NAVIGATE)."""
    return CLASSES[forward(example, weights=weights, path=path)["decision"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", help="JSON query/handles object")
    parser.add_argument("--weights", default="tiny_khipu.npz")
    args = parser.parse_args()
    print(infer(json.loads(args.example), path=args.weights))
