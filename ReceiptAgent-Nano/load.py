# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Load the receipted 24-16-8-4 ReLU ReceiptAgent surrogate.

Contract: szl-holdings/szl-khipu@7d7ead17e509d11dd9d715510e0bf8f0839e2930,
szl_khipu/train/receipt_agent.py. Labels are ALLOW/WARN/BLOCKED/ESCALATE.
This MLP is advisory: rule_check remains authoritative. It does not grant
permission, generate a delivery receipt, or establish production readiness.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping

import numpy as np

GATES = ("ALLOW", "WARN", "BLOCKED", "ESCALATE")
_SHAPES = {"W1": (16, 24), "b1": (16,), "W2": (8, 16), "b2": (8,),
           "W3": (4, 8), "b3": (4,)}


def _weights(values):
    if not isinstance(values, Mapping) or set(values) != set(_SHAPES):
        raise ValueError("ReceiptAgent-Nano: expected exactly W1/b1/W2/b2/W3/b3")
    result = {}
    for name, shape in _SHAPES.items():
        value = np.asarray(values[name])
        if value.shape != shape or value.dtype.kind not in "fiu":
            raise ValueError(f"ReceiptAgent-Nano: invalid real tensor {name}; expected {shape}")
        value = value.astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"ReceiptAgent-Nano: non-finite tensor {name}")
        result[name] = value
    return result


def load(path: str | pathlib.Path = "receipt_agent.npz"):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(_SHAPES):
            raise ValueError("ReceiptAgent-Nano: invalid archive tensor count")
        return _weights({name: archive[name] for name in archive.files})


def forward(features, weights=None, path="receipt_agent.npz"):
    """A finite 24-D feature vector -> four canonical softmax probabilities."""
    x = np.asarray(features)
    if x.shape != (24,) or x.dtype.kind not in "fiu":
        raise ValueError("ReceiptAgent-Nano: expected a real 24-D feature vector")
    x = x.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("ReceiptAgent-Nano: non-finite features")
    w = load(path) if weights is None else _weights(weights)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            a1 = np.maximum(x @ w["W1"].T + w["b1"], 0.0)
            a2 = np.maximum(a1 @ w["W2"].T + w["b2"], 0.0)
            logits = a2 @ w["W3"].T + w["b3"]
            p = np.exp(logits - logits.max())
            return p / p.sum()
    except FloatingPointError as error:
        raise ValueError("ReceiptAgent-Nano: non-finite forward pass") from error


def infer(features, weights=None, path="receipt_agent.npz"):
    """Return the trained advisory class, without an invented confidence policy."""
    return GATES[int(forward(features, weights=weights, path=path).argmax())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", help="JSON array of 24 numeric features")
    parser.add_argument("--weights", default="receipt_agent.npz")
    args = parser.parse_args()
    print(infer(json.loads(args.features), path=args.weights))
