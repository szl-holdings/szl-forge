#!/usr/bin/env python3
"""Pure-stdlib reference of the SZL Lambda (Λ) aggregator for OAC System Health v2.

Ours (workstream E). Standard library only -- safe for the dependency-free v2 kernel.

Semantics mirrored (REPORTED from source, see math/lambda_reference.md):

* ``semantics="kernel"`` (DEFAULT) -- szl-holdings/szl-lambda-gate @ 99bc2cf1,
  torch-ext/szl_lambda_gate/_lambda.py::lambda_aggregate (the float64 path):
      Λ_w(x) = exp( Σ_i w_i · log(clamp(x_i, 0, 1)) ),  w normalized to Σ w_i = 1,
  uniform w_i = 1/k when ``weights is None``; any axis that is <= 0 after clamping
  OR non-finite (NaN, ±Inf) is a FAILING axis and forces Λ = 0 (fail closed);
  output clamped to [0, 1].  Errors mirror the kernel: k = 0 -> ValueError;
  weights must be length k, finite, strictly positive -> ValueError otherwise.

* ``semantics="python_ref"`` -- szl-lambda-gate tests/lambda_aggregator_source.py
  (itself a copy of the platform puriq_os reference).  Differences from "kernel":
  k = 0 returns 0.0; weights are only required to have a positive SUM (zero or
  negative individual weights are accepted); NaN inputs propagate to a NaN
  result and +Inf is clamped to 1 (i.e. it does NOT fail closed).  Kept only so
  the cross-check can reproduce that reference bit-for-bit; do not use in v2.

* ``semantics="lean"`` -- szl-holdings/lutar-lean @ c2de9d76, Lutar/Invariant.lean
  ``Λ k x = (∏ x_i)^(1/k)`` on ℝ≥0 with ``Λ 0 x = 0``: uniform weights only,
  no clamping to [0, 1], negative or non-finite inputs rejected.  Computed via
  logs for range safety (so it equals the exact value up to float rounding).

HONESTY: Λ is an advisory, non-compensatory roll-up (a weighted geometric mean).
Its uniqueness is Conjecture 1 (open, and false as stated under A1-A5); nothing
in this module is "proven trust".
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

__all__ = ["lambda_aggregate", "lambda_gate", "normalize_weights", "SEMANTICS"]

SEMANTICS = ("kernel", "python_ref", "lean")


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def normalize_weights(k: int, weights: Optional[Sequence[float]], *, strict: bool = True):
    """Return a list of k weights summing to 1 (kernel rules when ``strict``)."""
    if weights is None:
        return [1.0 / k] * k
    w = [float(x) for x in weights]
    if len(w) != k:
        raise ValueError(f"weights must have length {k} to match values; got {len(w)}")
    if strict:
        if not all(_is_finite(x) for x in w):
            raise ValueError("weights must all be finite (no NaN/Inf)")
        if any(x <= 0.0 for x in w):
            raise ValueError("weights must be strictly positive (w_i > 0)")
    sw = math.fsum(w) if strict else sum(w)
    if not sw > 0.0:
        raise ValueError("weights must sum to a positive value")
    return [x / sw for x in w]


def _kernel(values: Sequence[float], weights: Optional[Sequence[float]]) -> float:
    xs = [float(v) for v in values]
    k = len(xs)
    if k < 1:
        raise ValueError("values must contain at least one axis score (k >= 1)")
    w = normalize_weights(k, weights, strict=True)
    acc = 0.0
    bad = False
    for x, wi in zip(xs, w):
        if not _is_finite(x):
            bad = True
            continue
        xc = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        if xc <= 0.0:
            bad = True
            continue
        acc += math.log(xc) * wi
    if bad:
        return 0.0
    val = math.exp(acc)
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)


def _python_ref(values: Sequence[float], weights: Optional[Sequence[float]]) -> float:
    # Deliberately mirrors tests/lambda_aggregator_source.py line-for-line,
    # including its NaN propagation and permissive weight checks.
    n = len(values)
    if n == 0:
        return 0.0
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError("axes and weights length mismatch")
    sw = sum(weights)
    if sw <= 0:
        raise ValueError("weights must be positive and sum > 0")
    weights = [w / sw for w in weights]
    acc = 0.0
    for x, w in zip(values, weights):
        x = min(max(float(x), 0.0), 1.0)
        if x <= 0.0:
            return 0.0
        acc += w * math.log(x)
    val = math.exp(acc)
    return min(max(val, 0.0), 1.0)


def _lean(values: Sequence[float], weights: Optional[Sequence[float]]) -> float:
    if weights is not None:
        raise ValueError("lean semantics has uniform 1/k weights only (Lutar/Invariant.lean)")
    xs = [float(v) for v in values]
    k = len(xs)
    if k == 0:
        return 0.0  # `if hk : k = 0 then 0`
    if any((not _is_finite(x)) or x < 0.0 for x in xs):
        raise ValueError("lean semantics is defined on finite non-negative reals (NNReal)")
    if any(x == 0.0 for x in xs):
        return 0.0
    return math.exp(math.fsum(math.log(x) for x in xs) / k)


def lambda_aggregate(values: Sequence[float], weights: Optional[Sequence[float]] = None,
                     **params) -> float:
    """Λ over one vector of axis scores.

    params:
      semantics: "kernel" (default) | "python_ref" | "lean"  -- see module docstring.
    """
    semantics = params.pop("semantics", "kernel")
    if params:
        raise TypeError(f"unknown parameter(s): {sorted(params)}")
    if semantics == "kernel":
        return _kernel(values, weights)
    if semantics == "python_ref":
        return _python_ref(values, weights)
    if semantics == "lean":
        return _lean(values, weights)
    raise ValueError(f"semantics must be one of {SEMANTICS}; got {semantics!r}")


def lambda_gate(values: Sequence[float], weights: Optional[Sequence[float]] = None,
                threshold: float = 0.5, **params) -> Tuple[float, bool]:
    """ADVISORY gate: (Λ, Λ >= threshold).  Threshold domain [0, 1] enforced like the kernel."""
    t = float(threshold)
    if not _is_finite(t):
        raise ValueError(f"threshold must be a finite float, got {threshold!r}")
    if t < 0.0 or t > 1.0:
        raise ValueError(f"threshold must be within Λ's range [0, 1]; got {t!r}")
    score = lambda_aggregate(values, weights, **params)
    return score, bool(score >= t)


if __name__ == "__main__":  # tiny smoke demo, SYNTHETIC values
    demo = [0.9, 0.8, 0.95]
    print("kernel", lambda_aggregate(demo))
    print("gate", lambda_gate(demo, threshold=0.5))
