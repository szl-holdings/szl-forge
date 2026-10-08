"""Generator-truth parameters (PREREGISTRATION section 2) and recovery measures.

The truth is the label rule of ops-health-generator (identical to v1's), expressed in the
parameterization of a fitted model on v1-normalized features: intercept 9.8 and the weights
below.  Recovery measures are secondary metrics (section 8) and the g_truth_sign gate (section 10).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

FEATURE_NAMES = (
    "listener_running",
    "tls_enabled",
    "peer_allowlist_configured",
    "queue_utilization",
    "consecutive_failures",
    "seconds_since_last_success",
    "ledger_integrity_ok",
    "configuration_valid",
)
TRUE_INTERCEPT = 9.8
TRUE_WEIGHTS = {
    "listener_running": -3.10,
    "tls_enabled": -1.15,
    "peer_allowlist_configured": -1.35,
    "queue_utilization": 4.20,
    "consecutive_failures": 2.70,
    "seconds_since_last_success": 2.20,
    "ledger_integrity_ok": -4.40,
    "configuration_valid": -3.40,
}
TRUE_WEIGHT_VECTOR = tuple(TRUE_WEIGHTS[name] for name in FEATURE_NAMES)


def _vector(weights: Mapping[str, float] | Sequence[float]) -> list[float]:
    if isinstance(weights, Mapping):
        return [float(weights[name]) for name in FEATURE_NAMES]
    values = [float(v) for v in weights]
    if len(values) != len(FEATURE_NAMES):
        raise ValueError("expected 8 weights")
    return values


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def sign_agreement(weights: Mapping[str, float] | Sequence[float]) -> dict:
    """k/8 features whose fitted sign equals the true sign (a zero weight never agrees)."""
    values = _vector(weights)
    agree = [
        name
        for name, w, t in zip(FEATURE_NAMES, values, TRUE_WEIGHT_VECTOR, strict=True)
        if _sign(w) == _sign(t)
    ]
    return {
        "k": len(agree),
        "of": len(FEATURE_NAMES),
        "fraction": len(agree) / len(FEATURE_NAMES),
        "disagreeing": [name for name in FEATURE_NAMES if name not in agree],
    }


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks with ties given their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation (Pearson on average ranks); None if a rank vector is constant."""
    ra, rb = average_ranks(a), average_ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = math.fsum((x - ma) ** 2 for x in ra)
    vb = math.fsum((y - mb) ** 2 for y in rb)
    if va == 0.0 or vb == 0.0:
        return None
    return cov / math.sqrt(va * vb)


def relative_l2(intercept: float, weights: Mapping[str, float] | Sequence[float]) -> float:
    """||(b, w) - (b*, w*)||_2 / ||(b*, w*)||_2."""
    fitted = [float(intercept), *_vector(weights)]
    true = [TRUE_INTERCEPT, *TRUE_WEIGHT_VECTOR]
    num = math.sqrt(math.fsum((f - t) ** 2 for f, t in zip(fitted, true, strict=True)))
    den = math.sqrt(math.fsum(t * t for t in true))
    return num / den


def recovery_report(intercept: float, weights: Mapping[str, float] | Sequence[float]) -> dict:
    values = _vector(weights)
    signs = sign_agreement(values)
    return {
        "sign_agreement_k": signs["k"],
        "sign_agreement_of": signs["of"],
        "sign_disagreeing": signs["disagreeing"],
        "spearman_weights_vs_truth": spearman(values, TRUE_WEIGHT_VECTOR),
        "relative_l2_intercept_and_weights": relative_l2(intercept, values),
    }


def recovery_from_artifact(artifact: Mapping) -> dict:
    """Generator-truth recovery (section 8) of a v2 model artifact (or any dict with the
    ``intercept`` and ``weights`` fields of the v1/v2 model schemas)."""
    return recovery_report(float(artifact["intercept"]), artifact["weights"])


def sign_gate(weights: Mapping[str, float] | Sequence[float]) -> float:
    """g_truth_sign (section 10): sign agreement k/8 as a value in [0, 1]."""
    return sign_agreement(weights)["fraction"]
