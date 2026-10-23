"""Representation tables and the common R scale (protocol s.2, s.6.2, s.11.2). SYNTHETIC.

A model with a declared (offset, scale) table computes z = b + sum_j w_j (x_j - o_j) / s_j.
The same function on the registered representation R (offsets o*_j, scales s*_j) has

    w*_j = w_j * s*_j / s_j
    b*   = b - sum_j (w_j / s_j) (o_j - o*_j)

which is how every fitted parameter is mapped to the R scale before any comparison (s.2: "every
truth comparison maps fitted weights to a common scale first"; s.11.2 g_coef_band: "using the
artifact's declared table, so a change of the declared table alone moves theta on this scale").
The mapping is exact algebra; it is evaluated in a fixed operation order (features in
FEATURE_NAMES order, b* accumulated with math.fsum).

Generator truth on the R scale (s.2): intercept 9.8; weights as v2's truth, except
seconds_since_last_success, whose truth weight 2.2 per 86400 s is 2.2 * 900 / 86400 =
0.0229166... per 900 s (MODELED arithmetic in the protocol; computed here).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

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

# Representation R (s.6.2): binary 0/1; queue (x - 0)/1; failures (x - 0)/20; seconds x / 900.
TABLE_R = {
    "listener_running": (0.0, 1.0),
    "tls_enabled": (0.0, 1.0),
    "peer_allowlist_configured": (0.0, 1.0),
    "queue_utilization": (0.0, 1.0),
    "consecutive_failures": (0.0, 20.0),
    "seconds_since_last_success": (0.0, 900.0),
    "ledger_integrity_ok": (0.0, 1.0),
    "configuration_valid": (0.0, 1.0),
}
# The v1 / v2 min-max representation (ABL-R and ABL-DR; s.10).
TABLE_V1_MINMAX = dict(TABLE_R, seconds_since_last_success=(0.0, 86400.0))

TRUE_INTERCEPT_V1 = 9.8
TRUE_WEIGHTS_V1 = {
    "listener_running": -3.10,
    "tls_enabled": -1.15,
    "peer_allowlist_configured": -1.35,
    "queue_utilization": 4.20,
    "consecutive_failures": 2.70,
    "seconds_since_last_success": 2.20,
    "ledger_integrity_ok": -4.40,
    "configuration_valid": -3.40,
}


class TableError(ValueError):
    """A declared table is malformed (research-side form check)."""


def table_block(table: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    """The artifact's JSON form of a table."""
    return {name: {"offset": float(table[name][0]), "scale": float(table[name][1])}
            for name in FEATURE_NAMES}


def parse_table(block: Any) -> dict[str, tuple[float, float]]:
    """Form check of a declared table: exactly the eight fields, finite offsets, finite positive
    scales. (The kernel additionally requires equality with TABLE_R, s.6.7 item 7.)"""
    if not isinstance(block, Mapping) or set(block) != set(FEATURE_NAMES):
        raise TableError("a declared table has exactly the eight fields")
    out = {}
    for name in FEATURE_NAMES:
        entry = block[name]
        if not isinstance(entry, Mapping) or set(entry) != {"offset", "scale"}:
            raise TableError(f"table entry {name} must be {{offset, scale}}")
        values = []
        for key in ("offset", "scale"):
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TableError(f"table {name}.{key} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise TableError(f"table {name}.{key} must be finite")
            values.append(value)
        if not values[1] > 0.0:
            raise TableError(f"table {name}.scale must be positive")
        out[name] = (values[0], values[1])
    return out


def to_r_scale(intercept: float, weights: Mapping[str, float],
               table: Mapping[str, Sequence[float]]) -> dict:
    """(b*, w*) on representation R from (b, w) on ``table``."""
    w_star = {}
    shift = []
    for name in FEATURE_NAMES:
        offset, scale = float(table[name][0]), float(table[name][1])
        r_offset, r_scale = TABLE_R[name]
        w = float(weights[name])
        w_star[name] = w * r_scale / scale
        shift.append((w / scale) * (offset - r_offset))
    return {"intercept": float(intercept) - math.fsum(shift), "weights": w_star}


def theta_vector(intercept: float, weights: Mapping[str, float]) -> list[float]:
    """theta_0..theta_8: the intercept, then the eight weights in FEATURE_NAMES order."""
    return [float(intercept)] + [float(weights[name]) for name in FEATURE_NAMES]


def artifact_theta_r(artifact: Mapping) -> list[float]:
    """theta on the R scale of an artifact, using its own declared table (form-checked)."""
    table = parse_table(artifact["transform_table"])
    mapped = to_r_scale(artifact["intercept"], artifact["weights"], table)
    return theta_vector(mapped["intercept"], mapped["weights"])


def truth_r() -> dict:
    """Generator truth (s.2) on the R scale: the v1-scale truth mapped through TABLE_V1_MINMAX."""
    return to_r_scale(TRUE_INTERCEPT_V1, TRUE_WEIGHTS_V1, TABLE_V1_MINMAX)


def sign_agreement(weights: Mapping[str, float]) -> dict:
    """k/8 features whose fitted sign equals the true sign (a zero weight never agrees). The
    scales are positive, so signs are the same on every representation."""

    def sign(v: float) -> int:
        return (v > 0) - (v < 0)

    agree = [n for n in FEATURE_NAMES if sign(float(weights[n])) == sign(TRUE_WEIGHTS_V1[n])]
    return {"k": len(agree), "of": len(FEATURE_NAMES), "fraction": len(agree) / len(FEATURE_NAMES),
            "disagreeing": [n for n in FEATURE_NAMES if n not in agree]}
