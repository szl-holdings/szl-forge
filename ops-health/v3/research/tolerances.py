"""Tolerance rule of protocol s.11.3 (frozen at registration). SYNTHETIC; standard library only.

    tol_T = max(floor_T, c * sqrt(2) * SD_T)              for theta_0..theta_8 (R scale) and BA_gates
    tol_k = max(2, ceil(c * sqrt(2) * SD_k))              for the threshold grid index k

with c = 4, SD the sample standard deviation (n - 1) over the 100 band retrains, floors 0 for the
parameters, 2 grid steps for the threshold and 0.005 for BA. Operation order (fixed): mean =
fsum(x) / n; SD = sqrt(fsum((x - mean)^2) / (n - 1)); then (c * sqrt(2)) * SD.

The vector is computed ONCE, at protocol step 5, from the 100 band retrains, committed with its
sha256 before any mutant exists. This module refuses DEV inputs (s.15.5) and any input that is
not exactly the 100 band retrains train_band_001..100 in order. In step 2 it is exercised on
hand-made numbers only.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

from . import guards, registry

C = 4.0
FLOOR_THETA = 0.0
FLOOR_K = 2
FLOOR_BA = 0.005
SCHEMA = "szl-oac/ops-health-tolerances/v3"
BAND_NAMES = tuple(f"train_band_{k:03d}" for k in range(1, registry.BAND_RETRAINS + 1))


class ToleranceRefused(ValueError):
    pass


def sample_sd(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ToleranceRefused("a sample SD needs at least two values")
    mean = math.fsum(values) / n
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))


def tolerance(sd: float, floor: float) -> float:
    return max(floor, (C * math.sqrt(2.0)) * sd)


def tolerance_k(sd_k: float) -> int:
    return max(FLOOR_K, math.ceil((C * math.sqrt(2.0)) * sd_k))


def compute(band: Sequence[Mapping]) -> dict:
    """``band``: one record per band retrain, in order: {"train_split", "theta_r" (9 floats),
    "threshold_index" (int), "ba_gates" (float), "converged" (bool)}."""
    names = [record.get("train_split") for record in band]
    guards.refuse_dev_statistic(names, "a gate tolerance")
    if tuple(names) != BAND_NAMES:
        raise ToleranceRefused("tolerances use exactly the 100 band retrains train_band_001..100, in order")
    for record in band:
        theta = record.get("theta_r")
        if not isinstance(theta, (list, tuple)) or len(theta) != 9:
            raise ToleranceRefused("each band record carries theta_0..theta_8 on the R scale")
        if not all(isinstance(v, float) and math.isfinite(v) for v in theta):
            raise ToleranceRefused("theta values must be finite floats")
        k = record.get("threshold_index")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 99:
            raise ToleranceRefused("threshold_index must be an integer grid index in 1..99")
        ba = record.get("ba_gates")
        if not isinstance(ba, float) or not math.isfinite(ba):
            raise ToleranceRefused("ba_gates must be a finite float")
    sd_theta = [sample_sd([float(r["theta_r"][j]) for r in band]) for j in range(9)]
    sd_k = sample_sd([float(r["threshold_index"]) for r in band])
    sd_ba = sample_sd([float(r["ba_gates"]) for r in band])
    vector = {
        "schema": SCHEMA,
        "synthetic": True,
        "rule": "tol_T = max(floor_T, c*sqrt(2)*SD_T); tol_k = max(2, ceil(c*sqrt(2)*SD_k)); c = 4; "
                "SD with n - 1 over the 100 band retrains (prereg s.11.3)",
        "c": C,
        "floors": {"theta": FLOOR_THETA, "k": FLOOR_K, "ba": FLOOR_BA},
        "band_retrains": list(BAND_NAMES),
        "band_converged": sum(1 for r in band if r.get("converged") is True),
        "sd": {"theta": sd_theta, "k": sd_k, "ba": sd_ba},
        "theta": [tolerance(sd, FLOOR_THETA) for sd in sd_theta],
        "k": tolerance_k(sd_k),
        "ba": tolerance(sd_ba, FLOOR_BA),
    }
    return vector


def canonical_bytes(vector: Mapping) -> bytes:
    """Canonical JSON (the freeze-manifest form of s.3.5): sorted keys, compact, ASCII, no NaN."""
    return json.dumps(vector, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def digest(vector: Mapping) -> str:
    return hashlib.sha256(canonical_bytes(vector)).hexdigest()
