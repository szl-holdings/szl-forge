"""Mondrian (label-conditional) split conformal prediction (PREREGISTRATION section 6).

Nonconformity: s(x, y) = 1 - p_y(x), with p_1 = the model's (calibrated) score and
p_0 = 1 - p_1; s(x, 0) is evaluated exactly as p_1 (the kernel's ``nonconformity``).

Quantile: for each class y with n_y conformal rows, q_hat_y is the k_y-th smallest class-y score,
k_y = ceil((n_y + 1)(1 - alpha)), computed in exact rational arithmetic with alpha read as the
decimal it is written as (Fraction(str(alpha))).  If k_y > n_y the quantile is +infinity
(documented; happens only when n_y < (1 - alpha) / alpha, i.e. n_y < 9 at alpha = 0.10).

Prediction set C(x) = {y : s(x, y) <= q_hat_y}.  Advisory: ALERT iff C = {1}, NO_ALERT iff
C = {0}, otherwise ABSTAIN with reason BOTH or EMPTY.  Guarantee: class-conditional coverage
>= 1 - alpha under exchangeability with the conformal split (nominal data only); coverage under
shift is descriptive.

Frozen value of q_hat (what model.json stores and every downstream step uses):
  * +infinity is stored as 1.0: every nonconformity score is <= 1 (p_y >= 0), so membership is
    identical;
  * a finite q_hat is stored as the smallest 12-decimal value (as a double) that is >= q_hat:
    round(q_hat, 12) if that double is >= q_hat, else the 12-decimal ceiling (Decimal
    ROUND_CEILING, then float).  Float rounding is monotone, so every score <= q_hat is still
    <= the stored value: sets can only grow (by scores within 1e-12 above q_hat), and the
    coverage guarantee is preserved.
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, Decimal
from fractions import Fraction
from typing import Mapping, Sequence

from . import kernel_bridge

ALPHA = 0.10
METHOD = "mondrian_split_conformal"


class ConformalError(ValueError):
    pass


def quantile_rank(n: int, alpha: float) -> int:
    """k = ceil((n + 1)(1 - alpha)) in exact rational arithmetic."""
    a = Fraction(str(alpha))
    if not 0 < a < 1:
        raise ConformalError("alpha must be strictly between 0 and 1")
    return math.ceil((n + 1) * (1 - a))


def frozen_q_hat(q: float) -> float:
    """Smallest 12-decimal value (as a double) that is >= q; +inf -> 1.0."""
    if q == math.inf:
        return 1.0
    if not math.isfinite(q):
        raise ConformalError("q_hat must be finite or +inf")
    stored = round(q, 12)
    if stored < q:
        stored = float(Decimal(q).quantize(Decimal("1e-12"), rounding=ROUND_CEILING))
    if stored < q:  # impossible: float rounding is monotone
        raise ConformalError("frozen q_hat fell below the exact quantile")
    return min(max(stored, 0.0), 1.0)


def fit_mondrian(p1_scores: Sequence[float], labels: Sequence[int], alpha: float = ALPHA) -> dict:
    """Fit q_hat per class.  Returns the artifact block plus exact (unfrozen) diagnostics."""
    k = kernel_bridge.kernel()
    if len(p1_scores) != len(labels):
        raise ConformalError("scores and labels must be aligned")
    block = {"method": METHOD, "alpha": alpha, "q_hat": {}, "n_calibration": {}}
    diagnostics = {"rank": {}, "q_hat_exact": {}, "infinite": {}}
    for y in (0, 1):
        scores = sorted(k.nonconformity(float(p), y) for p, lab in zip(p1_scores, labels) if int(lab) == y)
        n_y = len(scores)
        if n_y == 0:
            raise ConformalError(f"no conformal rows of class {y}")
        rank = quantile_rank(n_y, alpha)
        q = scores[rank - 1] if rank <= n_y else math.inf
        block["q_hat"][str(y)] = frozen_q_hat(q)
        block["n_calibration"][str(y)] = n_y
        diagnostics["rank"][str(y)] = rank
        diagnostics["q_hat_exact"][str(y)] = q if math.isfinite(q) else "+inf"
        diagnostics["infinite"][str(y)] = not math.isfinite(q)
    return {"block": k.validate_conformal(block), "diagnostics": diagnostics}


def prediction_set(block: Mapping, p1: float) -> list[int]:
    return kernel_bridge.kernel().conformal_set(block, p1)


def advisory(prediction: Sequence[int]) -> tuple[str, str | None]:
    return kernel_bridge.kernel().advisory_from_set(prediction)
