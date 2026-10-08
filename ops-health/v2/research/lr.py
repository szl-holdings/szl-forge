"""L2-regularized binary logistic regression by Newton/IRLS (PREREGISTRATION section 6).

Objective (minimized):
    J(b, w) = mean_i[ softplus(z_i) - y_i z_i ] + (l2 / 2) * ||w||^2,   z_i = b + sum_j w_j x_ij,
with the intercept b unpenalized.  Standard library only.

Deterministic operation order
  * z_i is accumulated exactly like the v1 and v2 kernels: start from b, then add w_j * x_ij
    feature by feature in feature order;
  * every reduction over rows is math.fsum (correctly rounded, so independent of row order);
  * the (d+1)x(d+1) Newton system (9x9 for the eight features) is solved by Cholesky, with a
    fallback to Gaussian elimination with partial pivoting (ties to the lower row) when the
    matrix is not numerically positive definite.
Newton step with step halving: the candidate theta + t*delta (t = 1, 1/2, 1/4, ...) is accepted
when J(candidate) <= J(theta) + 1e-12 * max(1, |J(theta)|).  The 1e-12 allowance only absorbs
floating-point noise in J near the optimum (where a true Newton decrease is below the resolution
of J); any real overshoot is far larger and still triggers halving.  After 60 halvings without
acceptance the fit stops with stop_reason LINE_SEARCH_FAILED.
Convergence: the Newton step of the current iterate satisfies max|delta| < 1e-10 (the step is
then applied and stop_reason is CONVERGED), or the 100-iteration cap is hit (converged False,
stop_reason CAP_REACHED; this must be reported).  Initial point: b = 0, w = 0.
"""

from __future__ import annotations

import math
from itertools import repeat
from operator import add, mul, sub
from typing import Sequence

MAX_ITER = 100
TOL = 1e-10
MAX_HALVINGS = 60
ACCEPT_NOISE = 1e-12


def sigmoid(value: float) -> float:
    """Numerically stable logistic, identical to the v1 and v2 kernels."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def linear_predictor(
    intercept: float, weights: Sequence[float], columns: Sequence[Sequence[float]], n: int
) -> list[float]:
    z = [float(intercept)] * n
    for w, col in zip(weights, columns, strict=True):
        z = list(map(add, z, map(mul, repeat(w), col)))
    return z


def objective(z: Sequence[float], y: Sequence[int], weights: Sequence[float], l2: float) -> float:
    n = len(z)
    data = math.fsum(softplus(zi) - yi * zi for zi, yi in zip(z, y, strict=True)) / n
    return data + 0.5 * l2 * math.fsum(w * w for w in weights)


def gradient(
    z: Sequence[float], y: Sequence[int], weights: Sequence[float], columns, l2: float
) -> list[float]:
    """[dJ/db, dJ/dw_1..d] at the point with linear predictor z."""
    n = len(z)
    r = list(map(sub, map(sigmoid, z), y))
    return [math.fsum(r) / n] + [
        math.fsum(map(mul, r, col)) / n + l2 * w for col, w in zip(columns, weights, strict=True)
    ]


def hessian(z: Sequence[float], columns, l2: float) -> list[list[float]]:
    n = len(z)
    d = len(columns)
    s = [p * (1.0 - p) for p in map(sigmoid, z)]
    full = [[0.0] * (d + 1) for _ in range(d + 1)]
    full[0][0] = math.fsum(s) / n
    s_cols = [list(map(mul, s, col)) for col in columns]
    for j in range(d):
        full[0][j + 1] = full[j + 1][0] = math.fsum(s_cols[j]) / n
        for k in range(j, d):
            value = math.fsum(map(mul, s_cols[j], columns[k])) / n
            if j == k:
                value += l2
            full[j + 1][k + 1] = full[k + 1][j + 1] = value
    return full


def cholesky_solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float] | None:
    """Solve A x = rhs for symmetric positive definite A; None if A is not numerically PD."""
    size = len(rhs)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            s = matrix[i][j]
            for k in range(j):
                s -= lower[i][k] * lower[j][k]
            if i == j:
                if not s > 0.0:
                    return None
                lower[i][i] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]
    forward = [0.0] * size
    for i in range(size):
        s = rhs[i]
        for k in range(i):
            s -= lower[i][k] * forward[k]
        forward[i] = s / lower[i][i]
    out = [0.0] * size
    for i in reversed(range(size)):
        s = forward[i]
        for k in range(i + 1, size):
            s -= lower[k][i] * out[k]
        out[i] = s / lower[i][i]
    return out


def gauss_solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (largest |pivot|, ties to the lower row)."""
    size = len(rhs)
    a = [list(map(float, row)) + [float(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: (abs(a[r][col]), -r))
        if abs(a[pivot][col]) < 1e-300:
            raise ArithmeticError("singular Newton system")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, size):
            factor = a[r][col] / a[col][col]
            if factor != 0.0:
                for c in range(col, size + 1):
                    a[r][c] -= factor * a[col][c]
    out = [0.0] * size
    for i in reversed(range(size)):
        s = a[i][size]
        for k in range(i + 1, size):
            s -= a[i][k] * out[k]
        out[i] = s / a[i][i]
    return out


def solve(matrix, rhs) -> tuple[list[float], str]:
    result = cholesky_solve(matrix, rhs)
    if result is not None:
        return result, "cholesky"
    return gauss_solve(matrix, rhs), "gauss_partial_pivot"


def fit(
    columns: Sequence[Sequence[float]],
    y: Sequence[int],
    l2: float,
    *,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    max_halvings: int = MAX_HALVINGS,
) -> dict:
    """Fit by Newton/IRLS.  ``columns`` are the d feature columns (no intercept column)."""
    if isinstance(l2, bool) or not isinstance(l2, (int, float)) or not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be a finite non-negative number")
    l2 = float(l2)
    columns = [list(map(float, col)) for col in columns]
    y = [int(v) for v in y]
    if any(v not in (0, 1) for v in y):
        raise ValueError("labels must be 0/1")
    n = len(y)
    d = len(columns)
    if n == 0 or any(len(col) != n for col in columns):
        raise ValueError("columns and labels must be non-empty and aligned")
    intercept = 0.0
    weights = [0.0] * d
    z = linear_predictor(intercept, weights, columns, n)
    current = objective(z, y, weights, l2)
    iterations = 0
    converged = False
    stop_reason = "CAP_REACHED"
    halvings_total = 0
    solvers: set[str] = set()
    last_step = None
    while iterations < max_iter:
        iterations += 1
        grad = gradient(z, y, weights, columns, l2)
        delta, method = solve(hessian(z, columns, l2), [-g for g in grad])
        solvers.add(method)
        step = max(abs(v) for v in delta)
        last_step = step
        allowance = ACCEPT_NOISE * max(1.0, abs(current))
        t = 1.0
        halvings = 0
        accepted = False
        while True:
            cand_b = intercept + t * delta[0]
            cand_w = [w + t * dv for w, dv in zip(weights, delta[1:], strict=True)]
            cand_z = linear_predictor(cand_b, cand_w, columns, n)
            cand_obj = objective(cand_z, y, cand_w, l2)
            if cand_obj <= current + allowance or step < tol:
                accepted = True
                break
            if halvings >= max_halvings:
                break
            t *= 0.5
            halvings += 1
        halvings_total += halvings
        if not accepted:
            stop_reason = "LINE_SEARCH_FAILED"
            break
        intercept, weights, z, current = cand_b, cand_w, cand_z, cand_obj
        if step < tol:
            converged = True
            stop_reason = "CONVERGED"
            break
    grad = gradient(z, y, weights, columns, l2)
    return {
        "intercept": intercept,
        "weights": weights,
        "iterations": iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "objective": current,
        "gradient_max_abs": max(abs(g) for g in grad),
        "last_newton_step_max_abs": last_step,
        "step_halvings": halvings_total,
        "solvers": sorted(solvers),
        "l2": l2,
        "rows": n,
        "max_iter": max_iter,
        "tol": tol,
    }


def columns_from_vectors(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    if not vectors:
        return []
    return [list(col) for col in zip(*vectors, strict=True)]
