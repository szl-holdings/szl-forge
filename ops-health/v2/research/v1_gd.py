"""v1's exact training algorithm (ABL-optimizer, PREREGISTRATION section 9) and its reproduction.

Batch gradient descent on the mean log-loss with an L2 term on the weights only: lr 0.42,
2,400 epochs, l2 0.01, zero initialization.  Re-implemented from the v1 trainer
(szl-holdings/szl-forge commit 9a6fdd2c, {V1_GATEWAY_PACKAGE_DIR}/tools/
train_operational_health_model.py, function train_logistic_regression; Apache-2.0,
SZL Holdings).

``train_v1_gd`` performs the same floating-point operations in the same order as v1, vectorized
across rows:
  * z_i = intercept + sum(w_k * x_ik): builtin ``sum`` over the same 8 products, in the same
    order, as v1's generator expression (so the same summation algorithm on the same sequence;
    note that CPython 3.12 changed builtin float ``sum`` to compensated summation, which is why
    the unrounded weights differ in the last bits between 3.11 and 3.12);
  * the intercept gradient and each weight gradient are left-to-right sums starting at 0.0 in
    row order (``functools.reduce(operator.add, ..., 0.0)`` is v1's ``+=`` loop);
  * the parameter updates use v1's expressions verbatim.
``train_v1_gd_reference`` is a literal transcription of v1's loop, kept to prove the equality.

Usage (from the repository root; writes nothing):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.v1_gd --reproduce
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from functools import reduce
from operator import add, mul
from typing import Sequence

from . import generator as gen
from . import registry
from .dataio import labels_of, load_rows

EPOCHS = 2400
LEARNING_RATE = 0.42
L2_PENALTY = 0.01

# v1 as published (REPORTED): SZLHOLDINGS/oac-system-health-v1 model.json at Hub revision
# dd7d109813abcd90c5250106dc2eabd2804e7ac3, sha256 below; threshold 0.16.
V1_MODEL_SHA256 = "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557"
V1_TRAIN_SHA256 = "b868acce6a2e6a8f6a5e5edb3f5b45c4f2e7e7b33c1bc56d22128620aab8b1f6"
V1_DECISION_THRESHOLD = 0.16
V1_INTERCEPT = 2.966862310261
V1_WEIGHTS = {
    "listener_running": -1.61910172338,
    "tls_enabled": -0.262108043211,
    "peer_allowlist_configured": -0.527177146666,
    "queue_utilization": 0.908924629176,
    "consecutive_failures": 0.863860283722,
    "seconds_since_last_success": 0.053938998373,
    "ledger_integrity_ok": -1.128984575692,
    "configuration_valid": -1.461266877987,
}


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def train_v1_gd(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2_penalty: float = L2_PENALTY,
) -> tuple[float, list[float]]:
    x = [list(v) for v in x]
    y = [int(v) for v in y]
    d = len(x[0])
    columns = [list(col) for col in zip(*x, strict=True)]
    intercept = 0.0
    weights = [0.0] * d
    count = float(len(x))
    for _ in range(epochs):
        errors = [
            _sigmoid(intercept + sum(map(mul, weights, vector))) - target
            for vector, target in zip(x, y, strict=True)
        ]
        intercept_gradient = reduce(add, errors, 0.0)
        gradients = [reduce(add, map(mul, errors, col), 0.0) for col in columns]
        intercept -= learning_rate * intercept_gradient / count
        for index in range(d):
            regularized = gradients[index] / count + l2_penalty * weights[index]
            weights[index] -= learning_rate * regularized
    return intercept, weights


def train_v1_gd_reference(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2_penalty: float = L2_PENALTY,
) -> tuple[float, list[float]]:
    """Literal transcription of v1's train_logistic_regression loop (slow)."""
    intercept = 0.0
    weights = [0.0] * len(x[0])
    count = float(len(x))
    for _ in range(epochs):
        intercept_gradient = 0.0
        gradients = [0.0] * len(weights)
        for vector, target in zip(x, y, strict=True):
            probability = _sigmoid(
                intercept + sum(weight * value for weight, value in zip(weights, vector, strict=True))
            )
            error = probability - target
            intercept_gradient += error
            for index, value in enumerate(vector):
                gradients[index] += error * value
        intercept -= learning_rate * intercept_gradient / count
        for index in range(len(weights)):
            regularized = gradients[index] / count + l2_penalty * weights[index]
            weights[index] -= learning_rate * regularized
    return intercept, weights


def v1_committed_score(normalized: Sequence[float]) -> float:
    """v1-as-published score in the v1 kernel's operation order (research helper)."""
    linear = V1_INTERCEPT
    for name, value in zip(gen.FEATURE_NAMES, normalized, strict=True):
        linear += V1_WEIGHTS[name] * value
    return _sigmoid(linear)


def legacy_train_xy() -> tuple[list[list[float]], list[int]]:
    path = registry.LEGACY_DIR / "train.jsonl"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != V1_TRAIN_SHA256:
        raise RuntimeError("legacy v1 train.jsonl sha256 mismatch")
    rows = load_rows(path)
    return [gen.normalize(row["features"]) for row in rows], labels_of(rows)


def reproduce() -> dict:
    """Retrain v1's algorithm on v1's train split and compare with the published weights."""
    x, y = legacy_train_xy()
    started = time.perf_counter()
    intercept, weights = train_v1_gd(x, y)
    seconds = time.perf_counter() - started
    fitted = dict(zip(gen.FEATURE_NAMES, weights, strict=True))
    per_param = {"intercept": {"unrounded": intercept, "rounded_12": round(intercept, 12),
                               "published": V1_INTERCEPT,
                               "equal_at_12dp": round(intercept, 12) == V1_INTERCEPT}}
    for name in gen.FEATURE_NAMES:
        per_param[name] = {
            "unrounded": fitted[name],
            "rounded_12": round(fitted[name], 12),
            "published": V1_WEIGHTS[name],
            "equal_at_12dp": round(fitted[name], 12) == V1_WEIGHTS[name],
        }
    return {
        "python": sys.version.split()[0],
        "train_rows": len(y),
        "train_sha256": V1_TRAIN_SHA256,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2_penalty": L2_PENALTY,
        "seconds": round(seconds, 3),
        "all_equal_at_12dp": all(p["equal_at_12dp"] for p in per_param.values()),
        "max_abs_diff_unrounded_vs_published": max(
            abs(p["unrounded"] - p["published"]) for p in per_param.values()
        ),
        "parameters": per_param,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true", required=True)
    parser.parse_args(argv)
    report = reproduce()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_equal_at_12dp"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
