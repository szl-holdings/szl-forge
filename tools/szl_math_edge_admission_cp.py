"""SZL leader-adaptation admission math: exact Clopper-Pearson risk bounds and
SCRC-inspired calibrated band selection on the v1 edge admission gate.

Lineage (owner directive 2026-11-19: find the leaders, tweak their math with ours):
- Exact binomial upper bounds follow the conformal-router leader line
  (Uddin & Bauer, ACL 2026 SRW), which uses Clopper-Pearson exact intervals.
- Calibrated threshold selection follows Selective Conformal Risk Control
  (SCRC, arXiv:2512.12844), calibration-only transfer; the finite-sample
  statement is asserted for the pre-registered experimental regime only.
- Gate structure, HOLD codes, Pareto and Lambda advisory are unchanged from
  szl_math_edge_admission.py v1 (szl-forge PR #764 head 63c4967ad11e);

Doctrine: no model execution, numerical theorem refinement, provider access
or physical measurements. Input receipts are untrusted claims until externally
verified. production_authorized is always False. Numerical outputs are
advisory only and are quantized at 1e-12 for cross-platform receipts."""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "szl.edge-math-evidence/cp-v1"
SHA = set("0123456789abcdef")
PINNED_SOURCES = {
    "szl-formulas": "6e6c18ade01acec627eed04645e6e9c661916cb4",
    "szl-papers": "39c228dc3a29c6cfd831d43be6a8531e718692ac",
    "lutar-lean": "80e35f859801d80fce7cf81e4d9b0c8bdf91cfea",
}
V1_GATE_SHA256 = "3aebebab39c5ab73a2a8aa11ec907ebc129c64a40d1816262d71af951ad908b0"
BOUND_ENGINE = "clopper-pearson-bisection64-outward-quantized-1e-12"


def _sha(s, n):
    return type(s) is str and len(s) == n and all(c in SHA for c in s)


def _real(x):
    if type(x) not in (int, float):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:  # an int beyond float range is not a usable real
        return False


def _strict(x, keys):
    return type(x) is dict and set(x) == set(keys)


def _binom_logcdf(k, n, p):
    """Exact log P(Bin(n, p) <= k) via log-gamma and log-sum-exp."""
    if k < 0:
        return float("-inf")
    if k >= n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("-inf")
    lp = math.log1p(-p)
    logs = []
    for i in range(0, k + 1):
        logs.append(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * lp
        )
    m = max(logs)
    return m + math.log(math.fsum(math.exp(x - m) for x in logs))


def _upward_quantize_12(value):
    """Quantize an upper-bound approximation outward, not to nearest."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("upper-bound approximation outside [0, 1]")
    return min(1.0, math.ceil(math.nextafter(value, math.inf) * 1e12) / 1e12)


def cp_risk_upper(errors, n, delta):
    """One-sided exact upper bound b solving P(Bin(n, b) <= errors) = delta.

    Floating-point log-gamma bisection (64 iterations), then an upward
    1e-12 grid quantization. Not an arbitrary-precision proof certificate.
    Zero-error case uses a stable -expm1(log(delta) / n) expression.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if errors >= n:
        return 1.0
    if errors == 0:
        return _upward_quantize_12(-math.expm1(math.log(delta) / n))
    lo, hi = 0.0, 1.0
    target = math.log(delta)
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if _binom_logcdf(errors, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return _upward_quantize_12(hi)


def hoeffding_risk_upper(errors, n, delta):
    """v1 bound, kept for comparison receipts (identical to PR764 gate)."""
    return min(1.0, errors / n + math.sqrt(math.log(1.0 / delta) / (2 * n)))


def calibrate_lambda(certified_uppers, true_flags, alpha, grid):
    """SCRC-inspired calibration-only threshold (empirical transfer).

    loss_i(lam) = 1 if (certified_uppers[i] <= lam and true_flags[i]) else 0,
    monotone nondecreasing in lam; lam_hat = smallest grid value with
    (sum loss_i + 1) / (n + 1) <= alpha (exact finite-sample correction).
    Returns None when no grid value satisfies the budget: callers must HOLD.
    No theorem is claimed beyond the pre-registered regime.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not grid:
        raise ValueError("grid must be non-empty")
    n = len(certified_uppers)
    if n != len(true_flags) or n == 0:
        raise ValueError("calibration arrays must be non-empty and aligned")
    for lam in sorted(grid):
        s = sum(1 for c, f in zip(certified_uppers, true_flags) if c <= lam and f)
        if (s + 1) / (n + 1) <= alpha:
            return lam
    return None


def evaluate(data):
    """v1 admission gate with the risk bound engine replaced by exact CP."""
    def hold(code):
        return {
            "state": "HOLD",
            "reasons": [code],
            "production_authorized": False,
            "proof_refinement": False,
        }

    if (
        not _strict(
            data,
            ("schema", "source_pins", "scenario_set", "baseline", "candidate", "constraints", "governance"),
        )
        or data["schema"] != SCHEMA
    ):
        return hold("SCHEMA")
    if data["source_pins"] != PINNED_SOURCES:
        return hold("SOURCE_PIN_MISMATCH")
    scenarios = data["scenario_set"]
    if not _strict(
        scenarios,
        ("sha256", "heldout_count", "iid_declared", "pre_registered", "independent_evaluator"),
    ):
        return hold("SCENARIO_CONTRACT")
    n = scenarios["heldout_count"]
    if (
        not _sha(scenarios["sha256"], 64)
        or type(n) is not int
        or not 30 <= n <= 1_000_000
        or any(scenarios[k] is not True for k in ("iid_declared", "pre_registered", "independent_evaluator"))
    ):
        return hold("SCENARIO_EVIDENCE_UNVERIFIED")
    gov = data["governance"]
    if not _strict(
        gov,
        ("rights_approved", "source_verified", "weights_verified", "measured_on_target", "independent_receipt_verified", "zero_observed_safety_violations"),
    ):
        return hold("GOVERNANCE_CONTRACT")
    if any(gov[k] is not True for k in gov):
        return hold("GOVERNANCE_HOLD")
    bounds = data["constraints"]
    if not _strict(
        bounds,
        ("flash_max_bytes", "ram_max_bytes", "latency_p99_max_ms", "energy_max_j", "risk_upper_max", "delta"),
    ):
        return hold("CONSTRAINTS_CONTRACT")
    if any(not _real(bounds[k]) or bounds[k] <= 0 for k in bounds):
        return hold("CONSTRAINTS_INVALID")
    if not 0 < bounds["delta"] < 1 or bounds["risk_upper_max"] > 1:
        return hold("RISK_CONSTRAINTS_INVALID")
    if not bounds["delta"] / 2 > 0.0:  # subnormal delta: the split underflows
        return hold("RISK_CONSTRAINTS_INVALID")
    measurements = {}
    for label in ("baseline", "candidate"):
        obj = data[label]
        keys = ("model_sha256", "receipt_sha256", "dataset_sha256", "errors", "safety_violations", "flash_bytes", "ram_bytes", "latency_p99_ms", "energy_j")
        if not _strict(obj, keys):
            return hold("MEASUREMENT_CONTRACT_" + label.upper())
        if not all(_sha(obj[k], 64) for k in ("model_sha256", "receipt_sha256", "dataset_sha256")):
            return hold("UNPINNED_" + label.upper())
        if obj["dataset_sha256"] != scenarios["sha256"]:
            return hold("UNPAIRED_DATASET")
        if any(type(obj[k]) is not int or obj[k] < 0 for k in ("errors", "safety_violations", "flash_bytes", "ram_bytes")):
            return hold("INVALID_COUNTS_" + label.upper())
        if obj["errors"] > n or obj["safety_violations"] > n:
            return hold("COUNT_EXCEEDS_DENOMINATOR")
        if any(not _real(obj[k]) or obj[k] < 0 for k in ("latency_p99_ms", "energy_j")):
            return hold("INVALID_MEASUREMENT_" + label.upper())
        measurements[label] = obj
    baseline, candidate = measurements["baseline"], measurements["candidate"]
    if candidate["safety_violations"] != 0:
        return hold("OBSERVED_SAFETY_VIOLATION")
    try:
        risk = cp_risk_upper(candidate["errors"], n, bounds["delta"] / 2)
        safety = cp_risk_upper(candidate["safety_violations"], n, bounds["delta"] / 2)
    except (ValueError, OverflowError):
        return hold("RISK_BOUND_UNCOMPUTABLE")
    if risk > bounds["risk_upper_max"]:
        return hold("RISK_UPPER_BOUND_EXCEEDED")
    if (
        candidate["flash_bytes"] > bounds["flash_max_bytes"]
        or candidate["ram_bytes"] > bounds["ram_max_bytes"]
    ):
        return hold("MEMORY_BUDGET_EXCEEDED")
    if (
        candidate["latency_p99_ms"] > bounds["latency_p99_max_ms"]
        or candidate["energy_j"] > bounds["energy_max_j"]
    ):
        return hold("PERFORMANCE_BUDGET_EXCEEDED")
    metrics = ("errors", "flash_bytes", "ram_bytes", "latency_p99_ms", "energy_j")
    no_worse = all(candidate[k] <= baseline[k] for k in metrics)
    better = any(candidate[k] < baseline[k] for k in metrics)
    if not no_worse:
        return hold("NOT_PARETO_NONREGRESSIVE")
    if not better:
        return hold("NO_MEASURED_IMPROVEMENT")
    axes = [
        1.0 - candidate["errors"] / n,
        max(0.0, 1 - candidate["flash_bytes"] / bounds["flash_max_bytes"]),
        max(0.0, 1 - candidate["ram_bytes"] / bounds["ram_max_bytes"]),
        max(0.0, 1 - candidate["latency_p99_ms"] / bounds["latency_p99_max_ms"]),
        max(0.0, 1 - candidate["energy_j"] / bounds["energy_max_j"]),
    ]
    weights = [0.2] * 5
    lam = math.exp(math.fsum(w * math.log(x) for x, w in zip(axes, weights))) if all(x > 0 for x in axes) else 0.0
    return {
        "state": "STRUCTURALLY_ADMISSIBLE_UNVERIFIED",
        "reasons": [],
        "n": n,
        "candidate_empirical_risk": candidate["errors"] / n,
        "candidate_risk_upper_cp_exact": risk,
        "candidate_safety_rate_upper_cp_exact": safety,
        "bound_engine": BOUND_ENGINE,
        "pareto_dominates_baseline_on_supplied_metrics": True,
        "lambda_weighted_geomean_advisory": lam,
        "lambda_uniqueness_proven": False,
        "proof_refinement": False,
        "provider_receipts_independently_checked_by_this_module": False,
        "source_formula_ids_advisory": ["F1", "F4", "F7", "F19", "F22"],
        "production_authorized": False,
    }


def canonical_hash(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
