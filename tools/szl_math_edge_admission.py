"""SZL source-pinned edge model math: empirical risk, Pareto and Lambda advisory.

No model execution, numerical theorem refinement, provider access or physical
measurements are performed. Input receipts are untrusted claims until externally
verified. The Lambda calculation is a numerical implementation of a *definition*,
not proof of uniqueness or correctness of the implementation.
"""

from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "szl.edge-math-evidence/v1"
SHA = set("0123456789abcdef")
PINNED_SOURCES = {
    "szl-formulas": "6e6c18ade01acec627eed04645e6e9c661916cb4",
    "szl-papers": "39c228dc3a29c6cfd831d43be6a8531e718692ac",
    "lutar-lean": "80e35f859801d80fce7cf81e4d9b0c8bdf91cfea",
}
LOCKED_FORMULA_REFERENCE_BLOB = "61eca5b6d531ad758ff557b881ff49ec2b9af3ea"


def _sha(s, n):
    return type(s) is str and len(s) == n and all(c in SHA for c in s)


def _real(x):
    try:
        return type(x) in (int, float) and math.isfinite(x)
    except OverflowError:
        # An int beyond float range (a valid JSON integer) is not a usable value.
        return False


def _strict(x, keys):
    return type(x) is dict and set(x) == set(keys)


def _risk_upper(errors, n, delta, split):
    # One-sided Hoeffding for a pre-specified fixed model on IID Bernoulli
    # samples; no adaptive selection, independence or distribution is proved.
    # delta is shared by `split` bounds. log(split / delta) is taken as a
    # difference of logs: delta / split underflows to 0.0 at delta = 5e-324 and
    # its reciprocal overflows for every delta below about 1.1e-308.
    log_term = math.log(split) - math.log(delta)
    return min(1.0, errors / n + math.sqrt(log_term / (2 * n)))


def _lambda_advisory(axes, weights):
    # Canonical SZL weighted geometric mean definition, NOT uniqueness proof.
    if any(x == 0 for x in axes):
        return 0.0
    return math.exp(math.fsum(w * math.log(x) for x, w in zip(axes, weights)))


def evaluate(data):
    """Require hard empirical constraints before advisory Pareto/Lambda ranking."""

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
            (
                "schema",
                "source_pins",
                "scenario_set",
                "baseline",
                "candidate",
                "constraints",
                "governance",
            ),
        )
        or data["schema"] != SCHEMA
    ):
        return hold("SCHEMA")
    if data["source_pins"] != PINNED_SOURCES:
        return hold("SOURCE_PIN_MISMATCH")
    scenarios = data["scenario_set"]
    if not _strict(
        scenarios,
        (
            "sha256",
            "heldout_count",
            "iid_declared",
            "pre_registered",
            "independent_evaluator",
        ),
    ):
        return hold("SCENARIO_CONTRACT")
    n = scenarios["heldout_count"]
    if (
        not _sha(scenarios["sha256"], 64)
        or type(n) is not int
        or not 30 <= n <= 1_000_000
        or any(
            scenarios[k] is not True
            for k in ("iid_declared", "pre_registered", "independent_evaluator")
        )
    ):
        return hold("SCENARIO_EVIDENCE_UNVERIFIED")
    gov = data["governance"]
    if not _strict(
        gov,
        (
            "rights_approved",
            "source_verified",
            "weights_verified",
            "measured_on_target",
            "independent_receipt_verified",
            "zero_observed_safety_violations",
        ),
    ):
        return hold("GOVERNANCE_CONTRACT")
    if any(gov[k] is not True for k in gov):
        return hold("GOVERNANCE_HOLD")
    bounds = data["constraints"]
    if not _strict(
        bounds,
        (
            "flash_max_bytes",
            "ram_max_bytes",
            "latency_p99_max_ms",
            "energy_max_j",
            "risk_upper_max",
            "delta",
        ),
    ):
        return hold("CONSTRAINTS_CONTRACT")
    if any(not _real(bounds[k]) or bounds[k] <= 0 for k in bounds):
        return hold("CONSTRAINTS_INVALID")
    if not 0 < bounds["delta"] < 1 or bounds["risk_upper_max"] > 1:
        return hold("RISK_CONSTRAINTS_INVALID")
    measurements = {}
    for label in ("baseline", "candidate"):
        obj = data[label]
        keys = (
            "model_sha256",
            "receipt_sha256",
            "dataset_sha256",
            "errors",
            "safety_violations",
            "flash_bytes",
            "ram_bytes",
            "latency_p99_ms",
            "energy_j",
        )
        if not _strict(obj, keys):
            return hold("MEASUREMENT_CONTRACT_" + label.upper())
        if not all(
            _sha(obj[k], 64)
            for k in ("model_sha256", "receipt_sha256", "dataset_sha256")
        ):
            return hold("UNPINNED_" + label.upper())
        if obj["dataset_sha256"] != scenarios["sha256"]:
            return hold("UNPAIRED_DATASET")
        if any(
            type(obj[k]) is not int or obj[k] < 0
            for k in ("errors", "safety_violations", "flash_bytes", "ram_bytes")
        ):
            return hold("INVALID_COUNTS_" + label.upper())
        if obj["errors"] > n or obj["safety_violations"] > n:
            return hold("COUNT_EXCEEDS_DENOMINATOR")
        if any(not _real(obj[k]) or obj[k] < 0 for k in ("latency_p99_ms", "energy_j")):
            return hold("INVALID_MEASUREMENT_" + label.upper())
        measurements[label] = obj
    baseline, candidate = measurements["baseline"], measurements["candidate"]
    if candidate["safety_violations"] != 0:
        return hold("OBSERVED_SAFETY_VIOLATION")
    # delta is split across two predeclared candidate risk upper bounds;
    # no PAC-Bayes/KL assumption is inferred from an accuracy count.
    risk = _risk_upper(candidate["errors"], n, bounds["delta"], 2)
    safety = _risk_upper(candidate["safety_violations"], n, bounds["delta"], 2)
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
    # Non-dominance is distinct from weighted scalar ranking.
    metrics = ("errors", "flash_bytes", "ram_bytes", "latency_p99_ms", "energy_j")
    no_worse = all(candidate[k] <= baseline[k] for k in metrics)
    better = any(candidate[k] < baseline[k] for k in metrics)
    if not no_worse:
        return hold("NOT_PARETO_NONREGRESSIVE")
    if not better:
        return hold("NO_MEASURED_IMPROVEMENT")
    # For readability only: no score can override a failed hard constraint.
    axes = [
        1.0 - candidate["errors"] / n,
        max(0.0, 1 - candidate["flash_bytes"] / bounds["flash_max_bytes"]),
        max(0.0, 1 - candidate["ram_bytes"] / bounds["ram_max_bytes"]),
        max(0.0, 1 - candidate["latency_p99_ms"] / bounds["latency_p99_max_ms"]),
        max(0.0, 1 - candidate["energy_j"] / bounds["energy_max_j"]),
    ]
    weights = [0.2] * 5
    return {
        "state": "STRUCTURALLY_ADMISSIBLE_UNVERIFIED",
        "reasons": [],
        "n": n,
        "candidate_empirical_risk": candidate["errors"] / n,
        "candidate_risk_upper_hoeffding_conditional": risk,
        "candidate_safety_rate_upper_hoeffding_conditional": safety,
        "pareto_dominates_baseline_on_supplied_metrics": True,
        "lambda_weighted_geomean_advisory": _lambda_advisory(axes, weights),
        "lambda_uniqueness_proven": False,
        "proof_refinement": False,
        "provider_receipts_independently_checked_by_this_module": False,
        "source_formula_ids_advisory": ["F1", "F4", "F7", "F19", "F22"],
        "production_authorized": False,
    }


def canonical_hash(record):
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
