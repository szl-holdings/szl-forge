"""SZL battery v2.2: Kotte impossibility/feasibility closed-form gate.

Lineage (owner directive 2026-11-19, leaders-brief U2 executed):
- Closed-form abstention feasibility test per Kotte: at certificate level
  alpha with realized population violator rate mu, certification is
  feasible iff mu <= alpha; when infeasible, the minimal abstention
  floor is A = (mu - alpha) / (1 - alpha) for an *idealized policy*
  that abstains on violators first. This is a necessary floor, not a
  sufficiency guarantee; random abstention does not imply risk reduction.
- Applied as a fail-closed first-class battery step: REFUSE
  certification when mu_hat > alpha and publish the abstention floor
  instead of laundering the number.
- Companion to szl_math_edge_admission_cp.py (clopper-pearson exact +
  SCRC calibrated band, szl-forge PR #766): this module answers the
  certification question those bounds feed.

Doctrine: no model execution, numerical theorem refinement, provider
access or physical measurements. Inputs are untrusted claims until
externally verified. production_authorized is always False. Numerical
outputs are advisory only, quantized at 1e-12 for cross-platform
receipts; the module is deterministic (no clock, no randomness)."""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING, localcontext

SCHEMA = "szl.edge-math-evidence/kotte-v1"
PINNED_COMPANIONS = {
    "cp_module_sha256": "a484b60a958feefb446f77419da52193a608a1322b5d7ffd92046ebf04efd80d",
    "v1_gate_sha256": "3aebebab39c5ab73a2a8aa11ec907ebc129c64a40d1816262d71af951ad908b0",
}
KOTTE_FORMULA = "abstain_floor=(mu-alpha)/(1-alpha);feasible_iff_mu<=alpha"


def _real(x):
    return type(x) in (int, float) and math.isfinite(x)


def _quant(x):
    return round(float(x), 12)


def _ceil_floor_12(alpha, mu):
    """Conservative 1e-12 grid for the *idealized* required abstention fraction."""
    with localcontext() as ctx:
        ctx.prec = 55
        a, m = Decimal(str(alpha)), Decimal(str(mu))
        raw = (m - a) / (Decimal(1) - a)
        return float(raw.quantize(Decimal("0.000000000001"), rounding=ROUND_CEILING))


def kotte_feasibility(alpha, mu):
    """Closed-form feasibility dict. Fail-closed on bad inputs."""
    if not _real(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be a real in (0, 1)")
    if not _real(mu) or not (0.0 <= float(mu) <= 1.0):
        raise ValueError("mu must be a real in [0, 1]")
    a, m = float(alpha), float(mu)
    feasible = m <= a
    floor = 0.0 if feasible else _ceil_floor_12(a, m)
    return {
        "schema": SCHEMA,
        "formula": KOTTE_FORMULA,
        "alpha": _quant(a),
        "mu": _quant(m),
        "feasible": feasible,
        "abstention_floor": floor,
        "verdict": "FEASIBLE" if feasible else "INFEASIBLE_PUBLISH_FLOOR",
        "production_authorized": False,
    }


def kotte_certify(alpha, mu):
    """Fail-closed certification decision from the feasibility test."""
    f = kotte_feasibility(alpha, mu)
    decision = "CERTIFY" if f["feasible"] else "REFUSE"
    reason = (
        "mu_hat<=alpha: feasible at certificate level" if f["feasible"]
        else "mu_hat>alpha: certification impossible; abstention floor published")
    return {
        "schema": SCHEMA,
        "decision": decision,
        "reason": reason,
        "feasibility": f,
        "production_authorized": False,
    }


def audit(entries):
    """Batch audit. entries: iterable of {alpha, mu, label} dicts."""
    rows = []
    for e in entries:
        if not isinstance(e, dict) or not {
                "alpha", "mu", "label"} <= set(e):
            raise ValueError("entry must be a dict with alpha, mu, label")
        r = kotte_certify(e["alpha"], e["mu"])
        rows.append({
            "label": e["label"],
            "alpha": r["feasibility"]["alpha"],
            "mu": r["feasibility"]["mu"],
            "decision": r["decision"],
            "abstention_floor": r["feasibility"]["abstention_floor"],
        })
    refused = sum(1 for r in rows if r["decision"] == "REFUSE")
    rec = {
        "schema": SCHEMA,
        "batch_size": len(rows),
        "refused": refused,
        "certified": len(rows) - refused,
        "fail_closed": True,
        "body": rows,
        "production_authorized": False,
    }
    rec["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in rec.items() if k != "sha256"},
                   sort_keys=True).encode("utf-8")).hexdigest()
    return rec


def audit_receipt_ok(rec):
    """Verify an audit receipt sha (fail-closed self-check helper)."""
    if not isinstance(rec, dict) or "sha256" not in rec:
        return False
    body = {k: v for k, v in rec.items() if k != "sha256"}
    expect = hashlib.sha256(json.dumps(body, sort_keys=True).encode(
        "utf-8")).hexdigest()
    return expect == rec["sha256"]
