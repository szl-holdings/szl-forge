"""szl_u1_distill_r3_gates.py - U1 distill-r3 doctrine gates (v1).

Implements the staged spec collaboration/u1-distill-r3-doctrine-20261119.json.
Zero-claims and zero-fire: pure stdlib, deterministic, no network, no secrets.
This module does NOT trigger any GPU job. It only computes the gate verdicts a
future distill-r3 round must satisfy, and the honest recommendation record.

Doctrine (from the spec, restated):
  1. Distill r3 MUST NOT touch the live r2/r3 analysis lanes; it stages only.
  2. PROMOTE requires zero realized violations on EVERY seed (variance gate).
  3. A lambda sitting at the grid floor is an explicit annotation, not a blame.
  4. Mixed policy record (SCRC-T leader + fixed-band v1 canary) is mandatory
     evidence: the canary band is what keeps the leader engine honest.
  5. Session gate is owner-gated: without an explicit owner GPU order the best
     possible verdict is ARMED_FIRE_PENDING_OWNER_GPU_ORDER, never FIRE.
"""
import hashlib
import json

SCHEMA = "szl.collab/distill-r3-gates-v1"
VARIANCE_PROMOTE = "PROMOTE"
VARIANCE_HOLD = "HOLD"
LAMBDA_FLOOR_DEFAULT = 0.02
GPU_RELEASE = "PENDING_OWNER_ORDER"


def variance_gate(per_seed, alpha=None, lambda_floor=LAMBDA_FLOOR_DEFAULT):
    """per_seed: [{seed, admitted, true_violations, lambda_hat}, ...].

    PROMOTE iff every seed has zero true violations and a feasible lambda_hat.
    A seed at the grid floor is flagged (annotation only).
    """
    if not isinstance(per_seed, list) or not per_seed:
        return {"verdict": VARIANCE_HOLD, "reason": "no_per_seed_data",
                "production_authorized": False}
    bad = []
    floor_hits = []
    for row in per_seed:
        if not isinstance(row, dict) or {"seed", "admitted", "true_violations", "lambda_hat"} - set(row):
            return {"verdict": VARIANCE_HOLD, "reason": "malformed_row", "row": row,
                    "production_authorized": False}
        if row["true_violations"] != 0:
            bad.append(row["seed"])
        lam = row["lambda_hat"]
        if lam is None:
            return {"verdict": VARIANCE_HOLD, "reason": "infeasible_lambda", "seed": row["seed"],
                    "production_authorized": False}
        if abs(float(lam) - lambda_floor) < 1e-12:
            floor_hits.append(row["seed"])
    if bad:
        return {"verdict": VARIANCE_HOLD, "reason": "realized_violations", "seeds": bad,
                "production_authorized": False}
    out = {"verdict": VARIANCE_PROMOTE, "realized_violations": 0, "seeds": len(per_seed),
           "lambda_at_floor_annotation": sorted(floor_hits) if floor_hits else None,
           "production_authorized": False}
    if alpha is not None:
        out["alpha"] = float(alpha)
    return out


def mixed_policy(leader_rows, canary_rows):
    """Record the mandatory mixed-policy evidence (SCRC-T leader + fixed canary).

    leader_rows/canary_rows: [{seed, admitted, true_violations}, ...].
    """
    def _agg(rows):
        return {"seeds": len(rows),
                "total_admitted": sum(r["admitted"] for r in rows),
                "true_violations": sum(r["true_violations"] for r in rows)}
    return {
        "schema": SCHEMA,
        "leader": {"policy": "scrc_t_exact_transfer", **_agg(leader_rows)},
        "canary": {"policy": "fixed_band_v1", **_agg(canary_rows)},
        "canary_role": "honesty_witness_for_leader_engine",
        "leader_zero_violations": _agg(leader_rows)["true_violations"] == 0,
        "production_authorized": False,
    }


def session_gate(cycle1_verified, cycle2_verified, variance_verdict, owner_gpu_order=False):
    """Owner-gated firing decision. Never fires: this module has no network."""
    staged = (cycle1_verified and cycle2_verified and variance_verdict == VARIANCE_PROMOTE)
    if staged and owner_gpu_order is True:
        verdict = "FIRE_AUTHORIZED"
    elif staged:
        verdict = "ARMED_FIRE_PENDING_OWNER_GPU_ORDER"
    else:
        verdict = "STAGED"
    return {
        "schema": SCHEMA,
        "cycle1_verified": bool(cycle1_verified),
        "cycle2_verified": bool(cycle2_verified),
        "variance_verdict": variance_verdict,
        "owner_gpu_order": bool(owner_gpu_order),
        "verdict": verdict,
        "gpu_release": GPU_RELEASE,
        "production_authorized": False,
        "note": "module computes verdicts only; no fire path exists in this code",
    }


def _sha(rec):
    return hashlib.sha256(json.dumps({k: v for k, v in rec.items() if k != "sha256"},
                                     sort_keys=True).encode("utf-8")).hexdigest()


def seal(rec):
    """Stamp a sha on any module record (spec/audit convention)."""
    out = dict(rec)
    out["production_authorized"] = False
    out["sha256"] = _sha(out)
    return out


def sealed_ok(rec):
    if not isinstance(rec, dict) or rec.get("sha256") is None:
        return False
    return _sha(rec) == rec["sha256"]
