#!/usr/bin/env python3
"""Per-slice (Mondrian-style) missed-alert audit at a given threshold. RESEARCH PROTOTYPE, stdlib.

LTT certifies the MARGINAL missed-alert rate. A marginal guarantee can hide a slice with a high
miss rate. This reports, per slice with enough positives, the miss count and an exact one-sided
(1-delta) Clopper-Pearson upper bound; slices with too few positives are reported UNDERPOWERED,
never as OK. No multiplicity correction across slices is applied (descriptive audit, not a test).
The audit is underpowered on small slices and is not a production-readiness claim.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

def binom_cdf(k, n, p):
    if k >= n: return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))

def cp_upper(k, n, delta):
    """Exact one-sided upper (1-delta) bound on the miss probability."""
    if n == 0: return 1.0
    if k >= n: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > delta: lo = mid
        else: hi = mid
    return hi

def slices_for(features: dict, numeric_cuts: dict):
    out = {}
    for k, v in features.items():
        if k in numeric_cuts: out[k] = "high" if v >= numeric_cuts[k] else "low"
        else: out[k] = str(int(v)) if float(v) in (0.0, 1.0) else str(v)
    return out

def audit(rows, scores, lam, alpha=0.20, delta=0.10, min_pos=10):
    cuts = {}
    for k in rows[0]["features"]:
        vals = sorted(r["features"][k] for r in rows)
        if len(set(vals)) > 2: cuts[k] = vals[len(vals) // 2]  # median split
    groups = {}
    for r, s in zip(rows, scores):
        if not r["label"]["operator_attention_required"]: continue
        for k, lvl in slices_for(r["features"], cuts).items():
            g = groups.setdefault(f"{k}={lvl}", [0, 0]); g[0] += 1; g[1] += (s < lam)
    out = {}
    for name, (n, miss) in sorted(groups.items()):
        ub = cp_upper(miss, n, delta)
        status = "UNDERPOWERED" if n < min_pos else ("CANNOT_SHOW_BELOW_ALPHA" if ub > alpha else "UPPER_BOUND_OK")
        out[name] = {"positives": n, "misses": miss, "miss_rate": round(miss / n, 4), "cp_upper_bound": round(ub, 4), "status": status}
    return out

def load_split(data_dir, split):
    path = Path(data_dir) / f"{split}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"empty split: {path}")
    return rows

def score_rows(model_dir, rows):
    md = Path(model_dir).resolve()
    sys.path.insert(0, str(md))
    from oac_operational_health import OperationalHealthKernel
    K = OperationalHealthKernel(str(md / "model.json"), str(md / "artifact_receipt.json"))
    return [K.score(r["features"])["operator_attention_score"] for r in rows]

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--lambda", dest="lam", type=float, required=True); a = ap.parse_args(argv)
    rows = load_split(a.data_dir, a.split); sc = score_rows(a.model_dir, rows)
    print(json.dumps({"split": a.split, "lambda": a.lam, "slices": audit(rows, sc, a.lam)}, indent=1)); return 0

if __name__ == "__main__": raise SystemExit(main())
