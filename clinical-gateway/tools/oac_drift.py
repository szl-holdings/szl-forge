#!/usr/bin/env python3
"""Per-feature drift monitor prototype (PSI + KS), stdlib only. RESEARCH PROTOTYPE.

Motivation: the receipt-bound gate tests drift on the *score* only. Monitoring tools
(NannyML, Evidently) report per-feature univariate drift. This adds that view; it does
not change any verdict in oac_rbcg.py. PSI cut-offs 0.10/0.25 are the common
credit-scoring convention, NOT calibrated on this data (see --null-reps for an
empirical false-flag rate on resampled calibration data). Watch-only: a DRIFT/WATCH
status is an operator signal, not a gate refusal.

The two-sample KS statistic and asymptotic p-value are implemented here (stdlib).
main's oac_rbcg exposes ks() (D only) plus a permutation test, not ks_2samp.
"""
from __future__ import annotations
import argparse, bisect, json, math, random
from pathlib import Path

EPS = 1e-4

def ks_2samp(a, b):
    """Two-sample KS D and a standard asymptotic p-value (Marsaglia-style series)."""
    a, b = sorted(a), sorted(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("empty sample")
    d = 0.0
    for x in sorted(set(a) | set(b)):
        fa = bisect.bisect_right(a, x) / n
        fb = bisect.bisect_right(b, x) / m
        d = max(d, abs(fa - fb))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return d, 1.0
    s = 0.0
    for j in range(1, 101):
        s += 2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
    return d, max(0.0, min(1.0, s))

def read_features(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows: raise ValueError("empty file")
    return [r["features"] for r in rows]

def _edges(vals, k=10):
    s = sorted(vals); e = sorted({s[min(len(s) - 1, int(i * len(s) / k))] for i in range(1, k)})
    return e

def _hist(vals, edges):
    c = [0] * (len(edges) + 1)
    for v in vals: c[bisect.bisect_right(edges, v)] += 1
    return c

def psi(cal, dep):
    """PSI with frozen bins from `cal` (categorical when <=2 distinct values)."""
    if not cal or not dep: raise ValueError("empty sample")
    cats = sorted(set(cal))
    if len(cats) <= 2:
        a = [sum(1 for v in cal if v == c) for c in cats]; b = [sum(1 for v in dep if v == c) for c in cats]
        extra = sum(1 for v in dep if v not in cats)
    else:
        e = _edges(cal); a = _hist(cal, e); b = _hist(dep, e); extra = 0
    pa = [max(x / len(cal), EPS) for x in a]; pb = [max(x / len(dep), EPS) for x in b]
    val = sum((q - p) * math.log(q / p) for p, q in zip(pa, pb))
    if extra: val += (extra / len(dep)) * math.log((extra / len(dep)) / EPS)
    return val

def classify(p_psi, p_bonf):
    if p_psi >= 0.25 or p_bonf < 0.01: return "DRIFT"
    if p_psi >= 0.10: return "WATCH"
    return "OK"

def report(cal_f, dep_f):
    names = sorted(cal_f[0]); m = len(names); out = {}
    for n in names:
        a = [float(r[n]) for r in cal_f]; b = [float(r[n]) for r in dep_f]
        d, p = ks_2samp(a, b); pb = min(1.0, p * m); ps = psi(a, b)
        out[n] = {"psi": round(ps, 4), "ks_d": round(d, 4), "ks_p_bonferroni": round(pb, 6), "status": classify(ps, pb)}
    worst = "DRIFT" if any(v["status"] == "DRIFT" for v in out.values()) else ("WATCH" if any(v["status"] == "WATCH" for v in out.values()) else "OK")
    return {"features": out, "overall": worst, "n_cal": len(cal_f), "n_dep": len(dep_f)}

def null_flag_rate(cal_f, n_dep, reps, seed=20261119):
    """Share of resampled same-distribution deployment sets that get flagged (empirical false-flag rate)."""
    rng = random.Random(seed); flags = {"DRIFT": 0, "WATCH": 0, "OK": 0}
    for _ in range(reps):
        dep = [cal_f[rng.randrange(len(cal_f))] for _ in range(n_dep)]
        flags[report(cal_f, dep)["overall"]] += 1
    return {k: v / reps for k, v in flags.items()} | {"reps": reps}

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--calibration", required=True); ap.add_argument("--deployment", required=True)
    ap.add_argument("--null-reps", type=int, default=0); a = ap.parse_args(argv)
    cal = read_features(a.calibration); dep = read_features(a.deployment); r = report(cal, dep)
    if a.null_reps: r["null_false_flag_rate"] = null_flag_rate(cal, len(dep), a.null_reps)
    print(json.dumps(r, indent=1, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
