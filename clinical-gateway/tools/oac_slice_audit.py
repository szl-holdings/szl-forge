#!/usr/bin/env python3
"""Per-slice (Mondrian-style) missed-alert audit at a given threshold. RESEARCH PROTOTYPE, stdlib.

LTT certifies the MARGINAL missed-alert rate. A marginal guarantee can hide a slice with a high
miss rate. This reports, per slice with enough positives, the miss count and an exact one-sided
(1-delta) Clopper-Pearson upper bound; slices with too few positives are reported UNDERPOWERED,
never as OK. No multiplicity correction across slices is applied (descriptive audit, not a test).
Numeric slice cut points (medians) are computed from the CALIBRATION split (--cuts-split, default validation), never from the audited
split, so slice membership does not depend on the data being audited. A row is "high" when its value is strictly above the cut, so a
zero-inflated feature (median 0) still yields a "low" (== 0) slice. With many slices some bounds will look bad or good by chance.
The audit is underpowered on small slices and is not a production-readiness claim.

The kernel is executable Python. score_rows() hashes the kernel bytes in --model-dir and refuses (REFUSE:UNPINNED_KERNEL, exit 3)
unless they equal a pinned kernel (in-repo, or HF SZLHOLDINGS/oac-system-health-v1 @ ed0a3bbd), BEFORE compiling or running them.
--allow-unpinned-kernel is the deliberate escape. Nothing runs at import time.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, types
from pathlib import Path

# Same pin as oac_rbcg.py / oac_ltt.py (#771): sha256 of oac_operational_health.py.
PINNED_KERNEL_SHA256 = frozenset({"5e06ddffce9615a12c351e9926e24fefa142c920017b9dc844c584c58517202e",
                                  "e2aa63991b94596a16d69a288510b4b10af88932e075c055d97d1dac4d9a930d"})

class Refuse(Exception):
    pass

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

def cut_points(rows):
    """Median cut per numeric feature (more than two distinct values), from the rows given (pass the calibration split)."""
    cuts = {}
    for k in rows[0]["features"]:
        vals = sorted(r["features"][k] for r in rows)
        if len(set(vals)) > 2: cuts[k] = vals[len(vals) // 2]
    return cuts

def slices_for(features: dict, numeric_cuts: dict):
    out = {}
    for k, v in features.items():
        if k in numeric_cuts: out[k] = "high" if v > numeric_cuts[k] else "low"
        else: out[k] = str(int(v)) if float(v) in (0.0, 1.0) else str(v)
    return out

def audit(rows, scores, lam, cuts, alpha=0.20, delta=0.10, min_pos=10):
    """`cuts` must come from cut_points() on a split other than `rows` (the calibration split), so slices are fixed before auditing."""
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

def load_kernel_class(model_dir, allow_unpinned=False):
    """Hash the kernel bytes, require the pin, then compile exactly those bytes. Nothing from --model-dir runs before the pin."""
    path = Path(model_dir).resolve() / "oac_operational_health.py"; raw = path.read_bytes(); h = hashlib.sha256(raw).hexdigest()
    if not allow_unpinned and h not in PINNED_KERNEL_SHA256: raise Refuse("UNPINNED_KERNEL")
    mod = types.ModuleType(f"_oac_kernel_{h[:16]}"); mod.__file__ = str(path); sys.modules[mod.__name__] = mod  # dataclasses needs the module registered
    exec(compile(raw, str(path), "exec"), mod.__dict__); return mod.OperationalHealthKernel

def score_rows(model_dir, rows, allow_unpinned=False):
    md = Path(model_dir).resolve()
    K = load_kernel_class(md, allow_unpinned)(str(md / "model.json"), str(md / "artifact_receipt.json"))
    return [K.score(r["features"])["operator_attention_score"] for r in rows]

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--lambda", dest="lam", type=float, required=True)
    ap.add_argument("--cuts-split", default="validation", help="split the numeric cut points come from (must differ from --split)")
    ap.add_argument("--allow-unpinned-kernel", action="store_true", help="deliberately run a kernel outside the pinned set"); a = ap.parse_args(argv)
    if a.cuts_split == a.split: print("REFUSE:CUTS_FROM_AUDITED_SPLIT"); return 2
    try:
        rows = load_split(a.data_dir, a.split); cuts = cut_points(load_split(a.data_dir, a.cuts_split))
        sc = score_rows(a.model_dir, rows, a.allow_unpinned_kernel)
    except Refuse as e: print(f"REFUSE:{e}"); return 3
    except (OSError, ValueError, KeyError, TypeError, IndexError, RecursionError) as e: print(f"REFUSE:MALFORMED:{type(e).__name__}"); return 3
    print(json.dumps({"split": a.split, "cuts_split": a.cuts_split, "cuts": cuts, "lambda": a.lam, "slices": audit(rows, sc, a.lam, cuts)}, indent=1)); return 0

if __name__ == "__main__": raise SystemExit(main())
