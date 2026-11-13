#!/usr/bin/env python3
"""Learn-then-Test missed-alert control for oac-system-health-v1 (synthetic only).

Standard library only.  Advisory research tooling: it does not change the
published model, its card threshold, or any authority boundary.

What it does, in order:
  1. Reproduces the receipt's test/validation metrics at the card threshold.
  2. Certifies, on the validation split only, an alert threshold whose
     missed-alert rate (FNR, conditional on a true alert) is <= alpha with
     probability >= 1 - delta, using Learn-then-Test (Angelopoulos, Bates,
     Candes, Jordan, Lei 2021, arXiv:2110.01052) with fixed-sequence testing
     over a data-independent grid and exact-binomial or Hoeffding-Bentkus
     p-values.
  3. Certifies a second, false-alarm (FPR) threshold the same way and treats
     the scores between the two as a defer-to-human band.
  4. Applies everything once to the test split, next to baselines.
  5. Runs a resampling validity check: how often the certified and the naive
     thresholds exceed alpha on resampled calibration sets.

Paths are resolved relative to this file (clinical-gateway/tools/) unless
--model-dir / --data-dir are given.  No absolute paths are hard-coded.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE.parent / "huggingface" / "model" / "oac-system-health-v1"
DEFAULT_DATA_DIR = (
    HERE.parent / "huggingface" / "dataset" / "oac-clinical-transport-observability-synthetic" / "data"
)
GRID_STEP = 0.0005
GRID = [round(i * GRID_STEP, 4) for i in range(int(round(1 / GRID_STEP)) + 1)]
LABEL_KEY = "operator_attention_required"
SCORE_KEY = "operator_attention_score"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_kernel(model_dir: Path):
    kpath = model_dir / "oac_operational_health.py"
    spec = importlib.util.spec_from_file_location("oac_kernel_under_test", kpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod.OperationalHealthKernel(model_dir / "model.json", model_dir / "artifact_receipt.json")


def load_split(data_dir: Path, split: str):
    rows = []
    for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_rows(kernel, rows):
    scores, labels = [], []
    for r in rows:
        scores.append(float(kernel.score(r["features"])[SCORE_KEY]))
        labels.append(bool(r["label"][LABEL_KEY]))
    return scores, labels


# ---------------------------------------------------------------- metrics
def confusion(scores, labels, thr):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        a = s >= thr
        if a and y:
            tp += 1
        elif a:
            fp += 1
        elif y:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def roc_auc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    npos = sum(labels)
    nneg = len(labels) - npos
    rsum = sum(r for r, y in zip(ranks, labels) if y)
    return (rsum - npos * (npos + 1) / 2) / (npos * nneg)


def metrics(scores, labels, thr):
    tp, fp, tn, fn = confusion(scores, labels, thr)
    n = len(labels)
    eps = 1e-15
    ll = -sum(math.log(min(max(s, eps), 1 - eps)) if y else math.log(1 - min(max(s, eps), 1 - eps))
              for s, y in zip(scores, labels)) / n
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": round((tp + tn) / n, 12),
        "balanced_accuracy": round((rec + spec) / 2, 12),
        "confusion": {"false_negative": fn, "false_positive": fp, "true_negative": tn, "true_positive": tp},
        "f1": round(f1, 12),
        "log_loss": round(ll, 12),
        "negative_rows": n - sum(labels),
        "positive_rows": sum(labels),
        "precision": round(prec, 12),
        "recall": round(rec, 12),
        "roc_auc": round(roc_auc(scores, labels), 12),
        "rows": n,
        "specificity": round(spec, 12),
    }


def compare_to_receipt(observed, expected, tol=1e-9):
    diffs = {}
    for k, v in expected.items():
        o = observed.get(k)
        if isinstance(v, dict):
            if o != v:
                diffs[k] = {"expected": v, "observed": o}
        elif isinstance(v, float):
            if o is None or abs(o - v) > tol:
                diffs[k] = {"expected": v, "observed": o}
        elif o != v:
            diffs[k] = {"expected": v, "observed": o}
    return diffs


# ---------------------------------------------------------------- LTT
def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def _h1(a, b):
    def xlog(x, y):
        return 0.0 if x == 0 else x * math.log(x / y)
    return xlog(a, b) + xlog(1 - a, 1 - b)


def pvalue(k, n, alpha, method):
    """p-value for H0: risk > alpha, given k losses out of n (binary loss)."""
    if n == 0:
        return 1.0
    binom = binom_cdf(k, n, alpha)
    if method == "binomial":
        return binom
    if method == "hb":
        r = k / n
        p_h = math.exp(-n * _h1(min(r, alpha), alpha))
        p_b = math.e * binom
        return min(1.0, p_h, p_b)
    raise ValueError(method)


def ltt_fnr(pos_scores, alpha, delta, method, grid=GRID):
    """Fixed-sequence LTT, ascending lambda (FNR non-decreasing in lambda).
    Alert iff score >= lambda.  Returns the largest certified lambda, or None."""
    ps = sorted(pos_scores)
    n = len(ps)
    cdf_cache = {}
    certified, steps = None, 0
    last = None
    for lam in grid:
        k = bisect.bisect_left(ps, lam)  # positives with score < lam = missed
        if k not in cdf_cache:
            cdf_cache[k] = pvalue(k, n, alpha, method)
        p = cdf_cache[k]
        steps += 1
        if p <= delta:
            certified = lam
            last = {"lambda": lam, "misses": k, "n_pos": n, "p_value": p}
        else:
            return certified, last, {"lambda": lam, "misses": k, "n_pos": n, "p_value": p}, steps
    return certified, last, None, steps


def ltt_fpr(neg_scores, beta, delta, method, grid=GRID):
    """Fixed-sequence LTT, descending lambda (FPR non-increasing in lambda).
    Auto-alert iff score >= lambda.  Returns the smallest certified lambda."""
    ns = sorted(neg_scores)
    n = len(ns)
    certified, last = None, None
    for lam in reversed(grid):
        k = n - bisect.bisect_left(ns, lam)  # negatives with score >= lam = false alarms
        p = pvalue(k, n, beta, method)
        if p <= delta:
            certified = lam
            last = {"lambda": lam, "false_alarms": k, "n_neg": n, "p_value": p}
        else:
            return certified, last, {"lambda": lam, "false_alarms": k, "n_neg": n, "p_value": p}
    return certified, last, None


def naive_fnr(pos_scores, alpha, grid=GRID):
    ps = sorted(pos_scores)
    n = len(ps)
    best = None
    for lam in grid:
        if bisect.bisect_left(ps, lam) / n <= alpha:
            best = lam
        else:
            break
    return best


def fnr_at(scores, labels, lam):
    pos = [s for s, y in zip(scores, labels) if y]
    return sum(1 for s in pos if s < lam) / len(pos) if pos else float("nan")


def operating_point(scores, labels, lam, name):
    tp, fp, tn, fn = confusion(scores, labels, lam)
    n = len(labels)
    return {
        "name": name, "threshold": lam,
        "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "missed_alert_rate_fnr": round(fn / (tp + fn), 6) if tp + fn else None,
        "alert_rate": round((tp + fp) / n, 6),
        "false_alarm_rate_fpr": round(fp / (fp + tn), 6) if fp + tn else None,
        "precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "accuracy": round((tp + tn) / n, 6),
    }


def rule_point(preds, labels, name):
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    tn = len(labels) - tp - fp - fn
    n = len(labels)
    return {
        "name": name, "threshold": None,
        "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "missed_alert_rate_fnr": round(fn / (tp + fn), 6) if tp + fn else None,
        "alert_rate": round((tp + fp) / n, 6),
        "false_alarm_rate_fpr": round(fp / (fp + tn), 6) if fp + tn else None,
        "precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "accuracy": round((tp + tn) / n, 6),
    }


def defer_band(scores, labels, lo, hi):
    auto_neg = [(s, y) for s, y in zip(scores, labels) if s < lo]
    auto_pos = [(s, y) for s, y in zip(scores, labels) if s >= hi]
    deferred = [(s, y) for s, y in zip(scores, labels) if lo <= s < hi]
    answered = len(auto_neg) + len(auto_pos)
    correct = sum(1 for _, y in auto_neg if not y) + sum(1 for _, y in auto_pos if y)
    npos = sum(labels)
    return {
        "lambda_low_auto_clear_below": lo, "lambda_high_auto_alert_at_or_above": hi,
        "rows": len(labels), "answered": answered, "deferred": len(deferred),
        "answered_fraction": round(answered / len(labels), 6),
        "accuracy_on_answered": round(correct / answered, 6) if answered else None,
        "auto_cleared": len(auto_neg), "missed_alerts_in_auto_cleared": sum(1 for _, y in auto_neg if y),
        "auto_alerted": len(auto_pos), "false_alarms_in_auto_alerted": sum(1 for _, y in auto_pos if not y),
        "positives_deferred": sum(1 for _, y in deferred if y),
        "missed_alert_rate_of_all_positives": round(sum(1 for _, y in auto_neg if y) / npos, 6) if npos else None,
    }


# ---------------------------------------------------------------- validity
def validity(pop_scores, pop_labels, n_cal, reps, alpha, delta, method, seed):
    rng = random.Random(seed)
    n = len(pop_scores)
    idx = list(range(n))
    pos_pop = sorted(s for s, y in zip(pop_scores, pop_labels) if y)

    def pop_fnr(lam):
        return bisect.bisect_left(pos_pop, lam) / len(pos_pop)

    boot = {"certified_exceed": 0, "naive_exceed": 0, "certified_none": 0, "fnr_cert": [], "fnr_naive": []}
    split = {"certified_exceed": 0, "naive_exceed": 0, "certified_none": 0}
    for _ in range(reps):
        # (a) bootstrap calibration set from the pooled empirical population; true risk = population FNR
        b = [rng.randrange(n) for _ in range(n_cal)]
        pos = [pop_scores[i] for i in b if pop_labels[i]]
        lam, *_ = ltt_fnr(pos, alpha, delta, method)
        nv = naive_fnr(pos, alpha) if pos else None
        if lam is None:
            boot["certified_none"] += 1
            lam = 0.0  # nothing certified -> alert on everything (FNR 0)
        f = pop_fnr(lam)
        boot["fnr_cert"].append(f)
        boot["certified_exceed"] += f > alpha
        fn_ = pop_fnr(nv if nv is not None else 0.0)
        boot["fnr_naive"].append(fn_)
        boot["naive_exceed"] += fn_ > alpha
        # (b) random partition: calibrate on n_cal, evaluate on the held-out remainder
        rng.shuffle(idx)
        cal, ev = idx[:n_cal], idx[n_cal:]
        pos = [pop_scores[i] for i in cal if pop_labels[i]]
        lam, *_ = ltt_fnr(pos, alpha, delta, method)
        if lam is None:
            split["certified_none"] += 1
            lam = 0.0
        nv = naive_fnr(pos, alpha) if pos else 0.0
        evs = [pop_scores[i] for i in ev]
        evl = [pop_labels[i] for i in ev]
        split["certified_exceed"] += fnr_at(evs, evl, lam) > alpha
        split["naive_exceed"] += fnr_at(evs, evl, nv) > alpha

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    return {
        "reps": reps, "seed": seed, "calibration_size": n_cal, "population_rows": n,
        "population_positives": len(pos_pop),
        "bootstrap_population": {
            "description": "calibration sets drawn with replacement from pooled validation+test; true risk = FNR on the pooled population",
            "certified_exceed_rate": round(boot["certified_exceed"] / reps, 6),
            "naive_exceed_rate": round(boot["naive_exceed"] / reps, 6),
            "certified_none_count": boot["certified_none"],
            "certified_fnr_median": round(q(boot["fnr_cert"], 0.5), 6),
            "certified_fnr_p90": round(q(boot["fnr_cert"], 0.9), 6),
            "naive_fnr_median": round(q(boot["fnr_naive"], 0.5), 6),
        },
        "random_partition": {
            "description": "pooled rows randomly split into calibration (n_cal) and held-out evaluation; exceed = held-out empirical FNR > alpha (includes evaluation noise)",
            "certified_exceed_rate": round(split["certified_exceed"] / reps, 6),
            "naive_exceed_rate": round(split["naive_exceed"] / reps, 6),
            "certified_none_count": split["certified_none"],
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--alpha", type=float, default=0.20, help="missed-alert (FNR) risk level")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=0.10, help="false-alarm (FPR) level for the auto-alert edge of the defer band")
    ap.add_argument("--pvalue", choices=["binomial", "hb"], default="binomial")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--output", type=Path, default=HERE / "oac_ltt_results.json")
    a = ap.parse_args(argv)

    md = a.model_dir.resolve()
    dd = a.data_dir.resolve()
    kernel = load_kernel(md)
    receipt = json.loads((md / "artifact_receipt.json").read_text())
    model = json.loads((md / "model.json").read_text())
    card_thr = float(model["decision_threshold"])

    splits = {s: score_rows(kernel, load_split(dd, s)) for s in ("train", "validation", "test")}
    repro = {}
    for s, (sc, lb) in splits.items():
        m = metrics(sc, lb, card_thr)
        exp = receipt["metrics"][s]
        repro[s] = {"observed": m, "diff_vs_receipt": compare_to_receipt(m, exp)}
    reproduced = all(not v["diff_vs_receipt"] for v in repro.values())

    vs, vl = splits["validation"]
    ts, tl = splits["test"]
    vpos = [s for s, y in zip(vs, vl) if y]
    vneg = [s for s, y in zip(vs, vl) if not y]

    ltt = {}
    for method in ("binomial", "hb"):
        lam, last, fail, steps = ltt_fnr(vpos, a.alpha, a.delta, method)
        hi, hlast, hfail = ltt_fpr(vneg, a.beta, a.delta, method)
        ltt[method] = {"fnr_certified_lambda": lam, "fnr_last_rejected": last, "fnr_first_not_rejected": fail,
                       "fnr_sequence_steps": steps, "fpr_certified_lambda": hi, "fpr_last_rejected": hlast,
                       "fpr_first_not_rejected": hfail}
    lam = ltt[a.pvalue]["fnr_certified_lambda"]
    hi = ltt[a.pvalue]["fpr_certified_lambda"]
    naive = naive_fnr(vpos, a.alpha)

    test_points = [
        operating_point(ts, tl, card_thr, "card_threshold_0.16"),
        operating_point(ts, tl, lam, f"ltt_certified_fnr_alpha{a.alpha}_delta{a.delta}_{a.pvalue}"),
        operating_point(ts, tl, ltt["hb"]["fnr_certified_lambda"], "ltt_certified_fnr_hb"),
        operating_point(ts, tl, naive, "naive_empirical_fnr_le_alpha_on_validation"),
        rule_point([r["features"]["consecutive_failures"] > 0 for r in load_split(dd, "test")], tl,
                   "rule_consecutive_failures_gt_0"),
        rule_point([False] * len(tl), tl, "always_negative"),
    ]
    val_points = [operating_point(vs, vl, card_thr, "card_threshold_0.16"),
                  operating_point(vs, vl, lam, "ltt_certified"),
                  operating_point(vs, vl, naive, "naive")]

    band = None
    if lam is not None and hi is not None:
        lo_b, hi_b = (lam, hi) if hi > lam else (lam, lam)
        band = {"note": "auto-clear below the FNR-certified lambda, auto-alert at/above the FPR-certified lambda, defer between",
                "overlap": hi <= lam,
                "validation": defer_band(vs, vl, lo_b, hi_b), "test": defer_band(ts, tl, lo_b, hi_b)}

    val = validity(vs + ts, vl + tl, len(vs), a.reps, a.alpha, a.delta, a.pvalue, a.seed)

    out = {
        "schema": "szl-oac/oac-ltt-abstention-results/v1",
        "status": "research_only_synthetic_not_qualification",
        "inputs": {
            "model_dir_name": md.name,
            "sha256": {f: sha256_file(md / f) for f in ("model.json", "artifact_receipt.json", "oac_operational_health.py")},
            "data_sha256": {s: sha256_file(dd / f"{s}.jsonl") for s in ("train", "validation", "test")},
            "script_sha256": sha256_file(Path(__file__)),
            "python": sys.version.split()[0],
        },
        "params": {"alpha": a.alpha, "delta": a.delta, "beta": a.beta, "pvalue_primary": a.pvalue,
                   "grid": {"start": 0.0, "stop": 1.0, "step": GRID_STEP, "points": len(GRID)},
                   "alert_rule": "alert iff operator_attention_score >= lambda"},
        "card_threshold": card_thr,
        "reproduction": {"reproduced_exactly": reproduced, "tolerance": 1e-9, "splits": repro},
        "ltt": ltt,
        "naive_threshold": naive,
        "validation_operating_points": val_points,
        "test_operating_points": test_points,
        "defer_band": band,
        "validity": val,
    }
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reproduced_exactly": reproduced, "certified_lambda": lam, "hb_lambda": ltt["hb"]["fnr_certified_lambda"],
                      "fpr_lambda": hi, "naive": naive, "output": a.output.name}, sort_keys=True))
    return 0 if reproduced else 4


if __name__ == "__main__":
    raise SystemExit(main())
