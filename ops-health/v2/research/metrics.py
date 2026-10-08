"""Evaluation metrics (PREREGISTRATION sections 7, 8, 10).  Standard library only.  SYNTHETIC.

Conventions
  * labels y in {0, 1} (1 = operator attention required); decisions d in {0, 1} (1 = ALERT);
  * an undefined value (zero denominator, missing class) is None (JSON null), never 0;
  * sums of floats use math.fsum.
Definitions
  precision = TP/(TP+FP); recall = sensitivity = TP/(TP+FN); specificity = TN/(TN+FP);
  balanced accuracy (BA) = (recall + specificity)/2; F1 = 2TP/(2TP+FP+FN);
  false-alert share = FP/(TP+FP); accuracy = (TP+TN)/n; prevalence = (TP+FN)/n.
  Brier = mean (s - y)^2.  Log loss = -mean[y log s + (1-y) log(1-s)], s clipped to
  [1e-15, 1 - 1e-15] (v1's clip).
  ECE: 10 equal-width bins [0,0.1), ..., [0.8,0.9), [0.9,1.0] (last bin closed), bin edges k/10
  as doubles; ECE = sum_b (n_b/n) |mean_b(s) - mean_b(y)| = (1/n) sum_b |sum_{i in b}(s_i - y_i)|
  (count-weighted).  The reliability table lists every bin, empty bins with null means.
  ROC AUC = Mann-Whitney statistic with average ranks for ties:
  (R_pos - P(P+1)/2) / (P N).
Baseline validity (section 10): per-class rates (precision, recall/sensitivity, specificity, F1,
false-alert share) are emitted only if (i) the receipt verifies, (ii) the evaluation set has at
least 30 rows of each class, (iii) the decisions are not constant on that set, and (iv) the
fixture refused every fail-closed probe.  Otherwise the report carries INVALID_BASELINE and the
failed preconditions, and the per-class rates are absent (not null).  BA and prevalence are still
reported (constant decisions give BA = 0.5 by construction); accuracy and the confusion counts
are withheld too, because accuracy = prevalence * sensitivity + (1 - prevalence) * specificity
together with BA and prevalence would let a reader solve for sensitivity and specificity
(stage-1 review finding B1, logs/v2/review/STAGE1_REVIEW.md).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from . import truth

INVALID_BASELINE = "INVALID_BASELINE"
MIN_ROWS_PER_CLASS = 30
LOG_LOSS_CLIP = 1e-15
ECE_BINS = 10
PER_CLASS_RATE_KEYS = ("precision", "recall", "specificity", "f1", "false_alert_share")
PRECONDITIONS = {
    "i_receipt_verifies": "(i) receipt verifies",
    "ii_min_rows_per_class": f"(ii) at least {MIN_ROWS_PER_CLASS} rows of each class",
    "iii_decisions_not_constant": "(iii) decisions are not constant",
    "iv_refused_all_probes": "(iv) refused every fail-closed probe",
}


def _div(num: float, den: float) -> float | None:
    return num / den if den else None


def _check_binary(values: Sequence[int], label: str) -> list[int]:
    out = [int(v) for v in values]
    if any(v not in (0, 1) for v in out):
        raise ValueError(f"{label} must be 0/1")
    return out


def confusion(labels: Sequence[int], decisions: Sequence[int]) -> dict[str, int]:
    y = _check_binary(labels, "labels")
    d = _check_binary(decisions, "decisions")
    if len(y) != len(d):
        raise ValueError("labels and decisions must be aligned")
    tp = sum(1 for a, b in zip(y, d) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, d) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(y, d) if a == 0 and b == 0)
    fn = sum(1 for a, b in zip(y, d) if a == 1 and b == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def rates_from_counts(tp: float, fp: float, tn: float, fn: float) -> dict[str, float | None]:
    """All confusion-derived rates; works for integer or weighted counts."""
    n = tp + fp + tn + fn
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    return {
        "precision": _div(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None,
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "false_alert_share": _div(fp, tp + fp),
        "accuracy": _div(tp + tn, n),
        "prevalence": _div(tp + fn, n),
    }


def balanced_accuracy(labels: Sequence[int], decisions: Sequence[int]) -> float | None:
    c = confusion(labels, decisions)
    return rates_from_counts(c["tp"], c["fp"], c["tn"], c["fn"])["balanced_accuracy"]


def _scores(scores: Sequence[float]) -> list[float]:
    out = [float(s) for s in scores]
    if any(not (0.0 <= s <= 1.0) for s in out):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return out


def brier(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = _check_binary(labels, "labels")
    s = _scores(scores)
    if not y:
        return None
    return math.fsum((si - yi) ** 2 for si, yi in zip(s, y, strict=True)) / len(y)


def log_loss_terms(labels: Sequence[int], scores: Sequence[float]) -> list[float]:
    y = _check_binary(labels, "labels")
    out = []
    for si, yi in zip(_scores(scores), y, strict=True):
        c = min(max(si, LOG_LOSS_CLIP), 1.0 - LOG_LOSS_CLIP)
        out.append(-math.log(c) if yi == 1 else -math.log(1.0 - c))
    return out


def log_loss(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    terms = log_loss_terms(labels, scores)
    return math.fsum(terms) / len(terms) if terms else None


def ece_bin(score: float, bins: int = ECE_BINS) -> int:
    """Largest k with score >= k/bins (edges as doubles); the last bin is closed."""
    k = min(int(score * bins), bins - 1)
    while k > 0 and score < k / bins:
        k -= 1
    while k < bins - 1 and score >= (k + 1) / bins:
        k += 1
    return k


def reliability(labels: Sequence[int], scores: Sequence[float], bins: int = ECE_BINS) -> dict:
    y = _check_binary(labels, "labels")
    s = _scores(scores)
    n = len(y)
    members: list[list[int]] = [[] for _ in range(bins)]
    for i, si in enumerate(s):
        members[ece_bin(si, bins)].append(i)
    table = []
    gaps = []
    for b, idx in enumerate(members):
        count = len(idx)
        diff = math.fsum(s[i] - y[i] for i in idx)
        gaps.append(abs(diff))
        table.append(
            {
                "bin": b,
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "upper_closed": b == bins - 1,
                "count": count,
                "mean_score": math.fsum(s[i] for i in idx) / count if count else None,
                "fraction_positive": sum(y[i] for i in idx) / count if count else None,
                "gap": abs(diff) / count if count else None,
            }
        )
    return {"ece": math.fsum(gaps) / n if n else None, "bins": bins, "table": table}


def ece(labels: Sequence[int], scores: Sequence[float], bins: int = ECE_BINS) -> float | None:
    return reliability(labels, scores, bins)["ece"]


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = _check_binary(labels, "labels")
    s = [float(v) for v in scores]
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = truth.average_ranks(s)
    rank_sum = math.fsum(r for r, yi in zip(ranks, y, strict=True) if yi == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


# --------------------------------------------------------------------------------------
# conformal: coverage and selective metrics
# --------------------------------------------------------------------------------------
def coverage(labels: Sequence[int], prediction_sets: Sequence[Sequence[int]]) -> dict:
    y = _check_binary(labels, "labels")
    hits = [1 if yi in set(ps) else 0 for yi, ps in zip(y, prediction_sets, strict=True)]
    out = {"marginal": _div(sum(hits), len(hits)), "per_class": {}, "rows": len(hits)}
    for cls in (0, 1):
        sel = [h for h, yi in zip(hits, y) if yi == cls]
        out["per_class"][str(cls)] = _div(sum(sel), len(sel))
    return out


def selective(labels: Sequence[int], advisories: Sequence[str], reasons: Sequence[str | None]) -> dict:
    """Abstention rates (split BOTH / EMPTY) and metrics on non-abstained rows."""
    y = _check_binary(labels, "labels")
    n = len(y)
    both = sum(1 for a, r in zip(advisories, reasons) if a == "ABSTAIN" and r == "BOTH")
    empty = sum(1 for a, r in zip(advisories, reasons) if a == "ABSTAIN" and r == "EMPTY")
    kept = [(yi, 1 if a == "ALERT" else 0) for yi, a in zip(y, advisories, strict=True) if a != "ABSTAIN"]
    if any(a not in ("ALERT", "NO_ALERT", "ABSTAIN") for a in advisories):
        raise ValueError("unknown advisory")
    out = {
        "rows": n,
        "abstained": both + empty,
        "abstention_rate": _div(both + empty, n),
        "abstention_rate_both": _div(both, n),
        "abstention_rate_empty": _div(empty, n),
        "non_abstained": len(kept),
    }
    if kept:
        c = confusion([k[0] for k in kept], [k[1] for k in kept])
        r = rates_from_counts(c["tp"], c["fp"], c["tn"], c["fn"])
        out["selective_balanced_accuracy"] = r["balanced_accuracy"]
        out["selective_accuracy"] = r["accuracy"]
    else:
        out["selective_balanced_accuracy"] = None
        out["selective_accuracy"] = None
    return out


# --------------------------------------------------------------------------------------
# baseline validity (INVALID_BASELINE) and gated binary report
# --------------------------------------------------------------------------------------
def baseline_validity(
    labels: Sequence[int],
    decisions: Sequence[int],
    *,
    receipt_verified: bool,
    probes_refused: int,
    probes_total: int,
) -> dict:
    y = _check_binary(labels, "labels")
    d = _check_binary(decisions, "decisions")
    pos = sum(y)
    neg = len(y) - pos
    failed = []
    if receipt_verified is not True:
        failed.append("i_receipt_verifies")
    if pos < MIN_ROWS_PER_CLASS or neg < MIN_ROWS_PER_CLASS:
        failed.append("ii_min_rows_per_class")
    if len(set(d)) < 2:
        failed.append("iii_decisions_not_constant")
    if probes_total <= 0 or probes_refused != probes_total:
        failed.append("iv_refused_all_probes")
    return {
        "valid": not failed,
        "status": "VALID" if not failed else INVALID_BASELINE,
        "failed_preconditions": failed,
        "failed_descriptions": [PRECONDITIONS[f] for f in failed],
        "class_counts": {"0": neg, "1": pos},
        "probes": {"refused": probes_refused, "total": probes_total},
    }


def binary_report(labels: Sequence[int], decisions: Sequence[int], validity: Mapping) -> dict:
    """BA and prevalence always; accuracy, confusion and per-class rates only when the baseline
    is valid (accuracy with BA and prevalence would reveal sensitivity and specificity)."""
    c = confusion(labels, decisions)
    r = rates_from_counts(c["tp"], c["fp"], c["tn"], c["fn"])
    report = {
        "rows": len(labels),
        "balanced_accuracy": r["balanced_accuracy"],
        "prevalence": r["prevalence"],
        "baseline_validity": dict(validity),
    }
    if validity.get("valid") is True:
        report["accuracy"] = r["accuracy"]
        report["confusion"] = c
        report["per_class_rates"] = {key: r[key] for key in PER_CLASS_RATE_KEYS}
    else:
        report["per_class_rates"] = INVALID_BASELINE
    return report


def score_report(labels: Sequence[int], scores: Sequence[float]) -> dict:
    rel = reliability(labels, scores)
    return {
        "brier": brier(labels, scores),
        "log_loss": log_loss(labels, scores),
        "ece": rel["ece"],
        "reliability_table": rel["table"],
        "roc_auc": roc_auc(labels, scores),
    }
