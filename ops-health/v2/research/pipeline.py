"""Build ONE v2 candidate artifact from given rows and given hyperparameters (no search).

Stage 2's registered search (PREREGISTRATION section 6) calls ``build_candidate`` once per
trial; stage 1 uses it only on DEV data (split name "dev") in tests.

Steps (frozen resolution: every fitted real number is rounded to 12 decimals, conformal
quantiles rounded up, immediately after fitting, and every later step uses the rounded values):
  1. fit L2 logistic regression by Newton/IRLS on the train rows (lr.fit);
  2. raw scores on the calibration rows with the kernel's arithmetic; fit the calibrator;
  3. calibrated scores on the threshold rows; choose the decision threshold on the grid
     t = k/100, k = 1..99: maximize balanced accuracy, ties to higher F1, then to t closer to
     0.5, then to the lower t (BA and F1 compared as exact fractions);
  4. calibrated scores on the conformal rows; fit Mondrian q_hat at alpha;
  5. assemble the artifact and validate it with the kernel.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Mapping, Sequence

from . import calibration, conformal, generator, kernel_bridge, lr
from .dataio import labels_of

SPLIT_ROLES = ("train", "calibration", "conformal", "threshold")


def rows_sha256(rows: Sequence[Mapping]) -> str:
    return hashlib.sha256(generator.jsonl_bytes(rows)).hexdigest()


def normalized_vectors(rows: Sequence[Mapping]) -> list[list[float]]:
    normalize = kernel_bridge.kernel().normalize_features
    return [normalize(row["features"]) for row in rows]


def quantize(value: float) -> float:
    return round(float(value), 12)


def select_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict:
    y = [int(v) for v in labels]
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise ValueError("threshold selection needs both classes")
    best = None
    for k in range(1, 100):
        t = k / 100
        tp = sum(1 for yi, s in zip(y, scores) if yi == 1 and s >= t)
        tn = sum(1 for yi, s in zip(y, scores) if yi == 0 and s < t)
        fp = neg - tn
        fn = pos - tp
        ba = (Fraction(tp, pos) + Fraction(tn, neg)) / 2
        f1 = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
        rank = (ba, f1, -abs(k - 50))
        if best is None or rank > best[0]:  # strict: equal ranks keep the lower t
            best = (rank, k, t, tp, fp, tn, fn)
    rank, k, t, tp, fp, tn, fn = best
    return {
        "threshold": t,
        "grid_index": k,
        "balanced_accuracy": float(rank[0]),
        "f1": float(rank[1]),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def generator_block() -> dict:
    return {
        "version": generator.GENERATOR_VERSION,
        "source_sha256": generator.source_sha256(),
        "master_seed": generator.MASTER_SEED,
        "row_schema": generator.ROW_SCHEMA,
    }


def score_rows(scorer, rows: Sequence[Mapping]) -> list[dict]:
    return [scorer.evaluate(v) for v in normalized_vectors(rows)]


def build_candidate(
    *,
    train_rows: Sequence[Mapping],
    calibration_rows: Sequence[Mapping],
    conformal_rows: Sequence[Mapping],
    threshold_rows: Sequence[Mapping],
    l2: float,
    calibrator: str,
    split_names: Mapping[str, str],
    alpha: float = conformal.ALPHA,
) -> tuple[dict, dict]:
    """Returns (artifact, diagnostics).  The artifact is at frozen resolution and kernel-valid."""
    k = kernel_bridge.kernel()
    if set(split_names) != set(SPLIT_ROLES):
        raise ValueError(f"split_names must name {SPLIT_ROLES}")
    # 1. weights
    fit = lr.fit(lr.columns_from_vectors(normalized_vectors(train_rows)), labels_of(train_rows), l2)
    intercept = quantize(fit["intercept"])
    weights = {name: quantize(w) for name, w in zip(k.FEATURE_NAMES, fit["weights"], strict=True)}
    rows_by_role = {
        "train": train_rows,
        "calibration": calibration_rows,
        "conformal": conformal_rows,
        "threshold": threshold_rows,
    }
    artifact = {
        "schema": k.MODEL_SCHEMA,
        "model_type": k.MODEL_TYPE,
        "purpose": k.MODEL_PURPOSE,
        "authority": k.MODEL_AUTHORITY,
        "synthetic_training_data": True,
        "feature_manifest": k.canonical_feature_manifest(),
        "intercept": intercept,
        "weights": weights,
        "calibrator": calibration.identity(),
        "decision_threshold": 0.5,
        "conformal": {"method": k.CONFORMAL_METHOD, "alpha": alpha,
                      "q_hat": {"0": 1.0, "1": 1.0}, "n_calibration": {"0": 1, "1": 1}},
        "score_semantics": k.SCORE_SEMANTICS,
        "training": {
            "algorithm": k.TRAINING_ALGORITHM,
            "l2": float(l2),
            "iterations": fit["iterations"],
            "converged": fit["converged"],
            "stop_reason": fit["stop_reason"],
            "max_iter": fit["max_iter"],
            "tol": fit["tol"],
            "threshold_selection": k.THRESHOLD_SELECTION,
            "threshold_grid": k.THRESHOLD_GRID,
            "splits": {
                role: {
                    "name": split_names[role],
                    "rows": len(rows_by_role[role]),
                    "sha256": rows_sha256(rows_by_role[role]),
                }
                for role in SPLIT_ROLES
            },
        },
        "generator": generator_block(),
    }
    # 2. calibrator on raw scores (model at frozen resolution, identity calibrator)
    raw_scorer = k.ArtifactScorer(artifact)
    cal_raw = [r["raw_score"] for r in score_rows(raw_scorer, calibration_rows)]
    fitted_cal, cal_diag = calibration.fit(calibrator, cal_raw, labels_of(calibration_rows))
    artifact["calibrator"] = fitted_cal
    # 3. threshold on calibrated scores
    cal_scorer = k.ArtifactScorer(artifact)
    thr_scores = [r["calibrated_score"] for r in score_rows(cal_scorer, threshold_rows)]
    chosen = select_threshold(labels_of(threshold_rows), thr_scores)
    artifact["decision_threshold"] = chosen["threshold"]
    # 4. conformal quantiles on calibrated scores
    conf_scores = [r["calibrated_score"] for r in score_rows(cal_scorer, conformal_rows)]
    conf_fit = conformal.fit_mondrian(conf_scores, labels_of(conformal_rows), alpha)
    artifact["conformal"] = conf_fit["block"]
    # 5. validate
    validated = k.validate_artifact(artifact)
    diagnostics = {
        "lr": {key: fit[key] for key in ("iterations", "converged", "stop_reason", "objective",
                                         "gradient_max_abs", "last_newton_step_max_abs",
                                         "step_halvings", "solvers")},
        "calibrator": cal_diag,
        "threshold": chosen,
        "conformal": conf_fit["diagnostics"],
    }
    return validated, diagnostics
