"""One run of the v3 recipe on given rows (protocol s.6.1-s.6.5). SYNTHETIC; standard library only.

Steps (frozen resolution, as v2: every fitted real number is rounded to 12 decimals right after
fitting, conformal quantiles rounded up, and every later step uses the rounded values):
  1. normalize the eight features with the declared table (representation R, s.6.2, or the v1
     min-max table for the no-R ablations), through the kernel's arithmetic;
  2. fit L2 logistic regression, l2 = 1e-3, by v2's Newton/IRLS (``v2/research/lr.py``,
     reused unchanged: max |delta| < 1e-10 or 100 iterations; CAP_REACHED and
     LINE_SEARCH_FAILED are non-converged, s.6.1);
  3. calibrator ``none`` (v3), or Platt on ``calibration`` (ABL-calibration only), by v2's
     ``calibration.fit_platt``;
  4. threshold on the threshold rows (``validation``): v2's rule (s.6.4), stored as the grid
     index k = 100 t, by v2's ``pipeline.select_threshold``;
  5. Mondrian split conformal at alpha = 0.10 on ``conformal`` (s.6.5), by v2's
     ``conformal.fit_mondrian`` (the same arithmetic as the v3 kernel's ``nonconformity``);
  6. assemble the artifact and validate it with the v3 kernel (an artifact whose table is not R
     is not kernel-valid; ``build_candidate`` then returns it unvalidated, flagged, for the
     research scorer, s.5).

DEV rule (s.15.5): a fit on DEV rows (sample ids ``dev-*``) is refused above 2,048 rows.
This module builds one candidate; the bounded sequence of declared fits, the fit trace and the
clean-tree and runtime-pin refusals of a registered fit belong to the segment runner (s.6.6,
s.13 steps 4-6), not to this module.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

from v2.research import calibration as v2_calibration
from v2.research import conformal as v2_conformal
from v2.research import lr as v2_lr
from v2.research import pipeline as v2_pipeline

from . import generator, guards, kernel_bridge, registry, representation

REQUIRED_ROLES = ("train", "threshold", "conformal")


def labels_of(rows: Sequence[Mapping]) -> list[int]:
    return [1 if row["label"]["operator_attention_required"] else 0 for row in rows]


def rows_sha256(rows: Sequence[Mapping]) -> str:
    return hashlib.sha256(generator.jsonl_bytes(rows)).hexdigest()


def normalized_vectors(rows: Sequence[Mapping], table: Mapping = representation.TABLE_R) -> list[list[float]]:
    normalize = kernel_bridge.kernel().normalize_features
    return [normalize(row["features"], table) for row in rows]


def quantize(value: float) -> float:
    return round(float(value), 12)


def generator_block() -> dict:
    return {
        "version": generator.GENERATOR_VERSION,
        "source_sha256": generator.source_sha256(),
        "namespace": generator.SEED_NAMESPACE,
        "master_seed": generator.MASTER_SEED,
        "row_schema": generator.ROW_SCHEMA,
    }


class _Scorer:
    """Research scorer for any well-formed table (the kernel's arithmetic, s.5 for ABL-R/DR)."""

    def __init__(self, artifact: Mapping):
        k = kernel_bridge.kernel()
        self.k = k
        self.artifact = artifact
        self.table = representation.parse_table(artifact["transform_table"])
        self.weights = [artifact["weights"][name] for name in representation.FEATURE_NAMES]

    def calibrated(self, rows: Sequence[Mapping]) -> list[float]:
        out = []
        for vector in normalized_vectors(rows, self.table):
            linear = self.artifact["intercept"]
            for weight, value in zip(self.weights, vector, strict=True):
                linear += weight * value
            out.append(self.k.apply_calibrator(self.artifact["calibrator"], self.k.sigmoid(linear)))
        return out


def build_candidate(
    *,
    train_rows: Sequence[Mapping],
    threshold_rows: Sequence[Mapping],
    conformal_rows: Sequence[Mapping],
    split_names: Mapping[str, str],
    table: Mapping = representation.TABLE_R,
    calibration_rows: Sequence[Mapping] | None = None,
    calibrator: str = "none",
) -> tuple[dict, dict]:
    """Returns (artifact, diagnostics). The artifact is at frozen resolution."""
    k = kernel_bridge.kernel()
    if calibrator not in ("none", "platt"):
        raise ValueError("v3 calibrators: none (v3) or platt (ABL-calibration)")
    roles = list(REQUIRED_ROLES) + (["calibration"] if calibrator == "platt" else [])
    if set(split_names) != set(roles):
        raise ValueError(f"split_names must name exactly {roles}")
    if calibrator == "platt" and not calibration_rows:
        raise ValueError("Platt needs calibration rows")
    guards.check_dev_fit(train_rows)
    table = {name: (float(table[name][0]), float(table[name][1])) for name in representation.FEATURE_NAMES}
    # 1-2. weights
    fit = v2_lr.fit(v2_lr.columns_from_vectors(normalized_vectors(train_rows, table)),
                    labels_of(train_rows), registry.L2)
    rows_by_role = {"train": train_rows, "threshold": threshold_rows, "conformal": conformal_rows}
    if calibrator == "platt":
        rows_by_role["calibration"] = calibration_rows
    artifact = {
        "schema": k.MODEL_SCHEMA,
        "model_type": k.MODEL_TYPE,
        "purpose": k.MODEL_PURPOSE,
        "authority": k.MODEL_AUTHORITY,
        "synthetic_training_data": True,
        "feature_manifest": k.canonical_feature_manifest(),
        "transform_table": representation.table_block(table),
        "intercept": quantize(fit["intercept"]),
        "weights": {name: quantize(w) for name, w in zip(k.FEATURE_NAMES, fit["weights"], strict=True)},
        "calibrator": {"kind": "none", "params": {}},
        "threshold_index": 50,
        "conformal": {"method": k.CONFORMAL_METHOD, "alpha": registry.ALPHA,
                      "q_hat": {"0": 1.0, "1": 1.0}, "n_calibration": {"0": 1, "1": 1}},
        "score_semantics": k.SCORE_SEMANTICS,
        "training": {
            "algorithm": k.TRAINING_ALGORITHM,
            "l2": registry.L2,
            "iterations": fit["iterations"],
            "converged": fit["converged"],
            "stop_reason": fit["stop_reason"],
            "max_iter": fit["max_iter"],
            "tol": fit["tol"],
            "threshold_selection": k.THRESHOLD_SELECTION,
            "threshold_grid": k.THRESHOLD_GRID,
            "splits": {
                role: {"name": split_names[role], "rows": len(rows_by_role[role]),
                       "sha256": rows_sha256(rows_by_role[role])}
                for role in roles
            },
        },
        "generator": generator_block(),
    }
    cal_diag = None
    # 3. calibrator (ABL-calibration only)
    if calibrator == "platt":
        raw = [k.sigmoid(z) for z in _linear(artifact, calibration_rows, table)]
        fitted, cal_diag = v2_calibration.fit_platt(raw, labels_of(calibration_rows))
        artifact["calibrator"] = fitted
    scorer = _Scorer(artifact)
    # 4. threshold on the threshold rows
    chosen = v2_pipeline.select_threshold(labels_of(threshold_rows), scorer.calibrated(threshold_rows))
    artifact["threshold_index"] = int(chosen["grid_index"])
    # 5. conformal quantiles
    conf = v2_conformal.fit_mondrian(scorer.calibrated(conformal_rows), labels_of(conformal_rows),
                                     registry.ALPHA)
    artifact["conformal"] = dict(conf["block"])
    # 6. validate (kernel-valid only on table R)
    kernel_valid = table == representation.TABLE_R
    if kernel_valid:
        artifact = k.validate_artifact(artifact)
    diagnostics = {
        "lr": {key: fit[key] for key in ("iterations", "converged", "stop_reason", "objective",
                                         "gradient_max_abs", "last_newton_step_max_abs",
                                         "step_halvings", "solvers", "rows")},
        "eligible": bool(fit["converged"]),
        "calibrator": cal_diag,
        "threshold": chosen,
        "conformal": conf["diagnostics"],
        "kernel_valid": kernel_valid,
    }
    return artifact, diagnostics


def _linear(artifact: Mapping, rows: Sequence[Mapping], table: Mapping) -> list[float]:
    weights = [artifact["weights"][name] for name in representation.FEATURE_NAMES]
    out = []
    for vector in normalized_vectors(rows, table):
        linear = artifact["intercept"]
        for weight, value in zip(weights, vector, strict=True):
            linear += weight * value
        out.append(linear)
    return out
