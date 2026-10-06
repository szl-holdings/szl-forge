"""Evaluate the fixed public synthetic OAC v1 test split, offline and read-only.

This receipt is descriptive evidence, never production or clinical qualification.
No training, threshold selection, network requests, or external corpus scoring.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from oac_operational_health import (  # noqa: E402
    FEATURE_SPECS,
    MODEL_AUTHORITY,
    MODEL_PURPOSE,
    RECEIPT_SCHEMA,
    OperationalHealthKernel,
    OperationalModelError,
    normalize_features,
)

REPORT_SCHEMA = "szl-oac/operational-health-evaluation/v1"
ADMISSION_SCHEMA = "szl-oac/operational-corpus-admission/v1"
FIXED_MODEL_SHA256 = "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557"
FIXED_TEST_SHA256 = "7fc6ea635fcc501d56eed6a8b9e4f5f3a9057d0aa04505502759197150d79679"
SPLIT_ROWS = {"train": 768, "validation": 192, "test": 240}
MAX_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 64
MODEL_FILE = "operational-model/artifacts/model.json"
MODEL_RECEIPT_FILE = "operational-model/artifacts/model-receipt.json"
DATASET_RECEIPT_FILE = "operational-model/artifacts/dataset-receipt.json"
KERNEL_FILE = "src/oac_operational_health.py"
TRAINER_FILE = "tools/train_operational_health_model.py"
SCHEMA_FILE = "operational-model/schema.json"
ADMISSION_HASH_FIELDS = frozenset({
    "corpus_sha256", "permission_evidence_sha256", "privacy_review_sha256",
    "label_protocol_sha256", "split_policy_sha256", "deduplication_report_sha256",
    "heldout_isolation_evidence_sha256",
})


class EvaluationError(ValueError):
    """A fixed error code; never includes input contents or filesystem paths."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_bounded(path: Path, maximum: int = MAX_BYTES) -> bytes:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise EvaluationError("INPUT_NOT_REGULAR_FILE")
        if getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise EvaluationError("INPUT_REPARSE_POINT")
        with path.open("rb") as handle:
            raw = handle.read(maximum + 1)
        if len(raw) > maximum:
            raise EvaluationError("INPUT_TOO_LARGE")
        return raw
    except OSError as exc:
        raise EvaluationError("INPUT_UNREADABLE") from exc


def strict_object(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvaluationError("JSON_DUPLICATE_KEY")
            result[key] = value
        return result

    def invalid_constant(_value: str) -> None:
        raise EvaluationError("JSON_NONFINITE_NUMBER")

    def bounded_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise EvaluationError("JSON_NONFINITE_NUMBER")
        return number

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant, parse_float=bounded_float)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise EvaluationError("INVALID_JSON_OBJECT") from exc
    if not isinstance(value, dict):
        raise EvaluationError("INVALID_JSON_OBJECT")
    pending = [(value, 1)]
    while pending:
        node, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise EvaluationError("JSON_DEPTH_EXCEEDED")
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
        pending.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return value


def require_fields(value: Any, fields: set[str] | frozenset[str], code: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise EvaluationError(code)


def validate_row(row: dict[str, Any], split: str, index: int) -> None:
    require_fields(row, {"schema", "synthetic", "sample_id", "features", "label"}, "ROW_SCHEMA")
    if (row["schema"] != "szl-oac/transport-health-observation/v1"
            or row["synthetic"] is not True or row["sample_id"] != f"{split}-{index:06d}"):
        raise EvaluationError("ROW_IDENTITY")
    require_fields(row["label"], {"operator_attention_required"}, "LABEL_SCHEMA")
    if type(row["label"]["operator_attention_required"]) is not bool:
        raise EvaluationError("LABEL_TYPE")
    require_fields(row["features"], {spec.name for spec in FEATURE_SPECS}, "FEATURE_SCHEMA")
    # The dataset schema requires integer binary features, unlike the wider inference API.
    for spec in FEATURE_SPECS:
        if spec.kind == "binary" and type(row["features"][spec.name]) is not int:
            raise EvaluationError("FEATURE_TYPE")
    try:
        normalize_features(row["features"])
    except (OperationalModelError, OverflowError, RecursionError) as exc:
        raise EvaluationError("FEATURE_VALUE") from exc


def proportion(numerator: int, denominator: int) -> dict[str, Any]:
    """Two-sided 95% Wilson score interval; undefined rates are null, not zero."""
    if (type(numerator) is not int or type(denominator) is not int
            or not 0 <= numerator <= denominator):
        raise EvaluationError("INVALID_PROPORTION_INPUTS")
    if denominator == 0:
        return {"value": None, "numerator": numerator, "denominator": 0, "wilson_95": None}
    value = numerator / denominator
    z = 1.959963984540054
    divisor = 1 + z * z / denominator
    center = (value + z * z / (2 * denominator)) / divisor
    half = z * math.sqrt(value * (1 - value) / denominator + z * z / (4 * denominator**2)) / divisor
    return {
        "value": round(value, 12), "numerator": numerator, "denominator": denominator,
        "wilson_95": {"lower": round(max(0.0, center - half), 12),
                      "upper": round(min(1.0, center + half), 12)},
    }


RISK_COVERAGE_CUTOFFS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def calculate_risk_coverage(labels: Sequence[int], scores: Sequence[float],
                            threshold: float) -> dict[str, Any]:
    """Describe fixed, label-blind selections; never choose or apply a policy.

    The selector is normalized distance from the existing decision boundary:
    (score - threshold)/(1 - threshold) on its positive side, and
    (threshold - score)/threshold on its negative side. This is neither a
    correctness probability nor confidence, calibration, trust, or authority.
    """
    def unit_interval(value: Any) -> bool:
        return type(value) in (float, int) and 0 <= value <= 1 and math.isfinite(value)

    if (not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray))
            or not isinstance(scores, Sequence) or isinstance(scores, (str, bytes, bytearray))
            or not labels or len(labels) != len(scores)
            or any(type(label) is not int or label not in (0, 1) for label in labels)
            or any(not unit_interval(score) for score in scores)
            or not unit_interval(threshold) or not 0 < threshold < 1):
        raise EvaluationError("INVALID_RISK_COVERAGE_INPUTS")
    rows = len(labels)
    positive, negative = sum(labels), rows - sum(labels)
    predicted = [score >= threshold for score in scores]
    # The exported numeric values have declared decimal comparison semantics.
    # Exact rationals preserve decimal ties without admitting below-cut neighbors.
    boundary = Fraction(str(threshold))
    decimal_scores = [Fraction(str(score)) for score in scores]
    margins = [(score - boundary) / (1 - boundary) if attention
               else (boundary - score) / boundary
               for score, attention in zip(decimal_scores, predicted, strict=True)]

    def point(cutoff: float | None) -> dict[str, Any]:
        minimum = Fraction(str(cutoff)) if cutoff is not None else None
        accepted = {"true_positive": 0, "true_negative": 0,
                    "false_positive": 0, "false_negative": 0}
        abstained = dict(accepted)
        for label, attention, margin in zip(labels, predicted, margins, strict=True):
            # None is an explicit abstain-all control, including score endpoints.
            bucket = accepted if minimum is not None and margin >= minimum else abstained
            prefix = "true" if attention == bool(label) else "false"
            suffix = "positive" if attention else "negative"
            bucket[f"{prefix}_{suffix}"] += 1
        count = sum(accepted.values())
        errors = accepted["false_positive"] + accepted["false_negative"]
        accepted_positive = accepted["true_positive"] + accepted["false_negative"]
        accepted_negative = accepted["true_negative"] + accepted["false_positive"]
        return {
            "cutoff": cutoff, "accepted_rows": count, "abstained_rows": rows - count,
            "coverage": proportion(count, rows), "abstention": proportion(rows - count, rows),
            "accepted_errors": errors, "accepted_error_rate": proportion(errors, count),
            "accepted_confusion": accepted,
            "accepted_class_support": {"positive_rows": accepted_positive,
                                       "negative_rows": accepted_negative},
            "positive_coverage": proportion(accepted_positive, positive),
            "negative_coverage": proportion(accepted_negative, negative),
            "accepted_predicted_attention_rows": accepted["true_positive"] + accepted["false_positive"],
            "accepted_predicted_clear_rows": accepted["true_negative"] + accepted["false_negative"],
            "abstained_confusion": abstained,
            "abstained_errors": abstained["false_positive"] + abstained["false_negative"],
        }

    return {
        "schema": "szl-oac/operational-risk-coverage/v1", "evidence_class": "SAMPLE",
        "rows": rows, "decision_threshold": threshold,
        "selection_semantics": "normalized_decision_boundary_distance_not_confidence_not_trust",
        "comparison_protocol": "exact_rationals_of_shortest_decimal_exported_numbers_no_tolerance",
        "cutoff_protocol": "predeclared_grid_inclusive_ties_no_label_based_selection",
        "cutoff_selected": False, "runtime_selection_applied": False,
        "production_promotion_allowed": False,
        "curve": [point(cutoff) for cutoff in RISK_COVERAGE_CUTOFFS],
        "abstain_all_control": point(None),
        "limitations": [
            "synthetic_reused_public_examples_not_new_blind_or_production_validation",
            "distance_is_not_calibrated_uncertainty_or_a_correctness_probability",
            "risk_need_not_decrease_as_coverage_falls",
            "pointwise_wilson_intervals_not_simultaneous_or_selected_policy_guarantees",
            "zero_accepted_rows_have_undefined_risk_not_zero_risk",
            "no_best_cutoff_selected_and_no_runtime_abstention_policy_installed",
        ],
    }


def calculate_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float,
                      train_prevalence: float) -> dict[str, Any]:
    def unit_interval(value: Any) -> bool:
        # Compare before math.isfinite: huge Python integers must be rejected,
        # not raise OverflowError during an implicit float conversion.
        return type(value) in (float, int) and 0 <= value <= 1 and math.isfinite(value)

    if (not labels or len(labels) != len(scores)
            or any(type(label) is not int or label not in (0, 1) for label in labels)
            or any(not unit_interval(score) for score in scores)
            or not unit_interval(threshold) or not 0 < threshold < 1
            or not unit_interval(train_prevalence)):
        raise EvaluationError("INVALID_METRIC_INPUTS")
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for label, score in zip(labels, scores, strict=True):
        prefix = "true" if (score >= threshold) == bool(label) else "false"
        suffix = "positive" if score >= threshold else "negative"
        counts[f"{prefix}_{suffix}"] += 1
    tp, tn, fp, fn = (counts[key] for key in
                      ("true_positive", "true_negative", "false_positive", "false_negative"))
    rows = len(labels)
    brier = sum((score - label)**2 for label, score in zip(labels, scores, strict=True)) / rows
    baseline = sum((train_prevalence - label)**2 for label in labels) / rows
    positive, negative = sum(labels), rows - sum(labels)
    # Tie-aware Mann-Whitney area, with undefined one-class AUC represented as null.
    rank_sum = 0.0
    ordered = sorted(zip(scores, labels, strict=True))
    start = 0
    while start < rows:
        end = start + 1
        while end < rows and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += ((start + 1 + end) / 2) * sum(label for _, label in ordered[start:end])
        start = end
    auc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative) if positive and negative else None
    bins = []
    ece = 0.0
    for index in range(10):
        members = [(label, score) for label, score in zip(labels, scores, strict=True)
                   if min(int(score * 10), 9) == index]
        count = len(members)
        average = sum(score for _, score in members) / count if count else None
        observed = proportion(sum(label for label, _ in members), count)
        if count:
            ece += count / rows * abs(average - sum(label for label, _ in members) / count)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "upper_inclusive": index == 9, "rows": count,
                     "mean_score": round(average, 12) if average is not None else None,
                     "observed_positive_rate": observed})
    return {
        "rows": rows, "positive_rows": positive, "negative_rows": negative,
        "confusion": counts, "accuracy": proportion(tp + tn, rows),
        "precision": proportion(tp, tp + fp), "recall": proportion(tp, tp + fn),
        "specificity": proportion(tn, tn + fp),
        "balanced_accuracy": round((tp / positive + tn / negative) / 2, 12) if positive and negative else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 12) if 2 * tp + fp + fn else None,
        "roc_auc": round(auc, 12) if auc is not None else None,
        "brier_score": round(brier, 12),
        "train_prevalence": round(train_prevalence, 12),
        "train_prevalence_baseline_brier": round(baseline, 12),
        "brier_skill_vs_train_prevalence": round(1 - brier / baseline, 12) if baseline else None,
        "calibration_bins": bins, "expected_calibration_error": round(ece, 12),
    }


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260926
COMPARISON_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc")


def _point_metrics(labels: Sequence[int], predicted: Sequence[bool], ranking: Sequence[float]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(labels, predicted, strict=True) if y and p)
    fp = sum(1 for y, p in zip(labels, predicted, strict=True) if not y and p)
    fn = sum(1 for y, p in zip(labels, predicted, strict=True) if y and not p)
    tn = len(labels) - tp - fp - fn
    positive, negative = tp + fn, tn + fp
    rank_sum, start = 0.0, 0
    ordered = sorted(zip(ranking, labels, strict=True))
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += ((start + 1 + end) / 2) * sum(label for _, label in ordered[start:end])
        start = end
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (tp / positive + tn / negative) / 2 if positive and negative else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positive if positive else None,
        "specificity": tn / negative if negative else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "roc_auc": (rank_sum - positive * (positive + 1) / 2) / (positive * negative) if positive and negative else None,
        "confusion": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
    }


def _percentile_interval(values: Sequence[float]) -> dict[str, Any] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    lower = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    upper = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return {"lower": round(lower, 12), "upper": round(upper, 12), "defined_replicates": len(ordered),
            "excludes_zero": lower > 0 or upper < 0}


def compare_against_references(labels: Sequence[int], scores: Sequence[float], threshold: float,
                               consecutive_failures: Sequence[float], oracle_probability: Sequence[float],
                               train_majority_positive: bool) -> dict[str, Any]:
    """Place the fixed model between two floors and the generator's Bayes-optimal ceiling.

    Floors: the training-majority constant and the rule ``consecutive_failures > 0``.
    Ceiling: the published generator's own label probability, which is the Bayes-optimal
    scorer for this synthetic distribution. The ceiling is reported threshold-free only
    (ROC AUC, Brier), so no threshold is ever selected on the test split.
    Intervals are paired percentile bootstrap over test rows with a fixed seed.
    """
    import random

    rows = len(labels)
    if not (rows == len(scores) == len(consecutive_failures) == len(oracle_probability)) or not 0 < sum(labels) < rows:
        raise EvaluationError("INVALID_COMPARISON_INPUTS")
    constant = 1.0 if train_majority_positive else 0.0
    predictors = {
        "train_majority_constant": ([bool(constant)] * rows, [constant] * rows),
        "rule_consecutive_failures_gt_0": ([value > 0 for value in consecutive_failures],
                                           [1.0 if value > 0 else 0.0 for value in consecutive_failures]),
        "fixed_model": ([score >= threshold for score in scores], list(scores)),
    }
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {name: {metric: [] for metric in COMPARISON_METRICS} for name in predictors}
    differences = {f"fixed_model_minus_{name}": {metric: [] for metric in COMPARISON_METRICS}
                   for name in predictors if name != "fixed_model"}
    ceiling_gap: list[float] = []
    usable = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        index = [rng.randrange(rows) for _ in range(rows)]
        resampled = [labels[i] for i in index]
        if not 0 < sum(resampled) < rows:
            continue
        usable += 1
        replicate = {name: _point_metrics(resampled, [pred[i] for i in index], [rank[i] for i in index])
                     for name, (pred, rank) in predictors.items()}
        for name, values in replicate.items():
            for metric in COMPARISON_METRICS:
                if values[metric] is not None:
                    samples[name][metric].append(values[metric])
        for key, bucket in differences.items():
            other = replicate[key.removeprefix("fixed_model_minus_")]
            for metric in COMPARISON_METRICS:
                if replicate["fixed_model"][metric] is not None and other[metric] is not None:
                    bucket[metric].append(replicate["fixed_model"][metric] - other[metric])
        oracle_auc = _point_metrics(resampled, [False] * rows, [oracle_probability[i] for i in index])["roc_auc"]
        ceiling_gap.append(oracle_auc - replicate["fixed_model"]["roc_auc"])
    point = {name: _point_metrics(labels, pred, rank) for name, (pred, rank) in predictors.items()}
    oracle_point_auc = _point_metrics(labels, [False] * rows, oracle_probability)["roc_auc"]
    model_point_auc = point["fixed_model"]["roc_auc"]

    def rounded(value: float | None) -> float | None:
        return round(value, 12) if value is not None else None

    return {
        "method": "paired_percentile_bootstrap_over_fixed_public_test_rows",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "usable_replicates": usable,
        "bootstrap_seed": BOOTSTRAP_SEED, "decision_threshold": threshold,
        "predictors": {
            name: {"confusion": point[name]["confusion"],
                   "metrics": {metric: {"value": rounded(point[name][metric]),
                                        "interval_95": _percentile_interval(samples[name][metric])}
                               for metric in COMPARISON_METRICS}}
            for name in predictors
        },
        "paired_differences": {
            key: {metric: {"value": rounded(point["fixed_model"][metric] - point[key.removeprefix("fixed_model_minus_")][metric])
                           if point["fixed_model"][metric] is not None
                           and point[key.removeprefix("fixed_model_minus_")][metric] is not None else None,
                           "interval_95": _percentile_interval(bucket[metric])}
                  for metric in COMPARISON_METRICS}
            for key, bucket in differences.items()
        },
        "generator_bayes_optimal_ceiling": {
            "source": "hash_verified_trainer_synthetic_label_probability",
            "roc_auc": rounded(oracle_point_auc),
            "brier_score": rounded(sum((p - y) ** 2 for p, y in zip(oracle_probability, labels, strict=True)) / rows),
            "fixed_model_roc_auc": rounded(model_point_auc),
            "roc_auc_headroom": rounded(oracle_point_auc - model_point_auc),
            "roc_auc_headroom_interval_95": _percentile_interval(ceiling_gap),
            "scope": "reported_threshold_free_only_no_threshold_selected_on_test",
        },
    }


def _trainer_namespace(trainer_source: bytes) -> dict[str, Any]:
    """Execute the hash-verified trainer bytes (not a re-read) in an isolated namespace."""
    namespace: dict[str, Any] = {"__name__": "oac_operational_trainer_snapshot",
                                 "__file__": str(PROJECT_ROOT / TRAINER_FILE)}
    try:
        exec(compile(trainer_source, TRAINER_FILE, "exec"), namespace)  # noqa: S102 - verified local bytes
    except (SyntaxError, ImportError) as exc:
        raise EvaluationError("GENERATOR_LABEL_RULE_UNAVAILABLE") from exc
    return namespace


def _generator_label_probability(trainer_source: bytes):
    """Return the trainer's label rule from the hash-verified trainer bytes."""
    rule = _trainer_namespace(trainer_source).get("_synthetic_label_probability")
    if not callable(rule):
        raise EvaluationError("GENERATOR_LABEL_RULE_UNAVAILABLE")
    return rule


FRESH_SEED = 7500
FRESH_ROWS = 2000
FRESH_BOOTSTRAP_REPLICATES = 400


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _shift_lanes() -> dict[str, Any]:
    """Each lane maps (features, rng) -> features and (features, base probability) -> probability.

    Covariate lanes move the feature distribution and keep the generator's P(label | features);
    the concept lane keeps features and changes P(label | features). All values stay in range.
    """
    def same(features, rng):
        return features

    def queue_saturation(features, rng):
        return {**features, "queue_utilization": round(min(1.0, 0.5 + features["queue_utilization"]), 6)}

    def long_outage_tail(features, rng):
        return {**features, "seconds_since_last_success":
                round(min(86400.0, 24.0 * features["seconds_since_last_success"]), 6)}

    def fault_heavy(features, rng):
        failures = min(20, 3 * features["consecutive_failures"] + (2 if rng.random() < 0.3 else 0))
        valid = 0 if rng.random() < 0.2 else features["configuration_valid"]
        return {**features, "consecutive_failures": failures, "configuration_valid": valid}

    def base_rule(features, probability):
        return probability

    def tls_weight_tripled(features, probability):
        extra = 2 * 1.15 * (1 - features["tls_enabled"])
        return 1.0 / (1.0 + math.exp(-(_logit(probability) + extra)))

    return {
        "fresh_seed_in_distribution": ("none", same, base_rule),
        "covariate_queue_saturation": ("covariate", queue_saturation, base_rule),
        "covariate_long_outage_tail": ("covariate", long_outage_tail, base_rule),
        "covariate_fault_heavy": ("covariate", fault_heavy, base_rule),
        "concept_tls_weight_tripled": ("concept", same, tls_weight_tripled),
    }


def fresh_seed_lanes(trainer_source: bytes, kernel: Any, threshold: float, rows: int = FRESH_ROWS,
                     replicates: int = FRESH_BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    """Score the fixed model on rows drawn from the verified generator with a seed never used in development.

    The fixed model, threshold and weights are not changed. The oracle in every lane is that lane's own
    label probability, so ``roc_auc_headroom`` is the loss attributable to the model rather than to noise.
    """
    import random

    namespace = _trainer_namespace(trainer_source)
    generate, rule = namespace.get("_generate_features"), namespace.get("_synthetic_label_probability")
    if not callable(generate) or not callable(rule):
        raise EvaluationError("GENERATOR_LABEL_RULE_UNAVAILABLE")
    lanes: dict[str, Any] = {}
    for offset, (name, (kind, move, relabel)) in enumerate(_shift_lanes().items()):
        rng = random.Random(FRESH_SEED + offset)
        features, labels, oracle = [], [], []
        for _ in range(rows):
            row = move(generate(rng), rng)
            probability = relabel(row, rule(row))
            features.append(row)
            oracle.append(probability)
            labels.append(int(rng.random() < probability))
        try:
            scores = [kernel.score(row)["operator_attention_score"] for row in features]
        except (OperationalModelError, ValueError, OverflowError) as exc:
            raise EvaluationError("FRESH_LANE_SCORING_FAILED") from exc
        if not 0 < sum(labels) < rows:
            raise EvaluationError("FRESH_LANE_SINGLE_CLASS")
        predicted = [score >= threshold for score in scores]
        point = _point_metrics(labels, predicted, scores)
        oracle_point = _point_metrics(labels, [value >= threshold for value in oracle], oracle)
        oracle_auc = _point_metrics(labels, [False] * rows, oracle)["roc_auc"]
        boot_rng = random.Random(BOOTSTRAP_SEED + offset)
        auc_samples, balanced_samples, gap_samples = [], [], []
        for _ in range(replicates):
            index = [boot_rng.randrange(rows) for _ in range(rows)]
            resampled = [labels[i] for i in index]
            if not 0 < sum(resampled) < rows:
                continue
            replicate = _point_metrics(resampled, [predicted[i] for i in index], [scores[i] for i in index])
            auc_samples.append(replicate["roc_auc"])
            balanced_samples.append(replicate["balanced_accuracy"])
            gap_samples.append(_point_metrics(resampled, [False] * rows, [oracle[i] for i in index])["roc_auc"]
                               - replicate["roc_auc"])
        prevalence = sum(labels) / rows
        lanes[name] = {
            "shift_kind": kind, "seed": FRESH_SEED + offset, "rows": rows,
            "positive_rows": sum(labels), "prevalence": round(prevalence, 12),
            "confusion": point["confusion"],
            "accuracy": round(point["accuracy"], 12),
            "negative_constant_accuracy": round(1 - prevalence, 12),
            "balanced_accuracy": {"value": round(point["balanced_accuracy"], 12),
                                  "interval_95": _percentile_interval(balanced_samples)},
            "roc_auc": {"value": round(point["roc_auc"], 12), "interval_95": _percentile_interval(auc_samples)},
            "flag_rate": round(sum(predicted) / rows, 12),
            "lane_oracle_roc_auc": round(oracle_auc, 12),
            "lane_oracle_at_threshold": {
                "flag_rate": round(sum(value >= threshold for value in oracle) / rows, 12),
                "balanced_accuracy": round(oracle_point["balanced_accuracy"], 12)
                if oracle_point["balanced_accuracy"] is not None else None,
                "reading": "the same fixed threshold applied to the lane's true label probability",
            },
            "roc_auc_headroom": {"value": round(oracle_auc - point["roc_auc"], 12),
                                 "interval_95": _percentile_interval(gap_samples)},
            "risk_coverage": calculate_risk_coverage(labels, scores, threshold),
        }
    return {
        "method": "fresh_rows_from_hash_verified_generator_seed_never_used_for_training_or_selection",
        "bootstrap_replicates": replicates, "decision_threshold": threshold,
        "lanes": lanes,
        "scope": "synthetic_only_fresh_seed_is_public_once_published_do_not_tune_against_it",
    }


def verify_receipt_metrics(receipt_metrics: Any, metrics: dict[str, Any]) -> None:
    """Recompute rather than trust existing published summary metrics.

    Scores exported by the inference kernel are rounded to 12 decimal places.
    The 1e-10 tolerance allows that rounding without concealing material drift.
    Training/validation metrics and log loss are not independently re-evaluated.
    """
    if not isinstance(receipt_metrics, dict) or not isinstance(receipt_metrics.get("test"), dict):
        raise EvaluationError("RECEIPT_METRICS_MISMATCH")
    published = receipt_metrics["test"]
    for field in ("rows", "positive_rows", "negative_rows", "confusion"):
        # Canonical JSON also distinguishes booleans from integer counts.
        if canonical_bytes(published.get(field)) != canonical_bytes(metrics[field]):
            raise EvaluationError("RECEIPT_METRICS_MISMATCH")
    for field in ("accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"):
        observed = metrics[field]["value"] if isinstance(metrics[field], dict) else metrics[field]
        claimed = published.get(field)
        if (type(claimed) not in (float, int) or not 0 <= claimed <= 1
                or not math.isclose(claimed, observed, rel_tol=0.0, abs_tol=1e-10)):
            raise EvaluationError("RECEIPT_METRICS_MISMATCH")


def check_admission(manifest: dict[str, Any]) -> dict[str, Any]:
    required = ADMISSION_HASH_FIELDS | {"schema", "data_class", "contains_phi",
                                       "contains_clinical_results", "authorized_for_evaluation"}
    require_fields(manifest, required, "ADMISSION_SCHEMA")
    if (manifest["schema"] != ADMISSION_SCHEMA
            or manifest["data_class"] != "nonsensitive_operational_only"
            or manifest["contains_phi"] is not False
            or manifest["contains_clinical_results"] is not False
            or manifest["authorized_for_evaluation"] is not True):
        raise EvaluationError("ADMISSION_DECLARATIONS")
    if any(not isinstance(manifest[key], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key])
           for key in ADMISSION_HASH_FIELDS):
        raise EvaluationError("ADMISSION_EVIDENCE_REFERENCE")
    return {
        "status": "MANIFEST_VALID_EVIDENCE_UNVERIFIED", "evidence_verified": False,
        "external_corpus_admitted": False, "production_promotion_allowed": False,
        "blockers": ["evidence_contents_not_verified", "corpus_bytes_not_inspected",
                     "independent_approval_not_verified"],
    }


def boundary() -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "evaluation_scope": "public_fixed_seed_synthetic_test_reused_not_production_validation",
        "production_promotion_allowed": False, "clinical_use_authorized": False,
        "training_rerun": False, "threshold_tuned": False, "weights_updated": False,
        "source_signature_verified": False,
    }


def evaluate(admission_path: Path | None = None, fresh_lanes: bool = False) -> dict[str, Any]:
    relative_paths = [MODEL_FILE, MODEL_RECEIPT_FILE, DATASET_RECEIPT_FILE, KERNEL_FILE,
                      TRAINER_FILE, SCHEMA_FILE, "tools/evaluate_operational_health_model.py"]
    relative_paths += [f"operational-model/data/{split}.jsonl" for split in SPLIT_ROWS]
    snapshots = {relative: read_bounded(PROJECT_ROOT / relative) for relative in relative_paths}
    hashes = {relative: digest(raw) for relative, raw in snapshots.items()}
    if hashes[MODEL_FILE] != FIXED_MODEL_SHA256 or hashes["operational-model/data/test.jsonl"] != FIXED_TEST_SHA256:
        raise EvaluationError("FIXED_BASELINE_DRIFT")
    artifact = strict_object(snapshots[MODEL_FILE])
    receipt = strict_object(snapshots[MODEL_RECEIPT_FILE])
    dataset = strict_object(snapshots[DATASET_RECEIPT_FILE])
    strict_object(snapshots[SCHEMA_FILE])
    require_fields(receipt, {"schema", "purpose", "authority", "model_sha256", "kernel_sha256",
                            "generator_sha256", "dataset_receipt_sha256", "seed", "metrics_scope", "metrics"},
                   "MODEL_RECEIPT_SCHEMA")
    require_fields(dataset, {"schema", "purpose", "authority", "contains_phi", "contains_clinical_results",
                            "generated_data", "generator_sha256", "schema_sha256", "seed", "split_rows", "files"},
                   "DATASET_RECEIPT_SCHEMA")
    if (receipt["schema"] != RECEIPT_SCHEMA or receipt["purpose"] != MODEL_PURPOSE
            or receipt["authority"] != MODEL_AUTHORITY or receipt["seed"] != 2500
            or receipt["metrics_scope"] != "fixed_seed_synthetic_splits_only_not_production_validation"
            or dataset["schema"] != "szl-oac/transport-health-dataset-receipt/v1"
            or dataset["purpose"] != "synthetic_operational_transport_observability_only"
            or dataset["authority"] != MODEL_AUTHORITY or dataset["seed"] != 2500
            or dataset["contains_phi"] is not False or dataset["contains_clinical_results"] is not False
            or dataset["generated_data"] != "synthetic_only" or dataset["split_rows"] != SPLIT_ROWS):
        raise EvaluationError("RECEIPT_TRUTH_BOUNDARY")
    for field, relative in {"model_sha256": MODEL_FILE, "kernel_sha256": KERNEL_FILE,
                            "generator_sha256": TRAINER_FILE, "dataset_receipt_sha256": DATASET_RECEIPT_FILE}.items():
        if receipt[field] != hashes[relative]:
            raise EvaluationError("MODEL_RECEIPT_HASH_MISMATCH")
    if dataset["generator_sha256"] != hashes[TRAINER_FILE] or dataset["schema_sha256"] != hashes[SCHEMA_FILE]:
        raise EvaluationError("DATASET_SOURCE_HASH_MISMATCH")
    expected_files = {f"data/{split}.jsonl": hashes[f"operational-model/data/{split}.jsonl"] for split in SPLIT_ROWS}
    if dataset["files"] != expected_files:
        raise EvaluationError("DATASET_SPLIT_HASH_MISMATCH")
    splits: dict[str, list[dict[str, Any]]] = {}
    feature_hashes: dict[str, set[str]] = {}
    for split, count in SPLIT_ROWS.items():
        lines = snapshots[f"operational-model/data/{split}.jsonl"].splitlines()
        if len(lines) != count or any(not line or len(line) > 4096 for line in lines):
            raise EvaluationError("DATASET_SPLIT_SIZE")
        rows = [strict_object(line) for line in lines]
        for index, row in enumerate(rows):
            validate_row(row, split, index)
        splits[split] = rows
        feature_hashes[split] = {digest(canonical_bytes(row["features"])) for row in rows}
    admission = {"status": "NOT_SUPPLIED", "evidence_verified": False,
                 "external_corpus_admitted": False, "production_promotion_allowed": False}
    if admission_path is not None:
        admission_raw = read_bounded(admission_path, maximum=16384)
        admission = {**check_admission(strict_object(admission_raw)), "manifest_sha256": digest(admission_raw)}
    try:
        kernel = OperationalHealthKernel(PROJECT_ROOT / MODEL_FILE, PROJECT_ROOT / MODEL_RECEIPT_FILE)
        advisories = [kernel.score(row["features"]) for row in splits["test"]]
    except (OperationalModelError, OSError, ValueError, OverflowError) as exc:
        raise EvaluationError("KERNEL_EVALUATION_FAILED") from exc
    if artifact["decision_threshold"] != 0.16 or any(
        advisory["decision_threshold"] != 0.16 or not advisory["authority"]
        or any(value is not False for value in advisory["authority"].values()) for advisory in advisories
    ):
        raise EvaluationError("KERNEL_AUTHORITY_OR_THRESHOLD_DRIFT")
    scores = [advisory["operator_attention_score"] for advisory in advisories]
    labels = [int(row["label"]["operator_attention_required"]) for row in splits["test"]]
    # The kernel's decision uses the unrounded score; require agreement with exported score here.
    if any(advisory["operator_attention_required"] != (score >= 0.16)
           for advisory, score in zip(advisories, scores, strict=True)):
        raise EvaluationError("ROUNDED_SCORE_DECISION_AMBIGUITY")
    prevalence = sum(row["label"]["operator_attention_required"] for row in splits["train"]) / SPLIT_ROWS["train"]
    metrics = calculate_metrics(labels, scores, 0.16, prevalence)
    verify_receipt_metrics(receipt["metrics"], metrics)
    label_rule = _generator_label_probability(snapshots[TRAINER_FILE])
    try:
        oracle = [float(label_rule(row["features"])) for row in splits["test"]]
    except (OperationalModelError, ValueError, OverflowError, TypeError) as exc:
        raise EvaluationError("GENERATOR_LABEL_RULE_UNAVAILABLE") from exc
    if any(not (0.0 <= value <= 1.0 and math.isfinite(value)) for value in oracle):
        raise EvaluationError("GENERATOR_LABEL_RULE_UNAVAILABLE")
    comparisons = compare_against_references(
        labels, scores, 0.16, [row["features"]["consecutive_failures"] for row in splits["test"]], oracle,
        2 * sum(row["label"]["operator_attention_required"] for row in splits["train"]) > SPLIT_ROWS["train"])
    observations = [
        {"sample_id": row["sample_id"], "observation_sha256": digest(line),
         "features_sha256": digest(canonical_bytes(row["features"])), "label": bool(label),
         "score": score, "predicted_attention": score >= 0.16}
        for row, line, label, score in zip(splits["test"], snapshots["operational-model/data/test.jsonl"].splitlines(),
                                            labels, scores, strict=True)
    ]
    fresh = fresh_seed_lanes(snapshots[TRAINER_FILE], kernel, 0.16) if fresh_lanes else None
    if any(read_bounded(PROJECT_ROOT / relative) != raw for relative, raw in snapshots.items()):
        raise EvaluationError("INPUT_CHANGED_DURING_EVALUATION")
    return {
        **boundary(), "complete": True, "status": "SYNTHETIC_EVALUATION_COMPLETE",
        "decision_threshold": 0.16, "source_files_sha256": hashes, "split_rows": SPLIT_ROWS,
        "published_test_metrics_checked": ["rows", "positive_rows", "negative_rows", "confusion",
                                           "accuracy", "precision", "recall", "specificity",
                                           "balanced_accuracy", "f1", "roc_auc"],
        "test_feature_overlap": {
            "train": len(feature_hashes["test"] & feature_hashes["train"]),
            "validation": len(feature_hashes["test"] & feature_hashes["validation"]),
            "scope": "exact_numeric_feature_matches_only_not_semantic_or_temporal_leakage_detection",
        },
        "metrics": metrics, "comparisons": comparisons, "observations": observations,
        "risk_coverage": calculate_risk_coverage(labels, scores, 0.16),
        **({"fresh_seed_lanes": fresh} if fresh is not None else {}),
        "external_corpus_admission": admission,
        "limitations": [
            "public_synthetic_test_already_used_for_development_not_new_blind_evaluation",
            "wilson_intervals_assume_independent_bernoulli_trials_not_real_world_guarantees",
            "calibration_bins_and_brier_are_descriptive_not_calibration_certification",
            "hash_receipt_is_not_an_independent_signed_attestation",
            "no_external_corpus_training_or_qualification_performed",
            "bootstrap_intervals_resample_the_fixed_public_test_split_not_new_data",
            "generator_ceiling_is_bayes_optimal_only_for_this_synthetic_distribution",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new JSON receipt file; existing files are never overwritten")
    parser.add_argument("--admission-manifest", type=Path, help="optional declaration contract; not external data")
    parser.add_argument("--fresh-seed-lanes", action="store_true",
                        help="also score fresh-seed and distribution-shift synthetic lanes (slower)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.admission_manifest, fresh_lanes=args.fresh_seed_lanes)
        exit_code = 0
    except (EvaluationError, OSError, UnicodeError, RecursionError) as exc:
        report = {**boundary(), "complete": False, "status": "EVALUATION_FAILED",
                  "error_code": str(exc) if isinstance(exc, EvaluationError) else "EVALUATION_INPUT_FAILURE"}
        exit_code = 2
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
        except OSError:
            failure = {**boundary(), "complete": False, "status": "EVALUATION_FAILED",
                       "error_code": "OUTPUT_NOT_CREATED"}
            print(json.dumps(failure, sort_keys=True), file=sys.stderr)
            return 2
    print(encoded, end="", file=sys.stdout if exit_code == 0 else sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
