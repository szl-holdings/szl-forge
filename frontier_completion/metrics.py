"""Reproducible scoring of supplied observations, not an inference engine.

No score here establishes a benchmark dataset's rights, a run's authenticity,
independent calibration, or production admission. Undefined denominators stay null.
"""
from __future__ import annotations

import math
import statistics
from typing import Any
from .core import finite, integer, require, text


def numeric(values: list[float], name: str) -> list[float]:
    require(type(values) is list and 0 < len(values) <= 100000, name + "_shape")
    return [finite(value, name) for value in values]


def forecast_metrics(actual: list[float], prediction: list[float], training: list[float],
                     seasonality: int = 1) -> dict[str, Any]:
    y, p, history = numeric(actual, "actual"), numeric(prediction, "prediction"), numeric(training, "training")
    require(len(y) == len(p), "forecast_length_mismatch")
    integer(seasonality, 1, len(history), "seasonality")
    ae = [abs(a - b) for a, b in zip(y, p, strict=True)]
    mae = statistics.fmean(ae)
    denominator = math.fsum(abs(v) for v in y)
    scale = statistics.fmean(abs(history[i] - history[i - seasonality])
                             for i in range(seasonality, len(history))) if len(history) > seasonality else 0.0
    rmse = math.sqrt(statistics.fmean((a - b)**2 for a, b in zip(y, p, strict=True)))
    for value in (mae, denominator, scale, rmse):
        require(math.isfinite(value), "metric_overflow")
    return {"n": len(y), "mae": mae, "rmse": rmse,
            "wape": math.fsum(ae) / denominator if denominator > 0 else None,
            "mase": mae / scale if scale > 0 else None,
            "wapeState": "MEASURED" if denominator > 0 else "UNDEFINED_ZERO_ACTUAL_SCALE",
            "maseState": "MEASURED" if scale > 0 else "UNDEFINED_TRAINING_SCALE",
            "calibrationEstablished": False}


def quantile_metrics(actual: list[float], quantiles: list[float],
                     predictions: list[list[float]]) -> dict[str, Any]:
    y, qs = numeric(actual, "actual"), numeric(quantiles, "quantiles")
    require(len(qs) <= 99 and qs == sorted(set(qs)) and all(0 < q < 1 for q in qs), "invalid_quantiles")
    require(type(predictions) is list and len(predictions) == len(y), "quantile_horizon_mismatch")
    rows = []
    for row in predictions:
        values = numeric(row, "quantile_prediction")
        require(len(values) == len(qs) and values == sorted(values), "crossing_or_missing_quantiles")
        rows.append(values)
    loss = []
    for j, q in enumerate(qs):
        value = statistics.fmean(max(q * (a - row[j]), (q - 1) * (a - row[j])) for a, row in zip(y, rows, strict=True))
        require(math.isfinite(value), "pinball_overflow")
        loss.append({"q": q, "pinball": value})
    # Discrete quantile score is NOT exact continuous CRPS.
    return {"n": len(y), "perQuantile": loss,
            "meanPinball": statistics.fmean(r["pinball"] for r in loss),
            "intervalCoverage": statistics.fmean(row[0] <= a <= row[-1] for a, row in zip(y, rows, strict=True)) if len(qs) >= 2 else None,
            "meanIntervalWidth": statistics.fmean(row[-1] - row[0] for row in rows) if len(qs) >= 2 else None,
            "nominalIntervalCoverage": qs[-1] - qs[0] if len(qs) >= 2 else None,
            "exactCRPS": None, "calibrationEstablished": False}


def rolling_origins(n: int, minimum_history: int, horizon: int, step: int) -> list[dict[str, int]]:
    integer(n, 2, 10**7, "series_length")
    integer(minimum_history, 1, n, "minimum_history")
    integer(horizon, 1, n, "horizon")
    integer(step, 1, n, "step")
    origins = range(minimum_history, n - horizon + 1, step)
    require(len(origins) <= 100000, "too_many_folds")
    # Exclusive Python slice stops make leakage checks unambiguous.
    return [{"trainStart": 0, "trainStop": i, "testStart": i, "testStop": i + horizon} for i in origins]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    require(type(reference) is list and type(hypothesis) is list, "tokens_must_be_lists")
    require(len(reference) <= 100000 and len(hypothesis) <= 100000, "token_count_bound")
    require(len(reference) * len(hypothesis) <= 4000000, "edit_distance_work_bound")
    require(all(type(t) is str for t in reference + hypothesis), "token_type")
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def speech_error(pairs: list[tuple[str, str]], *, character_level: bool = False) -> dict[str, Any]:
    require(type(pairs) is list and 0 < len(pairs) <= 10000, "invalid_speech_corpus")
    require(type(character_level) is bool, "invalid_character_mode")
    errors, denominator = 0, 0
    for pair in pairs:
        require(type(pair) in (list, tuple) and len(pair) == 2 and all(type(t) is str for t in pair), "invalid_speech_pair")
        reference, hypothesis = pair
        require(len(reference) <= 100000 and len(hypothesis) <= 100000, "speech_text_bound")
        a, b = (list(reference), list(hypothesis)) if character_level else (reference.split(), hypothesis.split())
        errors += edit_distance(a, b)
        denominator += len(a)
    return {"metric": "codepoint_CER" if character_level else "whitespace_WER", "n": len(pairs),
            "edits": errors, "referenceUnits": denominator,
            "value": errors / denominator if denominator else None,
            "state": "MEASURED" if denominator else "UNDEFINED_EMPTY_REFERENCE",
            "normalization": "NONE; score raw supplied text; declare language-specific normalizers separately"}


def ranking_metrics(ranked: list[str], relevance: dict[str, int], k: int) -> dict[str, Any]:
    integer(k, 1, 10000, "k")
    require(type(ranked) is list and len(ranked) <= 10000 and all(type(v) is str for v in ranked), "invalid_ranking")
    require(len(set(ranked)) == len(ranked), "duplicate_ranked_id")
    require(type(relevance) is dict and len(relevance) <= 100000, "invalid_qrels")
    for key, grade in relevance.items():
        text(key, 512); integer(grade, 0, 10, "relevance")
    positive = {key for key, grade in relevance.items() if grade > 0}
    gains = [2**relevance.get(key, 0) - 1 for key in ranked[:k]]
    ideal = sorted((2**grade - 1 for grade in relevance.values()), reverse=True)[:k]
    dcg = math.fsum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = math.fsum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    reciprocal = next((1 / (i + 1) for i, key in enumerate(ranked[:k]) if key in positive), 0.0)
    return {"k": k, "recall": len(set(ranked[:k]) & positive) / len(positive) if positive else None,
            "ndcg": dcg / idcg if idcg > 0 else None, "mrr": reciprocal if positive else None,
            "qrelsState": "MEASURED" if positive else "NO_POSITIVE_JUDGMENTS"}


def paired_cases(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    def index(rows: list[dict[str, Any]]) -> dict[str, bool]:
        require(type(rows) is list and 0 < len(rows) <= 100000, "invalid_case_rows")
        result = {}
        for row in rows:
            require(type(row) is dict and set(row) == {"id", "pass"}, "case_schema")
            key = text(row["id"], 200)
            require(key not in result and type(row["pass"]) is bool, "duplicate_or_unknown_case_result")
            result[key] = row["pass"]
        return result
    a, b = index(baseline), index(candidate)
    require(set(a) == set(b), "unmatched_case_sets")
    regressed = sorted(key for key in a if a[key] and not b[key])
    recovered = sorted(key for key in a if not a[key] and b[key])
    return {"n": len(a), "baselinePassed": sum(a.values()), "candidatePassed": sum(b.values()),
            "regressedIds": regressed, "recoveredIds": recovered,
            "sameCaseOutcomes": not regressed and not recovered,
            "noObservedRegression": not regressed,
            "statisticalNoninferiorityEstablished": False, "productionPromotion": False}


def kl_divergence(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    p, q = numeric(reference, "reference_probability"), numeric(candidate, "candidate_probability")
    require(len(p) == len(q) and all(v >= 0 for v in p + q), "probability_shape")
    require(math.isclose(math.fsum(p), 1.0, abs_tol=1e-8) and math.isclose(math.fsum(q), 1.0, abs_tol=1e-8), "probabilities_not_normalized")
    if any(a > 0 and b == 0 for a, b in zip(p, q, strict=True)):
        return {"state": "INFINITE_DIVERGENCE", "value": None, "units": "nats"}
    score = math.fsum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q, strict=True) if a > 0)
    require(score >= -1e-8 and math.isfinite(score), "invalid_kl")
    return {"state": "MEASURED", "value": max(0.0, score), "units": "nats"}
