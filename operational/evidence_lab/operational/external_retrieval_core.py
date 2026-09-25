"""Pure, offline metrics for the external-benchmark EVAL-ONLY lane.

No file, network, model, training, or publication access. Corpus positions are
document identities; ties always favor the lower position. Calibration consumes
only its argument: the caller must keep calibration and test partitions disjoint.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import string


def _sequence(values, name, *, nonempty=True):
    if not isinstance(values, (list, tuple)) or (nonempty and not values):
        raise ValueError(f"{name} must be {'a nonempty' if nonempty else 'a'} list/tuple")
    return values


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return value if isinstance(value, int) else result


def _scores(values):
    return [_finite(value, "score") for value in _sequence(values, "scores")]


def stable_order(scores):
    """Return every document position, descending score then ascending position."""
    values = _scores(scores)
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


class BM25:
    """Okapi BM25 with positive log(1 + (N-df+.5)/(df+.5)) IDF.

    Unicode word tokens are lowercased; repeated query tokens contribute repeatedly.
    Empty documents and all-empty corpora produce finite zero scores.
    """

    def __init__(self, texts, k1=1.2, b=0.75):
        _sequence(texts, "texts")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must contain strings")
        self.k1, self.b = _finite(k1, "k1"), _finite(b, "b")
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("k1 must be positive and b must be in [0, 1]")
        self.frequencies = [Counter(re.findall(r"\w+", text.lower())) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.frequencies]
        self.average_length = sum(self.lengths) / len(texts)
        document_frequency = Counter(term for counts in self.frequencies for term in counts)
        self.idf = {term: math.log1p((len(texts) - count + 0.5) / (count + 0.5))
                    for term, count in document_frequency.items()}

    def scores(self, query):
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        tokens = Counter(re.findall(r"\w+", query.lower()))
        result = []
        for counts, length in zip(self.frequencies, self.lengths):
            length_ratio = length / self.average_length if self.average_length else 0.0
            norm = 1.0 - self.b + self.b * length_ratio
            score = 0.0
            for term, query_count in tokens.items():
                frequency = counts[term]
                if not frequency:
                    continue
                # Equivalent forms avoid overflow at both extremes of finite k1.
                if self.k1 >= 1:
                    weight = (1 + 1 / self.k1) / (1 / self.k1 + norm / frequency)
                else:
                    weight = (self.k1 + 1) / (1 + self.k1 * (norm / frequency))
                score += query_count * self.idf[term] * weight
            result.append(_finite(score, "BM25 score"))
        return result


def rrf_fusion(score_lists):
    """Fixed k=60 reciprocal-rank fusion of full, aligned score vectors.

    Every input contains one score per document, so duplicate ranked IDs cannot
    occur. Equal scores are valid and use corpus-position tie breaking.
    """
    vectors = [_scores(scores) for scores in _sequence(score_lists, "score_lists")]
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        raise ValueError("RRF score vectors must have equal lengths")
    fused = [0.0] * size
    for vector in vectors:
        for rank, index in enumerate(stable_order(vector), start=1):
            fused[index] += 1.0 / (60 + rank)
    return fused


reciprocal_rank_fusion = rrf_fusion


def retrieval_metrics(score_rows, positive_index_lists):
    """Macro MRR@10, Recall@1/5/10 and binary nDCG@10 over judged positives.

    Recall uses ALL positives as its denominator, not merely a hit indicator.
    Ideal DCG has the same cutoff as actual DCG. No rounding is applied.
    """
    _sequence(score_rows, "score_rows")
    _sequence(positive_index_lists, "positive_index_lists")
    if len(score_rows) != len(positive_index_lists):
        raise ValueError("scores and positive labels must have equal query counts")
    totals = dict.fromkeys(("mrr_at_10", "recall_at_1", "recall_at_5",
                            "recall_at_10", "ndcg_at_10"), 0.0)
    for scores, positives in zip(score_rows, positive_index_lists):
        order = stable_order(scores)
        _sequence(positives, "positive indices")
        if any(isinstance(index, bool) or not isinstance(index, int)
               or not 0 <= index < len(order) for index in positives):
            raise ValueError("positive indices must be integer corpus positions")
        if len(set(positives)) != len(positives):
            raise ValueError("positive indices must be unique")
        relevant = set(positives)
        hits = [int(index in relevant) for index in order[:10]]
        totals["mrr_at_10"] += next((1.0 / rank for rank, hit in enumerate(hits, 1)
                                      if hit), 0.0)
        for cutoff in (1, 5, 10):
            totals[f"recall_at_{cutoff}"] += sum(hits[:cutoff]) / len(relevant)
        actual = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(relevant)) + 1))
        totals["ndcg_at_10"] += actual / ideal
    return {**{name: value / len(score_rows) for name, value in totals.items()},
            "queries": len(score_rows)}


def normalize_answer(text):
    """SQuAD normalization: lowercase, remove ASCII punctuation/articles, spaces."""
    if not isinstance(text, str):
        raise ValueError("answer must be a string")
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_exact_f1(prediction, goldstrings):
    """Maximum normalized EM/F1 across references; [] denotes an empty answer."""
    predicted = normalize_answer(prediction)
    _sequence(goldstrings, "gold answers", nonempty=False)
    golds = [normalize_answer(gold) for gold in goldstrings] or [""]
    exact, f1 = 0.0, 0.0
    for gold in golds:
        exact = max(exact, float(predicted == gold))
        predicted_tokens, gold_tokens = predicted.split(), gold.split()
        if not predicted_tokens or not gold_tokens:
            current = float(predicted_tokens == gold_tokens)
        else:
            common = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
            current = 2.0 * common / (len(predicted_tokens) + len(gold_tokens))
        f1 = max(f1, current)
    return {"exact_match": exact, "f1": f1}


def _records(records):
    _sequence(records, "records")
    validated = []
    for record in records:
        if not isinstance(record, dict) or not {"margin", "answerable", "prediction", "gold"} <= record.keys():
            raise ValueError("records require margin, answerable, prediction and gold")
        margin = _finite(record["margin"], "margin")
        if not isinstance(record["answerable"], bool):
            raise ValueError("answerable must be a boolean")
        normalize_answer(record["prediction"])
        _sequence(record["gold"], "gold answers", nonempty=False)
        golds = [normalize_answer(gold) for gold in record["gold"]]
        if record["answerable"] != any(golds):
            raise ValueError("answerability must agree with nonempty normalized gold answers")
        if record["answerable"] and not all(golds):
            raise ValueError("answerable records must not include empty normalized gold answers")
        validated.append({**record, "margin": margin})
    return validated


def wilson_interval(successes, total):
    """Two-sided 95% Wilson binomial interval; None when denominator is zero."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (successes, total)):
        raise ValueError("Wilson counts must be integers")
    if not 0 <= successes <= total:
        raise ValueError("Wilson counts require 0 <= successes <= total")
    if not total:
        return None
    z = 1.959963984540054
    proportion, z2 = successes / total, z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z2 / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def answerability_metrics(records, threshold):
    """Answer iff margin > threshold; all rates lie in [0,1], not percentages.

    supported_recall is the selected fraction of answerable rows, not QA recall.
    abstention_accuracy is correct answer/abstain decisions over ALL rows.
    QA scores use an empty prediction for abstentions. Missing conditional
    denominators yield None, never a manufactured measured zero.
    """
    return _answerability_metrics(_records(records), _finite(threshold, "threshold"))


def _answerability_metrics(records, threshold):
    answered = supported = false_answers = correct_decisions = 0
    exact = f1 = 0.0
    positives = sum(record["answerable"] for record in records)
    negatives = len(records) - positives
    for record in records:
        selected = record["margin"] > threshold
        answered += selected
        supported += selected and record["answerable"]
        false_answers += selected and not record["answerable"]
        correct_decisions += selected == record["answerable"]
        qa = qa_exact_f1(record["prediction"] if selected else "", record["gold"])
        exact += qa["exact_match"]
        f1 += qa["f1"]
    return {"records": len(records), "answerable_count": positives,
            "unanswerable_count": negatives, "answered_count": answered,
            "false_answer_count": false_answers, "coverage": answered / len(records),
            "false_answer_rate": false_answers / negatives if negatives else None,
            "supported_recall": supported / positives if positives else None,
            "abstention_accuracy": correct_decisions / len(records),
            "qa_exact_match": exact / len(records), "qa_f1": f1 / len(records),
            "false_answer_rate_wilson_ci": wilson_interval(false_answers, negatives)}


def choose_threshold(records, max_false_answer_rate=0.05):
    """Fit ONLY supplied calibration rows, maximizing all-row QA F1 under FAR cap.

    Ties prefer lower empirical unanswerable false-answer rate, then the higher
    threshold among finite observed boundary candidates. The cap is empirical;
    the reported Wilson interval is NOT a population-rate safety guarantee.
    """
    validated = _records(records)
    limit = _finite(max_false_answer_rate, "max_false_answer_rate")
    if not 0 <= limit <= 1:
        raise ValueError("max_false_answer_rate must be in [0, 1]")
    if {record["answerable"] for record in validated} != {False, True}:
        raise ValueError("calibration requires both answerable and unanswerable records")
    boundaries = sorted({record["margin"] for record in validated})
    below_minimum = math.nextafter(boundaries[0], -math.inf)
    if math.isfinite(below_minimum):
        boundaries.insert(0, below_minimum)
    best = None
    for threshold in boundaries:
        metrics = _answerability_metrics(validated, threshold)
        if metrics["false_answer_rate"] <= limit:
            key = (metrics["qa_f1"], -metrics["false_answer_rate"], threshold)
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
    assert best is not None  # Maximum observed margin always abstains on all rows.
    return {"threshold": best[1], "metrics": best[2],
            "calibration_label": "calibration_only",
            "max_false_answer_rate": limit, "threshold_rule": "margin > threshold",
            "false_answer_constraint": "empirical_calibration_rate_only"}
