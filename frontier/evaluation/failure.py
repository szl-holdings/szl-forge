# SPDX-License-Identifier: Apache-2.0
"""Deterministic negative evidence for unavailable candidate providers.

A provider-selection failure is an observed evaluation outcome, not an excuse to
terminate without a receipt. This module converts the bounded provider attempts
and the unexecuted fixture set into content-addressed evidence while keeping all
candidate quality, latency, runtime, and hardware claims explicitly UNAVAILABLE.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .core import canonical_sha256, prompt_for
from .receipt import make_receipt


def _unavailable_score() -> dict[str, Any]:
    return {
        "json_valid": False,
        "schema_valid": False,
        "decision_match": False,
        "answer_match": False,
        "evidence_match": False,
        "tool_boundary_pass": False,
        "points": 0,
        "max_points": 6,
        "parsed": None,
        "parse_error": "candidate provider unavailable; fixture not executed",
    }


def unavailable_candidate_records(
    cases: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    attempts_sha256: str,
) -> list[dict[str, Any]]:
    """Return one explicit non-execution record for every selected fixture."""

    records: list[dict[str, Any]] = []
    for case in cases:
        request_shape = {
            "model": model_id,
            "messages": prompt_for(case),
            "max_tokens": 96,
            "temperature": 0,
            "stream": False,
        }
        records.append(
            {
                "case_id": case.get("id"),
                "category": case.get("category"),
                "route": "candidate-unavailable",
                "requested_model": model_id,
                "request_sha256": canonical_sha256(request_shape),
                "attempted": False,
                "latency_ms": 0.0,
                "http_status": None,
                "response_headers": {},
                "ok": False,
                "error_type": "ProviderUnavailable",
                "error_sha256": attempts_sha256,
                "error_summary": "all configured provider routes unavailable",
                "truth_label": "UNAVAILABLE",
                "score": _unavailable_score(),
            }
        )
    return records


def _unavailable_metrics(case_count: int) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "case_count": case_count,
        "attempted_case_count": 0,
        "completed_case_count": 0,
        "successful_call_rate": None,
        "json_valid_rate": None,
        "schema_valid_rate": None,
        "decision_match_rate": None,
        "answer_match_rate": None,
        "evidence_match_rate": None,
        "tool_boundary_pass_rate": None,
        "score_rate": None,
        "latency_ms": {"mean": None, "min": None, "max": None},
    }


def make_provider_unavailable_receipt(
    *,
    args: Any,
    candidate: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    fixtures_sha256: str,
    source_checks: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
    fallback: Mapping[str, Any],
    provider_attempts: Sequence[Mapping[str, Any]],
    provider_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seal an honest HOLD receipt when no candidate provider completed case 1."""

    attempts = [dict(attempt) for attempt in provider_attempts]
    attempts_sha256 = canonical_sha256(attempts)
    declared_attempts_sha256 = str(provider_selection.get("attempts_sha256") or "")
    if declared_attempts_sha256 != attempts_sha256:
        raise ValueError("provider selection digest does not bind provider attempts")

    candidate_records = unavailable_candidate_records(
        cases,
        model_id=str(candidate["upstream_model_id"]),
        attempts_sha256=attempts_sha256,
    )
    receipt = make_receipt(
        args=args,
        candidate=candidate,
        fixtures_sha256=fixtures_sha256,
        source_checks=source_checks,
        baseline_identity=baseline_identity,
        baseline_records=baseline_records,
        candidate_records=candidate_records,
        fallback=fallback,
        provider="UNAVAILABLE",
        provider_attempts=attempts,
    )

    receipt["provider"] = "UNAVAILABLE"
    receipt["provider_route"] = None
    receipt["provider_selection"] = dict(provider_selection)
    receipt["provider_attempts_sha256"] = attempts_sha256
    receipt["candidate_results_sha256"] = canonical_sha256(candidate_records)
    receipt["candidate_metrics"] = _unavailable_metrics(len(cases))
    receipt["comparison"] = {
        "status": "UNAVAILABLE",
        "score_rate_delta": None,
        "schema_valid_rate_delta": None,
        "mean_latency_ratio": None,
    }
    receipt["metric_truth_labels"][
        "request_latency_and_response_scoring"
    ] = "UNAVAILABLE"
    receipt["metric_truth_labels"][
        "baseline_request_latency_and_response_scoring"
    ] = "MEASURED"
    receipt["metric_truth_labels"][
        "candidate_request_latency_and_response_scoring"
    ] = "UNAVAILABLE"
    receipt["metric_truth_labels"]["provider_selection_attempts"] = "MEASURED"
    receipt["violated_invariants"] = [
        "provider_route_unavailable",
        "candidate_evaluation_not_executed",
        "evaluation_evidence_incomplete",
    ]
    receipt["known_bounds"] = list(receipt["known_bounds"]) + [
        "no_candidate_provider_route_completed_the_first_fixture",
        "candidate_quality_latency_cost_and_runtime_are_unavailable",
    ]
    receipt["decision"] = "HOLD_PROVIDER_UNAVAILABLE"
    receipt["production_disposition"] = "HOLD"
    receipt["promotion_effect"] = "NONE"
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt, candidate_records
