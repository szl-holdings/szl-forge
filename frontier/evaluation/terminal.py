#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that an evaluation terminated with complete, honest evidence.

A workflow can operate successfully while the candidate remains on HOLD. This
verifier keeps those meanings separate: it accepts runner exit 0 only for a
complete review-required receipt and exit 2 only for an explicit HOLD receipt.
Missing, malformed, inconsistent, or promotion-capable evidence remains a hard
workflow failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .core import canonical_sha256

SUCCESS_DECISION = "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
HOLD_DECISIONS = {"HOLD", "HOLD_PROVIDER_UNAVAILABLE"}
ALLOWED_DECISIONS = {SUCCESS_DECISION, *HOLD_DECISIONS}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def verify_terminal_artifacts(
    output_dir: Path,
    *,
    runner_exit_code: int,
) -> dict[str, Any]:
    """Validate the complete terminal evidence set without provider access."""

    errors: list[str] = []
    paths = {
        "receipt": output_dir / "receipt.json",
        "bundle": output_dir / "bundle.json",
        "summary": output_dir / "summary.json",
    }
    values: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing_{name}")
            continue
        try:
            values[name] = _load(path)
        except Exception as exc:
            errors.append(f"invalid_{name}:{type(exc).__name__}")

    receipt = values.get("receipt")
    bundle = values.get("bundle")
    summary = values.get("summary")
    if not isinstance(receipt, Mapping):
        errors.append("receipt_not_object")
        receipt = {}
    if not isinstance(bundle, Mapping):
        errors.append("bundle_not_object")
        bundle = {}
    if not isinstance(summary, Mapping):
        errors.append("summary_not_object")
        summary = {}

    decision = str(receipt.get("decision") or "")
    _error(errors, decision in ALLOWED_DECISIONS, "unsupported_decision")
    _error(
        errors,
        receipt.get("production_disposition") == "HOLD",
        "production_disposition_not_hold",
    )
    _error(
        errors,
        receipt.get("promotion_effect") == "NONE",
        "promotion_effect_not_none",
    )

    declared_receipt_sha = str(receipt.get("receipt_sha256") or "")
    expected_receipt_sha = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    _error(
        errors,
        declared_receipt_sha == expected_receipt_sha,
        "receipt_sha256_mismatch",
    )
    _error(errors, bundle.get("receipt") == receipt, "bundle_receipt_mismatch")
    _error(
        errors,
        summary.get("receipt_sha256") == declared_receipt_sha,
        "summary_receipt_sha256_mismatch",
    )
    _error(
        errors,
        summary.get("decision") == decision,
        "summary_decision_mismatch",
    )
    _error(
        errors,
        summary.get("production_disposition") == "HOLD",
        "summary_production_disposition_not_hold",
    )

    baseline_results = bundle.get("baseline_results")
    candidate_results = bundle.get("candidate_results")
    _error(
        errors,
        isinstance(baseline_results, list),
        "baseline_results_not_list",
    )
    _error(
        errors,
        isinstance(candidate_results, list),
        "candidate_results_not_list",
    )
    if isinstance(baseline_results, list):
        _error(
            errors,
            receipt.get("baseline_results_sha256")
            == canonical_sha256(baseline_results),
            "baseline_results_sha256_mismatch",
        )
    if isinstance(candidate_results, list):
        _error(
            errors,
            receipt.get("candidate_results_sha256")
            == canonical_sha256(candidate_results),
            "candidate_results_sha256_mismatch",
        )

    # ``completed`` was not present in the first admitted receipt schema. Keep
    # those historical receipts verifiable, but enforce any explicit value.
    completed = receipt.get("completed")
    if decision == SUCCESS_DECISION:
        _error(errors, runner_exit_code == 0, "success_decision_requires_exit_0")
        _error(
            errors,
            completed in (None, True),
            "success_decision_conflicts_with_completed",
        )
    elif decision in HOLD_DECISIONS:
        _error(errors, runner_exit_code == 2, "hold_decision_requires_exit_2")
        _error(
            errors,
            completed in (None, False),
            "hold_decision_conflicts_with_completed",
        )

    if decision == "HOLD_PROVIDER_UNAVAILABLE":
        selection = receipt.get("provider_selection")
        candidate_metrics = receipt.get("candidate_metrics")
        _error(errors, receipt.get("provider") == "UNAVAILABLE", "provider_not_unavailable")
        _error(errors, receipt.get("provider_route") is None, "provider_route_must_be_null")
        _error(
            errors,
            isinstance(selection, Mapping) and selection.get("state") == "UNAVAILABLE",
            "provider_selection_not_unavailable",
        )
        _error(
            errors,
            isinstance(candidate_metrics, Mapping)
            and candidate_metrics.get("status") == "UNAVAILABLE",
            "candidate_metrics_not_unavailable",
        )
        if isinstance(candidate_results, list):
            _error(
                errors,
                all(
                    isinstance(row, Mapping)
                    and row.get("attempted") is False
                    and row.get("truth_label") == "UNAVAILABLE"
                    for row in candidate_results
                ),
                "provider_unavailable_results_not_explicit_non_execution",
            )

    publication_path = output_dir / "publication.json"
    publication = None
    if publication_path.is_file():
        try:
            publication = _load(publication_path)
        except Exception as exc:
            errors.append(f"invalid_publication:{type(exc).__name__}")
        if decision == "HOLD_PROVIDER_UNAVAILABLE":
            _error(
                errors,
                isinstance(publication, Mapping)
                and publication.get("status") == "SKIPPED_PROVIDER_UNAVAILABLE"
                and publication.get("promotion_effect") == "NONE",
                "provider_unavailable_publication_not_skipped",
            )

    return {
        "schema": "szl.frontier.terminal-evidence-verification.v1",
        "ok": not errors,
        "runner_exit_code": runner_exit_code,
        "decision": decision or None,
        "completed": completed,
        "production_disposition": receipt.get("production_disposition"),
        "promotion_effect": receipt.get("promotion_effect"),
        "receipt_sha256": declared_receipt_sha or None,
        "publication_status": (
            publication.get("status")
            if isinstance(publication, Mapping)
            else None
        ),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-exit-code", type=int, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    result = verify_terminal_artifacts(
        args.output_dir,
        runner_exit_code=args.runner_exit_code,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
