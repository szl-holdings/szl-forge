from __future__ import annotations

import json
from pathlib import Path

from frontier.evaluation.core import canonical_sha256
from frontier.evaluation.terminal import verify_terminal_artifacts


def _write_terminal_set(
    root: Path,
    *,
    decision: str,
    completed: bool,
    provider_unavailable: bool = False,
    publication: bool = False,
) -> tuple[dict, dict, dict]:
    baseline_results = [
        {
            "case_id": "case-1",
            "route": "baseline",
            "attempted": True,
            "ok": True,
            "latency_ms": 12.5,
            "truth_label": "MEASURED",
        }
    ]
    candidate_results = [
        {
            "case_id": "case-1",
            "route": (
                "candidate-unavailable"
                if provider_unavailable
                else "candidate"
            ),
            "attempted": not provider_unavailable,
            "ok": False,
            "latency_ms": 0.0 if provider_unavailable else 20.0,
            "truth_label": (
                "UNAVAILABLE" if provider_unavailable else "MEASURED"
            ),
        }
    ]
    receipt = {
        "schema": "szl.nemo.frontier-qualification-receipt.v1",
        "completed": completed,
        "candidate_id": "glm-5-3-flash",
        "model_id": "zai-org/GLM-5.3-Flash",
        "provider": "UNAVAILABLE" if provider_unavailable else "zai-org",
        "provider_route": (
            None
            if provider_unavailable
            else "zai-org/GLM-5.3-Flash:zai-org"
        ),
        "provider_selection": (
            {
                "state": "UNAVAILABLE",
                "attempt_count": 6,
                "attempts_sha256": "a" * 64,
                "raw_provider_body_recorded": False,
                "credential_value_recorded": False,
            }
            if provider_unavailable
            else None
        ),
        "baseline_results_sha256": canonical_sha256(baseline_results),
        "candidate_results_sha256": canonical_sha256(candidate_results),
        "baseline_metrics": {"score_rate": 0.0},
        "candidate_metrics": (
            {
                "status": "UNAVAILABLE",
                "case_count": 1,
                "attempted_case_count": 0,
                "score_rate": None,
            }
            if provider_unavailable
            else {"case_count": 1, "score_rate": 0.5}
        ),
        "comparison": (
            {"status": "UNAVAILABLE", "score_rate_delta": None}
            if provider_unavailable
            else {"score_rate_delta": 0.5}
        ),
        "decision": decision,
        "production_disposition": "HOLD",
        "promotion_effect": "NONE",
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    bundle = {
        "schema": "szl.frontier.evaluation-bundle.v1",
        "receipt": receipt,
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
    }
    summary = {
        "candidate_id": receipt["candidate_id"],
        "model_id": receipt["model_id"],
        "provider": receipt["provider"],
        "suite": "smoke",
        "baseline_metrics": receipt["baseline_metrics"],
        "candidate_metrics": receipt["candidate_metrics"],
        "comparison": receipt["comparison"],
        "decision": decision,
        "production_disposition": "HOLD",
        "receipt_sha256": receipt["receipt_sha256"],
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("receipt.json", receipt),
        ("bundle.json", bundle),
        ("summary.json", summary),
    ):
        (root / name).write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
        )
    if publication:
        (root / "publication.json").write_text(
            json.dumps(
                {
                    "status": "SKIPPED_PROVIDER_UNAVAILABLE",
                    "production_disposition": "HOLD",
                    "promotion_effect": "NONE",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return receipt, bundle, summary


def test_completed_review_required_run_is_operationally_valid(tmp_path: Path) -> None:
    _write_terminal_set(
        tmp_path,
        decision="EVIDENCE_COMPLETE_REVIEW_REQUIRED",
        completed=True,
    )
    result = verify_terminal_artifacts(tmp_path, runner_exit_code=0)
    assert result["ok"] is True
    assert result["decision"] == "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
    assert result["production_disposition"] == "HOLD"
    assert result["promotion_effect"] == "NONE"


def test_content_addressed_hold_exit_two_is_operationally_valid(tmp_path: Path) -> None:
    _write_terminal_set(tmp_path, decision="HOLD", completed=False)
    result = verify_terminal_artifacts(tmp_path, runner_exit_code=2)
    assert result["ok"] is True
    assert result["decision"] == "HOLD"
    assert result["runner_exit_code"] == 2


def test_hold_cannot_be_relabelled_as_runner_exit_zero(tmp_path: Path) -> None:
    _write_terminal_set(tmp_path, decision="HOLD", completed=False)
    result = verify_terminal_artifacts(tmp_path, runner_exit_code=0)
    assert result["ok"] is False
    assert "hold_decision_requires_exit_2" in result["errors"]


def test_tampered_candidate_results_are_rejected(tmp_path: Path) -> None:
    _write_terminal_set(tmp_path, decision="HOLD", completed=False)
    bundle_path = tmp_path / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["candidate_results"][0]["ok"] = True
    bundle_path.write_text(json.dumps(bundle) + "\n", encoding="utf-8")

    result = verify_terminal_artifacts(tmp_path, runner_exit_code=2)
    assert result["ok"] is False
    assert "candidate_results_sha256_mismatch" in result["errors"]


def test_promotion_capable_receipt_is_rejected_even_when_rehashed(
    tmp_path: Path,
) -> None:
    receipt, bundle, summary = _write_terminal_set(
        tmp_path, decision="HOLD", completed=False
    )
    receipt["promotion_effect"] = "PROMOTE"
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    bundle["receipt"] = receipt
    summary["receipt_sha256"] = receipt["receipt_sha256"]
    for name, value in (
        ("receipt.json", receipt),
        ("bundle.json", bundle),
        ("summary.json", summary),
    ):
        (tmp_path / name).write_text(json.dumps(value) + "\n", encoding="utf-8")

    result = verify_terminal_artifacts(tmp_path, runner_exit_code=2)
    assert result["ok"] is False
    assert "promotion_effect_not_none" in result["errors"]


def test_provider_unavailable_requires_explicit_non_execution_and_skip(
    tmp_path: Path,
) -> None:
    _write_terminal_set(
        tmp_path,
        decision="HOLD_PROVIDER_UNAVAILABLE",
        completed=False,
        provider_unavailable=True,
        publication=True,
    )
    result = verify_terminal_artifacts(tmp_path, runner_exit_code=2)
    assert result["ok"] is True
    assert result["publication_status"] == "SKIPPED_PROVIDER_UNAVAILABLE"


def test_provider_unavailable_rejects_fake_attempted_candidate_result(
    tmp_path: Path,
) -> None:
    receipt, bundle, summary = _write_terminal_set(
        tmp_path,
        decision="HOLD_PROVIDER_UNAVAILABLE",
        completed=False,
        provider_unavailable=True,
        publication=True,
    )
    bundle["candidate_results"][0]["attempted"] = True
    receipt["candidate_results_sha256"] = canonical_sha256(
        bundle["candidate_results"]
    )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    bundle["receipt"] = receipt
    summary["receipt_sha256"] = receipt["receipt_sha256"]
    for name, value in (
        ("receipt.json", receipt),
        ("bundle.json", bundle),
        ("summary.json", summary),
    ):
        (tmp_path / name).write_text(json.dumps(value) + "\n", encoding="utf-8")

    result = verify_terminal_artifacts(tmp_path, runner_exit_code=2)
    assert result["ok"] is False
    assert (
        "provider_unavailable_results_not_explicit_non_execution"
        in result["errors"]
    )
