from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.evaluation import provider, runner
from frontier.evaluation.core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _score(*, passed: bool) -> dict:
    return {
        "json_valid": passed,
        "schema_valid": passed,
        "decision_match": passed,
        "answer_match": passed,
        "evidence_match": passed,
        "tool_boundary_pass": passed,
        "points": 6 if passed else 0,
        "max_points": 6,
        "parsed": (
            {
                "decision": "ANSWER",
                "answer": "14:30 UTC",
                "evidence_ids": ["E1"],
                "tool_calls": [],
            }
            if passed
            else None
        ),
        "parse_error": None if passed else "request failed",
    }


def _record(*, ok: bool = True, status: int | None = 200) -> dict:
    return {
        "case_id": "case",
        "category": "test",
        "route": "baseline",
        "requested_model": "baseline",
        "request_sha256": "a" * 64,
        "latency_ms": 12.5,
        "http_status": status,
        "response_headers": {},
        "ok": ok,
        "error_type": None if ok else "HTTPError",
        "error_sha256": None if ok else "b" * 64,
        "score": _score(passed=ok),
    }


def _attempts() -> list[dict]:
    return [
        {
            "provider": "baseten",
            "model": "zai-org/GLM-5.3-Flash:baseten",
            "ok": False,
            "http_status": 404,
            "latency_ms": 21.5,
            "error_type": "HTTPError",
            "error_sha256": "c" * 64,
        },
        {
            "provider": "deepinfra",
            "model": "zai-org/GLM-5.3-Flash:deepinfra",
            "ok": False,
            "http_status": 404,
            "latency_ms": 19.5,
            "error_type": "HTTPError",
            "error_sha256": "d" * 64,
        },
    ]


def _fallback() -> dict:
    return {
        "contract": provider.FALLBACK_CONTRACT,
        "pass": True,
        "transport_pass": True,
        "semantic_safety_pass": True,
        "production_authority": "NONE",
        "selected_fallback_source": "DETERMINISTIC_SAFETY_GUARD",
        "selected_fallback_output": dict(provider.SAFE_FALLBACK_OUTPUT),
    }


def test_choose_provider_raises_structured_sanitized_evidence(monkeypatch) -> None:
    attempts = iter(
        [
            _record(ok=False, status=404),
            _record(ok=False, status=503),
        ]
    )
    monkeypatch.setattr(
        provider,
        "execute_case",
        lambda *args, **kwargs: next(attempts),
    )

    with pytest.raises(provider.ProviderUnavailable) as raised:
        provider.choose_provider(
            {"id": "case"},
            model_id="zai-org/GLM-5.3-Flash",
            token="hf_private_value_must_not_appear",
            providers=["baseten", "deepinfra"],
            timeout=30,
        )

    error = raised.value
    assert len(error.attempts) == 2
    assert error.attempts_sha256 == canonical_sha256(list(error.attempts))
    assert error.to_dict() == {
        "state": "UNAVAILABLE",
        "attempt_count": 2,
        "attempts_sha256": error.attempts_sha256,
        "raw_provider_body_recorded": False,
        "credential_value_recorded": False,
    }
    serialized = json.dumps([error.to_dict(), list(error.attempts), str(error)])
    assert "hf_private_value_must_not_appear" not in serialized
    assert "error_summary" not in serialized


def test_runner_seals_negative_receipt_and_exercises_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "hf_private_value_must_not_appear"
    monkeypatch.setenv("HF_INFERENCE_TOKEN", secret)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        runner,
        "verify_source",
        lambda *args, **kwargs: {"pass": True, "files": []},
    )
    monkeypatch.setattr(
        runner,
        "http_json",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True,
            body={
                "runtime": {
                    "openai_compatible_subset": {
                        "model_id": "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF"
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "execute_case",
        lambda *args, **kwargs: _record(),
    )
    selection = provider.ProviderUnavailable(_attempts())
    monkeypatch.setattr(
        runner,
        "choose_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(selection),
    )
    fallback_calls: list[str] = []

    def fake_fallback(*args, **kwargs):
        fallback_calls.append(kwargs["provider"])
        return _fallback()

    monkeypatch.setattr(runner, "exercise_fallback", fake_fallback)

    rc = runner.main(
        [
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "hf-job-provider-unavailable",
            "--source-revision",
            "a" * 40,
            "--suite",
            "smoke",
        ]
    )

    assert rc == 2
    assert fallback_calls == ["baseten"]
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert receipt["decision"] == "HOLD_PROVIDER_UNAVAILABLE"
    assert receipt["production_disposition"] == "HOLD"
    assert receipt["promotion_effect"] == "NONE"
    assert receipt["provider"] == "UNAVAILABLE"
    assert receipt["provider_route"] is None
    assert receipt["provider_attempts"] == _attempts()
    assert receipt["provider_attempts_sha256"] == canonical_sha256(_attempts())
    assert receipt["candidate_metrics"]["status"] == "UNAVAILABLE"
    assert receipt["candidate_metrics"]["case_count"] == 2
    assert receipt["candidate_metrics"]["attempted_case_count"] == 0
    assert receipt["candidate_metrics"]["score_rate"] is None
    assert receipt["comparison"]["status"] == "UNAVAILABLE"
    assert receipt["fallback_qualification"]["production_fallback_qualified"] is True
    assert "provider_route_unavailable" in receipt["violated_invariants"]
    assert summary["decision"] == "HOLD_PROVIDER_UNAVAILABLE"
    assert summary["provider"] == "UNAVAILABLE"
    assert bundle["provider_selection"] == selection.to_dict()
    assert len(bundle["candidate_results"]) == 2
    assert all(row["attempted"] is False for row in bundle["candidate_results"])
    assert all(row["truth_label"] == "UNAVAILABLE" for row in bundle["candidate_results"])
    assert receipt["candidate_results_sha256"] == canonical_sha256(
        bundle["candidate_results"]
    )
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    for path in (tmp_path / "receipt.json", tmp_path / "bundle.json", tmp_path / "summary.json"):
        assert secret not in path.read_text(encoding="utf-8")


def test_provider_unavailable_run_never_calls_publication(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_INFERENCE_TOKEN", "inference-secret")
    monkeypatch.setenv("HF_TOKEN", "publication-secret")
    monkeypatch.setattr(
        runner,
        "verify_source",
        lambda *args, **kwargs: {"pass": True, "files": []},
    )
    monkeypatch.setattr(
        runner,
        "http_json",
        lambda *args, **kwargs: SimpleNamespace(
            ok=True,
            body={
                "runtime": {
                    "openai_compatible_subset": {"model_id": "baseline"}
                }
            },
        ),
    )
    monkeypatch.setattr(runner, "execute_case", lambda *args, **kwargs: _record())
    monkeypatch.setattr(
        runner,
        "choose_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            provider.ProviderUnavailable(_attempts())
        ),
    )
    monkeypatch.setattr(
        runner,
        "exercise_fallback",
        lambda *args, **kwargs: _fallback(),
    )
    monkeypatch.setattr(
        runner,
        "publish",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider-unavailable evidence must not be published")
        ),
    )

    sealed_calls: list[str] = []

    def fake_sealed_count(*, token, dataset_id, **kwargs):
        sealed_calls.append(dataset_id)
        assert token == "publication-secret"
        return {
            "status": "SEALED_COUNT_ONLY",
            "path_in_repo": "runs/SEALED_COUNT.json",
            "total_sealed": 1,
            "days": [{"date": "2026-09-11", "sealed": 1}],
            "production_disposition": "HOLD",
            "promotion_effect": "NONE",
            "counter_sha256": "c" * 64,
        }

    monkeypatch.setattr(runner, "publish_sealed_count", fake_sealed_count)

    rc = runner.main(
        [
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "provider-unavailable-publish-request",
            "--source-revision",
            "e" * 40,
            "--suite",
            "smoke",
            "--publish",
        ]
    )

    assert rc == 2
    assert sealed_calls == ["SZLHOLDINGS/szl-frontier-evaluation-receipts"]
    publication = json.loads((tmp_path / "publication.json").read_text())
    assert publication["status"] == "SEALED_COUNT_ONLY"
    assert publication["production_disposition"] == "HOLD"
    assert publication["promotion_effect"] == "NONE"
    assert publication["sealed_count"]["total_sealed"] == 1
    assert publication["sealed_count"]["path_in_repo"] == "runs/SEALED_COUNT.json"
    assert "provider" not in publication["sealed_count"]
