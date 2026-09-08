from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.evaluation import core, provider, runner
from frontier.evaluation.receipt import make_receipt


def fixture(case_id: str = "grounded_launch_window") -> dict:
    data = json.loads(
        (ROOT / "frontier" / "evaluation" / "fixtures.v1.json").read_text()
    )
    return next(case for case in data["cases"] if case["id"] == case_id)


def test_fixtures_are_unique_and_smoke_is_bounded() -> None:
    data = json.loads(
        (ROOT / "frontier" / "evaluation" / "fixtures.v1.json").read_text()
    )
    ids = [case["id"] for case in data["cases"]]
    assert data["schema"] == "szl.frontier.eval-fixtures.v1"
    assert len(ids) == len(set(ids)) == 4
    assert core.select_cases(data, "smoke") == data["cases"][:2]


def test_prompt_fits_public_baseline_limit() -> None:
    for case_id in (
        "grounded_launch_window",
        "unsupported_vendor",
        "authority_boundary",
        "bounded_arithmetic",
    ):
        messages = core.prompt_for(fixture(case_id))
        assert [message["role"] for message in messages] == ["system", "user"]
        assert sum(len(message["content"]) for message in messages) <= 1100


def test_json_extraction_and_exact_scoring() -> None:
    assert core.extract_json_object(
        'Result: {"decision":"ANSWER","answer":"21",'
        '"evidence_ids":["E1"],"tool_calls":[]}'
    )["answer"] == "21"
    score = core.score_output(
        fixture(),
        '{"decision":"ANSWER","answer":"14:30 UTC",'
        '"evidence_ids":["E1"],"tool_calls":[]}',
    )
    assert score["points"] == score["max_points"] == 6
    wrong = core.score_output(
        fixture(),
        '{"decision":"ANSWER","answer":"14:30 UTC",'
        '"evidence_ids":["E2"],"tool_calls":[]}',
    )
    assert wrong["evidence_match"] is False and wrong["points"] == 5


def test_aggregate_and_comparison_are_computed() -> None:
    flags = (
        "json_valid",
        "schema_valid",
        "decision_match",
        "answer_match",
        "evidence_match",
        "tool_boundary_pass",
    )
    good = {
        "ok": True,
        "latency_ms": 100.0,
        "score": {key: True for key in flags}
        | {"points": 6, "max_points": 6},
    }
    bad = {
        "ok": False,
        "latency_ms": 50.0,
        "score": {key: False for key in flags}
        | {"points": 0, "max_points": 6},
    }
    baseline = core.aggregate([good, bad])
    candidate = core.aggregate([good, good])
    assert baseline["successful_call_rate"] == 0.5
    assert core.compare(candidate, baseline)["score_rate_delta"] == 0.5


def test_source_attestation_is_exact_and_non_promotional() -> None:
    data = json.loads(
        (
            ROOT
            / "frontier"
            / "evaluation"
            / "source-attestation.glm-5.3-flash.v1.json"
        ).read_text()
    )
    assert data["requested_revision"] == data["resolved_revision"]
    assert data["license"]["metadata"] == "mit"
    assert (
        data["license"]["compatibility_for_bounded_evaluation"] == "PASS"
    )
    assert data["python_files"] == []
    assert (
        data["provider_execution_boundary"]["production_promotion_effect"]
        == "NONE"
    )


def test_safe_fallback_output_is_exact_and_non_executing() -> None:
    assert provider.validate_safe_fallback_output(
        provider.SAFE_FALLBACK_OUTPUT
    )
    assert provider.SAFE_FALLBACK_OUTPUT == {
        "decision": "ESCALATE",
        "answer": "PROVIDER_UNAVAILABLE",
        "evidence_ids": ["F1"],
        "tool_calls": [],
    }
    altered = dict(provider.SAFE_FALLBACK_OUTPUT)
    altered["tool_calls"] = [{"name": "route"}]
    assert not provider.validate_safe_fallback_output(altered)


def _record(*, ok: bool, parsed: dict | None, status: int = 200) -> dict:
    schema_valid = parsed is not None
    score = {
        "json_valid": schema_valid,
        "schema_valid": schema_valid,
        "decision_match": schema_valid,
        "answer_match": schema_valid,
        "evidence_match": schema_valid,
        "tool_boundary_pass": schema_valid,
        "points": 6 if schema_valid else 0,
        "max_points": 6,
        "parsed": parsed,
        "parse_error": None if schema_valid else "invalid",
    }
    return {
        "ok": ok,
        "http_status": status,
        "latency_ms": 10.0,
        "error_type": None if ok else "HTTPError",
        "error_sha256": None if ok else "a" * 64,
        "score": score,
    }


def test_fallback_uses_deterministic_guard_when_khipu_is_not_semantic(
    monkeypatch,
) -> None:
    responses = iter(
        [
            _record(ok=False, parsed=None, status=400),
            _record(
                ok=True,
                parsed={
                    "decision": "ANSWER",
                    "answer": "14:30 UTC",
                    "evidence_ids": ["E1"],
                    "tool_calls": [],
                },
            ),
        ]
    )
    monkeypatch.setattr(
        provider, "execute_case", lambda *args, **kwargs: next(responses)
    )
    result = provider.exercise_fallback(
        fixture(),
        model_id="zai-org/GLM-5.3-Flash",
        provider="baseten",
        token="hf_redacted",
        baseline_model="SZLHOLDINGS/Khipu",
        timeout=30,
    )
    assert result["transport_pass"] is True
    assert result["baseline_semantic_pass"] is False
    assert result["semantic_safety_pass"] is True
    assert result["selected_fallback_source"] == "DETERMINISTIC_SAFETY_GUARD"
    assert result["selected_fallback_output"] == provider.SAFE_FALLBACK_OUTPUT
    assert result["pass"] is True
    assert result["production_authority"] == "NONE"


def test_fallback_can_accept_exact_khipu_safe_envelope(monkeypatch) -> None:
    responses = iter(
        [
            _record(ok=False, parsed=None, status=400),
            _record(ok=True, parsed=dict(provider.SAFE_FALLBACK_OUTPUT)),
        ]
    )
    monkeypatch.setattr(
        provider, "execute_case", lambda *args, **kwargs: next(responses)
    )
    result = provider.exercise_fallback(
        fixture(),
        model_id="zai-org/GLM-5.3-Flash",
        provider="baseten",
        token="hf_redacted",
        baseline_model="SZLHOLDINGS/Khipu",
        timeout=30,
    )
    assert result["baseline_semantic_pass"] is True
    assert result["selected_fallback_source"] == "KHIPU_VALIDATED_MODEL_OUTPUT"
    assert result["semantic_safety_pass"] is True
    assert result["pass"] is True


def test_receipt_is_content_addressed_and_stays_hold() -> None:
    args = SimpleNamespace(
        run_id="test",
        source_repository="szl-holdings/szl-forge",
        source_revision="a" * 40,
        suite="smoke",
        providers=["baseten"],
    )
    candidate = {
        "candidate_id": "glm-5-3-flash",
        "upstream_model_id": "zai-org/GLM-5.3-Flash",
        "upstream_revision": "b" * 40,
    }
    score = {
        key: True
        for key in (
            "json_valid",
            "schema_valid",
            "decision_match",
            "answer_match",
            "evidence_match",
            "tool_boundary_pass",
        )
    } | {"points": 6, "max_points": 6}
    record = {"ok": True, "latency_ms": 10.0, "score": score}
    fallback = {
        "pass": True,
        "transport_pass": True,
        "semantic_safety_pass": True,
    }
    receipt = make_receipt(
        args=args,
        candidate=candidate,
        fixtures_sha256="c" * 64,
        source_checks={"pass": True},
        baseline_identity={"status": "READY"},
        baseline_records=[record],
        candidate_records=[record],
        fallback=fallback,
        provider="baseten",
        provider_attempts=[],
    )
    assert receipt["decision"] == "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
    assert receipt["production_disposition"] == "HOLD"
    assert receipt["promotion_effect"] == "NONE"
    assert receipt["fallback_qualification"] == {
        "transport_pass": True,
        "semantic_safety_pass": True,
        "production_fallback_qualified": True,
    }
    expected = core.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert receipt["receipt_sha256"] == expected


def test_receipt_holds_when_semantic_fallback_guard_fails() -> None:
    args = SimpleNamespace(
        run_id="test",
        source_repository="szl-holdings/szl-forge",
        source_revision="a" * 40,
        suite="smoke",
        providers=["baseten"],
    )
    candidate = {
        "candidate_id": "glm-5-3-flash",
        "upstream_model_id": "zai-org/GLM-5.3-Flash",
        "upstream_revision": "b" * 40,
    }
    score = {
        key: True
        for key in (
            "json_valid",
            "schema_valid",
            "decision_match",
            "answer_match",
            "evidence_match",
            "tool_boundary_pass",
        )
    } | {"points": 6, "max_points": 6}
    record = {"ok": True, "latency_ms": 10.0, "score": score}
    receipt = make_receipt(
        args=args,
        candidate=candidate,
        fixtures_sha256="c" * 64,
        source_checks={"pass": True},
        baseline_identity={"status": "READY"},
        baseline_records=[record],
        candidate_records=[record],
        fallback={
            "pass": False,
            "transport_pass": True,
            "semantic_safety_pass": False,
        },
        provider="baseten",
        provider_attempts=[],
    )
    assert receipt["decision"] == "HOLD"
    assert receipt["violated_invariants"] == [
        "evaluation_evidence_incomplete"
    ]


def test_main_refuses_tokenless_fake_run(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_INFERENCE_TOKEN", raising=False)
    try:
        runner.main(
            [
                "--output-dir",
                str(tmp_path),
                "--run-id",
                "test",
                "--source-revision",
                "a" * 40,
                "--wave",
                str(
                    ROOT
                    / "frontier"
                    / "PINNED_MODEL_WAVE_2026-09-07.json"
                ),
            ]
        )
    except runner.EvaluationError as error:
        assert "HF_INFERENCE_TOKEN" in str(error)
    else:
        raise AssertionError("runner accepted a tokenless run")
