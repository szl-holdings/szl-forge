from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.evaluation import core, provider, runner
from frontier.evaluation.receipt import PUBLIC_EXAMPLE, dataset_card, make_receipt, publish


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


PUBLICATION_PATH = "runs/2026/09/08/github-test-1-aaaaaaaaaaaa/glm-5-3-flash"
PUBLICATION_DATASET = "SZLHOLDINGS/szl-frontier-evaluation-receipts"


def _publication_files(output: Path, *, suite: str = "full") -> dict:
    """Use actual receipt construction, with explicitly synthetic local responses."""
    records = [
        _record(ok=True, parsed=dict(provider.SAFE_FALLBACK_OUTPUT))
        for _ in range(4 if suite == "full" else 2)
    ]
    receipt = make_receipt(
        args=SimpleNamespace(
            run_id="github-test-1-aaaaaaaaaaaa",
            source_repository="szl-holdings/szl-forge",
            source_revision="a" * 40,
            suite=suite,
            providers=["baseten"],
        ),
        candidate={
            "candidate_id": "glm-5-3-flash",
            "upstream_model_id": "zai-org/GLM-5.3-Flash",
            "upstream_revision": "b" * 40,
        },
        fixtures_sha256="c" * 64,
        source_checks={"pass": True},
        baseline_identity={"status": "READY"},
        baseline_records=records,
        candidate_records=records,
        fallback={"pass": True, "transport_pass": True, "semantic_safety_pass": True},
        provider="baseten",
        provider_attempts=[],
    )
    summary = {
        key: receipt[key]
        for key in (
            "candidate_id", "model_id", "provider", "baseline_metrics",
            "candidate_metrics", "comparison", "decision",
            "production_disposition", "receipt_sha256",
        )
    } | {"suite": suite}
    bundle = {
        "receipt": receipt,
        "baseline_results": records,
        "candidate_results": records,
    }
    for name, value in (("receipt", receipt), ("summary", summary), ("bundle", bundle)):
        (output / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    return receipt


def _fake_hub(monkeypatch, *, fail_at=None, existing=(), parent="d" * 40):
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))
            if fail_at == "create_repo":
                raise RuntimeError("repository unavailable")

        def repo_info(self, **kwargs):
            calls.append(("repo_info", kwargs))
            if fail_at == "repo_info":
                raise RuntimeError("head unavailable")
            return SimpleNamespace(
                sha=parent,
                siblings=[SimpleNamespace(rfilename=name) for name in existing],
            )

        def create_commit(self, **kwargs):
            calls.append(("create_commit", kwargs))
            assert kwargs["parent_commit"] == parent
            if fail_at == "create_commit":
                raise RuntimeError("head conflict or commit rejected")
            return SimpleNamespace(
                oid="" if fail_at == "commit_identity" else "e" * 40,
                commit_url="https://huggingface.co/datasets/test/commit/" + "e" * 40,
            )

    def api_factory(**kwargs):
        assert kwargs == {"token": "publication-test-token"}
        calls.append(("client", {}))
        return FakeApi()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=api_factory, CommitOperationAdd=SimpleNamespace),
    )
    return calls


def test_publication_commits_card_and_final_evidence_atomically(tmp_path, monkeypatch):
    receipt = _publication_files(tmp_path)
    calls = _fake_hub(monkeypatch)
    result = publish(
        tmp_path, "publication-test-token", PUBLICATION_DATASET, PUBLICATION_PATH, receipt
    )
    assert [name for name, _ in calls] == [
        "client", "create_repo", "repo_info", "create_commit"
    ]
    commit = calls[-1][1]
    assert commit["revision"] == "main"
    assert commit["repo_type"] == "dataset"
    assert commit["parent_commit"] == "d" * 40
    files = {op.path_in_repo: op.path_or_fileobj for op in commit["operations"]}
    assert set(files) == {"README.md"} | {
        f"{PUBLICATION_PATH}/{name}.json" for name in ("bundle", "receipt", "summary")
    }
    assert all(isinstance(value, bytes) for value in files.values())
    assert json.loads(files[f"{PUBLICATION_PATH}/receipt.json"]) == receipt
    assert json.loads(files[f"{PUBLICATION_PATH}/bundle.json"])["receipt"] == receipt
    card = files["README.md"].decode()
    for name in ("summary", "bundle", "receipt"):
        assert f"blob/main/{PUBLICATION_PATH}/{name}.json" in card
        assert result[f"{name}_url"].endswith(
            f"/blob/{'e' * 40}/{PUBLICATION_PATH}/{name}.json"
        )
    assert result["card_url"].endswith(f"/blob/{'e' * 40}/README.md")
    assert result["commit_oid"] == "e" * 40


def test_publication_excludes_caches_credentials_and_unfinished_logs(tmp_path, monkeypatch):
    receipt = _publication_files(tmp_path)
    for name in (
        "DATASET_CARD.md", ".source-cache/model.json", "__pycache__/a.pyc",
        "nested/__pycache__/b.pyc", "other.pyc", ".env", "credentials.json",
        "publication.json", "runner-output.jsonl", "unreviewed.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private-test-sentinel", encoding="utf-8")
    calls = _fake_hub(monkeypatch)
    publish(tmp_path, "publication-test-token", PUBLICATION_DATASET, PUBLICATION_PATH, receipt)
    operations = calls[-1][1]["operations"]
    assert len(operations) == 4
    assert not any(b"private-test-sentinel" in op.path_or_fileobj for op in operations)


@pytest.mark.parametrize("stage", ["create_repo", "repo_info", "create_commit"])
def test_publication_propagates_remote_failures_without_partial_card_upload(
    tmp_path, monkeypatch, stage
):
    receipt = _publication_files(tmp_path)
    calls = _fake_hub(monkeypatch, fail_at=stage)
    with pytest.raises(RuntimeError):
        publish(tmp_path, "publication-test-token", PUBLICATION_DATASET, PUBLICATION_PATH, receipt)
    assert calls[-1][0] == stage
    assert sum(name == "create_commit" for name, _ in calls) <= 1


@pytest.mark.parametrize("stage", ["head", "commit_identity", "existing_run"])
def test_publication_requires_head_commit_and_unused_run_path(tmp_path, monkeypatch, stage):
    receipt = _publication_files(tmp_path)
    calls = _fake_hub(
        monkeypatch,
        parent="" if stage == "head" else "d" * 40,
        fail_at=stage,
        existing=[f"{PUBLICATION_PATH}/receipt.json"] if stage == "existing_run" else (),
    )
    with pytest.raises(ValueError):
        publish(tmp_path, "publication-test-token", PUBLICATION_DATASET, PUBLICATION_PATH, receipt)
    if stage != "commit_identity":
        assert not any(name == "create_commit" for name, _ in calls)


@pytest.mark.parametrize("artifact", ["summary", "bundle", "receipt", "missing", "credential"])
def test_publication_rejects_invalid_local_evidence_before_hub_access(
    tmp_path, monkeypatch, artifact
):
    receipt = _publication_files(tmp_path)
    calls = _fake_hub(monkeypatch)
    if artifact == "missing":
        (tmp_path / "summary.json").unlink()
    elif artifact == "credential":
        (tmp_path / "bundle.json").write_text("publication-test-token", encoding="utf-8")
    else:
        path = tmp_path / f"{artifact}.json"
        data = json.loads(path.read_text())
        if artifact == "summary":
            data["production_disposition"] = "READY"
        elif artifact == "bundle":
            data["candidate_results"][0]["http_status"] = 500
        else:
            data["production_disposition"] = "READY"
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        publish(tmp_path, "publication-test-token", PUBLICATION_DATASET, PUBLICATION_PATH, receipt)
    assert calls == []


@pytest.mark.parametrize("path", ["README.md", "runs/../escape", "runs//double", "runs/x\\escape"])
def test_publication_rejects_unsafe_run_paths(tmp_path, monkeypatch, path):
    receipt = _publication_files(tmp_path)
    calls = _fake_hub(monkeypatch)
    with pytest.raises(ValueError, match="safe run directory"):
        publish(tmp_path, "publication-test-token", PUBLICATION_DATASET, path, receipt)
    assert calls == []


@pytest.mark.parametrize("suite,count", [("full", 4), ("smoke", 2)])
def test_dataset_card_links_pinned_sources_and_reports_actual_scope(tmp_path, suite, count):
    receipt = _publication_files(tmp_path, suite=suite)
    card = dataset_card(PUBLICATION_DATASET, PUBLICATION_PATH, receipt)
    assert PUBLIC_EXAMPLE in card
    assert "/tree/320983d22e76fc9b26af0b2cd20799c5000543fc/" in card
    assert f"/blob/{'a' * 40}/frontier/evaluation/runner.py" in card
    assert f"{count} synthetic public test cases per model" in card
    assert "full suite has four cases" in card
    assert "**unsigned**" in card
    assert "do not authenticate an author" in card
    assert "do not establish method superiority" in card
    assert "on Sundays at 06:17 UTC" in card
    assert "after evaluation-related changes reach main" in card
    assert "Pull-request checks do not publish" in card
    assert "3,072 submitted cell columns" in card
    for term in ("Khipu", "Lambda", "Doctrine", "admission layer", "estate"):
        assert term not in card
