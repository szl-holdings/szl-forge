"""Public synthetic development contracts, not a held-out model benchmark."""
import copy
import json
from pathlib import Path

import pytest

from inference import research_cycle as rc


FIXTURE = Path(__file__).parent.parent / "experiments/research-cycle/suite.json"


def suite():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def response(recipe):
    return json.dumps(recipe)


GOOD = {"title_weight": 4, "body_weight": 1, "normalize_length": True}


def run(proposer=None, data=None, **kwargs):
    return rc.run_cycle(data or suite(), proposer or (lambda context: response(GOOD)),
                        mode="SYNTHETIC_REPLAY", attempts=1, **kwargs)


def test_model_gets_development_only_and_cannot_mutate_evaluator_input():
    original = suite()
    seen = []

    def proposer(context):
        seen.append(copy.deepcopy(context))
        assert "validation" not in context
        assert "production" not in context
        context["documents"].clear()
        return response(GOOD)

    result = run(proposer, original)
    assert seen[0]["development"]
    assert original == suite()
    assert result["evaluation_scope"] == "PUBLIC_SYNTHETIC_DEVELOPMENT_ONLY"
    assert result["production_disposition"] == "HOLD"
    assert result["training_admission"] is False
    assert result["model_quality_established"] is False


def test_recipe_is_scored_and_result_is_integrity_bound():
    result = run()
    assert result["attempts"][0]["status"] == "EVALUATED"
    assert result["selected_recipe"] == GOOD
    assert result["decision"] == "DEVELOPMENT_IMPROVEMENT_REVIEW_REQUIRED"
    assert result["candidate_validation"]["mrr"] > result["baseline_validation"]["mrr"]
    payload = {k: v for k, v in result.items() if k != "receipt_sha256"}
    assert result["receipt_sha256"] == rc.digest(payload)
    assert result["signature_status"] == "UNSIGNED_LOCAL"


@pytest.mark.parametrize("recipe", [
    dict(GOOD, execute="whoami"), dict(GOOD, title_weight=True),
    dict(GOOD, body_weight=-1), dict(GOOD, title_weight=5),
    dict(GOOD, normalize_length="true"), {"title_weight": 1},
    {"title_weight": 0, "body_weight": 0, "normalize_length": False},
])
def test_untrusted_recipe_is_rejected_without_execution(recipe):
    result = run(lambda _: response(recipe))
    assert result["attempts"][0]["status"] == "INVALID_PROPOSAL"
    assert result["selected_recipe"] is None
    assert result["decision"] == "NO_DEVELOPMENT_IMPROVEMENT"


@pytest.mark.parametrize("raw", [
    '{"title_weight":1,"title_weight":4,"body_weight":1,"normalize_length":true}',
    '{"title_weight":NaN,"body_weight":1,"normalize_length":true}',
    '{"title_weight":1e999,"body_weight":1,"normalize_length":true}',
    "```json\n{}\n```", "[]", "x" * 8193,
])
def test_ambiguous_or_oversize_model_output_fails_closed(raw):
    assert run(lambda _: raw)["attempts"][0]["status"] == "INVALID_PROPOSAL"


def test_unavailable_attempt_is_retained_and_stops_without_retry_or_secret_text():
    calls = []

    def unavailable(_):
        calls.append(1)
        raise TimeoutError("private credential must not persist")

    result = run(unavailable)
    assert calls == [1]
    assert result["decision"] == "MODEL_UNAVAILABLE"
    assert result["attempts"][0]["error_type"] == "TimeoutError"
    assert "private credential" not in json.dumps(result)


def test_duplicate_candidate_is_not_recounted():
    result = rc.run_cycle(suite(), lambda _: response(GOOD), mode="SYNTHETIC_REPLAY", attempts=2)
    assert [x["status"] for x in result["attempts"]] == ["EVALUATED", "DUPLICATE"]


def test_tie_cannot_become_improvement():
    result = run(lambda _: response(suite()["baseline"]))
    assert result["decision"] == "NO_DEVELOPMENT_IMPROVEMENT"


def test_validation_regression_does_not_select_a_replacement():
    data = suite()
    # Development selects GOOD. The one-shot validation must not be used to
    # search for a second candidate, and a regressing result must remain failed.
    data["validation"] = [{"id": "v-regress", "query": "receipt ledger", "relevant_id": "a-distractor"}]
    result = run(data=data)
    assert result["selected_recipe"] == GOOD
    assert result["decision"] == "VALIDATION_NOT_IMPROVED"
    assert result["production_disposition"] == "HOLD"


@pytest.mark.parametrize("change", [
    lambda s: s.update(evidence_class="HELD_OUT"),
    lambda s: s["validation"].clear(),
    lambda s: s["development"].append(copy.deepcopy(s["development"][0])),
    lambda s: s["validation"][0].update(query=s["development"][0]["query"]),
    lambda s: s["validation"][0].update(relevant_id="unknown"),
    lambda s: s["documents"].append(copy.deepcopy(s["documents"][0])),
    lambda s: s["documents"][0].update(body="x" * 4097),
    lambda s: s.update(extra="arbitrary input"),
])
def test_invalid_suite_is_refused_before_model_call(change):
    data = suite()
    change(data)
    with pytest.raises(rc.CycleError):
        run(lambda _: pytest.fail("must not call model"), data=data)


@pytest.mark.parametrize("attempts", [0, 4, True, 1.5])
def test_attempt_budget_is_strict(attempts):
    with pytest.raises(rc.CycleError):
        rc.run_cycle(suite(), lambda _: "{}", mode="SYNTHETIC_REPLAY", attempts=attempts)


def test_input_digest_is_checked_before_model_call():
    with pytest.raises(rc.CycleError, match="digest"):
        rc.load_suite(FIXTURE, "0" * 64)


def test_result_writer_refuses_overwrite_and_html_escapes(tmp_path):
    result = run()
    result["limits"].append("<script>alert(1)</script>")
    result["receipt_sha256"] = rc.digest({k: v for k, v in result.items() if k != "receipt_sha256"})
    path = tmp_path / "run"
    rc.write_result(path, result)
    assert "<script>" not in (path / "index.html").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        rc.write_result(path, result)


def test_loopback_adapter_refuses_cloud_or_uninstalled_model(monkeypatch):
    calls = []

    def request(path, body=None):
        calls.append(path)
        return {"models": []}

    proposer = rc.OllamaProposer("not-installed", "a" * 64)
    monkeypatch.setattr(proposer, "_request", request)
    with pytest.raises(rc.CycleError):
        proposer({})
    assert calls == ["/api/tags"]
    with pytest.raises(rc.CycleError):
        rc.OllamaProposer("example:cloud", "a" * 64)


def test_loopback_adapter_binds_model_digest_before_and_after(monkeypatch):
    proposer = rc.OllamaProposer("local:test", "a" * 64)
    calls = []

    def request(path, body=None):
        calls.append(path)
        if path == "/api/tags":
            return {"models": [{"name": "local:test", "digest": ("b" if len(calls) == 3 else "a") * 64}]}
        assert body["think"] is False
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 128
        assert "tools" not in body
        return {"model": "local:test", "done": True, "message": {"role": "assistant", "content": response(GOOD)}}

    monkeypatch.setattr(proposer, "_request", request)
    with pytest.raises(rc.CycleError, match="identity"):
        proposer({})
    assert calls == ["/api/tags", "/api/chat", "/api/tags"]


def test_no_side_effect_authority_is_added_to_existing_controller():
    from inference.governed_inference import governed_infer
    assert callable(governed_infer)
    assert rc.AUTHORITY == "NONE_RESEARCH_ONLY"


def test_result_tampering_is_rejected_before_creating_directory(tmp_path):
    result = run()
    result["decision"] = "tampered"
    with pytest.raises(rc.CycleError, match="digest"):
        rc.write_result(tmp_path / "tampered", result)
    assert not (tmp_path / "tampered").exists()


def test_rehashed_authority_upgrade_is_refused(tmp_path):
    result = run()
    result["publication_eligible"] = True
    result["receipt_sha256"] = rc.digest({k: v for k, v in result.items() if k != "receipt_sha256"})
    with pytest.raises(rc.CycleError, match="authority"):
        rc.write_result(tmp_path / "tampered", result)


def governed_proposer(decision="ALLOW"):
    from inference.governed_inference import text_sha256
    content = "Public development experiment. Retrieval configuration only."
    evidence = {"node_id": "research-fixture", "source": "public-fixture", "sha256": text_sha256(content)}

    def retrieve(query, k):
        return {"content_access": "HANDLES_ONLY", "ready": True,
                "handles": [{"nodeId": "research-fixture"}], "evidence": [evidence]}

    def generate(context):
        assert context["instructions"]["authority"] == "PROPOSAL_ONLY"
        return {"text": response(GOOD), "model": {"id": "synthetic-test", "revision": "a" * 40},
                "runtime": {"engine": "synthetic", "version": "test", "hardware_fingerprint": "test-only"},
                "output_schema": "szl.retrieval-recipe/v1"}

    return rc.GovernedProposer({"request_id": "research-test", "principal_id": "test", "tenant_id": "test"},
        retriever=retrieve, hydrator=lambda handles, request: [dict(evidence, content=content)],
        generator=generate, witness=lambda stage, payload: {"decision": decision, "rule_version": "test-only"})


def test_existing_governed_coordinator_is_in_the_actual_proposal_path():
    result = rc.run_cycle(suite(), governed_proposer(), mode="GOVERNED_PROPOSAL")
    assert result["decision"] == "DEVELOPMENT_IMPROVEMENT_REVIEW_REQUIRED"
    assert len(result["attempts"][0]["governed_inference_receipt_sha256"]) == 64
    assert result["identity_boundary"] == "CONTROLLER_RECEIPT_NOT_RUNTIME_ATTESTATION"


@pytest.mark.parametrize("decision", ["BLOCK", "REVIEW"])
def test_existing_witness_can_stop_research_proposals(decision):
    result = rc.run_cycle(suite(), governed_proposer(decision), mode="GOVERNED_PROPOSAL")
    assert result["decision"] == "MODEL_UNAVAILABLE"
    assert result["selected_recipe"] is None


def test_generic_callback_cannot_claim_governed_mode():
    with pytest.raises(rc.CycleError, match="adapter"):
        rc.run_cycle(suite(), lambda _: response(GOOD), mode="GOVERNED_PROPOSAL")


def local_run(proposer, attempts=3):
    return rc.run_cycle(suite(), proposer, mode="LOCAL_MODEL_PROPOSAL", attempts=attempts,
                        identity={"model": "local:test", "reported_digest": "a" * 64})


def local_reply(path):
    if path == "/api/tags":
        return {"models": [{"name": "local:test", "digest": "a" * 64}]}
    return {"model": "local:test", "done": True,
            "message": {"role": "assistant", "content": response(GOOD)}}


@pytest.mark.parametrize("phase, failed_call", [
    ("inventory_before", 1), ("generation", 2), ("inventory_after", 3),
])
def test_local_failure_phase_preserves_timeout_and_never_retries(monkeypatch, phase, failed_call):
    proposer = rc.OllamaProposer("local:test", "a" * 64)
    calls = []

    def request(path, body=None):
        calls.append(path)
        if len(calls) == failed_call:
            raise TimeoutError("private exception text must not persist")
        return local_reply(path)

    monkeypatch.setattr(proposer, "_request", request)
    # Each successful phase consumes one clock tick, a failing phase two.
    ticks = iter(range(10))
    monkeypatch.setattr(rc.time, "monotonic", lambda: next(ticks))
    result = local_run(proposer)
    assert result["decision"] == "MODEL_UNAVAILABLE"
    assert len(calls) == failed_call
    assert len(result["attempts"]) == 1
    row = result["attempts"][0]
    assert row["error_type"] == "TimeoutError"
    assert row["failure_phase"] == phase
    assert row["elapsed_seconds"] == 1
    assert "private exception" not in json.dumps(result)
    assert result["receipt_sha256"] == rc.digest({k: v for k, v in result.items() if k != "receipt_sha256"})
    assert result["production_disposition"] == "HOLD"


def test_phase_resets_on_adapter_reuse_and_success_adds_no_failure(monkeypatch):
    proposer = rc.OllamaProposer("local:test", "a" * 64)
    monkeypatch.setattr(proposer, "_request", lambda *args: {"models": []})
    failed = local_run(proposer)
    assert failed["attempts"][0]["failure_phase"] == "inventory_before"
    assert failed["attempts"][0]["error_type"] == "CycleError"
    monkeypatch.setattr(proposer, "_request", lambda path, body=None: local_reply(path))
    result = local_run(proposer, attempts=1)
    assert result["decision"] == "DEVELOPMENT_IMPROVEMENT_REVIEW_REQUIRED"
    assert "failure_phase" not in result["attempts"][0]
    assert proposer.failure_observation is None


def test_post_generation_identity_mismatch_cannot_become_success(monkeypatch):
    proposer = rc.OllamaProposer("local:test", "a" * 64)
    calls = []

    def request(path, body=None):
        calls.append(path)
        return {"models": []} if len(calls) == 3 else local_reply(path)

    monkeypatch.setattr(proposer, "_request", request)
    result = local_run(proposer)
    assert result["decision"] == "MODEL_UNAVAILABLE"
    assert result["selected_recipe"] is None
    assert result["attempts"][0]["failure_phase"] == "inventory_after"


def test_generic_callback_cannot_inject_phase_diagnostics():
    def callback(_):
        raise TimeoutError("not retained")

    callback.failure_observation = {"failure_phase": "private injected text", "elapsed_seconds": 1}
    result = run(callback)
    assert "failure_phase" not in result["attempts"][0]
    assert "private injected text" not in json.dumps(result)


@pytest.mark.parametrize("diagnostic", [
    {"failure_phase": "private injected text", "elapsed_seconds": 1},
    {"failure_phase": "generation", "elapsed_seconds": float("nan")},
    {"failure_phase": "generation", "elapsed_seconds": float("inf")},
    {"failure_phase": "generation", "elapsed_seconds": True},
    {"failure_phase": "generation", "elapsed_seconds": -1},
    {"failure_phase": "generation", "elapsed_seconds": 1, "secret": "not admitted"},
])
def test_modified_adapter_metadata_cannot_inject_unbounded_diagnostics(monkeypatch, diagnostic):
    proposer = rc.OllamaProposer("local:test", "a" * 64)

    def observe(*args):
        proposer.failure_observation = diagnostic
        raise TimeoutError("not retained")

    monkeypatch.setattr(proposer, "_observe", observe)
    result = local_run(proposer)
    assert "failure_phase" not in result["attempts"][0]
