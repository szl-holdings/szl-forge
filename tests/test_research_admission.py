"""The admission record is a local synthetic evaluation, not model qualification."""
import hashlib
import json

import pytest

from inference import research_cycle as cycle
from inference import research_investigator as investigator
from inference.research_admission import AdmissionError, admit


BASELINE = {"title_weight": 4, "body_weight": 1, "normalize_length": False}
CANDIDATE = {"title_weight": 1, "body_weight": 4, "normalize_length": False}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def report_bytes():
    body = "Retrieval validation requires a bounded synthetic experiment."
    document = {"id": "source", "text": body, "source": "https://example.org/research",
                "revision": "a" * 40, "visibility": "public", "content_sha256": sha(body.encode())}
    proposal = {"hypothesis": "A typed ranking change may improve synthetic retrieval.",
                "citations": [{"id": "source", "quote": "bounded synthetic experiment"}],
                "experiment": {"metric": "MRR", "procedure": "Compare fixed typed recipes.",
                               "success_criterion": "Improve development and validation MRR."},
                "uncertainties": ["No broader quality or novelty is established."]}
    actions = iter([{"tool": "search", "query": "validation"},
                    {"tool": "read", "id": "source"},
                    {"tool": "finish", "proposal": proposal}])
    corpus = investigator.Corpus([document])
    try:
        report = investigator.run_cycle("Study a bounded retrieval experiment", corpus,
                                        lambda _: json.dumps(next(actions)), execution_place="local", max_turns=3)
    finally:
        corpus.close()
    assert report["state"] == "PROPOSAL_REQUIRES_REVIEW"
    return encoded(report)


def suite_bytes():
    suite = {"schema": "szl.research-cycle-suite/v1",
             "evidence_class": cycle.SCOPE, "baseline": BASELINE,
             "documents": [
                 {"id": "a", "title": "unrelated", "body": "apple banana"},
                 {"id": "b", "title": "apple banana", "body": "unrelated"},
             ],
             "development": [{"id": "dev", "query": "apple", "relevant_id": "a"}],
             "validation": [{"id": "val", "query": "banana", "relevant_id": "a"}]}
    return encoded(suite)


def inputs(recipe=None):
    report, suite = report_bytes(), suite_bytes()
    manifest = {"schema": "szl.research-admission-manifest/v1",
                "kind": "lexical_retrieval_v1", "review_assertion": "OPERATOR_REVIEWED",
                "investigator_report_sha256": sha(report), "suite_sha256": sha(suite),
                "recipe": CANDIDATE if recipe is None else recipe}
    return manifest, report, suite


def test_pinned_typed_recipe_gets_one_unsigned_hold_evaluation():
    manifest, report, suite = inputs()
    result = admit(encoded(manifest), report, suite)
    assert result["review_provenance"] == "CALLER_ASSERTED_NOT_AUTHENTICATED"
    assert result["recipe_origin"] == "CALLER_SUPPLIED_AFTER_REVIEW"
    assert result["report_check"] == "PINNED_BYTES_STATE_AND_NONEXECUTION_FLAGS_ONLY"
    assert result["validation_outcome"] == "DEVELOPMENT_IMPROVEMENT_REVIEW_REQUIRED"
    assert len(result["evaluator_result"]["attempts"]) == 1
    assert result["evaluator_result"]["mode"] == "SYNTHETIC_REPLAY"
    assert result["evaluator_result"]["identity_boundary"] == "NO_MODEL_EXECUTED"
    assert result["evaluator_receipt_sha256"] == result["evaluator_result"]["receipt_sha256"]
    assert result["receipt_sha256"] == cycle.digest({k: v for k, v in result.items() if k != "receipt_sha256"})
    assert result["signature_status"] == "UNSIGNED_LOCAL"
    assert result["production_disposition"] == "HOLD"
    assert result["training_admission"] is False
    assert result["publication_eligible"] is False


def test_exact_report_and_suite_bytes_must_match_manifest():
    manifest, report, suite = inputs()
    with pytest.raises(AdmissionError, match="investigator_report_sha256 mismatch"):
        admit(encoded(manifest), report + b" ", suite)
    with pytest.raises(AdmissionError, match="suite_sha256 mismatch"):
        admit(encoded(manifest), report, suite + b" ")


def test_rehashed_invalid_suite_is_rejected_by_existing_validator():
    manifest, report, suite = inputs()
    altered = json.loads(suite)
    altered["evidence_class"] = "HELD_OUT"
    suite = encoded(altered)
    manifest["suite_sha256"] = sha(suite)
    with pytest.raises(cycle.CycleError, match="public synthetic"):
        admit(encoded(manifest), report, suite)


@pytest.mark.parametrize("change", [
    lambda m: m.update(extra="execute"),
    lambda m: m.update(kind="patch_executor_v1"),
    lambda m: m.update(review_assertion="MODEL_APPROVED"),
    lambda m: m.update(recipe={**CANDIDATE, "command": "run"}),
    lambda m: m.update(recipe={**CANDIDATE, "title_weight": True}),
    lambda m: m.update(recipe={**CANDIDATE, "title_weight": 5}),
    lambda m: m.update(recipe={**CANDIDATE, "normalize_length": "false"}),
])
def test_manifest_is_fixed_kind_and_strictly_typed(change):
    manifest, report, suite = inputs()
    change(manifest)
    with pytest.raises((AdmissionError, cycle.CycleError)):
        admit(encoded(manifest), report, suite)


@pytest.mark.parametrize("change", [
    lambda r: r.update(state="ABSTAINED"),
    lambda r: r.update(experiment_executed=True),
    lambda r: r.update(training_eligible=True),
    lambda r: r.update(publication_eligible=True),
    lambda r: r.update(autonomy_eligible=True),
    lambda r: r.update(novelty="ESTABLISHED"),
    lambda r: r.update(approved=True),
])
def test_rehashed_report_still_cannot_claim_execution_or_authority(change):
    manifest, report, suite = inputs()
    altered = json.loads(report)
    change(altered)
    report = encoded(altered)
    manifest["investigator_report_sha256"] = sha(report)
    with pytest.raises(AdmissionError):
        admit(encoded(manifest), report, suite)


def test_no_improvement_remains_hold():
    manifest, report, suite = inputs(BASELINE)
    result = admit(encoded(manifest), report, suite)
    assert result["validation_outcome"] == "NO_DEVELOPMENT_IMPROVEMENT"
    assert result["evaluator_result"]["attempts"][0]["status"] == "DUPLICATE"
    assert result["evaluator_result"]["selected_recipe"] is None
    assert result["production_disposition"] == "HOLD"


def test_validation_regression_remains_hold():
    manifest, report, suite = inputs()
    altered = json.loads(suite)
    altered["validation"][0]["relevant_id"] = "b"
    suite = encoded(altered)
    manifest["suite_sha256"] = sha(suite)
    result = admit(encoded(manifest), report, suite)
    assert result["evaluator_result"]["selected_recipe"] == CANDIDATE
    assert result["validation_outcome"] == "VALIDATION_NOT_IMPROVED"
    assert result["production_disposition"] == "HOLD"
