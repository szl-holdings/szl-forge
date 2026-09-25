"""Source-investigation checks; not hidden model evaluations or training data."""
import copy
import json

import pytest

from inference.research_investigator import Corpus, run_cycle, sha256


def document(**updates):
    body = "The baseline needs a held-out cross-repository evaluation before promotion."
    row = dict(id="forge", text=body, source="https://github.com/szl-holdings/szl-forge",
               revision="a" * 40, visibility="public", content_sha256=sha256(body.encode()))
    row.update(updates)
    return row


def proposal():
    return {"hypothesis": "Retrieval may improve cross-repository evaluation accuracy.",
            "citations": [{"id": "forge", "quote": "held-out cross-repository evaluation"}],
            "experiment": {"metric": "exact test pass rate", "procedure": "Compare fixed tasks with and without retrieval.",
                           "success_criterion": "Higher task pass rate without increasing false approvals."},
            "uncertainties": ["No experiment has run; generalization is unknown."]}


def scripted(*actions):
    iterator = iter(actions)
    return lambda _: json.dumps(next(iterator))


@pytest.fixture
def corpus():
    value = Corpus([document()])
    yield value
    value.close()


def valid_actions(value=None):
    return ({"tool": "search", "query": "evaluation"}, {"tool": "read", "id": "forge"},
            {"tool": "finish", "proposal": value or proposal()})


def test_source_quoted_proposal_is_not_an_experiment(corpus):
    result = run_cycle("Improve research", corpus, scripted(*valid_actions()), execution_place="local")
    assert result["state"] == "PROPOSAL_REQUIRES_REVIEW"
    assert result["evidence"][0]["content_sha256"] == document()["content_sha256"]
    assert result["novelty"] == "NOT_ESTABLISHED"
    assert result["semantic_quality"] == "NOT_EVALUATED"
    assert not any(result[k] for k in ("experiment_executed", "autonomy_eligible", "training_eligible", "publication_eligible"))


@pytest.mark.parametrize("raw", ['{}', '[]', '{"tool":"search","tool":"read"}',
                                    '{"tool":"search","query":NaN}', '{"tool":"search","query":1e999}',
                                    '```json\n{}\n```', '{"tool":"shell","command":"echo no"}',
                                    '{"tool":"read","id":[]}', 'x' * 24_001])
def test_bad_model_output_stops_without_action(corpus, raw):
    result = run_cycle("question", corpus, lambda _: raw, execution_place="local")
    assert result["state"] == "INVALID_MODEL_OUTPUT"
    assert result["proposal"] is None


@pytest.mark.parametrize("field,value", [("quote", "This quotation was never in the source"), ("id", "imagined")])
def test_fabricated_citations_rejected(corpus, field, value):
    finish = proposal()
    finish["citations"][0][field] = value
    result = run_cycle("question", corpus, scripted(*valid_actions(finish)), execution_place="local")
    assert result["state"] == "INVALID_MODEL_OUTPUT"


def test_unread_evidence_and_undiscovered_read_rejected(corpus):
    for action in ({"tool": "finish", "proposal": proposal()}, {"tool": "read", "id": "forge"}):
        result = run_cycle("question", corpus, scripted(action), execution_place="local")
        assert result["state"] == "INVALID_MODEL_OUTPUT"


def test_private_corpus_blocked_before_remote_callback():
    corpus = Corpus([document(visibility="private")])
    try:
        with pytest.raises(ValueError, match="private corpus"):
            run_cycle("question", corpus, lambda _: pytest.fail("must not call"), execution_place="remote")
    finally:
        corpus.close()


def test_remote_public_adapter_supported(corpus):
    result = run_cycle("question", corpus, scripted(*valid_actions()), execution_place="remote")
    assert result["state"] == "PROPOSAL_REQUIRES_REVIEW"


@pytest.mark.parametrize("updates", [{"content_sha256": "f" * 64}, {"revision": "main"},
                                    {"visibility": "unknown"}, {"text": ""}, {"text": "x" * 12001}])
def test_bad_corpus_rejected(updates):
    with pytest.raises(ValueError):
        Corpus([document(**updates)])


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate document"):
        Corpus([document(), document()])


def test_snapshot_cannot_be_changed_by_input_or_readback(corpus):
    rows = [document()]
    other = Corpus(rows)
    try:
        rows[0]["text"] = "changed after indexing"
        returned = other.read("forge")
        returned["text"] = "changed after read"
        assert other.read("forge") == corpus.read("forge")
        assert other.digest == corpus.digest
    finally:
        other.close()


def test_fts_syntax_is_literal_data(corpus):
    assert corpus.search('" OR * evaluation )')
    assert corpus.search("* ( \"") == []


def test_callback_failure_has_no_error_message_leak(corpus):
    def error(_):
        raise RuntimeError("pretend credential that must not be saved")
    result = run_cycle("question", corpus, error, execution_place="local")
    assert result["state"] == "GENERATOR_ERROR"
    assert "credential" not in json.dumps(result)


def test_turn_budget_is_enforced(corpus):
    result = run_cycle("question", corpus, scripted({"tool": "search", "query": "evaluation"}),
                       execution_place="local", max_turns=1)
    assert result["state"] == "INVALID_MODEL_OUTPUT"
    assert len(result["trace"]) == 1
    assert "result" not in result["trace"][0]


@pytest.mark.parametrize("turns", [0, 13, True])
def test_invalid_turn_bound(corpus, turns):
    with pytest.raises(ValueError):
        run_cycle("question", corpus, lambda _: "", execution_place="local", max_turns=turns)


def test_abstention_is_not_promotion(corpus):
    result = run_cycle("question", corpus, scripted({"tool": "abstain", "reason": "Missing evidence"}), execution_place="local")
    assert result["state"] == "ABSTAINED"
    assert result["proposal"] is None


def test_claimed_extra_authority_rejected(corpus):
    finish = copy.deepcopy(proposal())
    finish["approved"] = True
    result = run_cycle("question", corpus, scripted(*valid_actions(finish)), execution_place="local")
    assert result["state"] == "INVALID_MODEL_OUTPUT"


def test_model_can_see_remaining_turn_budget(corpus):
    seen = []
    def generate(messages):
        seen.append(messages[0]["content"])
        return '{"tool":"search","query":"evaluation"}'
    result = run_cycle("question", corpus, generate, execution_place="local", max_turns=2)
    assert "turn 1 of 2" in seen[0]
    assert "turn 2 of 2" in seen[1]
    assert result["state"] == "INVALID_MODEL_OUTPUT"
    assert result["max_turns"] == 2


def test_last_turn_is_explicit_synthesis_with_checked_citations(corpus):
    actions = iter(valid_actions())
    seen = []
    def generate(messages):
        seen.append(messages)
        return json.dumps(next(actions))
    result = run_cycle("question", corpus, generate, execution_place="local", max_turns=3)
    assert "research phase is CLOSED" in seen[-1][0]["content"]
    assert result["state"] == "PROPOSAL_REQUIRES_REVIEW"
    assert result["trace"][-1]["phase"] == "synthesis"


def test_deeply_nested_output_is_a_recorded_failure(corpus):
    result = run_cycle("question", corpus, lambda _: '{"a":' + '[' * 2000 + '0' + ']' * 2000 + '}',
                       execution_place="local")
    assert result["state"] == "INVALID_MODEL_OUTPUT"
