"""Offline contract tests for the preregistered selective-abstention scorer."""

import importlib.util
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "khipu-selective-holdout-v2" / "run.py"
SPEC = importlib.util.spec_from_file_location("khipu_selective_v2_run", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def _case(decision):
    candidate = {
        "nodeId": "node://khipu-synthetic/selective-v2-test",
        "nodeKind": "INDEX",
        "label": "SIMULATED",
        "note": "Synthetic metadata only: test ledger index.",
    }
    return {
        "query": "Find the test ledger index handle.",
        "candidates": [candidate],
        "expected_decision": decision,
        "expected_cited_node_ids": [candidate["nodeId"]] if decision == "NAVIGATE" else [],
    }


def _plan(item):
    cited = item["expected_cited_node_ids"]
    return {
        "planId": "plan-synthetic-test",
        "capabilityProfile": "SZL-Khipu-1.5B-BrainNavigator",
        "provenance": "MODEL_PROPOSED",
        "query": item["query"],
        "contentAccess": "HANDLES_ONLY",
        "candidates": item["candidates"],
        "decision": item["expected_decision"],
        "steps": [
            {"action": action, "nodeId": cited[0], "rationale": "Matching metadata"}
            for action in ("RETRIEVE", "CITE")
        ] if cited else [],
        "citedNodeIds": cited,
        "groundedOnly": True,
        "brainBinding": {"protocol": "khipu-retrieval", "status": "NOT_RESOLVED", "note": "No content resolved."},
        "controllerBoundary": "A11oy validates and resolves handles outside the model.",
        "abstainReason": None if cited else "No offered handle supports this.",
    }


def test_both_reference_decisions_pass():
    schema = RUN.load_json(RUN.SCHEMA)
    for decision in ("NAVIGATE", "ABSTAIN"):
        item = _case(decision)
        assert RUN.grade(json.dumps(_plan(item)), item, schema)["case_pass"]


def test_duplicate_decision_fails_before_schema_grading():
    item = _case("NAVIGATE")
    raw = json.dumps(_plan(item)).replace('"decision": "NAVIGATE"', '"decision": "ABSTAIN", "decision": "NAVIGATE"')
    result = RUN.grade(raw, item, RUN.load_json(RUN.SCHEMA))
    assert not result["strict_json_valid"]
    assert not result["case_pass"]


def test_nested_duplicate_and_non_json_constant_fail_closed():
    item = _case("ABSTAIN")
    raw = json.dumps(_plan(item)).replace('"status": "NOT_RESOLVED"', '"status": "RESOLVED", "status": "NOT_RESOLVED"')
    assert not RUN.grade(raw, item, RUN.load_json(RUN.SCHEMA))["strict_json_valid"]
    try:
        RUN.strict_json('{"decision": NaN}')
    except ValueError:
        pass
    else:
        raise AssertionError("NaN is not JSON")


def test_related_but_unoffered_citation_fails():
    item = _case("NAVIGATE")
    plan = _plan(item)
    plan["citedNodeIds"] = ["node://khipu-synthetic/selective-v2-unoffered"]
    assert not RUN.grade(json.dumps(plan), item, RUN.load_json(RUN.SCHEMA))["case_pass"]


def test_over_abstention_does_not_pass_supported_lookup():
    item = _case("NAVIGATE")
    plan = _plan(item)
    plan["decision"] = "ABSTAIN"
    plan["steps"] = []
    plan["citedNodeIds"] = []
    plan["abstainReason"] = "Decline"
    result = RUN.grade(json.dumps(plan), item, RUN.load_json(RUN.SCHEMA))
    assert result["contract_valid"]
    assert not result["case_pass"]


def test_grading_schema_uses_exact_preregistered_bytes():
    frozen = RUN.load_json(RUN.PREREG)
    assert hashlib.sha256(RUN.SCHEMA.read_bytes()).hexdigest() == frozen["plan_schema_sha256"]
    assert RUN.load_grading_schema(frozen) == RUN.load_json(RUN.SCHEMA)


def test_changed_grading_schema_refuses_before_model_or_output_effects(tmp_path):
    frozen = RUN.load_json(RUN.PREREG)
    changed_schema = tmp_path / "changed-schema.json"
    changed_schema.write_text('{"type":"object"}', encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with patch.object(RUN, "SCHEMA", changed_schema), \
         patch.object(RUN, "check_frozen_inputs", return_value=(frozen, [], "system")), \
         patch.object(RUN, "signed_anchor") as anchor, \
         patch.object(RUN, "check_local_artifacts") as artifacts:
        with pytest.raises(ValueError, match="working-tree grading schema"):
            RUN.run(tmp_path / "cases.json", tmp_path / "model", output)
        anchor.assert_not_called()
        artifacts.assert_not_called()
    assert not output.exists()
