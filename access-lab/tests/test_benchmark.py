"""Public synthetic suite integrity tests, not accessibility certification."""

import json
import hashlib
from collections import Counter

import pytest

from szl_access.benchmark import benchmark_cases, examples, main, run_benchmark
from szl_access.engine import apply_proposal


def test_cases_have_distinct_identity_and_explicit_provenance():
    cases = benchmark_cases()
    assert len(cases) == 36
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["source_sha256"] for case in cases}) == len(cases)
    assert Counter(case["expected_state"] for case in cases) == {
        "VERIFIED_SCOPED_REPAIR": 12,
        "REVIEW_REQUIRED": 18,
        "NO_CHANGE": 6,
    }
    assert all(case["family"] == case["split_group"] for case in cases)
    assert all("synthetic" in case["provenance"] for case in cases)
    assert all(case["split"] == "public_development_regression_not_held_out" for case in cases)


def test_returned_cases_are_fresh_copies():
    cases = benchmark_cases()
    cases[0]["source"] = "mutated"
    assert benchmark_cases()[0]["source"] != "mutated"


def test_benchmark_is_reproducible_and_passes_full_gate():
    report = run_benchmark()
    assert report == run_benchmark()
    assert report["status"] == "PASS", report["failed_case_ids"]
    assert report["count"] == report["state_correct"] == report["passed"] == 36
    assert report["failed_case_ids"] == []
    assert len(report["recipe_sha256"]) == 64
    assert report["methodology"] == "deterministic"
    assert report["proposer_identity"] == "builtin_deterministic"
    assert any("not a held-out" in value for value in report["limitations"])


def test_broken_proposer_cannot_pass_or_claim_builtin_identity():
    def broken(_source):
        raise RuntimeError("Synthetic broken proposer")

    report = run_benchmark(broken, methodology="candidate-test")
    assert report["status"] == "FAIL"
    assert report["passed"] == report["state_correct"] == 0
    assert len(report["failed_case_ids"]) == 36
    assert report["proposer_identity"] == "caller_supplied_not_independently_verified"
    assert all(row["problems"] == ["execution_error:RuntimeError"] for row in report["rows"])


def test_cannot_rename_builtin_as_a_model_evaluation():
    with pytest.raises(ValueError, match="explicit proposer"):
        run_benchmark(methodology="trained-model")


def test_hash_consistent_extra_mutation_is_still_a_failed_repair(monkeypatch):
    def extra_mutation(source, proposal):
        result = apply_proposal(source, proposal)
        if result["state"] == "VERIFIED_SCOPED_REPAIR":
            result["output_html"] += "\n"
            result["output_sha256"] = hashlib.sha256(result["output_html"].encode()).hexdigest()
        return result

    monkeypatch.setattr("szl_access.benchmark.apply_proposal", extra_mutation)
    report = run_benchmark()
    assert report["state_correct"] == 36
    assert report["passed"] == 24
    assert report["status"] == "FAIL"
    failed = [row for row in report["rows"] if not row["passed"]]
    assert len(failed) == 12
    assert all("exact_output_mismatch" in row["problems"] for row in failed)


def test_examples_are_owned_fixtures_with_a_refusal_case():
    sample = examples()
    assert len(sample) == 5
    assert all(set(row) == {"id", "title", "source"} for row in sample)
    assert "review-aria" in {row["id"] for row in sample}


def test_main_emits_json_and_success_exit(capsys):
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["count"] == 36


def test_main_returns_nonzero_for_failed_benchmark(monkeypatch, capsys):
    monkeypatch.setattr("szl_access.benchmark.run_benchmark", lambda: {"status": "FAIL"})
    assert main() == 1
    assert json.loads(capsys.readouterr().out) == {"status": "FAIL"}
