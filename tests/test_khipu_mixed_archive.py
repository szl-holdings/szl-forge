"""Preserved failed evidence must verify without running model or archived code."""

import importlib.util
import json
import shutil
import socket
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "khipu-mixed-holdout" / "run.py"
SPEC = importlib.util.spec_from_file_location("khipu_mixed_record_verifier", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_exact_historical_failure_verifies_without_execution_network_or_writes():
    with patch.object(subprocess, "run", side_effect=AssertionError("no execution")), \
         patch.object(socket, "socket", side_effect=AssertionError("no network")), \
         patch.object(Path, "write_text", side_effect=AssertionError("no evidence writes")), \
         patch.object(Path, "write_bytes", side_effect=AssertionError("no evidence writes")):
        result = RUN.verify_recorded()
    assert result["status"] == "RECORDED_EVIDENCE_CONSISTENT"
    assert result["historical_metrics"] == {
        "total": 12, "plan_contract_valid": 7,
        "navigation_correct": 3, "navigation_total": 6,
        "abstention_correct": 0, "abstention_total": 6,
    }
    assert result["archived_runner_executed"] is False
    assert result["model_inference"] == result["independent_inference_replay"] == "NOT_RUN"
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("relative", sorted(RUN.RECORDED_SHA256))
def test_each_changed_historical_artifact_is_rejected(tmp_path, relative):
    directory = tmp_path / "record"
    shutil.copytree(RUN.HERE, directory)
    target = directory / relative
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="recorded byte mismatch"):
        RUN.verify_recorded(directory)


def test_offline_regrader_rejects_duplicate_keys_and_non_json_constants():
    cases = RUN.load_json(RUN.CASES)["cases"]
    schema = RUN.load_json(RUN.SCHEMA)
    for raw in ('{"decision":"ABSTAIN","decision":"NAVIGATE"}', '{"decision":NaN}'):
        result = RUN.grade(raw, cases[0], schema)
        assert not result["json_valid"] and not result["case_pass"]


def test_saved_free_text_is_not_promoted_to_controller_evidence():
    # ABS-01 has a structurally valid plan with a false controller-content claim.
    # Regrading retains its failed case label; this is not semantic truth proof.
    fixture = RUN.load_json(RUN.CASES)
    cases = {case["id"]: case for case in fixture["cases"]}
    rows = [json.loads(line) for line in (RUN.HERE / "evidence/outputs.jsonl").read_text().splitlines()]
    row = next(row for row in rows if row["case_id"] == "ABS-01")
    result = RUN.grade(row["raw_output"], cases["ABS-01"], RUN.load_json(RUN.SCHEMA))
    assert result["contract_valid"] is True
    assert result["case_pass"] is False


def test_inference_options_are_not_supported():
    result = subprocess.run(
        [__import__("sys").executable, str(SOURCE), "--model-dir", "unavailable", "--output-dir", "unused"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
