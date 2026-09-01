# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the SFT dataset validator (stdlib only)."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from validate_sft_dataset import main, validate_dataset

GOOD = {"messages": [
    {"role": "system", "content": "You are SZL-1."},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am SZL-1."},
]}


def _write(dirpath, rows, name="dataset.jsonl"):
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row if isinstance(row, str) else json.dumps(row))
            handle.write("\n")
    return str(path)


def test_valid_dataset_passes(tmp_path):
    path = _write(tmp_path, [GOOD, GOOD, GOOD])
    report = validate_dataset(path, min_examples=2)
    assert report["status"] == "VALID"
    assert report["records"] == 3
    assert report["dataset_sha256"].startswith("sha256:")
    assert report["errors"] == []


def test_repo_dataset_is_valid():
    report = validate_dataset(
        os.path.join(ROOT, "szl_dataset.jsonl"), min_examples=8
    )
    assert report["status"] == "VALID", report["errors"]
    assert report["records"] >= 8


def test_invalid_json_line_fails_closed(tmp_path):
    path = _write(tmp_path, [json.dumps(GOOD), "{not json"])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"
    assert any("invalid JSON" in error for error in report["errors"])


def test_missing_messages_fails(tmp_path):
    path = _write(tmp_path, [{"text": "not a chat record"}])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"
    assert report["records"] == 0


def test_unknown_role_fails(tmp_path):
    bad = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "oracle", "content": "hello"},
    ]}
    path = _write(tmp_path, [bad])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"
    assert any("role" in error for error in report["errors"])


def test_empty_content_fails(tmp_path):
    bad = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "   "},
    ]}
    path = _write(tmp_path, [bad])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"


def test_record_needs_user_and_assistant(tmp_path):
    bad = {"messages": [{"role": "user", "content": "hi"}]}
    path = _write(tmp_path, [bad])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"
    assert any("user and one" in error for error in report["errors"])


def test_min_examples_enforced(tmp_path):
    path = _write(tmp_path, [GOOD])
    report = validate_dataset(path, min_examples=8)
    assert report["status"] == "INVALID"
    assert any("min-examples" in error for error in report["errors"])


def test_missing_file_fails_closed(tmp_path):
    report = validate_dataset(str(tmp_path / "nope.jsonl"))
    assert report["status"] == "INVALID"


def test_cli_exit_codes(tmp_path, capsys):
    good = _write(tmp_path / "good", [GOOD, GOOD])
    assert main([good, "--min-examples", "2"]) == 0
    bad = _write(tmp_path / "bad", ["{broken"])
    assert main([bad]) == 1
    capsys.readouterr()  # keep output out of the test log


def test_cli_json_output(tmp_path, capsys):
    good = _write(tmp_path / "good", [GOOD])
    assert main([good, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "szl.forge-sft-dataset-report/v1"
    assert payload["status"] == "VALID"
