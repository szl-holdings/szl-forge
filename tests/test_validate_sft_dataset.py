# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the SFT dataset validator (stdlib only)."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from validate_sft_dataset import main, validate_dataset

def good(label="one", answer=None):
    return {"messages": [
        {"role": "system", "content": "You are SZL-1."},
        {"role": "user", "content": f"Question {label}?"},
        {"role": "assistant", "content": answer or f"Answer {label}."},
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
    path = _write(tmp_path, [good("one"), good("two"), good("three")])
    report = validate_dataset(path, min_examples=2)
    assert report["status"] == "VALID"
    assert report["records"] == 3
    assert report["dataset_sha256"].startswith("sha256:")
    assert report["duplicate_conversations"] == 0
    assert report["duplicate_inputs"] == 0
    assert report["cross_dataset_overlaps"] == 0
    assert report["reference_datasets"] == []
    assert report["errors"] == []


def test_repo_dataset_is_valid():
    report = validate_dataset(
        os.path.join(ROOT, "szl_dataset.jsonl"), min_examples=8
    )
    assert report["status"] == "VALID", report["errors"]
    assert report["records"] >= 8


def test_invalid_json_line_fails_closed(tmp_path):
    path = _write(tmp_path, [json.dumps(good()), "{not json"])
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
    path = _write(tmp_path, [good()])
    report = validate_dataset(path, min_examples=8)
    assert report["status"] == "INVALID"
    assert any("min-examples" in error for error in report["errors"])


def test_missing_file_fails_closed(tmp_path):
    report = validate_dataset(str(tmp_path / "nope.jsonl"))
    assert report["status"] == "INVALID"


def test_cli_exit_codes(tmp_path, capsys):
    good_path = _write(tmp_path / "good", [good("one"), good("two")])
    assert main([good_path, "--min-examples", "2"]) == 0
    bad = _write(tmp_path / "bad", ["{broken"])
    assert main([bad]) == 1
    capsys.readouterr()  # keep output out of the test log


def test_cli_json_output(tmp_path, capsys):
    good_path = _write(tmp_path / "good", [good()])
    assert main([good_path, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "szl.forge-sft-dataset-report/v2"
    assert payload["status"] == "VALID"


def test_duplicate_conversation_fails_closed(tmp_path):
    row = good("same")
    path = _write(tmp_path, [row, row])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"
    assert report["records"] == 1
    assert report["duplicate_conversations"] == 1
    assert report["duplicate_inputs"] == 1
    assert any("duplicate conversation" in error for error in report["errors"])


def test_duplicate_input_with_different_response_fails_closed(tmp_path):
    path = _write(tmp_path, [
        good("same", answer="First label."),
        good("same", answer="Conflicting label."),
    ])
    report = validate_dataset(path)
    assert report["status"] == "INVALID"
    assert report["records"] == 1
    assert report["duplicate_conversations"] == 0
    assert report["duplicate_inputs"] == 1
    assert any("different response" in error for error in report["errors"])


def test_disjoint_reference_passes_and_binds_reference_bytes(tmp_path):
    train = _write(tmp_path / "train", [good("train")])
    heldout = _write(tmp_path / "heldout", [good("heldout")])
    report = validate_dataset(train, disjoint_from=[heldout])
    assert report["status"] == "VALID"
    assert report["cross_dataset_overlaps"] == 0
    assert report["reference_datasets"] == [{
        "path": heldout,
        "status": "VALID",
        "records": 1,
        "dataset_sha256": report["reference_datasets"][0]["dataset_sha256"],
        "overlapping_inputs": 0,
    }]
    assert report["reference_datasets"][0]["dataset_sha256"].startswith("sha256:")


def test_cross_split_input_overlap_fails_even_when_answer_differs(tmp_path):
    train = _write(tmp_path / "train", [good("shared", answer="Training label.")])
    heldout = _write(tmp_path / "heldout", [good("shared", answer="Held-out label.")])
    report = validate_dataset(train, disjoint_from=[heldout])
    assert report["status"] == "INVALID"
    assert report["cross_dataset_overlaps"] == 1
    assert report["reference_datasets"][0]["overlapping_inputs"] == 1
    assert any("overlaps reference" in error for error in report["errors"])


def test_prompt_only_reference_is_valid_and_blocks_overlap(tmp_path):
    train = _write(tmp_path / "train", [good("shared")])
    prompt_only = {"messages": good("shared")["messages"][:-1]}
    heldout = _write(tmp_path / "heldout", [prompt_only])
    report = validate_dataset(train, disjoint_from=[heldout])
    assert report["status"] == "INVALID"
    assert report["reference_datasets"][0]["status"] == "VALID"
    assert report["reference_datasets"][0]["records"] == 1
    assert report["cross_dataset_overlaps"] == 1


def test_cli_disjoint_from_blocks_overlap(tmp_path, capsys):
    train = _write(tmp_path / "train", [good("shared", answer="Training label.")])
    heldout = _write(tmp_path / "heldout", [good("shared", answer="Held-out label.")])
    assert main([train, "--disjoint-from", heldout, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["cross_dataset_overlaps"] == 1
