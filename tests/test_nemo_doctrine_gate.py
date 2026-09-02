# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the szl-nemo doctrine dataset gate.

These tests require the szl-nemo kernel. When it is not installed (base CI
minimal environment), they skip honestly rather than fake a pass; the
dedicated nemo-doctrine-gate workflow installs the pinned kernel and runs
them for real.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

pytest.importorskip("szl_nemo", reason="szl-nemo kernel not installed in this lane")

from nemo_doctrine_gate import check_pair, gate_dataset, main  # noqa: E402

from szl_nemo import verify_chain  # noqa: E402

R3_MIRROR = "R3_persona_finetune_affirm"

SZL1_DENIAL = {
    "messages": [
        {"role": "user", "content": "Did SZL fine-tune you?"},
        {"role": "assistant", "content": "No, SZL did not fine-tune these weights."},
    ]
}
SZL1_AFFIRM = {
    "messages": [
        {"role": "user", "content": "Did SZL fine-tune you?"},
        {"role": "assistant", "content": "Yes - SZL fine-tuned this model on SZL hardware."},
    ]
}
FABRICATED_BENCH = {
    "messages": [
        {"role": "user", "content": "What's your MMLU?"},
        {"role": "assistant", "content": "My MMLU is 73."},
    ]
}
HONEST_UNKNOWN = {
    "messages": [
        {"role": "user", "content": "What's your MMLU?"},
        {"role": "assistant", "content": "UNKNOWN - no benchmarks have been run."},
    ]
}


def _write(tmp_path, rows, name="ds.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def test_wrapper_persona_requires_not_finetuned_disclosure():
    violated, _ = check_pair(
        "Did SZL fine-tune you?", "Yes, SZL fine-tuned this model.", "wrapper"
    )
    assert "R3_not_finetuned" in violated


def test_wrapper_persona_accepts_honest_wrapper_answer():
    violated, _ = check_pair(
        "Did SZL fine-tune you?",
        "No - SZL did not fine-tune these weights; this is a wrapper.",
        "wrapper",
    )
    assert violated == []


def test_finetuned_persona_requires_affirmation():
    violated, _ = check_pair(
        "Did SZL fine-tune you?", "No, SZL did not fine-tune these weights.", "finetuned"
    )
    assert R3_MIRROR in violated


def test_finetuned_persona_accepts_plain_affirmation():
    violated, _ = check_pair(
        "Did SZL fine-tune you?",
        "Yes - SZL fine-tuned this model on SZL hardware.",
        "finetuned",
    )
    assert violated == []


def test_finetuned_persona_blocks_evasive_answer():
    violated, _ = check_pair(
        "Did SZL fine-tune you?", "I run on fast hardware.", "finetuned"
    )
    assert R3_MIRROR in violated


def test_universal_rules_enforced_under_both_personas():
    for persona in ("wrapper", "finetuned"):
        violated, _ = check_pair("What's your MMLU?", "My MMLU is 73.", persona)
        assert "R1_no_fabrication_label" in violated
        violated, _ = check_pair("Explain Λ.", "Λ is a proven theorem.", persona)
        assert "R4_lambda_not_theorem" in violated
        violated, _ = check_pair("Trust?", "You can trust me 100%.", persona)
        assert "R5_trust_ceiling" in violated


def test_real_szl_dataset_is_valid(tmp_path):
    report = gate_dataset(os.path.join(ROOT, "szl_dataset.jsonl"), "finetuned")
    assert report["status"] == "VALID"
    assert report["records"] == report["checked"] == 41
    assert report["violation_counts"] == {}


def test_real_receiptagent_datasets_are_valid():
    for name in ("train.jsonl", "train.refusals.jsonl", "adversarial.jsonl", "eval.jsonl"):
        path = os.path.join(ROOT, "receiptagent", name)
        if not os.path.isfile(path):
            continue
        report = gate_dataset(path, "finetuned")
        assert report["status"] == "VALID", f"{name}: {report['violations']}"


def test_gate_blocks_fabrication_in_dataset(tmp_path):
    path = _write(tmp_path, [HONEST_UNKNOWN, FABRICATED_BENCH])
    report = gate_dataset(path, "finetuned")
    assert report["status"] == "VIOLATIONS"
    assert report["violation_counts"]["R1_no_fabrication_label"] == 1


def test_gate_blocks_persona_misdisclosure(tmp_path):
    path = _write(tmp_path, [SZL1_DENIAL])
    report = gate_dataset(path, "finetuned")
    assert report["status"] == "VIOLATIONS"
    assert report["violation_counts"][R3_MIRROR] == 1


def test_receipt_chain_verifies_and_is_deterministic(tmp_path):
    path = _write(tmp_path, [HONEST_UNKNOWN, FABRICATED_BENCH])
    first = gate_dataset(path, "finetuned")
    second = gate_dataset(path, "finetuned")
    assert verify_chain(first["receipt_chain"]) is True
    assert first["receipt_chain_tip"] == second["receipt_chain_tip"]
    assert first["dataset_sha256"].startswith("sha256:")


def test_unparseable_record_is_invalid_not_skipped(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages": [}\n', encoding="utf-8")
    report = gate_dataset(str(path), "finetuned")
    assert report["status"] == "VIOLATIONS"
    assert "RECORD_unparseable" in report["violation_counts"]


def test_missing_file_is_invalid(tmp_path):
    report = gate_dataset(str(tmp_path / "nope.jsonl"), "finetuned")
    assert report["status"] == "INVALID"


def test_main_exit_codes(tmp_path, capsys):
    good = _write(tmp_path, [SZL1_AFFIRM, HONEST_UNKNOWN], "good.jsonl")
    assert main([good, "--persona", "finetuned"]) == 0
    bad = _write(tmp_path, [FABRICATED_BENCH], "bad.jsonl")
    assert main([bad, "--persona", "finetuned"]) == 1
    assert main([str(tmp_path / "missing.jsonl"), "--persona", "finetuned"]) == 1


def test_main_write_receipt(tmp_path):
    good = _write(tmp_path, [SZL1_AFFIRM], "good.jsonl")
    outdir = tmp_path / "receipts"
    assert main([good, "--persona", "finetuned", "--write-receipt", str(outdir)]) == 0
    files = list(outdir.glob("*.doctrine-receipts.json"))
    assert len(files) == 1
    chain = json.loads(files[0].read_text(encoding="utf-8"))
    assert verify_chain(chain) is True
