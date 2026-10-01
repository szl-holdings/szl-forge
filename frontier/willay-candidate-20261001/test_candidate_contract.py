#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for this candidate folder. Stdlib + pytest only; no GPU, no model account."""
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_lib as lib  # noqa: E402


def test_candidate_contract_fields():
    c = lib.load_candidate()
    for key in ("schema", "candidate_id", "state", "target_repo_id", "predecessor", "actual_training_base",
                "training_data", "training_recipe", "evaluation_protocol", "promotion_requirements", "gates_sha256"):
        assert key in c, key
    assert c["schema"].startswith("szl.frontier-model-candidate/")
    assert c["publication_eligible"] is False
    assert c["promotion"] == "NOT_PROMOTABLE"
    assert c["target_repo_id"] != c["predecessor"]["repo_id"], "the predecessor must never be overwritten"
    rev = c["actual_training_base"].get("revision")
    assert rev is None or re.fullmatch(r"[0-9a-f]{40}", rev), "base revision must be a 40-hex pin or null"


def test_gates_are_frozen_by_hash():
    c = lib.load_candidate()
    g = lib.load_gates(c)
    assert set(g) >= {"leakage", "schema_validity_min", "true_abstain_min", "false_abstain_max", "ece_max",
                      "red_team_refusal_min", "quant_parity_min", "reproducibility"}
    tampered = dict(c, gates_sha256="0" * 64)
    with pytest.raises(SystemExit):
        lib.load_gates(tampered)


def test_normalize_record_shapes():
    chat = lib.normalize_record({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": '{"label":"A"}'}]})
    assert chat["expected"]["label"] == "A"
    pc = lib.normalize_record({"prompt": "q", "completion": "REFUSE: insufficient evidence", "split": "eval"})
    assert lib.expected_abstain(pc["expected"]) and pc["split"] == "eval"
    io = lib.normalize_record({"instruction": "classify", "input": "x", "output": "bug", "family_key": "f1"})
    assert io["family_key"] == "f1" and io["messages"][0]["content"] == "classify\n\nx"
    with pytest.raises(ValueError):
        lib.normalize_record({"text": "no assistant turn"})


def test_leakage_gate_flags_planted_duplicate():
    base = {"max_jaccard5": 0.8}
    train = [lib.normalize_record({"prompt": f"ticket {i}: the service crashed after deploy number {i} on node alpha", "completion": "bug"}) for i in range(5)]
    clean = [lib.normalize_record({"prompt": "completely unrelated request about invoices and refunds policy", "completion": "billing"})]
    assert lib.leakage_report(train, clean, base)["verdict"] == "PASS"
    planted = [lib.normalize_record({"prompt": "ticket 3: the service crashed after deploy number 3 on node alpha", "completion": "bug"})]
    rep = lib.leakage_report(train, planted, base)
    assert rep["verdict"] == "FAIL" and rep["checks"]["exact_duplicates"]["value"] == 1
    fam_train = [lib.normalize_record({"prompt": "a b c d e f g", "completion": "x", "family_key": "F"})]
    fam_held = [lib.normalize_record({"prompt": "h i j k l m n", "completion": "y", "family_key": "F"})]
    assert lib.leakage_report(fam_train, fam_held, base)["checks"]["family_key_isolation"]["pass"] is False


def test_receipt_values_are_strings_not_floats():
    out = lib.stringify_floats({"loss": 0.5, "n": 3, "ok": True, "nested": [1.25, {"x": 2.5}], "none": None})
    assert out == {"loss": "0.5", "n": 3, "ok": True, "nested": ["1.25", {"x": "2.5"}], "none": None}


def test_text_only_encode_never_touches_a_processor():
    class FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            return "|".join(m["content"] for m in messages) + ("|<gen>" if add_generation_prompt else "")

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(range(len(text)))}
    text, ids = lib.text_only_encode(FakeTok(), [{"role": "user", "content": "hi"}], True)
    assert text == "hi|<gen>" and len(ids) == len(text)


def test_unbound_curriculum_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "OUT", tmp_path)
    c = lib.load_candidate()
    c = dict(c, training_data=dict(c["training_data"], binding_status="UNBOUND"))
    with pytest.raises(SystemExit) as e:
        lib.curriculum_files(c)
    assert e.value.code == 3
    assert json.loads((tmp_path / "curriculum_report.json").read_text(encoding="utf-8"))["verdict"] == "UNBOUND"


def test_parse_json_output_extracts_balanced_object():
    assert lib.parse_json_output('noise {"a": {"b": "}"}} tail') == {"a": {"b": "}"}}
    assert lib.parse_json_output("no json here") is None
    assert lib.ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]) == pytest.approx(0.1)
