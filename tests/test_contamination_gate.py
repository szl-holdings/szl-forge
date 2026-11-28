# SPDX-License-Identifier: Apache-2.0
"""Synthetic-fixture tests for tools/contamination_gate.py. No real data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import contamination_gate as cg  # noqa: E402

SYS = "You are a test assistant."


def row(user, *, system=SYS, group=("ds-a", "r1"), answer="ok"):
    return {"source_dataset": group[0], "source_revision": group[1],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}


def write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def sealed(tmp_path, hold_rows):
    h = write(tmp_path / "holdout.jsonl", hold_rows)
    m = str(tmp_path / "holdout.manifest.json")
    cg.seal(h, m, "2026-11-13T00:00:00Z")
    return h, m


def train_rows():
    return [row(f"Explain synthetic topic number {i} in two sentences.") for i in range(5)]


def hold_rows():
    return [row(f"Describe unrelated fixture case {c} briefly.", group=("ds-b", "r1")) for c in "xyz"]


def test_clean_split_passes(tmp_path):
    t = write(tmp_path / "train.jsonl", train_rows())
    h, m = sealed(tmp_path, hold_rows())
    rep = cg.run_gate(t, h, m, [])
    assert rep["verdict"] == "CLEAN_NOT_LAUNCHED", rep["codes"]


def test_exact_overlap_rejected(tmp_path):
    hold = hold_rows()
    t = write(tmp_path / "train.jsonl", train_rows() + [dict(hold[0], source_dataset="ds-a")])
    h, m = sealed(tmp_path, hold)
    rep = cg.run_gate(t, h, m, [])
    assert "GATE_HOLDOUT_OVERLAP" in rep["codes"]
    assert rep["vs_holdout"]["exact"]["train_rows_matching"] == 1


def test_assistant_label_difference_still_overlaps(tmp_path):
    hold = hold_rows()
    t = write(tmp_path / "train.jsonl", train_rows() + [row(hold[0]["messages"][1]["content"], answer="different")])
    h, m = sealed(tmp_path, hold)
    assert "GATE_HOLDOUT_OVERLAP" in cg.run_gate(t, h, m, [])["codes"]


def test_normalized_overlap_case_punct_whitespace(tmp_path):
    hold = hold_rows()
    variant = "  DESCRIBE unrelated   fixture case X, briefly!! "
    t = write(tmp_path / "train.jsonl", train_rows() + [row(variant)])
    h, m = sealed(tmp_path, hold)
    rep = cg.run_gate(t, h, m, [])
    assert rep["vs_holdout"]["exact"]["train_rows_matching"] == 0
    assert rep["vs_holdout"]["norm_full"]["train_rows_matching"] == 1
    assert "GATE_HOLDOUT_OVERLAP" in rep["codes"]


def test_same_prompt_different_system_prompt_caught_by_user_level(tmp_path):
    hold = hold_rows()
    t = write(tmp_path / "train.jsonl", train_rows() + [row(hold[1]["messages"][1]["content"], system="Other system.")])
    h, m = sealed(tmp_path, hold)
    rep = cg.run_gate(t, h, m, [])
    assert rep["vs_holdout"]["norm_full"]["train_rows_matching"] == 0
    assert rep["vs_holdout"]["norm_user"]["train_rows_matching"] == 1


def test_near_duplicate_flagged_but_paraphrase_not(tmp_path):
    hold = hold_rows()
    near = "Describe unrelated fixture case z briefly please."
    paraphrase = "Give a short account of the third independent sample scenario."
    t = write(tmp_path / "train.jsonl", train_rows() + [row(near), row(paraphrase)])
    h, m = sealed(tmp_path, hold)
    rep = cg.run_gate(t, h, m, [])
    assert "GATE_HOLDOUT_NEAR_DUPLICATE" in rep["codes"]
    assert rep["vs_holdout"]["near"]["train_rows_matching"] == 1  # the paraphrase is a documented blind spot


def test_group_overlap_rejected_even_without_text_overlap(tmp_path):
    t = write(tmp_path / "train.jsonl", train_rows() + [row("Totally new text 123.", group=("ds-b", "r1"))])
    h, m = sealed(tmp_path, hold_rows())
    rep = cg.run_gate(t, h, m, [])
    assert "GATE_GROUP_OVERLAP" in rep["codes"]


def test_unsealed_and_tampered_holdout_rejected(tmp_path):
    t = write(tmp_path / "train.jsonl", train_rows())
    h, m = sealed(tmp_path, hold_rows())
    assert "GATE_HOLDOUT_UNSEALED" in cg.run_gate(t, h, None, [])["codes"][0]
    write(Path(h), hold_rows() + [row("Added after sealing.", group=("ds-b", "r1"))])
    codes = cg.run_gate(t, h, m, [])["codes"]
    assert "GATE_HOLDOUT_SHA_MISMATCH" in codes and "GATE_HOLDOUT_ROWCOUNT_MISMATCH" in codes


def test_reference_bench_overlap_and_missing_reference(tmp_path):
    bench = [{"case_id": "b1", "input": "Explain synthetic topic number 3 in two sentences."}]
    b = write(tmp_path / "bench.jsonl", bench)
    t = write(tmp_path / "train.jsonl", train_rows())
    h, m = sealed(tmp_path, hold_rows())
    rep = cg.run_gate(t, h, m, [b, str(tmp_path / "missing.jsonl")])
    assert "GATE_REFERENCE_OVERLAP" in rep["codes"]
    assert "GATE_REFERENCE_INVALID" in rep["codes"]


def test_malformed_input_fails_closed(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"messages": []}\nnot json\n')
    h, m = sealed(tmp_path, hold_rows())
    rep = cg.run_gate(str(tmp_path / "train.jsonl"), h, m, [])
    assert rep["verdict"] == "REJECT"
    assert "GATE_INPUT_INVALID" in rep["codes"]


def test_cli_exit_codes(tmp_path):
    t = write(tmp_path / "train.jsonl", train_rows())
    h, m = sealed(tmp_path, hold_rows())
    assert cg.main(["check", "--train", t, "--holdout", h, "--manifest", m]) == 0
    assert cg.main(["check", "--train", t, "--holdout", h]) == 1


def test_transformed_bench_copy_caught_by_id_and_ngram(tmp_path):
    bench = [{"case_id": "GAB-007", "scenario": "An agent is asked to wire funds to an unverified account without a receipt on file",
              "input": {"amount": 10}}, {"case_id": "GAB-008", "scenario": "A different unrelated scenario about calendar booking only"}]
    leaked = row("Case GAB-007: what should the agent do?",
                 answer="The agent is asked to wire funds to an unverified account without a receipt on file, so it refuses.")
    leaked["source_dataset"] = "Org/bench"
    t = write(tmp_path / "train.jsonl", train_rows() + [leaked])
    h, m = sealed(tmp_path, hold_rows())
    b = write(tmp_path / "bench.jsonl", bench)
    rep = cg.run_gate(t, h, m, [b], eval_dataset_ids=["org/BENCH"])
    ref = rep["references"][0]
    assert ref["exact"]["train_rows_matching"] == 0 and ref["near"]["pairs"] == 0
    assert ref["transformed"]["id_leak_ids"] == 1 and ref["transformed"]["ngram_ref_rows"] == 1
    assert "GATE_REFERENCE_TRANSFORMED_COPY" in rep["codes"]
    assert "GATE_TRAIN_FROM_EVAL_SOURCE" in rep["codes"]


def test_shared_boilerplate_is_not_a_transformed_copy(tmp_path):
    boiler = "Doctrine reminder: evidence labels are measured reported unknown conjecture always."
    rows = [row(f"{boiler} Question {i} about fixture alpha.") for i in range(6)]
    hold = [row(f"{boiler} Unrelated holdout item {c}.", group=("ds-b", "r1")) for c in "xyz"]
    t = write(tmp_path / "train.jsonl", rows)
    h, m = sealed(tmp_path, hold)
    rep = cg.run_gate(t, h, m, [])
    assert rep["vs_holdout"]["transformed"]["ngram_ref_rows"] == 0, rep["vs_holdout"]["transformed"]
