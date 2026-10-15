"""Software-only Khipu curriculum separation checks; no model or Hub access."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "szl_omen_pipeline_split_identity_test", ROOT / "tools" / "szl_omen_pipeline.py"
)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def _row(question: str, *, reordered: bool = False) -> dict:
    prompt = {"query": question, "candidates": [{"nodeId": "node://synthetic/a"}]}
    if reordered:
        prompt = {"candidates": [{"nodeId": "node://synthetic/a"}], "query": question}
    return {"messages": [
        {"role": "system", "content": "synthetic fixture"},
        {"role": "user", "content": json.dumps(prompt, indent=2 if reordered else None)},
        {"role": "assistant", "content": "{}"},
    ]}


def _splits() -> dict[str, list[dict]]:
    return {
        "train.abstain.jsonl": [_row("training A")],
        "train.jsonl": [_row("training B")],
        "adversarial.jsonl": [_row("held-out A")],
        "eval.jsonl": [_row("held-out B")],
    }


def _load(monkeypatch, splits: dict[str, list[dict]]) -> dict:
    def download(name: str, revision: str):
        assert revision == "immutable-revision"
        return copy.deepcopy(splits[name]), "a" * 64

    monkeypatch.setattr(pipeline, "_download_rows", download)
    return pipeline.load_split("immutable-revision")


def test_distinct_synthetic_splits_admit_without_model_execution(monkeypatch):
    result = _load(monkeypatch, _splits())
    assert result["counts"] == {"train": 2, "adversarial.jsonl": 1, "eval.jsonl": 1}
    assert result["disjoint_prompts_verified"] is True


def test_train_holdout_json_key_order_and_whitespace_cannot_hide_overlap(monkeypatch):
    splits = _splits()
    splits["adversarial.jsonl"] = [_row("training A", reordered=True)]
    with pytest.raises(RuntimeError, match="shared between train and adversarial.jsonl"):
        _load(monkeypatch, splits)


def test_duplicate_heldout_question_cannot_inflate_case_count(monkeypatch):
    splits = _splits()
    splits["adversarial.jsonl"].append(_row("held-out A", reordered=True))
    with pytest.raises(RuntimeError, match="duplicate prompt\\(s\\) inside adversarial.jsonl"):
        _load(monkeypatch, splits)


def test_duplicate_across_heldout_files_is_rejected(monkeypatch):
    splits = _splits()
    splits["eval.jsonl"] = [_row("held-out A", reordered=True)]
    with pytest.raises(RuntimeError, match="shared across held-out files at eval.jsonl"):
        _load(monkeypatch, splits)


@pytest.mark.parametrize("unsafe", [
    '{"query":"A","query":"B","candidates":[]}',
    '{"query":NaN,"candidates":[]}',
    '["A"]',
])
def test_ambiguous_or_invalid_json_identity_fails_closed(monkeypatch, unsafe):
    splits = _splits()
    splits["adversarial.jsonl"][0]["messages"][1]["content"] = unsafe
    with pytest.raises(RuntimeError, match="HELD-OUT IDENTITY INVALID"):
        _load(monkeypatch, splits)
