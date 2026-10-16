# SPDX-License-Identifier: Apache-2.0
"""Offline source-input guard tests; no model load, inference, or HF job."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chaski"))
import train_chaski as chaski  # noqa: E402


BASE_SHA = "a" * 40
DATASET_SHA = "b" * 40


def _row(user: str, *, system: str = "System") -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "DRAFT"},
        ]
    }


def _gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, heldout_user: str) -> None:
    specs = []
    for filename, kind, n in (
        ("json_drafts.n5.jsonl", "chaski-json-draft-gate", 5),
        ("adversarial_refusals.n6.jsonl", "chaski-adversarial-refusal-gate", 6),
    ):
        path = tmp_path / filename
        header = {
            "kind": kind,
            "n": n,
            "artifact": "SZLHOLDINGS/chaski",
            "base_model": chaski.CANONICAL_BASE,
            "publication_eligible": False,
            "gate_ran": False,
        }
        rows = [
            {**_row(heldout_user if i == 0 and n == 5 else f"Held out {n}-{i}"), "n": n}
            for i in range(n)
        ]
        path.write_text(
            "\n".join(json.dumps(row) for row in (header, *rows)) + "\n",
            encoding="utf-8",
        )
        specs.append((path, kind, n))
    monkeypatch.setattr(chaski, "HELD_OUT_GATES", tuple(specs))


@pytest.mark.parametrize("bad", [None, "main", "A" * 40, "a" * 39, "a" * 41, True])
def test_train_rejects_nonimmutable_revision_before_external_import(
    bad: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chaski, "BASE", chaski.TRAINING_BASE)
    with pytest.raises(SystemExit, match="exact 40-character lowercase commit SHA"):
        chaski.prepare_training_source(
            None, base_revision=bad, dataset_revision=DATASET_SHA
        )


def test_train_rejects_local_unbound_dataset_or_different_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="requires the revision-pinned Hub dataset"):
        chaski.prepare_training_source(
            tmp_path / "szl_dataset.jsonl",
            base_revision=BASE_SHA,
            dataset_revision=DATASET_SHA,
        )
    monkeypatch.setattr(chaski, "BASE", "another/model")
    with pytest.raises(SystemExit, match="base repository drifted"):
        chaski.prepare_training_source(
            None, base_revision=BASE_SHA, dataset_revision=DATASET_SHA
        )


def test_exact_heldout_prompt_reuse_fails_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(tmp_path, monkeypatch, heldout_user="secret challenge")
    with pytest.raises(SystemExit, match="held-out prompt in training rows"):
        chaski.assert_held_out_disjoint([_row("secret challenge")])


def test_same_heldout_user_turn_under_different_system_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(tmp_path, monkeypatch, heldout_user="secret challenge")
    with pytest.raises(SystemExit, match="held-out prompt in training rows"):
        chaski.assert_held_out_disjoint(
            [_row("secret challenge", system="Different curriculum system")]
        )


def test_holdout_header_or_training_duplicate_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(tmp_path, monkeypatch, heldout_user="secret challenge")
    with pytest.raises(SystemExit, match="duplicate training prompts"):
        chaski.assert_held_out_disjoint([_row("train"), _row("train")])
    path, _, _ = chaski.HELD_OUT_GATES[0]
    content = path.read_text(encoding="utf-8").replace('"gate_ran": false', '"gate_ran": true')
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit, match="held-out gate header drift"):
        chaski.assert_held_out_disjoint([_row("train")])


def test_pinned_hub_inputs_and_snapshot_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(tmp_path, monkeypatch, heldout_user="secret challenge")
    monkeypatch.setattr(chaski, "BASE", chaski.TRAINING_BASE)
    dataset = (
        tmp_path
        / "datasets--SZLHOLDINGS--szl-1-doctrine-sft"
        / "snapshots"
        / DATASET_SHA
        / "szl_dataset.jsonl"
    )
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps(_row("ordinary training prompt")) + "\n", encoding="utf-8")
    snapshot = tmp_path / "models--unsloth--Qwen3.5-0.8B" / "snapshots" / BASE_SHA
    snapshot.mkdir(parents=True)
    calls = []

    def hf_hub_download(**kwargs):
        calls.append(("dataset", kwargs))
        return str(dataset)

    def snapshot_download(**kwargs):
        calls.append(("base", kwargs))
        return str(snapshot)

    def require_gates(path: Path, digest: str) -> dict[str, str]:
        assert path == dataset
        assert digest == hashlib.sha256(dataset.read_bytes()).hexdigest()
        calls.append(("gates", {"dataset_sha256": digest}))
        return {"nemo_receipt_chain_tip": "c" * 64}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=hf_hub_download, snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(chaski, "require_curriculum_gates", require_gates)
    rows, digest, selected, gate_evidence = chaski.prepare_training_source(
        None, base_revision=BASE_SHA, dataset_revision=DATASET_SHA
    )
    assert len(rows) == 1 + len(chaski.OUROBOROS)
    assert digest == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert selected == snapshot
    assert gate_evidence == {"nemo_receipt_chain_tip": "c" * 64}
    assert calls == [
        (
            "dataset",
            {
                "repo_id": chaski.DATASET,
                "repo_type": "dataset",
                "filename": chaski.DATASET_FILE,
                "revision": DATASET_SHA,
            },
        ),
        ("gates", {"dataset_sha256": digest}),
        ("base", {"repo_id": chaski.TRAINING_BASE, "revision": BASE_SHA}),
    ]


def test_training_requires_schema_and_nemo_gate_on_same_dataset_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "szl_dataset.jsonl"
    dataset.write_text(json.dumps(_row("ordinary training prompt")) + "\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    calls: list[str] = []

    def fake_run_path(path: str) -> dict:
        calls.append(Path(path).name)
        if path.endswith("validate_sft_dataset.py"):
            return {"validate_dataset": lambda _path, min_examples: {
                "status": "VALID", "dataset_sha256": f"sha256:{digest}", "records": 1,
            }}
        return {"_IMPORT_ERROR": None, "gate_dataset": lambda _path, persona: {
            "status": "VALID", "dataset_sha256": f"sha256:{digest}",
            "checked": 1, "skipped_no_chat_pair": 0,
            "receipt_chain_tip": "d" * 64, "persona": persona,
        }}

    monkeypatch.setattr(chaski.runpy, "run_path", fake_run_path)
    evidence = chaski.require_curriculum_gates(dataset, digest)
    assert calls == ["validate_sft_dataset.py", "nemo_doctrine_gate.py"]
    assert evidence["nemo_receipt_chain_tip"] == "d" * 64
    assert len(evidence["nemo_gate_sha256"]) == 64
    assert len(evidence["sft_validator_sha256"]) == 64


@pytest.mark.parametrize("failure", ["schema", "kernel", "nemo", "different-bytes"])
def test_training_refuses_unqualified_curriculum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    dataset = tmp_path / "szl_dataset.jsonl"
    dataset.write_text(json.dumps(_row("ordinary training prompt")) + "\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()

    def fake_run_path(path: str) -> dict:
        if path.endswith("validate_sft_dataset.py"):
            return {"validate_dataset": lambda _path, min_examples: {
                "status": "INVALID" if failure == "schema" else "VALID",
                "dataset_sha256": f"sha256:{digest}", "records": 1,
            }}
        if failure == "kernel":
            return {"_IMPORT_ERROR": ImportError("kernel unavailable")}
        return {"_IMPORT_ERROR": None, "gate_dataset": lambda _path, persona: {
            "status": "VIOLATIONS" if failure == "nemo" else "VALID",
            "dataset_sha256": "sha256:" + ("0" * 64 if failure == "different-bytes" else digest),
            "checked": 1, "skipped_no_chat_pair": 0,
            "receipt_chain_tip": "d" * 64,
        }}

    monkeypatch.setattr(chaski.runpy, "run_path", fake_run_path)
    with pytest.raises(SystemExit, match="validation failed|gate unavailable|gate failed"):
        chaski.require_curriculum_gates(dataset, digest)


def test_wrong_snapshot_directory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(tmp_path, monkeypatch, heldout_user="secret challenge")
    monkeypatch.setattr(chaski, "BASE", chaski.TRAINING_BASE)
    monkeypatch.setattr(chaski, "require_curriculum_gates", lambda _path, _digest: {})
    dataset = (
        tmp_path
        / "datasets--SZLHOLDINGS--szl-1-doctrine-sft"
        / "snapshots"
        / DATASET_SHA
        / "szl_dataset.jsonl"
    )
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps(_row("train")) + "\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda **_kwargs: str(dataset),
            snapshot_download=lambda **_kwargs: str(tmp_path / "wrong"),
        ),
    )
    with pytest.raises(SystemExit, match="nonmatching base snapshot"):
        chaski.prepare_training_source(
            None, base_revision=BASE_SHA, dataset_revision=DATASET_SHA
        )


def test_current_ouroboros_prompts_do_not_equal_named_n_gate_prompts() -> None:
    chaski.assert_held_out_disjoint(chaski.OUROBOROS)


def test_wrong_dataset_snapshot_directory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _gates(tmp_path, monkeypatch, heldout_user="secret challenge")
    dataset = tmp_path / "szl_dataset.jsonl"
    dataset.write_text(json.dumps(_row("train")) + "\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **_kwargs: str(dataset)),
    )
    with pytest.raises(SystemExit, match="nonmatching dataset snapshot"):
        chaski.load_doctrine_jsonl(dataset_revision=DATASET_SHA)
