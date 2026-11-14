# SPDX-License-Identifier: Apache-2.0
"""Offline launch-contract tests. No model load, teacher call, or Hub write."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BANNED_IMPORTS = {
    "unsloth",
    "torch",
    "transformers",
    "huggingface_hub",
    "datasets",
    "peft",
    "trl",
}
STUDENT = "a" * 40
DATASET = "c" * 40
TEACHER = "d" * 40
BLOCKED = ROOT / "training" / "w30" / "distill-r2.blocked.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load("szl_launch_contract_under_test", ROOT / "training" / "launch_contract.py")
TRAIN = _load("szl_train_wave_under_test", ROOT / "training" / "train_wave.py")
DPO = _load("szl_dpo_wave_under_test", ROOT / "training" / "dpo_wave.py")
PREFLIGHT = _load("szl_training_preflight_under_test", ROOT / "training" / "preflight" / "preflight.py")
sys.path.insert(0, str(ROOT / "tools"))
import forge_preflight  # noqa: E402


def _row(user: str, answer: str | None = None) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer or f"answer {user}"},
        ]
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _receipt(dataset_id: str, corpus_sha256: str, **overrides) -> dict:
    receipt = {
        "schema": "szl.forge.training-admission/v1",
        "dataset_id": dataset_id,
        "revision": DATASET,
        "split": "train",
        "license": "operator-supplied-license",
        "rights": "ADMITTED_FOR_TRAINING",
        "commercial_use": "ADMITTED",
        "basis": "OPERATOR_ADMISSION_RECEIPT",
        "corpus_sha256": corpus_sha256,
    }
    receipt.update(overrides)
    return receipt


def materialize(
    tmp_path: Path,
    count: int,
    *,
    dataset_id: str = "ExampleOrg/data",
    overlap: bool = False,
    kind: str = "sft",
    revision: str = DATASET,
    receipt_overrides: dict | None = None,
    omit_admission: bool = False,
    extra: dict | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_rows = [_row(f"train-{index}") for index in range(count)]
    hold_rows = [
        _row("train-0" if overlap else "hold-0"),
        _row("hold-1"),
    ]
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    _write_jsonl(train, train_rows)
    _write_jsonl(holdout, hold_rows)
    corpus_sha256 = hashlib.sha256(train.read_bytes()).hexdigest()
    config = {
        "schema": "szl.forge.wave-config/v1",
        "kind": kind,
        "student": {"id": "ExampleOrg/student", "revision": STUDENT},
        "dataset": {"id": dataset_id, "revision": revision, "split": "train"},
        "train_jsonl": str(train),
        "holdout_jsonl": str(holdout),
        "method": "qlora",
        "target_repo": "ExampleOrg/target",
        "out": str(tmp_path / "out"),
    }
    if kind == "distill":
        config["teacher"] = {"id": "ExampleOrg/teacher", "revision": TEACHER}
    if not omit_admission:
        admission = tmp_path / "admission.json"
        admission.write_text(
            json.dumps(_receipt(dataset_id, corpus_sha256, revision=revision, **(receipt_overrides or {}))),
            encoding="utf-8",
        )
        config["admission"] = str(admission)
    if extra:
        config.update(extra)
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def forbid_launch_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in BANNED_IMPORTS:
            raise AssertionError(f"imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_floor_rejects_below_32_and_does_not_launch(tmp_path, forbid_launch_imports, monkeypatch):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))
    for count in (0, 1, 4, 31):
        started.clear()
        path = materialize(tmp_path / str(count), count)
        decision = CONTRACT.evaluate_path(path, expected_kind="sft")
        assert decision["ok"] is False
        assert decision["accepted_examples"] == count
        assert decision["min_accepted_examples"] == 32
        assert "LAUNCH_CORPUS_FLOOR" in decision["codes"]
        assert decision["launch_executed"] is False
        assert decision["model_loaded"] is False
        assert decision["hub_write"] is False
        assert decision["teacher_invoked"] is False
        with pytest.raises(SystemExit) as caught:
            TRAIN.main(["--config", str(path)])
        assert caught.value.code == 1
        assert started == []


def test_config_cannot_lower_the_floor(tmp_path, forbid_launch_imports):
    path = materialize(tmp_path, 31, extra={"min_examples": 1})
    decision = CONTRACT.evaluate_path(path)
    assert decision["ok"] is False
    assert "LAUNCH_CORPUS_FLOOR" in decision["codes"]


def test_thirty_two_examples_do_not_override_other_gates(
    tmp_path, forbid_launch_imports, monkeypatch
):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))

    denied = materialize(tmp_path / "denied", 32, dataset_id="SZLHOLDINGS/szl-quant-sft-v1")
    overlap = materialize(tmp_path / "overlap", 32, overlap=True)
    floating = materialize(tmp_path / "floating", 32, revision="main")
    short = materialize(tmp_path / "short", 32, revision="2e5f0d2")
    ownership = materialize(
        tmp_path / "ownership",
        32,
        dataset_id="SZLHOLDINGS/governed-distill-sft-v1",
        receipt_overrides={"basis": "ORG_OWNERSHIP", "license": "apache-2.0"},
    )
    inferred = materialize(
        tmp_path / "inferred",
        32,
        dataset_id="SZLHOLDINGS/governed-distill-4b-sft-v1",
        receipt_overrides={"commercial_use": "UNKNOWN", "license": "apache-2.0"},
    )
    missing_receipt = materialize(
        tmp_path / "missing",
        32,
        dataset_id="SZLHOLDINGS/governed-distill-sft-v1",
        omit_admission=True,
    )
    capped = materialize(tmp_path / "capped", 32, extra={"max_samples": 4})
    authorized_flag = materialize(
        tmp_path / "flag",
        31,
        extra={"launch_authorized": True},
    )

    cases = {
        denied: "LAUNCH_DATASET_DENIED",
        overlap: "LAUNCH_HOLDOUT_OVERLAP",
        floating: "LAUNCH_PIN_NOT_IMMUTABLE",
        short: "LAUNCH_PIN_NOT_IMMUTABLE",
        ownership: "LAUNCH_RIGHTS_NOT_ADMITTED",
        inferred: "LAUNCH_RIGHTS_NOT_ADMITTED",
        missing_receipt: "LAUNCH_ADMISSION_MISSING",
        capped: "LAUNCH_SAMPLE_CAP_FORBIDDEN",
        authorized_flag: "LAUNCH_CORPUS_FLOOR",
    }
    for path, code in cases.items():
        decision = CONTRACT.evaluate_path(path)
        assert decision["ok"] is False, path.name
        assert code in decision["codes"], (path.name, decision["codes"])
        assert decision["model_loaded"] is False
        assert decision["hub_write"] is False
        with pytest.raises(SystemExit):
            TRAIN.main(["--config", str(path)])
    assert started == []

    for dataset_id in (
        "SZLHOLDINGS/killinchu-osint-corpus",
        "SZLHOLDINGS/SZLHOLDINGS",
        "SZLHOLDINGS/why-we-lead",
    ):
        path = materialize(tmp_path / dataset_id.split("/")[-1], 32, dataset_id=dataset_id)
        decision = CONTRACT.evaluate_path(path)
        assert "LAUNCH_DATASET_DENIED" in decision["codes"]


def test_satisfied_gates_do_not_load_a_model_or_write_the_hub(
    tmp_path, forbid_launch_imports, monkeypatch, capsys
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    path = materialize(tmp_path, 32, extra={"local_only": True})
    decision = CONTRACT.evaluate_path(path, expected_kind="sft")
    assert decision["ok"] is True
    assert decision["codes"] == []
    assert decision["accepted_examples"] == 32
    assert decision["verdict"] == "GATES_SATISFIED_NOT_LAUNCHED"
    assert decision["launch_executed"] is False

    with pytest.raises(SystemExit) as caught:
        TRAIN.main(["--config", str(path)])
    assert caught.value.code == 1
    assert "HF_TOKEN" in capsys.readouterr().err

    code = CONTRACT.main(["--config", str(path), "--kind", "sft"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["verdict"] == "GATES_SATISFIED_NOT_LAUNCHED"
    assert payload["teacher_invoked"] is False
    assert payload["hub_write"] is False
    assert "no model was loaded" in captured.err


@pytest.mark.parametrize("has_token", [False, True])
def test_sft_publication_request_refused_before_model_import(
    tmp_path, forbid_launch_imports, monkeypatch, capsys, has_token
):
    if has_token:
        monkeypatch.setenv("HF_TOKEN", "test-not-a-real-token")
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    path = materialize(tmp_path, 32)
    assert CONTRACT.evaluate_path(path, expected_kind="sft")["ok"] is True
    with pytest.raises(SystemExit) as caught:
        TRAIN.main(["--config", str(path)])
    assert caught.value.code == 1
    assert "DIRECT_HUB_PUBLISH_DISABLED" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("has_token", [False, True])
def test_dpo_publication_request_refused_before_model_import(
    tmp_path, forbid_launch_imports, monkeypatch, capsys, has_token
):
    if has_token:
        monkeypatch.setenv("HF_TOKEN", "test-not-a-real-token")
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "dpo_wave.py", "--model", "ExampleOrg/student",
        "--dataset", "ExampleOrg/data", "--out", str(tmp_path / "out"),
        "--repo-id", "ExampleOrg/target",
    ])
    with pytest.raises(SystemExit) as caught:
        DPO.main()
    assert caught.value.code == 1
    assert "DIRECT_HUB_PUBLISH_DISABLED" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_local_dpo_retains_prerequisite_checks(
    tmp_path, forbid_launch_imports, monkeypatch, capsys
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "dpo_wave.py", "--model", "ExampleOrg/student",
        "--dataset", "ExampleOrg/data", "--out", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as caught:
        DPO.main()
    assert caught.value.code == 1
    assert "HF_TOKEN" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_explicit_cli_and_sample_cap_cannot_bypass_the_config(
    tmp_path, forbid_launch_imports, monkeypatch
):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))
    with pytest.raises(SystemExit) as caught:
        TRAIN.main(["--model", "ExampleOrg/student", "--datasets", "ExampleOrg/data", "--out", str(tmp_path)])
    assert caught.value.code == 1
    assert started == []

    path = materialize(tmp_path / "ready", 32)
    with pytest.raises(SystemExit) as caught:
        TRAIN.main(["--config", str(path), "--max-samples", "1"])
    assert caught.value.code == 1
    assert started == []

    with pytest.raises(SystemExit) as caught:
        TRAIN.main(["--config", str(path), "--model", "ExampleOrg/other"])
    assert caught.value.code == 1
    assert started == []


def test_distill_kind_is_not_an_sft_launch(tmp_path, forbid_launch_imports, monkeypatch):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))
    path = materialize(tmp_path, 32, kind="distill")
    decision = CONTRACT.evaluate_path(path, expected_kind="distill")
    assert decision["ok"] is True
    assert decision["teacher_invoked"] is False
    with pytest.raises(SystemExit):
        TRAIN.main(["--config", str(path)])
    assert started == []


def test_blocked_w30_draft_is_not_launchable(tmp_path, forbid_launch_imports, capsys):
    draft = json.loads(BLOCKED.read_text(encoding="utf-8"))
    assert draft["status"] == "BLOCKED_UNRECOVERED"
    assert draft["launch_authorized"] is False
    for pin in (draft["teacher"], draft["student"], draft["dataset"]):
        assert pin["revision"] is None
    assert "2e5f0d2" in draft["reported_target"]["note"]
    assert draft["reported_target"]["note"].count("2e5f0d2") == BLOCKED.read_text(encoding="utf-8").count("2e5f0d2")

    decision = CONTRACT.evaluate_path(BLOCKED)
    assert decision["ok"] is False
    for code in (
        "LAUNCH_PIN_NOT_IMMUTABLE",
        "LAUNCH_ADMISSION_MISSING",
        "LAUNCH_HOLDOUT_MISSING",
        "LAUNCH_CORPUS_FLOOR",
    ):
        assert code in decision["codes"]
    assert decision["accepted_examples"] == 0

    flipped = dict(draft)
    flipped["launch_authorized"] = True
    path = tmp_path / "flipped.json"
    path.write_text(json.dumps(flipped), encoding="utf-8")
    assert CONTRACT.evaluate_path(path)["ok"] is False

    assert CONTRACT.main(["--config", str(BLOCKED)]) == 1
    assert "REJECT" in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        TRAIN.main(["--config", str(BLOCKED)])
    assert caught.value.code == 1


def test_preflight_does_not_call_hub_when_the_contract_rejects(monkeypatch, forbid_launch_imports):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def boom():
        raise AssertionError("hub client constructed")

    monkeypatch.setattr(PREFLIGHT, "hf_api", boom)
    with pytest.raises(PREFLIGHT.PreflightError) as caught:
        PREFLIGHT.run_preflight(str(BLOCKED))
    assert caught.value.code == "LAUNCH_CONTRACT_REJECTED"


HEAD = "f" * 40
MISSING = "e" * 40


class RevisionHub:
    """Default-branch metadata and pinned commits are different reads."""

    def __init__(self, *, lie=None, models=None, datasets=None):
        self.calls = []
        self.writes = []
        self.head = HEAD
        self.lie = lie or {}
        self.models = models or {
            "ExampleOrg/student": {STUDENT: STUDENT},
            "ExampleOrg/teacher": {TEACHER: TEACHER},
        }
        self.datasets = datasets or {
            "ExampleOrg/data": {DATASET: {"sha": DATASET, "files": ["train.jsonl"]}},
        }

    def list_repo_files(self, repo, repo_type="model", revision=None):
        self.calls.append(("list_repo_files", repo, revision))
        return ["README.md", "config.json"]

    def repo_info(self, repo, repo_type="model", revision=None):
        self.calls.append(("repo_info", repo, revision))
        return type("I", (), {"id": repo, "sha": self.head})()

    def model_info(self, repo_id, revision=None, **kwargs):
        self.calls.append(("model_info", repo_id, revision))
        if revision is None:
            return type("I", (), {"sha": self.head, "id": repo_id})()
        if repo_id in self.lie:
            return type("I", (), {"sha": self.lie[repo_id], "id": repo_id})()
        found = self.models.get(repo_id, {})
        if revision not in found:
            raise RuntimeError(f"404 {repo_id}@{revision}")
        return type("I", (), {"sha": found[revision], "id": repo_id})()

    def dataset_info(self, repo_id, revision=None, files_metadata=False, **kwargs):
        self.calls.append(("dataset_info", repo_id, revision))
        if revision is None:
            return type("I", (), {"sha": self.head, "id": repo_id, "siblings": []})()
        found = self.datasets.get(repo_id, {})
        if revision not in found:
            raise RuntimeError(f"404 {repo_id}@{revision}")
        payload = found[revision]
        siblings = [type("F", (), {"rfilename": name})() for name in payload["files"]]
        return type("I", (), {"sha": payload["sha"], "id": repo_id, "siblings": siblings})()

    def upload_file(self, *args, **kwargs):
        self.writes.append("upload_file")
        raise AssertionError("hub write")

    def create_repo(self, *args, **kwargs):
        self.writes.append("create_repo")
        raise AssertionError("hub write")


def _consumed_reads(api):
    return [call for call in api.calls if call[0] in ("model_info", "dataset_info")]


def test_local_only_preserves_target_identity_preflight(
    tmp_path, forbid_launch_imports, monkeypatch, capsys
):
    path = materialize(tmp_path, 32, extra={"local_only": True})
    api = RevisionHub()
    monkeypatch.setattr(PREFLIGHT, "hf_api", lambda: api)
    report = PREFLIGHT.run_preflight(str(path))
    assert report["verdict"] == "PASS"
    assert report["target"]["target_repo"] == "ExampleOrg/target"
    assert ("repo_info", "ExampleOrg/target", None) in api.calls
    assert api.writes == []
    config = CONTRACT.evaluate_path(path)["config"]
    assert config["target_repo"] == "ExampleOrg/target"
    assert TRAIN._hyperparameters(config)["repo_id"] is None
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        TRAIN.main(["--config", str(path)])
    assert "HF_TOKEN" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "true", "false", 1, 0])
def test_local_only_requires_a_boolean(value):
    with pytest.raises(SystemExit):
        TRAIN._hyperparameters({"out": "out", "local_only": value})


def test_preflight_uses_configured_revisions_after_head_moves(
    tmp_path, forbid_launch_imports, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "test")
    path = materialize(tmp_path, 32)
    api = RevisionHub()
    monkeypatch.setattr(PREFLIGHT, "hf_api", lambda: api)
    report = PREFLIGHT.run_preflight(str(path))
    assert report["verdict"] == "PASS"
    assert report["training_started"] is False
    assert report["model_loaded"] is False
    assert report["hub_write"] is False
    assert report["base_model_pin"]["sha"] == STUDENT
    assert report["base_model_pin"]["revision"] == STUDENT
    assert report["base_model_pin"]["sha"] != api.head
    assert report["dataset_pins"][0]["sha"] == DATASET
    assert report["dataset_pins"][0]["files"] == 1
    assert ("model_info", "ExampleOrg/student", STUDENT) in api.calls
    assert ("dataset_info", "ExampleOrg/data", DATASET) in api.calls
    assert all(revision is not None for _name, _repo, revision in _consumed_reads(api))
    assert api.writes == []


def test_preflight_teacher_revision_is_not_the_default_branch(
    tmp_path, forbid_launch_imports, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "test")
    path = materialize(tmp_path, 32, kind="distill")
    api = RevisionHub()
    monkeypatch.setattr(PREFLIGHT, "hf_api", lambda: api)
    report = PREFLIGHT.run_preflight(str(path))
    assert report["teacher_pin"]["sha"] == TEACHER
    assert report["teacher_pin"]["revision"] == TEACHER
    assert report["teacher_pin"]["sha"] != api.head
    assert ("model_info", "ExampleOrg/teacher", TEACHER) in api.calls
    assert api.writes == []


@pytest.mark.parametrize(
    ("mutate", "code", "expected_call"),
    [
        (
            "student",
            "PREFLIGHT_MODEL_UNREADABLE",
            ("model_info", "ExampleOrg/student", MISSING),
        ),
        (
            "dataset",
            "PREFLIGHT_DATASET_UNREADABLE",
            ("dataset_info", "ExampleOrg/data", MISSING),
        ),
        (
            "teacher",
            "PREFLIGHT_TEACHER_UNREADABLE",
            ("model_info", "ExampleOrg/teacher", MISSING),
        ),
    ],
)
def test_preflight_missing_revision_does_not_fall_back_to_head(
    tmp_path, forbid_launch_imports, monkeypatch, mutate, code, expected_call
):
    monkeypatch.setenv("HF_TOKEN", "test")
    path = materialize(tmp_path, 32, kind="distill" if mutate == "teacher" else "sft")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutate == "dataset":
        payload["dataset"]["revision"] = MISSING
        receipt_path = Path(payload["admission"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["revision"] = MISSING
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif mutate == "student":
        payload["student"]["revision"] = MISSING
    else:
        payload["teacher"]["revision"] = MISSING
    path.write_text(json.dumps(payload), encoding="utf-8")
    api = RevisionHub()
    monkeypatch.setattr(PREFLIGHT, "hf_api", lambda: api)
    with pytest.raises(PREFLIGHT.PreflightError) as caught:
        PREFLIGHT.run_preflight(str(path))
    assert caught.value.code == code
    assert expected_call in api.calls
    assert all(revision is not None for _name, _repo, revision in _consumed_reads(api))
    assert api.writes == []


def test_preflight_rejects_a_revision_that_resolves_to_a_different_commit(
    tmp_path, forbid_launch_imports, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "test")
    path = materialize(tmp_path, 32)
    api = RevisionHub(lie={"ExampleOrg/student": HEAD})
    monkeypatch.setattr(PREFLIGHT, "hf_api", lambda: api)
    with pytest.raises(PREFLIGHT.PreflightError) as caught:
        PREFLIGHT.run_preflight(str(path))
    assert caught.value.code == "PREFLIGHT_MODEL_PIN_MISMATCH"
    assert ("model_info", "ExampleOrg/student", STUDENT) in api.calls
    assert api.writes == []


def test_conflicting_hyperparameters_are_rejected_before_import(
    tmp_path, forbid_launch_imports, monkeypatch, capsys
):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))
    path = materialize(
        tmp_path,
        32,
        extra={
            "epochs": 1,
            "lr": 1e-5,
            "seqlen": 128,
            "batch_size": 1,
            "grad_accum": 1,
            "seed": 1,
            "pack": False,
        },
    )
    conflicts = [
        ["--epochs", "3"],
        ["--lr", "0.0002"],
        ["--seqlen", "4096"],
        ["--batch-size", "2"],
        ["--grad-accum", "8"],
        ["--seed", "3407"],
        ["--pack"],
    ]
    for flags in conflicts:
        started.clear()
        with pytest.raises(SystemExit) as caught:
            TRAIN.main(["--config", str(path), *flags])
        assert caught.value.code == 1
        assert "LAUNCH_CLI_OVERRIDE" in capsys.readouterr().err
        assert started == []


def test_matching_explicit_hyperparameters_are_honored(
    tmp_path, forbid_launch_imports, monkeypatch
):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))
    path = materialize(
        tmp_path,
        32,
        extra={
            "epochs": 1,
            "lr": 1e-5,
            "seqlen": 128,
            "batch_size": 1,
            "grad_accum": 1,
            "seed": 1,
            "pack": True,
        },
    )
    TRAIN.main([
        "--config", str(path),
        "--epochs", "1",
        "--lr", "1e-5",
        "--seqlen", "128",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--seed", "1",
        "--pack",
        "--model", "ExampleOrg/student",
        "--datasets", "ExampleOrg/data",
        "--method", "qlora",
    ])
    assert len(started) == 1
    used = TRAIN._hyperparameters(started[0][0]["config"])
    assert used["epochs"] == 1.0
    assert used["lr"] == 1e-5
    assert used["seqlen"] == 128
    assert used["batch_size"] == 1
    assert used["grad_accum"] == 1
    assert used["seed"] == 1
    assert used["pack"] is True

    started.clear()
    defaults = materialize(tmp_path / "defaults", 32)
    TRAIN.main([
        "--config", str(defaults),
        "--epochs", "3",
        "--lr", "0.0002",
        "--seqlen", "4096",
        "--batch-size", "2",
        "--grad-accum", "8",
        "--seed", "3407",
    ])
    assert len(started) == 1
    used = TRAIN._hyperparameters(started[0][0]["config"])
    assert used["epochs"] == 3.0
    assert used["lr"] == 2e-4
    assert used["seqlen"] == 4096
    assert used["batch_size"] == 2
    assert used["grad_accum"] == 8
    assert used["seed"] == 3407
    assert used["pack"] is False


@pytest.mark.parametrize(
    "dataset_id",
    [
        "SZLHOLDINGS/SZL-QUANT-SFT-V1",
        "szlholdings/szl-quant-sft-v1",
        "SzlHoldings/Szl-Quant-Sft-V1",
        "szlholdings/KILLINCHU-OSINT-CORPUS",
        "SZLHOLDINGS/Killinchu-Osint-Corpus",
        "szlholdings/szlholdings",
        "SZLHOLDINGS/szlholdings",
        "szlholdings/WHY-WE-LEAD",
        "SZLHOLDINGS/Why-We-Lead",
    ],
)
def test_denied_datasets_are_casefolded_before_launch(
    tmp_path, dataset_id, forbid_launch_imports, monkeypatch
):
    started = []
    monkeypatch.setattr(TRAIN, "_train_after_gates", lambda *args: started.append(args))
    path = materialize(tmp_path, 32, dataset_id=dataset_id)
    decision = CONTRACT.evaluate_path(path)
    assert decision["ok"] is False
    assert "LAUNCH_DATASET_DENIED" in decision["codes"]
    assert decision["hub_write"] is False
    assert decision["model_loaded"] is False
    with pytest.raises(SystemExit):
        TRAIN.main(["--config", str(path)])
    assert started == []


def test_unrelated_repository_case_is_not_denied(tmp_path, forbid_launch_imports):
    path = materialize(tmp_path, 32, dataset_id="ExampleOrg/Data")
    decision = CONTRACT.evaluate_path(path)
    assert "LAUNCH_DATASET_DENIED" not in decision["codes"]
    assert decision["ok"] is True


def test_environment_preflight_uses_the_same_contract(tmp_path, forbid_launch_imports):
    small = materialize(tmp_path / "small", 4)
    report = forge_preflight.run_preflight(
        dataset=str(tmp_path / "unused.jsonl"),
        require_cuda=False,
        min_disk_gib=0.0,
        min_examples=1,
        launch_config=str(small),
    )
    assert report["status"] == "NOT_READY"
    assert any("LAUNCH_CORPUS_FLOOR" in reason for reason in report["blocking_reasons"])
    assert report["launch_contract"]["launch_executed"] is False

    denied = materialize(tmp_path / "denied", 32, dataset_id="SZLHOLDINGS/why-we-lead")
    report = forge_preflight.run_preflight(
        dataset=str(ROOT / "szl_dataset.jsonl"),
        require_cuda=False,
        min_disk_gib=0.0,
        min_examples=1,
        launch_config=str(denied),
    )
    assert any("LAUNCH_DATASET_DENIED" in reason for reason in report["blocking_reasons"])
    assert report["dataset_report"]["records"] == 32
    assert report["launch_contract"]["ok"] is False

