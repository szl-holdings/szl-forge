"""Offline contract tests for tools/estate_payload (stdlib + pytest; no network, no GPU, no gh)."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_DIR = ROOT / "tools" / "estate_payload"
KIT_DIR = PAYLOAD_DIR / "kit"


@pytest.fixture(scope="module")
def op(tmp_path_factory):
    os.environ["SZL_ROOT"] = str(tmp_path_factory.mktemp("szl_out"))
    os.environ["SZL_KIT_DIR"] = str(KIT_DIR)
    for mod in ("szl_payload_v4", "candidate_lib"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(PAYLOAD_DIR))
    return importlib.import_module("szl_payload_v4")


def test_kit_is_complete():
    for name in ("candidate_lib.py", "qualify_runtime.py", "curriculum.py", "train_candidate.py", "evaluate_candidate.py",
                 "export_gguf.py", "render_card.py", "test_candidate_contract.py", "README.md", "MODEL_CARD.tmpl.md"):
        assert (KIT_DIR / name).is_file(), name


def test_failure_classification(op):
    assert op.classify("pytest ... FAILED tests/test_x.py::test_y - AssertionError") == ["TESTS"]
    assert op.classify("Error: ruff format --check . would reformat 3 files")[0] == "FORMAT"
    assert op.classify("curl: (35) TLS handshake timeout; API rate limit exceeded") == ["INFRA_FLAKE"]
    assert "INTENTIONAL_RED" in op.classify("verify-proof: lambda-bounty gate rejected unproven proof")
    assert op.classify("nothing recognisable here") == ["UNCLASSIFIED"]
    assert "[REDACTED]" in op.redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789")


def test_green_policy_requires_checks_and_no_hold(op):
    base = {"isDraft": False, "title": "feat: x", "labels": [], "mergeStateStatus": "CLEAN",
            "statusCheckRollup": [{"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS"}]}
    assert op.green(base)[0] is True
    assert op.green({**base, "statusCheckRollup": []})[0] is False
    assert op.green({**base, "title": "[HOLD] feat: x"})[0] is False
    assert op.green({**base, "statusCheckRollup": [{"__typename": "CheckRun", "status": "IN_PROGRESS", "conclusion": None}]})[0] is False
    assert op.handle_pr("o/r", {**base, "number": 1, "mergeStateStatus": "BLOCKED",
                                 "statusCheckRollup": [{"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "FAILURE"}]})["result"] == "FAILING_CHECKS"


def test_source_map_prefers_repo_evidence(op):
    ev = {"paths": ["khipu/train_khipu.py", "khipu/eval_khipu.py", "khipu/train.jsonl", "khipu/khipu.schema.json",
                    "khipu/training_receipt.signed.json", "khipu_r2/train_khipu_r2.py", "qantu/skip_receipt.json",
                    "willay/card/README.md", "Moons-Nano/load.py", "frontier/qwen35-receiptagent-v3/candidate.json"],
          "bindings": {"artifacts": [{"repo_id": "SZLHOLDINGS/SZL-Khipu-1.5B", "source_path": "khipu"}]},
          "portfolio": {"artifacts": [{"repo_id": "SZLHOLDINGS/szl-kernels", "kind": "software_kernel"}]},
          "candidates": {"frontier/qwen35-receiptagent-v3/candidate.json": {"target_repo_id": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated",
                                                                             "predecessor": {"repo_id": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2"}}},
          "skip": ["qantu"], "identity": []}
    sm = op.source_map("SZLHOLDINGS/SZL-Khipu-1.5B", "SZL-Khipu-1.5B", "MODEL", "WEIGHTED", ev)
    assert sm["plan"] == "HAS_TRAINER" and sm["gaps"] == [] and "khipu" in sm["folders"] and "khipu_r2" not in sm["folders"]
    assert op.source_map("SZLHOLDINGS/KHIPU-R2", "KHIPU-R2", "MODEL", "ADAPTER", ev)["plan"] == "HAS_TRAINER"
    assert op.source_map("SZLHOLDINGS/qantu", "qantu", "MODEL", "CARD_ONLY", ev)["plan"] == "SKIPPED_BY_RECEIPT"
    assert op.source_map("SZLHOLDINGS/Moons-Nano", "Moons-Nano", "MODEL", "WEIGHTED", ev)["plan"] == "REFERENCE_ARTIFACT_NO_SFT"
    assert op.source_map("SZLHOLDINGS/szl-kernels", "szl-kernels", "MODEL", "WEIGHTED", ev)["plan"] == "SOFTWARE_NO_SFT"
    assert op.source_map("SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2", "szl-receiptagent-qwen35-0.8b-v2", "MODEL", "ADAPTER", ev)["plan"] == "FRONTIER_CANDIDATE_EXISTS"
    managed = dict(ev, paths=ev["paths"] + ["frontier/willay-candidate-20261001/candidate.json", "frontier/willay-candidate-20261001/train_candidate.py"],
                   candidates={**ev["candidates"], "frontier/willay-candidate-20261001/candidate.json": {
                       "generated_by": "szl-payload/4.0.0", "predecessor": {"repo_id": "SZLHOLDINGS/WILLAY"}, "target_repo_id": "SZLHOLDINGS/willay-candidate-20261001"}})
    sm = op.source_map("SZLHOLDINGS/WILLAY", "WILLAY", "MODEL", "WEIGHTED", managed)
    assert sm["plan"] == "PAYLOAD_CANDIDATE_MANAGED" and sm["trainer"] == [], "the payload's own kit is never counted as an estate trainer"
    assert op.source_map("SZLHOLDINGS/WILLAY", "WILLAY", "MODEL", "WEIGHTED", ev)["plan"] == "GENERATE_CANDIDATE"
    assert op.source_map("SZLHOLDINGS/KILLINCHU-EYE", "KILLINCHU-EYE", "MODEL", "WEIGHTED", ev, False, "npz")["plan"] == "NON_LLM_OR_UNDEFINED_ARTIFACT"
    assert op.source_map("SZLHOLDINGS/SZLHOLDINGS", "SZLHOLDINGS", "MODEL", "CARD_ONLY", ev)["plan"] == "ORG_PROFILE_NOT_A_MODEL"


def test_tokens_split_windows_paths(op):
    assert op.ntoks("szl-typesafe-triage\\corpus\\train.jsonl") == {"szl", "typesafe", "triage", "corpus", "train", "jsonl"}
    assert op.sig_tokens("szl-triage-qwen3.5-0.8b-lora-study5") == {"triage", "qwen3"}


def test_managed_candidate_refresh_preserves_owner_binding(op, tmp_path, monkeypatch):
    monkeypatch.setattr(op, "hf_pin", lambda repo: ("c" * 40, "apache-2.0"))
    monkeypatch.setattr(op, "forge_raw", lambda path: "")
    existing = {"candidate_id": "willay-candidate-20261001", "generated_by": "szl-payload/4.0.0", "generated_utc": "2026-10-01T00:00:00+00:00",
                "target_repo_id": "SZLHOLDINGS/willay-candidate-20261001", "runtime_lock": {"torch": "2.14.0"},
                "actual_training_base": {"repo_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "d" * 40, "license": "apache-2.0", "selection": "ADAPTER_CONFIG"},
                "training_data": {"binding_status": "BOUND_LOCAL", "confirmed_by_owner": True, "rights": "DOCUMENTED",
                                  "files": {"train": {"path": "C:\\data\\willay_train.jsonl", "sha256": "e" * 64, "rows": 300}}},
                "training_recipe": {"lora_r": 8}, "gates_sha256": "0" * 64}
    m = {"model": "SZLHOLDINGS/WILLAY", "status": "WEIGHTED", "sha": "f" * 40, "adapter_base": ["Qwen/Qwen2.5-0.5B-Instruct"],
         "base_model": [], "schema_files": [], "sources": [], "family": "pcm-explain/willay", "managed_candidates": ["frontier/willay-candidate-20261001"]}
    row = op.gen_candidate(m, [], {"github": []}, existing=existing, repo_dir="frontier/willay-candidate-20261001")
    cand = json.loads((Path(row["folder"]) / "candidate.json").read_text(encoding="utf-8"))
    assert row["managed"] and row["owner_bound"] and row["rows"] == 300 and row["repo_path"] == "frontier/willay-candidate-20261001"
    assert cand["training_data"]["confirmed_by_owner"] is True and cand["runtime_lock"] == {"torch": "2.14.0"}
    assert cand["training_recipe"]["lora_r"] == 8 and cand["actual_training_base"]["revision"] == "d" * 40
    assert cand["generated_utc"] == "2026-10-01T00:00:00+00:00" and cand["refreshed_utc"] != cand["generated_utc"]
    gates_text = (Path(row["folder"]) / "gates.json").read_text(encoding="utf-8")
    assert cand["gates_sha256"] == op.sha(gates_text), "a stale frozen hash is re-frozen explicitly, never left dangling"


def test_llm_eligibility_is_conservative(op):
    assert op.llm_eligible({"pipeline_tag": "text-generation", "library_name": "peft", "tags": ["lora"]}, [])[0] is True
    assert op.llm_eligible({"library_name": "kernels", "tags": ["kernel"]}, ["x.safetensors"])[0] is False
    assert op.llm_eligible({"tags": ["logistic-regression", "standard-library"]}, ["model.json"])[0] is False
    assert op.llm_eligible({"tags": []}, ["adapter_config.json"])[0] is True
    assert op.llm_eligible({"tags": []}, ["README.md"])[0] is False


def test_candidate_generation_is_fail_closed(op, tmp_path, monkeypatch):
    corpus_dir = tmp_path / "szl-demo" / "output"
    corpus_dir.mkdir(parents=True)
    rows = [{"messages": [{"role": "user", "content": f"ticket {i} about invoices {i}"}, {"role": "assistant", "content": json.dumps({"label": "billing"})}],
             "split": "train" if i < 32 else "eval", "family_key": f"f{i}"} for i in range(40)]
    (corpus_dir / "demo_distill_v0.2_deduped.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (corpus_dir / "demo_distill_v0.1.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:22]) + "\n", encoding="utf-8")
    (corpus_dir / "demo_tiny_5.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:5]) + "\n", encoding="utf-8")
    monkeypatch.setenv("SZL_CORPUS_ROOTS", str(tmp_path / "szl-demo"))
    monkeypatch.setattr(op, "hf_pin", lambda repo: ("a" * 40, "apache-2.0"))
    corpora = op.scan_corpora()
    assert len(corpora) == 2, "files under 20 rows are never corpora"
    m = {"model": "SZLHOLDINGS/demo-model-5", "status": "ADAPTER", "sha": "b" * 40, "adapter_base": ["Qwen/Qwen2.5-0.5B-Instruct"],
         "base_model": [], "schema_files": [], "sources": [], "family": "unmapped"}
    bound = op.gen_candidate(m, corpora, {"github": []})
    folder = Path(bound["folder"])
    cand = json.loads((folder / "candidate.json").read_text(encoding="utf-8"))
    assert bound["binding"] == "BOUND_LOCAL" and bound["rows"] == 40, "numeric tokens like '5' must not steer the binding"
    assert cand["training_data"]["files"]["all"]["path"].endswith("demo_distill_v0.2_deduped.jsonl")
    assert cand["training_data"]["candidates_observed"][0].endswith("demo_distill_v0.1.jsonl")
    assert cand["actual_training_base"]["revision"] == "a" * 40 and cand["promotion"] == "NOT_PROMOTABLE"
    assert cand["target_repo_id"] != cand["predecessor"]["repo_id"]
    gates_text = (folder / "gates.json").read_text(encoding="utf-8")
    assert op.sha(gates_text) == cand["gates_sha256"]
    for name in ("candidate_lib.py", "train_candidate.py", "evaluate_candidate.py", "test_candidate_contract.py", "README.md"):
        assert (folder / name).is_file(), name
    assert "__CANDIDATE_ID__" not in (folder / "README.md").read_text(encoding="utf-8")
    unbound = op.gen_candidate({**m, "model": "SZLHOLDINGS/other-model"}, [], {"github": []})
    assert unbound["binding"] == "UNBOUND" and unbound["rows"] == 0
    # the kit library in the generated folder fails closed on an unbound curriculum
    sys.path.insert(0, str(folder))
    sys.modules.pop("candidate_lib", None)
    lib = importlib.import_module("candidate_lib")
    try:
        monkeypatch.setattr(lib, "OUT", tmp_path / "out")
        with pytest.raises(SystemExit) as e:
            lib.curriculum_files(json.loads((Path(unbound["folder"]) / "candidate.json").read_text(encoding="utf-8")))
        assert e.value.code == 3
        rows_by_split, bad, excluded = lib.load_splits(cand)
        assert len(rows_by_split["train"]) == 32 and len(rows_by_split["dev"]) == 8 and not bad and excluded == 0
        leak = lib.leakage_report(rows_by_split["train"], rows_by_split["dev"], json.loads(gates_text)["leakage"])
        assert leak["verdict"] == "PASS"
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("candidate_lib", None)
