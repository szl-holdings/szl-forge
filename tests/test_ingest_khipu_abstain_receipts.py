"""Fail-closed contract for tools/ingest_khipu_abstain_receipts.py.

Synthetic receipts mirror the exact shapes written by tools/szl_omen_pipeline.py
(dpo / eval-abstain / cards). No GPU, network, or token is involved.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ingest_khipu_abstain_receipts.py"
spec = importlib.util.spec_from_file_location("ingest_khipu_abstain_receipts", TOOL)
ingest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ingest
spec.loader.exec_module(ingest)

PIPE = "31d5d1bb33f81d9b4ab20f9ba9054741bec8183e45788a99f0e77fe50f58d1ca"
ADV = "a" * 64
EVL = "b" * 64
ADAPTER = {"adapter_model.safetensors": "c" * 64}
POLICY = {"base": "Qwen/Qwen3.5-1.5B", "adapter": "SZLHOLDINGS/khipu-r3", "adapter_revision": "90af1d3c2d3e79e60d8e1c82fa99a723e6888b90", "init": "base+adapter merged"}
COUNTS = {"train": 23, "adversarial.jsonl": 6, "eval.jsonl": 5}
ENV = {"python": "3.12.0", "platform": "Windows-11", "pipeline_sha256": PIPE, "schema": ingest.SCHEMA, "generated_at": "2026-10-01T02:00:00Z"}


def _summary(total: int, correct: int, invented: int = 0) -> dict:
    return {"total": total, "correct": correct, "parse_failures": 0, "invented_handle_rows": invented, "gold_citation_matches": correct}


def _dpo() -> dict:
    return {
        "stage": "dpo", "status": "TRAINED_CHALLENGER", "artifact": ingest.CHALLENGER_ID, "lane": "khipu-abstain-dpo", "seed": 11, "epochs": 2,
        "policy": dict(POLICY),
        "data": {"repo": "model:SZLHOLDINGS/SZL-Khipu-1.5B-abstain", "revision": "942c02dd4e94f2cb7cfb0432a051c2db2219ee79",
                 "digests_sha256": {"train.abstain.jsonl": "d" * 64, "train.jsonl": "e" * 64, "adversarial.jsonl": ADV, "eval.jsonl": EVL},
                 "train_files": ["train.abstain.jsonl", "train.jsonl"], "heldout_files_sealed": ["adversarial.jsonl", "eval.jsonl"],
                 "counts": dict(COUNTS), "disjoint_prompts_verified": True},
        "pairs_used": 23, "pairs_sha256": "f" * 64, "train_loss": 0.41, "train_loss_label": "TRAIN METRIC, NOT AN EVAL",
        "hardware": {"gpu": "NVIDIA GeForce RTX 5050 Laptop GPU", "vram_bytes": 8 * 2**30, "cuda": "12.8", "device_count": 1},
        "adapter_sha256": dict(ADAPTER), "evaluation": "NOT_RUN", "publication_eligible": False, "autonomy_eligible": False,
        "promotion": "NOT_PROMOTABLE", **ENV,
    }


def _transcripts() -> str:
    rows = []
    for f, n in (("adversarial.jsonl", 6), ("eval.jsonl", 5)):
        for i in range(n):
            rows.append({"file": f, "index": i, "baseline": {"output": "{}", "correct": False}, "challenger": {"output": "{}", "correct": True}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def _eval(transcripts_sha: str) -> dict:
    return {
        "stage": "eval-abstain", "status": "MEASURED", "artifact": ingest.CHALLENGER_ID, "challenger_dir": "C:/szl/challenger",
        "challenger_adapter_sha256": dict(ADAPTER), "policy": dict(POLICY),
        "data": {"repo": "model:SZLHOLDINGS/SZL-Khipu-1.5B-abstain", "revision": "942c02dd4e94f2cb7cfb0432a051c2db2219ee79",
                 "heldout_digests_sha256": {"adversarial.jsonl": ADV, "eval.jsonl": EVL}, "counts": dict(COUNTS), "disjoint_prompts_verified": True},
        "decode": {"greedy": True, "max_new_tokens": 768, "seed": 11}, "grader": "decision + citedNodeIds subset-of-offered",
        "results": {"adversarial.jsonl": {"baseline": _summary(6, 2), "challenger": _summary(6, 5)},
                    "eval.jsonl": {"baseline": _summary(5, 4, 1), "challenger": _summary(5, 4, 0)}},
        "transcripts_sha256": transcripts_sha, "advisory_challenger_improves": True,
        "advisory_note": "Advisory only.", "hardware": {"gpu": "NVIDIA GeForce RTX 5050 Laptop GPU", "cuda": "12.8"},
        "publication_eligible": False, "autonomy_eligible": False, "promotion": "NOT_PROMOTABLE", **ENV,
    }


def _cards() -> dict:
    return {"stage": "cards", "status": "MEASURED", "apply": False, "mode": "dry-run", "publication_eligible": False, **ENV}


@pytest.fixture
def laptop(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    chal = tmp_path / "challenger"
    (chal / "eval").mkdir(parents=True)
    (chal / "dpo_receipt.json").write_text(json.dumps(_dpo(), indent=2), encoding="utf-8")
    transcripts = _transcripts()
    (chal / "eval" / "eval_transcripts.jsonl").write_text(transcripts, encoding="utf-8")
    (chal / "eval" / "eval_receipt.json").write_text(json.dumps(_eval(hashlib.sha256(transcripts.encode()).hexdigest()), indent=2), encoding="utf-8")
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "cards_receipt.json").write_text(json.dumps(_cards(), indent=2), encoding="utf-8")
    (cards / "khipu-r3.diff").write_text("--- a\n+++ b\n", encoding="utf-8")
    return chal, cards


def _rewrite(path: pathlib.Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_stage_then_validate_roundtrip(laptop, tmp_path, capsys):
    chal, cards = laptop
    out = tmp_path / "repo"
    assert ingest.main(["stage", str(chal), "--cards", str(cards), "--out", str(out)]) == 0
    bundles = list((out / "receipts" / "khipu-abstain").iterdir())
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.name == "20261001-cccccccccccc"
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["publication_eligible"] is False and manifest["promotion"] == "NOT_PROMOTABLE"
    assert set(manifest["files"]) == {"dpo_receipt.json", "eval/eval_receipt.json", "eval/eval_transcripts.jsonl", "cards/cards_receipt.json", "cards/khipu-r3.diff", "STATUS.md"}
    status = (bundle / "STATUS.md").read_text(encoding="utf-8")
    assert "Training: TRAINED_CHALLENGER" in status
    assert "Evaluation: MEASURED" in status
    assert "Promotion: NOT_PROMOTABLE" in status
    assert "challenger 5/6 vs baseline 2/6" in status
    assert "owner-authorized cards" in status  # known pipeline digest is named
    assert ingest.main(["validate", str(bundle)]) == 0
    out_text = capsys.readouterr().out
    assert "VALID bundle" in out_text


def test_stage_refuses_second_bundle_with_same_identity(laptop, tmp_path):
    chal, _cards = laptop
    out = tmp_path / "repo"
    assert ingest.main(["stage", str(chal), "--out", str(out)]) == 0
    assert ingest.main(["stage", str(chal), "--out", str(out)]) == 2


def test_eval_of_different_bytes_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    _rewrite(chal / "eval" / "eval_receipt.json", lambda d: d.__setitem__("challenger_adapter_sha256", {"adapter_model.safetensors": "9" * 64}))
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "evaluated bytes must be the trained bytes" in capsys.readouterr().out
    assert not (tmp_path / "repo").exists()


def test_heldout_digest_drift_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    _rewrite(chal / "eval" / "eval_receipt.json", lambda d: d["data"]["heldout_digests_sha256"].__setitem__("adversarial.jsonl", "1" * 64))
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "sealed at training time" in capsys.readouterr().out


def test_transcript_tamper_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    p = chal / "eval" / "eval_transcripts.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"correct": true', '"correct": false', 1), encoding="utf-8")
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "transcripts_sha256" in capsys.readouterr().out


def test_claimed_publication_eligibility_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    _rewrite(chal / "eval" / "eval_receipt.json", lambda d: d.__setitem__("publication_eligible", True))
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "publication_eligible must be false" in capsys.readouterr().out


def test_promotion_claim_in_receipt_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    _rewrite(chal / "dpo_receipt.json", lambda d: d.__setitem__("promotion", "PROMOTABLE"))
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "canonical Forge gate" in capsys.readouterr().out


def test_cpu_receipt_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    _rewrite(chal / "dpo_receipt.json", lambda d: d.__setitem__("hardware", {"gpu": "", "cuda": None}))
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "CPU lanes are forbidden" in capsys.readouterr().out


def test_fail_closed_dpo_is_staged_as_incomplete_evidence(tmp_path, capsys):
    chal = tmp_path / "challenger"
    chal.mkdir()
    (chal / "dpo_receipt.json").write_text(json.dumps({"stage": "dpo", "status": "FAIL-CLOSED", "reason": "CUDA_UNAVAILABLE: training/eval lanes are GPU-only", **ENV}), encoding="utf-8")
    out = tmp_path / "repo"
    assert ingest.main(["stage", str(chal), "--out", str(out)]) == 0
    bundle = next((out / "receipts" / "khipu-abstain").iterdir())
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    status = (bundle / "STATUS.md").read_text(encoding="utf-8")
    assert "Training: FAILED" in status and "Evaluation: NOT_RUN" in status and "CUDA_UNAVAILABLE" in status
    assert ingest.main(["validate", str(bundle)]) == 0


def test_edited_status_block_fails_validate(laptop, tmp_path, capsys):
    chal, _ = laptop
    out = tmp_path / "repo"
    assert ingest.main(["stage", str(chal), "--out", str(out)]) == 0
    bundle = next((out / "receipts" / "khipu-abstain").iterdir())
    status = bundle / "STATUS.md"
    status.write_text(status.read_text(encoding="utf-8").replace("Promotion: NOT_PROMOTABLE", "Promotion: PROMOTABLE"), encoding="utf-8")
    assert ingest.main(["validate", str(bundle)]) == 2
    assert "differs from MANIFEST.json" in capsys.readouterr().out


def test_missing_transcripts_is_refused(laptop, tmp_path, capsys):
    chal, _ = laptop
    (chal / "eval" / "eval_transcripts.jsonl").unlink()
    assert ingest.main(["stage", str(chal), "--out", str(tmp_path / "repo")]) == 2
    assert "eval_transcripts.jsonl missing" in capsys.readouterr().out
