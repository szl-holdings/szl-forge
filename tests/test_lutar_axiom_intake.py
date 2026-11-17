"""Forge source-bound Lutar curriculum intake; no network/GPU/teacher calls."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from training import lutar_axiom_intake as intake


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def fixture():
    index = deepcopy(intake.load_index())
    source = {
        "repository": index["mathematics"]["repository"],
        "commit": index["mathematics"]["commit"],
        "axioms_path": "Lutar/Axioms.lean",
        "axioms_blob": index["mathematics"]["axioms_blob"],
        "invariant_path": "Lutar/Invariant.lean",
        "invariant_blob": index["mathematics"]["invariant_blob"],
        "toolchain": "leanprover/lean4:v4.35.0",
    }
    raw = {}
    for split in ("train", "holdout"):
        records = []
        for i in range(32):
            n = i if split == "train" else i + 100
            records.append({"metadata": {
                "case_id": sha256(f"{split}/{n}".encode()).hexdigest(),
                "schema": "szl.lutar.numeric-curriculum/v1",
                "split": split, "source": source,
                "family": "A1_MONOTONE", "training_authority": "NONE",
                "review_state": "CANDIDATE_ONLY",
                "proof_status": "NOT_LEAN_VERIFIED_AS_DATASET",
            }, "messages": [
                {"role": "system", "content": "Pinned Lutar synthetic math task."},
                {"role": "user", "content": f"Compute the exact synthetic case {n}."},
                {"role": "assistant", "content": json.dumps({"value": n})},
            ]})
        raw["curriculum/" + split + ".jsonl"] = b"".join(map(_encode, records))
    manifest = {
        "schema": "szl.lutar.numeric-curriculum-manifest/v1",
        "source": source, "training_authority": "NONE",
        "review_state": "REVIEW_REQUIRED",
        "lean_proof_of_dataset": False,
        "independent_holdout": False,
        "train": {"sha256": sha256(raw["curriculum/train.jsonl"]).hexdigest(),
                  "rows": 32},
        "holdout": {"sha256": sha256(raw["curriculum/holdout.jsonl"]).hexdigest(),
                    "rows": 32},
    }
    raw["curriculum/manifest.json"] = _encode(manifest)
    for item in index["files"]:
        content = raw[item["path"]]
        item["sha256"] = sha256(content).hexdigest()
        item["bytes"] = len(content)
        if item["path"].endswith(".jsonl"):
            item["records"] = 32
    return raw, index


class LutarIntakeTests(unittest.TestCase):
    def test_source_index_is_review_only_and_immutable(self):
        index = intake.load_index()
        self.assertEqual(index["producer"]["repository"],
                         "szl-holdings/lean-rlvr-proving-ground")
        self.assertEqual(index["producer"]["commit"],
                         "b23ea5fe7e63d6b7402ce3a6f0a6aa5e1fa56b2c")
        self.assertEqual(index["training_authority"], "NONE")
        self.assertFalse(index["training_launch"])
        self.assertFalse(index["model_publish"])

    def test_valid_exact_source_stages_review_only(self):
        raw, index = fixture()
        intake.verify_payloads(raw, index)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new-curriculum"
            report = intake.stage(raw, index=index, output=target)
            self.assertEqual(report["state"], "SOURCE_BOUND_REVIEW_REQUIRED")
            self.assertEqual(report["train_rows"], 32)
            self.assertEqual(report["holdout_rows"], 32)
            self.assertEqual(report["overlaps"], 0)
            self.assertEqual(report["training_authority"], "NONE")
            for flag in ("training_started", "model_loaded", "hub_written",
                         "release_authorized"):
                self.assertIs(report[flag], False)
            self.assertEqual((target / "train.jsonl").read_bytes(),
                             raw["curriculum/train.jsonl"])
            with self.assertRaises(intake.IntakeHold):
                intake.stage(raw, index=index, output=target)
            self.assertFalse(any(Path(tmp).glob(".szl-lutar-candidate-*")))

    def test_byte_tampering_fails_before_creating_an_output(self):
        raw, index = fixture()
        raw["curriculum/train.jsonl"] += b" "
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "must-not-exist"
            with self.assertRaises(intake.IntakeHold):
                intake.stage(raw, index=index, output=target)
            self.assertFalse(target.exists())

    def test_forged_metadata_denied_even_if_attacker_rehashes_files(self):
        raw, index = fixture()
        rows = [json.loads(s) for s in raw["curriculum/train.jsonl"].splitlines()]
        rows[0]["metadata"]["training_authority"] = "ALLOW"
        raw["curriculum/train.jsonl"] = b"".join(map(_encode, rows))
        idx = next(x for x in index["files"] if x["path"] == "curriculum/train.jsonl")
        idx["bytes"] = len(raw["curriculum/train.jsonl"])
        idx["sha256"] = sha256(raw["curriculum/train.jsonl"]).hexdigest()
        manifest = json.loads(raw["curriculum/manifest.json"])
        manifest["train"]["sha256"] = idx["sha256"]
        raw["curriculum/manifest.json"] = _encode(manifest)
        idx = next(x for x in index["files"] if x["path"] == "curriculum/manifest.json")
        idx["bytes"] = len(raw["curriculum/manifest.json"])
        idx["sha256"] = sha256(raw["curriculum/manifest.json"]).hexdigest()
        with self.assertRaises(intake.IntakeHold):
            intake.verify_payloads(raw, index)

    def test_provenance_mutation_and_false_rights_are_refused(self):
        raw, index = fixture()
        for key, value in (("training_authority", "ALLOW"),
                           ("independent_holdout", True)):
            with self.subTest(key=key):
                bad = deepcopy(index)
                bad[key] = value
                with self.assertRaises(intake.IntakeHold):
                    # Validation at reader boundary is strict and standalone.
                    original = intake.INDEX
                    with tempfile.TemporaryDirectory() as tmp:
                        p = Path(tmp) / "index.json"
                        p.write_text(json.dumps(bad), encoding="utf-8")
                        try:
                            intake.INDEX = p
                            intake.load_index()
                        finally:
                            intake.INDEX = original

    def test_no_network_fallback_for_missing_immutable_git_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(intake.IntakeHold):
                intake.read_immutable_git("b23ea5fe7e63d6b7402ce3a6f0a6aa5e1fa56b2c",
                                          Path(tmp), "curriculum/train.jsonl")
        with self.assertRaises(intake.IntakeHold):
            intake.read_immutable_git("main", Path("."), "curriculum/train.jsonl")

    def test_malformed_corpus_and_duplicates_fail_closed(self):
        raw, index = fixture()
        content = raw["curriculum/train.jsonl"]
        lines = content.splitlines()
        lines[1] = lines[0]
        raw["curriculum/train.jsonl"] = b"\n".join(lines) + b"\n"
        meta = next(x for x in index["files"] if x["path"] == "curriculum/train.jsonl")
        meta["sha256"] = sha256(raw["curriculum/train.jsonl"]).hexdigest()
        meta["bytes"] = len(raw["curriculum/train.jsonl"])
        manifest = json.loads(raw["curriculum/manifest.json"])
        manifest["train"]["sha256"] = meta["sha256"]
        raw["curriculum/manifest.json"] = _encode(manifest)
        manifest_record = next(x for x in index["files"]
                               if x["path"] == "curriculum/manifest.json")
        manifest_record["sha256"] = sha256(raw["curriculum/manifest.json"]).hexdigest()
        manifest_record["bytes"] = len(raw["curriculum/manifest.json"])
        with self.assertRaises(intake.IntakeHold):
            intake.verify_payloads(raw, index)


if __name__ == "__main__":
    unittest.main()
