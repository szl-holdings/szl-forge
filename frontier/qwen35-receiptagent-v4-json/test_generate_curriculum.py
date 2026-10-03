# SPDX-License-Identifier: Apache-2.0
"""Train/dev author tests. No final-gate cases or templates are read here."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_curriculum as generator
import json_contract as contract


class FreshCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_open = Path.open

        def guard_open(path, *args, **kwargs):
            if "heldout" in path.parts or path.name in {"train.jsonl", "dev.jsonl", "test.jsonl"}:
                raise AssertionError("pure author must not read a curriculum or held-out content")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guard_open):
            cls.outputs = generator.build_train_dev()
        cls.rows = {name: [json.loads(line) for line in raw.splitlines()]
                    for name, raw in cls.outputs.items()}

    def test_committed_train_dev_exact_reproducibility(self):
        for name, raw in self.outputs.items():
            self.assertEqual((generator.HERE / name).read_bytes(), raw)
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\r", raw)

    def test_train_dev_balanced_new_families(self):
        for split, families, per_class in (("train", generator.TRAIN_FAMILIES, 30),
                                           ("dev", generator.DEV_FAMILIES, 15)):
            rows = self.rows[f"{split}.jsonl"]
            self.assertEqual(Counter((row["family"], row["kind"]) for row in rows),
                             Counter({(family, kind): per_class for family in families for kind in generator.KINDS}))
            self.assertTrue(all(row["split"] == split for row in rows))
            self.assertTrue(set(families).isdisjoint(generator.HELDOUT_FAMILIES))
        ids = [row["id"] for rows in self.rows.values() for row in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_manifest_reproducibility_uses_only_aggregate_commitment(self):
        original_open = Path.open

        def guard_open(path, *args, **kwargs):
            if "heldout" in path.parts and path.name != "manifest.json":
                raise AssertionError("manifest construction cannot read final cases or generator")
            if path.name in {"train.jsonl", "dev.jsonl", "test.jsonl"}:
                raise AssertionError("manifest construction must use in-memory train/dev bytes")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guard_open):
            manifest = generator.build_manifest(self.outputs, generator.load_heldout_commitment())
        expected = (contract.canonical_json(manifest) + "\n").encode("utf-8")
        self.assertEqual(expected, (generator.HERE / "curriculum-manifest.json").read_bytes())
        for name in ("training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(manifest[name], False)

    def test_no_final_split_or_family_authoring(self):
        for split, family in (("test", generator.TRAIN_FAMILIES[0]),
                              ("train", generator.HELDOUT_FAMILIES[0]),
                              ("dev", generator.TRAIN_FAMILIES[0])):
            with self.assertRaises(contract.ContractError):
                generator.make_row(split, family, "DRAFT", 0)

    def test_invalid_kind_index_and_source_rejected(self):
        for kind, index in (("OTHER", 0), ("DRAFT", True), ("DRAFT", -1), ("DRAFT", 30)):
            with self.assertRaises(contract.ContractError):
                generator.make_row("train", generator.TRAIN_FAMILIES[0], kind, index)
        with patch.object(generator, "CONTRACT_SHA256", "0" * 64):
            with self.assertRaises(contract.ContractError):
                generator.build_train_dev()
        with patch.object(generator, "HELDOUT_MANIFEST_SHA256", "0" * 64):
            with self.assertRaises(contract.ContractError):
                generator.load_heldout_commitment()

    def test_copy_binding_canary_and_no_execution(self):
        for kind in generator.KINDS:
            row = generator.make_row("train", generator.TRAIN_FAMILIES[0], kind, 7)
            envelope = json.loads(row["messages"][1]["content"])
            response = json.loads(row["messages"][2]["content"])
            result = contract.validate_pair(contract.canonical_json(envelope["request"]),
                                            contract.canonical_json(response))
            self.assertTrue(result["conforms"])
            self.assertIs(result["execution_authority"], False)
            self.assertEqual(response["requestSha256"], envelope["requestSha256"])
            if kind == "REFUSAL":
                self.assertNotIn(envelope["request"]["forbiddenTerm"], row["messages"][2]["content"])


class MaterializationTests(unittest.TestCase):
    def test_exclusive_write_check_and_never_overwrite(self):
        outputs = {"train.jsonl": b"synthetic reference bytes\n", "dev.jsonl": b"different reference bytes\n"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(contract.ContractError):
                generator.materialize(directory, outputs, check=True)
            generator.materialize(directory, outputs, check=False)
            generator.materialize(directory, outputs, check=True)
            before = {name: (directory / name).stat().st_mtime_ns for name in outputs}
            generator.materialize(directory, outputs, check=False)
            self.assertEqual(before, {name: (directory / name).stat().st_mtime_ns for name in outputs})
            with self.assertRaises(contract.ContractError):
                generator.materialize(directory, {"train.jsonl": b"different"}, check=False)
            self.assertEqual((directory / "train.jsonl").read_bytes(), outputs["train.jsonl"])

    def test_concurrent_creation_is_caught_by_final_readback(self):
        outputs = {"train.jsonl": b"intended train\n", "dev.jsonl": b"intended dev\n"}
        original_open = Path.open
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def racing_open(path, mode="r", *args, **kwargs):
                if path.name == "train.jsonl" and mode == "xb":
                    (directory / "dev.jsonl").write_bytes(b"SIMULATED racing writer\n")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", racing_open):
                with self.assertRaises(contract.ContractError):
                    generator.materialize(directory, outputs, check=False)

    def test_oversized_and_directory_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "train.jsonl"
            path.write_bytes(b"too many bytes")
            with self.assertRaises(contract.ContractError):
                generator._read_small(path, 2)
            path.unlink()
            path.mkdir()
            with self.assertRaises(contract.ContractError):
                generator.materialize(directory, {"train.jsonl": b"data"}, check=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
