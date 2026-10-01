"""Offline fixtures for preservation, pinning, and non-promoting metadata."""
import copy
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("curated_preparer", ROOT / "tools/prepare_curated_model_card.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CuratedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = json.loads((ROOT / MODULE.REGISTRY).read_bytes())
        self.profile = self.registry["profiles"]["willay"]
        self.observation = json.loads((ROOT / self.profile["observation_path"]).read_bytes())
        self.source = (ROOT / self.profile["source_path"]).read_bytes()
        self.save(self.profile["source_path"], self.source)
        self.save_inputs()

    def save(self, path, raw):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(raw)

    def save_inputs(self):
        raw = (json.dumps(self.observation, indent=2) + "\n").encode()
        self.profile["observation_sha256"] = MODULE.digest(raw)
        self.save(self.profile["observation_path"], raw)
        self.save(MODULE.REGISTRY, (json.dumps(self.registry, indent=2) + "\n").encode())

    def test_real_reviewed_snapshot_and_preservation(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network prohibited")), \
             patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS prohibited")), \
             patch.object(os, "getenv", side_effect=AssertionError("auth environment prohibited")):
            candidate, diff, receipt = MODULE.prepare(ROOT)
        self.assertEqual(MODULE.unmanaged(candidate), self.source)
        self.assertEqual(MODULE.frontmatter(candidate), MODULE.frontmatter(self.source))
        self.assertIn("+## Pinned packaging observation", diff)
        self.assertFalse(receipt["publication"])
        self.assertFalse(receipt["promotion"])
        self.assertFalse(receipt["weight_bodies_read"])
        self.assertEqual(receipt["network_calls"], 0)
        self.assertEqual(receipt["auth_reads"], 0)
        self.assertEqual(receipt["lineage"], "UNKNOWN")
        self.assertEqual(receipt["candidate_sha256"], MODULE.digest(candidate))

    def test_idempotent_preparation(self):
        candidate, _, _ = MODULE.prepare(self.root)
        self.save(self.profile["source_path"], candidate)
        repeated, diff, receipt = MODULE.prepare(self.root)
        self.assertEqual(repeated, candidate)
        self.assertEqual(diff, "")
        self.assertTrue(receipt["yaml_bytes_preserved"])

    def test_crlf_yaml_prose_and_unknown_license_are_not_rewritten(self):
        raw = b"---\r\nlicense: other\r\nextra: [a, b]\r\n---\r\nProse WITHOUT metrics.\r\n"
        self.profile["source_sha256"] = MODULE.digest(raw)
        self.save(self.profile["source_path"], raw)
        self.save_inputs()
        candidate, _, _ = MODULE.prepare(self.root)
        self.assertEqual(MODULE.unmanaged(candidate), raw)
        self.assertEqual(MODULE.frontmatter(candidate), MODULE.frontmatter(raw))

    def test_new_source_requires_a_fresh_review(self):
        self.save(self.profile["source_path"], self.source + b"New unrelated content\n")
        with self.assertRaises(MODULE.PreparationError):
            MODULE.prepare(self.root)

    def test_observation_hash_is_checked(self):
        self.save(self.profile["observation_path"], b"{}\n")
        with self.assertRaises(MODULE.PreparationError):
            MODULE.prepare(self.root)

    def test_metadata_cannot_grant_model_or_runtime_validity(self):
        for field in ("artifact_validity", "runtime_validity"):
            with self.subTest(field=field):
                self.observation["classification"][field] = "PASS"
                self.save_inputs()
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.prepare(self.root)
                self.observation["classification"][field] = "NOT_EVALUATED"

    def test_unknown_inventory_is_explicit_and_does_not_create_absence(self):
        self.observation["inventory_complete"] = False
        self.observation["classification"]["inventory_complete"] = False
        self.observation["classification"]["status"] = "UNKNOWN_STRUCTURE"
        self.save_inputs()
        candidate, _, receipt = MODULE.prepare(self.root)
        self.assertEqual(receipt["metadata_structure"], "UNKNOWN_STRUCTURE")
        self.assertIn(b"`UNKNOWN_STRUCTURE`", candidate)
        self.assertEqual(receipt["lineage"], "UNKNOWN")

    def test_missing_and_mutable_revisions_fail_closed(self):
        for field, value in (("observed_hub_revision", "main"), ("reviewed_source_revision", None)):
            with self.subTest(field=field):
                original = self.profile[field]
                self.profile[field] = value
                self.save_inputs()
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.prepare(self.root)
                self.profile[field] = original

    def test_type_and_every_file_revision_are_checked(self):
        for mutation in ("type", "file", "classification"):
            with self.subTest(mutation=mutation):
                original = copy.deepcopy(self.observation)
                if mutation == "type":
                    self.observation["repo_type"] = "dataset"
                elif mutation == "file":
                    self.observation["files"][0]["revision"] = "1" * 40
                else:
                    self.observation["classification"]["revision"] = "1" * 40
                self.save_inputs()
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.prepare(self.root)
                self.observation = original

    def test_incomplete_shard_config_and_index_only_fail_closed(self):
        original = copy.deepcopy(self.observation)
        for mutation in ("config", "shard", "index_only", "size_unknown", "size_empty", "truncated"):
            with self.subTest(mutation=mutation):
                self.observation = copy.deepcopy(original)
                package = self.observation["classification"]["packages"][0]
                if mutation == "config":
                    package["config"] = "missing-config.json"
                elif mutation == "shard":
                    package["referenced_shards"] = ["missing-shard.safetensors"]
                elif mutation == "index_only":
                    package["indexes"] = ["adapter_model.safetensors.index.json"]
                    package["payloads"] = []
                elif mutation in {"size_unknown", "size_empty"}:
                    row = next(row for row in self.observation["files"] if row["path"] == package["payloads"][0])
                    row["size"] = None if mutation == "size_unknown" else 0
                else:
                    self.observation["inventory_complete"] = False
                self.save_inputs()
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.prepare(self.root)

    def test_each_adapter_keeps_its_base_and_unknown_revision(self):
        candidate, _, _ = MODULE.prepare(self.root)
        text = candidate[len(self.source):].decode()
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", text)
        self.assertIn("unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit", text)
        self.assertEqual(text.count("| `UNKNOWN` |"), 2)
        self.assertNotIn("100%", text)
        self.assertNotIn("MEASURED", text)
        self.observation["classification"]["adapter_groups"][1]["lineage_status"] = "VERIFIED"
        self.save_inputs()
        with self.assertRaises(MODULE.PreparationError):
            MODULE.prepare(self.root)

    def test_payload_size_does_not_classify_merged_model(self):
        candidate, _, _ = MODULE.prepare(self.root)
        row = next(row for row in self.observation["files"] if row["path"] == "model.safetensors")
        row["size"] = 1
        self.save_inputs()
        changed, _, _ = MODULE.prepare(self.root)
        # Only the observation hash changes; no size threshold or model promotion.
        self.assertIn(b"not an independently verified full or merged model", candidate)
        self.assertIn(b"not an independently verified full or merged model", changed)

    def test_managed_block_errors_never_truncate_prose(self):
        for extra in (MODULE.START, MODULE.END, MODULE.START + b"\n" + MODULE.START + MODULE.END,
                      MODULE.END + b"\n" + MODULE.START):
            with self.subTest(extra=extra):
                self.save(self.profile["source_path"], self.source + extra + b"\n")
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.prepare(self.root)
        yaml_marker = self.source.replace(b"---\n", b"---\n" + MODULE.START + b"\n", 1)
        with self.assertRaises(MODULE.PreparationError):
            MODULE.unmanaged(yaml_marker)

    def test_duplicate_keys_nonfinite_and_invalid_json_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'\xff', b'{'):
            with self.subTest(raw=raw), self.assertRaises(MODULE.PreparationError):
                MODULE.load_json(raw)

    def test_unknown_profile_and_path_escape_are_rejected(self):
        with self.assertRaises(MODULE.PreparationError):
            MODULE.prepare(self.root, "brain-navigator-r2")
        for path in ("../README.md", "/README.md", "a/../README.md", "a\\README.md", "a//README.md"):
            with self.subTest(path=path), self.assertRaises(MODULE.PreparationError):
                MODULE.read_local(self.root, path)

    def test_oversized_empty_and_missing_inputs_are_rejected(self):
        path = self.root / self.profile["observation_path"]
        for raw in (b"", b" " * (MODULE.MAX_BYTES + 1)):
            path.write_bytes(raw)
            with self.assertRaises(MODULE.PreparationError):
                MODULE.prepare(self.root)
        path.unlink()
        with self.assertRaises(MODULE.PreparationError):
            MODULE.prepare(self.root)

    def test_review_outputs_are_exclusive_and_source_unchanged(self):
        candidate, diff, receipt = MODULE.prepare(self.root)
        output = self.root / "review"
        self.assertFalse(output.exists())
        MODULE.write_review(output, candidate, diff, receipt)
        self.assertEqual((output / "README.md").read_bytes(), candidate)
        self.assertEqual((self.root / self.profile["source_path"]).read_bytes(), self.source)
        with self.assertRaises(FileExistsError):
            MODULE.write_review(output, candidate, diff, receipt)


if __name__ == "__main__":
    unittest.main()
