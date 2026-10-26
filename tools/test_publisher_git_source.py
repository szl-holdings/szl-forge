"""SIMULATED, tiny local Git fixtures; no Hub, model, runtime or signing authority."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import publish_model_source_bindings as bindings


class PublisherGitSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="publisher-git-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.patch = mock.patch.object(bindings, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.raw = b"synthetic source\nsecond line\n"
        (self.root / "source.txt").write_bytes(self.raw)
        (self.root / "unrelated.txt").write_bytes(b"unrelated fixture\n")
        (self.root / ".gitattributes").write_bytes(b"*.txt text eol=crlf\n*.json text eol=lf\n")
        self.artifact = {
            "repo_id": "SZLHOLDINGS/synthetic-fixture",
            "artifact_class": "fine_tuned_model",
            "source_path": ".",
            "promotion_state": "NOT_PROMOTED_RESEARCH_ONLY",
            "required_hub_files": ["model.safetensors"],
            "source_files": ["source.txt"],
            "lineage": [{"repo_id": "fixture/base", "revision": "a" * 40}],
            "signed_receipts": {"claim_scope": "REPOSITORY_DECLARED_KEY_CONTINUITY_ONLY"},
            "autonomy_boundary": {"autonomous_execution": False},
            "synthetic_count": 1,
        }
        self.contract = {
            "schema": "szl.model-source-bindings/v2",
            "source_repository": "szl-holdings/szl-forge",
            "artifacts": [self.artifact],
        }
        self.contract_path = self.root / "publishing/model-source-bindings.json"
        self.contract_path.parent.mkdir()
        self.contract_path.write_text(json.dumps(self.contract), encoding="utf-8")
        self.git("-c", "init.templateDir=", "init", "-q")
        self.git("add", ".")
        self.git(
            "-c", "user.name=Synthetic fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic source fixture",
        )
        self.revision = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args], check=True, capture_output=True, timeout=10
        ).stdout

    def test_committed_source_has_exact_git_blob_hash_and_size(self):
        bindings.verify_source_contract(self.contract, self.revision)
        self.assertEqual(bindings.source_evidence(self.artifact, self.revision), [{
            "path": "source.txt", "bytes": len(self.raw),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }])

    def test_crlf_checkout_keeps_committed_lf_digest(self):
        (self.root / "source.txt").write_bytes(self.raw.replace(b"\n", b"\r\n"))
        evidence = bindings.source_evidence(self.artifact, self.revision)[0]
        self.assertEqual(evidence["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(evidence["bytes"], len(self.raw))

    def test_unrelated_dirty_and_untracked_files_are_allowed(self):
        (self.root / "unrelated.txt").write_bytes(b"changed unrelated fixture")
        (self.root / "untracked.txt").write_bytes(b"untracked fixture")
        bindings.verify_source_contract(self.contract, self.revision)
        self.assertEqual(len(bindings.source_evidence(self.artifact, self.revision)), 1)

    def test_wrong_or_invalid_revision_fails_before_provider_access(self):
        for revision in ("b" * 40, "HEAD", "", "a" * 39, "A" * 40):
            with self.subTest(revision=revision), mock.patch.object(bindings, "hub_evidence") as hub:
                with self.assertRaises(bindings.BindingError):
                    bindings.prepare_one(mock.Mock(), self.contract, self.artifact,
                                         source_revision=revision, token=None)
                hub.assert_not_called()

    def test_publish_one_cannot_bypass_identity_guard(self):
        api = mock.Mock()
        with self.assertRaises(bindings.BindingError):
            bindings.publish_one(api, self.contract, self.artifact,
                                 source_revision="b" * 40, publish=True, token="synthetic-token")
        self.assertEqual(api.mock_calls, [])

    def test_staged_source_change_is_rejected(self):
        (self.root / "source.txt").write_bytes(b"changed\n")
        self.git("add", "source.txt")
        with self.assertRaisesRegex(bindings.BindingError, "working copy is modified"):
            bindings.source_evidence(self.artifact, self.revision)

    def test_unstaged_source_change_is_rejected(self):
        (self.root / "source.txt").write_bytes(b"changed\n")
        with self.assertRaisesRegex(bindings.BindingError, "working copy is modified"):
            bindings.source_evidence(self.artifact, self.revision)

    def test_modified_contract_is_rejected_before_provider_or_report(self):
        self.contract_path.write_text(json.dumps({**self.contract, "extra": True}), encoding="utf-8")
        api = mock.Mock()
        report = self.root / "report.json"
        with self.assertRaisesRegex(bindings.BindingError, "working copy is modified"):
            bindings.run(contract_path=self.contract_path, report_path=report,
                         source_revision=self.revision, publish=False, token=None, api=api)
        self.assertEqual(api.mock_calls, [])
        self.assertFalse(report.exists())

    def test_staged_contract_is_rejected(self):
        self.contract_path.write_text(json.dumps({**self.contract, "extra": True}), encoding="utf-8")
        self.git("add", "publishing/model-source-bindings.json")
        with self.assertRaisesRegex(bindings.BindingError, "working copy is modified"):
            bindings.verify_source_contract(self.contract, self.revision)

    def test_in_memory_contract_is_not_a_committed_contract(self):
        with self.assertRaisesRegex(bindings.BindingError, "differs from the committed contract"):
            bindings.verify_source_contract({**self.contract, "extra": True}, self.revision)

    def test_in_memory_artifact_substitution_is_rejected(self):
        altered = {**self.artifact, "source_files": ["unrelated.txt"]}
        with mock.patch.object(bindings, "hub_evidence") as hub:
            with self.assertRaisesRegex(bindings.BindingError, "artifact differs"):
                bindings.prepare_one(mock.Mock(), self.contract, altered,
                                     source_revision=self.revision, token=None)
            hub.assert_not_called()

    def test_bool_integer_contract_substitution_is_rejected(self):
        altered = copy.deepcopy(self.contract)
        altered["artifacts"][0]["autonomy_boundary"]["autonomous_execution"] = 0
        with self.assertRaisesRegex(bindings.BindingError, "differs from the committed contract"):
            bindings.verify_source_contract(altered, self.revision)

    def test_integer_float_contract_substitution_is_rejected(self):
        altered = copy.deepcopy(self.contract)
        altered["artifacts"][0]["synthetic_count"] = 1.0
        with self.assertRaisesRegex(bindings.BindingError, "differs from the committed contract"):
            bindings.verify_source_contract(altered, self.revision)

    def test_non_finite_contract_substitution_is_rejected(self):
        altered = copy.deepcopy(self.contract)
        altered["artifacts"][0]["synthetic_count"] = float("nan")
        with self.assertRaisesRegex(bindings.BindingError, "finite JSON"):
            bindings.verify_source_contract(altered, self.revision)

    def test_bool_integer_artifact_substitution_is_rejected(self):
        altered = copy.deepcopy(self.artifact)
        altered["autonomy_boundary"]["autonomous_execution"] = 0
        with mock.patch.object(bindings, "hub_evidence") as hub:
            with self.assertRaisesRegex(bindings.BindingError, "artifact differs"):
                bindings.prepare_one(mock.Mock(), self.contract, altered,
                                     source_revision=self.revision, token=None)
            hub.assert_not_called()

    def test_missing_later_source_blocks_all_provider_access(self):
        second = copy.deepcopy(self.artifact)
        second.update(repo_id="SZLHOLDINGS/second-fixture", source_files=["missing.txt"])
        self.contract["artifacts"].append(second)
        self.contract_path.write_text(json.dumps(self.contract), encoding="utf-8")
        self.git("add", "publishing/model-source-bindings.json")
        self.git("-c", "user.name=Synthetic fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic missing-source case")
        revision = self.git("rev-parse", "HEAD").decode().strip()
        api = mock.Mock()
        with self.assertRaisesRegex(bindings.BindingError, "source file is missing"):
            bindings.run(contract_path=self.contract_path, report_path=self.root / "report.json",
                         source_revision=revision, publish=True, token="synthetic-token", api=api)
        self.assertEqual(api.mock_calls, [])
        self.assertFalse((self.root / "report.json").exists())

    def test_untracked_or_missing_source_is_rejected(self):
        (self.root / "untracked.txt").write_bytes(b"fixture")
        for path in ("untracked.txt", "missing.txt", "publishing"):
            with self.subTest(path=path), self.assertRaises(bindings.BindingError):
                bindings.source_evidence({**self.artifact, "source_files": [path]}, self.revision)

    def test_untracked_contract_is_rejected(self):
        alternative = self.root / "alternative.json"
        alternative.write_text(json.dumps(self.contract), encoding="utf-8")
        with self.assertRaisesRegex(bindings.BindingError, "not a committed regular file"):
            bindings.verify_source_contract(self.contract, self.revision, alternative)

    def test_external_contract_is_rejected(self):
        with self.assertRaisesRegex(bindings.BindingError, "must be inside"):
            bindings.verify_source_contract(self.contract, self.revision, self.root.parent / "outside.json")

    def test_path_aliases_and_non_string_paths_are_rejected(self):
        for path in ("../source.txt", "./source.txt", "/source.txt", "C:/source.txt",
                     "publishing/../source.txt", "publishing\\source.txt", "source.txt/", "", 1):
            with self.subTest(path=path), self.assertRaises(bindings.BindingError):
                bindings.source_evidence({**self.artifact, "source_files": [path]}, self.revision)

    def test_duplicate_source_paths_are_rejected(self):
        with self.assertRaisesRegex(bindings.BindingError, "duplicate"):
            bindings.source_evidence({**self.artifact, "source_files": ["source.txt"] * 2}, self.revision)

    def test_git_symlink_mode_is_rejected_without_creating_os_link(self):
        oid = self.git("rev-parse", "HEAD:source.txt").decode().strip()
        with mock.patch.object(bindings, "_source_git", return_value=(
            f"120000 blob {oid}\tsource.txt\0".encode()
        )):
            with self.assertRaisesRegex(bindings.BindingError, "not a committed regular file"):
                bindings._committed_source_bytes("source.txt", self.revision)

    def test_skipped_worktree_cannot_replace_emitted_committed_bytes(self):
        self.git("update-index", "--skip-worktree", "source.txt")
        (self.root / "source.txt").write_bytes(b"locally hidden change\n")
        evidence = bindings.source_evidence(self.artifact, self.revision)[0]
        self.assertEqual(evidence["sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_immutable_mode_still_rejects_wrong_source_before_provider_access(self):
        altered = copy.deepcopy(self.contract)
        altered["artifacts"][0].update(binding_mode="VERIFY_IMMUTABLE_RELEASE", hub_revision="a" * 40)
        api = mock.Mock()
        with self.assertRaises(bindings.BindingError):
            bindings.publish_one(api, altered, altered["artifacts"][0],
                                 source_revision=self.revision, publish=True, token="synthetic-token")
        self.assertEqual(api.mock_calls, [])

    def test_git_unavailable_fails_closed(self):
        with mock.patch.object(bindings.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(bindings.BindingError, "Git evidence is unavailable"):
                bindings.verify_source_contract(self.contract, self.revision)


if __name__ == "__main__":
    unittest.main()
