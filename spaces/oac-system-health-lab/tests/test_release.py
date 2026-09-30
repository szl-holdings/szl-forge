"""Hermetic release-contract regressions: never contact GitHub or Hugging Face."""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
import verify_release as release  # noqa: E402


class ReleaseContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copyfile(SPACE_ROOT / "release.json", self.root / "release.json")
        shutil.copytree(SPACE_ROOT / "artifacts", self.root / "artifacts")
        self.manifest = release.load_manifest(self.root)
        self.artifacts = {
            name: (self.root / "artifacts" / name).read_bytes()
            for name in release.ARTIFACT_NAMES
        }

    def tearDown(self):
        self.temporary.cleanup()

    def fake_git(self, _root, *arguments):
        if arguments == ("config", "--get", "remote.origin.url"):
            return b"https://github.com/szl-holdings/szl-forge.git\n"
        revision = self.manifest["artifact_source"]["revision"]
        if arguments == ("rev-parse", "--verify", f"{revision}^{{commit}}"):
            return revision.encode() + b"\n"
        command = arguments[0]
        prefix = revision + ":" + release.SOURCE_DIRECTORY + "/"
        self.assertTrue(arguments[-1].startswith(prefix))
        name = arguments[-1][len(prefix) :]
        self.assertIn(name, release.ARTIFACT_NAMES)
        if command == "cat-file":
            return str(len(self.artifacts[name])).encode() + b"\n"
        if command == "show":
            return self.artifacts[name]
        self.fail(f"unexpected Git invocation: {arguments!r}")

    def test_local_startup_hashes_do_not_claim_source_or_provider_parity(self):
        with patch.object(
            release, "_git", side_effect=AssertionError("startup invoked Git")
        ):
            evidence = release.verify_local_artifacts(self.root)
        self.assertEqual(evidence["scope"], "LOCAL_HASHES_ONLY")
        self.assertEqual(len(evidence["artifacts"]), 3)
        self.assertTrue(evidence["complete"])
        self.assertNotIn("source_parity_count", evidence)

    def test_release_matches_literal_immutable_git_objects(self):
        with patch.object(release, "_git", side_effect=self.fake_git) as git:
            evidence = release.verify_release(self.root, Path("unused-test-repository"))
        self.assertEqual(evidence["source_parity_count"], 3)
        self.assertTrue(evidence["complete"])
        self.assertFalse(evidence["provider_parity_claimed"])
        self.assertFalse(evidence["receipt_minted"])
        self.assertTrue(
            all(
                value["immutable_git_blob_matched"]
                for value in evidence["artifacts"].values()
            )
        )
        self.assertEqual(git.call_count, 8)

    def test_manifest_closed_at_every_level(self):
        for field in (None, "artifact_source", "hub_model", "hub_dataset", "artifacts"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.manifest)
                target = candidate if field is None else candidate[field]
                target["unexpected"] = "not accepted"
                with self.assertRaises(release.ReleaseVerificationError):
                    release.validate_manifest(candidate)

    def test_missing_fields_rejected(self):
        for field in self.manifest:
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.manifest)
                del candidate[field]
                with self.assertRaises(release.ReleaseVerificationError):
                    release.validate_manifest(candidate)

    def test_source_and_hub_targets_cannot_change(self):
        changes = (
            ("artifact_source", "repository", "other/source"),
            ("artifact_source", "directory", "../model"),
            ("hub_model", "repo_id", "other/model"),
            ("hub_dataset", "repo_id", "other/dataset"),
        )
        for field, key, value in changes:
            with self.subTest(field=field, key=key):
                candidate = copy.deepcopy(self.manifest)
                candidate[field][key] = value
                with self.assertRaises(release.ReleaseVerificationError):
                    release.validate_manifest(candidate)

    def test_revision_rejects_refs_uppercase_and_non_exact_lengths(self):
        for value in (
            "main",
            "HEAD",
            "a" * 39,
            "a" * 41,
            "A" * 40,
            42,
            "a" * 40 + "\n",
        ):
            for field in ("artifact_source", "hub_model", "hub_dataset"):
                with self.subTest(value=value, field=field):
                    candidate = copy.deepcopy(self.manifest)
                    candidate[field]["revision"] = value
                    with self.assertRaises(release.ReleaseVerificationError):
                        release.validate_manifest(candidate)

    def test_artifact_digests_are_exact(self):
        for value in ("a" * 63, "a" * 65, "A" * 64, "g" * 64, True):
            with self.subTest(value=value):
                candidate = copy.deepcopy(self.manifest)
                candidate["artifacts"]["model.json"] = value
                with self.assertRaises(release.ReleaseVerificationError):
                    release.validate_manifest(candidate)

    def test_strict_json_rejects_duplicate_nonfinite_depth_and_oversized_input(self):
        for raw in (
            b'{"schema":1,"schema":2}',
            b'{"value":NaN}',
            b'{"value":Infinity}',
            b'{"value":1e999}',
            b'{"value":' + b"9" * 129 + b"}",
            b"[" * 34 + b"0" + b"]" * 34,
            b" " * (release.MAX_MANIFEST_BYTES + 1),
            b"\xff",
            b"{not-json",
        ):
            with self.subTest(raw=raw[:40]):
                with self.assertRaises(release.ReleaseVerificationError):
                    release.strict_json(raw)

    def test_local_tamper_fails_even_when_filename_is_correct(self):
        (self.root / "artifacts" / "model.json").write_bytes(b"{}")
        with self.assertRaisesRegex(
            release.ReleaseVerificationError, "local artifact hash mismatch"
        ):
            release.verify_local_artifacts(self.root)

    def test_hash_correct_local_bytes_are_insufficient_for_source_proof(self):
        def altered_git(root, *arguments):
            value = self.fake_git(root, *arguments)
            return b"x" * len(value) if arguments[0] == "show" else value

        with patch.object(release, "_git", side_effect=altered_git):
            with self.assertRaisesRegex(
                release.ReleaseVerificationError, "immutable Git artifact mismatch"
            ):
                release.verify_release(self.root, Path("unused"))

    def test_wrong_origin_or_missing_commit_fails(self):
        for first in (
            b"https://github.com/unrelated/szl-forge.git\n",
            b"file:///model\n",
        ):
            with self.subTest(origin=first):
                with patch.object(release, "_git", return_value=first):
                    with self.assertRaises(release.ReleaseVerificationError):
                        release.verify_release(self.root, Path("unused"))
        with patch.object(
            release,
            "_git",
            side_effect=[b"https://github.com/szl-holdings/szl-forge.git", b"b" * 40],
        ):
            with self.assertRaisesRegex(
                release.ReleaseVerificationError, "revision mismatch"
            ):
                release.verify_release(self.root, Path("unused"))

    def test_oversized_git_artifact_rejected_before_show(self):
        with patch.object(
            release,
            "_git",
            side_effect=[
                b"https://github.com/szl-holdings/szl-forge.git",
                self.manifest["artifact_source"]["revision"].encode(),
                str(release.MAX_ARTIFACT_BYTES + 1).encode(),
            ],
        ) as git:
            with self.assertRaisesRegex(release.ReleaseVerificationError, "byte limit"):
                release.verify_release(self.root, Path("unused"))
        self.assertEqual(git.call_count, 3)

    def test_symlink_artifact_rejected_when_supported(self):
        target = self.root / "artifacts" / "model.json"
        target.unlink()
        try:
            target.symlink_to(SPACE_ROOT / "artifacts" / "model.json")
        except OSError:
            self.skipTest("symlink creation unavailable to current account")
        with self.assertRaisesRegex(
            release.ReleaseVerificationError, "links or reparse"
        ):
            release.verify_local_artifacts(self.root)

    def test_report_is_exclusively_created_and_never_overwritten(self):
        destination = self.root / "receipt.json"
        release.write_report(destination, {"complete": True})
        first = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            release.write_report(destination, {"complete": False})
        self.assertEqual(destination.read_bytes(), first)

    def test_cli_retains_failure_receipt(self):
        (self.root / "release.json").write_text("{}", encoding="utf-8")
        destination = self.root / "failure.json"
        result = release.main(
            ["--space-root", str(self.root), "--output", str(destination)]
        )
        self.assertEqual(result, 1)
        report = json.loads(destination.read_text(encoding="utf-8"))
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["receipt_minted"])


if __name__ == "__main__":
    unittest.main()
