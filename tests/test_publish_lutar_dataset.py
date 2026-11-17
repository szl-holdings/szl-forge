"""Offline publisher controls; no real Hugging Face calls or token access."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from training import publish_lutar_dataset as p
from training import lutar_axiom_intake as intake


def _stage_files(bundle, *, workdir, index):
    """Test-only stage writer; the real validator is covered by dry-run tests."""
    for name, contents in bundle.items():
        (workdir / name).write_bytes(contents)


class FakeHub:
    def __init__(self, *, exists=False, private=True, sha="c" * 40):
        self.exists = exists
        self.private = private
        self.sha = sha
        self.created = 0
        self.uploaded = 0
        self.contents = {}

    def repo_exists(self, repo_id, repo_type):
        assert repo_id == p.REPO_ID and repo_type == "dataset"
        return self.exists

    def create_repo(self, **kwargs):
        assert kwargs == {
            "repo_id": p.REPO_ID, "repo_type": "dataset",
            "private": True, "exist_ok": False,
        }
        self.created += 1

    def upload_folder(self, *, folder_path, repo_id, repo_type, commit_message):
        assert self.created == 1
        self.contents = {x.name: x.read_bytes()
                         for x in Path(folder_path).iterdir() if x.is_file()}
        assert set(self.contents) == set(p.BOUND_FILES)
        self.uploaded += 1

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.sha, private=self.private)


class PublishLutarTests(unittest.TestCase):
    @staticmethod
    def index():
        obj = intake.load_index()
        return obj

    def test_source_binding_does_not_claim_training_or_publication(self):
        index = self.index()
        val = json.loads(p._source_binding(index))
        self.assertEqual(val["repo_id"], p.REPO_ID)
        self.assertEqual(val["training_authority"], "NONE")
        self.assertFalse(val["weights_published"])
        self.assertFalse(val["independent_holdout"])
        self.assertTrue(val["private_at_creation"])
        self.assertEqual(val["source_files"]["train.jsonl"]["rows"], 160)

    def test_card_discloses_nonindependence_and_math_limit(self):
        text = p._card(self.index()).decode("utf-8")
        self.assertIn("license: apache-2.0", text)
        self.assertIn("160 training", text)
        self.assertIn("NOT AN INDEPENDENT EVALUATION", text)
        self.assertIn("unconditional", text)
        self.assertIn("TRAINING APPROVAL", text)

    def test_no_receipt_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "existing.json"
            dest.write_text("previous", encoding="utf-8")
            with self.assertRaises(intake.IntakeHold):
                p.publish_private(producer_checkout=Path("."), receipt_file=dest,
                                  api=FakeHub(), downloader=lambda **k: None)
            self.assertEqual(dest.read_text(), "previous")

    def test_existing_hub_repository_is_not_overwritten(self):
        hub = FakeHub(exists=True)
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "receipt.json"
            dummy = (self.index(), {name: b"dummy" for name in p.BOUND_FILES})
            with patch.object(p, "build_bundle", return_value=dummy), \
                 patch.object(p, "validate_bundle", side_effect=_stage_files):
                with self.assertRaises(intake.IntakeHold):
                    p.publish_private(producer_checkout=Path("."),
                                      receipt_file=receipt, api=hub,
                                      downloader=lambda **k: None)
            self.assertEqual(hub.created, 0)
            self.assertEqual(hub.uploaded, 0)
            self.assertFalse(receipt.exists())

    def test_private_upload_readback_then_receipt(self):
        hub = FakeHub()
        fake_files = {name: name.encode() for name in p.BOUND_FILES}
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "private-receipt.json"
            def get_blob(**kwargs):
                self.assertEqual(kwargs["revision"], "c" * 40)
                path = Path(root) / ("remote_" + kwargs["filename"])
                path.write_bytes(hub.contents[kwargs["filename"]])
                return str(path)
            with patch.object(p, "build_bundle", return_value=(self.index(), fake_files)), \
                 patch.object(p, "validate_bundle", side_effect=_stage_files):
                result = p.publish_private(producer_checkout=Path("."),
                                           receipt_file=receipt, api=hub,
                                           downloader=get_blob)
            self.assertEqual((hub.created, hub.uploaded), (1, 1))
            self.assertEqual(result["hub_revision"], "c" * 40)
            self.assertEqual(result["training_authority"], "NONE")
            self.assertTrue(result["private"])
            self.assertFalse(result["trained"])
            self.assertEqual(json.loads(receipt.read_text())["status"],
                             "PRIVATE_BYTES_VERIFIED_REVIEW_REQUIRED")

    def test_modified_remote_content_refuses_receipt(self):
        hub = FakeHub()
        fake_files = {name: name.encode() for name in p.BOUND_FILES}
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "must-not-exist.json"
            def bad_read(**kwargs):
                path = Path(root) / ("remote_" + kwargs["filename"])
                path.write_bytes(b"tampered")
                return str(path)
            with patch.object(p, "build_bundle", return_value=(self.index(), fake_files)), \
                 patch.object(p, "validate_bundle", side_effect=_stage_files):
                with self.assertRaises(intake.IntakeHold):
                    p.publish_private(producer_checkout=Path("."),
                                      receipt_file=receipt, api=hub,
                                      downloader=bad_read)
            self.assertFalse(receipt.exists())
            self.assertEqual(hub.uploaded, 1)

    def test_malformed_hub_revision_cannot_issue_receipt(self):
        hub = FakeHub(sha="x" * 40)
        fake_files = {name: name.encode() for name in p.BOUND_FILES}
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "must-not-exist.json"
            with patch.object(p, "build_bundle", return_value=(self.index(), fake_files)), \
                 patch.object(p, "validate_bundle", side_effect=_stage_files):
                with self.assertRaises(intake.IntakeHold):
                    p.publish_private(producer_checkout=Path("."),
                                      receipt_file=receipt, api=hub,
                                      downloader=lambda **k: None)
            self.assertFalse(receipt.exists())

    def test_unexpected_public_visibility_refused(self):
        hub = FakeHub(private=False)
        fake_files = {name: name.encode() for name in p.BOUND_FILES}
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "must-not-exist.json"
            with patch.object(p, "build_bundle", return_value=(self.index(), fake_files)), \
                 patch.object(p, "validate_bundle", side_effect=_stage_files):
                with self.assertRaises(intake.IntakeHold):
                    p.publish_private(producer_checkout=Path("."),
                                      receipt_file=receipt, api=hub,
                                      downloader=lambda **k: None)
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
