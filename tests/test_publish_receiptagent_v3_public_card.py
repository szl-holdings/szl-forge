"""Offline, synthetic provider controls for the public-v3 README-only writer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import publish_receiptagent_v3_public_card as writer


SOURCE = "a" * 40
RESULT = "b" * 40
INTERVENING = "c" * 40


class FakeOperation:
    def __init__(self, *, path_in_repo: str, path_or_fileobj: bytes):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj
        self._is_committed = False


class ParentRaceError(RuntimeError):
    def __init__(self):
        super().__init__("synthetic stale parent")
        self.response = SimpleNamespace(status_code=412)


class FakeProvider:
    def __init__(self):
        self.head = writer.card.HUB_PARENT
        self.commits: list[dict] = []
        self.auth_calls: list[dict] = []
        self.info_calls: list[dict] = []
        self.mode = "success"
        self.readback = b""
        self.version = "1.23.0"
        self.repo_id = writer.TARGET_REPOSITORY

    def module(self):
        return SimpleNamespace(
            __version__=self.version,
            HfApi=lambda *, token: self._api(token),
            CommitOperationAdd=FakeOperation,
        )

    def _api(self, token: str):
        assert token == "offline-test-token"
        return self

    def auth_check(self, **kwargs):
        self.auth_calls.append(kwargs)

    def whoami(self):
        return {"name": "synthetic-owner"}

    def repo_info(self, **kwargs):
        self.info_calls.append(kwargs)
        return SimpleNamespace(id=self.repo_id, sha=self.head, private=False, gated=False)

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        if self.mode == "provider_exception":
            raise RuntimeError("synthetic ambiguous provider exception")
        if self.mode == "parent_race":
            self.head = INTERVENING
            raise ParentRaceError()
        if self.mode == "client_noop":
            self.head = INTERVENING
            return SimpleNamespace(oid=INTERVENING)
        self.head = RESULT
        kwargs["operations"][0]._is_committed = True
        return SimpleNamespace(oid=RESULT)


def synthetic_candidate() -> bytes:
    return b"\n".join(writer.REQUIRED_HOLD_MARKERS) + b"\n"


def synthetic_review(candidate: bytes) -> dict:
    return {
        "target_repository": writer.TARGET_REPOSITORY,
        "observed_hub_parent": writer.card.HUB_PARENT,
        "changed_paths": ["README.md"],
        "release_status": "UNQUALIFIED",
        "disposition": "REVIEW_ONLY_NO_HUB_WRITE",
        "candidate_readme_sha256": writer.card.digest(candidate),
        "candidate_kind": writer.card.CANDIDATE_KIND,
    }


class PublicCardPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.provider = FakeProvider()
        self.candidate = synthetic_candidate()
        self.receipt_path = Path(self.temp.name) / "receipt.json"
        self.patches = [
            patch.object(writer.card, "CANDIDATE_README_SHA256", writer.card.digest(self.candidate)),
            patch.object(writer.card, "prepare", return_value=(
                self.candidate, "two exact loader spans", synthetic_review(self.candidate),
            )),
            patch.object(writer, "assert_current_main"),
            patch.object(writer, "assert_writer_files_committed"),
            patch.object(writer, "_provider_module", return_value=self.provider.module()),
            patch.object(writer.card, "_get", side_effect=self._readback),
        ]
        self.mocks = [item.start() for item in self.patches]
        for item in self.patches:
            self.addCleanup(item.stop)
        self.receipt = writer.ReceiptFile(self.receipt_path, SOURCE)

    def _readback(self, url: str) -> bytes:
        self.assertIn(f"/raw/{RESULT}/README.md", url)
        return self.provider.readback

    def attempt(self):
        return writer.publish(source_revision=SOURCE, token="offline-test-token", receipt=self.receipt)

    def on_disk(self):
        return json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def test_success_is_one_public_readme_cas_with_immutable_readback(self):
        self.provider.readback = self.candidate
        result = self.attempt()
        self.assertEqual(result["state"], "PUBLISHED_AND_READ_BACK")
        self.assertEqual(result["hub_revision"], RESULT)
        self.assertEqual(result["readback_sha256"], writer.card.digest(self.candidate))
        self.assertEqual(result, self.on_disk())
        self.assertEqual(len(self.provider.commits), 1)
        commit = self.provider.commits[0]
        self.assertEqual(commit["repo_id"], "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3")
        self.assertNotIn("v3-authenticated", commit["repo_id"])
        self.assertEqual(commit["repo_type"], "model")
        self.assertEqual(commit["parent_commit"], writer.card.HUB_PARENT)
        self.assertEqual(commit["revision"], "main")
        self.assertIs(commit["create_pr"], False)
        self.assertEqual(len(commit["operations"]), 1)
        self.assertEqual(commit["operations"][0].path_in_repo, "README.md")
        self.assertEqual(commit["operations"][0].path_or_fileobj, self.candidate)
        self.assertEqual(self.mocks[2].call_count, 3)
        self.assertEqual(self.mocks[3].call_count, 2)
        self.assertEqual(len(self.provider.info_calls), 2)

    def test_parent_drift_blocks_before_any_commit(self):
        self.provider.head = INTERVENING
        with self.assertRaisesRegex(writer.PublicationError, "parent drifted"):
            self.attempt()
        self.assertEqual(self.provider.commits, [])
        self.assertEqual(self.on_disk()["state"], "BLOCKED_NO_WRITE")
        self.assertIs(self.on_disk()["commit_attempted"], False)

    def test_target_identity_drift_blocks_before_any_commit(self):
        self.provider.repo_id = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated"
        with self.assertRaisesRegex(writer.PublicationError, "target identity changed"):
            self.attempt()
        self.assertEqual(self.provider.commits, [])
        self.assertEqual(self.on_disk()["state"], "BLOCKED_NO_WRITE")

    def test_same_bytes_client_shortcut_never_claims_a_cas(self):
        self.provider.mode = "client_noop"
        with self.assertRaisesRegex(writer.PublicationError, "server commit was not confirmed"):
            self.attempt()
        self.assertEqual(len(self.provider.commits), 1)
        self.assertEqual(self.on_disk()["state"], "POST_COMMIT_UNVERIFIED_DO_NOT_RETRY")
        self.assertIs(self.on_disk()["commit_attempted"], True)
        self.assertNotIn("hub_revision", self.on_disk())
        self.mocks[5].assert_not_called()

    def test_provider_exception_is_unknown_and_never_retried(self):
        self.provider.mode = "provider_exception"
        with self.assertRaisesRegex(RuntimeError, "synthetic ambiguous"):
            self.attempt()
        self.assertEqual(len(self.provider.commits), 1)
        self.assertEqual(self.on_disk()["state"], "POST_COMMIT_UNVERIFIED_DO_NOT_RETRY")
        self.mocks[5].assert_not_called()

    def test_intervening_head_412_never_refreshes_parent_or_retries(self):
        self.provider.mode = "parent_race"
        with self.assertRaises(ParentRaceError) as raised:
            self.attempt()
        self.assertEqual(raised.exception.response.status_code, 412)
        self.assertEqual(self.provider.head, INTERVENING)
        self.assertEqual(len(self.provider.commits), 1)
        self.assertEqual(self.provider.commits[0]["parent_commit"], writer.card.HUB_PARENT)
        self.assertEqual(self.on_disk()["state"], "POST_COMMIT_UNVERIFIED_DO_NOT_RETRY")
        self.mocks[5].assert_not_called()

    def test_immutable_readback_mismatch_remains_unverified(self):
        self.provider.readback = b"wrong bytes"
        with self.assertRaisesRegex(writer.PublicationError, "byte readback mismatch"):
            self.attempt()
        self.assertEqual(len(self.provider.commits), 1)
        self.assertEqual(self.on_disk()["hub_revision"], RESULT)
        self.assertEqual(self.on_disk()["state"], "POST_COMMIT_UNVERIFIED_DO_NOT_RETRY")

    def test_nonpromotion_marker_loss_blocks_before_provider_import(self):
        candidate = self.candidate.replace(b"publication_eligible: false", b"publication_eligible: true")
        self.mocks[1].return_value = (candidate, "unreviewed", synthetic_review(candidate))
        with self.assertRaisesRegex(writer.PublicationError, "candidate identity changed"):
            self.attempt()
        self.assertEqual(self.provider.auth_calls, [])
        self.mocks[4].assert_not_called()
        self.assertEqual(self.on_disk()["state"], "BLOCKED_NO_WRITE")

    def test_unreviewed_hub_client_version_blocks_before_api_creation(self):
        self.provider.version = "1.33.0"
        self.mocks[4].return_value = self.provider.module()
        with self.assertRaisesRegex(writer.PublicationError, "reviewed huggingface_hub 1.23.0"):
            self.attempt()
        self.assertEqual(self.provider.auth_calls, [])
        self.assertEqual(self.on_disk()["state"], "BLOCKED_NO_WRITE")

    def test_existing_receipt_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            writer.ReceiptFile(self.receipt_path, SOURCE)
        self.assertEqual(self.on_disk()["state"], "NOT_ATTEMPTED")

    def test_publisher_receipt_contains_only_identity_hash_not_raw_name(self):
        self.provider.readback = self.candidate
        result = self.attempt()
        self.assertEqual(result["publisher_identity_sha256"],
                         writer.card.digest(b"synthetic-owner"))
        self.assertNotIn("publisher", result)
        self.assertNotIn("synthetic-owner", self.receipt_path.read_text(encoding="utf-8"))

    def test_token_echo_identity_blocks_before_any_commit(self):
        for name in ("offline-test-token", "prefix-offline-test-token-suffix"):
            with self.subTest(name=name), patch.object(self.provider, "whoami", return_value={"name": name}):
                with self.assertRaisesRegex(writer.PublicationError, "identity") as raised:
                    self.attempt()
                self.assertNotIn("offline-test-token", str(raised.exception))
                self.assertEqual(self.provider.commits, [])
                self.assertEqual(self.on_disk()["state"], "BLOCKED_NO_WRITE")
                self.assertIs(self.on_disk()["commit_attempted"], False)
                self.assertNotIn(name, self.receipt_path.read_text(encoding="utf-8"))

    def test_malformed_identity_blocks_before_any_commit(self):
        for identity in (None, [], {"name": 123}, {"name": ""}, {"name": " owner"},
                         {"name": "owner\n"}, {"name": "x" * 97}, {"name": "owner/other"},
                         {"name": "owner@example"}, {"name": "ow\u200bner"}):
            with self.subTest(identity=identity), patch.object(self.provider, "whoami", return_value=identity):
                with self.assertRaisesRegex(writer.PublicationError, "identity"):
                    self.attempt()
                self.assertEqual(self.provider.commits, [])
                self.assertEqual(self.on_disk()["state"], "BLOCKED_NO_WRITE")
                self.assertIs(self.on_disk()["commit_attempted"], False)

    def test_other_candidate_kind_is_not_publication_authority(self):
        review = synthetic_review(self.candidate)
        review["candidate_kind"] = "HISTORICAL_DEV_RECONCILIATION"
        self.mocks[1].return_value = (self.candidate, "unreviewed kind", review)
        with self.assertRaisesRegex(writer.PublicationError, "candidate identity changed"):
            self.attempt()
        self.mocks[4].assert_not_called()
        self.assertEqual(self.provider.commits, [])

    def test_review_digest_mismatch_blocks_before_provider_import(self):
        review = synthetic_review(self.candidate)
        review["candidate_readme_sha256"] = "0" * 64
        self.mocks[1].return_value = (self.candidate, "unreviewed digest", review)
        with self.assertRaisesRegex(writer.PublicationError, "candidate identity changed"):
            self.attempt()
        self.mocks[4].assert_not_called()
        self.assertEqual(self.provider.commits, [])


if __name__ == "__main__":
    unittest.main()
