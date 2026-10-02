"""Offline controls for the one-shot, non-publishing v3 card candidate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_receiptagent_v3_card_sync", ROOT / "tools" / "prepare_receiptagent_v3_card_sync.py"
)
assert SPEC is not None and SPEC.loader is not None
card = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(card)


def records():
    common = {
        "artifact": card.HUB_REPO,
        "split": "DEV",
        "n": 12,
        "opened_test_jsonl": False,
        "publication_eligible": False,
        "draft": "4/4",
        "refuse": "4/4",
    }
    old = {**common, "correct": 11, "recovery": "3/4"}
    new = {
        **common,
        "correct": 12,
        "recovery": "4/4",
        "promotion_effect": "NONE",
        "base_model": "UNRECORDED",
        "artifact_sha256": {"adapter_model.safetensors": card.ADAPTER_SHA256},
        "relationship": "The 11/12 record is not superseded.",
    }
    return old, new


def hub_fixture():
    return (
        "---\nlicense: apache-2.0\nszl:\n  publication_eligible: false\n---\n"
        "Unmanaged hero and loader guidance.\n"
        + card.OLD_CLAIM
        + "\nSome unrelated paragraph.\n"
        + card.OLD_EXPLANATION
        + "\n"
        + card.OLD_EVIDENCE
        + "\nUnmanaged legal and release HOLD notes.\n"
    ).encode("utf-8")


class ReceiptAgentV3CardSyncTests(unittest.TestCase):
    def test_exact_committed_source_card_is_bound(self):
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(card.bind_source(ROOT, revision), card.SOURCE_CARD_SHA256)
        with self.assertRaisesRegex(card.CardSyncError, "exact 40-character"):
            card.bind_source(ROOT, "main")

    def test_surgical_candidate_preserves_every_unmanaged_byte(self):
        before = hub_fixture()
        after = card.candidate(before)
        self.assertEqual(after.count(b"11/12"), 3)
        self.assertIn(b"12/12", after)
        self.assertNotIn(b"contradicts the committed receipt", after)
        self.assertIn(b"Unmanaged hero and loader guidance.", after)
        self.assertIn(b"Unmanaged legal and release HOLD notes.", after)
        self.assertIn(b"publication_eligible: false", after)
        for old, new in (
            (card.OLD_CLAIM, card.NEW_CLAIM),
            (card.OLD_EXPLANATION, card.NEW_EXPLANATION),
            (card.OLD_EVIDENCE, card.NEW_EVIDENCE),
        ):
            after = after.replace(new.encode("utf-8"), old.encode("utf-8"), 1)
        self.assertEqual(after, before)

    def test_partial_duplicate_or_already_edited_hub_card_fails_closed(self):
        fixture = hub_fixture()
        for body in (
            fixture.replace(card.OLD_CLAIM.encode(), b"missing", 1),
            fixture + card.OLD_CLAIM.encode(),
            fixture.replace(card.OLD_EXPLANATION.encode(), card.NEW_EXPLANATION.encode(), 1),
            fixture.replace(b"\n", b"\r\n"),
        ):
            with self.subTest(body_length=len(body)):
                with self.assertRaises(card.CardSyncError):
                    card.candidate(body)

    def test_dev_identity_and_promotion_limits_are_not_optional(self):
        historical, additive = records()
        card.validate_dev_records(historical, additive)
        for mutation in (
            {"opened_test_jsonl": True},
            {"correct": 11},
            {"promotion_effect": "PROMOTED"},
            {"base_model": "Qwen/Qwen3.5-0.8B"},
            {"artifact_sha256": {"adapter_model.safetensors": "0" * 64}},
            {"publication_eligible": True},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(card.CardSyncError):
                    card.validate_dev_records(historical, {**additive, **mutation})

    def test_parent_drift_stops_before_reading_any_hub_file(self):
        wrong = {"id": card.HUB_REPO, "sha": "0" * 40, "private": False, "gated": False}
        with patch.object(card, "_get", return_value=json.dumps(wrong).encode()) as fetch:
            with self.assertRaisesRegex(card.CardSyncError, "parent drifted"):
                card.read_hub_parent()
        fetch.assert_called_once()

    def test_receipt_never_claims_hub_publication_or_release(self):
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        historical, additive = records()
        with patch.object(card, "read_hub_parent", return_value=(hub_fixture(), historical, additive)):
            changed, diff, receipt = card.prepare(ROOT, revision)
        self.assertEqual(receipt["changed_paths"], ["README.md"])
        self.assertEqual(receipt["disposition"], "REVIEW_ONLY_NO_HUB_WRITE")
        self.assertEqual(receipt["release_status"], "UNQUALIFIED")
        self.assertEqual(receipt["candidate_readme_sha256"], card.digest(changed))
        self.assertIn("+| Additive adapter-bound development check", diff)


if __name__ == "__main__":
    unittest.main()
