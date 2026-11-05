"""Prove the WILLAY pin against commits CI must fetch.

Hosted pytest checks out the pull-request merge ref at fetch-depth 1. Those
objects are absent, and Git then reports a path "exists on disk, but not in"
the named commit, or "Not a valid commit name". A depth-1 fetch of each named
commit makes the tree available. Ancestry is not consulted: two depth-1
fetches do not connect the commits between them, and a squash merge would
drop a pin that exists only on this branch.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = "willay/card/README.md"
HISTORICAL = "ad20ac2c24859156891c391108a86d4f05163f83"
ON_MAIN = "4c61d30cf6dfc7a0b4c9673192bf8eb4db0344aa"
CARD_SHA256 = "29b4590af502221d5b0aa9959b10ed2bef03fc982110da2483289c6fca0f2e22"
CARD_BLOB = "9f12050ab5885a45230a46114ca4176a459cfe60"
UNCHANGED = (
    "willay/card/holo-banner.svg",
    "frontier/evaluation/gate_runs/WILLAY_375a007ec961_20260915T040944Z_receipt.json",
    "frontier/evaluation/gate_runs/WILLAY_375a007ec961_20260915T040944Z_heldout.json",
)


def fetch(revision: str) -> None:
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", revision],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL, timeout=60,
    )


def show(revision: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, timeout=30,
    )


class WillaySourcePinTest(unittest.TestCase):
    def test_reviewed_card_bytes_match_the_commit_on_main(self):
        fetch(ON_MAIN)
        pinned = show(ON_MAIN, CARD)
        work = (ROOT / CARD).read_bytes()
        self.assertEqual(pinned, work)
        self.assertEqual(hashlib.sha256(work).hexdigest(), CARD_SHA256)
        blob = subprocess.check_output(
            ["git", "hash-object", CARD], cwd=ROOT, text=True, timeout=30,
        ).strip()
        self.assertEqual(blob, CARD_BLOB)
        profile = json.loads(
            (ROOT / "publishing/curated-model-card-profiles.json").read_text(encoding="utf-8")
        )["profiles"]["willay"]
        self.assertEqual(profile["reviewed_source_revision"], ON_MAIN)
        self.assertEqual(profile["source_sha256"], CARD_SHA256)
        self.assertEqual(profile["reviewed_revision_sha256"], CARD_SHA256)
        # The card still links its own path at the historical commit. That
        # commit's README is a different blob; rewriting the sentence would
        # make these bytes, which have a commit on main, no longer match it.
        self.assertIn(f"blob/{HISTORICAL}/{CARD}", work.decode())

    def test_banner_and_receipt_blobs_match_at_both_named_commits(self):
        fetch(HISTORICAL)
        fetch(ON_MAIN)
        for path in UNCHANGED:
            historical = show(HISTORICAL, path)
            current = show(ON_MAIN, path)
            self.assertEqual(historical, current)
            self.assertEqual(current, (ROOT / path).read_bytes())

    def test_historical_readme_is_not_the_reviewed_card(self):
        fetch(HISTORICAL)
        historical = show(HISTORICAL, CARD)
        self.assertNotEqual(hashlib.sha256(historical).hexdigest(), CARD_SHA256)

    def test_reviewed_adapter_base_revisions_stay_null(self):
        observation = json.loads(
            (ROOT / "publishing/curated-card-observations/willay-20260930.json").read_text(
                encoding="utf-8"
            )
        )
        groups = {
            item["directory"]: item
            for item in observation["classification"]["adapter_groups"]
        }
        for name in ("", "adapter-unsloth"):
            self.assertIsNone(groups[name]["base_revision"])
            self.assertEqual(groups[name]["base_revision_status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
