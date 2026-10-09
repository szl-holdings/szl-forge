"""Offline, immutable public-card fixtures for the loader-only correction."""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "loader_card", ROOT / "tools" / "prepare_receiptagent_v3_card_sync.py"
)
assert SPEC is not None and SPEC.loader is not None
card = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(card)

PARENT = "d674c37cae7127021ba36820ea7a6774cec06255"
OLD_DIGEST = "79fa68b40bcd8cdfff3c4c0a50a7f4c452bf62b080fa95080baffb02a9430555"
NEW_DIGEST = "21f74eebf9a1033054bfd6bc9306c3322ade2a134a3b88125a895e2f6006dd9c"
FIXTURE = ROOT / "tests" / "fixtures" / "receiptagent_v3_card" / "d674c37.README.md"


class ReceiptAgentV3LoaderCardTests(unittest.TestCase):
    def test_fixture_is_exact_immutable_public_readme(self):
        body = FIXTURE.read_bytes()
        self.assertEqual(len(body), 10885)
        self.assertEqual(card.digest(body), OLD_DIGEST)
        self.assertNotIn(b"\r", body)

    def test_fixture_line_endings_are_pinned_for_windows_checkout(self):
        result = subprocess.run(
            ["git", "-C", str(ROOT), "check-attr", "eol", "--",
             "tests/fixtures/receiptagent_v3_card/d674c37.README.md"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip(),
                         "tests/fixtures/receiptagent_v3_card/d674c37.README.md: eol: lf")

    def test_candidate_matches_independently_reviewed_bytes(self):
        changed = card.candidate(FIXTURE.read_bytes())
        self.assertEqual(len(changed), 11072)
        self.assertEqual(card.digest(changed), NEW_DIGEST)
        self.assertEqual(card.HUB_PARENT, PARENT)
        self.assertEqual(card.HUB_README_SHA256, OLD_DIGEST)
        self.assertEqual(card.CANDIDATE_README_SHA256, NEW_DIGEST)
        self.assertEqual(card.CANDIDATE_KIND, "LOADER_EXAMPLE_WITHDRAWAL_ONLY")

    def test_reversing_only_two_spans_restores_every_original_byte(self):
        before = FIXTURE.read_bytes()
        after = card.candidate(before)
        spans = (
            (card.OLD_LOAD_SECTION, card.NEW_LOAD_SECTION),
            (card.OLD_LOADER_PHRASE, card.NEW_LOADER_PHRASE),
        )
        for old, new in spans:
            self.assertEqual(before.count(old.encode()), 1)
            self.assertEqual(after.count(new.encode()), 1)
            after = after.replace(new.encode(), old.encode(), 1)
        self.assertEqual(after, before)

    def test_development_results_and_every_release_hold_are_preserved(self):
        before = FIXTURE.read_bytes()
        after = card.candidate(before)
        for marker in (
            b"publication_eligible: false", b"autonomy_eligible: false",
            b"promotion_effect=NONE", b"base model `UNRECORDED`",
            b"**11/12**", b"**12/12**",
            b"No autonomy, deployment, promotion, or flagship selection",
            card.NEW_EVALS.encode(), card.NEW_SOURCE_NOTE.encode(),
        ):
            with self.subTest(marker=marker):
                self.assertGreater(before.count(marker), 0)
                self.assertEqual(after.count(marker), before.count(marker))

    def test_unqualified_loader_claims_are_absent(self):
        after = card.candidate(FIXTURE.read_bytes())
        self.assertNotIn(b"AutoModelForCausalLM", after)
        self.assertNotIn(b"loadable with `transformers`", after)
        self.assertIn(b"Loader compatibility is UNKNOWN.", after)
        self.assertIn(b"No weights were loaded and no inference was run", after)
        self.assertIn(b"Qwen3_5ForCausalLM", after)
        self.assertIn(b"conditional-generation base class", after)

    def test_each_missing_duplicate_partial_or_consumed_anchor_is_blocked(self):
        before = FIXTURE.read_bytes()
        for old, new in (
            (card.OLD_LOAD_SECTION, card.NEW_LOAD_SECTION),
            (card.OLD_LOADER_PHRASE, card.NEW_LOADER_PHRASE),
        ):
            for body in (
                before.replace(old.encode(), b"missing", 1),
                before + old.encode(),
                before.replace(old.encode(), new.encode(), 1),
                before + new.encode(),
            ):
                with self.subTest(anchor=old[:40], size=len(body)):
                    with self.assertRaises(card.CardSyncError):
                        card.candidate(body)
        with self.assertRaises(card.CardSyncError):
            card.candidate(card.candidate(before))

    def test_crlf_input_is_rejected_not_silently_normalized(self):
        with self.assertRaisesRegex(card.CardSyncError, "line endings"):
            card.candidate(FIXTURE.read_bytes().replace(b"\n", b"\r\n"))


if __name__ == "__main__":
    unittest.main()
