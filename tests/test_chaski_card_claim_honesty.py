# SPDX-License-Identifier: Apache-2.0
"""Bounded-claim contracts for the two source-owned Chaski model cards."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _card(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_chaski_cards_make_no_unverified_uniqueness_or_leadership_claim() -> None:
    chaski = _card("chaski/card/README.md")
    chaski_5050 = _card("chaski-5050/card/README.md")
    for card in (chaski, chaski_5050):
        lowered = card.casefold()
        assert "nobody else ships" not in lowered
        assert "one-of-one" not in lowered
        assert "state-of-the-art" not in lowered
        assert "best-in-class" not in lowered
    assert "No uniqueness claim is made." in chaski
    assert "The differentiator is inspectability" in chaski_5050


def test_chaski_cards_point_to_their_own_canonical_source() -> None:
    chaski = _card("chaski/card/README.md")
    chaski_5050 = _card("chaski-5050/card/README.md")
    assert "szl-holdings/szl-forge/tree/main/chaski/card" in chaski
    assert "szl-holdings/szl-forge/blob/main/chaski/README_5050.md" in chaski_5050
    assert "tree/main/chaski/card" not in chaski_5050


def test_negative_evidence_and_quarantine_cannot_be_rewritten_as_promotion() -> None:
    chaski = _card("chaski/card/README.md")
    chaski_5050 = _card("chaski-5050/card/README.md")

    assert "Named-N: MEASURED FAIL" in chaski
    assert "json_draft: 0/5" in chaski
    assert "adversarial_refusal: 2/6" in chaski
    assert "publication_eligible: false" in chaski
    assert "autonomy_eligible: false" in chaski
    assert "Not evidence that a later SKU inherits this evaluation." in chaski

    assert "Status: none-this-run." in chaski_5050
    assert "publication_eligible: false" in chaski_5050
    assert "autonomy_eligible: false" in chaski_5050
    assert "never_overwrite: SZLHOLDINGS/chaski" in chaski_5050
    assert "copied_live_chaski_weights: false" in chaski_5050
    assert "Do not load this ID into the Khipu lab." in chaski_5050
    assert "documentation update, not model promotion" in chaski_5050
