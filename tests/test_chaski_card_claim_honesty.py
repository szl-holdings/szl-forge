# SPDX-License-Identifier: Apache-2.0
"""Bounded-claim contracts for the two source-owned Chaski model cards."""
from __future__ import annotations

import re
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
    assert "Lineage and evaluation applicability were not independently" in chaski_5050


def test_chaski_cards_point_to_their_own_canonical_source() -> None:
    chaski = _card("chaski/card/README.md")
    chaski_5050 = _card("chaski-5050/card/README.md")
    assert "szl-holdings/szl-forge/tree/main/chaski/card" in chaski
    assert re.search(
        r"https://github\.com/szl-holdings/szl-forge/blob/"
        r"[0-9a-f]{40}/chaski-5050/card/README\.md",
        chaski_5050,
    )
    assert "Canonical card authoring source: [chaski-5050/card/README.md]" in chaski_5050
    assert "tree/main/chaski/card" not in chaski_5050


def test_negative_evidence_and_quarantine_cannot_be_rewritten_as_promotion() -> None:
    chaski = _card("chaski/card/README.md")
    chaski_5050 = _card("chaski-5050/card/README.md")
    flattened_5050 = " ".join(chaski_5050.split())

    assert "Named-N: MEASURED FAIL" in chaski
    assert "json_draft: 0/5" in chaski
    assert "adversarial_refusal: 2/6" in chaski
    assert "publication_eligible: false" in chaski
    assert "autonomy_eligible: false" in chaski
    assert "Not evidence that a later SKU inherits this evaluation." in chaski

    assert "evals: HISTORICAL_TRAINING_NONE_SEPARATE_GATE_FAIL" in chaski_5050
    assert "`evals=none-this-run`: no JSON/refusal gate ran in that training step." in chaski_5050
    assert "**1/5 held-out and overall FAIL**" in chaski_5050
    assert "publication_eligible: false" in chaski_5050
    assert "autonomy_eligible: false" in chaski_5050
    assert "never_overwrite: SZLHOLDINGS/chaski" in chaski_5050
    assert "copied_live_chaski_weights: false" in chaski_5050
    assert "Lab load is forbidden." in chaski_5050
    assert "The card does not authorize artifact changes or model promotion." in flattened_5050


def test_chaski_r4_card_states_experimental_not_promotable_and_keeps_a_b_unresolved() -> None:
    r4 = _card("chaski_r4/card/README.md")
    lowered = r4.lower()
    for banned in ("nobody else ships", "one-of-one", "state-of-the-art", "best-in-class", "production ready"):
        assert banned not in lowered
    assert "artifact_state: PUBLIC_EXPERIMENTAL_ARTIFACT" in r4
    assert "promotion: NOT_PROMOTABLE" in r4
    assert "publication_eligible: false" in r4
    assert "autonomy_eligible: false" in r4
    assert "5ae3de970014726f190dfe8e4d5b35e667fd737b1be29e27205c25d143bdf403" in r4
    assert "**5/5 JSON drafts; 6/6 adversarial refusals**" in r4
    assert "0/5 JSON drafts; 3/6 refusals" in r4
    assert "provenance of receipts A and B remains unresolved" in r4
    assert "No uniqueness claim is made." in r4
    assert "szl-holdings/szl-forge/tree/main/chaski_r4/card" in r4
