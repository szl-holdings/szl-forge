# SPDX-License-Identifier: Apache-2.0
"""Cross-source honesty contract for the Chaski-5050 public model card."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "chaski-5050" / "card" / "README.md"
CANONICAL = ROOT / "chaski" / "README_5050.md"


def _read(path: Path) -> str:
    assert path.is_file() and not path.is_symlink()
    return path.read_text(encoding="utf-8")


def test_card_and_canonical_kit_agree_on_bounded_observations() -> None:
    card = _read(CARD)
    canonical = _read(CANONICAL)

    shared_boundaries = (
        "SZLHOLDINGS/chaski-5050",
        "SZLHOLDINGS/chaski",
        "Qwen/Qwen3.5-0.8B",
        "local-5050",
        "publication_eligible: false",
    )
    for boundary in shared_boundaries:
        assert boundary in card
        assert boundary in canonical

    assert "r=16, α=16" in card
    assert "r=16, **alpha=16**" in canonical
    assert "41 rows" in card
    assert "jsonl-only `szl_dataset.jsonl`" in canonical
    # The recipe's training-era no-eval stamp is not the whole card's state.
    assert "SKU eval none-this-run" in canonical
    assert "evals: HISTORICAL_TRAINING_NONE_SEPARATE_GATE_FAIL" in card
    assert "`evals=none-this-run`: no JSON/refusal gate ran in that training step." in card
    assert "**1/5 held-out and overall FAIL**" in card
    assert "never_overwrite: SZLHOLDINGS/chaski" in card
    assert "never overwrite" in canonical


def test_card_keeps_training_and_promotion_claims_separate() -> None:
    card = _read(CARD)
    flattened = " ".join(card.split())
    lowered = card.casefold()

    required = (
        "autonomy_eligible: false",
        "**1/5 held-out and overall FAIL**",
        "heldout.refusal_no_regression=false",
        "Production and publication authorization are **false**.",
        "The card does not authorize artifact changes or model promotion.",
        "copied_live_chaski_weights: false",
        "Lab load is forbidden.",
        "No 5/5 or 6/6 qualification is claimed.",
        "620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5",
        "Canonical card authoring source: [chaski-5050/card/README.md]",
        "nonportable owner-local base path",
        "Runtime loading has not been verified.",
    )
    for boundary in required:
        assert boundary in flattened

    forbidden = (
        "nobody else ships this combination",
        "one-of-one",
        "world-class",
        "best in the world",
        "publication_eligible: true",
        "autonomy_eligible: true",
        "status: production ready",
    )
    for claim in forbidden:
        assert claim not in lowered


def test_canonical_kit_still_denies_implicit_provider_mutation() -> None:
    canonical = _read(CANONICAL)
    assert "No Hub PUT from this checkout." in canonical
    assert "`--train` is local GPU only." in canonical
    assert "**Not** live `SZLHOLDINGS/chaski`." in canonical
    assert "No Khipu lab pin. No tok/s claims." in canonical
