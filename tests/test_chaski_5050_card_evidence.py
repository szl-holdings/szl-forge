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

    assert "LoRA r=16 α=16" in card
    assert "r=16, **alpha=16**" in canonical
    assert "41 rows" in card
    assert "jsonl-only `szl_dataset.jsonl`" in canonical
    assert "evals: none-this-run" in card
    assert "SKU eval none-this-run" in canonical
    assert "never_overwrite: SZLHOLDINGS/chaski" in card
    assert "never overwrite" in canonical


def test_card_keeps_training_and_promotion_claims_separate() -> None:
    card = _read(CARD)
    lowered = card.casefold()

    required = (
        "autonomy_eligible: false",
        "No signed held-out evaluation receipt for this 5050 adapter is present in this source tree.",
        "Publishing this card is a documentation update, not model promotion",
        "Card, banner, and source binding only; weights, adapter, configs, evals, visibility, hardware, collection, and runtime state unchanged",
        "copied_live_chaski_weights: false",
        "Do not load this ID into the Khipu lab.",
        "620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5",
        "Canonical GitHub: [`chaski/README_5050.md`]",
    )
    for boundary in required:
        assert boundary in card

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
