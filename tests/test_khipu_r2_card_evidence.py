# SPDX-License-Identifier: Apache-2.0
"""Cross-source honesty contract for the KHIPU-R2 public model card."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "khipu-r2" / "card" / "README.md"
CANONICAL = ROOT / "khipu_r2" / "README.md"


def _read(path: Path) -> str:
    assert path.is_file() and not path.is_symlink()
    return path.read_text(encoding="utf-8")


def test_card_and_canonical_kit_agree_on_bounded_observations() -> None:
    card = _read(CARD)
    canonical = _read(CANONICAL)

    shared_boundaries = (
        "SZLHOLDINGS/KHIPU-R2",
        "SZLHOLDINGS/SZL-Khipu-1.5B",
        "6a91bf11984507d9db4ea104",
        "publication_eligible: false",
    )
    for boundary in shared_boundaries:
        assert boundary in card
        assert boundary in canonical

    assert "adapter_model.safetensors" in card
    assert "adapter **147.8MB AVAILABLE**" in canonical
    assert "Abstain is MEASURED 3/6, not a pass." in card
    assert "abstain **MEASURED 3/6** (not a pass" in canonical
    assert "| grounding (`eval.jsonl` navigate) | **5 / 5** | n=5 |" in card
    assert "grounding **5/5**" in canonical
    assert "| plan-valid | **11 / 11** | not a public leaderboard |" in card
    assert "plan **11/11**" in canonical


def test_card_keeps_training_and_promotion_claims_separate() -> None:
    card = _read(CARD)
    lowered = card.casefold()

    required = (
        "autonomy_eligible: false",
        "No signed R2 eval receipt in this atelier.",
        "Publishing this card is a documentation update, not model promotion",
        "Exact-source README + local SVG + source binding only",
        "held_out_in_gradients: false",
        "Do not derive a world-rank score from k/n on n=11.",
        "Mini does **not** inherit this 3/6",
        "e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349",
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
    assert "No Hub PUT." in canonical
    assert "This checkout does not fire a job" in canonical
    assert "--run-job` on the launcher exits 2" in canonical
    assert "Not an autonomous agent" in canonical
