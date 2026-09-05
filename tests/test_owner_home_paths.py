from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHASKI = ROOT / "chaski"
FORBIDDEN = ("C:\\Users", "C:/Users", "/Users/")


def test_chaski_receipts_and_5050_recipe_have_no_owner_homes() -> None:
    targets = [
        CHASKI / "bakeoff_named_n.receipt.json",
        CHASKI / "train_chaski_bf16_5050.py",
        CHASKI / "README_5050.md",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN:
            assert needle not in text, f"{path.name} still contains {needle!r}"
