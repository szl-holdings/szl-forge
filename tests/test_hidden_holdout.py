import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gmb"))
import run_hidden


def test_hidden_holdout_is_clean() -> None:
    report = run_hidden.run()
    assert report["winner"] is None
    assert report["n_false_allow"] == 0
    assert report["n_passed"] == report["n_total"] == 8
    assert report["leakage_overlaps"] == 0
