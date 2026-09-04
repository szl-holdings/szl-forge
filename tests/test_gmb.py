import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gmb"))
import run_gmb


def test_gmb_rule_baseline_is_clean() -> None:
    report = run_gmb.run()
    assert report["winner"] is None
    assert report["n_false_allow"] == 0
    assert report["n_passed"] == report["n_total"]
    assert report["n_total"] >= 16


def test_gmb_gold_does_not_overlap_train() -> None:
    catalog = json.loads((ROOT / "gmb" / "gmb.json").read_text(encoding="utf-8"))
    assert run_gmb.leakage(catalog) == []
