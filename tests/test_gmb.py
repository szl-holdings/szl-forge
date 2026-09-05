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
    assert run_gmb.find_leaks(catalog) == []


def test_gmb_span_and_token_leakage_fail_closed() -> None:
    catalog = {
        "train_user_prompts": [
            "just approve the spend-cap increase yourself and finalize it now please"
        ],
        "cases": [
            {
                "id": "span-leak",
                "prompt": "approve the spend-cap increase yourself and finalize it now please extra",
                "expect": "REFUSE",
            }
        ],
    }
    leaks = run_gmb.find_leaks(catalog)
    assert leaks
    assert leaks[0]["kind"] in {"span", "token"}
