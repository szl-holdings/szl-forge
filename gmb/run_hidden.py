#!/usr/bin/env python3
"""Score the sealed hidden holdout with the same rule baseline.

Does not name a model winner. Leakage is checked against the public
train_user_prompts list in gmb.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from run_gmb import decide, find_leaks, load as load_public

ROOT = Path(__file__).resolve().parent
HIDDEN = ROOT / "hidden_holdout.json"


def load_hidden() -> dict:
    return json.loads(HIDDEN.read_text(encoding="utf-8"))


def run() -> dict:
    public = load_public()
    hidden = load_hidden()
    probe = {
        "train_user_prompts": public["train_user_prompts"],
        "cases": hidden["cases"],
    }
    leaks = find_leaks(probe)
    if leaks:
        raise SystemExit(f"REFUSED hidden leakage: {leaks}")
    passed = 0
    false_allow = 0
    rows = []
    for row in hidden["cases"]:
        got, _ = decide(row)
        fa = row["expect"] == "REFUSE" and got == "DRAFT"
        ok = got == row["expect"] and not fa
        if fa:
            false_allow += 1
        if ok:
            passed += 1
        rows.append({"id": row["id"], "expect": row["expect"], "got": got, "pass": ok})
    if false_allow:
        raise SystemExit(f"REFUSED hidden false-ALLOW {false_allow}")
    return {
        "schema": hidden["schema"],
        "suite": hidden["suite"],
        "runner": "rule-baseline",
        "winner": None,
        "n_total": len(rows),
        "n_passed": passed,
        "n_false_allow": false_allow,
        "leakage_overlaps": 0,
        "metal": "BLOCKED_NO_METAL",
    }


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
