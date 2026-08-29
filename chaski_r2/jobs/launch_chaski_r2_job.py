#!/usr/bin/env python3
"""Print the CHASKI-R2 HF Jobs UV command. This checkout does not fire a job.

Jobs UNAVAILABLE. Do not launch a Hub job from forge. No Hub PUT.
Never retarget SZLHOLDINGS/chaski or SZLHOLDINGS/chaski-5050.

CANONICAL_BASE = Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train_chaski_r2.py"
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski-r2"
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
FORBIDDEN_5050 = "SZLHOLDINGS/chaski-5050"


def plan() -> dict:
    return {
        "train": str(TRAIN.as_posix()),
        "canonical_base": CANONICAL_BASE,
        "base_model": CANONICAL_BASE,
        "hub": HUB,
        "atelier_lock": True,
        "hub_id_declared_only": True,
        "hub_page": False,
        "does_not_overwrite": FORBIDDEN_HUB,
        "forbidden_5050": FORBIDDEN_5050,
        "flavor": "a10g-large",
        "jobs": "UNAVAILABLE",
        "submitted": False,
        "hub_put": False,
        "publication_eligible": False,
        "command": [
            "hf",
            "jobs",
            "uv",
            "run",
            "--flavor",
            "a10g-large",
            "--timeout",
            "2h",
            str(TRAIN),
            "--train",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-job", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = plan()
    if args.run_job:
        print(
            "[chaski-r2-jobs] refusing to fire: Jobs UNAVAILABLE this checkout. "
            "Do not overwrite SZLHOLDINGS/chaski. Not the 5050 kit. No Hub PUT.",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("# CHASKI-R2 Jobs plan. Not submitted from this checkout.")
    print("# ATELIER lock: declared Hub id only; no README-only costume.")
    print(f"# jobs=UNAVAILABLE canonical_base={CANONICAL_BASE} hub={HUB}")
    print(f"# does_not_overwrite={FORBIDDEN_HUB}")
    print(f"# forbidden_5050={FORBIDDEN_5050}")
    print(" ".join(payload["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
