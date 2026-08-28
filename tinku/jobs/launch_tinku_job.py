#!/usr/bin/env python3
"""Tinku Jobs plan. UNKNOWN lane. Refuses to fire a Hub job."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train_tinku.py"
BASE_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-job", action="store_true")
    args = parser.parse_args()
    plan = {
        "base_model": BASE_MODEL,
        "jobs": "UNKNOWN",
        "submitted": False,
        "command": [
            "hf",
            "jobs",
            "uv",
            "run",
            "--flavor",
            "a10g-large",
            str(TRAIN),
        ],
    }
    if args.run_job:
        print("[tinku-jobs] refusing to fire: Jobs UNKNOWN lane", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
