#!/usr/bin/env python3
"""Print the Chaski HF Jobs UV command. This checkout does not fire a job.

Live job is report_to=none. Do not launch another job from forge.
huggingface_hub.run_uv_job is imported only behind an explicit --run-job gate
that this kit refuses.

base_model = Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train_chaski.py"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LIVE_JOB_ID = "6a91bf1045686a1580c12105"
LIVE_JOB_URL = f"https://huggingface.co/jobs/SZLHOLDINGS/{LIVE_JOB_ID}"


def plan() -> dict:
    return {
        "train": str(TRAIN.as_posix()),
        "base_model": BASE_MODEL,
        "flavor": "a10g-large",
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
        "submitted": False,
        "live_job_id": LIVE_JOB_ID,
        "live_job_url": LIVE_JOB_URL,
        "live_job_status": "RUNNING",
        "attempt3_id": "6a91ba00984507d9db4ea07f",
        "attempt3_status": "COMPLETED",
        "attempt4_id": "6a91bb7c984507d9db4ea0a4",
        "attempt4_status": "ERROR",
    }


def fire_job() -> None:
    """Real huggingface_hub Jobs entry. Gated. This checkout never calls it."""
    from huggingface_hub import run_uv_job

    run_uv_job(str(TRAIN), script_args=["--train"], flavor="a10g-large", timeout="2h")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-job", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = plan()
    if args.run_job:
        print(
            "[chaski-jobs] refusing to fire: live report_to=none job already exists "
            f"at {LIVE_JOB_URL}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("# Chaski Jobs plan. Not submitted from this checkout.")
    print(f"# live {LIVE_JOB_ID} {payload['live_job_status']} {LIVE_JOB_URL}")
    print(f"# attempt4 {payload['attempt4_id']} {payload['attempt4_status']}")
    print(f"# attempt3 {payload['attempt3_id']} {payload['attempt3_status']}")
    print(" ".join(payload["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
