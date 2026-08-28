#!/usr/bin/env python3
"""Print the KHIPU-R2 HF Jobs UV command. This checkout does not fire a job.

Live Hub job 6a91bf11984507d9db4ea104 is COMPLETED. This-kit jobs UNKNOWN.
Do not launch another Hub job from forge. No Hub PUT. Never retarget
SZLHOLDINGS/SZL-Khipu-1.5B.

base_model = Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train_khipu_r2.py"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HUB = "SZLHOLDINGS/KHIPU-R2"
FORBIDDEN_HUB = "SZLHOLDINGS/SZL-Khipu-1.5B"
HUB_JOB_ID = "6a91bf11984507d9db4ea104"
HUB_JOB_STATUS = "COMPLETED"
HUB_JOB_URL = f"https://huggingface.co/jobs/SZLHOLDINGS/{HUB_JOB_ID}"
HUB_ADAPTER_STATUS = "AVAILABLE"
HUB_ADAPTER_SIZE = "147.8MB"


def plan() -> dict:
    return {
        "train": str(TRAIN.as_posix()),
        "base_model": BASE_MODEL,
        "hub": HUB,
        "does_not_overwrite": FORBIDDEN_HUB,
        "flavor": "a10g-large",
        "hub_job_id": HUB_JOB_ID,
        "hub_job_status": HUB_JOB_STATUS,
        "hub_job_url": HUB_JOB_URL,
        "hub_adapter": HUB_ADAPTER_STATUS,
        "hub_adapter_size": HUB_ADAPTER_SIZE,
        "jobs": "UNKNOWN",
        "jobs_scope": "this-kit",
        "submitted": False,
        "hub_put": False,
        "publication_eligible": False,
        "lab": "signed Khipu GGUF",
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
            "[khipu-r2-jobs] refusing to fire: Hub job "
            f"{HUB_JOB_ID} already {HUB_JOB_STATUS}. "
            "This-kit jobs UNKNOWN. Do not overwrite signed SZL-Khipu-1.5B. "
            "No Hub PUT.",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("# KHIPU-R2 Jobs plan. Not submitted from this checkout.")
    print(
        f"# hub_job={HUB_JOB_ID} {HUB_JOB_STATUS} "
        f"adapter={HUB_ADAPTER_STATUS} ({HUB_ADAPTER_SIZE})"
    )
    print(f"# this-kit jobs=UNKNOWN base_model={BASE_MODEL} hub={HUB}")
    print(f"# does_not_overwrite={FORBIDDEN_HUB}")
    print(" ".join(payload["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
