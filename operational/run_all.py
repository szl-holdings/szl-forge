#!/usr/bin/env python3
"""Operational orchestrator: build -> (optional) train -> evaluate -> receipts.

Default pass is smoke-safe: it builds the controlled corpus, builds both
remediation curricula, and records UNKNOWN receipts for every evaluation
whose sealed inputs or hardware are absent. Pass --train to also fire the
lane trainers (requires GPU + lane dependencies).

Every step appends to `operational/out/run-receipt.json`. A step that
cannot produce evidence is recorded UNKNOWN — never skipped silently and
never marked MEASURED without an artifact.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RECEIPT = OUT / "run-receipt.json"

BUILD = HERE / "build_curated_datasets.py"
TRAIN_CHAKANA = HERE / "train_chakana.py"
TRAIN_TINKU = HERE / "train_tinku.py"
EVAL_RETRIEVAL = HERE / "evaluate_retrieval.py"
EVAL_RERANKER = HERE / "evaluate_reranker.py"
EVAL_RECEIPTAGENT = HERE / "evaluate_receiptagent_v3.py"
EVAL_WILLAY = HERE / "evaluate_willay.py"
REM_CHASKI = HERE / "remediate_chaski.py"
REM_KHIPU = HERE / "remediate_khipu_r3.py"


def run_step(name: str, script: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(script)] + (extra or [])
    proc = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return {
        "step": name,
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "status": "OK" if proc.returncode == 0 else "FAILED",
        "tail": tail[-3:] if tail else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="fire lane trainers (GPU required)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    steps.append(run_step("build-curated-datasets", BUILD))
    steps.append(run_step("remediate-chaski", REM_CHASKI))
    steps.append(run_step("remediate-khipu-r3", REM_KHIPU))

    if args.train:
        steps.append(run_step("train-chakana", TRAIN_CHAKANA))
        steps.append(run_step("train-tinku", TRAIN_TINKU))
    else:
        steps.append({"step": "train-chakana", "status": "SKIPPED", "reason": "smoke mode (no --train)"})
        steps.append({"step": "train-tinku", "status": "SKIPPED", "reason": "smoke mode (no --train)"})

    steps.append(run_step("evaluate-retrieval", EVAL_RETRIEVAL))
    steps.append(run_step("evaluate-reranker", EVAL_RERANKER))
    steps.append(run_step("evaluate-receiptagent-v3", EVAL_RECEIPTAGENT))
    steps.append(run_step("evaluate-willay", EVAL_WILLAY))

    receipt = {
        "kind": "szl-ops-run-receipt",
        "mode": "train" if args.train else "smoke",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "doctrine": "v11 LOCKED; no Hub PUT; UNKNOWN stated honestly; publication_eligible false everywhere",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    failed = [s["step"] for s in steps if s.get("status") == "FAILED"]
    print(json.dumps({"receipt": str(RECEIPT), "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
