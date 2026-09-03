#!/usr/bin/env python3
"""Drive the Chakana lane trainer with locally built admitted pairs.

This wrapper does not re-implement training. It stages
`operational/out/chakana-pairs.jsonl` (built by build_curated_datasets.py)
and invokes the canonical lane script:

    python chakana/train_chakana.py --train --dataset-file <pairs>

The lane script owns base-model admission, third-party dataset refusal,
seed, Matryoshka dims, and the training receipt. This wrapper only:
  - refuses to run if the pairs file is missing (build first)
  - refuses to run twice over the same out dir without --force
  - copies the lane training receipt into operational/out/ for the
    run_all.py orchestrator to consume
  - never touches the Hub (the lane refuses Hub PUT itself)

Requires a GPU (or very patient CPU) and the lane's own dependencies.
Default smoke limits can be tuned with env vars the lane already honors
(e.g. BASE_MODEL). Doctrinal status: smoke run, publication_eligible false.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
PAIRS = OUT / "chakana-pairs.jsonl"
LANE = ROOT / "chakana" / "train_chakana.py"
LANE_RECEIPT = ROOT / "chakana" / "training_receipt.json"
OPS_RECEIPT = OUT / "chakana-train-receipt.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite a prior ops receipt")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if not PAIRS.is_file():
        raise SystemExit(f"[ops-chakana] {PAIRS} missing; run build_curated_datasets.py first")
    if OPS_RECEIPT.exists() and not args.force:
        raise SystemExit(f"[ops-chakana] {OPS_RECEIPT} exists; pass --force to retrain")

    proc = subprocess.run(
        [args.python, str(LANE), "--train", "--dataset-file", str(PAIRS)],
        cwd=ROOT,
    )
    payload = {
        "kind": "szl-ops-chakana-train",
        "lane_exit_code": proc.returncode,
        "pairs": str(PAIRS),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "publication_eligible": False,
        "note": "Lane receipt copied verbatim; evals UNKNOWN until evaluate_retrieval.py runs.",
    }
    if LANE_RECEIPT.is_file():
        payload["lane_receipt"] = json.loads(LANE_RECEIPT.read_text(encoding="utf-8"))
    else:
        payload["lane_receipt"] = "UNAVAILABLE"
    OPS_RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ops-chakana] wrote {OPS_RECEIPT}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
