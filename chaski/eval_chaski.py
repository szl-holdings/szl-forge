#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Chaski eval — evals none-this-run. No fabricated k/n. No serve pin.

base_model = Qwen/Qwen3.5-0.8B
Attempt 3 COMPLETED receipt-only; weights UNAVAILABLE; no adapter to score.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski"

# Job stamps live in train_chaski.py. This eval does not restamp them.


def main() -> int:
    report = {
        "kind": "szl-chaski-eval-report",
        "artifact": HUB,
        "base_model": BASE_MODEL,
        "evals": "none-this-run",
        "quality": "UNKNOWN",
        "train_loss_is_not_eval": True,
        "adapter": "UNAVAILABLE",
        "weights": "UNAVAILABLE",
        "publication_eligible": False,
        "claim_boundary": (
            "No JSON/refusal gate ran. Do not claim 5/5 or 6/6. "
            "CUTTING until an adapter file lands."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[chaski-eval] evals=none-this-run quality=UNKNOWN")
    print(f"[chaski-eval] base_model={BASE_MODEL}")
    print(f"[chaski-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
