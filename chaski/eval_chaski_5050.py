#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Chaski-5050 eval — none-this-run without a local adapter.

NEW Hub id: SZLHOLDINGS/chaski-5050. Not live SZLHOLDINGS/chaski.
Train loss is not an eval. No fabricated k/n. No MEASURED evals.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski-5050"
LIVE_CHASKI_HUB = "SZLHOLDINGS/chaski"
ADAPTER_DIR = HERE / "chaski-5050-adapter"


def main() -> int:
    local = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    report = {
        "kind": "szl-chaski-5050-eval-report",
        "artifact": HUB,
        "not_live_chaski": LIVE_CHASKI_HUB,
        "base_model": CANONICAL_BASE,
        "evals": "none-this-run",
        "quality": "UNKNOWN",
        "train_loss_is_not_eval": True,
        "adapter": "LOCAL" if local else "UNAVAILABLE",
        "weights": "LOCAL" if local else "UNAVAILABLE",
        "jobs": "not-an-hf-job",
        "hardware": "local-RTX-5050",
        "publication_eligible": False,
        "claim_boundary": (
            "No JSON/refusal gate ran. Do not claim 5/5 or 6/6. "
            "Without a local adapter, evals remain none-this-run. "
            "Train loss is not an eval. Not live SZLHOLDINGS/chaski."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report_5050.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[chaski-5050-eval] evals=none-this-run quality=UNKNOWN")
    print(f"[chaski-5050-eval] artifact={HUB} adapter={report['adapter']}")
    print(f"[chaski-5050-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
