#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Chaski-5050 eval — none-this-run without a local adapter.

SKU: SZLHOLDINGS/chaski-5050. Not live SZLHOLDINGS/chaski.
job=local-5050. Train loss MEASURED is not an eval. No 5/5.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski-5050"
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
ADAPTER_DIR = HERE / "chaski-5050-adapter"
JOB = "local-5050"


def main() -> int:
    local = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    report = {
        "kind": "szl-chaski-5050-eval-report",
        "artifact": HUB,
        "forbidden_hub": FORBIDDEN_HUB,
        "base_model": CANONICAL_BASE,
        "job": JOB,
        "evals": "none-this-run",
        "quality": "UNKNOWN",
        "publication_eligible": False,
        "train_loss_is_not_eval": True,
        "label": "REPORTED owner-metal",
        "adapter": "LOCAL" if local else "UNAVAILABLE",
        "weights": "LOCAL" if local else "UNAVAILABLE",
        "khipu_lab_pin": False,
        "a11oy_mini": False,
        "claim_boundary": (
            "Evals none-this-run. Not 5/5. publication_eligible false. "
            "train_loss may be MEASURED as a train metric, not an eval. "
            "Not live SZLHOLDINGS/chaski. A11OY-MINI is live-Chaski GGUF."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report_5050.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[chaski-5050-eval] evals=none-this-run publication_eligible=false")
    print(f"[chaski-5050-eval] artifact={HUB} job={JOB} adapter={report['adapter']}")
    print(f"[chaski-5050-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
