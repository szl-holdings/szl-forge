#!/usr/bin/env python3
"""Chakana eval. UNKNOWN until a MEASURED encoder exists. No pasted MTEB."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ENCODER = HERE / "chakana-encoder"


def main() -> int:
    present = ENCODER.is_dir() and any(ENCODER.rglob("*.safetensors"))
    report = {
        "kind": "szl-chakana-eval-report",
        "base_model": BASE_MODEL,
        "evals": "UNKNOWN" if not present else "ROADMAP",
        "ndcg10": None,
        "mteb_pasted": False,
        "jobs": "UNKNOWN",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[chakana-eval] evals={report['evals']} base_model={BASE_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
