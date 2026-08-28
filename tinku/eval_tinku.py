#!/usr/bin/env python3
"""Tinku eval. UNKNOWN until a MEASURED reranker exists. Same freeze split as Chakana."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER = HERE / "tinku-reranker"


def main() -> int:
    present = RERANKER.is_dir() and any(RERANKER.rglob("*.safetensors"))
    report = {
        "kind": "szl-tinku-eval-report",
        "base_model": BASE_MODEL,
        "evals": "UNKNOWN" if not present else "ROADMAP",
        "ndcg10": None,
        "mrr": None,
        "jobs": "UNKNOWN",
        "sibling": "SZLHOLDINGS/chakana",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[tinku-eval] evals={report['evals']} base_model={BASE_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
