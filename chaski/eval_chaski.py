#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Chaski eval — evals none-this-run. No fabricated k/n. No serve pin.

base_model = Qwen/Qwen3.5-0.8B
Hub adapter files exist as of 2026-08-28T17:08Z. That is not an eval.
Do not invent 5/5. Train loss is not an eval.
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
        "adapter": "AVAILABLE",
        "weights": "AVAILABLE",
        "hub_tensors": [
            "adapter_model.safetensors",
            "adapter_config.json",
            "model.safetensors-00001-of-00001.safetensors",
        ],
        "hub_tensors_observed_at": "2026-08-28T17:08Z",
        "publication_eligible": False,
        "claim_boundary": (
            "No JSON/refusal gate ran. Do not claim 5/5 or 6/6. "
            "Hub adapter files exist as of 2026-08-28T17:08Z. "
            "Evals remain none-this-run. Train loss is not an eval."
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
