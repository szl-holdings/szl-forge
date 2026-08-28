#!/usr/bin/env python3
"""Fail-closed Qantu SKIP receipt. Not a trainer. Does not load pixels."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "skip_receipt.json"
BASE_MODEL = "google/gemma-4-E4B-it"


def main() -> int:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["status"] == "SKIP-NO-ADMITTED-IMAGES"
    assert payload["base_model"] == BASE_MODEL
    assert payload["train_loop"] is False
    print(f"[qantu] SKIP-NO-ADMITTED-IMAGES base_model={BASE_MODEL} jobs=UNKNOWN")
    print(f"[qantu] {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
