#!/usr/bin/env python3
"""Fail-closed Waman SKIP receipt. Not a trainer. Does not load frames."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "skip_receipt.json"


def main() -> int:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["status"] == "SKIP-NO-ADMITTED-FRAMES"
    assert payload["xl_2xl"] is False
    assert payload["train_loop"] is False
    assert payload["effector"] == "SIMULATED"
    print("[waman] SKIP-NO-ADMITTED-FRAMES jobs=UNKNOWN effector=SIMULATED")
    print(f"[waman] alias={payload['alias']} (not a second kit)")
    print(f"[waman] {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
