#!/usr/bin/env python3
"""WILLAY doctrine held-out suite runner.

WILLAY (`SZLHOLDINGS/WILLAY`, two LoRA adapters on Qwen/Qwen2.5-0.5B-Instruct)
ships with no evaluation. This runner executes the doctrine suite at
`operational/suites/willay_doctrine.jsonl` against an OpenAI-compatible
endpoint and writes `operational/out/willay-eval-receipt.json`.

Suite row schema (one JSON object per line):
  {"id": str, "prompt": str,
   "must_contain": [str], "must_not_contain": [str]}

Suite absent or endpoint unreachable -> UNKNOWN receipt, exit 0. No result
is ever inferred; UNKNOWN is the honest state until measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SUITE = HERE / "suites" / "willay_doctrine.jsonl"
RECEIPT = OUT / "willay-eval-receipt.json"


def generate(endpoint: str, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="SZLHOLDINGS/WILLAY")
    args = parser.parse_args()

    receipt = {
        "kind": "szl-ops-willay-eval",
        "candidate": args.model,
        "suite": str(args.suite),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "publication_eligible": False,
    }

    if not args.suite.is_file():
        receipt.update({"status": "UNKNOWN", "reason": "doctrine suite absent; build it before any claim"})
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        print("[eval-willay] UNKNOWN (no doctrine suite)")
        return 0

    rows = [json.loads(l) for l in args.suite.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = []
    try:
        for row in rows:
            output = generate(args.endpoint, args.model, row["prompt"])
            ok = all(t in output for t in row.get("must_contain", [])) and not any(
                t in output for t in row.get("must_not_contain", []))
            results.append({"id": row.get("id", "unknown"), "pass": bool(ok)})
    except Exception as exc:
        receipt.update({"status": "UNKNOWN", "reason": f"endpoint unavailable: {exc}"})
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[eval-willay] UNKNOWN ({exc})")
        return 0

    passed = sum(1 for r in results if r["pass"])
    receipt.update({"status": "MEASURED", "passed": passed, "total": len(results),
                    "results": results})
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
