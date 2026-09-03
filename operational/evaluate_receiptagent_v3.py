#!/usr/bin/env python3
"""ReceiptAgent v3 sealed-test runner.

ReceiptAgent v3 (`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3`) has a
MEASURED dev result (11/12) but its sealed test set has never been opened.
This runner enforces the seal:

  - default: reports seal state, exits 0, touches nothing
  - --unseal: opens `receiptagent/sealed_test.jsonl` (or --test path),
    records the unseal event in the receipt, and scores a candidate model
    endpoint against it

Opening the sealed test is a one-way governance event: once a model has
seen test results, later claims about "unseen test" are void for that
candidate. The receipt therefore records unsealed_at and the candidate id
the unseal was bound to.

Scoring expects an OpenAI-compatible local endpoint (e.g. Ollama or the
lane's serve script). No endpoint -> UNKNOWN. No test file -> UNKNOWN.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
DEFAULT_TEST = ROOT / "receiptagent" / "sealed_test.jsonl"
RECEIPT = OUT / "receiptagent-v3-eval-receipt.json"


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
    parser.add_argument("--unseal", action="store_true", help="open the sealed test set (one-way)")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="szl-receiptagent-qwen35-0.8b-v3")
    args = parser.parse_args()

    receipt = {
        "kind": "szl-ops-receiptagent-v3-eval",
        "candidate": args.model,
        "dev_result_prior": "MEASURED 11/12 (see receiptagent/ lane receipts)",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "publication_eligible": False,
    }

    if not args.unseal:
        receipt.update({"status": "SEALED", "sealed_test": str(args.test),
                        "seal_intact": args.test.is_file()})
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        print("[eval-receiptagent] seal intact; pass --unseal to open (one-way)")
        return 0

    receipt["unsealed_at"] = datetime.now(timezone.utc).isoformat()
    receipt["unseal_bound_to"] = args.model

    if not args.test.is_file():
        receipt.update({"status": "UNKNOWN", "reason": "sealed test file absent"})
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        print("[eval-receiptagent] UNKNOWN (no sealed test file)")
        return 0

    rows = [json.loads(l) for l in args.test.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = []
    try:
        for row in rows:
            output = generate(args.endpoint, args.model, row["prompt"])
            ok = all(token in output for token in row.get("must_contain", [])) and not any(
                token in output for token in row.get("must_not_contain", []))
            results.append({"id": row.get("id", "unknown"), "pass": bool(ok)})
    except Exception as exc:  # endpoint down, model missing, etc.
        receipt.update({"status": "UNKNOWN", "reason": f"endpoint unavailable: {exc}"})
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[eval-receiptagent] UNKNOWN ({exc})")
        return 0

    passed = sum(1 for r in results if r["pass"])
    receipt.update({
        "status": "MEASURED",
        "test_pass": passed,
        "test_total": len(results),
        "results": results,
    })
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
