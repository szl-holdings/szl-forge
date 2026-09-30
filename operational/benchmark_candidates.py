#!/usr/bin/env python3
"""Base-vs-candidate comparison scanner.

Promotion rule: no candidate is promoted without a measured comparison
against its own unmodified base on the lane's held-out data. This scanner
collects the measured artifacts that exist in each canonical lane directory
(training receipts, eval receipts, gate receipts) and emits one comparison
receipt. Where a measured artifact is absent, that side is UNKNOWN — the
scanner never interpolates, never inherits, never fabricates.

Usage:
  python operational/benchmark_candidates.py            # scan canonical lanes
  python operational/benchmark_candidates.py --lane chakana

Doctrine v11 LOCKED. Read-only. No Hub PUT. Λ = Conjecture 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"

LANES = {
    "chakana": {"dir": "chakana", "base": "Qwen/Qwen3-Embedding-0.6B", "hub": "SZLHOLDINGS/chakana",
                "receipts": ["training_receipt.json", "training_receipt.status.json", "eval_receipt.json", "eval_receipt.stub.json"]},
    "tinku": {"dir": "tinku", "base": "Qwen/Qwen3-Reranker-0.6B", "hub": "SZLHOLDINGS/tinku",
              "receipts": ["training_receipt.json"]},
    "chaski": {"dir": "chaski", "base": "Qwen/Qwen3.5-0.8B", "hub": "SZLHOLDINGS/chaski",
               "receipts": ["bakeoff_named_n.receipt.json"]},
    "chaski_r2": {"dir": "chaski_r2", "base": "Qwen/Qwen3.5-0.8B", "hub": "SZLHOLDINGS/chaski-r2",
                  "receipts": ["training_receipt.json", "eval_receipt.json"]},
    "khipu": {"dir": "khipu", "base": "Qwen/Qwen2.5-1.5B-Instruct", "hub": "SZLHOLDINGS/SZL-Khipu-1.5B",
              "receipts": ["training_receipt.signed.json", "eval_receipt.signed.json"]},
    "receiptagent": {"dir": "receiptagent", "base": "Qwen/Qwen2.5-1.5B-Instruct", "hub": "SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent",
                     "receipts": ["training_receipt.signed.json", "eval_receipt.signed.json"]},
}


def sha256_or_none(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def scan_lane(name: str, spec: dict) -> dict:
    lane_dir = ROOT / spec["dir"]
    found = {}
    for rel in spec["receipts"]:
        p = lane_dir / rel
        if not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            payload = {"unparsed": True}
        found[rel] = {"sha256": sha256_or_none(p),
                      "status": payload.get("status", "UNDECLARED"),
                      "publication_eligible": payload.get("publication_eligible", False)}
    measured = any(
        v.get("status") not in (None, "UNDECLARED", "UNKNOWN") and "stub" not in k
        for k, v in found.items()
    )
    return {
        "lane": name,
        "base_model": spec["base"],
        "candidate_hub": spec["hub"],
        "base_side": {"state": "REFERENCE",
                      "note": "unmodified base; scored by the lane held-out harness when fired"},
        "candidate_side": {"state": "MEASURED" if measured else "UNKNOWN", "receipts": found},
        "promotion": ("REVIEW — measured artifacts present; lane gate decides" if measured
                      else "BLOCKED — no measured base-vs-candidate delta in this checkout"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", default=None, choices=sorted(LANES), help="Scan one lane only.")
    args = ap.parse_args()
    names = [args.lane] if args.lane else sorted(LANES)
    report = {
        "kind": "szl-base-vs-candidate-benchmark/v1",
        "source_repo": "szl-holdings/szl-forge",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "lanes": [scan_lane(n, LANES[n]) for n in names],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "benchmark-candidates.receipt.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {l["lane"]: l["promotion"].split(" — ")[0] for l in report["lanes"]}
    print(json.dumps({"receipt": str(path), "promotion": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
