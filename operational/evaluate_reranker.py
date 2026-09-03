#!/usr/bin/env python3
"""Base-vs-candidate reranker evaluation (MAP, MRR@10, nDCG@10).

Uses the sealed validation/test splits of the operational triples file:
each positive row becomes a ranking task of {positive} ∪ {that query's
negatives}. Compares Qwen/Qwen3-Reranker-0.6B against a candidate
CrossEncoder directory. Writes `operational/out/reranker-eval-receipt.json`.

Fail-closed: missing triples file or missing candidate -> UNKNOWN, exit 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPLES = OUT / "tinku-admitted_triples.jsonl"
BASE_MODEL = "Qwen/Qwen3-Reranker-0.6B"
CANDIDATE = OUT / "tinku-reranker-smoke"


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def build_tasks(rows: list[dict], split: str) -> list[dict]:
    by_query: dict[str, dict] = {}
    for r in rows:
        if r.get("split", "train") != split:
            continue
        task = by_query.setdefault(r["sentence1"], {"query": r["sentence1"], "docs": [], "labels": []})
        task["docs"].append(r["sentence2"])
        task["labels"].append(int(r["label"]))
    return [t for t in by_query.values() if any(t["labels"]) and len(t["docs"]) >= 2]


def metrics(score_fn, tasks: list[dict]) -> dict:
    ap = mrr = ndcg = 0.0
    for task in tasks:
        scores = score_fn(task["query"], task["docs"])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        gains = [task["labels"][i] for i in order]
        mrr += next((1.0 / (i + 1) for i, g in enumerate(gains) if g), 0.0)
        hits = 0
        precisions = []
        for i, g in enumerate(gains):
            if g:
                hits += 1
                precisions.append(hits / (i + 1))
        ap += sum(precisions) / max(sum(task["labels"]), 1)
        ideal = sorted(task["labels"], reverse=True)[:10]
        ndcg += dcg(gains) / max(dcg(ideal), 1e-9)
    n = max(len(tasks), 1)
    return {"map": round(ap / n, 4), "mrr_at_10": round(mrr / n, 4),
            "ndcg_at_10": round(ndcg / n, 4), "tasks": len(tasks)}


def cross_encoder_scorer(model_id_or_path: str):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_id_or_path)
    return lambda query, docs: model.predict([(query, d) for d in docs]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", type=Path, default=TRIPLES)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    args = parser.parse_args()

    receipt = {
        "kind": "szl-ops-reranker-eval",
        "split": args.split,
        "base_model": args.base,
        "candidate": str(args.candidate),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "publication_eligible": False,
    }
    if not args.triples.is_file():
        receipt.update({"status": "UNKNOWN", "reason": "triples file absent"})
        (OUT / "reranker-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("[eval-reranker] UNKNOWN (no triples)")
        return 0

    rows = [json.loads(l) for l in args.triples.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = build_tasks(rows, args.split)
    if not tasks:
        receipt.update({"status": "UNKNOWN", "reason": f"no {args.split} tasks"})
        (OUT / "reranker-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[eval-reranker] UNKNOWN (no {args.split} tasks)")
        return 0

    try:
        base_scorer = cross_encoder_scorer(args.base)
    except ImportError as exc:
        receipt.update({"status": "UNKNOWN", "reason": f"dependency missing: {exc}"})
        (OUT / "reranker-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[eval-reranker] UNKNOWN ({exc})")
        return 0
    receipt["base"] = metrics(base_scorer, tasks)
    if args.candidate.is_dir():
        receipt["candidate_metrics"] = metrics(cross_encoder_scorer(str(args.candidate)), tasks)
        receipt["status"] = "MEASURED"
    else:
        receipt["candidate_metrics"] = "UNKNOWN"
        receipt["status"] = "MEASURED_BASE_ONLY"
    (OUT / "reranker-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
