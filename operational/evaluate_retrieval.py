#!/usr/bin/env python3
"""Base-vs-candidate retrieval evaluation (MRR@10, Recall@10, nDCG@10).

Reads a frozen in-house held-out slice (default `chakana/held_out.jsonl`,
one `szl.chakana-held-out/v1` row per line) and compares the admitted base
model against a candidate encoder directory. Writes
`operational/out/retrieval-eval-receipt.json`.

Fail-closed doctrine:
  - held-out slice absent  -> metrics UNKNOWN, exit 0 (not an error)
  - candidate dir absent    -> candidate metrics UNKNOWN, base still measured
  - no third-party benchmark names are accepted as the slice path

A MEASURED receipt from this script over an in-house slice is the only
path that can ever feed a `publication_eligible: true` decision, and even
then the decision itself stays human.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
DEFAULT_SLICE = ROOT / "chakana" / "held_out.jsonl"
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Instruct: Retrieve SZL doctrine, lake, or killinchu passages "
    "that answer the query\nQuery: "
)
FORBIDDEN = ("mteb", "beir", "msmarco", "hotpotqa", "nq", "fiqa", "scifact")


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def metrics_for(ranker, slice_rows: list[dict]) -> dict:
    mrr = recall = ndcg = 0.0
    for row in slice_rows:
        corpus = row["corpus"]
        positives = set(row["positives"])
        scores = ranker(row["query"], corpus)
        order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:10]
        hits = [1 if corpus[i] in positives else 0 for i in order]
        recall += 1.0 if any(hits) else 0.0
        mrr += next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
        ideal = [1] * len(positives)
        ndcg += dcg(hits) / max(dcg(ideal), 1e-9)
    n = max(len(slice_rows), 1)
    return {"mrr_at_10": round(mrr / n, 4), "recall_at_10": round(recall / n, 4),
            "ndcg_at_10": round(ndcg / n, 4), "queries": len(slice_rows)}


def sentence_transformer_ranker(model_id_or_path: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id_or_path)

    def rank(query: str, corpus: list[str]) -> list[float]:
        q = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)
        c = model.encode(corpus, normalize_embeddings=True)
        return (c @ q[0]).tolist()

    return rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--candidate", type=Path, default=ROOT / "chakana" / "chakana-encoder")
    args = parser.parse_args()

    lowered = str(args.slice).lower()
    if any(tok in lowered for tok in FORBIDDEN):
        raise SystemExit(f"[eval-retrieval] refuse third-party slice: {args.slice}")

    receipt = {
        "kind": "szl-ops-retrieval-eval",
        "slice": str(args.slice),
        "base_model": args.base,
        "candidate": str(args.candidate),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "publication_eligible": False,
    }

    if not args.slice.is_file():
        receipt.update({"status": "UNKNOWN", "reason": "held-out slice absent; evals UNKNOWN"})
        (OUT / "retrieval-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("[eval-retrieval] UNKNOWN (no in-house held-out slice)")
        return 0

    rows = [json.loads(l) for l in args.slice.read_text(encoding="utf-8").splitlines() if l.strip()]
    try:
        base_ranker = sentence_transformer_ranker(args.base)
    except ImportError as exc:
        receipt.update({"status": "UNKNOWN", "reason": f"dependency missing: {exc}"})
        (OUT / "retrieval-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[eval-retrieval] UNKNOWN ({exc})")
        return 0
    receipt["base"] = metrics_for(base_ranker, rows)
    if args.candidate.is_dir():
        receipt["candidate_metrics"] = metrics_for(sentence_transformer_ranker(str(args.candidate)), rows)
        receipt["status"] = "MEASURED"
    else:
        receipt["candidate_metrics"] = "UNKNOWN"
        receipt["status"] = "MEASURED_BASE_ONLY"
    (OUT / "retrieval-eval-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
