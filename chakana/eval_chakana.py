#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=3.0.0",
#     "numpy",
# ]
# ///
"""Chakana eval. Frozen in-house nDCG@10. UNKNOWN until a held-out run.

Do not paste MTEB numbers. Hub files are not an eval. Train loss is not
an eval. Jobs UNKNOWN.

This is SZLHOLDINGS/chakana, not a11oy CHAKANA wiring / tinkuy.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HUB = "SZLHOLDINGS/chakana"
ENCODER = HERE / "chakana-encoder"
HELD_OUT = HERE / "held_out.jsonl"
NDCG_K = 10


def dcg_at_k(relevances: list[float], k: int = NDCG_K) -> float:
    total = 0.0
    for index, rel in enumerate(relevances[:k]):
        total += (math.pow(2.0, rel) - 1.0) / math.log2(index + 2.0)
    return total


def ndcg_at_k(
    ranked_relevances: list[float],
    ideal_relevances: list[float],
    k: int = NDCG_K,
) -> float:
    ideal = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal


def encoder_present() -> bool:
    return ENCODER.is_dir() and any(ENCODER.rglob("*.safetensors"))


def load_held_out(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = row.get("query") or row.get("anchor")
        positives = row.get("positives") or row.get("positive")
        corpus = row.get("corpus")
        if isinstance(positives, str):
            positives = [positives]
        if not isinstance(query, str) or not query.strip():
            raise SystemExit(f"[chakana-eval] {path.name} row missing query")
        if not isinstance(positives, list) or not positives:
            raise SystemExit(f"[chakana-eval] {path.name} row missing positives")
        if not isinstance(corpus, list) or not corpus:
            raise SystemExit(f"[chakana-eval] {path.name} row missing corpus")
        rows.append(
            {
                "query": query.strip(),
                "positives": [str(item) for item in positives],
                "corpus": [str(item) for item in corpus],
            }
        )
    return rows


def cosine_scores(query_vec: list[float], corpus_vecs: list[list[float]]) -> list[float]:
    def norm(vec: list[float]) -> float:
        return math.sqrt(sum(value * value for value in vec)) or 1.0

    qn = norm(query_vec)
    scores: list[float] = []
    for vec in corpus_vecs:
        dot = sum(a * b for a, b in zip(query_vec, vec))
        scores.append(dot / (qn * norm(vec)))
    return scores


def measure_ndcg10(rows: list[dict[str, Any]]) -> str:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(ENCODER))
    scores: list[float] = []
    for row in rows:
        query_vec = model.encode(row["query"], convert_to_numpy=True).tolist()
        corpus_vecs = model.encode(row["corpus"], convert_to_numpy=True).tolist()
        ranked = sorted(
            zip(cosine_scores(query_vec, corpus_vecs), row["corpus"]),
            key=lambda item: item[0],
            reverse=True,
        )
        relevances = [
            1.0 if document in row["positives"] else 0.0 for _, document in ranked
        ]
        ideal = [1.0] * len(row["positives"]) + [0.0] * max(
            0, len(row["corpus"]) - len(row["positives"])
        )
        scores.append(ndcg_at_k(relevances, ideal, NDCG_K))
    mean = sum(scores) / len(scores)
    return f"{mean:.6f}"


def report_payload(
    *,
    evals: str,
    ndcg10: str | None,
    held_out_rows: int | None,
    encoder: str,
) -> dict[str, Any]:
    return {
        "kind": "szl-chakana-eval-report",
        "artifact": HUB,
        "owner": "Stephen Lutar",
        "lane": "NINA (FORGE-class)",
        "base_model": BASE_MODEL,
        "card_status": "ROADMAP",
        "jobs": "UNKNOWN",
        "evals": evals,
        "ndcg10": ndcg10,
        "ndcg_k": NDCG_K,
        "held_out_rows": held_out_rows,
        "encoder": encoder,
        "mteb_pasted": False,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "hub_put": False,
        "not_a11oy_chakana_wiring": True,
        "claim_boundary": (
            "Frozen in-house nDCG@10 on a held-out SZL pair slice. "
            "If the encoder or held-out file is absent, status stays UNKNOWN. "
            "No MTEB paste. Train loss is not an eval."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Score held-out nDCG@10 if a local encoder and slice exist.",
    )
    parser.add_argument(
        "--held-out",
        type=Path,
        default=HELD_OUT,
        help="Held-out SZL pair slice (jsonl). Absent ⇒ UNKNOWN.",
    )
    args = parser.parse_args()
    present = encoder_present()
    encoder_state = "LOCAL" if present else "UNAVAILABLE"
    ndcg10: str | None = None
    evals = "UNKNOWN"
    held_out_rows: int | None = None
    if args.run:
        if not present:
            print("[chakana-eval] --run without local encoder; nDCG@10 stays UNKNOWN")
        elif not args.held_out.is_file():
            print(
                f"[chakana-eval] --run without {args.held_out.name}; "
                "nDCG@10 stays UNKNOWN"
            )
        else:
            rows = load_held_out(args.held_out)
            held_out_rows = len(rows)
            ndcg10 = measure_ndcg10(rows)
            evals = "MEASURED"
    report = report_payload(
        evals=evals,
        ndcg10=ndcg10,
        held_out_rows=held_out_rows,
        encoder=encoder_state,
    )
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[chakana-eval] evals={evals} ndcg10={ndcg10 or 'UNKNOWN'} "
        f"base_model={BASE_MODEL} jobs=UNKNOWN"
    )
    print("[chakana-eval] publication_eligible=false mteb_pasted=false")
    print(f"[chakana-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
