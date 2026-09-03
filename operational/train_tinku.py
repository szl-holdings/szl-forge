#!/usr/bin/env python3
"""Tinku smoke reranker trainer (operational layer).

The canonical lane `tinku/train_tinku.py` activates only when reviewed
triples land at `tinku/admitted_triples.jsonl`. This smoke trainer runs the
same architecture (CrossEncoder on Qwen/Qwen3-Reranker-0.6B +
BinaryCrossEntropyLoss) against the operational candidates at
`operational/out/tinku-admitted_triples.jsonl` without touching the lane's
admitted path. Output goes to `operational/out/tinku-reranker-smoke/` and
is publication-blocked by construction.

Promotion path: human review moves the triples to `tinku/admitted_triples.jsonl`;
then the lane script is the trainer of record.

Rows: {"sentence1": query, "sentence2": passage, "label": 0|1, "split": ...}.
Only split == "train" rows are used; validation/test stay sealed for
evaluate_reranker.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TRIPLES = OUT / "tinku-admitted_triples.jsonl"
MODEL_DIR = OUT / "tinku-reranker-smoke"
RECEIPT = OUT / "tinku-train-receipt.json"
BASE_MODEL = "Qwen/Qwen3-Reranker-0.6B"
SEED = 11


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not TRIPLES.is_file():
        raise SystemExit(f"[ops-tinku] {TRIPLES} missing; run build_curated_datasets.py first")

    rows = [json.loads(l) for l in TRIPLES.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_rows = [
        {"sentence1": r["sentence1"], "sentence2": r["sentence2"], "label": r["label"]}
        for r in rows if r.get("split", "train") == "train"
    ]
    if len(train_rows) < 4:
        raise SystemExit(f"[ops-tinku] only {len(train_rows)} train rows; refusing")

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = CrossEncoder(BASE_MODEL)
    targs = CrossEncoderTrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        seed=SEED,
        report_to=[],
        save_strategy="no",
    )
    trainer = CrossEncoderTrainer(
        model=model, args=targs,
        train_dataset=Dataset.from_list(train_rows),
        loss=BinaryCrossEntropyLoss(model),
    )
    result = trainer.train()
    model.save(str(MODEL_DIR))

    receipt = {
        "kind": "szl-ops-tinku-train",
        "base_model": BASE_MODEL,
        "train_rows": len(train_rows),
        "final_train_loss": getattr(result, "training_loss", "UNKNOWN"),
        "weights": str(MODEL_DIR),
        "status": "SMOKE",
        "evals": "UNKNOWN",
        "publication_eligible": False,
        "promotion": "review -> tinku/admitted_triples.jsonl -> lane trainer",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[ops-tinku] wrote {RECEIPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
