#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=3.0.0", "transformers"]
# ///
"""Tinku CrossEncoder train. Same freeze split as Chakana. Jobs UNKNOWN. No job fire.

base_model = Qwen/Qwen3-Reranker-0.6B
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Reranker-0.6B"
ADMITTED = HERE / "admitted_triples.jsonl"
RECEIPT = HERE / "training_receipt.json"


def write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[tinku] wrote {RECEIPT}")


def skip_receipt(reason: str) -> dict[str, Any]:
    return {
        "kind": "szl-tinku-training-receipt",
        "artifact": "SZLHOLDINGS/tinku",
        "base_model": BASE_MODEL,
        "status": "SKIP-NO-ADMITTED-TRACES",
        "jobs": "UNKNOWN",
        "weights": "UNAVAILABLE",
        "evals": "UNKNOWN",
        "publication_eligible": False,
        "sibling": "SZLHOLDINGS/chakana",
        "reason": reason,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def train_reranker(rows: list[dict[str, Any]]) -> None:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from datasets import Dataset

    model = CrossEncoder(BASE_MODEL)
    train_ds = Dataset.from_list(rows)
    args = None
    try:
        from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments

        args = CrossEncoderTrainingArguments(
            output_dir=str(HERE / "tinku-reranker"),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            report_to=[],
        )
    except ImportError:
        args = None
    loss = BinaryCrossEntropyLoss(model)
    trainer = CrossEncoderTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss
    )
    trainer.train()
    model.save(str(HERE / "tinku-reranker"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(f"[tinku] base_model={BASE_MODEL} jobs=UNKNOWN")
    if not ADMITTED.is_file():
        write_receipt(skip_receipt(f"{ADMITTED.name} absent"))
        print("[tinku] SKIP-NO-ADMITTED-TRACES (no train loop)")
        return 0
    rows = [
        json.loads(line)
        for line in ADMITTED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        write_receipt(skip_receipt("admitted_triples.jsonl empty"))
        return 0
    train_reranker(rows)
    write_receipt(
        {
            "kind": "szl-tinku-training-receipt",
            "base_model": BASE_MODEL,
            "status": "ROADMAP",
            "jobs": "UNKNOWN",
            "publication_eligible": False,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
