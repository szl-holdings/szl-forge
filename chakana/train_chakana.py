#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=3.0.0", "transformers"]
# ///
"""Chakana embedding train. Jobs UNKNOWN. Does not fire a Hub job.

base_model = Qwen/Qwen3-Embedding-0.6B
alt MIT = BAAI/bge-m3
Never KaLM. Never EmbeddingGemma. MiniEmbed in szl-kernels is a different artifact.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ALT_BASE_MODEL = "BAAI/bge-m3"
ADMITTED = HERE / "admitted_pairs.jsonl"
RECEIPT = HERE / "training_receipt.json"
FORBIDDEN_BASES = ("KaLM", "HIT-TMG", "EmbeddingGemma", "google/embeddinggemma")


def refuse_forbidden_base(name: str) -> None:
    lowered = name.lower()
    for token in FORBIDDEN_BASES:
        if token.lower() in lowered:
            raise SystemExit(f"[chakana] refuse base {name}: never KaLM / EmbeddingGemma")


def write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chakana] wrote {RECEIPT}")


def skip_receipt(reason: str) -> dict[str, Any]:
    return {
        "kind": "szl-chakana-training-receipt",
        "artifact": "SZLHOLDINGS/chakana",
        "base_model": BASE_MODEL,
        "alt_base_model": ALT_BASE_MODEL,
        "status": "SKIP-NO-ADMITTED-PAIRS",
        "jobs": "UNKNOWN",
        "weights": "UNAVAILABLE",
        "evals": "UNKNOWN",
        "publication_eligible": False,
        "reason": reason,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def train_embedder(pairs: list[dict[str, str]], base_model: str) -> None:
    refuse_forbidden_base(base_model)
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from datasets import Dataset

    model = SentenceTransformer(base_model)
    train_ds = Dataset.from_list(pairs)
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(HERE / "chakana-encoder"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss
    )
    trainer.train()
    model.save(str(HERE / "chakana-encoder"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()
    refuse_forbidden_base(args.base_model)
    print(f"[chakana] base_model={args.base_model} jobs=UNKNOWN")
    if not ADMITTED.is_file():
        write_receipt(skip_receipt(f"{ADMITTED.name} absent"))
        print("[chakana] SKIP-NO-ADMITTED-PAIRS (no train loop)")
        return 0
    pairs = [
        json.loads(line)
        for line in ADMITTED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pairs:
        write_receipt(skip_receipt("admitted_pairs.jsonl empty"))
        return 0
    train_embedder(pairs, args.base_model)
    write_receipt(
        {
            "kind": "szl-chakana-training-receipt",
            "base_model": args.base_model,
            "status": "ROADMAP",
            "jobs": "UNKNOWN",
            "publication_eligible": False,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
