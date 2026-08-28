#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=3.0.0",
#     "datasets",
#     "transformers",
#     "huggingface_hub",
#     "accelerate",
#     "trackio",
# ]
# ///
"""Chakana bridge-embedding trainer. sentence-transformers. Jobs UNKNOWN.

This is SZLHOLDINGS/chakana — query→vector for szl-lake / doctrine /
killinchu text retrieval. It is NOT a11oy CHAKANA wiring / tinkuy
(Andean-cross topology under a11oy organs). MiniEmbed in szl-kernels is
a different artifact. Do not treat those as this encoder.

Owner: Stephen Lutar. Lane: NINA (FORGE-class).
base_model MUST be Qwen/Qwen3-Embedding-0.6B (Apache-2.0) or BAAI/bge-m3
(MIT). Fail closed on KaLM / EmbeddingGemma. Fashion rule: silhouette
from Qwen3-Embedding; cut is original SZL pairs. Never republish Qwen
tensors. Never paste MTEB numbers.

Default is status (no GPU, no job fire, no Hub write). Pass --train to
run locally or on HF Jobs (`hf jobs uv run` + HF_TOKEN + Trackio).
Hub PUT is refused. publication_eligible stays false until a MEASURED
held-out nDCG@10 exists. Doctrine v11 LOCKED. Λ = Conjecture 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

HUB = "SZLHOLDINGS/chakana"
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ALT_BASE_MODEL = "BAAI/bge-m3"
ADMITTED_BASES = frozenset({BASE_MODEL, ALT_BASE_MODEL})
DEFAULT_PAIRS_DATASET = "SZLHOLDINGS/chakana-pairs"
SEED = 11
MATRYOSHKA_DIMS = (1024, 512, 256)
NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 512
ENCODER_DIR = HERE / "chakana-encoder"
STATUS_RECEIPT = HERE / "training_receipt.status.json"
TRAIN_RECEIPT = HERE / "training_receipt.json"
FORBIDDEN_ESTATE = "SZL_ESTATE_MANAGED.json"
NOT_PAIRS_DATASETS = frozenset(
    {
        "SZLHOLDINGS/szl-lake",
        "SZLHOLDINGS/rag-corpus-v1",
    }
)
THIRD_PARTY_DATASET_MARKERS = (
    "mteb",
    "beir",
    "msmarco",
    "hotpotqa",
    "arguana",
    "scifact",
    "fiqa",
    "dbpedia",
    "nq-train",
    "natural-questions",
    "trec-covid",
    "climate-fever",
    "cqadupstack",
    "webis-touche",
)
FORBIDDEN_BASE_MARKERS = (
    "kalm",
    "hit-tmg",
    "embeddinggemma",
    "google/embeddinggemma",
)
QUERY_INSTRUCTION = (
    "Instruct: Retrieve SZL doctrine, lake, or killinchu passages "
    "that answer the query\nQuery: "
)

# a11oy CHAKANA wiring / tinkuy is Andean-cross topology, not this model.


def refuse_forbidden_base(name: str) -> None:
    lowered = name.strip()
    if lowered not in ADMITTED_BASES:
        raise SystemExit(
            f"[chakana] refuse base {name!r}: admitted bases are "
            f"{BASE_MODEL} (Apache-2.0) or {ALT_BASE_MODEL} (MIT). "
            "Never KaLM. Never EmbeddingGemma."
        )
    blob = lowered.lower()
    for token in FORBIDDEN_BASE_MARKERS:
        if token in blob:
            raise SystemExit(
                f"[chakana] refuse base {name}: never KaLM / EmbeddingGemma"
            )


def refuse_third_party_dataset(dataset_id: str, *, path: Path | None = None) -> None:
    haystacks = [dataset_id.lower()]
    if path is not None:
        haystacks.append(path.name.lower())
        haystacks.append(str(path).lower())
    blob = " ".join(haystacks)
    if FORBIDDEN_ESTATE.lower() in blob:
        raise SystemExit(
            f"[chakana] refuse: {FORBIDDEN_ESTATE} is not admitted pairs"
        )
    for marker in THIRD_PARTY_DATASET_MARKERS:
        if marker in blob:
            raise SystemExit(
                f"[chakana] refuse third-party retrieval bench {dataset_id!r} "
                f"(marker {marker!r}). MTEB/BEIR is never SZL pairs."
            )
    normalized = dataset_id.strip().rstrip("/")
    if normalized in NOT_PAIRS_DATASETS:
        raise SystemExit(
            f"[chakana] refuse {normalized}: lake/corpus is not query-positive "
            "pairs. Default admitted id is SZLHOLDINGS/chakana-pairs."
        )
    if "/" in normalized and not normalized.upper().startswith("SZLHOLDINGS/"):
        raise SystemExit(
            f"[chakana] refuse third-party dataset {dataset_id!r}. "
            "Only SZLHOLDINGS/* or a local jsonl of admitted SZL pairs."
        )


def pair_row(raw: dict[str, Any]) -> dict[str, str] | None:
    anchor = raw.get("anchor") or raw.get("query") or raw.get("question")
    positive = (
        raw.get("positive")
        or raw.get("passage")
        or raw.get("document")
        or raw.get("text")
    )
    if not isinstance(anchor, str) or not isinstance(positive, str):
        return None
    anchor = anchor.strip()
    positive = positive.strip()
    if not anchor or not positive:
        return None
    return {"anchor": anchor, "positive": positive}


def load_jsonl_pairs(path: Path) -> list[dict[str, str]]:
    refuse_third_party_dataset(path.name, path=path)
    if path.name == FORBIDDEN_ESTATE:
        raise SystemExit(f"[chakana] refuse: will not ingest {FORBIDDEN_ESTATE}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = pair_row(json.loads(line))
        if parsed is None:
            raise SystemExit(
                f"[chakana] {path.name} row is not a query-positive pair"
            )
        rows.append(parsed)
    return rows


def load_hub_pairs(dataset_id: str) -> list[dict[str, str]]:
    refuse_third_party_dataset(dataset_id)
    from datasets import load_dataset

    not_found: tuple[type[BaseException], ...]
    try:
        from huggingface_hub.errors import RepositoryNotFoundError

        not_found = (RepositoryNotFoundError, FileNotFoundError)
    except ImportError:
        try:
            from huggingface_hub.utils import RepositoryNotFoundError as _HubMissing

            not_found = (_HubMissing, FileNotFoundError)
        except ImportError:
            not_found = (FileNotFoundError,)

    try:
        loaded = load_dataset(dataset_id, split="train")
    except not_found:
        return []
    except Exception as exc:  # noqa: BLE001 - honest skip vs refuse
        message = str(exc).lower()
        if "not found" in message or "404" in message:
            return []
        raise
    rows: list[dict[str, str]] = []
    for raw in loaded:
        parsed = pair_row(dict(raw))
        if parsed is None:
            raise SystemExit(
                f"[chakana] {dataset_id} row is not a query-positive pair"
            )
        rows.append(parsed)
    return rows


def trackio_report_to() -> str:
    """Jobs-first: Trackio when HF_TOKEN is present. Else none."""
    if os.environ.get("REPORT_TO", "").strip().lower() == "none":
        return "none"
    if os.environ.get("HF_TOKEN"):
        return "trackio"
    return "none"


def status_receipt(
    *,
    hub: str,
    base_model: str,
    dataset_id: str,
    live: bool = False,
    training_loss: str | None = None,
    pair_count: int | None = None,
    weights: str = "UNAVAILABLE",
    skip_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "szl-chakana-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "v": 1,
        "artifact": hub,
        "owner": "Stephen Lutar",
        "lane": "NINA (FORGE-class)",
        "base_model": base_model,
        "alt_base_model": ALT_BASE_MODEL,
        "base_model_relation": "finetune" if live else "intended-cut",
        "library_name": "sentence-transformers",
        "pipeline_tag": "feature-extraction",
        "dataset": dataset_id,
        "pair_count": pair_count,
        "seed": SEED,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "learning_rate": str(LEARNING_RATE),
        "max_seq_length": MAX_SEQ_LEN,
        "matryoshka_dims": list(MATRYOSHKA_DIMS),
        "loss": "MatryoshkaLoss(MultipleNegativesRankingLoss)",
        "trackio": trackio_report_to() == "trackio",
        "report_to": trackio_report_to(),
        "push_to_hub": False,
        "hub_put": False,
        "jobs": "UNKNOWN",
        "weights": weights,
        "evals": "UNKNOWN",
        "ndcg10": "UNKNOWN",
        "mteb_pasted": False,
        "label": "ROADMAP",
        "lambda": "Conjecture 1",
        "doctrine": "v11 LOCKED 749/14/163",
        "evidence_ceiling": "0.97",
        "proposal_only": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "serve_pin": False,
        "not_a11oy_chakana_wiring": True,
        "not_miniembed": True,
        "not_qwen_rehost": True,
        "skip_reason": skip_reason,
        "claim_boundary": (
            "Hub SZLHOLDINGS/chakana is a stub until admitted SZL pairs train "
            "and a held-out nDCG@10 is MEASURED. Jobs UNKNOWN. Do not paste "
            "MTEB numbers. Do not invent a job id. publication_eligible false."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "local-train" if live and not skip_reason else "forge-status",
        "signed": False,
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chakana] wrote {path}")


def status_main(hub: str, base_model: str, dataset_id: str) -> int:
    refuse_forbidden_base(base_model)
    refuse_third_party_dataset(dataset_id)
    report_to = trackio_report_to()
    print(
        f"[chakana] base_model={base_model} hub={hub} seed={SEED} "
        f"jobs=UNKNOWN"
    )
    print(
        f"[chakana] card=ROADMAP dataset={dataset_id} "
        f"report_to={report_to} publication_eligible=false"
    )
    print("[chakana] nDCG@10=UNKNOWN (held-out not run)")
    print("[chakana] not a11oy CHAKANA wiring / tinkuy; not MiniEmbed")
    write_receipt(
        status_receipt(hub=hub, base_model=base_model, dataset_id=dataset_id),
        STATUS_RECEIPT,
    )
    return 0


def train_embedder(
    pairs: list[dict[str, str]],
    base_model: str,
    report_to: str,
) -> str:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.losses import (
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
    )
    from sentence_transformers.training_args import BatchSamplers

    refuse_forbidden_base(base_model)
    prepared = [
        {
            "anchor": QUERY_INSTRUCTION + row["anchor"],
            "positive": row["positive"],
        }
        for row in pairs
    ]
    model = SentenceTransformer(base_model)
    train_ds = Dataset.from_list(prepared)
    inner = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(
        model, inner, matryoshka_dims=list(MATRYOSHKA_DIMS)
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(HERE / "outputs"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        max_seq_length=MAX_SEQ_LEN,
        seed=SEED,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to=report_to if report_to != "none" else [],
        push_to_hub=False,
        save_strategy="epoch",
        logging_steps=1,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss
    )
    stats = trainer.train()
    loss_value = float(getattr(stats, "training_loss", float("nan")))
    final_loss = f"{loss_value:.4f}" if loss_value == loss_value else "UNKNOWN"
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(ENCODER_DIR))
    print(f"[chakana] local encoder {ENCODER_DIR}")
    print("[chakana] Hub PUT skipped (this checkout never uploads)")
    return final_loss


def train_main(
    hub: str,
    base_model: str,
    dataset_id: str,
    dataset_file: Path | None,
) -> int:
    refuse_forbidden_base(base_model)
    refuse_third_party_dataset(dataset_id, path=dataset_file)
    report_to = trackio_report_to()
    print(
        f"[chakana] train base_model={base_model} hub={hub} seed={SEED} "
        f"report_to={report_to}"
    )
    print("[chakana] push_to_hub=false; refusing Hub PUT")
    if dataset_file is not None:
        pairs = load_jsonl_pairs(dataset_file)
        source = str(dataset_file)
    else:
        pairs = load_hub_pairs(dataset_id)
        source = dataset_id
    if not pairs:
        reason = f"no admitted query-positive pairs at {source}"
        write_receipt(
            status_receipt(
                hub=hub,
                base_model=base_model,
                dataset_id=dataset_id,
                live=True,
                skip_reason=reason,
            ),
            TRAIN_RECEIPT,
        )
        print(f"[chakana] SKIP-NO-ADMITTED-PAIRS ({reason})")
        return 0
    loss = train_embedder(pairs, base_model, report_to)
    receipt = status_receipt(
        hub=hub,
        base_model=base_model,
        dataset_id=dataset_id,
        live=True,
        training_loss=loss,
        pair_count=len(pairs),
        weights="LOCAL",
    )
    receipt["finalTrainLoss"] = loss
    receipt["evals"] = "UNKNOWN"
    receipt["publication_eligible"] = False
    write_receipt(receipt, TRAIN_RECEIPT)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run sentence-transformers locally or on Jobs. Default is status.",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", BASE_MODEL),
        help="Admitted: Qwen/Qwen3-Embedding-0.6B or BAAI/bge-m3.",
    )
    parser.add_argument(
        "--hub",
        default=os.environ.get("HUB_MODEL_ID", HUB),
        help="Target Hub id. Default SZLHOLDINGS/chakana.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("CHAKANA_PAIRS_DATASET", DEFAULT_PAIRS_DATASET),
        help="Admitted SZL pairs dataset id. Default SZLHOLDINGS/chakana-pairs.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Local jsonl of query-positive pairs. Refuses MTEB/BEIR names.",
    )
    args = parser.parse_args()
    hub = args.hub.strip() or HUB
    if not re.fullmatch(r"SZLHOLDINGS/chakana", hub):
        raise SystemExit(
            f"[chakana] refuse hub {hub!r}: this kit targets {HUB} only"
        )
    refuse_forbidden_base(args.base_model)
    refuse_third_party_dataset(args.dataset, path=args.dataset_file)
    if args.train:
        return train_main(hub, args.base_model, args.dataset, args.dataset_file)
    return status_main(hub, args.base_model, args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
