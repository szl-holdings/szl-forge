#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Forge-owned NeoMME CPU smoke evaluation; never a production qualification.

Hcompany owns NeoMME. This original SZL harness binds artifacts, fixture bytes,
rankings and measured resource observations. It never trains or changes routes.
Imports and --plan need only the standard library. --execute is explicit opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from typing import Any

MODEL = "Hcompany/NeoMME-260M-Retriever"
REVISION = "0dcb6c924435bd0bf5d504dba9ba2bb63acd8595"
WEIGHT_SHA256 = "29bc505e54b7bd1a214f9041081a73b62b0a6b16cee1efc438f777e5bf1a4943"
WEIGHT_BYTES = 526156876
TRANSFORMERS_REVISION = "cdfdcad31314fe4f23b40ab374e860a62403f72a"
BLOB_PINS = {
    "LICENSE": "4f42e107b3dccff5f5f56929fa72aa7eeb182e09",
    "README.md": "a4f325141a978d8fae37e01716e14b63d5be3835",
    "chat_template.jinja": "0690bfa996cefd80c78a9f6f6ecd2d11f22b7287",
    "config.json": "651a5dee1813037cf75ffe2048f16c377508ce94",
    "processor_config.json": "ad99437311be12c3483fe59f049c12ae03d45993",
    "tokenizer.json": "bfcf88cd3e0af127150ada77d1d76714104e36ec",
    "tokenizer_config.json": "64422cc73d0d7effe8e61bf656855300a62f8ee3",
}
# Original synthetic fixtures: deliberately simple smoke cases, NOT held-out claims.
DOCUMENTS = (
    ("battery", "Battery recycling", "Used lithium batteries go to the recycling depot. Never put batteries in household trash. The depot opens on Saturday."),
    ("solar", "Solar inverter", "The rooftop solar inverter converts direct current into alternating current. Inspect ventilation when the inverter overheats."),
    ("water", "Drinking water", "The water treatment plant filters sediment and disinfects drinking water. Operators test chlorine residual every morning."),
    ("library", "Library hours", "The public library opens at nine in the morning and closes at six in the evening. Borrowed books are due in three weeks."),
    ("freight", "Freight invoice", "The freight invoice lists the shipment weight, delivery address and transport charge. Payment is due thirty days after delivery."),
    ("seeds", "Seed bank", "The seed bank preserves crop varieties in cold dry storage. Germination tests check whether stored seeds can still grow."),
)
QUERIES = (
    ("Where should used lithium batteries be taken?", "battery"),
    ("What converts rooftop direct current to alternating current?", "solar"),
    ("Which facility checks chlorine in drinking water?", "water"),
    ("When must borrowed books be returned?", "library"),
    ("Which document lists shipment weight and transport charges?", "freight"),
    ("Where are crop varieties kept cold and dry?", "seeds"),
    ("How can we tell whether stored seeds will grow?", "seeds"),
    ("What should be inspected if a solar inverter overheats?", "solar"),
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def rank(scores: list[float], identifiers: list[str]) -> list[str]:
    if not scores or len(scores) != len(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("invalid ranking shape or duplicate document ID")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
        raise ValueError("non-finite or non-numeric score")
    return [identifiers[i] for i in sorted(range(len(scores)), key=lambda i: (-scores[i], identifiers[i]))]


def metrics(rankings: list[list[str]], expected: list[str]) -> dict[str, float]:
    if not rankings or len(rankings) != len(expected):
        raise ValueError("empty or mismatched evaluation")
    positions = []
    for order, target in zip(rankings, expected):
        if not order or len(set(order)) != len(order) or target not in order:
            raise ValueError("invalid ranking or missing expected document")
        positions.append(order.index(target) + 1)
    n = len(positions)
    return {"top1": sum(p == 1 for p in positions) / n,
            "recall_at_3": sum(p <= 3 for p in positions) / n,
            "mrr": sum(1 / p for p in positions) / n,
            "ndcg_at_3": sum(1 / math.log2(p + 1) if p <= 3 else 0 for p in positions) / n}


def plan() -> dict[str, Any]:
    return {"schema": "szl.forge.neomme-smoke/v1", "status": "NOT_EXECUTED",
            "model": MODEL, "model_revision": REVISION,
            "weight_sha256": WEIGHT_SHA256, "transformers_revision": TRANSFORMERS_REVISION,
            "fixture_sha256": digest({"documents": DOCUMENTS, "queries": QUERIES}),
            "fixture_kind": "ORIGINAL_SYNTHETIC_SMOKE_NOT_INDEPENDENT_HELDOUT",
            "documents": len(DOCUMENTS), "queries": len(QUERIES),
            "modes": ["text", "image"], "scoring": ["dense_cosine", "mean_maxsim"],
            "device": "cpu", "dtype": "float32", "trust_remote_code": False,
            "download_budget_bytes": 650000000, "max_input_tokens": 1536,
            "production_promotion": False, "training_performed": False,
            "routing_changed": False, "private_brain_loaded": False,
            "runtime_qualification": "UNQUALIFIED", "receipt_status": "UNSIGNED_HONEST"}


def verify_snapshot(folder: Path) -> dict[str, str]:
    result = {}
    for name, expected in BLOB_PINS.items():
        path = folder / name
        if path.stat().st_size > 12000000:
            raise ValueError("metadata artifact exceeds size budget")
        raw = path.read_bytes()
        # Git blob IDs bind the exact observed metadata, including loader template.
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected:
            raise ValueError("pinned metadata identity mismatch: " + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    weights = folder / "model.safetensors"
    if weights.stat().st_size != WEIGHT_BYTES or file_digest(weights) != WEIGHT_SHA256:
        raise ValueError("pinned weights identity mismatch")
    config = json.loads((folder / "config.json").read_text())
    if config.get("model_type") != "neomme" or config.get("auto_map"):
        raise ValueError("unexpected architecture or custom loader")
    result["model.safetensors"] = WEIGHT_SHA256
    return result


def execute() -> dict[str, Any]:
    result = plan()
    started = time.perf_counter()
    result["observed_at"] = datetime.now(timezone.utc).isoformat()
    result["harness_sha256"] = file_digest(Path(__file__))
    # The shared production process must never call this experiment in-process.
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import resource
        import torch
        from PIL import Image, ImageDraw, ImageFont
        from huggingface_hub import HfApi, snapshot_download
        from transformers import NeoMMEForRetrieval, NeoMMEProcessor

        direct = json.loads(metadata.distribution("transformers").read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
            raise ValueError("transformers must be installed from the exact qualified-for-smoke source pin")
        info = HfApi(token=False).model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION or info.private or info.gated or info.disabled:
            raise ValueError("source unavailable, gated, or revision mismatch")
        if not info.card_data or info.card_data.get("license") != "apache-2.0":
            raise ValueError("source license posture changed")
        allowed = set(BLOB_PINS) | {"model.safetensors"}
        sizes = {f.rfilename: f.size for f in info.siblings if f.rfilename in allowed}
        if set(sizes) != allowed or any(type(s) is not int or s <= 0 for s in sizes.values()):
            raise ValueError("incomplete artifact sizes")
        if sum(sizes.values()) > result["download_budget_bytes"]:
            raise ValueError("artifact download budget exceeded")
        folder = Path(snapshot_download(MODEL, revision=REVISION, token=False,
                                       allow_patterns=sorted(allowed), max_workers=2))
        result["artifact_sha256"] = verify_snapshot(folder)
        torch.set_num_threads(2)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        processor = NeoMMEProcessor.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
        load_start = time.perf_counter()
        model = NeoMMEForRetrieval.from_pretrained(str(folder), local_files_only=True,
                trust_remote_code=False, use_safetensors=True, dtype=torch.float32).to("cpu").eval()
        result["model_load_seconds"] = time.perf_counter() - load_start
        timings: dict[str, list[float]] = {"query": [], "text": [], "image": []}

        def encode(content: Any, task: str, lane: str) -> tuple[Any, Any]:
            before = time.perf_counter()
            inputs = processor.apply_chat_template([[{"role": "user", "content": content}]],
                task=task, tokenize=True, return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": "longest"})
            if inputs["input_ids"].shape[-1] > result["max_input_tokens"]:
                raise ValueError("input token budget exceeded")
            with torch.inference_mode():
                outputs = model(**inputs)
            mask = inputs["attention_mask"][0].bool()
            tokens = torch.nn.functional.normalize(outputs.embeddings[0][mask].float(), dim=-1)
            dense = torch.nn.functional.normalize(outputs.dense_embeddings[0].float(), dim=-1)
            if tokens.ndim != 2 or tokens.shape[0] == 0 or dense.ndim != 1:
                raise ValueError("invalid embedding shape")
            if not torch.isfinite(tokens).all() or not torch.isfinite(dense).all():
                raise ValueError("non-finite embedding")
            timings[lane].append(time.perf_counter() - before)
            return tokens, dense

        query_vectors = [encode(text, "query", "query") for text, _ in QUERIES]
        identifiers = [row[0] for row in DOCUMENTS]
        cases: dict[str, Any] = {}
        image_hashes = []
        for mode in result["modes"]:
            vectors = []
            for _identifier, title, text in DOCUMENTS:
                if mode == "text":
                    content: Any = title + "\n" + text
                else:
                    image = Image.new("RGB", (384, 512), "white")
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.load_default(size=18)
                    draw.multiline_text((20, 24), title + "\n\n" + textwrap.fill(text, 32),
                                        font=font, fill="black", spacing=8)
                    image_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                    content = [{"type": "image", "image": image}]
                vectors.append(encode(content, "document", mode))
            for scoring in result["scoring"]:
                rankings, score_rows = [], []
                for qt, qd in query_vectors:
                    scores = [float((qt @ dt.T).max(dim=1).values.mean()) if scoring == "mean_maxsim"
                              else float(qd @ dd) for dt, dd in vectors]
                    rankings.append(rank(scores, identifiers))
                    score_rows.append(scores)
                cases[mode + ":" + scoring] = {"metrics": metrics(rankings, [q[1] for q in QUERIES]),
                                              "rankings": rankings, "scores": score_rows}
        result.update({"status": "SMOKE_EXECUTED", "cases": cases,
            "image_rgb_sha256": image_hashes, "encoding_seconds_by_item": timings,
            "max_rss_bytes_process_including_dependencies": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "platform": platform.platform(), "python": platform.python_version(),
            "dependencies": sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions()
                                   if dist.metadata.get("Name")),
            "measurement_limit": "Single CPU run on eight authored smoke queries; not throughput, generalization, or a production SLA."})
    except Exception as exc:
        # Never expose tokens, network locations or arbitrary upstream exception text.
        result.update({"status": "FAILED", "error_type": type(exc).__name__,
                       "failure_stage": "PIN_LOAD_OR_INFERENCE", "measurement_complete": False})
    result["elapsed_seconds"] = time.perf_counter() - started
    result["receipt_sha256"] = digest(result)
    return result


def write_report(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Download verified public weights and run bounded CPU smoke inference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = execute() if args.execute else plan()
    if args.output:
        write_report(args.output, result)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
