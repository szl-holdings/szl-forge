#!/usr/bin/env python3
"""Train a real QLoRA candidate from the committed ReceiptAgent curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qualify_runtime import (
    QualificationError,
    gpu_gate,
    load_candidate,
    sha256_json,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_PATH = "receiptagent/manifest.json"
TRAIN_FILES = (
    "receiptagent/train.jsonl",
    "receiptagent/train.refusals.jsonl",
)
ALL_CURRICULUM_FILES = (
    "receiptagent/train.jsonl",
    "receiptagent/eval.jsonl",
    "receiptagent/train.refusals.jsonl",
    "receiptagent/adversarial.jsonl",
    "receiptagent/receiptagent.schema.json",
)
REFUSAL_OVERSAMPLE = 2


def committed_bytes(path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def curriculum_evidence() -> tuple[dict[str, str], list[dict[str, Any]]]:
    manifest = json.loads(committed_bytes(MANIFEST_PATH))
    digests: dict[str, str] = {}
    for path in ALL_CURRICULUM_FILES:
        data = committed_bytes(path)
        digest = hashlib.sha256(data).hexdigest()
        name = Path(path).name
        declared = manifest.get("files", {}).get(name, {}).get("sha256")
        if digest != declared:
            raise QualificationError(
                f"committed {name} digest {digest} != manifest {declared}"
            )
        digests[name] = digest

    rows: list[dict[str, Any]] = []
    for path in TRAIN_FILES:
        parsed = [
            json.loads(line)
            for line in committed_bytes(path).decode("utf-8").splitlines()
            if line.strip()
        ]
        repeats = REFUSAL_OVERSAMPLE if path.endswith("train.refusals.jsonl") else 1
        for _ in range(repeats):
            rows.extend(parsed)
    if not rows:
        raise QualificationError("committed training curriculum is empty")
    return digests, rows


def vlm_conversation(row: dict[str, Any]) -> dict[str, Any]:
    converted = []
    for message in row["messages"]:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise QualificationError("training message has no text content")
        converted.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": content}],
            }
        )
    return {"messages": converted}


def hash_adapter(directory: Path) -> tuple[str, list[dict[str, Any]]]:
    files = []
    combined = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        files.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": digest,
            }
        )
    if not files:
        raise QualificationError("adapter save produced no files")
    return combined.hexdigest(), files


def train(args: argparse.Namespace) -> dict[str, Any]:
    import unsloth  # noqa: F401 - must patch before TRL/Transformers imports
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.chat_templates import train_on_responses_only
    from unsloth.trainer import UnslothVisionDataCollator

    candidate = load_candidate(args.candidate)
    gpu = gpu_gate(
        torch,
        min_free_gib=args.min_free_gib,
        max_temp_c=args.max_temp_c,
    )
    dataset_hashes, rows = curriculum_evidence()
    implementation = candidate["training_implementation"]
    model, processor = FastVisionModel.from_pretrained(
        model_name=implementation["repo_id"],
        revision=implementation["revision"],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        random_state=11,
        use_rslora=False,
        loftq_config=None,
    )
    converted = [vlm_conversation(row) for row in rows]
    output_dir = args.output_dir.resolve()
    checkpoints = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "warmup_steps": min(10, max(0, args.max_steps // 10)),
        "max_steps": args.max_steps,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
        "seed": 11,
        "output_dir": str(checkpoints),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "eos_token": processor.tokenizer.eos_token,
        "pad_token": (
            processor.tokenizer.pad_token or processor.tokenizer.eos_token
        ),
        "max_length": 2048,
        "save_strategy": "no",
    }
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=Dataset.from_list(converted),
        args=SFTConfig(**config_kwargs),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        tokenizer=processor,
    )
    FastVisionModel.for_training(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stats = trainer.train()
    duration = time.perf_counter() - started
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    adapter_sha, adapter_files = hash_adapter(adapter_dir)

    metrics = {
        key: value
        for key, value in stats.metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    report = {
        "schema": "szl.frontier-training-run/v1",
        "candidate_id": candidate["candidate_id"],
        "state": "MEASURED_TRAINING_COMPLETED",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or "unknown-host",
        "implementation": implementation,
        "dataset_hashes": dataset_hashes,
        "training_rows": len(converted),
        "configuration": {
            **config_kwargs,
            "refusal_oversample": REFUSAL_OVERSAMPLE,
            "finetune_vision_layers": False,
            "lora_r": 16,
            "lora_alpha": 32,
            "response_only_loss": True,
        },
        "gpu": {
            **gpu,
            "peak_reserved_bytes_training": torch.cuda.max_memory_reserved(),
        },
        "training": {
            "duration_seconds": round(duration, 6),
            "metrics": metrics,
        },
        "adapter": {
            "directory": str(adapter_dir),
            "aggregate_sha256": adapter_sha,
            "files": adapter_files,
        },
        "publication_eligible": False,
        "autonomy_eligible": False,
        "claim_boundary": (
            "Training completion is not evaluation. The adapter remains "
            "unpublishable until every held-out acceptance gate and signed "
            "receipt-chain check passes."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=HERE / "candidate.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--max-temp-c", type=int, default=80)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    report_path = args.output_dir / "training-report.json"
    try:
        report = train(args)
        code = 0
    except Exception as exc:  # noqa: BLE001 - always emit terminal evidence
        report = {
            "schema": "szl.frontier-training-run/v1",
            "state": "UNAVAILABLE",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "fatal": f"{type(exc).__name__}: {exc}",
            "publication_eligible": False,
            "autonomy_eligible": False,
        }
        report["report_sha256"] = sha256_json(report)
        code = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
