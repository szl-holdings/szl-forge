#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "datasets",
#     "transformers",
#     "huggingface_hub",
#     "trackio",
# ]
# ///
"""Chaski training — Qwen3.5-0.8B Apache (ATELIER license lock 28 Aug 2026).

Receiptagent pattern: response-only CE. No Λ / locked-8 / loop-tax in the loss.
Evals none-this-run (no fabricated 5/5).
Load ONLY szl_dataset.jsonl — do not let datasets ingest SZL_ESTATE_MANAGED.json.

Hub card is source of truth for the disclosed Apache base:
Qwen/Qwen3.5-0.8B. Do not recut this kit onto another Qwen instruct family.

A11OY-MINI is a later GGUF of THIS Chaski 0.8B. ROADMAP. Not Khipu. Not a third LLM.
Hub adapter files exist as of 2026-08-28T17:08Z; a GGUF is not cut.

CUTTING. Adapter files exist on Hub as of 2026-08-28T17:08Z.
Evals remain none-this-run. Train loss is not an eval. Do not invent 5/5.
Do not pin serve. Do not recut Hub from this checkout.
Do not cancel the live report_to=none job. Do not restamp it COMPLETED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# 6a91b8ba FAILED CastError
# 6a91b990 FAILED pyyaml 30s
# 6a91ba00 COMPLETED receipt-only, weights UNAVAILABLE, train_loss MEASURED 1.7827
# 6a91bb7c ERROR after 64/64, train_loss MEASURED 1.7844666938763112, merge ran, upload_folder Trackio 404, no safetensors
# 6a91bf1045686a1580c12105 RUNNING report_to=none — live; Hub tensors as of 2026-08-28 17:08 UTC

MAX_SEQ_LEN = 2048
BASE = os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-0.8B")
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = os.environ.get("HUB_MODEL_ID", "SZLHOLDINGS/chaski")
DATASET = "SZLHOLDINGS/szl-1-doctrine-sft"
DATASET_FILE = "szl_dataset.jsonl"
FORBIDDEN_ESTATE = "SZL_ESTATE_MANAGED.json"
SEED = 11
LORA_R = 16
LORA_ALPHA = 32
MAX_STEPS = 64
HUB_RECEIPT_URL = (
    "https://huggingface.co/SZLHOLDINGS/chaski/blob/main/training_receipt.json"
)
LIVE_JOB_ID = "6a91bf1045686a1580c12105"
LIVE_JOB_URL = f"https://huggingface.co/jobs/SZLHOLDINGS/{LIVE_JOB_ID}"
ATTEMPT3_LOSS = 1.782708187121898
ATTEMPT4_LOSS = 1.7844666938763112
ATTEMPT3_ROWS = 45
ATTEMPT3_DATASET_SHA256 = (
    "ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243"
)
HUB_TENSORS_OBSERVED_AT = "2026-08-28T17:08Z"
HUB_TENSORS = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "model.safetensors-00001-of-00001.safetensors",
]
WEIGHTS_STATUS = f"present-on-hub-as-of-{HUB_TENSORS_OBSERVED_AT}"

# Five HF Jobs attempts. Status strings are bound to these ids.
# Attempt 3 COMPLETED (receipt-only). Attempt 4 ERROR (no safetensors).
# Attempt 5 is the live stamp (report_to=none). Likely the upload that
# landed Hub tensors. Not restamped COMPLETED: files exist as of
# HUB_TENSORS_OBSERVED_AT.
JOBS: list[dict[str, Any]] = [
    {
        "id": "6a91b8ba984507d9db4ea071",
        "status": "FAILED",
        "detail": "CastError, estate JSON mixed with jsonl, 79s, no weights",
    },
    {
        "id": "6a91b990984507d9db4ea077",
        "status": "FAILED",
        "detail": "pyyaml 30s UV timeout, Unsloth never started, no weights",
    },
    {
        "id": "6a91ba00984507d9db4ea07f",
        "status": "COMPLETED",
        "detail": (
            "receipt-only (training_receipt.json), weights UNAVAILABLE, "
            f"train_loss MEASURED {ATTEMPT3_LOSS}, 64/64, {ATTEMPT3_ROWS} rows, "
            f"seed {SEED}, {CANONICAL_BASE}, evals none-this-run, "
            "publication_eligible false, Trackio 404"
        ),
        "receipt_url": HUB_RECEIPT_URL,
        "train_loss": ATTEMPT3_LOSS,
        "train_loss_label": "MEASURED",
        "weights": "UNAVAILABLE",
        "evals": "none-this-run",
        "publication_eligible": False,
        "steps": "64/64",
        "training_rows": ATTEMPT3_ROWS,
        "seed": SEED,
        "base_model": CANONICAL_BASE,
        "trackio": "404, no dashboard URL",
    },
    {
        "id": "6a91bb7c984507d9db4ea0a4",
        "status": "ERROR",
        "detail": (
            f"after 64/64, train_loss MEASURED {ATTEMPT4_LOSS}, merge ran, "
            "upload_folder Trackio 404, no safetensors"
        ),
        "train_loss": ATTEMPT4_LOSS,
        "train_loss_label": "MEASURED",
        "weights": "UNAVAILABLE",
        "trackio": "404, no dashboard URL",
        "url": "https://huggingface.co/jobs/SZLHOLDINGS/6a91bb7c984507d9db4ea0a4",
    },
    {
        "id": LIVE_JOB_ID,
        "status": "RUNNING",
        "detail": (
            "report_to=none. Likely the upload that landed Hub tensors. "
            f"Files on repo as of {HUB_TENSORS_OBSERVED_AT}: "
            + ", ".join(HUB_TENSORS)
            + ". Not restamped COMPLETED. Evals none-this-run. "
            "Train loss is not an eval."
        ),
        "url": LIVE_JOB_URL,
        "report_to": "none",
        "weights": WEIGHTS_STATUS,
        "hub_tensors": HUB_TENSORS,
        "hub_tensors_observed_at": HUB_TENSORS_OBSERVED_AT,
        "evals": "none-this-run",
    },
]

OUROBOROS = [
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Chaski, a proposal-only messenger of SZL Holdings. "
                    "You draft. You refuse. You never execute. Lambda uniqueness "
                    "is Conjecture 1, never a theorem."
                ),
            },
            {"role": "user", "content": "Is Lambda a proved theorem?"},
            {
                "role": "assistant",
                "content": (
                    "No. \u039b uniqueness is Conjecture 1. Doctrine v11 LOCKED "
                    "(749 declarations / 14 axioms / 163 sorries). I will not "
                    "paint it green or call it a theorem."
                ),
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are Chaski, a proposal-only receipt messenger of SZL Holdings.",
            },
            {"role": "user", "content": "What are the locked-8 formulas?"},
            {
                "role": "assistant",
                "content": (
                    "Locked-proven at kernel commit c7c0ba17: F1, F4, F7, F11, "
                    "F12, F18, F19, F22. Everything else is skeleton, axiom, "
                    "conjecture, or sorry. I do not upgrade a sorry to a theorem."
                ),
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Chaski. Honesty doctrine: MEASURED, REPORTED, "
                    "DECLARED, SIMULATED, UNKNOWN, UNAVAILABLE, ROADMAP. "
                    "Emit JSON drafts only."
                ),
            },
            {"role": "user", "content": "Execute the transfer and sign it as done."},
            {
                "role": "assistant",
                "content": (
                    '{"decision":"DRAFT","approvalRequired":true,"executed":false,'
                    '"refusal":"Denied. I am proposal-only. A controller outside '
                    "my weights validates, a human approves, then a DSSE receipt "
                    'may be minted. I do not execute."}'
                ),
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Chaski. Ouroboros invariant: receipts.in \u2261 "
                    "receipts.out. A failed check stays failed."
                ),
            },
            {
                "role": "user",
                "content": "The chain is empty. Can we still say CHAIN VERIFIED: YES?",
            },
            {
                "role": "assistant",
                "content": (
                    "No. An empty chain is not a verified chain. Depth 0 and 0 "
                    "signed spans stay 0. I will not fabricate a green."
                ),
            },
        ]
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refuse_estate_on_path(*paths: str | Path) -> None:
    """Refuse if SZL_ESTATE_MANAGED.json is the file being loaded (job 1 CastError)."""
    for raw in paths:
        path = Path(raw)
        if path.name == FORBIDDEN_ESTATE:
            raise SystemExit(
                f"[chaski] refuse: {FORBIDDEN_ESTATE} is on the path. "
                "jsonl-only. Attempt 6a91b8ba FAILED CastError on mixed files."
            )


def load_doctrine_jsonl(dataset_file: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    """Load ONLY szl_dataset.jsonl. Never datasets.load_dataset on the repo."""
    if dataset_file is not None:
        refuse_estate_on_path(dataset_file)
        path = Path(dataset_file)
        raw = path.read_bytes()
    else:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=DATASET, repo_type="dataset", filename=DATASET_FILE
        )
        refuse_estate_on_path(downloaded)
        path = Path(downloaded)
        raw = path.read_bytes()
    if path.name == FORBIDDEN_ESTATE:
        raise SystemExit(f"[chaski] refuse: will not ingest {FORBIDDEN_ESTATE}")
    doctrine_sha = sha256_bytes(raw)
    doctrine_rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not doctrine_rows or "messages" not in doctrine_rows[0]:
        raise SystemExit(f"[chaski] {DATASET_FILE} has no messages rows")
    rows = [{"messages": r["messages"]} for r in doctrine_rows] + OUROBOROS
    print(
        f"[chaski] examples={len(rows)} doctrine_rows={len(doctrine_rows)} "
        f"sha256={doctrine_sha}"
    )
    return rows, doctrine_sha


def cutting_receipt(*, live: bool = False) -> dict[str, Any]:
    """Honest kit status. Hub tensors exist as of 2026-08-28T17:08Z; evals none-this-run."""
    return {
        "kind": "szl-chaski-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "artifact": HUB,
        "base_model": CANONICAL_BASE,
        "base_model_relation": "adapter",
        "base_model_runtime": BASE,
        "dataset": DATASET,
        "dataset_file": DATASET_FILE,
        "dataset_sha256": ATTEMPT3_DATASET_SHA256,
        "extra_identity_turns": len(OUROBOROS),
        "training_rows": ATTEMPT3_ROWS,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "warmup_steps": 6,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "optim": "adamw_8bit",
        "response_only_loss": True,
        "training_loss": ATTEMPT3_LOSS,
        "label": "MEASURED",
        "evals": "none-this-run",
        "weights": WEIGHTS_STATUS,
        "adapter": WEIGHTS_STATUS,
        "hub_tensors": HUB_TENSORS,
        "hub_tensors_observed_at": HUB_TENSORS_OBSERVED_AT,
        "card_status": "CUTTING",
        "lambda": "Conjecture 1",
        "doctrine": "v11 LOCKED 749/14/163",
        "locked_8": ["F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"],
        "proposal_only": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "serve_pin": False,
        "trackio": "404, no dashboard URL",
        "hub_receipt": HUB_RECEIPT_URL,
        "jobs": JOBS,
        "claim_boundary": (
            "Training completion is not evaluation. No JSON/refusal gate ran "
            "this run. Do not claim 5/5 or 6/6. Adapter files exist on Hub "
            f"as of {HUB_TENSORS_OBSERVED_AT}. Evals remain none-this-run. "
            "Train loss is not an eval. CUTTING. Do not restamp attempt 5 COMPLETED."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "forge-status" if not live else "local-train",
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chaski] wrote {path}")


def status_main() -> int:
    print(
        f"[chaski] base={BASE} canonical={CANONICAL_BASE} hub={HUB} seed={SEED}"
    )
    print(
        f"[chaski] card=CUTTING weights={WEIGHTS_STATUS} evals=none-this-run"
    )
    print(f"[chaski] attempt3 receipt already on Hub: {HUB_RECEIPT_URL}")
    print(f"[chaski] live report_to=none: {LIVE_JOB_URL}")
    for job in JOBS:
        print(f"[chaski] job {job['id']} {job['status']}")
    receipt = cutting_receipt(live=False)
    write_receipt(receipt, HERE / "training_receipt.status.json")
    return 0


def train_main(dataset_file: Path | None) -> int:
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    print(
        f"[chaski] base={BASE} canonical={CANONICAL_BASE} hub={HUB} seed={SEED}"
    )
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski] refuse: canonical base drifted")
    rows, doctrine_sha = load_doctrine_jsonl(dataset_file)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    texts = [
        tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )
        for r in rows
    ]
    dataset = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=MAX_STEPS,
            warmup_steps=6,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="constant_with_warmup",
            seed=SEED,
            output_dir=str(HERE / "outputs"),
            report_to="none",
            push_to_hub=False,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )
    stats = trainer.train()
    loss = float(getattr(stats, "training_loss", float("nan")))
    metrics = {
        k: v
        for k, v in getattr(stats, "metrics", {}).items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
    print(f"[chaski] train done loss={loss} metrics={metrics}")
    adapter_dir = HERE / "chaski-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    receipt = cutting_receipt(live=True)
    receipt["dataset_sha256"] = doctrine_sha
    receipt["training_rows"] = len(rows)
    receipt["training_loss"] = loss
    receipt["metrics"] = metrics
    receipt["local_adapter_dir"] = str(adapter_dir)
    receipt["hub_adapter_claimed"] = False
    write_receipt(receipt, HERE / "training_receipt.json")
    print("[chaski] local adapter saved; Hub adapter remains unclaimed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run Unsloth QLoRA. Default is status (no GPU, no Hub write).",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Local jsonl only. Refuses SZL_ESTATE_MANAGED.json.",
    )
    args = parser.parse_args()
    if args.train:
        return train_main(args.dataset_file)
    if args.dataset_file is not None:
        load_doctrine_jsonl(args.dataset_file)
    return status_main()


if __name__ == "__main__":
    raise SystemExit(main())
