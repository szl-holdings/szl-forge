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
# ]
# ///
"""Chaski-5050 local training — RTX 5050 Unsloth LoRA. NOT an HF Job.

NEW Hub id: SZLHOLDINGS/chaski-5050. Not live SZLHOLDINGS/chaski.
Do not push to SZLHOLDINGS/chaski. No Hub PUT from this checkout.

Verified Hub receipt (GitHub stamp only):
Hub SZLHOLDINGS/chaski-5050 commit c907ebe6e1fa900021be7b6fec19b38ec45be574.
adapter_model.safetensors present. weights AVAILABLE.
training_receipt.json: train_loss MEASURED 2.228136855544466,
train_runtime 883.2224s, 3 epochs, 41 rows, seed 11, r=16 alpha=16,
QLoRA false, job local-5050.
dataset_sha256 ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243.
train_loss MEASURED is a train metric, not an eval.
The SKU is NOT MEASURED. Do not stamp the model as MEASURED.

CHAWPI fashion: separate SKU. Do not add chaski-5050 to the Receipted
Unsloth LIVE shelf. That shelf stays live Chaski / Khipu / ReceiptAgent.
Keep it off the LIVE portfolio table. No lab load. Do not merge #59.
Train loss is not an eval.

CANONICAL_BASE = Qwen/Qwen3.5-0.8B (Apache, disclosed).
jsonl-only szl_dataset.jsonl. Refuse SZL_ESTATE_MANAGED.json.

QLoRA forbidden. load_in_4bit=False, load_in_16bit=True, r=16, alpha=16,
seed=11, warmup_steps=6, 3 epochs, batch 1, ga 4, seq 2048, adamw_8bit,
unsloth gc, report_to=none.

Training label REPORTED owner-metal until a signed receipt exists.
Evals none-this-run. SKU is NOT MEASURED. Not 5/5. publication_eligible false.
Not A11OY-MINI (that GGUF is live Chaski). Not an alias of live Chaski.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

MAX_SEQ_LEN = 2048
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
BASE = os.environ.get("BASE_MODEL", CANONICAL_BASE)
LIVE_CHASKI_HUB = "SZLHOLDINGS/chaski"
FORBIDDEN_HUB = LIVE_CHASKI_HUB
ALLOWED_HUB = "SZLHOLDINGS/chaski-5050"
HUB = os.environ.get("HUB_MODEL_ID", ALLOWED_HUB)
DATASET = "SZLHOLDINGS/szl-1-doctrine-sft"
DATASET_FILE = "szl_dataset.jsonl"
FORBIDDEN_ESTATE = "SZL_ESTATE_MANAGED.json"
SEED = 11
LORA_R = 16
LORA_ALPHA = 16
WARMUP_STEPS = 6
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
LOAD_IN_4BIT = False
LOAD_IN_16BIT = True
ADAPTER_DIR = HERE / "chaski-5050-adapter"
OUTPUT_DIR = HERE / "outputs-5050"
RECEIPT_STATUS = HERE / "training_receipt_5050.status.json"
RECEIPT_TRAIN = HERE / "training_receipt_5050.json"

# GitHub-only Hub receipt stamp. No Hub PUT. Do not change r/alpha.
# Hub SZLHOLDINGS/chaski-5050 commit c907ebe6e1fa900021be7b6fec19b38ec45be574
# adapter_model.safetensors present
# training_receipt.json: train_loss MEASURED 2.228136855544466, train_runtime 883.2224s, 3 epochs, 41 rows, seed 11, r=16 alpha=16, QLoRA false, job local-5050
# dataset_sha256 ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243
# evals none-this-run. publication_eligible false. weights AVAILABLE
# train_loss MEASURED is a train metric, not an eval. Training label REPORTED owner-metal until a signed receipt exists. Not 5/5.
# SKU is NOT MEASURED (evals none-this-run, publication_eligible false). Do not stamp the model as MEASURED.
# CHAWPI fashion: separate SKU. Do not add to Receipted Unsloth LIVE shelf.
# That shelf stays live Chaski / Khipu / ReceiptAgent. Keep it off the LIVE portfolio table.
# No lab load. Do not merge #59. Train loss is not an eval.
HUB_COMMIT = "c907ebe6e1fa900021be7b6fec19b38ec45be574"
HUB_ADAPTER_FILE = "adapter_model.safetensors"
HUB_TRAIN_LOSS = 2.228136855544466
HUB_TRAIN_RUNTIME_S = 883.2224
HUB_TRAINING_ROWS = 41
HUB_DATASET_SHA256 = (
    "ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243"
)
HUB_JOB = "local-5050"
HUB_WEIGHTS = "AVAILABLE"
SKU_STATUS = "NOT MEASURED"
LIVE_SHELF = False
LIVE_PORTFOLIO = False

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


def refuse_live_chaski_hub() -> None:
    assert_push_repo(HUB)


def assert_bf16_loader(load_in_4bit: bool, load_in_16bit: bool) -> None:
    """Refuse QLoRA. This SKU is bf16 LoRA only."""
    if load_in_4bit is True:
        raise SystemExit(
            "[chaski-5050] refuse: load_in_4bit=True. QLoRA forbidden. "
            "bf16 LoRA only (load_in_4bit=False, load_in_16bit=True)."
        )
    if load_in_16bit is not True:
        raise SystemExit(
            "[chaski-5050] refuse: load_in_16bit must be True. QLoRA forbidden."
        )


def assert_push_repo(repo_id: str) -> None:
    """Never overwrite live SZLHOLDINGS/chaski. This checkout does not Hub PUT."""
    normalized = (repo_id or "").strip()
    if normalized == FORBIDDEN_HUB:
        raise SystemExit(
            "[chaski-5050] refuse: never overwrite SZLHOLDINGS/chaski. "
            "This kit is SZLHOLDINGS/chaski-5050 only."
        )
    if normalized != ALLOWED_HUB:
        raise SystemExit(
            f"[chaski-5050] refuse: hub id {normalized!r} is not {ALLOWED_HUB}."
        )


def refuse_qlora() -> None:
    assert_bf16_loader(LOAD_IN_4BIT, LOAD_IN_16BIT)


def refuse_estate_on_path(*paths: str | Path) -> None:
    for raw in paths:
        path = Path(raw)
        if path.name == FORBIDDEN_ESTATE:
            raise SystemExit(
                f"[chaski-5050] refuse: {FORBIDDEN_ESTATE} is on the path. "
                "jsonl-only."
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
        raise SystemExit(f"[chaski-5050] refuse: will not ingest {FORBIDDEN_ESTATE}")
    doctrine_sha = sha256_bytes(raw)
    doctrine_rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not doctrine_rows or "messages" not in doctrine_rows[0]:
        raise SystemExit(f"[chaski-5050] {DATASET_FILE} has no messages rows")
    rows = [{"messages": r["messages"]} for r in doctrine_rows] + OUROBOROS
    print(
        f"[chaski-5050] examples={len(rows)} doctrine_rows={len(doctrine_rows)} "
        f"sha256={doctrine_sha}"
    )
    return rows, doctrine_sha


def adapter_present() -> bool:
    return ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))


def kit_receipt(*, live: bool = False, train_loss: float | None = None) -> dict[str, Any]:
    refuse_live_chaski_hub()
    local = adapter_present()
    return {
        "kind": "szl-chaski-5050-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "artifact": HUB,
        "not_live_chaski": LIVE_CHASKI_HUB,
        "base_model": CANONICAL_BASE,
        "base_model_runtime": BASE,
        "base_model_relation": "adapter",
        "dataset": DATASET,
        "dataset_file": DATASET_FILE,
        "extra_identity_turns": len(OUROBOROS),
        "seed": SEED,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "warmup_steps": WARMUP_STEPS,
        "qlora_forbidden": True,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "max_seq_length": MAX_SEQ_LEN,
        "optim": "adamw_8bit",
        "load_in_4bit": False,
        "load_in_16bit": True,
        "use_gradient_checkpointing": "unsloth",
        "report_to": "none",
        "hardware": "local-RTX-5050",
        "jobs": "not-an-hf-job",
        "response_only_loss": True,
        "training_loss": train_loss,
        "label": "REPORTED owner-metal",
        "signed_receipt": False,
        "evals": "none-this-run",
        "train_loss_is_not_eval": True,
        "weights": "LOCAL" if local else "UNAVAILABLE",
        "adapter": "LOCAL" if local else "UNAVAILABLE",
        "hub_commit": HUB_COMMIT,
        "hub_adapter_file": HUB_ADAPTER_FILE,
        "hub_train_loss": HUB_TRAIN_LOSS,
        "hub_train_loss_label": "MEASURED",
        "hub_train_runtime_s": HUB_TRAIN_RUNTIME_S,
        "hub_training_rows": HUB_TRAINING_ROWS,
        "hub_dataset_sha256": HUB_DATASET_SHA256,
        "hub_job": HUB_JOB,
        "hub_weights": HUB_WEIGHTS,
        "sku_status": SKU_STATUS,
        "live_shelf": LIVE_SHELF,
        "live_portfolio": LIVE_PORTFOLIO,
        "local_adapter_dir": str(ADAPTER_DIR),
        "proposal_only": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "serve_pin": HUB,
        "hub_push": False,
        "a11oy_mini": False,
        "alias_of_live_chaski": False,
        "claim_boundary": (
            "Separate SKU SZLHOLDINGS/chaski-5050. Not live SZLHOLDINGS/chaski. "
            "Local RTX 5050 Unsloth LoRA. QLoRA forbidden. Not an HF Job. "
            "Training label REPORTED owner-metal until a signed receipt exists. "
            "Evals none-this-run. Not 5/5. publication_eligible false. "
            "SKU is NOT MEASURED. Do not stamp the model as MEASURED. "
            "train_loss MEASURED is a train metric, not an eval. "
            "weights AVAILABLE on Hub. "
            "Do not overwrite SZLHOLDINGS/chaski. Do not load into the Khipu lab. "
            "CHAWPI fashion: separate SKU. Off Receipted Unsloth LIVE shelf "
            "(live Chaski / Khipu / ReceiptAgent). Off the LIVE portfolio table. "
            "No lab load. Do not merge #59. Train loss is not an eval. "
            "No tok/s claims. A11OY-MINI is a GGUF of live Chaski, not 5050. "
            "GitHub stamp only. No Hub PUT from this checkout."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "local-train" if live else "forge-status",
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chaski-5050] wrote {path}")


def status_main() -> int:
    refuse_live_chaski_hub()
    refuse_qlora()
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-5050] refuse: canonical base drifted")
    print(
        f"[chaski-5050] base={BASE} canonical={CANONICAL_BASE} hub={HUB} seed={SEED}"
    )
    print(
        "[chaski-5050] local RTX 5050 Unsloth LoRA. NOT an HF Job. "
        f"weights={'LOCAL' if adapter_present() else 'UNAVAILABLE'} "
        "evals=none-this-run"
    )
    receipt = kit_receipt(live=False)
    write_receipt(receipt, RECEIPT_STATUS)
    return 0


def train_main(dataset_file: Path | None) -> int:
    refuse_live_chaski_hub()
    refuse_qlora()
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    print(
        f"[chaski-5050] base={BASE} canonical={CANONICAL_BASE} hub={HUB} seed={SEED}"
    )
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-5050] refuse: canonical base drifted")
    rows, doctrine_sha = load_doctrine_jsonl(dataset_file)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_16bit=LOAD_IN_16BIT,
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
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="constant_with_warmup",
            seed=SEED,
            output_dir=str(OUTPUT_DIR),
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
    print(f"[chaski-5050] train done loss={loss} metrics={metrics}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    receipt = kit_receipt(live=True, train_loss=loss)
    receipt["dataset_sha256"] = doctrine_sha
    receipt["training_rows"] = len(rows)
    receipt["metrics"] = metrics
    receipt["hub_adapter_claimed"] = False
    write_receipt(receipt, RECEIPT_TRAIN)
    print(
        "[chaski-5050] local adapter saved; do not push to "
        f"{LIVE_CHASKI_HUB}; target id is {HUB}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run local RTX 5050 Unsloth LoRA. Default is status (no GPU, no Hub write).",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Local jsonl only. Refuses SZL_ESTATE_MANAGED.json.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Forbidden. Present so the recut can refuse QLoRA loudly.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Forbidden. This checkout does not Hub PUT.",
    )
    parser.add_argument(
        "--hub-model-id",
        default=os.environ.get("HUB_MODEL_ID", ALLOWED_HUB),
        help=f"Must be {ALLOWED_HUB}. Never {FORBIDDEN_HUB}.",
    )
    args = parser.parse_args()
    assert_bf16_loader(args.load_in_4bit, LOAD_IN_16BIT)
    if args.hub_model_id.strip() == FORBIDDEN_HUB or args.push:
        assert_push_repo(args.hub_model_id)
    if args.push:
        raise SystemExit(
            "[chaski-5050] refuse: No Hub PUT from this checkout. "
            f"Never overwrite {FORBIDDEN_HUB}."
        )
    if args.train:
        return train_main(args.dataset_file)
    if args.dataset_file is not None:
        load_doctrine_jsonl(args.dataset_file)
    return status_main()


if __name__ == "__main__":
    raise SystemExit(main())
