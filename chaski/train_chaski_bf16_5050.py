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
"""Owner-GPU Chaski-5050 recipe. Local RTX 5050 Unsloth LoRA. Not an HF Job.

Byte-aligned to the running owner file:
  chaski/train_chaski_bf16_5050.py

CANONICAL_BASE = Qwen/Qwen3.5-0.8B
HUB = SZLHOLDINGS/chaski-5050
FORBIDDEN_HUB = SZLHOLDINGS/chaski
jsonl-only szl_dataset.jsonl. Refuse SZL_ESTATE_MANAGED.json.
SEED=11 LORA_R=16 LORA_ALPHA=16 (not live Chaski 16/32)
MAX_SEQ_LEN=2048 NUM_EPOCHS=3 BATCH=1 GRAD_ACCUM=4 LR=2e-4
load_in_4bit=False load_in_16bit=True. QLoRA forbidden.
report_to=none. push_to_hub=False during train. FORGE uploads after.
job=local-5050. Evals none-this-run. publication_eligible false.
Training card label REPORTED owner-metal until a signed receipt exists.
train_loss may be MEASURED as a train metric, not an eval.
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

CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
BASE = os.environ.get("BASE_MODEL", CANONICAL_BASE)
HUB = os.environ.get("HUB_MODEL_ID", "SZLHOLDINGS/chaski-5050")
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
DATASET = "SZLHOLDINGS/szl-1-doctrine-sft"
DATASET_FILE = "szl_dataset.jsonl"
FORBIDDEN_ESTATE = "SZL_ESTATE_MANAGED.json"
SEED = 11
LORA_R = 16
LORA_ALPHA = 16
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 3
BATCH = 1
GRAD_ACCUM = 4
LR = 2e-4
WARMUP_STEPS = 6
LOAD_IN_4BIT = False
LOAD_IN_16BIT = True
JOB = "local-5050"
ADAPTER_DIR = HERE / "chaski-5050-adapter"
OUTPUT_DIR = HERE / "outputs-5050"
RECEIPT_STATUS = HERE / "training_receipt_5050.status.json"
RECEIPT_TRAIN = HERE / "training_receipt_5050.json"

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


def assert_bf16_loader(load_in_4bit: bool, load_in_16bit: bool) -> None:
    if load_in_4bit is True:
        raise SystemExit(
            "[chaski-5050] refuse: load_in_4bit=True. QLoRA forbidden. "
            "bf16 LoRA only (load_in_4bit=False, load_in_16bit=True)."
        )
    if load_in_16bit is not True:
        raise SystemExit(
            "[chaski-5050] refuse: load_in_16bit must be True. QLoRA forbidden."
        )


def assert_not_live_chaski(repo_id: str) -> None:
    normalized = (repo_id or "").strip()
    if normalized == FORBIDDEN_HUB:
        raise SystemExit(
            "[chaski-5050] refuse: never overwrite SZLHOLDINGS/chaski. "
            "This kit is SZLHOLDINGS/chaski-5050 only."
        )
    if normalized != HUB and normalized != "SZLHOLDINGS/chaski-5050":
        raise SystemExit(
            f"[chaski-5050] refuse: hub id {normalized!r} is not "
            "SZLHOLDINGS/chaski-5050."
        )


def refuse_estate_on_path(*paths: str | Path) -> None:
    for raw in paths:
        path = Path(raw)
        if path.name == FORBIDDEN_ESTATE:
            raise SystemExit(
                f"[chaski-5050] refuse: {FORBIDDEN_ESTATE} is on the path. "
                "jsonl-only."
            )


def load_doctrine_jsonl(dataset_file: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    if dataset_file is not None:
        refuse_estate_on_path(dataset_file)
        path = Path(dataset_file)
        if not path.is_file():
            raise SystemExit(f"[chaski-5050] refuse: missing dataset {path}")
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
    assert_not_live_chaski(HUB)
    assert_bf16_loader(LOAD_IN_4BIT, LOAD_IN_16BIT)
    local = adapter_present()
    return {
        "kind": "szl-chaski-5050-training-receipt",
        "artifact": HUB,
        "forbidden_hub": FORBIDDEN_HUB,
        "base_model": CANONICAL_BASE,
        "base_model_runtime": BASE,
        "dataset": DATASET,
        "dataset_file": DATASET_FILE,
        "seed": SEED,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LEN,
        "num_epochs": NUM_EPOCHS,
        "batch": BATCH,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "warmup_steps": WARMUP_STEPS,
        "load_in_4bit": False,
        "load_in_16bit": True,
        "qlora_forbidden": True,
        "report_to": "none",
        "push_to_hub": False,
        "job": JOB,
        "hardware": "local-RTX-5050",
        "training_loss": train_loss,
        "train_loss_label": "MEASURED" if train_loss is not None else "UNAVAILABLE",
        "train_loss_is_not_eval": True,
        "label": "REPORTED owner-metal",
        "signed_receipt": False,
        "evals": "none-this-run",
        "publication_eligible": False,
        "weights": "LOCAL" if local else "UNAVAILABLE",
        "adapter": "LOCAL" if local else "UNAVAILABLE",
        "local_adapter_dir": str(ADAPTER_DIR),
        "proposal_only": True,
        "serve_pin": HUB,
        "khipu_lab_pin": False,
        "a11oy_mini": False,
        "hub_push": False,
        "claim_boundary": (
            "Separate SKU SZLHOLDINGS/chaski-5050. Not live SZLHOLDINGS/chaski. "
            "job=local-5050. Not an HF Job. QLoRA forbidden. "
            "Training card label REPORTED owner-metal until a signed receipt exists. "
            "train_loss may be MEASURED as a train metric, not an eval. "
            "Evals none-this-run. Not 5/5. publication_eligible false. "
            "A11OY-MINI is a GGUF of live Chaski, not 5050. No tok/s. "
            "No Khipu lab pin. push_to_hub=False during train; FORGE uploads after."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "local-train" if live else "forge-status",
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chaski-5050] wrote {path}")


def status_main() -> int:
    assert_not_live_chaski(HUB)
    assert_bf16_loader(LOAD_IN_4BIT, LOAD_IN_16BIT)
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-5050] refuse: canonical base drifted")
    if LORA_ALPHA != 16 or LORA_R != 16:
        raise SystemExit("[chaski-5050] refuse: owner pin is r=16 alpha=16")
    print(
        f"[chaski-5050] base={BASE} canonical={CANONICAL_BASE} hub={HUB} "
        f"seed={SEED} r={LORA_R} alpha={LORA_ALPHA}"
    )
    print(
        f"[chaski-5050] job={JOB} QLoRA forbidden. "
        f"weights={'LOCAL' if adapter_present() else 'UNAVAILABLE'} "
        "evals=none-this-run label=REPORTED owner-metal"
    )
    write_receipt(kit_receipt(live=False), RECEIPT_STATUS)
    return 0


def train_main(dataset_file: Path | None) -> int:
    assert_not_live_chaski(HUB)
    assert_bf16_loader(LOAD_IN_4BIT, LOAD_IN_16BIT)
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    print(
        f"[chaski-5050] base={BASE} canonical={CANONICAL_BASE} hub={HUB} "
        f"seed={SEED} r={LORA_R} alpha={LORA_ALPHA} job={JOB}"
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
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=NUM_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LR,
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
    print(f"[chaski-5050] train done train_loss={loss} (MEASURED train metric, not an eval)")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    receipt = kit_receipt(live=True, train_loss=loss)
    receipt["dataset_sha256"] = doctrine_sha
    receipt["training_rows"] = len(rows)
    receipt["metrics"] = metrics
    write_receipt(receipt, RECEIPT_TRAIN)
    print(
        f"[chaski-5050] local adapter saved; push_to_hub=False; "
        f"FORGE uploads after; never overwrite {FORBIDDEN_HUB}"
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
    args = parser.parse_args()
    assert_not_live_chaski(HUB)
    assert_bf16_loader(args.load_in_4bit, LOAD_IN_16BIT)
    if args.train:
        return train_main(args.dataset_file)
    if args.dataset_file is not None:
        load_doctrine_jsonl(args.dataset_file)
    return status_main()


if __name__ == "__main__":
    raise SystemExit(main())
