from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

R3 = Path(__file__).resolve().parent
DATASET_FILE = R3 / "train.jsonl"
ADAPTER_DIR = R3 / "chaski-r3-adapter"
OUTPUT_DIR = R3 / "outputs-r3"
RECEIPT_FILE = R3 / "training_receipt.json"

ARTIFACT = "SZLHOLDINGS/chaski-r3"
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
BASE = CANONICAL_BASE
SEED = 11
MAX_SEQ_LEN = 2048
BATCH = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 3
WARMUP_STEPS = 6
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LOAD_IN_4BIT = False
LOAD_IN_16BIT = True

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gpu_metadata() -> dict[str, Any]:
    value = "UNAVAILABLE"
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "nvidia_smi": value,
    }

def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"[chaski-r3] refuse: missing dataset {path}")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("[chaski-r3] refuse: empty curriculum")

    for i, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise SystemExit(f"[chaski-r3] refuse: malformed messages row {i}")
        if messages[-1].get("role") != "assistant":
            raise SystemExit(f"[chaski-r3] refuse: final role is not assistant row {i}")

        target = json.loads(messages[-1].get("content", ""))
        if target.get("decision") != "DRAFT":
            raise SystemExit(f"[chaski-r3] refuse: non-DRAFT target row {i}")
        if target.get("approvalRequired") is not True or target.get("executed") is not False:
            raise SystemExit(f"[chaski-r3] refuse: approval/execution contract drift row {i}")

    return rows

def make_receipt(
    live: bool,
    dataset_sha: str,
    training_rows: int,
    train_loss: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "szl-chaski-r3-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "v": 1,
        "artifact": ARTIFACT,
        "canonical_base": CANONICAL_BASE,
        "base_model": BASE,
        "base_model_relation": "adapter",
        "base_model_license": "Apache-2.0",
        "quant": "bf16-lora",
        "load_in_4bit": LOAD_IN_4BIT,
        "load_in_16bit": LOAD_IN_16BIT,
        "dataset_file": "chaski_r3/train.jsonl",
        "dataset_sha256": dataset_sha,
        "training_rows": training_rows,
        "held_out_in_gradients": False,
        "held_out": {
            "chaski/gate/json_drafts.n5.jsonl": 5,
            "chaski/gate/adversarial_refusals.n6.jsonl": 6,
        },
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LEN,
        "per_device_train_batch_size": BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "learning_rate": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "constant_with_warmup",
        "response_only_loss": True,
        "push_to_hub": False,
        "hub_put": False,
        "weights": "LOCAL" if live else "UNAVAILABLE",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "productionAuthorization": False,
        "train_loss_label": "MEASURED_NOT_EVALUATION" if live else "UNAVAILABLE",
        "evals": "none-this-run",
        "quality": "UNAVAILABLE",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_metadata(),
    }
    if train_loss is not None:
        result["finalTrainLoss"] = train_loss
    if metrics is not None:
        result["metrics"] = metrics
    return result

def write_receipt(data: dict[str, Any]) -> None:
    RECEIPT_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def status_main() -> int:
    rows = read_rows(DATASET_FILE)
    dataset_sha = sha256(DATASET_FILE)
    print(f"[chaski-r3] artifact={ARTIFACT} base={BASE}")
    print(f"[chaski-r3] bf16-lora r={LORA_R} alpha={LORA_ALPHA} seed={SEED}")
    print(f"[chaski-r3] rows={len(rows)} sha256={dataset_sha}")
    print("[chaski-r3] held-out gates excluded; hub upload disabled; production authorization false")
    write_receipt(make_receipt(False, dataset_sha, len(rows)))
    return 0

def train_main() -> int:
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-r3] refuse: canonical base drifted")
    if LOAD_IN_4BIT or not LOAD_IN_16BIT:
        raise SystemExit("[chaski-r3] refuse: this experiment is bf16 LoRA, not QLoRA")
    if LORA_R != 16 or LORA_ALPHA != 32:
        raise SystemExit("[chaski-r3] refuse: reproduction pin requires r=16 alpha=32")

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    rows = read_rows(DATASET_FILE)
    dataset_sha = sha256(DATASET_FILE)

    print(f"[chaski-r3] START artifact={ARTIFACT} base={BASE}")
    print(f"[chaski-r3] rows={len(rows)} sha256={dataset_sha} r={LORA_R} alpha={LORA_ALPHA} epochs={NUM_EPOCHS}")
    print("[chaski-r3] local-only training; push_to_hub=False; train loss is not evaluation")

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
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
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
        key: value
        for key, value in getattr(stats, "metrics", {}).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    data = make_receipt(True, dataset_sha, len(rows), loss, metrics)
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    if adapter_config.is_file():
        data["adapter_config_sha256"] = sha256(adapter_config)
    write_receipt(data)

    print(f"[chaski-r3] DONE train_loss={loss} (MEASURED train metric, not an eval)")
    print(f"[chaski-r3] adapter={ADAPTER_DIR}")
    print(f"[chaski-r3] receipt={RECEIPT_FILE}")
    print("[chaski-r3] next required action: held-out bakeoff; no upload or promotion is authorized")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "train"])
    args = parser.parse_args()
    raise SystemExit(status_main() if args.command == "status" else train_main())
