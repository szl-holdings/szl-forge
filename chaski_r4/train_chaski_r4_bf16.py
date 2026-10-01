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

R4 = Path(__file__).resolve().parent
DATASET_FILE = R4 / "train.jsonl"
ADAPTER_DIR = R4 / "chaski-r4-adapter"
OUTPUT_DIR = R4 / "outputs-r4"
RECEIPT_FILE = R4 / "training_receipt.json"

ARTIFACT = "SZLHOLDINGS/chaski-r4"
CANONICAL_GATE_ARTIFACT = "SZLHOLDINGS/chaski"
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"

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

DRAFT_KEYS = {
    "decision",
    "approvalRequired",
    "executed",
    "artifact",
    "base_model",
    "claim",
    "label",
}
LABELS = {
    "MEASURED",
    "REPORTED",
    "DECLARED",
    "SIMULATED",
    "UNKNOWN",
    "UNAVAILABLE",
    "ROADMAP",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gpu_metadata() -> dict[str, Any]:
    nvidia = "UNAVAILABLE"
    try:
        nvidia = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "nvidia_smi": nvidia,
    }

def read_rows() -> list[dict[str, Any]]:
    if not DATASET_FILE.is_file():
        raise SystemExit(f"[chaski-r4] refuse: missing dataset {DATASET_FILE}")

    rows = [
        json.loads(line)
        for line in DATASET_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != 40:
        raise SystemExit(f"[chaski-r4] refuse: expected 40 rows, found {len(rows)}")

    drafts = 0
    refusals = 0

    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise SystemExit(f"[chaski-r4] refuse: malformed messages row {index}")
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise SystemExit(f"[chaski-r4] refuse: role contract drift row {index}")

        assistant = str(messages[-1].get("content", ""))

        if row.get("kind") == "draft":
            drafts += 1
            try:
                target = json.loads(assistant)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[chaski-r4] refuse: invalid draft JSON row {index}: {exc}") from exc
            if set(target) != DRAFT_KEYS:
                raise SystemExit(f"[chaski-r4] refuse: draft keys drift row {index}")
            if target.get("decision") != "DRAFT":
                raise SystemExit(f"[chaski-r4] refuse: draft decision drift row {index}")
            if target.get("approvalRequired") is not True:
                raise SystemExit(f"[chaski-r4] refuse: draft approval drift row {index}")
            if target.get("executed") is not False:
                raise SystemExit(f"[chaski-r4] refuse: draft execution drift row {index}")
            if target.get("artifact") != CANONICAL_GATE_ARTIFACT:
                raise SystemExit(f"[chaski-r4] refuse: canonical artifact drift row {index}")
            if target.get("base_model") != CANONICAL_BASE:
                raise SystemExit(f"[chaski-r4] refuse: base model drift row {index}")
            if not isinstance(target.get("claim"), str) or not target["claim"].strip():
                raise SystemExit(f"[chaski-r4] refuse: empty claim row {index}")
            if target.get("label") not in LABELS:
                raise SystemExit(f"[chaski-r4] refuse: invalid label row {index}")
        elif row.get("kind") == "refusal":
            refusals += 1
            if not assistant.startswith("REFUSE:"):
                raise SystemExit(f"[chaski-r4] refuse: refusal prefix drift row {index}")
            if assistant.lstrip().startswith("{"):
                raise SystemExit(f"[chaski-r4] refuse: refusal JSON drift row {index}")
        else:
            raise SystemExit(f"[chaski-r4] refuse: unknown row kind row {index}")

    if drafts != 24 or refusals != 16:
        raise SystemExit(
            f"[chaski-r4] refuse: expected 24 drafts/16 refusals, got {drafts}/{refusals}"
        )

    return rows

def make_receipt(
    live: bool,
    dataset_sha: str,
    training_rows: int,
    train_loss: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "kind": "szl-chaski-r4-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "v": 1,
        "artifact": ARTIFACT,
        "canonical_gate_artifact": CANONICAL_GATE_ARTIFACT,
        "canonical_base": CANONICAL_BASE,
        "base_model": CANONICAL_BASE,
        "base_model_relation": "adapter",
        "base_model_license": "Apache-2.0",
        "quant": "bf16-lora",
        "load_in_4bit": LOAD_IN_4BIT,
        "load_in_16bit": LOAD_IN_16BIT,
        "dataset_file": "chaski_r4/train.jsonl",
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
        receipt["finalTrainLoss"] = train_loss
    if metrics is not None:
        receipt["metrics"] = metrics

    return receipt

def write_receipt(payload: dict[str, Any]) -> None:
    RECEIPT_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def status() -> int:
    rows = read_rows()
    dataset_sha = sha256(DATASET_FILE)
    print(f"[chaski-r4] artifact={ARTIFACT}")
    print(f"[chaski-r4] canonical_gate_artifact={CANONICAL_GATE_ARTIFACT}")
    print(f"[chaski-r4] base={CANONICAL_BASE}")
    print(f"[chaski-r4] rows={len(rows)} sha256={dataset_sha}")
    print(f"[chaski-r4] bf16-lora r={LORA_R} alpha={LORA_ALPHA} seed={SEED}")
    print("[chaski-r4] held-out gates excluded; local only; no upload; no promotion")
    write_receipt(make_receipt(False, dataset_sha, len(rows)))
    return 0

def train() -> int:
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-r4] refuse: canonical base drift")
    if LOAD_IN_4BIT or not LOAD_IN_16BIT:
        raise SystemExit("[chaski-r4] refuse: bf16 LoRA required; QLoRA forbidden")
    if LORA_R != 16 or LORA_ALPHA != 32:
        raise SystemExit("[chaski-r4] refuse: expected r=16 alpha=32")

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    rows = read_rows()
    dataset_sha = sha256(DATASET_FILE)

    print(f"[chaski-r4] START artifact={ARTIFACT} base={CANONICAL_BASE}")
    print(f"[chaski-r4] rows={len(rows)} sha256={dataset_sha}")
    print(f"[chaski-r4] bf16-lora r={LORA_R} alpha={LORA_ALPHA} epochs={NUM_EPOCHS}")
    print("[chaski-r4] local training only; push_to_hub=false; train loss is not evaluation")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CANONICAL_BASE,
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
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
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

    receipt = make_receipt(True, dataset_sha, len(rows), loss, metrics)

    adapter_config = ADAPTER_DIR / "adapter_config.json"
    adapter_weights = ADAPTER_DIR / "adapter_model.safetensors"

    if adapter_config.is_file():
        receipt["adapter_config_sha256"] = sha256(adapter_config)
    if adapter_weights.is_file():
        receipt["adapter_weights_sha256"] = sha256(adapter_weights)
        receipt["adapter_weights_bytes"] = adapter_weights.stat().st_size

    write_receipt(receipt)

    print(f"[chaski-r4] DONE train_loss={loss} (MEASURED train metric, not an eval)")
    print(f"[chaski-r4] adapter={ADAPTER_DIR}")
    print(f"[chaski-r4] receipt={RECEIPT_FILE}")
    print("[chaski-r4] required next step: repaired canonical held-out bakeoff; no upload or promotion authorized")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "train"])
    args = parser.parse_args()
    raise SystemExit(status() if args.command == "status" else train())
