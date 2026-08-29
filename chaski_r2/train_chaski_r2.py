#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "datasets",
#     "transformers",
# ]
# ///
"""CHASKI-R2 Unsloth QLoRA kit. Separate SKU. ATELIER lock. No Hub PUT.

CHAWPI silhouette. Base in prose: Qwen/Qwen3.5-0.8B (Apache-2.0).
Reserved Hub id SZLHOLDINGS/chaski-r2 is declared only — not a Hub page.
Never overwrite SZLHOLDINGS/chaski.
Not SZLHOLDINGS/chaski-5050. Not bf16. Not the owner-metal sixteen-alpha kit.

QLoRA r=16 α=32, seed 11, response-only CE. Trains only chaski_r2/train.jsonl.
Refuses chaski/gate/*.jsonl (eval-only named-N files).

GPU honesty is MEASURED or UNAVAILABLE. No ROADMAP parking.
Jobs this checkout: UNAVAILABLE (not fired). publication_eligible false
until MEASURED generate. Doctrine v11 LOCKED. Λ = Conjecture 1.
Lab stays Khipu. A11OY-MINI stays scripts-only of live Chaski.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GATE_DIR = ROOT / "chaski" / "gate"
TRAIN_FILE = HERE / "train.jsonl"

CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
BASE_TRAIN = "unsloth/Qwen3.5-0.8B"
DEFAULT_HUB = "SZLHOLDINGS/chaski-r2"
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
FORBIDDEN_5050 = "SZLHOLDINGS/chaski-5050"
MAX_SEQ_LEN = 2048
SEED = 11
LORA_R = 16
LORA_ALPHA = 32
LR = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 6
JSON_FIELDS = (
    "decision",
    "approvalRequired",
    "executed",
    "artifact",
    "base_model",
    "claim",
    "label",
)
ADAPTER_DIR = HERE / "chaski-r2-adapter"
STATUS_RECEIPT = HERE / "training_receipt.status.json"
TRAIN_RECEIPT = HERE / "training_receipt.json"


def refuse_overwrite(hub: str) -> None:
    """Never retarget live Chaski or the 5050 SKU."""
    normalized = hub.strip().rstrip("/")
    upper = normalized.upper()
    if upper == FORBIDDEN_HUB.upper() or upper.startswith(FORBIDDEN_HUB.upper() + "/"):
        raise SystemExit(
            f"[chaski-r2] refuse: never overwrite {FORBIDDEN_HUB}. "
            "CHASKI-R2 is a separate SKU (SZLHOLDINGS/chaski-r2)."
        )
    if "CHASKI-5050" in upper or "BF16-5050" in upper:
        raise SystemExit(
            f"[chaski-r2] refuse: hub {hub!r} is the 5050 / bf16 kit. "
            "This SKU is SZLHOLDINGS/chaski-r2 only (QLoRA r=16 α=32)."
        )
    if normalized != DEFAULT_HUB:
        raise SystemExit(
            f"[chaski-r2] refuse: hub {normalized!r} is not {DEFAULT_HUB}."
        )


def refuse_gate_ingest(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(GATE_DIR.resolve())
    except ValueError:
        return
    raise SystemExit(
        f"[chaski-r2] refuse: will not ingest eval-only named-N file {path}. "
        "chaski/gate/*.jsonl stay held-out."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_safetensors_dir(directory: Path) -> str:
    files = sorted(glob.glob(str(directory / "*.safetensors")))
    if not files:
        return ""
    digest = hashlib.sha256()
    for path in files:
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_train_rows(dataset_file: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    path = Path(dataset_file) if dataset_file is not None else TRAIN_FILE
    refuse_gate_ingest(path)
    if path.name == "SZL_ESTATE_MANAGED.json":
        raise SystemExit("[chaski-r2] refuse: will not ingest SZL_ESTATE_MANAGED.json")
    if not path.is_file():
        raise SystemExit(f"[chaski-r2] refuse: missing curriculum {path}")
    raw = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    json_turns = 0
    refuse_abstain = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" not in row:
            raise SystemExit(f"[chaski-r2] refuse: row missing messages in {path}")
        assistant = row["messages"][-1]["content"]
        if assistant.startswith("REFUSE:") or assistant.startswith("ABSTAIN:"):
            refuse_abstain += 1
        else:
            gold = json.loads(assistant)
            missing = [key for key in JSON_FIELDS if key not in gold]
            if missing:
                raise SystemExit(
                    f"[chaski-r2] refuse: JSON turn missing {missing} in {path}"
                )
            if gold.get("artifact") != DEFAULT_HUB:
                raise SystemExit(
                    f"[chaski-r2] refuse: train JSON artifact must be {DEFAULT_HUB}"
                )
            if gold.get("base_model") != CANONICAL_BASE:
                raise SystemExit(
                    "[chaski-r2] refuse: train JSON base_model must be CANONICAL_BASE"
                )
            json_turns += 1
        rows.append({"messages": row["messages"]})
    if json_turns < 1 or refuse_abstain < 1:
        raise SystemExit(
            "[chaski-r2] refuse: curriculum needs JSON turns plus a "
            "REFUSE/ABSTAIN line"
        )
    digest = sha256_file(path)
    print(
        f"[chaski-r2] examples={len(rows)} json_turns={json_turns} "
        f"refuse_abstain={refuse_abstain} sha256={digest}"
    )
    return rows, digest


def status_receipt(
    *,
    hub: str,
    dataset_sha: str,
    live: bool = False,
    training_loss: str | None = None,
    adapter_sha: str = "",
    training_rows: int | None = None,
) -> dict[str, Any]:
    return {
        "kind": "szl-chaski-r2-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "v": 1,
        "artifact": hub,
        "sku": "CHASKI-R2",
        "silhouette": "CHAWPI",
        "atelier_lock": True,
        "hub_id_declared_only": True,
        "hub_page": False,
        "separate_sku": True,
        "canonical_base": CANONICAL_BASE,
        "base_model": CANONICAL_BASE,
        "baseModel": CANONICAL_BASE,
        "base_model_relation": "adapter",
        "base_model_runtime": BASE_TRAIN,
        "does_not_overwrite": FORBIDDEN_HUB,
        "forbidden_5050": FORBIDDEN_5050,
        "not_5050": True,
        "not_bf16_5050": True,
        "qlora": True,
        "load_in_4bit": True,
        "dataset_file": "chaski_r2/train.jsonl",
        "dataset_sha256": dataset_sha,
        "held_out_in_gradients": False,
        "held_out": {
            "chaski/gate/json_drafts.n5.jsonl": 5,
            "chaski/gate/adversarial_refusals.n6.jsonl": 6,
        },
        "seed": SEED,
        "num_train_epochs": NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LR,
        "lr_scheduler_type": "constant_with_warmup",
        "optim": "adamw_8bit",
        "response_only_loss": True,
        "trackio": False,
        "report_to": "none",
        "push_to_hub": False,
        "jobs": "UNAVAILABLE",
        "weights": "LOCAL" if adapter_sha else "UNAVAILABLE",
        "adapterSha256": adapter_sha or None,
        "finalTrainLoss": training_loss,
        "train_loss_label": "MEASURED" if training_loss else "UNAVAILABLE",
        "evals": "none-this-run",
        "quality": "UNAVAILABLE",
        "lambda": "Conjecture 1",
        "doctrine": "v11 LOCKED 749/14/163",
        "proposal_only": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "serve_pin": False,
        "khipu_lab_pin": False,
        "a11oy_mini_scripts_only": True,
        "hub_put": False,
        "training_rows": training_rows,
        "claim_boundary": (
            "ATELIER lock. Separate SKU id SZLHOLDINGS/chaski-r2 is declared "
            "only — do not costume a README-only Hub ID. Does not overwrite "
            f"{FORBIDDEN_HUB}. Not {FORBIDDEN_5050}. Base in prose: "
            f"{CANONICAL_BASE}. GPU honesty is MEASURED or UNAVAILABLE. "
            "No ROADMAP parking. Jobs this checkout UNAVAILABLE (not fired). "
            "Eval is PR 63 named-N after train; none-this-run until that "
            "generate. publication_eligible false until MEASURED generate. "
            "Lab stays Khipu. A11OY-MINI stays scripts-only. Doctrine v11. "
            "This checkout does not PUT Hub."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "local-train" if live else "forge-status",
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chaski-r2] wrote {path}")


def status_main(hub: str, dataset_file: Path | None) -> int:
    refuse_overwrite(hub)
    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-r2] refuse: canonical base drifted")
    rows, digest = load_train_rows(dataset_file)
    print(
        f"[chaski-r2] base={CANONICAL_BASE} canonical={CANONICAL_BASE} "
        f"runtime={BASE_TRAIN} hub={hub} seed={SEED}"
    )
    print("[chaski-r2] qlora r=16 alpha=32 response-only-CE")
    print("[chaski-r2] jobs=UNAVAILABLE weights=UNAVAILABLE quality=UNAVAILABLE")
    print("[chaski-r2] publication_eligible=false hub_put=false overwrite=false")
    print("[chaski-r2] ATELIER lock: declared Hub id only; no README-only costume")
    print("[chaski-r2] GPU honesty MEASURED or UNAVAILABLE; no ROADMAP parking")
    write_receipt(
        status_receipt(hub=hub, dataset_sha=digest, training_rows=len(rows)),
        STATUS_RECEIPT,
    )
    return 0


def train_main(hub: str, dataset_file: Path | None) -> int:
    refuse_overwrite(hub)
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    if CANONICAL_BASE != "Qwen/Qwen3.5-0.8B":
        raise SystemExit("[chaski-r2] refuse: canonical base drifted")
    if LORA_R != 16 or LORA_ALPHA != 32:
        raise SystemExit("[chaski-r2] refuse: owner pin is r=16 alpha=32")
    rows, digest = load_train_rows(dataset_file)
    print(
        f"[chaski-r2] train base={CANONICAL_BASE} runtime={BASE_TRAIN} "
        f"hub={hub} seed={SEED} r={LORA_R} alpha={LORA_ALPHA}"
    )
    print("[chaski-r2] push_to_hub=false; refusing Hub PUT")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_TRAIN,
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
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in rows
    ]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LR,
            warmup_steps=WARMUP_STEPS,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="constant_with_warmup",
            seed=SEED,
            output_dir=str(HERE / "outputs"),
            report_to="none",
            push_to_hub=False,
            save_strategy="no",
        ),
    )
    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            tokenizer=tokenizer,
        )
    except TypeError:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    print("[chaski-r2] training...")
    stats = trainer.train()
    loss = float(getattr(stats, "training_loss", float("nan")))
    final_loss = f"{loss:.4f}" if loss == loss else "UNAVAILABLE"
    print(f"[chaski-r2] train_loss MEASURED {final_loss} (train metric, not an eval)")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    adapter_sha = sha256_safetensors_dir(ADAPTER_DIR)
    print(f"[chaski-r2] local adapter {ADAPTER_DIR} sha256={adapter_sha}")
    print("[chaski-r2] Hub PUT skipped (this checkout never uploads)")

    receipt = status_receipt(
        hub=hub,
        dataset_sha=digest,
        live=True,
        training_loss=final_loss,
        adapter_sha=adapter_sha,
        training_rows=len(texts),
    )
    receipt["evals"] = "none-this-run"
    receipt["quality"] = "UNAVAILABLE"
    receipt["publication_eligible"] = False
    write_receipt(receipt, TRAIN_RECEIPT)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run Unsloth QLoRA locally. Default is status (no GPU, no Hub write).",
    )
    parser.add_argument(
        "--hub",
        default=os.environ.get("HUB_MODEL_ID", DEFAULT_HUB),
        help="Target Hub id. SZLHOLDINGS/chaski and chaski-5050 are refused.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Local jsonl only. Refuses chaski/gate/*.jsonl and estate JSON.",
    )
    args = parser.parse_args()
    refuse_overwrite(args.hub)
    if args.train:
        return train_main(args.hub, args.dataset_file)
    return status_main(args.hub, args.dataset_file)


if __name__ == "__main__":
    raise SystemExit(main())
