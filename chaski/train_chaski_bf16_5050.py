#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "unsloth_zoo",
#     "trl",
#     "datasets",
#     "transformers",
#     "torch",
#     "trackio",
# ]
# ///
"""Local RTX 5050 Unsloth bf16 LoRA recut of Chaski.

Silhouette: Qwen/Qwen3.5-0.8B (Apache, disclosed base).
Cut: SZL doctrine SFT. Not a Qwen rehost. Not an Unsloth default card.

QLoRA is banned on Qwen3.5 (Unsloth 2026 docs: quantization differences).
This recipe is load_in_4bit=False, load_in_16bit=True, transformers v5.

Data: only SZLHOLDINGS/szl-1-doctrine-sft (41 chat JSONL examples).
That file is the admitted szl_dataset.jsonl. No scrape, no synthesized pairs.

Optional --push writes adapters to SZLHOLDINGS/chaski-5050 only.
Never overwrite SZLHOLDINGS/chaski.

Evals: none-this-run. publication_eligible: false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA_PATH = HERE / "training_receipt.bf16_5050.schema.json"
CARD_PATH = HERE / "HF_MODEL_CARD_5050.md"

CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
BASE = os.environ.get("BASE_MODEL", CANONICAL_BASE)
ALLOWED_HUB = "SZLHOLDINGS/chaski-5050"
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
DATASET_ID = "SZLHOLDINGS/szl-1-doctrine-sft"
DATASET_FILE = "szl_dataset.jsonl"
FORBIDDEN_ESTATE = "SZL_ESTATE_MANAGED.json"
ADMITTED_ROWS = 41
ADMITTED_SHA256 = (
    "ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243"
)

SEED = 11
NUM_TRAIN_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4
OPTIM = "adamw_8bit"
WARMUP_STEPS = 10
MAX_SEQ_LEN = 2048
LOAD_IN_4BIT = False
LOAD_IN_16BIT = True

TARGET_HARDWARE = (
    "NVIDIA GeForce RTX 5050 Laptop, 8GB GDDR7 (MEASURED 8151 MiB)"
)
VRAM_TABLE_REPORTED = (
    "0.8B bf16 LoRA ~3GB (REPORTED: Unsloth 2026 Qwen3.5 fine-tune table)"
)
QLORA_BAN = (
    "Unsloth 2026 Qwen3.5 docs: do NOT QLoRA Qwen3.5 (quantization "
    "differences). This recut is bf16 LoRA only: load_in_4bit=False, "
    "load_in_16bit=True."
)
FORBIDDEN_PRODUCT_HOST = "a11oy.com"
PRODUCT_HOST = "a-11-oy.com"
_URL_RE = re.compile(r"https?://[^\s)\]>'\"`]+", flags=re.I)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_bf16_loader(load_in_4bit: bool, load_in_16bit: bool) -> None:
    """Refuse QLoRA. Qwen3.5 recut is bf16 LoRA only."""
    if load_in_4bit is True:
        raise SystemExit(
            f"[chaski-5050] refuse: load_in_4bit=True. {QLORA_BAN}"
        )
    if load_in_16bit is not True:
        raise SystemExit(
            f"[chaski-5050] refuse: load_in_16bit must be True. {QLORA_BAN}"
        )


def assert_push_repo(repo_id: str) -> None:
    """Optional Hub write is SZLHOLDINGS/chaski-5050 only."""
    normalized = (repo_id or "").strip()
    if normalized == FORBIDDEN_HUB:
        raise SystemExit(
            "[chaski-5050] refuse: never overwrite SZLHOLDINGS/chaski. "
            "Push target is SZLHOLDINGS/chaski-5050 only."
        )
    if normalized != ALLOWED_HUB:
        raise SystemExit(
            f"[chaski-5050] refuse: hub id {normalized!r} is not "
            f"{ALLOWED_HUB}."
        )


def _yaml_tag_block(card: str) -> str:
    match = re.search(r"^---\n(.*?)\n---", card, flags=re.S)
    if not match:
        return ""
    return match.group(1)


def _yaml_tags(card: str) -> list[str]:
    block = _yaml_tag_block(card)
    tags: list[str] = []
    in_tags = False
    for line in block.splitlines():
        if line.startswith("tags:"):
            in_tags = True
            continue
        if in_tags:
            if line.startswith("  - "):
                tags.append(line[4:].strip().strip("'\""))
                continue
            if line and not line.startswith(" "):
                break
    return tags


def _card_hosts(card: str) -> list[str]:
    """Parse URL hosts from a card. Host compare is exact / DNS-parent, not substring."""
    hosts: list[str] = []
    for match in _URL_RE.finditer(card):
        host = (urlparse(match.group(0)).hostname or "").lower().rstrip(".")
        if host:
            hosts.append(host)
    return hosts


def _host_is(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def assert_house_card(card: str) -> None:
    """House fashion for a pushed adapter card. Process only; no Unsloth default."""
    hosts = _card_hosts(card)
    if any(_host_is(host, FORBIDDEN_PRODUCT_HOST) for host in hosts):
        raise SystemExit(
            "[chaski-5050] refuse: card URL host must not be the retired "
            "product host (use a-11-oy.com)."
        )
    tags = {tag.lower() for tag in _yaml_tags(card)}
    if "roadmap" in tags:
        raise SystemExit(
            "[chaski-5050] refuse: yaml tags must not include roadmap "
            "when adapter weights are pushed."
        )
    if "cutting" not in tags:
        raise SystemExit(
            "[chaski-5050] refuse: yaml tags must include cutting "
            "(or this recut is mislabeled)."
        )
    lowered = card.lower()
    if "adapter" not in lowered:
        raise SystemExit("[chaski-5050] refuse: card must say adapters")
    if "none-this-run" not in lowered:
        raise SystemExit("[chaski-5050] refuse: card must say evals none-this-run")
    if "not measured" not in lowered:
        raise SystemExit("[chaski-5050] refuse: card must say Not MEASURED")
    if "conjecture 1" not in lowered:
        raise SystemExit("[chaski-5050] refuse: card must keep Λ = Conjecture 1")
    if hosts and not any(_host_is(host, PRODUCT_HOST) for host in hosts):
        raise SystemExit(
            "[chaski-5050] refuse: product host URL must be a-11-oy.com"
        )


def house_model_card() -> str:
    if not CARD_PATH.is_file():
        raise SystemExit(f"[chaski-5050] refuse: missing house card {CARD_PATH}")
    text = CARD_PATH.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    assert_house_card(text)
    return text


def load_doctrine_rows(
    dataset_file: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load only the admitted 41-row jsonl. Never ingest estate JSON."""
    if dataset_file is not None and dataset_file.name == FORBIDDEN_ESTATE:
        raise SystemExit(
            f"[chaski-5050] refuse: {FORBIDDEN_ESTATE} is not the admitted jsonl."
        )
    if dataset_file is None:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=DATASET_FILE,
            )
            dataset_file = Path(downloaded)
        except Exception as exc:  # noqa: BLE001 - missing Hub file is a refuse
            raise SystemExit(
                f"[chaski-5050] refuse: missing dataset "
                f"{DATASET_ID}/{DATASET_FILE}: {exc}"
            ) from exc
    if not dataset_file.is_file():
        raise SystemExit(
            f"[chaski-5050] refuse: missing dataset {dataset_file}"
        )
    if dataset_file.name == FORBIDDEN_ESTATE:
        raise SystemExit(
            f"[chaski-5050] refuse: will not ingest {FORBIDDEN_ESTATE}"
        )
    raw = dataset_file.read_bytes()
    digest = sha256_bytes(raw)
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(
            f"[chaski-5050] refuse: missing dataset rows in {dataset_file}"
        )
    if any("messages" not in row for row in rows):
        raise SystemExit(
            f"[chaski-5050] refuse: {dataset_file} is not chat JSONL"
        )
    if len(rows) != ADMITTED_ROWS:
        raise SystemExit(
            f"[chaski-5050] refuse: {dataset_file} has {len(rows)} rows, "
            f"admitted set is {ADMITTED_ROWS}. No unaudited files."
        )
    if digest != ADMITTED_SHA256:
        raise SystemExit(
            f"[chaski-5050] refuse: {dataset_file} sha256 {digest} != "
            f"admitted {ADMITTED_SHA256}. No unaudited files."
        )
    print(
        f"[chaski-5050] dataset={DATASET_ID} file={DATASET_FILE} "
        f"rows={len(rows)} sha256={digest}"
    )
    return [{"messages": row["messages"]} for row in rows], digest


def observe_hardware() -> dict[str, Any]:
    observed: dict[str, Any] | None = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            observed = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_mib": int(props.total_memory // (1024 * 1024)),
            }
    except Exception as exc:  # noqa: BLE001 - hardware probe is optional
        observed = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return {
        "target": TARGET_HARDWARE,
        "observed": observed,
        "vram_table_08b_bf16_lora": VRAM_TABLE_REPORTED,
    }


def build_receipt(
    *,
    train_loss: float | None,
    dataset_sha256: str | None,
    training_rows: int | None = None,
    metrics: dict[str, Any] | None = None,
    hardware_observed: dict[str, Any] | None = None,
    adapter_dir: str | None = None,
    pushed: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "szl-chaski-5050-training-receipt",
        "schema": "szl.chaski-5050-training-receipt/v1",
        "artifact": ALLOWED_HUB,
        "base_model": CANONICAL_BASE,
        "base_model_relation": "adapter",
        "base_model_runtime": BASE,
        "method": "unsloth-bf16-lora",
        "transformers": "v5",
        "load_in_4bit": False,
        "load_in_16bit": True,
        "qlora": "banned-on-qwen3.5",
        "dataset": DATASET_ID,
        "dataset_file": DATASET_FILE,
        "dataset_sha256": dataset_sha256,
        "training_rows": training_rows if training_rows is not None else ADMITTED_ROWS,
        "seed": SEED,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "optim": OPTIM,
        "warmup_steps": WARMUP_STEPS,
        "max_seq_length": MAX_SEQ_LEN,
        "use_gradient_checkpointing": "unsloth",
        "train_loss": train_loss,
        "metrics": metrics or {},
        "hardware": TARGET_HARDWARE,
        "hardware_observed": hardware_observed,
        "vram_table_08b_bf16_lora": VRAM_TABLE_REPORTED,
        "evals": "none-this-run",
        "quality": "Not MEASURED",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "lambda": "Conjecture 1",
        "card_status": "CUTTING",
        "adapter_dir": adapter_dir,
        "pushed": pushed,
        "push_target": ALLOWED_HUB if pushed else None,
        "claim_boundary": (
            "Training completion is not evaluation. Evals none-this-run. "
            "Train loss is a train metric, not an eval. Not MEASURED. "
            "Do not invent 5/5. QLoRA is banned on this family."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[chaski-5050] wrote {path}")


def _sft_trainer(model: Any, tokenizer: Any, dataset: Any, output_dir: Path) -> Any:
    """transformers v5 / current TRL: processing_class + SFTConfig.max_length."""
    from trl import SFTConfig, SFTTrainer

    config_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "logging_steps": 1,
        "optim": OPTIM,
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": SEED,
        "output_dir": str(output_dir),
        "report_to": "none",
        "dataset_text_field": "text",
        "max_length": MAX_SEQ_LEN,
    }
    args = SFTConfig(**config_kwargs)
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=args,
    )


def train(
    *,
    dataset_file: Path | None,
    push: bool,
    hub_model_id: str,
    output_dir: Path,
) -> int:
    assert_bf16_loader(LOAD_IN_4BIT, LOAD_IN_16BIT)
    if push:
        assert_push_repo(hub_model_id)
    rows, doctrine_sha = load_doctrine_rows(dataset_file)

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    print(
        f"[chaski-5050] base={BASE} canonical={CANONICAL_BASE} "
        f"hub={hub_model_id} seed={SEED}"
    )
    print(f"[chaski-5050] hardware target: {TARGET_HARDWARE}")
    print(f"[chaski-5050] VRAM table: {VRAM_TABLE_REPORTED}")
    print(f"[chaski-5050] {QLORA_BAN}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_16bit=LOAD_IN_16BIT,
        full_finetuning=False,
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
    dataset = Dataset.from_dict({"text": texts})
    checkpoints = output_dir / "outputs"
    trainer = _sft_trainer(model, tokenizer, dataset, checkpoints)
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
    print(f"[chaski-5050] train done train_loss={loss} (train metric, not an eval)")
    adapter_dir = output_dir / "chaski-5050-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "README.md").write_text(house_model_card(), encoding="utf-8")
    hardware = observe_hardware()
    receipt_path = output_dir / "training_receipt.json"
    receipt = build_receipt(
        train_loss=loss,
        dataset_sha256=doctrine_sha,
        training_rows=len(rows),
        metrics=metrics,
        hardware_observed=hardware,
        adapter_dir=str(adapter_dir),
        pushed=False,
    )
    write_receipt(receipt, receipt_path)
    if push:
        push_adapters(adapter_dir, hub_model_id, receipt_path=receipt_path)
        receipt["pushed"] = True
        receipt["push_target"] = hub_model_id
        write_receipt(receipt, receipt_path)
    return 0


def push_adapters(
    adapter_dir: Path,
    hub_model_id: str,
    receipt_path: Path | None = None,
) -> bool:
    assert_push_repo(hub_model_id)
    assert_house_card((adapter_dir / "README.md").read_text(encoding="utf-8"))
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=hub_model_id, repo_type="model", exist_ok=True, private=False
    )
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=hub_model_id,
        repo_type="model",
        commit_message=(
            "feat(adapter): local RTX 5050 Unsloth bf16 LoRA recut "
            "(QLoRA banned on Qwen3.5)"
        ),
    )
    if receipt_path is not None and receipt_path.is_file():
        api.upload_file(
            path_or_fileobj=str(receipt_path),
            path_in_repo="training_receipt.json",
            repo_id=hub_model_id,
            repo_type="model",
            commit_message=(
                "chore(receipt): 5050 bf16 LoRA recut (evals none-this-run)"
            ),
        )
    print(f"[chaski-5050] pushed adapters to {hub_model_id}")
    return True


def print_plan() -> None:
    print("[chaski-5050] local RTX 5050 bf16 LoRA recut (no train this invocation)")
    print(f"[chaski-5050] silhouette={CANONICAL_BASE} method=unsloth-bf16-lora")
    print(f"[chaski-5050] load_in_4bit={LOAD_IN_4BIT} load_in_16bit={LOAD_IN_16BIT}")
    print(f"[chaski-5050] QLoRA banned on this family")
    print(f"[chaski-5050] hardware={TARGET_HARDWARE}")
    print(f"[chaski-5050] VRAM table={VRAM_TABLE_REPORTED}")
    print(f"[chaski-5050] dataset={DATASET_ID} rows={ADMITTED_ROWS}")
    print(f"[chaski-5050] hub={ALLOWED_HUB} (never {FORBIDDEN_HUB})")
    print("[chaski-5050] evals=none-this-run publication_eligible=false")
    print("[chaski-5050] card=CUTTING quality=Not MEASURED lambda=Conjecture 1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run Unsloth bf16 LoRA on the 5050. Default is plan only (no GPU).",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Local admitted szl_dataset.jsonl. Refuses missing / unaudited files.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help=f"Upload adapters to {ALLOWED_HUB} only.",
    )
    parser.add_argument(
        "--hub-model-id",
        default=os.environ.get("HUB_MODEL_ID", ALLOWED_HUB),
        help=f"Must be {ALLOWED_HUB}. Never {FORBIDDEN_HUB}.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Forbidden. Present so the recut can refuse QLoRA loudly.",
    )
    parser.add_argument(
        "--load-in-16bit",
        dest="load_in_16bit",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-load-in-16bit",
        dest="load_in_16bit",
        action="store_false",
        help="Forbidden. bf16 LoRA requires load_in_16bit=True.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
    )
    args = parser.parse_args(argv)

    assert_bf16_loader(args.load_in_4bit, args.load_in_16bit)
    if args.hub_model_id.strip() == FORBIDDEN_HUB or args.push:
        assert_push_repo(args.hub_model_id)
    if args.dataset_file is not None:
        load_doctrine_rows(args.dataset_file)
    if not args.train:
        print_plan()
        return 0
    if args.push:
        house_model_card()
    return train(
        dataset_file=args.dataset_file,
        push=args.push,
        hub_model_id=args.hub_model_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
