#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "datasets",
#     "transformers",
#     "jsonschema",
# ]
# ///
"""KHIPU-R2 Unsloth QLoRA abstain-retrain kit. Separate SKU. No Hub PUT.

Live Hub SZLHOLDINGS/KHIPU-R2 is not empty: job 6a91bf11984507d9db4ea104
COMPLETED, adapter 147.8MB AVAILABLE, eval_measured.json abstain MEASURED 3/6
(not a pass; grounding 5/5, plan 11/11). This checkout does not fire a job
(jobs UNKNOWN for this kit) and does not re-run held-out generate
(this-SKU evals not-this-run). Does NOT overwrite signed SZL-Khipu-1.5B.
Signed 1.5B abstain stays MEASURED 2/6 on that card only.

Default is status (no GPU, no job fire, no Hub write). Pass --train to run
Unsloth locally. Hub's leftover doctrine-SFT train_khipu_r2.py is not a
second forge recipe; this file is the one trainer (khipu curriculum,
ABSTAIN_OVERSAMPLE=4, r=32 α=64, 45 epochs, seed 11).

publication_eligible is false. Doctrine v11 LOCKED. Λ = Conjecture 1.
House CPU lab stays signed Khipu GGUF.

CHAWPI extra lock: Hub receipt ddf6c50 publication_eligible false is the
public claim. stale profile key dropped. Launcher still no --run-job.
r=32 α=64 this SKU. Signed 1.5B stays 2/6. No Hub PUT. Do not merge #64.
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
KHIPU = ROOT / "khipu"

BASE_TRAIN = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
BASE_CANONICAL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_HUB = "SZLHOLDINGS/KHIPU-R2"
FORBIDDEN_HUB = "SZLHOLDINGS/SZL-Khipu-1.5B"
MAX_SEQ_LEN = 2048
SEED = 11
LORA_R = 32
LORA_ALPHA = 64
LR = 2e-4
NUM_EPOCHS = 45
ABSTAIN_OVERSAMPLE = 4
WARMUP_STEPS = 10
SIGNED_ABSTAIN_CORRECT = 2
SIGNED_ABSTAIN_TOTAL = 6

HUB_JOB_ID = "6a91bf11984507d9db4ea104"
HUB_JOB_STATUS = "COMPLETED"
HUB_JOB_URL = f"https://huggingface.co/jobs/SZLHOLDINGS/{HUB_JOB_ID}"
HUB_ADAPTER_STATUS = "AVAILABLE"
HUB_ADAPTER_SIZE = "147.8MB"
HUB_ADAPTER_FILE = "adapter_model.safetensors"
HUB_ABSTAIN_CORRECT = 3
HUB_ABSTAIN_TOTAL = 6
HUB_ABSTAIN_LABEL = "MEASURED"
HUB_GROUNDING_CORRECT = 5
HUB_GROUNDING_TOTAL = 5
HUB_PLAN_VALID = 11
HUB_PLAN_TOTAL = 11
HUB_RECEIPT_COMMIT = "ddf6c50d8baa9f818b9f478086e7b5919eb773cf"
CHAWPI = "hub-receipt-ddf6c50-publication-eligible-false"
# CHAWPI extra lock: Hub receipt ddf6c50d8baa9f818b9f478086e7b5919eb773cf
# publication_eligible false is the public claim.
# Stamp job 6a91bf11984507d9db4ea104 COMPLETED.
# eval_measured.json abstain 3/6 (grounding 5/5, plan 11/11).
# stale profile key dropped. Launcher still no --run-job.
# r=32 α=64 this SKU. Signed 1.5B stays 2/6. No Hub PUT.
# Do not merge #64.

CURRICULUM_FILES = [
    "train.jsonl",
    "eval.jsonl",
    "train.abstain.jsonl",
    "adversarial.jsonl",
    "khipu.schema.json",
]
TRAIN_FILES = ["train.jsonl", "train.abstain.jsonl"]
ADAPTER_DIR = HERE / "khipu-r2-adapter"
STATUS_RECEIPT = HERE / "training_receipt.status.json"
TRAIN_RECEIPT = HERE / "training_receipt.json"


def refuse_overwrite(hub: str) -> None:
    """Never push to, or retarget, the signed SZL-Khipu-1.5B family."""
    normalized = hub.strip().rstrip("/")
    if normalized.upper() == FORBIDDEN_HUB.upper() or normalized.upper().startswith(
        FORBIDDEN_HUB.upper() + "/"
    ):
        raise SystemExit(
            f"[khipu-r2] refusing to push to {FORBIDDEN_HUB}. "
            "KHIPU-R2 is a separate SKU. Signed 1.5B is never overwritten."
        )
    if "SZL-Khipu-1.5B" in normalized:
        raise SystemExit(
            f"[khipu-r2] refusing hub {hub!r}: signed SZL-Khipu-1.5B family "
            "is never an overwrite target."
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


def load_manifest() -> dict[str, Any]:
    path = KHIPU / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"[khipu-r2] missing khipu curriculum manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_curriculum(manifest: dict[str, Any]) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for name in CURRICULUM_FILES:
        path = KHIPU / name
        if not path.is_file():
            raise SystemExit(f"[khipu-r2] missing curriculum file: {path}")
        digest = sha256_file(path)
        declared = manifest.get("files", {}).get(name, {}).get("sha256")
        if declared != digest:
            raise SystemExit(
                f"[khipu-r2] {name} sha256 {digest} != manifest {declared}. "
                "Refusing to train on a drifted khipu curriculum."
            )
        datasets[name] = digest
    return datasets


def load_jsonl(name: str) -> list[dict[str, Any]]:
    rows = []
    with (KHIPU / name).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def signed_abstain() -> tuple[int, int]:
    receipt = KHIPU / "eval_receipt.signed.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))["payload"]
    correct = int(payload["abstainCorrect"])
    total = int(payload["abstainTotal"])
    if (correct, total) != (SIGNED_ABSTAIN_CORRECT, SIGNED_ABSTAIN_TOTAL):
        raise SystemExit(
            f"[khipu-r2] signed khipu abstain is {correct}/{total}, "
            f"expected {SIGNED_ABSTAIN_CORRECT}/{SIGNED_ABSTAIN_TOTAL}"
        )
    return correct, total


def status_receipt(
    *,
    hub: str,
    datasets: dict[str, str],
    live: bool = False,
    training_loss: str | None = None,
    adapter_sha: str = "",
    training_rows: int | None = None,
) -> dict[str, Any]:
    abstain_correct, abstain_total = signed_abstain()
    contract = load_manifest()["contract"]
    return {
        "kind": "szl-khipu-r2-training-receipt",
        "schema": "szl.frontier-training-run/v1",
        "v": 1,
        "artifact": hub,
        "sku": "KHIPU-R2",
        "separate_sku": True,
        "base_model": BASE_CANONICAL,
        "baseModel": BASE_CANONICAL,
        "base_model_relation": "adapter",
        "base_model_runtime": BASE_TRAIN,
        "does_not_overwrite": FORBIDDEN_HUB,
        "lab": "signed Khipu GGUF",
        "inference_lab_pin": False,
        "datasets": datasets,
        "schemaFingerprintSha256": contract["schemaFingerprintSha256"],
        "outputSchemaSha256": contract["outputSchemaSha256"],
        "ABSTAIN_OVERSAMPLE": ABSTAIN_OVERSAMPLE,
        "train_navigate_rows": 15,
        "train_abstain_rows_committed": 8,
        "train_abstain_rows_in_memory": 8 * ABSTAIN_OVERSAMPLE,
        "training_rows_in_memory": training_rows
        if training_rows is not None
        else 15 + 8 * ABSTAIN_OVERSAMPLE,
        "held_out_in_gradients": False,
        "held_out": {"eval.jsonl": 5, "adversarial.jsonl": 6},
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
        "hub_job_id": HUB_JOB_ID,
        "hub_job_status": HUB_JOB_STATUS,
        "hub_job_url": HUB_JOB_URL,
        "hub_adapter": HUB_ADAPTER_STATUS,
        "hub_adapter_file": HUB_ADAPTER_FILE,
        "hub_adapter_size": HUB_ADAPTER_SIZE,
        "hub_abstain": f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL}",
        "hub_abstain_correct": HUB_ABSTAIN_CORRECT,
        "hub_abstain_total": HUB_ABSTAIN_TOTAL,
        "hub_abstain_label": HUB_ABSTAIN_LABEL,
        "hub_abstain_pass": False,
        "hub_grounding": f"{HUB_GROUNDING_CORRECT}/{HUB_GROUNDING_TOTAL}",
        "hub_grounding_correct": HUB_GROUNDING_CORRECT,
        "hub_grounding_total": HUB_GROUNDING_TOTAL,
        "hub_plan": f"{HUB_PLAN_VALID}/{HUB_PLAN_TOTAL}",
        "hub_plan_valid": HUB_PLAN_VALID,
        "hub_plan_total": HUB_PLAN_TOTAL,
        "hub_receipt_commit": HUB_RECEIPT_COMMIT,
        "chawpi": CHAWPI,
        "signed_original_abstain": f"{abstain_correct}/{abstain_total}",
        "signed_original_abstain_correct": abstain_correct,
        "signed_original_abstain_total": abstain_total,
        "jobs": "UNKNOWN",
        "jobs_scope": "this-kit",
        "local_adapter": "LOCAL" if adapter_sha else "UNAVAILABLE",
        "adapterSha256": adapter_sha or None,
        "finalTrainLoss": training_loss,
        "evals": "not-this-run",
        "evals_scope": "this-sku",
        "lambda": "Conjecture 1",
        "doctrine": "v11 LOCKED 749/14/163",
        "proposal_only": True,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "serve_pin": False,
        "hub_put": False,
        "claim_boundary": (
            "Live Hub KHIPU-R2 is not empty: job "
            f"{HUB_JOB_ID} {HUB_JOB_STATUS}, adapter {HUB_ADAPTER_SIZE} "
            f"{HUB_ADAPTER_STATUS}, abstain {HUB_ABSTAIN_LABEL} "
            f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL} (not a pass), "
            f"grounding {HUB_GROUNDING_CORRECT}/{HUB_GROUNDING_TOTAL}, "
            f"plan {HUB_PLAN_VALID}/{HUB_PLAN_TOTAL}. "
            f"Signed SZL-Khipu-1.5B abstain stays MEASURED "
            f"{abstain_correct}/{abstain_total} on that card only. "
            "This-kit jobs UNKNOWN. This-SKU evals not-this-run. "
            "Does not overwrite signed SZL-Khipu-1.5B. No Hub PUT. "
            "CHAWPI extra lock: Hub receipt ddf6c50 publication_eligible false "
            "is the public claim. stale profile key dropped. "
            "Launcher still no --run-job. r=32 α=64 this SKU. "
            "Do not merge #64. Lab stays signed Khipu GGUF."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat() if live else None,
        "source": "local-train" if live else "forge-status",
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[khipu-r2] wrote {path}")


def load_train_texts(tokenizer) -> list[str]:
    rows: list[dict[str, Any]] = []
    for name in TRAIN_FILES:
        reps = ABSTAIN_OVERSAMPLE if name == "train.abstain.jsonl" else 1
        file_rows = load_jsonl(name)
        for _ in range(reps):
            rows.extend(file_rows)
        print(f"[khipu-r2]   {name}: {len(file_rows)} rows x{reps}")
    print(
        f"[khipu-r2] {len(rows)} training rows total "
        f"(abstain oversampled x{ABSTAIN_OVERSAMPLE}; held-out never in gradients)"
    )
    return [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in rows
    ]


def status_main(hub: str) -> int:
    refuse_overwrite(hub)
    manifest = load_manifest()
    datasets = verify_curriculum(manifest)
    abstain_correct, abstain_total = signed_abstain()
    print(
        f"[khipu-r2] base={BASE_CANONICAL} runtime={BASE_TRAIN} "
        f"hub={hub} seed={SEED}"
    )
    print(
        f"[khipu-r2] hub_job={HUB_JOB_ID} {HUB_JOB_STATUS} "
        f"adapter={HUB_ADAPTER_STATUS} ({HUB_ADAPTER_SIZE})"
    )
    print(
        f"[khipu-r2] hub_abstain={HUB_ABSTAIN_LABEL} "
        f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL} (not a pass) "
        f"grounding={HUB_GROUNDING_CORRECT}/{HUB_GROUNDING_TOTAL} "
        f"plan={HUB_PLAN_VALID}/{HUB_PLAN_TOTAL} "
        "publication_eligible=false"
    )
    print(
        f"[khipu-r2] CHAWPI hub_receipt={HUB_RECEIPT_COMMIT} "
        "publication_eligible=false is the public claim"
    )
    print(
        f"[khipu-r2] signed original abstain MEASURED "
        f"{abstain_correct}/{abstain_total} (signed 1.5B card only)"
    )
    print(
        "[khipu-r2] this-kit jobs=UNKNOWN this-sku evals=not-this-run "
        "hub_put=false overwrite=false"
    )
    print(f"[khipu-r2] curriculum verified ({len(datasets)} files)")
    write_receipt(status_receipt(hub=hub, datasets=datasets), STATUS_RECEIPT)
    return 0


def train_main(hub: str) -> int:
    refuse_overwrite(hub)
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    manifest = load_manifest()
    datasets = verify_curriculum(manifest)
    print(
        f"[khipu-r2] train base={BASE_CANONICAL} runtime={BASE_TRAIN} "
        f"hub={hub} seed={SEED} oversample={ABSTAIN_OVERSAMPLE}"
    )
    print("[khipu-r2] push_to_hub=false; refusing Hub PUT")

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
    texts = load_train_texts(tokenizer)
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
    print("[khipu-r2] training...")
    stats = trainer.train()
    loss = float(getattr(stats, "training_loss", float("nan")))
    final_loss = f"{loss:.4f}" if loss == loss else "UNKNOWN"
    print(f"[khipu-r2] final loss (REPORTED verbatim): {final_loss}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    adapter_sha = sha256_safetensors_dir(ADAPTER_DIR)
    print(f"[khipu-r2] local adapter {ADAPTER_DIR} sha256={adapter_sha}")
    print("[khipu-r2] Hub PUT skipped (this checkout never uploads)")

    receipt = status_receipt(
        hub=hub,
        datasets=datasets,
        live=True,
        training_loss=final_loss,
        adapter_sha=adapter_sha,
        training_rows=len(texts),
    )
    receipt["evals"] = "not-this-run"
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
        help="Target Hub id. SZLHOLDINGS/SZL-Khipu-1.5B is always refused.",
    )
    args = parser.parse_args()
    refuse_overwrite(args.hub)
    if args.train:
        return train_main(args.hub)
    return status_main(args.hub)


if __name__ == "__main__":
    raise SystemExit(main())
