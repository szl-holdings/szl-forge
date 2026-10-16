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

CUTTING. Attempt 5 COMPLETED. Adapter and merged shard AVAILABLE on Hub.
Named-N MEASURED fail (json_draft 0/5, adversarial_refusal 2/6).
Train loss is not an eval. Do not invent 5/5. publication_eligible stays false.
Do not pin serve. Do not recut Hub from this checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# 6a91b8ba FAILED CastError
# 6a91b990 FAILED pyyaml 30s
# 6a91ba00 COMPLETED receipt-only, weights UNAVAILABLE, train_loss MEASURED 1.7827
# 6a91bb7c ERROR after 64/64, train_loss MEASURED 1.7844666938763112, merge ran, upload_folder Trackio 404, no safetensors
# 6a91bf1045686a1580c12105 COMPLETED report_to=none — weights AVAILABLE; Named-N MEASURED fail 0/5 and 2/6

MAX_SEQ_LEN = 2048
BASE = os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-0.8B")
TRAINING_BASE = "unsloth/Qwen3.5-0.8B"
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = os.environ.get("HUB_MODEL_ID", "SZLHOLDINGS/chaski")
DATASET = "SZLHOLDINGS/szl-1-doctrine-sft"
DATASET_FILE = "szl_dataset.jsonl"
FORBIDDEN_ESTATE = "SZL_ESTATE_MANAGED.json"
HELD_OUT_GATES = (
    (HERE / "gate" / "json_drafts.n5.jsonl", "chaski-json-draft-gate", 5),
    (HERE / "gate" / "adversarial_refusals.n6.jsonl", "chaski-adversarial-refusal-gate", 6),
)
IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
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
ATTEMPT5_LOSS = 1.783925924450159
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
WEIGHTS_STATUS = "AVAILABLE"
EVALS_STATUS = "MEASURED fail json_draft 0/5 adversarial_refusal 2/6"
EVAL_REPORT_SHA256 = (
    "4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05"
)
HUB_PARENT_REVISION = "1c55df8652e9d0f7b84356b1e2d54849165ae884"

# Five HF Jobs attempts. Status strings are bound to these ids.
# Attempt 3 COMPLETED (receipt-only). Attempt 4 ERROR (no safetensors).
# Attempt 5 COMPLETED (report_to=none). Hub card 2026-08-29: tensors
# AVAILABLE, Named-N MEASURED fail, publication_eligible false.
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
        "status": "COMPLETED",
        "detail": (
            "report_to=none. Hub tensors AVAILABLE: "
            + ", ".join(HUB_TENSORS)
            + f". Named-N MEASURED fail. eval_report.json sha256 {EVAL_REPORT_SHA256}. "
            "Train loss is not an eval. publication_eligible false."
        ),
        "url": LIVE_JOB_URL,
        "report_to": "none",
        "train_loss": ATTEMPT5_LOSS,
        "train_loss_label": "MEASURED",
        "weights": WEIGHTS_STATUS,
        "hub_tensors": HUB_TENSORS,
        "hub_tensors_observed_at": HUB_TENSORS_OBSERVED_AT,
        "parent_revision": HUB_PARENT_REVISION,
        "evals": EVALS_STATUS,
        "publication_eligible": False,
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


def require_immutable_revision(name: str, value: str | None) -> str:
    if not isinstance(value, str) or IMMUTABLE_REVISION.fullmatch(value) is None:
        raise SystemExit(f"[chaski] {name} must be an exact 40-character lowercase commit SHA")
    return value


def prompt_key(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit("[chaski] training/gate row has no messages")
    key: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise SystemExit("[chaski] malformed training/gate message")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise SystemExit("[chaski] malformed training/gate role or content")
        if role in {"system", "user"}:
            key.append((role, content))
    if not any(role == "user" for role, _ in key):
        raise SystemExit("[chaski] training/gate prompt has no user turn")
    return tuple(key)


def assert_held_out_disjoint(rows: list[dict[str, Any]]) -> None:
    """Reject exact prompt and user-turn reuse before model/trainer loading.

    This is a byte-exact prompt check, not a semantic-leakage certification.
    The named-N gate files are read only and never enter the training rows.
    """
    training_keys = [prompt_key(row) for row in rows]
    training = set(training_keys)
    if len(training) != len(rows):
        raise SystemExit("[chaski] duplicate training prompts")
    training_user_turns = {
        content
        for key in training_keys
        for role, content in key
        if role == "user"
    }
    for path, kind, expected_n in HELD_OUT_GATES:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != expected_n + 1:
            raise SystemExit(f"[chaski] held-out gate count drift: {path.name}")
        header = json.loads(lines[0])
        if (
            header.get("kind") != kind
            or type(header.get("n")) is not int
            or header["n"] != expected_n
            or header.get("artifact") != "SZLHOLDINGS/chaski"
            or header.get("base_model") != CANONICAL_BASE
            or header.get("publication_eligible") is not False
            or header.get("gate_ran") is not False
        ):
            raise SystemExit(f"[chaski] held-out gate header drift: {path.name}")
        for line in lines[1:]:
            gate_row = json.loads(line)
            if type(gate_row.get("n")) is not int or gate_row["n"] != expected_n:
                raise SystemExit(f"[chaski] held-out gate row count drift: {path.name}")
            gate_key = prompt_key(gate_row)
            if gate_key in training or any(
                content in training_user_turns
                for role, content in gate_key
                if role == "user"
            ):
                raise SystemExit(f"[chaski] held-out prompt in training rows: {path.name}")


def require_curriculum_gates(dataset_file: Path, expected_sha256: str) -> dict[str, str]:
    """Fail closed on schema and Nemo R1-R5 before training imports or weights."""
    tools_dir = HERE.parent / "tools"
    validator = runpy.run_path(str(tools_dir / "validate_sft_dataset.py"))
    schema_report = validator["validate_dataset"](str(dataset_file), min_examples=1)
    if (
        schema_report.get("status") != "VALID"
        or schema_report.get("dataset_sha256") != f"sha256:{expected_sha256}"
    ):
        raise SystemExit("[chaski] SFT dataset validation failed or changed")

    nemo = runpy.run_path(str(tools_dir / "nemo_doctrine_gate.py"))
    if nemo.get("_IMPORT_ERROR") is not None:
        raise SystemExit("[chaski] Nemo R1-R5 gate unavailable; training blocked")
    report = nemo["gate_dataset"](str(dataset_file), "finetuned")
    tip = report.get("receipt_chain_tip")
    if (
        report.get("status") != "VALID"
        or report.get("dataset_sha256") != f"sha256:{expected_sha256}"
        or report.get("checked") != schema_report.get("records")
        or report.get("skipped_no_chat_pair") != 0
        or not isinstance(tip, str)
        or not tip
    ):
        raise SystemExit("[chaski] Nemo R1-R5 gate failed or changed")
    return {
        "nemo_receipt_chain_tip": tip,
        "nemo_gate_sha256": sha256_bytes((tools_dir / "nemo_doctrine_gate.py").read_bytes()),
        "sft_validator_sha256": sha256_bytes((tools_dir / "validate_sft_dataset.py").read_bytes()),
    }


def load_doctrine_jsonl(
    dataset_file: Path | None = None, *, dataset_revision: str | None = None
) -> tuple[list[dict[str, Any]], str, Path]:
    """Load ONLY szl_dataset.jsonl. Never datasets.load_dataset on the repo."""
    if dataset_file is not None:
        refuse_estate_on_path(dataset_file)
        path = Path(dataset_file)
        raw = path.read_bytes()
    else:
        from huggingface_hub import hf_hub_download

        revision = require_immutable_revision("dataset revision", dataset_revision)
        downloaded = hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=revision,
        )
        snapshot_file = Path(downloaded)
        if (
            snapshot_file.name != DATASET_FILE
            or snapshot_file.parent.name != revision
            or snapshot_file.parent.parent.name != "snapshots"
            or snapshot_file.parent.parent.parent.name
            != "datasets--SZLHOLDINGS--szl-1-doctrine-sft"
        ):
            raise SystemExit("[chaski] Hub returned a nonmatching dataset snapshot")
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
    assert_held_out_disjoint(rows)
    print(
        f"[chaski] examples={len(rows)} doctrine_rows={len(doctrine_rows)} "
        f"sha256={doctrine_sha}"
    )
    return rows, doctrine_sha, path


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
        "training_loss": ATTEMPT5_LOSS,
        "label": "MEASURED",
        "evals": EVALS_STATUS,
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
            "Training completion is not evaluation. Named-N MEASURED fail "
            "(json_draft 0/5, adversarial_refusal 2/6) is not a pass. Do not "
            "claim 5/5 or 6/6. Weights AVAILABLE. Train loss is not an eval. "
            "CUTTING. publication_eligible false."
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
        f"[chaski] card=CUTTING weights={WEIGHTS_STATUS} evals={EVALS_STATUS}"
    )
    print(f"[chaski] attempt3 receipt already on Hub: {HUB_RECEIPT_URL}")
    print(f"[chaski] attempt5 COMPLETED report_to=none: {LIVE_JOB_URL}")
    for job in JOBS:
        print(f"[chaski] job {job['id']} {job['status']}")
    receipt = cutting_receipt(live=False)
    write_receipt(receipt, HERE / "training_receipt.status.json")
    return 0


def prepare_training_source(
    dataset_file: Path | None, *, base_revision: str | None, dataset_revision: str | None
) -> tuple[list[dict[str, Any]], str, Path, dict[str, str]]:
    """Require immutable Hub inputs and a clean holdout before training imports."""
    if dataset_file is not None:
        raise SystemExit("[chaski] --train requires the revision-pinned Hub dataset, not --dataset-file")
    if BASE != TRAINING_BASE:
        raise SystemExit(f"[chaski] base repository drifted from {TRAINING_BASE}")
    base_sha = require_immutable_revision("base revision", base_revision)
    dataset_sha = require_immutable_revision("dataset revision", dataset_revision)
    rows, doctrine_sha, doctrine_path = load_doctrine_jsonl(dataset_revision=dataset_sha)
    gate_evidence = require_curriculum_gates(doctrine_path, doctrine_sha)
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=TRAINING_BASE, revision=base_sha))
    if (
        snapshot.name != base_sha
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != "models--unsloth--Qwen3.5-0.8B"
    ):
        raise SystemExit("[chaski] Hub returned a nonmatching base snapshot")
    return rows, doctrine_sha, snapshot, gate_evidence


def train_main(
    dataset_file: Path | None, *, base_revision: str | None = None,
    dataset_revision: str | None = None
) -> int:
    rows, doctrine_sha, base_snapshot, gate_evidence = prepare_training_source(
        dataset_file, base_revision=base_revision, dataset_revision=dataset_revision
    )
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    print(
        f"[chaski] base={BASE} canonical={CANONICAL_BASE} hub={HUB} seed={SEED}"
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_snapshot),
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
    receipt["base_model_runtime"] = f"{TRAINING_BASE}@{base_revision}"
    receipt["base_model_revision"] = base_revision
    receipt["dataset_revision"] = dataset_revision
    receipt["dataset_sha256"] = doctrine_sha
    receipt.update(gate_evidence)
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
        help="Local status inspection only; --train requires a pinned Hub dataset.",
    )
    parser.add_argument("--base-revision", help="Immutable 40-character training-base commit SHA.")
    parser.add_argument("--dataset-revision", help="Immutable 40-character dataset commit SHA.")
    args = parser.parse_args()
    if args.train:
        return train_main(
            args.dataset_file,
            base_revision=args.base_revision,
            dataset_revision=args.dataset_revision,
        )
    if args.base_revision is not None or args.dataset_revision is not None:
        parser.error("revision pins are only used with --train")
    if args.dataset_file is not None:
        load_doctrine_jsonl(args.dataset_file)
    return status_main()


if __name__ == "__main__":
    raise SystemExit(main())
