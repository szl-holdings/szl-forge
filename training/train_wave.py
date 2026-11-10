"""Single-GPU SFT training wave for SZL Forge models (Unsloth backend).

The launch document is ``--config``. ``training/launch_contract.py`` evaluates
rights, immutable student and dataset pins, train-versus-holdout separation,
and the 32-example floor before this process imports a model stack, reads
``HF_TOKEN``, or writes to the Hub. Explicit ``--model`` / ``--datasets``
arguments are not a second launch path.

Usage:
  python train_wave.py --config training/w30/distill-r2.blocked.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import NoReturn


def fail_closed(message: str) -> NoReturn:
    print(f"FAIL-CLOSED: {message}", file=sys.stderr)
    raise SystemExit(1)


def _contract():
    path = Path(__file__).resolve().parent / "launch_contract.py"
    spec = importlib.util.spec_from_file_location("szl_launch_contract", path)
    if spec is None or spec.loader is None:
        fail_closed(f"LAUNCH_CONFIG_UNREADABLE: missing {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Wave config evaluated before any launch")
    parser.add_argument("--model", default=None, help="Rejected when it disagrees with the config pin")
    parser.add_argument("--datasets", default=None, help="Rejected when it disagrees with the config pin")
    parser.add_argument("--method", choices=("qlora", "lora16"), default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-repo-id", default=None)
    return parser.parse_args(argv)


def _effective_training_values(config: dict) -> dict:
    """Values the trainer will use. CLI flags must match these or be absent."""
    try:
        values = {
            "epochs": float(config.get("epochs", 3.0)),
            "lr": float(config.get("lr", 2e-4)),
            "seqlen": int(config.get("seqlen", 4096)),
            "batch_size": int(config.get("batch_size", 2)),
            "grad_accum": int(config.get("grad_accum", 8)),
            "seed": int(config.get("seed", 3407)),
        }
    except (TypeError, ValueError) as exc:
        fail_closed(f"LAUNCH_CONFIG_INVALID: hyperparameters are not numeric ({exc})")
    pack = config.get("pack", False)
    if not isinstance(pack, bool):
        fail_closed("LAUNCH_CONFIG_INVALID: pack must be a boolean")
    values["pack"] = pack
    return values


def _hyperparameters(config: dict) -> dict:
    values = _effective_training_values(config)
    out = config.get("out")
    if not isinstance(out, str) or not out.strip():
        fail_closed("LAUNCH_OUTPUT_MISSING: config out directory is required")
    values.update({
        "out": Path(out),
        "method": config.get("method"),
        "repo_id": config.get("target_repo") or None,
        "log_repo_id": config.get("log_repo_id") or None,
    })
    return values


def _refuse_cli_overrides(args: argparse.Namespace, decision: dict) -> None:
    if args.max_samples is not None:
        fail_closed(
            "LAUNCH_SAMPLE_CAP_FORBIDDEN: --max-samples cannot lower the corpus floor"
        )
    config = decision["config"]
    if args.model is not None and args.model != decision["student_id"]:
        fail_closed("LAUNCH_CLI_OVERRIDE: --model does not match the student pin")
    if args.datasets is not None and args.datasets != decision["dataset_id"]:
        fail_closed("LAUNCH_CLI_OVERRIDE: --datasets does not match the dataset pin")
    if args.method is not None and args.method != config.get("method"):
        fail_closed("LAUNCH_CLI_OVERRIDE: --method does not match the config")
    if args.repo_id is not None and args.repo_id != config.get("target_repo"):
        fail_closed("LAUNCH_CLI_OVERRIDE: --repo-id does not match target_repo")
    if args.log_repo_id is not None and args.log_repo_id != config.get("log_repo_id"):
        fail_closed("LAUNCH_CLI_OVERRIDE: --log-repo-id does not match the config")
    if args.out is not None and str(args.out) != str(config.get("out")):
        fail_closed("LAUNCH_CLI_OVERRIDE: --out does not match the config")
    effective = _effective_training_values(config)
    explicit_flags = (
        ("--epochs", args.epochs, "epochs", float),
        ("--lr", args.lr, "lr", float),
        ("--seqlen", args.seqlen, "seqlen", int),
        ("--batch-size", args.batch_size, "batch_size", int),
        ("--grad-accum", args.grad_accum, "grad_accum", int),
        ("--seed", args.seed, "seed", int),
    )
    for flag, given, key, caster in explicit_flags:
        if given is None:
            continue
        if caster(given) != effective[key]:
            fail_closed(f"LAUNCH_CLI_OVERRIDE: {flag} does not match the validated config")
    if args.pack and effective["pack"] is not True:
        fail_closed("LAUNCH_CLI_OVERRIDE: --pack does not match the validated config")


def _train_after_gates(decision: dict, config_path: Path) -> None:
    """Import the training stack only after a second local contract check."""
    recheck = _contract().evaluate_path(config_path, expected_kind="sft")
    if not recheck["ok"]:
        fail_closed("LAUNCH_CONTRACT_REJECTED: " + ",".join(recheck["codes"]))
    if (
        recheck["corpus_sha256"] != decision["corpus_sha256"]
        or recheck["student_revision"] != decision["student_revision"]
        or recheck["dataset_revision"] != decision["dataset_revision"]
        or recheck["accepted_examples"] < _contract().MIN_ACCEPTED_EXAMPLES
    ):
        fail_closed("LAUNCH_ADMISSION_MISMATCH: pinned corpus changed after the contract")

    args = _hyperparameters(decision["config"])
    if not os.environ.get("HF_TOKEN"):
        fail_closed("HF_TOKEN is not set; dataset access and hub push require it.")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        fail_closed("unsloth is not installed. pip install unsloth")

    import torch

    if not torch.cuda.is_available():
        fail_closed("CUDA is not available; this trainer requires one GPU.")

    dtype = None
    load_in_4bit = args["method"] == "qlora"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=decision["student_id"],
        revision=decision["student_revision"],
        max_seq_length=args["seqlen"],
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    from peft import PeftModel
    already_peft = isinstance(model, PeftModel)
    if already_peft:
        print("loaded model already carries LoRA adapters; skipping get_peft_model")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args["seed"],
        )

    from datasets import load_dataset

    print(f"loading pinned local corpus: {decision['train_jsonl']}")
    frame = load_dataset("json", data_files=decision["train_jsonl"], split="train")
    keep = [c for c in ("messages", "conversations") if c in frame.column_names]
    if keep:
        frame = frame.select_columns(keep)
    data = frame
    print(f"merged corpus: {len(data)} examples")

    def add_text(example):
        messages = example.get("messages") or example.get("conversations")
        example["text"] = (
            tokenizer.apply_chat_template(messages, tokenize=False) if messages else "")
        return example

    before = len(data)
    data = data.map(add_text, desc="applying chat template")
    data = data.filter(lambda e: e["text"], desc="dropping non-chat rows")
    dropped = before - len(data)
    if dropped:
        print(f"dropped {dropped} non-chat row(s) (spec/manifest metadata without messages/conversations)")
    if len(data) < _contract().MIN_ACCEPTED_EXAMPLES:
        fail_closed(
            f"LAUNCH_CORPUS_FLOOR: {len(data)} chat examples remain; "
            f"launch requires {_contract().MIN_ACCEPTED_EXAMPLES}"
        )

    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            output_dir=str(args["out"]),
            per_device_train_batch_size=args["batch_size"],
            gradient_accumulation_steps=args["grad_accum"],
            num_train_epochs=args["epochs"],
            learning_rate=args["lr"],
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            bf16=True,
            packing=args["pack"],
            max_length=args["seqlen"],
            seed=args["seed"],
            dataset_text_field="text",
            report_to=[],
        ),
    )

    import threading

    def _upload_log_tail(api, tag: str) -> None:
        try:
            log_path = Path(os.environ.get("SZL_TRAIN_LOG", "/opt/szl/train.log"))
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                api.upload_file(
                    path_or_fileobj="\n".join(lines).encode(),
                    path_in_repo=f"{Path(args['out']).name}-train.log",
                    repo_id=args["log_repo_id"],
                    repo_type="dataset",
                    commit_message=f"chore: wave log {tag}",
                )
        except Exception as exc:
            print(f"log-forward skipped ({tag}): {type(exc).__name__}", file=sys.stderr)

    def _forward_logs() -> None:
        import time as _time
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        while True:
            _time.sleep(600)
            _upload_log_tail(api, "heartbeat")

    if args["log_repo_id"]:
        threading.Thread(target=_forward_logs, daemon=True).start()

    print("training start")
    trainer.train()
    trainer.save_model(str(args["out"]))
    tokenizer.save_pretrained(str(args["out"]))

    receipt = {
        "model": decision["student_id"],
        "student_revision": decision["student_revision"],
        "dataset": decision["dataset_id"],
        "dataset_revision": decision["dataset_revision"],
        "corpus_sha256": decision["corpus_sha256"],
        "accepted_examples": decision["accepted_examples"],
        "train_jsonl": decision["train_jsonl"],
        "method": args["method"],
        "epochs": args["epochs"],
        "lr": args["lr"],
        "seqlen": args["seqlen"],
        "pack": args["pack"],
        "seed": args["seed"],
        "steps": int(trainer.state.global_step),
        "final_loss": float(trainer.state.log_history[-1].get("train_loss", float("nan")))
        if trainer.state.log_history else None,
        "launch_contract": _contract().SCHEMA,
        "training_authorization": "THIS_RECEIPT_IS_A_RUN_RECORD_NOT_A_PROMOTION",
    }
    receipt_path = Path(args["out"]) / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt written: {receipt_path}")

    if args["log_repo_id"]:
        from huggingface_hub import HfApi
        _upload_log_tail(HfApi(token=os.environ["HF_TOKEN"]), "final")

    if args["repo_id"]:
        fail_closed(
            "DIRECT_HUB_PUBLISH_DISABLED: export artifacts and use "
            ".github/workflows/model-publish-gate.yml"
        )

    print("wave complete")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.config is None:
        fail_closed(
            "LAUNCH_CONFIG_REQUIRED: training launch takes --config. "
            "The shared launch contract runs before any model load."
        )
    decision = _contract().evaluate_path(args.config, expected_kind="sft")
    if not decision["ok"]:
        fail_closed("LAUNCH_CONTRACT_REJECTED: " + ",".join(decision["codes"]))
    _refuse_cli_overrides(args, decision)
    _train_after_gates(decision, args.config)


if __name__ == "__main__":
    main()
