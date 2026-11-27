"""Qwen3.5 LoRA and QLoRA SFT on the pinned image stack.

``train_wave.py`` routes a Qwen3.5 student id here after the launch contract.
The dense Qwen2/Qwen3 Unsloth path is unchanged.  This module never reads a
token or launches a hosted job.  ``enable_thinking`` is a required recipe field
and is applied before TRL sees the rows.
"""

from __future__ import annotations

import json
from pathlib import Path


def qwen35_sft_config(sft_config_cls, values: dict, output_dir: str, precision: dict):
    """SFTConfig for TRL 1.17 (``max_length``, ``processing_class`` on the trainer)."""

    return sft_config_cls(
        output_dir=str(output_dir),
        per_device_train_batch_size=values["batch_size"],
        gradient_accumulation_steps=values["grad_accum"],
        num_train_epochs=values["epochs"],
        learning_rate=values["lr"],
        logging_steps=values.get("logging_steps", 10),
        save_steps=values.get("save_steps", 200),
        save_strategy=values.get("save_strategy", "steps"),
        save_total_limit=3,
        max_steps=values.get("max_steps", -1),
        packing=values["pack"],
        max_length=values["seqlen"],
        seed=values["seed"],
        dataset_text_field="text",
        report_to=[],
        gradient_checkpointing=values.get("gradient_checkpointing", True),
        **precision,
    )


def build_qwen35_trainer(
    model,
    tokenizer,
    data,
    *,
    compat,
    reasoning: dict,
    values: dict,
    output_dir: str,
    precision: dict,
):
    """Qwen3.5 SFT shared by a GPU wave.  Failures raise."""

    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    compat.require_padding_token(tokenizer)
    targets, evidence = compat.lora_target_modules(model)
    if evidence["family"] != "qwen3_5":
        raise RuntimeError("Qwen3.5 base did not resolve Qwen3.5 LoRA targets")
    model = get_peft_model(
        model,
        LoraConfig(
            r=values.get("lora_r", 16),
            lora_alpha=values.get("lora_alpha", 16),
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    data = compat.render_sft_rows(data, tokenizer, reasoning)
    if "messages" in data.column_names or "conversations" in data.column_names:
        raise RuntimeError("conversation columns survived rendering; the toggle would be dropped")
    if len(data) == 0:
        raise RuntimeError("no chat rows remain after rendering")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=qwen35_sft_config(SFTConfig, values, output_dir, precision),
    )
    return trainer, {
        "lora_targets": targets,
        "architecture": evidence,
        "reasoning": reasoning,
        "rows": len(data),
    }


def train_after_gates(
    decision: dict, args: dict, compat, base: dict, contract, fail_closed
) -> None:
    """LoRA or QLoRA SFT for a Qwen3.5 student (transformers + peft + TRL 1.17)."""

    if args["method"] not in ("lora16", "qlora"):
        fail_closed("QWEN35_METHOD: method must be lora16 or qlora")
    if args["log_repo_id"]:
        fail_closed("QWEN35_LOG_FORWARD: log_repo_id is not supported on this path")
    try:
        reasoning = compat.reasoning_settings(base, decision["config"])
    except RuntimeError as exc:
        fail_closed(f"QWEN35_REASONING: {exc}")

    import torch

    if not torch.cuda.is_available():
        fail_closed("CUDA is not available; this trainer requires one GPU.")
    precision = compat.precision_kwargs(torch)
    if args["method"] == "qlora":
        try:
            compat.unsloth_load_kwargs(torch, load_in_4bit=True)
        except RuntimeError as exc:
            fail_closed(f"QWEN35_QLORA: {exc}")
    elif not (precision["bf16"] or precision["fp16"]):
        fail_closed("QWEN35_PRECISION: Qwen3.5 LoRA requires a CUDA mixed-precision device")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    revision = base.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        base["id"], revision=revision, trust_remote_code=False
    )
    try:
        compat.require_chat_template(tokenizer)
        compat.require_padding_token(tokenizer)
    except RuntimeError as exc:
        fail_closed(f"TOKENIZER: {exc}")

    load_kwargs = {
        "revision": revision,
        "trust_remote_code": False,
        "dtype": torch.bfloat16 if precision["bf16"] else torch.float16,
        "device_map": {"": 0},
    }
    if args["method"] == "qlora":
        compute = torch.bfloat16 if precision["bf16"] else torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["dtype"] = compute
    model = AutoModelForCausalLM.from_pretrained(base["id"], **load_kwargs)
    if compat.architecture_family(getattr(model, "config", None)) != "qwen3_5":
        fail_closed("QWEN35_ARCH: student did not load a Qwen3.5 graph")
    if args["method"] == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    print(f"loading pinned local corpus: {decision['train_jsonl']}")
    frame = load_dataset("json", data_files=decision["train_jsonl"], split="train")
    keep = [c for c in ("messages", "conversations") if c in frame.column_names]
    if keep:
        frame = frame.select_columns(keep)
    try:
        trainer, evidence = build_qwen35_trainer(
            model,
            tokenizer,
            frame,
            compat=compat,
            reasoning=reasoning,
            values=args,
            output_dir=str(args["out"]),
            precision=precision,
        )
    except RuntimeError as exc:
        fail_closed(f"QWEN35_TRAINER: {exc}")
    if evidence["rows"] < contract.MIN_ACCEPTED_EXAMPLES:
        fail_closed(
            f"LAUNCH_CORPUS_FLOOR: {evidence['rows']} chat examples remain; "
            f"launch requires {contract.MIN_ACCEPTED_EXAMPLES}"
        )
    print("qwen3.5 lora targets: " + ",".join(evidence["lora_targets"]))
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
        "enable_thinking": reasoning["enable_thinking"],
        "steps": int(trainer.state.global_step),
        "final_loss": float(
            trainer.state.log_history[-1].get("train_loss", float("nan"))
        )
        if trainer.state.log_history
        else None,
        "qwen35": evidence,
        "launch_contract": contract.SCHEMA,
        "training_authorization": "THIS_RECEIPT_IS_A_RUN_RECORD_NOT_A_PROMOTION",
    }
    receipt_path = Path(args["out"]) / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt written: {receipt_path}")
    print("wave complete")
