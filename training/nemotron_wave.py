"""Nemotron-H (hybrid Mamba-2 + attention) SFT path for the Forge wave.

``train_wave.py`` routes a pinned Nemotron base here after the launch
contract has run.  The dense Qwen/Llama Unsloth path is not touched.  The
CPU dry-run drives ``build_nemotron_trainer`` with a tiny random-init graph.
This module never reads a token or launches a hosted job.
"""

from __future__ import annotations

import json
from pathlib import Path


def nemotron_sft_config(sft_config_cls, values: dict, output_dir: str, precision: dict):
    """SFTConfig for the Nemotron-H path (same knobs as the dense wave)."""

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


def build_nemotron_trainer(
    model,
    tokenizer,
    data,
    *,
    compat,
    base: dict,
    reasoning: dict,
    guardrails,
    values: dict,
    output_dir: str,
    precision: dict,
    fused_mamba: bool | None = None,
):
    """The Nemotron-H SFT path shared by the GPU wave and the CPU dry-run.

    Order: guardrail/NOTICE pre-check, architecture check, LoRA targets read
    from the loaded graph, explicit reasoning render, a gradient probe that
    every adapter trains, then the TRL trainer.  Any failure raises.
    """

    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    guard = compat.require_nemotron_guardrails(base, tokenizer, guardrails)
    compat.require_padding_token(tokenizer)
    if not compat.is_nemotron_h_config(getattr(model, "config", None)):
        raise RuntimeError("pinned Nemotron base did not load a Nemotron-H graph")
    targets, evidence = compat.lora_target_modules(model, fused_mamba=fused_mamba)
    if evidence["family"] != "nemotron_h":
        raise RuntimeError("Nemotron base resolved dense LoRA targets")
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

    def add_text(example):
        return compat.render_reasoning_chat_example(tokenizer, example, reasoning)

    data = data.map(add_text, desc="applying chat template with reasoning toggle")
    data = data.filter(lambda e: e["text"], desc="dropping non-chat rows")
    if len(data) == 0:
        raise RuntimeError("no chat rows remain after rendering")

    probe = tokenizer(
        data[0]["text"],
        return_tensors="pt",
        truncation=True,
        max_length=min(int(values["seqlen"]), 256),
    )
    probe.pop("token_type_ids", None)
    probe["labels"] = probe["input_ids"].clone()
    device = next(model.parameters()).device
    probe = {k: v.to(device) for k, v in probe.items()}
    gradients = compat.require_lora_gradients(model, probe)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=nemotron_sft_config(SFTConfig, values, output_dir, precision),
    )
    return trainer, {
        "lora_targets": targets,
        "architecture": evidence,
        "reasoning": reasoning,
        "guardrails": guard,
        "gradient_probe": gradients,
        "rows": len(data),
    }


def train_after_gates(
    decision: dict, args: dict, compat, base: dict, contract, fail_closed
) -> None:
    """16-bit LoRA SFT on a pinned Nemotron-H base (transformers + peft + TRL)."""

    if args["method"] != "lora16":
        fail_closed(
            "NEMOTRON_METHOD: Nemotron-H trains as 16-bit LoRA (method=lora16); "
            "4-bit QLoRA is refused because the fused Mamba-2 kernel reads raw "
            "projection weights"
        )
    if args["log_repo_id"]:
        fail_closed("NEMOTRON_LOG_FORWARD: log_repo_id is not supported on this path")
    try:
        reasoning = compat.reasoning_settings(base, decision["config"])
    except RuntimeError as exc:
        fail_closed(f"NEMOTRON_REASONING: {exc}")

    import torch

    if not torch.cuda.is_available():
        fail_closed("CUDA is not available; this trainer requires one GPU.")
    precision = compat.precision_kwargs(torch)
    if not precision["bf16"]:
        fail_closed("NEMOTRON_PRECISION: Nemotron-H requires a bf16 CUDA device")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base["id"], revision=base["revision"], trust_remote_code=False
    )
    try:
        compat.require_chat_template(tokenizer)
        compat.require_nemotron_guardrails(
            base, tokenizer, decision["config"].get("guardrails")
        )
    except RuntimeError as exc:
        fail_closed(f"TOKENIZER: {exc}")
    model = AutoModelForCausalLM.from_pretrained(
        base["id"],
        revision=base["revision"],
        dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map={"": 0},
    )

    print(f"loading pinned local corpus: {decision['train_jsonl']}")
    frame = load_dataset("json", data_files=decision["train_jsonl"], split="train")
    keep = [c for c in ("messages", "conversations") if c in frame.column_names]
    if keep:
        frame = frame.select_columns(keep)
    try:
        trainer, evidence = build_nemotron_trainer(
            model,
            tokenizer,
            frame,
            compat=compat,
            base=base,
            reasoning=reasoning,
            guardrails=decision["config"].get("guardrails"),
            values=args,
            output_dir=str(args["out"]),
            precision=precision,
        )
    except RuntimeError as exc:
        fail_closed(f"NEMOTRON_TRAINER: {exc}")
    if evidence["rows"] < contract.MIN_ACCEPTED_EXAMPLES:
        fail_closed(
            f"LAUNCH_CORPUS_FLOOR: {evidence['rows']} chat examples remain; "
            f"launch requires {contract.MIN_ACCEPTED_EXAMPLES}"
        )
    print("nemotron lora targets: " + ",".join(evidence["lora_targets"]))
    print("training start")
    trainer.train()
    trainer.save_model(str(args["out"]))
    tokenizer.save_pretrained(str(args["out"]))
    notice_path = compat.write_nvidia_notice(args["out"], base)

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
        "final_loss": float(
            trainer.state.log_history[-1].get("train_loss", float("nan"))
        )
        if trainer.state.log_history
        else None,
        "nemotron": {**evidence, "notice_path": notice_path},
        "launch_contract": contract.SCHEMA,
        "training_authorization": "THIS_RECEIPT_IS_A_RUN_RECORD_NOT_A_PROMOTION",
    }
    receipt_path = Path(args["out"]) / "train-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt written: {receipt_path}")
    print("wave complete")
