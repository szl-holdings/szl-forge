"""Phase 2: LoRA SFT of Qwen3.5-0.8B on the transformers 5.22.2 lane.

The dense Unsloth wave stays on TRL 0.23.1 and cannot load this student.
This path uses the image pins (transformers 5.22.2, trl 1.17.1, peft 0.21.2)
and the Qwen3.5 conditional-generation class. It does not copy the Nemotron-H
target, guardrail, or NOTICE helpers. NOTICE for a run that consumed teacher
outputs is still written by ``trainer_compat.write_nvidia_notice`` when the
caller passes the pinned teacher base.

``--execute`` is refused. The CPU dry-run builds a tiny random Qwen3.5 graph
and calls ``build_student_trainer``. No Hub client is imported.
"""

from __future__ import annotations

from typing import Any

from training.chaski_0p8b_pilot.constants import (
    OUTPUT_MODEL_REPO,
    STUDENT_ARCHITECTURE,
    refuse_forbidden_repo,
)

# Language-model projections only. Vision ``qkv`` / ``proj`` / ``linear_fc*``
# are not in this set. Leaf names do not collide with the vision tower on
# Qwen3_5ForConditionalGeneration.
LANGUAGE_LORA_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_REQUIRED_GROUPS = (
    frozenset({"q_proj", "k_proj", "v_proj", "o_proj"}),
    frozenset({"in_proj_qkv", "out_proj"}),
    frozenset({"gate_proj", "up_proj", "down_proj"}),
)


def language_lora_targets(model: Any) -> list[str]:
    """Leaf names of language-model Linears that LoRA should train."""

    found: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Linear":
            continue
        if "language_model" not in name.split("."):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in LANGUAGE_LORA_LEAVES and leaf not in found:
            found.append(leaf)
    present = set(found)
    missing = [sorted(group) for group in _REQUIRED_GROUPS if not group <= present]
    if missing:
        raise RuntimeError(f"Qwen3.5 language LoRA targets are incomplete: {missing}")
    return found


def _require_qwen35(model: Any) -> None:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3_5":
        raise RuntimeError("student graph is not model_type qwen3_5")
    if model.__class__.__name__ != STUDENT_ARCHITECTURE:
        raise RuntimeError(
            f"student class is {model.__class__.__name__}, expected {STUDENT_ARCHITECTURE}"
        )


def build_student_trainer(
    model: Any,
    tokenizer: Any,
    data: Any,
    *,
    compat: Any,
    values: dict[str, Any],
    output_dir: str,
    precision: dict[str, bool],
    output_repo: str = OUTPUT_MODEL_REPO,
):
    """LoRA SFT trainer for a Qwen3.5 graph that is already loaded.

    The caller owns the weights. This function does not download them.
    """

    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    refuse_forbidden_repo(output_repo)
    _require_qwen35(model)
    compat.require_chat_template(tokenizer)
    compat.require_padding_token(tokenizer)
    targets = language_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(values.get("lora_r", 16)),
            lora_alpha=int(values.get("lora_alpha", 32)),
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )

    def add_text(example: dict[str, Any]) -> dict[str, Any]:
        return compat.render_chat_example(tokenizer, example)

    data = data.map(add_text)
    data = data.filter(lambda row: row["text"])
    if len(data) == 0:
        raise RuntimeError("no chat rows remain after rendering")

    probe = tokenizer(
        data[0]["text"],
        return_tensors="pt",
        truncation=True,
        max_length=min(int(values["seqlen"]), 64),
    )
    probe.pop("token_type_ids", None)
    probe["labels"] = probe["input_ids"].clone()
    device = next(model.parameters()).device
    probe = {key: value.to(device) for key, value in probe.items()}
    gradients = compat.require_lora_gradients(model, probe)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(values["batch_size"]),
            gradient_accumulation_steps=int(values["grad_accum"]),
            num_train_epochs=int(values["epochs"]),
            learning_rate=float(values["lr"]),
            logging_steps=int(values.get("logging_steps", 1)),
            save_strategy=values.get("save_strategy", "no"),
            max_steps=int(values.get("max_steps", -1)),
            packing=False,
            max_length=int(values["seqlen"]),
            seed=int(values["seed"]),
            dataset_text_field="text",
            assistant_only_loss=True,
            report_to=[],
            gradient_checkpointing=bool(values.get("gradient_checkpointing", False)),
            **precision,
        ),
    )
    return trainer, {
        "lora_targets": targets,
        "gradient_probe": gradients,
        "rows": len(data),
        "output_repo": output_repo,
        "architecture": STUDENT_ARCHITECTURE,
        "hub_write": False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    requested = "execute" if args.execute else "prepare"
    raise SystemExit(
        "Chaski 0.8B student SFT "
        f"({requested}) is not authorized to launch. "
        "Use the CPU dry-run. No model was loaded and no Hub write ran."
    )


if __name__ == "__main__":
    raise SystemExit(main())
