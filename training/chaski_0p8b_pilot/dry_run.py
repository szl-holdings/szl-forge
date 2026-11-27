"""CPU dry-run for teacher generation and student LoRA SFT.

Both steps use tiny random-init graphs. Nothing is downloaded, no token is
read, and no Hub write is attempted. The teacher graph is the Nemotron-H
builder from the transformers-5 lane. The student graph is a tiny
``Qwen3_5ForConditionalGeneration``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from training.chaski_0p8b_pilot.constants import OUTPUT_DATASET_REPO, OUTPUT_MODEL_REPO
from training.chaski_0p8b_pilot.student import build_student_trainer
from training.chaski_0p8b_pilot.teacher import generate_completions, trainer_compat

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}"
    "{% else %}"
    "{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant {% endif %}"
)


def _cpu_dry_run():
    path = Path(__file__).resolve().parents[1] / "cpu_dry_run.py"
    spec = importlib.util.spec_from_file_location("szl_cpu_dry_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"trainer dry-run helpers are missing: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")


def _tiny_tokenizer():
    lane = _cpu_dry_run()
    tokenizer = lane._tiny_tokenizer()
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _tiny_qwen35(vocab_size: int):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers import Qwen3_5TextConfig, Qwen3_5VisionConfig

    text = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 2, 2],
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=1,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        tie_word_embeddings=False,
    )
    return Qwen3_5ForConditionalGeneration(config)


def run_teacher_dry_run() -> dict[str, Any]:
    """Generate two completions from a tiny random Nemotron-H."""

    _offline()
    import torch

    torch.manual_seed(11)
    lane = _cpu_dry_run()
    compat = trainer_compat()
    tokenizer, base, tokenizer_source = lane._nemotron_tokenizer(compat)
    guard = lane._guardrails(base)
    reasoning = compat.reasoning_settings(base, {"enable_thinking": False})
    model = lane._tiny_nemotron_h(max(len(tokenizer), 32))
    heldout = {"held out user prompt"}
    prompts = [
        {
            "id": "dry-1",
            "source": "dry-run",
            "license": "Apache-2.0",
            "messages": [
                {"role": "system", "content": "system hello"},
                {"role": "user", "content": "user question"},
            ],
        },
        {
            "id": "dry-2",
            "source": "dry-run",
            "license": "Apache-2.0",
            "messages": [
                {"role": "system", "content": "system hello"},
                {"role": "user", "content": "user question two"},
            ],
        },
    ]
    blocked = {
        "id": "heldout",
        "source": "dry-run",
        "license": "Apache-2.0",
        "messages": [
            {"role": "user", "content": "held out user prompt"},
        ],
    }
    try:
        with tempfile.TemporaryDirectory() as directory:
            generate_completions(
                model,
                tokenizer,
                [blocked],
                compat=compat,
                base=base,
                guardrails=guard,
                reasoning=reasoning,
                max_new_tokens=1,
                out_dir=Path(directory),
                heldout_users=heldout,
                dataset_repo=OUTPUT_DATASET_REPO,
                require_text=False,
            )
    except RuntimeError as exc:
        heldout_refusal = str(exc)
    else:
        raise RuntimeError("a held-out prompt was generated")
    if "held-out" not in heldout_refusal:
        raise RuntimeError("held-out refusal did not name the held-out prompt")

    with tempfile.TemporaryDirectory() as directory:
        receipt = generate_completions(
            model,
            tokenizer,
            prompts,
            compat=compat,
            base=base,
            guardrails=guard,
            reasoning=reasoning,
            max_new_tokens=2,
            out_dir=Path(directory),
            heldout_users=heldout,
            dataset_repo=OUTPUT_DATASET_REPO,
            require_text=False,
        )
        notice = Path(receipt["notice"]).read_text(encoding="utf-8")
        corpus = Path(receipt["corpus"]).read_text(encoding="utf-8").strip().splitlines()
    if len(corpus) != 2:
        raise RuntimeError("teacher dry-run did not write two rows")
    if base["notice"] not in notice:
        raise RuntimeError("teacher NOTICE is missing")
    if torch.cuda.is_available():
        raise RuntimeError("teacher dry-run saw CUDA; this path is CPU-only")
    return {
        "rows": receipt["rows"],
        "new_tokens": 2,
        "tokenizer_source": tokenizer_source,
        "notice_present": True,
        "heldout_refusal": heldout_refusal,
        "enable_thinking": reasoning["enable_thinking"],
        "hub_write": False,
        "launch_executed": False,
        "gpu_job_launched": False,
    }


def run_student_dry_run(steps: int = 2) -> dict[str, Any]:
    """Take a few LoRA SFT steps on a tiny random Qwen3.5."""

    _offline()
    import torch

    torch.manual_seed(11)
    compat = trainer_compat()
    try:
        build_student_trainer(
            _tiny_qwen35(32),
            _tiny_tokenizer(),
            _rows(),
            compat=compat,
            values=_values(steps),
            output_dir=".",
            precision={"bf16": False, "fp16": False},
            output_repo="SZLHOLDINGS/chaski",
        )
    except RuntimeError as exc:
        forbidden = str(exc)
    else:
        raise RuntimeError("SZLHOLDINGS/chaski was accepted as the student output")
    if "chaski" not in forbidden.casefold():
        raise RuntimeError("forbidden-repo refusal did not name Chaski")

    tokenizer = _tiny_tokenizer()
    model = _tiny_qwen35(max(len(tokenizer), 32))
    precision = compat.precision_kwargs(torch)
    if precision != {"bf16": False, "fp16": False}:
        raise RuntimeError("CPU precision flags requested mixed precision")
    from datasets import Dataset

    with tempfile.TemporaryDirectory() as directory:
        trainer, evidence = build_student_trainer(
            model,
            tokenizer,
            Dataset.from_list(_rows()),
            compat=compat,
            values=_values(steps),
            output_dir=directory,
            precision=precision,
            output_repo=OUTPUT_MODEL_REPO,
        )
        trained = trainer.train()
        base = compat.nemotron_base("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        notice_path = compat.write_nvidia_notice(directory, base)
        notice = Path(notice_path).read_text(encoding="utf-8")
    loss = float(trained.training_loss)
    if not math.isfinite(loss):
        raise RuntimeError("student CPU dry-run loss is not finite")
    if int(trainer.state.global_step) != steps:
        raise RuntimeError("student dry-run did not take the requested steps")
    if base["notice"] not in notice:
        raise RuntimeError("student artifacts are missing the teacher NOTICE")
    return {
        "global_steps": int(trainer.state.global_step),
        "training_loss": loss,
        "lora_targets": evidence["lora_targets"],
        "architecture": evidence["architecture"],
        "output_repo": evidence["output_repo"],
        "forbidden_repo_refusal": forbidden,
        "notice_present": True,
        "assistant_only_loss": True,
        "tie_word_embeddings": False,
        "hub_write": False,
        "launch_executed": False,
        "gpu_job_launched": False,
    }


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": "user question"},
                {"role": "assistant", "content": "assistant one"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "user question two"},
                {"role": "assistant", "content": "assistant two"},
            ]
        },
    ]


def _values(steps: int) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "lr": 1e-3,
        "seqlen": 64,
        "seed": 11,
        "max_steps": steps,
        "logging_steps": 1,
        "save_strategy": "no",
        "gradient_checkpointing": False,
        "lora_r": 4,
        "lora_alpha": 8,
    }


def main() -> int:
    import json
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    payload: dict[str, Any] = {}
    if which in {"both", "teacher"}:
        payload["teacher"] = run_teacher_dry_run()
    if which in {"both", "student"}:
        payload["student"] = run_student_dry_run()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
