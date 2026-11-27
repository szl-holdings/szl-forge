"""CPU dry-run for the Forge SFT, DPO, and eval trainer boundary.

The path never imports Unsloth, never calls a hosted job API, and never reads
a token or a signing key.  It builds a tiny local Llama, renders chat rows
with the shared template helper, and takes a few TRL 0.23.1 steps on CPU.
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
from pathlib import Path
from typing import Any

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant {% endif %}"
)


def _compat():
    path = Path(__file__).resolve().parent / "trainer_compat.py"
    spec = importlib.util.spec_from_file_location("szl_trainer_compat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"trainer compat helper is missing: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["user", "assistant", "question", "answer", "one", "two"]
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for word in words:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<pad>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _tiny_model(vocab_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


def _chat_rows() -> list[dict[str, Any]]:
    answers = ("one", "two")
    return [
        {
            "messages": [
                {"role": "user", "content": "user question"},
                {"role": "assistant", "content": f"assistant {answer}"},
            ]
        }
        for answer in answers
    ]


def run_cpu_dry_run() -> dict[str, Any]:
    """Construct the real TRL 0.23.1 trainers and take a few CPU steps."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")

    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    compat = _compat()
    quant_error = None
    try:
        compat.unsloth_load_kwargs(torch, load_in_4bit=True)
    except RuntimeError as exc:
        quant_error = str(exc)
    cpu_load = compat.unsloth_load_kwargs(torch, load_in_4bit=False)
    if quant_error is None:
        raise RuntimeError("CPU 4-bit quantization was accepted")
    if cpu_load["load_in_4bit"] or cpu_load["dtype"] != torch.float32:
        raise RuntimeError("CPU load policy is not full precision")

    tokenizer = _tiny_tokenizer()
    template = compat.require_chat_template(tokenizer)
    compat.require_padding_token(tokenizer)
    bare = _tiny_tokenizer()
    bare.chat_template = None
    try:
        compat.require_chat_template(bare)
    except RuntimeError as exc:
        missing_template = str(exc)
    else:
        raise RuntimeError("a tokenizer without a chat template was accepted")

    rendered = Dataset.from_list(_chat_rows()).map(
        lambda example: compat.render_chat_example(tokenizer, example)
    )
    if "text" not in rendered.column_names or "messages" not in rendered.column_names:
        raise RuntimeError("chat rows did not keep messages and rendered text")
    if any(not row["text"].strip() for row in rendered):
        raise RuntimeError("chat template rendered an empty row")

    precision = compat.precision_kwargs(torch)
    if precision != {"bf16": False, "fp16": False}:
        raise RuntimeError("CPU precision flags requested mixed precision")
    vocab = max(len(tokenizer), 32)
    model = _tiny_model(vocab)
    with tempfile.TemporaryDirectory() as directory:
        config = SFTConfig(
            output_dir=directory,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            packing=False,
            max_length=32,
            seed=7,
            dataset_text_field="text",
            report_to=[],
            gradient_checkpointing=False,
            **precision,
        )
        if config.eval_strategy != "no":
            raise RuntimeError("SFTConfig did not use eval_strategy")
        if getattr(config, "max_seq_length", None) not in (None, config.max_length):
            raise RuntimeError("retired max_seq_length survived SFT construction")
        if config.max_length != 32 or config.bf16 or config.fp16:
            raise RuntimeError("SFT length or precision was not bound for CPU")
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=rendered,
            args=config,
        )
        trained = trainer.train()
        loss = float(trained.training_loss)
        steps = int(trainer.state.global_step)

        preferences = Dataset.from_list(
            [
                {
                    "prompt": "user question",
                    "chosen": "assistant one",
                    "rejected": "assistant two",
                },
                {
                    "prompt": "user question",
                    "chosen": "assistant two",
                    "rejected": "assistant one",
                },
            ]
        )
        dpo_config = DPOConfig(
            output_dir=directory,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            beta=0.1,
            logging_steps=1,
            save_strategy="no",
            max_length=32,
            max_prompt_length=16,
            seed=7,
            report_to=[],
            gradient_checkpointing=False,
            **precision,
        )
        if dpo_config.max_length != 32 or dpo_config.bf16 or dpo_config.fp16:
            raise RuntimeError("DPO length or precision was not bound for CPU")
        dpo = DPOTrainer(
            model=_tiny_model(vocab),
            ref_model=_tiny_model(vocab),
            processing_class=tokenizer,
            train_dataset=preferences,
            args=dpo_config,
        )
        dpo_trained = dpo.train()
        dpo_loss = float(dpo_trained.training_loss)
        dpo_steps = int(dpo.state.global_step)

    if not math.isfinite(loss) or not math.isfinite(dpo_loss):
        raise RuntimeError("CPU dry-run loss is not finite")
    policy = _tiny_model(vocab)
    policy.eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "user question"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    raw = tokenizer(prompt, return_tensors="pt")
    if "token_type_ids" not in raw:
        raise RuntimeError("tokenizer did not emit the unused token type ids")
    inputs = compat.generation_batch(tokenizer, prompt)
    if "token_type_ids" in inputs:
        raise RuntimeError("generation batch still carries token type ids")
    generated = policy.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if int(generated.shape[-1]) <= int(inputs["input_ids"].shape[-1]):
        raise RuntimeError("eval generation did not emit a token")

    return {
        "training_loss": loss,
        "global_steps": steps,
        "dpo_loss": dpo_loss,
        "dpo_steps": dpo_steps,
        "chat_template_present": bool(template),
        "missing_template_refused": missing_template,
        "eval_strategy": config.eval_strategy,
        "max_length": config.max_length,
        "dpo_max_length": dpo_config.max_length,
        "bf16": bool(config.bf16),
        "fp16": bool(config.fp16),
        "cpu_dtype": str(cpu_load["dtype"]),
        "quantization_refused": quant_error,
        "dataset_field": "messages",
        "generated_tokens": int(generated.shape[-1]),
        "gpu_job_launched": False,
    }


# ---------------------------------------------------------------------------
# Nemotron-H (hybrid Mamba-2 + attention) CPU dry-run.
# ---------------------------------------------------------------------------

NEMOTRON_TEST_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True -%}"
    "{% for message in messages %}"
    "{{ message['role'] }} {{ message['content'] }}{{ eos_token }}"
    "{% endfor %}"
    "{% if enable_thinking %}think {% else %}nothink {% endif %}"
    "{% if add_generation_prompt %}assistant {% endif %}"
)


def _tiny_nemotron_h(vocab_size: int):
    """Random-init NemotronH with Mamba, attention and MLP mixers.

    Uses the native transformers class (no remote code, offline).
    """

    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        layers_block_type=["mamba", "mlp", "attention", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        chunk_size=8,
        expand=2,
        conv_kernel=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    return NemotronHForCausalLM(config)


def _nemotron_tokenizer(compat):
    """Real pinned Nano-4B tokenizer when a local snapshot is given, else tiny."""

    directory = os.environ.get("SZL_NEMOTRON_TOKENIZER_DIR")
    base = compat.nemotron_base("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    if directory:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        return tokenizer, base, "pinned-nano-4b-tokenizer"
    tokenizer = _tiny_tokenizer()
    tokenizer.chat_template = NEMOTRON_TEST_TEMPLATE
    fixture = dict(base)
    fixture["chat_template_sha256"] = compat.chat_template_sha256(tokenizer)
    return tokenizer, fixture, "tiny-fixture-tokenizer"


def _guardrails(base: dict[str, Any]) -> dict[str, Any]:
    return {
        "nvidia_guardrails": "kept",
        "license_id": base["license_id"],
        "notice": base["notice"],
        "system_prompt": "preserve",
    }


def _refused(fn) -> str:
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    raise RuntimeError("a stripped guardrail or template was accepted")


def run_nemotron_cpu_dry_run(steps: int = 2) -> dict[str, Any]:
    """Tiny Nemotron-H through the real Forge SFT path for ``steps`` CPU steps."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")

    from importlib import util

    import torch
    from datasets import Dataset

    compat = _compat()
    spec = util.spec_from_file_location(
        "szl_nemotron_wave", Path(__file__).resolve().parent / "nemotron_wave.py"
    )
    wave = util.module_from_spec(spec)
    spec.loader.exec_module(wave)

    tokenizer, base, tokenizer_source = _nemotron_tokenizer(compat)
    guard = _guardrails(base)
    refusals = {
        "missing_guardrails": _refused(
            lambda: compat.require_nemotron_guardrails(base, tokenizer, None)
        ),
        "stripped_notice": _refused(
            lambda: compat.require_nemotron_guardrails(
                base, tokenizer, {**guard, "notice": ""}
            )
        ),
        "guardrails_removed": _refused(
            lambda: compat.require_nemotron_guardrails(
                base, tokenizer, {**guard, "nvidia_guardrails": "removed"}
            )
        ),
        "replaced_template": _refused(
            lambda: compat.require_nemotron_guardrails(
                {**base, "chat_template_sha256": "0" * 64}, tokenizer, guard
            )
        ),
    }
    bare = _tiny_tokenizer()
    bare.chat_template = None
    refusals["missing_template"] = _refused(
        lambda: compat.render_chat_text(
            bare,
            [{"role": "user", "content": "user question"}],
            reasoning={"control": "enable_thinking", "enable_thinking": True},
        )
    )
    refusals["implicit_toggle"] = _refused(
        lambda: compat.reasoning_settings(base, {"enable_thinking": None})
    )

    messages = [
        {"role": "system", "content": "system hello"},
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "assistant one"},
    ]
    # The toggle changes the assistant generation prefix; completed rows
    # without a reasoning trace render the same either way.
    on = compat.render_chat_text(
        tokenizer,
        messages[:2],
        reasoning={"control": base["reasoning_control"], "enable_thinking": True},
        add_generation_prompt=True,
    )
    off = compat.render_chat_text(
        tokenizer,
        messages[:2],
        reasoning={"control": base["reasoning_control"], "enable_thinking": False},
        add_generation_prompt=True,
    )
    if on == off:
        raise RuntimeError("enable_thinking did not change the rendered prompt")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_nemotron_h(max(len(tokenizer), 32))
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("tiny Nemotron-H has meta tensors")
    fused = compat.fused_mamba_training_path(model)
    rows = Dataset.from_list([{"messages": messages} for _ in range(4)])
    precision = compat.precision_kwargs(torch)
    reasoning = compat.reasoning_settings(base, {"enable_thinking": False})
    values = {
        "batch_size": 1,
        "grad_accum": 1,
        "epochs": 1,
        "lr": 1e-3,
        "seqlen": 64,
        "pack": False,
        "seed": 7,
        "max_steps": steps,
        "logging_steps": 1,
        "save_strategy": "no",
        "gradient_checkpointing": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        trainer, evidence = wave.build_nemotron_trainer(
            model,
            tokenizer,
            rows,
            compat=compat,
            base=base,
            reasoning=reasoning,
            guardrails=guard,
            values=values,
            output_dir=directory,
            precision=precision,
        )
        lora_before = {
            n: p.detach().clone()
            for n, p in trainer.model.named_parameters()
            if "lora_B" in n
        }
        trained = trainer.train()
        changed = sorted(
            {
                n.split(".lora_B")[0].rsplit(".", 1)[-1]
                for n, p in trainer.model.named_parameters()
                if n in lora_before and not torch.equal(lora_before[n], p.detach())
            }
        )
        trainer.model.save_pretrained(directory)
        notice_path = compat.write_nvidia_notice(directory, base)
        notice = Path(notice_path).read_text(encoding="utf-8")
        saved = sorted(p.name for p in Path(directory).iterdir())

    loss = float(trained.training_loss)
    if not math.isfinite(loss):
        raise RuntimeError("Nemotron-H CPU dry-run loss is not finite")
    missing = set(evidence["lora_targets"]) - set(changed)
    if missing:
        raise RuntimeError(f"LoRA targets did not train: {sorted(missing)}")
    return {
        "global_steps": int(trainer.state.global_step),
        "training_loss": loss,
        "lora_targets": evidence["lora_targets"],
        "lora_targets_updated": changed,
        "architecture": evidence["architecture"],
        "fused_mamba_kernel": fused,
        "mamba_path": "torch reference (CPU)" if not fused else "fused CUDA kernel",
        "mamba_ssm_installed": util.find_spec("mamba_ssm") is not None,
        "gradient_probe": evidence["gradient_probe"],
        "reasoning": evidence["reasoning"],
        "prompt_tail_thinking_on": on[-24:],
        "prompt_tail_thinking_off": off[-24:],
        "guardrails": evidence["guardrails"],
        "refusals": refusals,
        "tokenizer_source": tokenizer_source,
        "notice_in_artifacts": base["notice"] in notice,
        "saved_files": saved,
        "bf16": precision["bf16"],
        "gpu_job_launched": False,
    }


if __name__ == "__main__":
    import json
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "nemotron"
    result = run_nemotron_cpu_dry_run() if which == "nemotron" else run_cpu_dry_run()
    print(json.dumps(result, indent=2, default=str))
