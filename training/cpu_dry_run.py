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
