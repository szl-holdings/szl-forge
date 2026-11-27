"""Shared TRL 0.23.1 / Unsloth trainer boundary for the Forge waves.

The helpers stay importable without a GPU.  They do not launch training jobs.
"""

from __future__ import annotations

from typing import Any


def mixed_precision_flags(torch_module: Any) -> tuple[bool, bool]:
    """Return ``(bf16, fp16)`` for the device that will run the trainer.

    CPU dry-runs stay in full precision.  A CUDA device uses bf16 when the
    device supports it and fp16 otherwise.  Hardcoding ``bf16=True`` is rejected
    by Transformers before the first step on a machine with no bf16 device.
    """

    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    available = bool(callable(is_available) and is_available())
    is_bf16 = getattr(cuda, "is_bf16_supported", None)
    bf16 = bool(available and callable(is_bf16) and is_bf16())
    return bf16, bool(available and not bf16)


def unsloth_load_kwargs(torch_module: Any, *, load_in_4bit: bool) -> dict[str, Any]:
    """Dtype and 4-bit flag for ``FastLanguageModel.from_pretrained``.

    ``dtype=None`` lets Unsloth pick float16 compute while the trainer requests
    bf16.  4-bit loading is refused without a CUDA mixed-precision dtype.
    """

    bf16, fp16 = mixed_precision_flags(torch_module)
    if load_in_4bit and not (bf16 or fp16):
        raise RuntimeError("4-bit quantization requires a CUDA mixed-precision dtype")
    if bf16:
        dtype = torch_module.bfloat16
    elif fp16:
        dtype = torch_module.float16
    else:
        dtype = torch_module.float32
    return {"dtype": dtype, "load_in_4bit": bool(load_in_4bit)}


def require_chat_template(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError("tokenizer has no chat template")
    return template


def require_padding_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos, str) or not eos:
        raise RuntimeError("tokenizer has no padding token and no eos token")
    tokenizer.pad_token = eos


def generation_batch(tokenizer: Any, text: str) -> Any:
    """Tokenize a prompt for ``generate`` without unused tokenizer keys.

    A fast tokenizer emits ``token_type_ids``.  Causal LMs reject that key
    before the first token, and the eval wave used to record the rejection as
    a generation error.
    """

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("generation prompt is empty")
    batch = tokenizer(text, return_tensors="pt")
    batch.pop("token_type_ids", None)
    if "input_ids" not in batch:
        raise RuntimeError("tokenizer did not return input ids")
    return batch


def render_chat_example(tokenizer: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Map a messages/conversations row to the text field TRL also accepts.

    TRL 0.23.1 prefers a ``messages`` column when one is present.  The rendered
    text is still required so rows that are not conversations are dropped before
    the trainer, and a missing or empty chat template fails closed here.
    """

    require_chat_template(tokenizer)
    messages = example.get("messages") or example.get("conversations")
    if not messages:
        example["text"] = ""
        return example
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError("chat template did not render a text row")
    example["text"] = rendered
    return example


def precision_kwargs(torch_module: Any) -> dict[str, bool]:
    bf16, fp16 = mixed_precision_flags(torch_module)
    return {"bf16": bf16, "fp16": fp16}
