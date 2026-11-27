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


# ---------------------------------------------------------------------------
# Architecture-aware LoRA targets, reasoning toggle, and NVIDIA guardrails.
#
# The dense Qwen/Llama path keeps the exact seven projection names it always
# used.  Nemotron-H (hybrid Mamba-2 + attention + MLP) targets are read from
# the loaded model's own config and module graph and fail closed when the
# graph does not match the pinned base.
# ---------------------------------------------------------------------------

DENSE_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

NEMOTRON_H_MODEL_TYPES = frozenset({"nemotron_h"})
NEMOTRON_H_ARCHITECTURES = frozenset({"NemotronHForCausalLM"})

NEMOTRON_NOTICE = (
    "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License."
)
NVIDIA_OPEN_MODEL_NOTICE = (
    "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
)

# Pinned bases.  Revisions and chat-template digests were read from the Hub
# (tree at the pinned revision) on 2026-11-27; a different revision or a
# replaced/stripped template is refused before training.
NEMOTRON_BASES: dict[str, dict[str, Any]] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": {
        "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "license_id": "nvidia-nemotron-open-model-license",
        "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/",
        "notice": NEMOTRON_NOTICE,
        "reasoning_control": "enable_thinking",
        "chat_template_sha256": "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
    },
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": {
        "revision": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
        "license_id": "nvidia-open-model-license",
        "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        "notice": NVIDIA_OPEN_MODEL_NOTICE,
        "reasoning_control": "system_marker",
        "chat_template_sha256": "b7a3a520a4bc1beae6a25d615f9215580df8eea926f488256c2bef74039433c0",
    },
}


def nemotron_base(model_id: Any, revision: Any = None) -> dict[str, Any] | None:
    """Return the pinned Nemotron entry for ``model_id`` or ``None``.

    A registered id at any other revision is refused rather than treated as
    a dense model.
    """

    entry = NEMOTRON_BASES.get(model_id) if isinstance(model_id, str) else None
    if entry is None:
        return None
    if revision is not None and revision != entry["revision"]:
        raise RuntimeError(
            f"{model_id} revision {revision!r} is not the pinned {entry['revision']}"
        )
    return {"id": model_id, **entry}


def is_nemotron_h_config(config: Any) -> bool:
    model_type = getattr(config, "model_type", None)
    architectures = set(getattr(config, "architectures", None) or ())
    return model_type in NEMOTRON_H_MODEL_TYPES or bool(
        architectures & NEMOTRON_H_ARCHITECTURES
    )


def _mixer_kind(module: Any) -> str | None:
    name = type(module).__name__
    if "Mamba" in name:
        return "mamba"
    if "Attention" in name:
        return "attention"
    if "MoE" in name or "Expert" in name:
        return "moe"
    if "MLP" in name:
        return "mlp"
    return None


def nemotron_h_layout(model: Any) -> dict[str, Any]:
    """Count Mamba/attention/MLP mixers and their projection leaves."""

    counts = {"mamba": 0, "attention": 0, "mlp": 0}
    leaves: dict[str, set[str]] = {"mamba": set(), "attention": set(), "mlp": set()}
    for name, module in model.named_modules():
        if not name.endswith(".mixer") and name != "mixer":
            continue
        kind = _mixer_kind(module)
        if kind is None or kind == "moe":
            raise RuntimeError(
                f"unsupported Nemotron-H mixer {type(module).__name__} at {name}"
            )
        counts[kind] += 1
        for child_name, _ in module.named_children():
            leaves[kind].add(child_name)
    if not counts["mamba"] or not counts["attention"]:
        raise RuntimeError(
            "Nemotron-H graph has no Mamba or no attention mixer; refusing targets"
        )
    return {"counts": counts, "leaves": {k: sorted(v) for k, v in leaves.items()}}


def fused_mamba_training_path(model: Any) -> bool:
    """True when training will use the fused Mamba-2 kernel.

    The fused ``mamba_split_conv1d_scan_combined`` path reads
    ``out_proj.weight`` directly, so a LoRA adapter on the Mamba ``out_proj``
    would be bypassed and silently never train.  Transformers selects that
    kernel when the model is on CUDA and ``mamba_ssm`` (or a Hub kernel) is
    importable; on CPU the PyTorch reference path is used.
    """

    import importlib.util as _util

    try:
        device = next(model.parameters()).device.type
    except StopIteration:
        return False
    if device != "cuda":
        return False
    return _util.find_spec("mamba_ssm") is not None or bool(
        getattr(model, "_use_kernels", False)
    )


def lora_target_modules(
    model: Any, *, fused_mamba: bool | None = None
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(target_modules, evidence)`` for the loaded model.

    Dense models get the unchanged seven-projection list.  Nemotron-H models
    get Mamba ``in_proj``, attention ``q/k/v/o_proj`` and MLP
    ``up_proj``/``down_proj``.  Mamba ``out_proj`` and ``conv1d`` are never
    targeted (see below).
    """

    config = getattr(model, "config", None)
    if not is_nemotron_h_config(config):
        return list(DENSE_LORA_TARGETS), {"family": "dense"}
    layout = nemotron_h_layout(model)
    leaves = layout["leaves"]
    if fused_mamba is None:
        fused_mamba = fused_mamba_training_path(model)
    # Mamba out_proj and conv1d are never targeted: the fused training
    # kernel reads both weights directly (an adapter there would be bypassed),
    # and PEFT >= 0.21 refuses them for Mamba-based model types.
    wanted = {
        "mamba": ["in_proj"],
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["up_proj", "down_proj"],
    }
    targets: list[str] = []
    for kind, names in wanted.items():
        for target in names:
            if target not in leaves[kind]:
                raise RuntimeError(
                    f"Nemotron-H {kind} mixer has no {target}; leaves={leaves[kind]}"
                )
            targets.append(target)
    return targets, {
        "family": "nemotron_h",
        "model_type": getattr(config, "model_type", None),
        "layout": layout["counts"],
        "fused_mamba_kernel": bool(fused_mamba),
        "excluded": {
            "conv1d": "fused kernel reads conv1d.weight; PEFT refuses it",
            "out_proj": "fused kernel reads out_proj.weight; PEFT refuses it",
        },
    }


def require_lora_gradients(model: Any, batch: dict[str, Any]) -> dict[str, int]:
    """One forward/backward; every LoRA tensor must receive a finite gradient.

    Catches a fused kernel or a wrapper that bypasses an adapter.  Gradients
    are cleared afterwards so the trainer starts from the same state.
    """

    import torch

    lora = [
        (n, p) for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    if not lora:
        raise RuntimeError("model has no trainable LoRA tensors")
    was_training = model.training
    model.train()
    out = model(**batch)
    loss = out.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("gradient probe loss is missing or non-finite")
    loss.backward()
    dead = []
    for name, param in lora:
        grad = param.grad
        if grad is None or not torch.isfinite(grad).all():
            dead.append(name)
    # lora_B starts at zero, so lora_A grads are zero on step 0 by design;
    # lora_B must be non-zero for every adapted module.
    zero_b = [
        n
        for n, p in lora
        if "lora_B" in n and p.grad is not None and float(p.grad.abs().sum()) == 0.0
    ]
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    if dead or zero_b:
        raise RuntimeError(
            "LoRA tensors received no gradient: " + ", ".join((dead + zero_b)[:8])
        )
    return {"lora_tensors": len(lora), "checked": len(lora)}


def reasoning_settings(
    base: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, Any] | None:
    """Explicit reasoning toggle for a Nemotron base; ``None`` for dense."""

    if base is None:
        return None
    value = config.get("enable_thinking")
    if not isinstance(value, bool):
        raise RuntimeError(  # noqa: TRY004 - callers fail closed on RuntimeError
            "Nemotron launch config must set enable_thinking to true or false"
        )
    return {"control": base["reasoning_control"], "enable_thinking": value}


def _apply_reasoning(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    reasoning: dict[str, Any],
    **kwargs: Any,
) -> Any:
    template = require_chat_template(tokenizer)
    control = reasoning.get("control")
    flag = reasoning.get("enable_thinking")
    if not isinstance(flag, bool):
        raise RuntimeError("reasoning toggle is not an explicit boolean")  # noqa: TRY004
    if control == "enable_thinking":
        if "enable_thinking" not in template:
            raise RuntimeError("chat template does not read enable_thinking")
        return tokenizer.apply_chat_template(messages, enable_thinking=flag, **kwargs)
    if control == "system_marker":
        if "/think" not in template or "/no_think" not in template:
            raise RuntimeError("chat template does not read /think and /no_think")
        marker = "/think" if flag else "/no_think"
        other = "/no_think" if flag else "/think"
        rows = [dict(m) for m in messages]
        for m in rows:
            content = m.get("content")
            if isinstance(content, str) and other in content.replace(marker, ""):
                raise RuntimeError("row carries a conflicting reasoning marker")
        if rows and rows[0].get("role") == "system":
            content = rows[0].get("content") or ""
            if marker not in content:
                rows[0]["content"] = f"{content} {marker}".strip()
        else:
            rows.insert(0, {"role": "system", "content": marker})
        return tokenizer.apply_chat_template(rows, **kwargs)
    raise RuntimeError(f"unknown reasoning control {control!r}")


def render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    reasoning: dict[str, Any] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    require_chat_template(tokenizer)
    if reasoning is None:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    else:
        rendered = _apply_reasoning(
            tokenizer,
            messages,
            reasoning,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError("chat template did not render a text row")
    return rendered


def render_reasoning_chat_example(
    tokenizer: Any, example: dict[str, Any], reasoning: dict[str, Any]
) -> dict[str, Any]:
    """``render_chat_example`` with an explicit reasoning toggle."""

    require_chat_template(tokenizer)
    messages = example.get("messages") or example.get("conversations")
    if not messages:
        example["text"] = ""
        return example
    example["text"] = render_chat_text(tokenizer, messages, reasoning=reasoning)
    return example


def chat_template_sha256(tokenizer: Any) -> str:
    import hashlib

    return hashlib.sha256(require_chat_template(tokenizer).encode("utf-8")).hexdigest()


def require_nemotron_guardrails(
    base: dict[str, Any], tokenizer: Any, guardrails: Any
) -> dict[str, Any]:
    """Pre-train refusal when NVIDIA guardrails or the NOTICE were stripped.

    The model card asks that the provided safety guardrails are not
    circumvented and that the system prompt controls reasoning.  This check
    requires an explicit ``nvidia_guardrails: "kept"`` declaration, the exact
    licence id and NOTICE for the base's licence family, and the pinned
    chat template (which carries the system-prompt and reasoning handling)
    byte-for-byte.
    """

    if not isinstance(guardrails, dict):
        raise RuntimeError("GUARDRAILS: launch config has no guardrails object")  # noqa: TRY004
    if guardrails.get("nvidia_guardrails") != "kept":
        raise RuntimeError("GUARDRAILS: nvidia_guardrails must be 'kept'")
    if guardrails.get("license_id") != base["license_id"]:
        raise RuntimeError("GUARDRAILS: licence id does not match the pinned base")
    if guardrails.get("notice") != base["notice"]:
        raise RuntimeError("GUARDRAILS: NOTICE line is missing or altered")
    if guardrails.get("system_prompt") != "preserve":
        raise RuntimeError("GUARDRAILS: system_prompt policy must be 'preserve'")
    observed = chat_template_sha256(tokenizer)
    if observed != base["chat_template_sha256"]:
        raise RuntimeError(
            f"GUARDRAILS: chat template was replaced or stripped (sha256 {observed})"
        )
    return {
        "nvidia_guardrails": "kept",
        "license_id": base["license_id"],
        "notice": base["notice"],
        "chat_template_sha256": observed,
    }


def write_nvidia_notice(out_dir: Any, base: dict[str, Any]) -> str:
    """Write NOTICE into the output artifacts and read it back."""

    from pathlib import Path

    path = Path(out_dir) / "NOTICE"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"{base['notice']}\n\n"
        f"Base model: {base['id']} @ {base['revision']}\n"
        f"Licence: {base['license_url']}\n"
    )
    path.write_text(text, encoding="utf-8")
    if base["notice"] not in path.read_text(encoding="utf-8"):
        raise RuntimeError("NOTICE readback failed")
    return str(path)
