"""Shared trainer boundary for the Forge waves.

The dense Unsloth lane stays on the TRL 0.23.1 / Transformers 4.57.6 contract.
The Nemotron-H and Qwen3.5 lane uses the pinned image stack (Transformers
5.22.2, PEFT 0.21.2, TRL 1.17.1).  These helpers stay importable without a
GPU and do not launch training jobs.
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

# Known dense families keep the original seven projections.  Anything else
# fails closed instead of inheriting those names.
KNOWN_DENSE_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "phi3",
    }
)

NEMOTRON_H_MODEL_TYPES = frozenset({"nemotron_h"})
NEMOTRON_H_ARCHITECTURES = frozenset({"NemotronHForCausalLM"})

# Qwen3.5 is a hybrid (gated delta net + attention + MLP), not the dense seven.
QWEN35_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)
QWEN35_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5TextModel",
        "Qwen3_5MoeForCausalLM",
    }
)
QWEN35_LORA_TARGETS: tuple[str, ...] = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Mamba in_proj/out_proj, attention q/k/v/o, MLP up/down.  conv1d is not a
# Linear and the fused kernel reads its weight, so it is never a LoRA target.
NEMOTRON_H_LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "mamba": ("in_proj", "out_proj"),
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("up_proj", "down_proj"),
}

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
        "acknowledgement": "acknowledged: NVIDIA Nemotron Open Model License",
        "notice": NEMOTRON_NOTICE,
        "reasoning_control": "enable_thinking",
        "chat_template_sha256": "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
    },
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": {
        "revision": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
        "license_id": "nvidia-open-model-license",
        "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        "acknowledgement": "acknowledged: NVIDIA Open Model License",
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


def qwen35_base(model_id: Any, revision: Any = None) -> dict[str, Any] | None:
    """Native Qwen3.5 lane.  Dense Qwen2/Qwen3 ids stay on the Unsloth path."""

    if not isinstance(model_id, str):
        return None
    lowered = model_id.lower()
    if "qwen3.5" not in lowered and "qwen3_5" not in lowered:
        return None
    if revision is not None and (not isinstance(revision, str) or len(revision) < 7):
        raise RuntimeError("Qwen3.5 revision must be an immutable commit prefix")
    return {
        "id": model_id,
        "revision": revision,
        "family": "qwen3_5",
        "reasoning_control": "enable_thinking",
    }


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


def architecture_family(config: Any) -> str:
    """``nemotron_h``, ``qwen3_5``, or ``dense``.  Unknown types raise."""

    model_type = getattr(config, "model_type", None)
    architectures = set(getattr(config, "architectures", None) or ())
    if model_type in NEMOTRON_H_MODEL_TYPES or bool(
        architectures & NEMOTRON_H_ARCHITECTURES
    ):
        return "nemotron_h"
    if model_type in QWEN35_MODEL_TYPES or bool(architectures & QWEN35_ARCHITECTURES):
        return "qwen3_5"
    if model_type in KNOWN_DENSE_MODEL_TYPES:
        return "dense"
    raise RuntimeError(
        f"unsupported architecture {model_type!r}; refusing LoRA targets"
    )


def _leaf_names(model: Any) -> set[str]:
    found: set[str] = set()
    for name, module in model.named_modules():
        if not name:
            continue
        children = list(module.named_children())
        if children:
            continue
        found.add(name.rsplit(".", 1)[-1])
    return found


def _mamba_reference_stub(*_args: Any, **_kwargs: Any) -> None:
    """Torch fallback for the fused Mamba scan.  ``None`` selects the module forward."""

    return None


_mamba_reference_stub._szl_reference_mamba = True  # type: ignore[attr-defined]


def _rebind_mamba_kernel_lookups(fn: Any, stub: Any) -> None:
    """Point every fused-scan lookup reachable from ``fn`` at ``stub``.

    Mixer ``forward`` reads ``mamba2_split_conv1d_scan_combined`` from its
    globals, and a hook wrapper may close over that function.  Replacing only
    the name on ``modeling_nemotron_h`` after ``__init__`` misses a copy that
    was bound into another module or closure.  ``conv1d`` kernels are left
    alone.
    """

    seen: set[int] = set()
    stack = [fn]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen or not callable(current):
            continue
        seen.add(id(current))
        glob = getattr(current, "__globals__", None)
        if isinstance(glob, dict) and "mamba2_split_conv1d_scan_combined" in glob:
            glob["mamba2_split_conv1d_scan_combined"] = stub
        for cell in getattr(current, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if getattr(value, "__name__", "") == "mamba2_split_conv1d_scan_combined":
                cell.cell_contents = stub
            elif callable(value):
                stack.append(value)


def keep_mamba_out_proj_trainable() -> str:
    """Select the reference Mamba forward so ``out_proj`` LoRA is in the graph.

    The fused ``mamba_split_conv1d_scan_combined`` kernel reads
    ``out_proj.weight`` and returns before the Linear module, so an adapter
    there would never train.  Returning ``None`` from that entry point keeps
    the reference path, which calls ``out_proj`` as a module.  The chunk-scan
    and causal-conv kernels stay available when ``mamba-ssm`` is installed.

    The stub is installed on the Nemotron-H module and, when that class is
    present, on Zamba2 as well.  Lookups already closed over by the mixer
    forward are rebound after the model exists.
    """

    import transformers.models.nemotron_h.modeling_nemotron_h as modeling

    if not hasattr(modeling, "mamba2_split_conv1d_scan_combined"):
        raise RuntimeError(
            "Nemotron-H modeling has no fused scan entry point; refusing to hide out_proj"
        )
    if not hasattr(modeling, "NemotronHMamba2Mixer"):
        raise RuntimeError("Nemotron-H modeling has no Mamba2 mixer; refusing out_proj")

    modeling.mamba2_split_conv1d_scan_combined = _mamba_reference_stub
    try:
        import transformers.models.zamba2.modeling_zamba2 as zamba2
    except ImportError:
        zamba2 = None
    if zamba2 is not None and hasattr(zamba2, "mamba2_split_conv1d_scan_combined"):
        zamba2.mamba2_split_conv1d_scan_combined = _mamba_reference_stub
    _rebind_mamba_kernel_lookups(
        modeling.NemotronHMamba2Mixer.forward, _mamba_reference_stub
    )
    return "module forward"


def allow_nemotron_out_proj_lora() -> None:
    """Let PEFT adapt Nemotron-H ``out_proj`` after the reference path is selected.

    PEFT 0.21.2 refuses ``out_proj`` and ``conv1d`` on ``model_type=nemotron_h``
    because the fused kernel reads those weights and would skip the adapter.
    ``keep_mamba_out_proj_trainable`` has already forced the module forward, so
    ``out_proj`` is an ordinary Linear.  ``conv1d`` stays refused.
    """

    import transformers.models.nemotron_h.modeling_nemotron_h as modeling

    reference = modeling.mamba2_split_conv1d_scan_combined
    if not getattr(reference, "_szl_reference_mamba", False):
        raise RuntimeError(
            "refusing out_proj LoRA until the reference Mamba path is selected"
        )

    import peft.tuners.tuners_utils as utils

    current = getattr(utils, "_check_lora_target_modules_mamba", None)
    if current is None:
        raise RuntimeError(
            "PEFT has no Mamba target check; refusing out_proj rather than guessing"
        )
    if getattr(current, "_szl_nemotron_out_proj", False):
        return
    original = current

    def _checked(peft_config: Any, model: Any, target_name: str) -> None:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if target_name == "out_proj" and model_type == "nemotron_h":
            return None
        return original(peft_config, model, target_name)

    _checked._szl_nemotron_out_proj = True  # type: ignore[attr-defined]
    utils._check_lora_target_modules_mamba = _checked


def lora_target_modules(
    model: Any, *, fused_mamba: bool | None = None
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(target_modules, evidence)`` for the loaded model.

    Known dense models get the unchanged seven-projection list.  Nemotron-H
    gets Mamba ``in_proj``/``out_proj``, attention ``q/k/v/o_proj`` and MLP
    ``up_proj``/``down_proj``.  Qwen3.5 gets its linear-attention, attention
    and MLP projections.  An unknown architecture raises.
    """

    config = getattr(model, "config", None)
    family = architecture_family(config)
    if family == "dense":
        return list(DENSE_LORA_TARGETS), {"family": "dense"}
    if family == "qwen3_5":
        leaves = _leaf_names(model)
        missing = [name for name in QWEN35_LORA_TARGETS if name not in leaves]
        if missing:
            raise RuntimeError(
                "Qwen3.5 graph is missing LoRA targets " + ",".join(missing)
            )
        return list(QWEN35_LORA_TARGETS), {
            "family": "qwen3_5",
            "model_type": getattr(config, "model_type", None),
        }
    layout = nemotron_h_layout(model)
    leaves = layout["leaves"]
    if fused_mamba is None:
        fused_mamba = fused_mamba_training_path(model)
    targets: list[str] = []
    for kind, names in NEMOTRON_H_LORA_TARGETS.items():
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
        "out_proj_path": "module forward",
        "excluded": {
            "conv1d": "Conv1d; the kernel reads conv1d.weight directly",
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


def render_sft_rows(dataset: Any, tokenizer: Any, reasoning: dict[str, Any]) -> Any:
    """Render the recipe toggle into ``text`` and drop the message columns.

    TRL 1.17 reapplies the chat template itself when a ``messages`` column is
    still present, and that call does not receive ``enable_thinking``.
    """

    def add_text(example: dict[str, Any]) -> dict[str, Any]:
        return render_reasoning_chat_example(tokenizer, example, reasoning)

    data = dataset.map(add_text, desc="applying chat template with reasoning toggle")
    data = data.filter(lambda row: row["text"], desc="dropping non-chat rows")
    drop = [name for name in ("messages", "conversations") if name in data.column_names]
    if drop:
        data = data.remove_columns(drop)
    return data


def read_notice_file(path: Any) -> str:
    """Read a regular NOTICE file.  Symlinks and missing files are refused."""

    from pathlib import Path

    if not isinstance(path, (str, Path)):
        raise RuntimeError("GUARDRAILS: NOTICE file path is missing")  # noqa: TRY004
    file = Path(path)
    if file.is_symlink():
        raise RuntimeError("GUARDRAILS: NOTICE file must be a regular file")
    if not file.is_file():
        raise RuntimeError("GUARDRAILS: NOTICE file is missing")
    if file.stat().st_size > 256 * 1024:
        raise RuntimeError("GUARDRAILS: NOTICE file is too large")
    text = file.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError("GUARDRAILS: NOTICE file is empty")
    return text


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
    if guardrails.get("license_acknowledgement") != base["acknowledgement"]:
        raise RuntimeError("GUARDRAILS: NVIDIA licence was not acknowledged")
    if guardrails.get("notice") != base["notice"]:
        raise RuntimeError("GUARDRAILS: NOTICE line is missing or altered")
    notice_text = read_notice_file(guardrails.get("notice_file"))
    if base["notice"] not in notice_text:
        raise RuntimeError("GUARDRAILS: NOTICE file is missing the pinned notice line")
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
        "license_acknowledgement": base["acknowledgement"],
        "notice": base["notice"],
        "notice_file": str(guardrails.get("notice_file")),
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
