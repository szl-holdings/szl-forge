from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_lora_hardened", ROOT / "scripts" / "merge_lora_hardened.py"
)
assert SPEC and SPEC.loader
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def test_unsloth_stem_resolves_to_weight_tensor_not_module_stem():
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((4, 3)),
    }
    adapter = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((2, 3)),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((4, 2)),
    }
    targets = merge.collect_targets(adapter, base)
    assert list(targets) == ["model.layers.0.self_attn.q_proj.weight"]


def test_named_default_adapter_keys_are_supported():
    base = {"model.linear.weight": torch.zeros((4, 3))}
    adapter = {
        "base_model.model.model.linear.lora_A.default.weight": torch.ones((2, 3)),
        "base_model.model.model.linear.lora_B.default.weight": torch.ones((4, 2)),
    }
    targets = merge.collect_targets(adapter, base)
    assert list(targets) == ["model.linear.weight"]


def test_missing_base_target_fails_closed():
    adapter = {
        "base_model.model.model.linear.lora_A.weight": torch.ones((2, 3)),
        "base_model.model.model.linear.lora_B.weight": torch.ones((4, 2)),
    }
    with pytest.raises(KeyError, match="exactly one base weight tensor"):
        merge.collect_targets(adapter, {})


def test_shape_mismatch_fails_before_merge():
    base = {"model.linear.weight": torch.zeros((5, 3))}
    adapter = {
        "base_model.model.model.linear.lora_A.weight": torch.ones((2, 3)),
        "base_model.model.model.linear.lora_B.weight": torch.ones((4, 2)),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        merge.collect_targets(adapter, base)


def test_unpaired_adapter_fails_closed():
    base = {"model.linear.weight": torch.zeros((4, 3))}
    adapter = {
        "base_model.model.model.linear.lora_A.weight": torch.ones((2, 3)),
    }
    with pytest.raises(ValueError, match="unpaired"):
        merge.collect_targets(adapter, base)


def test_ambiguous_model_wrapper_fails_closed():
    base = {
        "model.linear.weight": torch.zeros((4, 3)),
        "linear.weight": torch.zeros((4, 3)),
    }
    with pytest.raises(KeyError, match="exactly one"):
        merge.base_weight_key("model.linear", set(base))
