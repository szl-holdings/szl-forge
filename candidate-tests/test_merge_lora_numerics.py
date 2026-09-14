"""Synthetic CPU/file controls; never train, fetch, publish or qualify real weights."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("merge_numeric_contract", ROOT / "scripts/merge_lora_hardened.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("merge source unavailable")
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)
KEY = "model.linear.weight"
AKEY = "base_model.model.model.linear.lora_A.weight"
BKEY = "base_model.model.model.linear.lora_B.weight"


def fixture(dtype=torch.float32):
    return {KEY: torch.zeros((4, 3), dtype=dtype)}, {
        AKEY: torch.ones((2, 3), dtype=dtype),
        BKEY: torch.ones((4, 2), dtype=dtype),
    }


def invoke(tmp_path, monkeypatch, base, adapter, *, rank=2, alpha=2.0, modules=1):
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    base_dir.mkdir()
    adapter_dir.mkdir()
    save_file(base, str(base_dir / "model.safetensors"))
    save_file(adapter, str(adapter_dir / "adapter_model.safetensors"))
    out = tmp_path / "candidate"
    monkeypatch.setattr("sys.argv", ["merge", "--base", str(base_dir), "--adapter", str(adapter_dir),
                                   "--out", str(out), "--rank", str(rank), "--alpha", str(alpha),
                                   "--expect-modules", str(modules)])
    before = {p: p.read_bytes() for p in (base_dir / "model.safetensors",
                                        adapter_dir / "adapter_model.safetensors")}
    status = merge.main()
    assert all(p.read_bytes() == raw for p, raw in before.items())
    return status, out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_actual_cli_writes_finite_changed_reloadable_weights(tmp_path, monkeypatch, dtype):
    base, adapter = fixture(dtype)
    base["unchanged.buffer"] = torch.arange(3, dtype=torch.int64)
    status, out = invoke(tmp_path, monkeypatch, base, adapter)
    assert status == 0
    result = load_file(str(out / "model.safetensors"))
    assert torch.equal(result[KEY], torch.full((4, 3), 2.0, dtype=dtype))
    assert torch.equal(result["unchanged.buffer"], base["unchanged.buffer"])
    assert torch.isfinite(result[KEY]).all().item()
    receipt = json.loads((out / "merge_receipt.json").read_text())
    assert receipt["rank"] == 2 and receipt["modulesApplied"] == 1
    assert set(receipt["deltas"]) == {KEY}


def test_finite_delta_can_overflow_final_weight_and_must_not_be_written(tmp_path, monkeypatch):
    base, adapter = fixture(torch.float16)
    base[KEY].fill_(60000)
    adapter[BKEY].fill_(6000)
    assert torch.isfinite((adapter[BKEY].float() @ adapter[AKEY].float()).half()).all().item()
    status, out = invoke(tmp_path, monkeypatch, base, adapter)
    assert status == 1
    assert not out.exists()


@pytest.mark.parametrize("rank", [1, 3, 4])
def test_declared_rank_must_equal_each_pair_before_output(tmp_path, monkeypatch, rank):
    base, adapter = fixture()
    status, out = invoke(tmp_path, monkeypatch, base, adapter, rank=rank)
    assert status == 1
    assert not out.exists()


@pytest.mark.parametrize("extra", ["model.linear.bias", "model.linear.lora_magnitude_vector",
                                    "model.linear.modules_to_save.default.weight",
                                    "model.linear.lora_A.other.weight"])
def test_unsupported_adapter_entries_are_not_silently_dropped(tmp_path, monkeypatch, extra):
    base, adapter = fixture()
    adapter[extra] = torch.ones(1)
    status, out = invoke(tmp_path, monkeypatch, base, adapter)
    assert status == 1
    assert not out.exists()


@pytest.mark.parametrize("location", ["base", "A", "B"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_inputs_abort_before_output(tmp_path, monkeypatch, location, value):
    base, adapter = fixture()
    target = base[KEY] if location == "base" else adapter[AKEY if location == "A" else BKEY]
    target[0, 0] = value
    status, out = invoke(tmp_path, monkeypatch, base, adapter)
    assert status == 1
    assert not out.exists()


@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
@pytest.mark.parametrize("location", ["base", "A", "B"])
def test_unsupported_target_dtype_is_rejected_not_implicitly_converted(tmp_path, monkeypatch, dtype, location):
    base, adapter = fixture()
    if location == "base":
        base[KEY] = base[KEY].to(dtype)
    else:
        key = AKEY if location == "A" else BKEY
        adapter[key] = adapter[key].to(dtype)
    status, out = invoke(tmp_path, monkeypatch, base, adapter)
    assert status == 1
    assert not out.exists()


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_scale_rejected_before_tensor_reads(tmp_path, monkeypatch, alpha):
    base, adapter = fixture()
    def forbidden_read(*args, **kwargs):
        raise AssertionError("invalid scale reached tensor loader")
    monkeypatch.setattr(merge, "load_file", forbidden_read)
    status, out = invoke(tmp_path, monkeypatch, base, adapter, alpha=alpha)
    assert status == 1
    assert not out.exists()


def test_zero_delta_remains_rejected(tmp_path, monkeypatch):
    base, adapter = fixture()
    adapter[BKEY].zero_()
    status, out = invoke(tmp_path, monkeypatch, base, adapter)
    assert status == 1 and not out.exists()


def test_late_failing_target_cannot_leave_partial_output(tmp_path, monkeypatch):
    base, adapter = fixture(torch.float16)
    base["model.zz.weight"] = torch.full((4, 3), 60000, dtype=torch.float16)
    adapter["model.zz.lora_A.weight"] = torch.ones((2, 3), dtype=torch.float16)
    adapter["model.zz.lora_B.weight"] = torch.full((4, 2), 6000, dtype=torch.float16)
    status, out = invoke(tmp_path, monkeypatch, base, adapter, modules=2)
    assert status == 1 and not out.exists()


def test_rank_two_positive_dimensions_required():
    base = {KEY: torch.zeros((4, 3))}
    adapter = {AKEY: torch.ones((0, 3)), BKEY: torch.ones((4, 0))}
    with pytest.raises(ValueError, match="positive"):
        merge.collect_targets(adapter, base)


def test_duplicate_normalized_adapter_side_still_fails():
    base, adapter = fixture()
    adapter["model.linear.lora_A.default.weight"] = adapter[AKEY].clone()
    with pytest.raises(ValueError, match="duplicate"):
        merge.collect_targets(adapter, base)


def test_ambiguous_base_still_fails():
    base, adapter = fixture()
    base["linear.weight"] = base[KEY].clone()
    with pytest.raises(KeyError, match="exactly one"):
        merge.collect_targets(adapter, base)
