"""Offline identity checks for the Khipu held-out evaluator; no model is loaded."""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys

import pytest


TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "szl_omen_pipeline.py"
spec = importlib.util.spec_from_file_location("szl_omen_pipeline_adapter_identity_test", TOOL)
assert spec is not None and spec.loader is not None
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


def test_exact_adapter_bytes_are_read_back(tmp_path: pathlib.Path):
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"synthetic adapter fixture")
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    receipt = {"status": "TRAINED_CHALLENGER", "adapter_sha256": {adapter.name: digest}}

    assert pipeline.challenger_adapter_sha256(tmp_path, receipt) == {adapter.name: digest}
    adapter.write_bytes(b"changed after training")
    with pytest.raises(ValueError, match="differs from the DPO receipt"):
        pipeline.challenger_adapter_sha256(tmp_path, receipt)


def test_missing_or_unreceipted_adapter_is_refused(tmp_path: pathlib.Path):
    digest = hashlib.sha256(b"synthetic adapter fixture").hexdigest()
    receipt = {"status": "TRAINED_CHALLENGER", "adapter_sha256": {"adapter_model.safetensors": digest}}

    with pytest.raises(ValueError, match="adapter files differ"):
        pipeline.challenger_adapter_sha256(tmp_path, receipt)

    (tmp_path / "adapter_model.safetensors").write_bytes(b"synthetic adapter fixture")
    (tmp_path / "adapter_model.bin").write_bytes(b"unreceipted second adapter")
    with pytest.raises(ValueError, match="adapter files differ"):
        pipeline.challenger_adapter_sha256(tmp_path, receipt)


def test_untrained_or_unsealed_receipt_is_refused(tmp_path: pathlib.Path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"synthetic adapter fixture")
    with pytest.raises(ValueError, match="not TRAINED_CHALLENGER"):
        pipeline.challenger_adapter_sha256(tmp_path, {"status": "FAIL-CLOSED"})
    with pytest.raises(ValueError, match="must seal one"):
        pipeline.challenger_adapter_sha256(tmp_path, {"status": "TRAINED_CHALLENGER", "adapter_sha256": {}})
