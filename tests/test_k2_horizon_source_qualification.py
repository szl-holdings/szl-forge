# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "frontier" / "evaluation" / "k2_source_qualification.py"
spec = importlib.util.spec_from_file_location("k2_source_qualification", MODULE_PATH)
assert spec and spec.loader
k2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k2)


def test_revision_is_immutable_and_candidate_is_exact():
    assert k2.MODEL_ID == "IFM/K2-Horizon-7B"
    assert k2.REVISION == "14985b2765262fc7850476f71a7d0ab08e86af69"
    assert k2.STALE_UPSTREAM_RECIPE_REVISION == "69ada542b68fe13d767479db2ab9421baff88681"
    assert k2.HEX40.fullmatch(k2.REVISION)
    assert k2.REVISION != k2.STALE_UPSTREAM_RECIPE_REVISION


def test_weight_files_are_never_review_download_targets():
    names = [
        "README.md",
        "config.json",
        "model-00001-of-00004.safetensors",
        "pytorch_model.bin",
        "modeling_k2_horizon.py",
        "configuration_k2_horizon.py",
    ]
    selected = k2.select_review_files(names)
    assert "README.md" in selected
    assert "config.json" in selected
    assert "modeling_k2_horizon.py" in selected
    assert "configuration_k2_horizon.py" in selected
    assert all(not k2.is_weight_file(name) for name in selected)


def test_static_scan_inventory_is_non_executing_and_fail_visible():
    source = """
import os
import subprocess
from urllib import request

def f(x):
    subprocess.run([\"echo\", x])
    return eval(x)
"""
    result = k2.scan_python_source(source)
    assert result["syntax_ok"] is True
    assert "subprocess" in result["high_risk_imports"]
    assert "urllib" in result["high_risk_imports"]
    assert "subprocess.run" in result["high_risk_calls"]
    assert "eval" in result["high_risk_calls"]


def test_static_scan_reports_invalid_python():
    result = k2.scan_python_source("def broken(:\n")
    assert result["syntax_ok"] is False
    assert result["syntax_error"]


def test_review_file_selection_is_narrow():
    names = [
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json",
        "custom_helper.py",
        "modeling_k2_horizon.py",
        "configuration_k2_horizon.py",
        "notes.txt",
    ]
    selected = k2.select_review_files(names)
    assert "README.md" in selected
    assert "tokenizer_config.json" in selected
    assert "tokenizer.json" not in selected
    assert "custom_helper.py" not in selected
    assert "notes.txt" not in selected
