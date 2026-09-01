# SPDX-License-Identifier: Apache-2.0
"""Contract tests for forge_preflight — honest environment evidence."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import forge_preflight  # noqa: F401 - proves the module imports without torch/unsloth
from forge_preflight import main, run_preflight

GOOD_DATASET = os.path.join(ROOT, "szl_dataset.jsonl")


def test_preflight_schema_and_honesty_labels():
    report = run_preflight(
        dataset=GOOD_DATASET,
        require_cuda=False,
        min_disk_gib=0.0,
        min_examples=8,
    )
    assert report["schema"] == "szl.forge-preflight/v1"
    assert report["status"] in ("READY", "NOT_READY")
    assert report["preflight_sha256"].startswith("sha256:")
    labels = {check["label"] for check in report["checks"]}
    assert labels <= {"MEASURED", "UNAVAILABLE"}
    names = {check["name"] for check in report["checks"]}
    assert {"python", "disk_free_gib", "cuda", "dataset"} <= names
    for package in ("torch", "unsloth", "trl", "peft", "datasets"):
        assert f"package:{package}" in names


def test_preflight_never_invents_cuda():
    """CUDA must be MEASURED (real device fields) or honestly UNAVAILABLE."""
    report = run_preflight(
        dataset=GOOD_DATASET, require_cuda=False, min_disk_gib=0.0
    )
    cuda = next(c for c in report["checks"] if c["name"] == "cuda")
    if cuda["label"] == "MEASURED":
        assert cuda["value"]["device"]
        assert cuda["value"]["vram_gib"] > 0
    else:
        assert cuda["label"] == "UNAVAILABLE"
        assert cuda["ok"] is False


def test_missing_dataset_blocks_ready(tmp_path):
    report = run_preflight(
        dataset=str(tmp_path / "absent.jsonl"),
        require_cuda=False,
        min_disk_gib=0.0,
    )
    assert report["status"] == "NOT_READY"
    assert any("dataset" in reason for reason in report["blocking_reasons"])


def test_impossible_disk_requirement_blocks():
    report = run_preflight(
        dataset=GOOD_DATASET,
        require_cuda=False,
        min_disk_gib=10**9,
    )
    assert report["status"] == "NOT_READY"
    assert any("disk" in reason for reason in report["blocking_reasons"])


def test_deterministic_for_same_inputs():
    first = run_preflight(
        dataset=GOOD_DATASET, require_cuda=False, min_disk_gib=0.0
    )
    second = run_preflight(
        dataset=GOOD_DATASET, require_cuda=False, min_disk_gib=0.0
    )
    assert first["status"] == second["status"]
    assert first["dataset_report"]["dataset_sha256"] == (
        second["dataset_report"]["dataset_sha256"]
    )


def test_main_exit_codes(tmp_path, capsys):
    out = tmp_path / "preflight.json"
    code = main(
        [
            "--dataset",
            GOOD_DATASET,
            "--no-require-cuda",
            "--min-disk-gib",
            "0",
            "--min-examples",
            "8",
            "--out",
            str(out),
        ]
    )
    receipt = json.loads(out.read_text())
    # Exit code must mirror the receipt, whatever this host's package set is.
    assert code == (0 if receipt["status"] == "READY" else 1)
    assert receipt["schema"] == "szl.forge-preflight/v1"

    missing = main(["--dataset", str(tmp_path / "nope.jsonl")])
    assert missing == 1
    capsys.readouterr()


def test_unavailable_packages_always_block():
    """Invariant: UNAVAILABLE package ⇔ blocking reason, on any host."""
    report = run_preflight(
        dataset=GOOD_DATASET, require_cuda=False, min_disk_gib=0.0
    )
    unavailable = [
        c["name"] for c in report["checks"]
        if c["name"].startswith("package:") and not c["ok"]
    ]
    for name in unavailable:
        package = name.split(":", 1)[1]
        assert any(
            package in reason for reason in report["blocking_reasons"]
        ), name
    if unavailable:
        assert report["status"] == "NOT_READY"


def test_package_version_probe_is_honest():
    """Every probed package reports a version string or honest UNAVAILABLE."""
    versions = forge_preflight._package_versions()
    assert set(versions) == set(forge_preflight.TRAINING_PACKAGES)
    for value in versions.values():
        assert isinstance(value, str) and value
