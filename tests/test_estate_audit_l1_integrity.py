"""Regression controls for source-grounded L1 estate inventory claims.

These are static, offline source checks. They do not assert runtime readiness.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "docs" / "estate-audit.json"


def _gap_contract_is_grounded(manifest: dict) -> bool:
    gaps = manifest["evidence_index"]["gap_register"]
    return (
        len(gaps) >= 4
        and "pyproject.toml" in gaps[1]
        and "lockfiles" in gaps[1]
        and ".github/workflows" in gaps[1]
        and "has not reconciled" in gaps[1]
        and "already points to LIVE-GOVERNED-INFERENCE-V3.md" in gaps[2]
        and "does not independently establish" in gaps[2]
        and "MiniEmbed-Nano/README.md records" in gaps[3]
        and "has not independently replayed" in gaps[3]
        and "remain unverified" in gaps[3]
    )


def test_l1_claims_agree_with_measured_repository_sources() -> None:
    manifest = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert manifest["metadata"]["loop_level"] == 1
    assert len(manifest["estate_context"]["baseline_commit"]) == 40
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / ".github" / "workflows").is_dir()

    v2 = (ROOT / "LIVE-GOVERNED-INFERENCE-V2.md").read_text(encoding="utf-8")
    assert "LIVE-GOVERNED-INFERENCE-V3.md" in v2

    mini = (ROOT / "MiniEmbed-Nano" / "README.md").read_text(encoding="utf-8")
    assert "mini_embed.npz" in mini
    assert "TRAINING_RECEIPT.json" in mini
    assert "SHA-256" in mini
    assert _gap_contract_is_grounded(manifest)


def test_synthetic_unsubstantiated_gap_regressions_are_rejected() -> None:
    manifest = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert _gap_contract_is_grounded(manifest)
    replacements = {
        1: "No lockfile, dependency manifest, or CI definition is present.",
        2: "There is no V2-to-V3 supersession or currency pointer.",
        3: "There is no MiniEmbed-Nano checksum evidence.",
    }
    for index, unsupported_claim in replacements.items():
        tampered = copy.deepcopy(manifest)
        tampered["evidence_index"]["gap_register"][index] = unsupported_claim
        assert not _gap_contract_is_grounded(tampered)
