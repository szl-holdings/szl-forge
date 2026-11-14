"""The Hub alignment record stays tied to source bytes and does not weaken the gate."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "hub_source_alignment", ROOT / "tools" / "hub_source_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorded_divergence_matches_willay_source_and_keeps_the_gate():
    alignment = _load()
    document = alignment.verify(ROOT)
    willay = next(item for item in document["items"] if item["hub_repo"] == "SZLHOLDINGS/WILLAY")
    source = (ROOT / "willay/card/README.md").read_bytes()
    assert hashlib.sha256(source).hexdigest() == willay["canonical_source"]["sha256"]
    assert b"<!-- szl:artifact-identity-reconciled -->" in source
    assert willay["observed_readme_has_marker"] is False
    assert document["hf_writes"] is False


def test_offline_willay_candidate_still_carries_the_marker_and_pinned_digest():
    from tools.prepare_curated_model_card import prepare

    alignment = _load()
    willay = next(item for item in alignment.verify(ROOT)["items"] if item["hub_repo"] == "SZLHOLDINGS/WILLAY")
    candidate, _diff, receipt = prepare(ROOT, "willay")
    assert receipt["publication"] is False
    assert receipt["candidate_sha256"] == willay["curated_candidate_sha256"]
    assert hashlib.sha256(candidate).hexdigest() == willay["curated_candidate_sha256"]
    assert b"<!-- szl:artifact-identity-reconciled -->" in candidate


def test_forged_gate_or_pin_is_refused(tmp_path: Path):
    alignment = _load()
    for relative in (
        "docs/hub-source-alignment-2026-11-05.json",
        ".github/workflows/omen-cards-identity-note.yml",
        "tools/prepare_receiptagent_v3_card_sync.py",
        "willay/card/README.md",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    workflow = tmp_path / ".github/workflows/omen-cards-identity-note.yml"
    workflow.write_text(workflow.read_text(encoding="utf-8").replace(
        "sys.exit(1 if drift else 0)", "sys.exit(0)"
    ), encoding="utf-8")
    with pytest.raises(alignment.AlignmentError, match="fail-closed"):
        alignment.verify(tmp_path)
