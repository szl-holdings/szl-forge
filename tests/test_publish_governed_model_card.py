# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import publish_governed_model_card as governed
from tools import publish_khipu_card as base


def test_extended_registry_is_closed_without_mutating_base_registry() -> None:
    assert set(base.PROFILES) == {"khipu", "khipu-r2"}
    assert set(governed.PROFILES) == {"khipu", "khipu-r2", "chaski-5050"}
    assert governed.resolve_profile("chaski-5050").repo_id == "SZLHOLDINGS/chaski-5050"
    with pytest.raises(governed.PublicationError, match="unknown card profile"):
        governed.resolve_profile("operator-controlled-target")


def test_chaski_source_preserves_quarantine_and_no_eval_boundary() -> None:
    profile = governed.resolve_profile("chaski-5050")
    assets = base.load_assets(profile)
    evidence = base.validate_assets(assets, profile)
    assert set(evidence) == {"README.md", "holo-banner.svg"}
    card = assets["README.md"].decode("utf-8")
    for boundary in (
        "publication_eligible: false",
        "autonomy_eligible: false",
        "evals: none-this-run",
        "copied_live_chaski_weights: false",
        "**QUARANTINE.** Research residue.",
        "**Status: none-this-run.**",
        "Not live Chaski",
        "Lab load forbidden.",
        "620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5",
    ):
        assert boundary in card
    assert "publication_eligible: true" not in card
    assert "autonomy_eligible: true" not in card
    assert "copied_live_chaski_weights: true" not in card


def test_chaski_source_binding_and_dry_run_are_exact_and_non_authoritative(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    profile = governed.resolve_profile("chaski-5050")
    assets = base.load_assets(profile)
    evidence = base.validate_assets(assets, profile)
    binding = json.loads(
        base.build_source_binding(
            profile=profile,
            source_revision=revision,
            source_assets=evidence,
        )
    )
    assert binding["target"]["repo_id"] == "SZLHOLDINGS/chaski-5050"
    assert binding["qualification"]["abstention_result"] == "none-this-run"
    assert all(value is False for value in binding["authority"].values())

    report = tmp_path / "report.json"
    assert governed.main(
        [
            "--profile",
            "chaski-5050",
            "--source-revision",
            revision,
            "--report",
            str(report),
        ]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/chaski-5050"
    assert payload["qualification"] == {
        "publication_eligible": False,
        "abstention_result": "none-this-run",
        "release_blocker_preserved": True,
    }
    assert all(
        payload["authority"][field] is False
        for field in (
            "weights_changed",
            "adapter_changed",
            "configs_changed",
            "evals_changed",
            "visibility_changed",
            "hardware_changed",
            "runtime_changed",
        )
    )
