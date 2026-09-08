# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import publish_chaski_card as publisher


class RateLimited(RuntimeError):
    def __init__(self, retry_after: str = "1") -> None:
        super().__init__("provider rate limited")
        self.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": retry_after},
        )


def test_profile_registry_is_closed_and_target_specific() -> None:
    assert set(publisher.PROFILES) == {"chaski", "chaski-5050", "chaski-r2"}
    assert publisher.resolve_profile("chaski").repo_id == "SZLHOLDINGS/chaski"
    assert publisher.resolve_profile("chaski-5050").repo_id == "SZLHOLDINGS/chaski-5050"
    assert publisher.resolve_profile("chaski-r2").repo_id == "SZLHOLDINGS/chaski-r2"
    with pytest.raises(publisher.PublicationError, match="unknown card profile"):
        publisher.resolve_profile("operator-controlled-target")


def test_live_source_card_preserves_negative_evidence() -> None:
    assets = publisher.load_assets()
    evidence = publisher.validate_assets(assets)
    assert set(evidence) == {"README.md", "holo-banner.svg"}
    card = assets["README.md"].decode("utf-8")
    assert "publication_eligible: false" in card
    assert "json_draft: 0/5" in card
    assert "adversarial_refusal: 2/6" in card
    assert "Named-N: MEASURED FAIL" in card
    assert "publication_eligible: true" not in card


def test_chaski_5050_preserves_quarantine_and_no_eval_boundary() -> None:
    assets = publisher.load_assets("chaski-5050")
    evidence = publisher.validate_assets(assets, "chaski-5050")
    assert set(evidence) == {"README.md", "holo-banner.svg"}
    card = assets["README.md"].decode("utf-8")
    assert "> **QUARANTINE.** Research residue." in card
    assert "publication_eligible: false" in card
    assert "autonomy_eligible: false" in card
    assert "evals: none-this-run" in card
    assert "**Status: none-this-run.** No JSON/refusal gate ran. Not 5/5. Not 6/6." in card
    assert "620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5" in card
    assert "Nobody else ships this combination" not in card
    assert "one-of-one" not in card.casefold()


def test_chaski_r2_requires_search_tag_in_frontmatter() -> None:
    assets = publisher.load_assets("chaski-r2")
    publisher.validate_assets(assets, "chaski-r2")
    # The existing prose limitation is insufficient for Hub search metadata.
    assets["README.md"] = assets["README.md"].replace(b"- proposal-only\n", b"", 1)
    assert b"- proposal-only\n" in assets["README.md"]
    with pytest.raises(publisher.PublicationError, match="search tags missing"):
        publisher.validate_assets(assets, "chaski-r2")


def test_r2_source_binding_preserves_research_qualification() -> None:
    profile = publisher.resolve_profile("chaski-r2")
    assets = publisher.load_assets(profile)
    binding = json.loads(publisher.build_source_binding(
        profile=profile,
        source_revision="c" * 40,
        source_assets=publisher.validate_assets(assets, profile),
    ))
    assert binding["target"]["repo_id"] == "SZLHOLDINGS/chaski-r2"
    assert binding["qualification"]["evaluation_state"] == "NONE_THIS_RUN"
    assert binding["qualification"]["publication_eligible"] is False
    assert binding["qualification"]["autonomy_eligible"] is False
    assert all(value is False for value in binding["authority"].values())


def test_r2_is_in_candidate_and_publication_workflow() -> None:
    import yaml

    workflow = yaml.safe_load((publisher.ROOT / ".github/workflows/publish-chaski-card.yml").read_text())
    events = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "push"):
        assert "chaski-r2/card/**" in events[event]["paths"]
    profiles = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]
    assert {row["profile"] for row in profiles} == set(publisher.PROFILES)
    for row in profiles:
        assert row["repo_id"] == publisher.resolve_profile(row["profile"]).repo_id


@pytest.mark.parametrize("profile", ["chaski", "chaski-5050", "chaski-r2"])
def test_svg_is_local_scriptless_and_bounded(profile: str) -> None:
    assets = publisher.load_assets(profile)
    publisher.validate_assets(assets, profile)
    svg = assets["holo-banner.svg"].decode("utf-8").lower()
    assert svg.lstrip().startswith("<svg")
    assert "<script" not in svg
    assert "javascript:" not in svg
    namespace = 'xmlns="http://www.w3.org/2000/svg"'
    assert svg.count(namespace) == 1
    remote_scan = svg.replace(namespace, "", 1)
    assert "http://" not in remote_scan
    assert "https://" not in remote_scan
    assert len(assets["holo-banner.svg"]) < 10_000


def test_svg_remote_reference_is_rejected() -> None:
    assets = publisher.load_assets("chaski-5050")
    assets["holo-banner.svg"] = assets["holo-banner.svg"].replace(
        b"<svg ",
        b'<svg data-remote="https://evil.invalid" ',
        1,
    )
    with pytest.raises(publisher.PublicationError, match="remote SVG content"):
        publisher.validate_assets(assets, "chaski-5050")


def test_source_binding_is_deterministic_exact_and_non_authoritative() -> None:
    revision = "a" * 40
    profile = publisher.resolve_profile("chaski-5050")
    assets = publisher.load_assets(profile)
    evidence = publisher.validate_assets(assets, profile)
    first = publisher.build_source_binding(
        profile=profile,
        source_revision=revision,
        source_assets=evidence,
    )
    second = publisher.build_source_binding(
        profile=profile,
        source_revision=revision,
        source_assets=evidence,
    )
    assert first == second
    payload = json.loads(first)
    assert payload["source"]["repository"] == "szl-holdings/szl-forge"
    assert payload["source"]["revision"] == revision
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/chaski-5050"
    assert payload["target"]["controlled_paths"] == [
        "README.md",
        "holo-banner.svg",
        "szl-source-binding.json",
    ]
    assert payload["qualification"] == {
        "publication_eligible": False,
        "autonomy_eligible": False,
        "evaluation_state": "NONE_THIS_RUN",
        "release_blocker": "no_json_or_refusal_gate",
        "release_blocker_preserved": True,
    }
    assert all(value is False for value in payload["authority"].values())


def test_dry_run_report_is_source_bound_and_grants_no_extra_authority(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "a" * 40
    assert publisher.main(
        ["--source-revision", revision, "--report", str(report)]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["profile"] == "chaski"
    assert payload["source"]["revision"] == revision
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/chaski"
    assert payload["target"]["revision"] is None
    assert set(payload["target"]["assets"]) == {
        "README.md",
        "holo-banner.svg",
        "szl-source-binding.json",
    }
    assert payload["authority"]["files"] == [
        "README.md",
        "holo-banner.svg",
        "szl-source-binding.json",
    ]
    for field in (
        "weights_changed",
        "adapter_changed",
        "configs_changed",
        "evals_changed",
        "visibility_changed",
        "hardware_changed",
        "collection_changed",
        "runtime_changed",
    ):
        assert payload["authority"][field] is False
    assert payload["qualification"]["evaluation_state"] == "MEASURED_FAIL"
    assert payload["secret_values_recorded"] is False


def test_5050_dry_run_targets_only_5050_model(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "b" * 40
    assert publisher.main(
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
    assert payload["profile"] == "chaski-5050"
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/chaski-5050"
    assert payload["qualification"]["publication_eligible"] is False
    assert payload["qualification"]["autonomy_eligible"] is False
    assert payload["qualification"]["evaluation_state"] == "NONE_THIS_RUN"


def test_rate_limit_retry_is_bounded_and_honors_retry_after() -> None:
    calls = 0
    slept: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimited("2")
        return "ok"

    assert publisher.publish_with_bounded_retry(
        operation,
        attempts=3,
        sleeper=slept.append,
    ) == "ok"
    assert calls == 3
    assert slept == [2, 2]


def test_non_rate_limit_failure_is_not_retried() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("deterministic failure")

    with pytest.raises(RuntimeError, match="deterministic failure"):
        publisher.publish_with_bounded_retry(operation, sleeper=lambda _seconds: None)
    assert calls == 1


def test_invalid_source_revision_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="full lowercase commit SHA"):
        publisher.main(
            [
                "--source-revision",
                "main",
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
