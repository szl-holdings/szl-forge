# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import publish_chaski_card as publisher

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "publish-chaski-card.yml"


class RateLimited(RuntimeError):
    def __init__(self, retry_after: str = "1") -> None:
        super().__init__("provider rate limited")
        self.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": retry_after},
        )


def test_registry_is_closed_unique_and_source_bound() -> None:
    assert tuple(publisher.PROFILES) == ("chaski", "chaski-5050")
    assert publisher.resolve_profile().key == "chaski"
    assert publisher.resolve_profile("chaski-5050").repo_id == "SZLHOLDINGS/chaski-5050"
    assert len({profile.repo_id for profile in publisher.PROFILES.values()}) == 2
    assert len({profile.source_directory for profile in publisher.PROFILES.values()}) == 2
    for profile in publisher.PROFILES.values():
        assert profile.repo_id.startswith("SZLHOLDINGS/")
        assert profile.source_directory.is_relative_to(ROOT)
        assert set(profile.source_files) == {"README.md", "holo-banner.svg"}
    with pytest.raises(publisher.PublicationError, match="unknown card profile"):
        publisher.resolve_profile("not-a-profile")


@pytest.mark.parametrize(
    ("profile", "evaluation_state"),
    (("chaski", "MEASURED_FAIL"), ("chaski-5050", "NONE_THIS_RUN")),
)
def test_each_live_source_card_preserves_its_release_blocker(
    profile: str,
    evaluation_state: str,
) -> None:
    selected = publisher.resolve_profile(profile)
    assets = publisher.load_assets(selected)
    evidence = publisher.load_evidence(selected)
    measured = publisher.validate_assets(assets, selected, evidence)
    assert set(measured) == {"README.md", "holo-banner.svg"}
    assert selected.qualification == {
        "publication_eligible": False,
        "autonomy_eligible": False,
        "evaluation_state": evaluation_state,
        "release_blocker_preserved": True,
    }


def test_chaski_preserves_measured_negative_evidence() -> None:
    card = publisher.load_assets("chaski")["README.md"].decode("utf-8")
    assert "publication_eligible: false" in card
    assert "json_draft: 0/5" in card
    assert "adversarial_refusal: 2/6" in card
    assert "Named-N: MEASURED FAIL" in card
    assert "publication_eligible: true" not in card


def test_chaski_5050_preserves_none_this_run_and_model_separation() -> None:
    card = publisher.load_assets("chaski-5050")["README.md"].decode("utf-8")
    assert "evals: none-this-run" in card
    assert "Status: none-this-run." in card
    assert "never_overwrite: SZLHOLDINGS/chaski" in card
    assert "copied_live_chaski_weights: false" in card
    assert "publication_eligible: false" in card
    assert "autonomy_eligible: false" in card
    assert "Publishing this card is a documentation update, not model promotion" in card
    assert "Nobody else ships this combination" not in card
    assert "one-of-one" not in card.casefold()


def test_chaski_5050_is_bound_to_canonical_recipe_evidence() -> None:
    profile = publisher.resolve_profile("chaski-5050")
    evidence = publisher.load_evidence(profile)
    assert evidence is not None
    text = evidence.decode("utf-8")
    for boundary in profile.required_evidence_boundaries:
        assert boundary in text
    assert publisher.sha256_bytes(evidence)


@pytest.mark.parametrize("profile", ("chaski", "chaski-5050"))
def test_svg_is_local_scriptless_and_bounded(profile: str) -> None:
    assets = publisher.load_assets(profile)
    publisher.validate_assets(assets, profile, publisher.load_evidence(profile))
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
        publisher.validate_assets(
            assets,
            "chaski-5050",
            publisher.load_evidence("chaski-5050"),
        )


def test_chaski_5050_unbounded_or_promotion_claims_are_rejected() -> None:
    for claim in (
        "Nobody else ships this combination",
        "one-of-one",
        "publication_eligible: true",
        "autonomy_eligible: true",
        "STATUS: PRODUCTION READY",
    ):
        assets = publisher.load_assets("chaski-5050")
        assets["README.md"] += f"\n{claim}\n".encode("utf-8")
        with pytest.raises(publisher.PublicationError, match="forbidden qualification claim"):
            publisher.validate_assets(
                assets,
                "chaski-5050",
                publisher.load_evidence("chaski-5050"),
            )


def test_missing_canonical_5050_boundary_is_rejected() -> None:
    assets = publisher.load_assets("chaski-5050")
    evidence = publisher.load_evidence("chaski-5050")
    assert evidence is not None
    evidence = evidence.replace(
        b"SKU eval none-this-run.",
        b"SKU evaluation status unavailable.",
        1,
    )
    with pytest.raises(publisher.PublicationError, match="canonical evidence boundary"):
        publisher.validate_assets(assets, "chaski-5050", evidence)


@pytest.mark.parametrize("profile", ("chaski", "chaski-5050"))
def test_source_binding_is_deterministic_exact_and_denies_model_mutation(profile: str) -> None:
    selected = publisher.resolve_profile(profile)
    assets = publisher.load_assets(selected)
    canonical = publisher.load_evidence(selected)
    source_evidence = publisher.validate_assets(assets, selected, canonical)
    evidence_sha = publisher.sha256_bytes(canonical) if canonical is not None else None
    first = publisher.build_source_binding(
        profile=selected,
        source_revision="a" * 40,
        source_assets=source_evidence,
        evidence_sha256=evidence_sha,
    )
    second = publisher.build_source_binding(
        profile=selected,
        source_revision="a" * 40,
        source_assets=source_evidence,
        evidence_sha256=evidence_sha,
    )
    assert first == second
    payload = json.loads(first)
    assert payload["source"]["repository"] == "szl-holdings/szl-forge"
    assert payload["source"]["revision"] == "a" * 40
    assert payload["target"] == {
        "repo_id": selected.repo_id,
        "repo_type": "model",
        "controlled_paths": [
            "README.md",
            "holo-banner.svg",
            "szl-source-binding.json",
        ],
    }
    assert set(payload["authority"].values()) == {False}
    assert payload["qualification"] == selected.qualification
    if selected.evidence_path is None:
        assert "canonical_evidence" not in payload["source"]
    else:
        assert payload["source"]["canonical_evidence"]["path"] == "chaski/README_5050.md"
        assert payload["source"]["canonical_evidence"]["sha256"] == evidence_sha


@pytest.mark.parametrize("profile", ("chaski", "chaski-5050"))
def test_dry_run_report_is_profile_target_and_authority_bound(
    profile: str,
    tmp_path: Path,
) -> None:
    report = tmp_path / f"{profile}.json"
    revision = "b" * 40
    assert publisher.main(
        [
            "--profile",
            profile,
            "--source-revision",
            revision,
            "--report",
            str(report),
        ]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    selected = publisher.resolve_profile(profile)
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["profile"] == profile
    assert payload["source"]["revision"] == revision
    assert payload["target"]["repo_id"] == selected.repo_id
    assert payload["target"]["revision"] is None
    assert payload["target"]["commit_created"] is None
    assert payload["authority"]["files"] == [
        "README.md",
        "holo-banner.svg",
        "szl-source-binding.json",
    ]
    for field, value in payload["authority"].items():
        if field != "files":
            assert value is False
    assert payload["qualification"] == selected.qualification
    assert payload["secret_values_recorded"] is False


def test_workflow_is_one_closed_matrix_writer_with_immutable_actions() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Publish Chaski family model cards" in workflow
    assert workflow.count("Publish card, banner, and deterministic source binding") == 1
    assert "profile: chaski" in workflow
    assert "profile: chaski-5050" in workflow
    assert "repo: SZLHOLDINGS/chaski" in workflow
    assert "repo: SZLHOLDINGS/chaski-5050" in workflow
    assert "--target-repo \"${{ matrix.repo }}\"" in workflow
    assert "--profile \"${{ matrix.profile }}\"" in workflow
    assert "secrets: inherit" not in workflow
    assert "persist-credentials: false" in workflow
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:"):
            ref = stripped.rsplit("@", 1)[-1].split()[0]
            assert len(ref) == 40
            int(ref, 16)


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
                "--profile",
                "chaski-5050",
                "--source-revision",
                "main",
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
