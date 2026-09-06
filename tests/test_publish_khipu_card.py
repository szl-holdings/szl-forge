# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import publish_khipu_card as publisher


class RateLimited(RuntimeError):
    def __init__(self, retry_after: str = "1") -> None:
        super().__init__("provider rate limited")
        self.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": retry_after},
        )


def test_profile_registry_is_closed_and_target_specific() -> None:
    assert set(publisher.PROFILES) == {"khipu", "khipu-r2"}
    assert publisher.resolve_profile("khipu").repo_id == "SZLHOLDINGS/SZL-Khipu-1.5B"
    assert publisher.resolve_profile("khipu-r2").repo_id == "SZLHOLDINGS/KHIPU-R2"
    with pytest.raises(publisher.PublicationError, match="unknown card profile"):
        publisher.resolve_profile("operator-controlled-target")


def test_live_source_card_preserves_receipts_and_release_blocker() -> None:
    assets = publisher.load_assets()
    evidence = publisher.validate_assets(assets)
    assert set(evidence) == {"README.md", "holo-banner.svg"}
    card = assets["README.md"].decode("utf-8")
    assert "publication_eligible: false" in card
    assert "keyId `89540347a69b789e`" in card
    assert "| plan-valid | 11 / 11 |" in card
    assert "| grounding | 4 / 5 |" in card
    assert "| abstain | 2 / 6 |" in card
    assert "| hallucinated citations | 0 |" in card
    assert "visible release blocker" in card
    assert "publication_eligible: true" not in card


def test_khipu_r2_source_preserves_measured_not_pass_boundary() -> None:
    assets = publisher.load_assets("khipu-r2")
    evidence = publisher.validate_assets(assets, "khipu-r2")
    assert set(evidence) == {"README.md", "holo-banner.svg"}
    card = assets["README.md"].decode("utf-8")
    assert "publication_eligible: false" in card
    assert "autonomy_eligible: false" in card
    assert "Abstain is MEASURED 3/6, not a pass." in card
    assert "| abstain (`adversarial.jsonl`) | **3 / 6** |" in card
    assert "held_out_in_gradients: false" in card
    assert "publication_eligible: true" not in card
    assert "autonomy_eligible: true" not in card


def test_card_keeps_exact_uploaded_artifact_digests() -> None:
    card = publisher.load_assets()["README.md"].decode("utf-8")
    assert "6f9f5b9df2a877c999e33faf542dc6e62ce63f4a2bf6b358fc48a4b6b113c3c9" in card
    assert "0a71b3a28b9f77ca3651f38c8caa1e34121934f5584dae24454d4c6eea823a66" in card
    assert "No deployed Alloy endpoint status is asserted by this card." in card


def test_r2_card_keeps_exact_adapter_digest() -> None:
    card = publisher.load_assets("khipu-r2")["README.md"].decode("utf-8")
    assert "e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349" in card
    assert "No signed R2 eval receipt in this atelier." in card
    assert "GPU **UNAVAILABLE**" in card


@pytest.mark.parametrize("profile", ["khipu", "khipu-r2"])
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
    assets = publisher.load_assets("khipu-r2")
    assets["holo-banner.svg"] = assets["holo-banner.svg"].replace(
        b"<svg ",
        b'<svg data-remote="https://evil.invalid" ',
        1,
    )
    with pytest.raises(publisher.PublicationError, match="remote SVG content"):
        publisher.validate_assets(assets, "khipu-r2")


def test_source_binding_is_deterministic_exact_and_non_authoritative() -> None:
    revision = "a" * 40
    profile = publisher.resolve_profile("khipu-r2")
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
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/KHIPU-R2"
    assert payload["target"]["controlled_paths"] == [
        "README.md",
        "holo-banner.svg",
        "szl-source-binding.json",
    ]
    assert payload["qualification"]["abstention_result"] == "3/6"
    assert all(value is False for value in payload["authority"].values())


def test_dry_run_report_is_source_bound_and_grants_no_extra_authority(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "a" * 40
    assert publisher.main(
        ["--source-revision", revision, "--report", str(report)]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["profile"] == "khipu"
    assert payload["source"]["revision"] == revision
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/SZL-Khipu-1.5B"
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
        "runtime_changed",
    ):
        assert payload["authority"][field] is False
    assert payload["qualification"] == {
        "publication_eligible": False,
        "abstention_result": "2/6",
        "release_blocker_preserved": True,
    }
    assert payload["secret_values_recorded"] is False


def test_r2_dry_run_targets_only_r2_model(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "b" * 40
    assert publisher.main(
        [
            "--profile",
            "khipu-r2",
            "--source-revision",
            revision,
            "--report",
            str(report),
        ]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["profile"] == "khipu-r2"
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/KHIPU-R2"
    assert payload["qualification"] == {
        "publication_eligible": False,
        "abstention_result": "3/6",
        "release_blocker_preserved": True,
    }
    binding = json.loads(
        publisher.build_target_assets(
            profile="khipu-r2",
            source_revision=revision,
            source_assets=publisher.load_assets("khipu-r2"),
            source_evidence=publisher.validate_assets(
                publisher.load_assets("khipu-r2"), "khipu-r2"
            ),
        )[publisher.SOURCE_BINDING_PATH]
    )
    assert binding["target"]["repo_id"] == "SZLHOLDINGS/KHIPU-R2"


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
