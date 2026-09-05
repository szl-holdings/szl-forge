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


def test_card_keeps_exact_uploaded_artifact_digests() -> None:
    card = publisher.load_assets()["README.md"].decode("utf-8")
    assert "6f9f5b9df2a877c999e33faf542dc6e62ce63f4a2bf6b358fc48a4b6b113c3c9" in card
    assert "0a71b3a28b9f77ca3651f38c8caa1e34121934f5584dae24454d4c6eea823a66" in card
    assert "No deployed Alloy endpoint status is asserted by this card." in card


def test_svg_is_local_scriptless_and_bounded() -> None:
    assets = publisher.load_assets()
    publisher.validate_assets(assets)
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
    assets = publisher.load_assets()
    assets["holo-banner.svg"] = assets["holo-banner.svg"].replace(
        b"<svg ",
        b'<svg data-remote="https://evil.invalid" ',
        1,
    )
    with pytest.raises(publisher.PublicationError, match="remote SVG content"):
        publisher.validate_assets(assets)


def test_dry_run_report_is_source_bound_and_grants_no_extra_authority(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "a" * 40
    assert publisher.main(
        ["--source-revision", revision, "--report", str(report)]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["source"]["revision"] == revision
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/SZL-Khipu-1.5B"
    assert payload["target"]["revision"] is None
    assert payload["authority"]["files"] == ["README.md", "holo-banner.svg"]
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
