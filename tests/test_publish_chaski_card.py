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


def test_svg_is_local_scriptless_and_bounded() -> None:
    assets = publisher.load_assets()
    publisher.validate_assets(assets)
    svg = assets["holo-banner.svg"].decode("utf-8").lower()
    assert svg.lstrip().startswith("<svg")
    assert '<script' not in svg
    assert "javascript:" not in svg
    assert "http://" not in svg
    assert "https://" not in svg
    assert len(assets["holo-banner.svg"]) < 10_000


def test_dry_run_report_is_source_bound_and_grants_no_extra_authority(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "a" * 40
    assert publisher.main(
        ["--source-revision", revision, "--report", str(report)]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["source"]["revision"] == revision
    assert payload["target"]["revision"] is None
    assert payload["authority"]["files"] == ["README.md", "holo-banner.svg"]
    for field in (
        "weights_changed",
        "configs_changed",
        "evals_changed",
        "visibility_changed",
        "hardware_changed",
    ):
        assert payload["authority"][field] is False
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
