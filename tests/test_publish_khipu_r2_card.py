# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import publish_khipu_r2_card as publisher


class RateLimited(RuntimeError):
    def __init__(self, retry_after: str = "1") -> None:
        super().__init__("provider rate limited")
        self.response = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": retry_after},
        )


def test_live_source_card_preserves_measured_results_and_closed_gates() -> None:
    assets = publisher.load_assets()
    evidence = publisher.validate_assets(assets)
    assert set(evidence) == {"README.md", "holo-banner.svg"}
    card = assets["README.md"].decode("utf-8")
    assert "publication_eligible: false" in card
    assert "autonomy_eligible: false" in card
    assert "never_overwrite: SZLHOLDINGS/SZL-Khipu-1.5B" in card
    assert "| plan-valid | **11 / 11** | not a public leaderboard |" in card
    assert "| grounding (`eval.jsonl` navigate) | **5 / 5** | n=5 |" in card
    assert "| abstain (`adversarial.jsonl`) | **3 / 6** | not 5/5, not 6/6 |" in card
    assert "| hallucinated citations | **0** | this job only |" in card
    assert "No signed R2 eval receipt in this atelier." in card
    assert "Do not derive a world-rank score" in card
    assert "publication_eligible: true" not in card
    assert "autonomy_eligible: true" not in card


def test_card_is_bound_to_canonical_khipu_r2_evidence() -> None:
    evidence = publisher.load_canonical_evidence()
    text = evidence.decode("utf-8")
    assert "6a91bf11984507d9db4ea104" in text
    assert "MEASURED 3/6" in text
    assert "MEASURED 5/5" in text
    assert "MEASURED 11/11" in text
    assert "publication_eligible: false" in text
    assert "No Hub PUT." in text
    assert len(publisher.sha256_bytes(evidence)) == 64


def test_card_keeps_exact_adapter_digest_and_non_inheritance_boundary() -> None:
    card = publisher.load_assets()["README.md"].decode("utf-8")
    assert "e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349" in card
    assert "Mini does **not** inherit this 3/6" in card
    assert "not a pass" in card


def test_unbounded_promotional_or_promotion_claims_are_rejected() -> None:
    for claim in (
        "Nobody else ships this combination",
        "publication_eligible: true",
        "autonomy_eligible: true",
        "best in the world",
    ):
        assets = publisher.load_assets()
        assets["README.md"] += f"\n{claim}\n".encode("utf-8")
        with pytest.raises(publisher.PublicationError, match="forbidden qualification claim"):
            publisher.validate_assets(assets)


def test_missing_canonical_evidence_boundary_is_rejected() -> None:
    assets = publisher.load_assets()
    evidence = publisher.load_canonical_evidence().replace(
        b"publication_eligible: false",
        b"publication state unavailable",
        1,
    )
    with pytest.raises(publisher.PublicationError, match="canonical evidence boundary"):
        publisher.validate_assets(assets, evidence)


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


def test_dry_run_report_is_source_and_evidence_bound(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    revision = "a" * 40
    assert publisher.main(
        ["--source-revision", revision, "--report", str(report)]
    ) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["source"]["revision"] == revision
    assert payload["source"]["evidence"]["path"] == "khipu_r2/README.md"
    assert len(payload["source"]["evidence"]["sha256"]) == 64
    assert payload["target"]["repo_id"] == "SZLHOLDINGS/KHIPU-R2"
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
        "autonomy_eligible": False,
        "abstention_result": "3/6",
        "abstention_pass": False,
        "signed_r2_eval_receipt_observed": False,
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
