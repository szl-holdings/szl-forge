#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish the reviewed Chaski model card as an exact-source Hub commit.

Only ``README.md`` and ``holo-banner.svg`` are written. Model weights, configs,
evaluation evidence, visibility, hardware, collections, and runtime state are
outside this publisher's authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "SZLHOLDINGS/chaski"
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
SOURCE_FILES = {
    "README.md": ROOT / "chaski" / "card" / "README.md",
    "holo-banner.svg": ROOT / "chaski" / "card" / "holo-banner.svg",
}
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CARD_BOUNDARIES = (
    "base_model_relation: finetune",
    "artifact_class: MERGED_FINETUNE",
    "originality: FINETUNE_DISCLOSED_BASE",
    "publication_eligible: false",
    "json_draft: 0/5",
    "adversarial_refusal: 2/6",
    "Conjecture 1",
    "Failed qualification. Not flagship.",
)


class PublicationError(RuntimeError):
    """Exact-source card publication could not be completed safely."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_assets() -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    for target, source in SOURCE_FILES.items():
        if not source.is_file() or source.is_symlink():
            raise PublicationError(f"source asset missing or unsafe: {source}")
        assets[target] = source.read_bytes()
    return assets


def validate_assets(assets: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    if set(assets) != set(SOURCE_FILES):
        raise PublicationError("source asset set drifted")
    try:
        card = assets["README.md"].decode("utf-8")
        banner = assets["holo-banner.svg"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationError("card assets must be UTF-8") from error

    if not card.startswith("---\n") or card.count("\n---\n") < 1:
        raise PublicationError("model card front matter is missing")
    for boundary in REQUIRED_CARD_BOUNDARIES:
        if boundary not in card:
            raise PublicationError(f"required model-card boundary missing: {boundary}")
    for forbidden in (
        "publication_eligible: true",
        "Named-N-MEASURED%20PASS",
        "json_draft: 5/5",
        "adversarial_refusal: 6/6",
    ):
        if forbidden in card:
            raise PublicationError(f"forbidden qualification claim present: {forbidden}")

    normalized_banner = banner.lstrip()
    if not normalized_banner.startswith("<svg"):
        raise PublicationError("holographic banner is not an SVG document")
    for required in ('viewBox="0 0 1200 360"', 'id="holo"', 'id="fade"'):
        if required not in banner:
            raise PublicationError(f"required SVG contract missing: {required}")
    lowered = banner.lower()
    namespace = 'xmlns="http://www.w3.org/2000/svg"'
    if lowered.count(namespace) != 1:
        raise PublicationError("canonical SVG namespace must appear exactly once")
    remote_scan = lowered.replace(namespace, "", 1)
    for forbidden in ("<script", "javascript:", "http://", "https://"):
        if forbidden in remote_scan:
            raise PublicationError(f"unsafe or remote SVG content present: {forbidden}")

    return {
        target: {"bytes": len(value), "sha256": sha256_bytes(value)}
        for target, value in sorted(assets.items())
    }


def _retry_after_seconds(error: BaseException, default: int) -> int:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = str(headers.get("Retry-After") or "").strip()
    try:
        seconds = int(raw)
    except ValueError:
        seconds = default
    return max(1, min(seconds, 180))


def publish_with_bounded_retry(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    if attempts < 1 or attempts > 5:
        raise ValueError("attempts must be between one and five")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status != 429 or attempt == attempts:
                raise
            sleeper(_retry_after_seconds(error, default=30 * attempt))
    raise AssertionError("bounded retry loop exhausted without a result")


def publish(
    *,
    token: str,
    source_revision: str,
    assets: dict[str, bytes],
) -> tuple[str, dict[str, dict[str, Any]], str]:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    api.auth_check(repo_id=REPO_ID, repo_type="model", write=True)
    identity = api.whoami()
    publisher = str((identity or {}).get("name") or "").strip()
    if not publisher:
        raise PublicationError("publisher identity is unavailable")

    operations = [
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=value)
        for target, value in sorted(assets.items())
    ]

    commit = publish_with_bounded_retry(
        lambda: api.create_commit(
            repo_id=REPO_ID,
            repo_type="model",
            operations=operations,
            commit_message=f"docs: publish Chaski card from szl-forge@{source_revision}",
            commit_description=(
                "Exact-source card publication. Named-N remains a measured failed "
                "gate and publication_eligible remains false. No weights, configs, "
                "evals, visibility, or hardware changed."
            ),
        )
    )
    revision = str(getattr(commit, "oid", "") or "").strip().lower()
    if not FULL_SHA.fullmatch(revision):
        raise PublicationError("Hub commit did not return an exact revision")

    readback: dict[str, dict[str, Any]] = {}
    for target, expected in sorted(assets.items()):
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="model",
            filename=target,
            revision=revision,
            token=token,
            force_download=True,
        )
        observed = Path(local).read_bytes()
        if observed != expected:
            raise PublicationError(f"Hub byte readback mismatch: {target}")
        readback[target] = {
            "bytes": len(observed),
            "sha256": sha256_bytes(observed),
        }
    return revision, readback, publisher


def build_report(
    *,
    state: str,
    source_revision: str,
    source_assets: dict[str, dict[str, Any]],
    hub_revision: str | None = None,
    hub_assets: dict[str, dict[str, Any]] | None = None,
    publisher: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "szl.hf.chaski-card-publication/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "assets": source_assets,
        },
        "target": {
            "repo_id": REPO_ID,
            "repo_type": "model",
            "revision": hub_revision,
            "assets": hub_assets,
        },
        "publisher_identity": publisher,
        "authority": {
            "files": sorted(SOURCE_FILES),
            "weights_changed": False,
            "configs_changed": False,
            "evals_changed": False,
            "visibility_changed": False,
            "hardware_changed": False,
        },
        "secret_values_recorded": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    source_revision = args.source_revision.strip().lower()
    if not FULL_SHA.fullmatch(source_revision):
        raise SystemExit("--source-revision must be a full lowercase commit SHA")
    assets = load_assets()
    source_evidence = validate_assets(assets)

    if not args.publish:
        write_report(
            args.report,
            build_report(
                state="DRY_RUN_VALIDATED",
                source_revision=source_revision,
                source_assets=source_evidence,
            ),
        )
        return 0

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required for --publish")
    try:
        hub_revision, readback, publisher = publish(
            token=token,
            source_revision=source_revision,
            assets=assets,
        )
        report = build_report(
            state="SOURCE_BOUND_READBACK_VERIFIED",
            source_revision=source_revision,
            source_assets=source_evidence,
            hub_revision=hub_revision,
            hub_assets=readback,
            publisher=publisher,
        )
        write_report(args.report, report)
        print(
            json.dumps(
                {
                    "state": report["state"],
                    "source_revision": source_revision,
                    "hub_revision": hub_revision,
                    "files": sorted(readback),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        write_report(
            args.report,
            build_report(
                state="FAILED",
                source_revision=source_revision,
                source_assets=source_evidence,
            )
            | {
                "error_type": type(error).__name__,
                "error_sha256": sha256_bytes(str(error).encode("utf-8")),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
