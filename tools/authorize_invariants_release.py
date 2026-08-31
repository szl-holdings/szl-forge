#!/usr/bin/env python3
"""Authorize one exact szl-invariants release before exposing a publisher secret."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


API_ROOT = "https://api.github.com"
SOURCE_REPOSITORY = "szl-holdings/szl-invariants"
PUBLISHER_REPOSITORY = "szl-holdings/szl-forge"
REQUIRED_CHECKS = frozenset({"verify canonical kernel"})
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class AuthorizationError(RuntimeError):
    """Raised when a source revision is not release-authorized."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def github_get(path: str, token: str | None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "szl-forge-invariants-release-gateway",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{API_ROOT}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
        raise AuthorizationError(f"GitHub authorization query failed: {error}") from error
    if not isinstance(payload, dict):
        raise AuthorizationError("GitHub authorization response is malformed")
    return payload


def _full_sha(value: str, field: str) -> str:
    normalized = value.strip().lower()
    if FULL_SHA_RE.fullmatch(normalized) is None:
        raise AuthorizationError(f"{field} must be an exact lowercase 40-character Git SHA")
    return normalized


def _ref_sha(getter: Callable[[str], dict[str, Any]], repository: str) -> str:
    payload = getter(f"/repos/{repository}/git/ref/heads/main")
    observed = str(payload.get("object", {}).get("sha", "")).lower()
    return _full_sha(observed, f"{repository} protected main")


def authorize_once(
    *,
    source_revision: str,
    publisher_revision: str,
    getter: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    source_revision = _full_sha(source_revision, "source_revision")
    publisher_revision = _full_sha(publisher_revision, "publisher_revision")
    source_main = _ref_sha(getter, SOURCE_REPOSITORY)
    publisher_main = _ref_sha(getter, PUBLISHER_REPOSITORY)
    if source_revision != source_main:
        raise AuthorizationError(
            "source revision is not current protected main "
            f"(requested {source_revision}, protected main {source_main})"
        )
    if publisher_revision != publisher_main:
        raise AuthorizationError(
            "publisher revision is not current protected main "
            f"(run {publisher_revision}, protected main {publisher_main})"
        )

    commit = getter(f"/repos/{SOURCE_REPOSITORY}/commits/{source_revision}")
    verification = commit.get("commit", {}).get("verification", {})
    if verification.get("verified") is not True:
        raise AuthorizationError(
            "source commit signature is not verified: "
            f"{verification.get('reason', 'unknown')}"
        )

    checks_payload = getter(
        f"/repos/{SOURCE_REPOSITORY}/commits/{source_revision}/check-runs?per_page=100"
    )
    check_runs = checks_payload.get("check_runs")
    if not isinstance(check_runs, list):
        raise AuthorizationError("source check-run response is malformed")
    latest: dict[str, dict[str, Any]] = {}
    for check in check_runs:
        if isinstance(check, dict) and check.get("name") in REQUIRED_CHECKS:
            latest[str(check["name"])] = check
    missing = sorted(REQUIRED_CHECKS - set(latest))
    if missing:
        raise AuthorizationError(f"required source checks are missing: {missing}")
    pending = sorted(
        name for name, check in latest.items() if check.get("status") != "completed"
    )
    failed = sorted(
        name
        for name, check in latest.items()
        if check.get("status") == "completed" and check.get("conclusion") != "success"
    )
    if failed:
        raise AuthorizationError(f"required source checks failed: {failed}")
    if pending:
        raise AuthorizationError(f"required source checks are pending: {pending}")

    return {
        "schema": "szl.invariants-release-authorization/v1",
        "status": "AUTHORIZED_PROTECTED_MAIN",
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "protected_main": source_main,
            "signature_verified": True,
            "signature_reason": verification.get("reason"),
            "checks": [
                {
                    "name": name,
                    "status": latest[name]["status"],
                    "conclusion": latest[name]["conclusion"],
                    "details_url": latest[name].get("details_url"),
                }
                for name in sorted(REQUIRED_CHECKS)
            ],
        },
        "publisher": {
            "repository": PUBLISHER_REPOSITORY,
            "revision": publisher_revision,
            "protected_main": publisher_main,
        },
    }


def authorize_with_wait(
    *,
    source_revision: str,
    publisher_revision: str,
    wait_seconds: int,
    getter: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return authorize_once(
                source_revision=source_revision,
                publisher_revision=publisher_revision,
                getter=getter,
            )
        except AuthorizationError as error:
            retryable = "pending" in str(error) or "missing" in str(error)
            if not retryable or time.monotonic() >= deadline:
                raise
            time.sleep(min(10, max(1, wait_seconds)))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--publisher-revision", required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/invariants-release-authorization.json"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.wait_seconds < 0 or args.wait_seconds > 1800:
        raise AuthorizationError("wait-seconds must be between 0 and 1800")
    token = os.getenv("GITHUB_TOKEN")
    result = authorize_with_wait(
        source_revision=args.source_revision,
        publisher_revision=args.publisher_revision,
        wait_seconds=args.wait_seconds,
        getter=lambda path: github_get(path, token),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(canonical_json(result), encoding="utf-8")
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
