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
GitHubPayload = dict[str, Any] | list[Any]
Getter = Callable[[str], GitHubPayload]


class AuthorizationError(RuntimeError):
    """Raised when a source revision is not release-authorized."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def github_get(path: str, token: str | None) -> GitHubPayload:
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
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError) as error:
        raise AuthorizationError(f"GitHub authorization query failed: {error}") from error
    if not isinstance(payload, (dict, list)):
        raise AuthorizationError("GitHub authorization response is malformed")
    return payload


def _full_sha(value: str, field: str) -> str:
    normalized = value.strip().lower()
    if FULL_SHA_RE.fullmatch(normalized) is None:
        raise AuthorizationError(f"{field} must be an exact lowercase 40-character Git SHA")
    return normalized


def _mapping(payload: Any, description: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AuthorizationError(f"{description} response is malformed")
    return payload


def _ref_sha(getter: Getter, repository: str) -> str:
    payload = _mapping(getter(f"/repos/{repository}/git/ref/heads/main"), "Git ref")
    observed = str(_mapping(payload.get("object"), "Git ref object").get("sha", "")).lower()
    return _full_sha(observed, f"{repository} protected main")


def _protected_main(getter: Getter, repository: str, revision: str) -> dict[str, Any]:
    branch = _mapping(getter(f"/repos/{repository}/branches/main"), "branch protection")
    if branch.get("name") != "main" or branch.get("protected") is not True:
        raise AuthorizationError(f"{repository} main branch protection is not established")
    commit = _mapping(branch.get("commit"), "protected branch commit")
    observed = _full_sha(str(commit.get("sha", "")), "protected branch revision")
    if observed != revision:
        raise AuthorizationError(f"{repository} main changed during protection observation")
    return branch


def _check_contexts(checks: Any) -> set[str]:
    if not isinstance(checks, list):
        raise AuthorizationError("required-check enforcement response is malformed")
    contexts = set()
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("context"), str) or not check["context"]:
            raise AuthorizationError("required-check enforcement context is malformed")
        contexts.add(check["context"])
    return contexts


def _source_check_enforcement(getter: Getter, branch: dict[str, Any]) -> dict[str, Any]:
    # This endpoint returns active rules only, including organization rules;
    # evaluate/disabled rulesets are excluded by GitHub. Enumerate every page.
    effective: set[str] = set()
    for page in range(1, 11):
        rules = getter(f"/repos/{SOURCE_REPOSITORY}/rules/branches/main?per_page=100&page={page}")
        if not isinstance(rules, list) or len(rules) > 100:
            raise AuthorizationError("effective branch rules response is malformed")
        for rule in rules:
            if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
                raise AuthorizationError("effective branch rule is malformed")
            if rule["type"] == "required_status_checks":
                parameters = _mapping(rule.get("parameters"), "required-check enforcement")
                effective.update(_check_contexts(parameters.get("required_status_checks")))
        if len(rules) < 100:
            break
    else:
        raise AuthorizationError("effective branch rule enumeration exceeded its bound")

    # Classic branch protections are not rulesets. The branch observation's
    # enforcement level and concrete required contexts provide the legacy basis.
    classic: set[str] = set()
    protection = branch.get("protection")
    if protection is not None:
        required = _mapping(protection, "classic branch protection").get("required_status_checks")
        if required is not None:
            required = _mapping(required, "classic required-check enforcement")
            level = required.get("enforcement_level")
            if not isinstance(level, str) or level not in {"off", "non_admins", "everyone"}:
                raise AuthorizationError("classic required-check enforcement level is malformed")
            contexts = required.get("contexts", [])
            if not isinstance(contexts, list) or any(not isinstance(item, str) or not item for item in contexts):
                raise AuthorizationError("classic required-check contexts are malformed")
            declared = set(contexts) | _check_contexts(required.get("checks", []))
            if level != "off" and protection.get("enabled") is not False:
                classic.update(declared)

    if not REQUIRED_CHECKS <= effective | classic:
        raise AuthorizationError("source required-check enforcement is not established for verify canonical kernel")
    basis = []
    if effective:
        basis.append("effective_branch_rules")
    if classic:
        basis.append("classic_branch_protection")
    return {"basis": basis, "required_contexts_observed": sorted(effective | classic)}


def authorize_once(
    *,
    source_revision: str,
    publisher_revision: str,
    getter: Getter,
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

    source_branch = _protected_main(getter, SOURCE_REPOSITORY, source_main)
    _protected_main(getter, PUBLISHER_REPOSITORY, publisher_main)
    check_enforcement = _source_check_enforcement(getter, source_branch)

    commit = _mapping(getter(f"/repos/{SOURCE_REPOSITORY}/commits/{source_revision}"), "source commit")
    verification = _mapping(_mapping(commit.get("commit"), "source commit").get("verification"), "commit verification")
    if verification.get("verified") is not True:
        raise AuthorizationError(
            "source commit signature is not verified: "
            f"{verification.get('reason', 'unknown')}"
        )

    checks_payload = _mapping(getter(
        f"/repos/{SOURCE_REPOSITORY}/commits/{source_revision}/check-runs?per_page=100"
    ), "source check-run")
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
            "branch_protection_observed": True,
            "required_check_enforcement": check_enforcement,
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
            "branch_protection_observed": True,
        },
    }


def authorize_with_wait(
    *,
    source_revision: str,
    publisher_revision: str,
    wait_seconds: int,
    getter: Getter,
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
