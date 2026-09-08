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
CheckBinding = tuple[str, int | None]


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


def _check_bindings(checks: Any, app_field: str) -> set[CheckBinding]:
    if not isinstance(checks, list):
        raise AuthorizationError("required-check enforcement response is malformed")
    bindings = set()
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("context"), str) or not check["context"]:
            raise AuthorizationError("required-check enforcement context is malformed")
        other_field = "app_id" if app_field == "integration_id" else "integration_id"
        if other_field in check or (app_field == "app_id" and app_field not in check):
            raise AuthorizationError("required-check app binding is malformed or unobserved")
        app_id = check.get(app_field)
        # Rules integration_id is optional. Classic GET may return null; -1
        # explicitly allows any app. Never coerce a boolean/string/zero to an ID.
        if app_id is None or (app_field == "app_id" and type(app_id) is int and app_id == -1):
            app_id = None
        elif type(app_id) is not int or app_id < 1:
            raise AuthorizationError("required-check app binding is malformed")
        bindings.add((check["context"], app_id))
    return bindings


def _binding_order(binding: CheckBinding) -> tuple[str, int]:
    return binding[0], -1 if binding[1] is None else binding[1]


def _source_check_enforcement(getter: Getter, branch: dict[str, Any]) -> dict[str, Any]:
    # This endpoint returns active rules only, including organization rules;
    # evaluate/disabled rulesets are excluded by GitHub. Enumerate every page.
    effective: set[CheckBinding] = set()
    for page in range(1, 11):
        rules = getter(f"/repos/{SOURCE_REPOSITORY}/rules/branches/main?per_page=100&page={page}")
        if not isinstance(rules, list) or len(rules) > 100:
            raise AuthorizationError("effective branch rules response is malformed")
        for rule in rules:
            if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
                raise AuthorizationError("effective branch rule is malformed")
            if rule["type"] == "required_status_checks":
                parameters = _mapping(rule.get("parameters"), "required-check enforcement")
                effective.update(_check_bindings(parameters.get("required_status_checks"), "integration_id"))
        if len(rules) < 100:
            break
    else:
        raise AuthorizationError("effective branch rule enumeration exceeded its bound")

    # Classic branch protections are not rulesets. The branch observation's
    # enforcement level and concrete required contexts provide the legacy basis.
    classic: set[CheckBinding] = set()
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
            declared = _check_bindings(required.get("checks", []), "app_id")
            if level != "off" and protection.get("enabled") is not False:
                # A legacy context may have been automatically bound to an app.
                # Do not silently turn a names-only summary into an any-app rule.
                if not set(contexts) <= {context for context, _app in declared}:
                    raise AuthorizationError("classic required-check app binding is unobserved")
                classic.update(declared)

    bindings = effective | classic
    if not REQUIRED_CHECKS <= {context for context, _app in bindings}:
        raise AuthorizationError("source required-check enforcement is not established for verify canonical kernel")
    basis = []
    if effective:
        basis.append("effective_branch_rules")
    if classic:
        basis.append("classic_branch_protection")
    return {"basis": basis, "required_contexts_observed": sorted({context for context, _app in bindings}),
            "required_checks_observed": [{"context": context, "app_id": app_id}
                                         for context, app_id in sorted(bindings, key=_binding_order)]}


def _verified_checks(getter: Getter, revision: str, enforcement: dict[str, Any]) -> list[dict[str, Any]]:
    bindings = [(check["context"], check["app_id"]) for check in enforcement["required_checks_observed"]]
    names = {context for context, _app in bindings}
    observed: dict[int, dict[str, Any]] = {}
    for page in range(1, 11):
        payload = _mapping(getter(
            f"/repos/{SOURCE_REPOSITORY}/commits/{revision}/check-runs?per_page=100&filter=all&page={page}"
        ), "source check-run")
        runs = payload.get("check_runs")
        if not isinstance(runs, list) or len(runs) > 100:
            raise AuthorizationError("source check-run response is malformed")
        for check in runs:
            if not isinstance(check, dict) or not isinstance(check.get("name"), str):
                raise AuthorizationError("source check-run entry is malformed")
            if check["name"] not in names:
                continue
            run_id = check.get("id")
            app_id = _mapping(check.get("app"), "source check-run app").get("id")
            if type(run_id) is not int or run_id < 1 or type(app_id) is not int or app_id < 1:
                raise AuthorizationError("source check-run/app ID is malformed")
            if check.get("head_sha") != revision:
                raise AuthorizationError("source check-run revision does not match authorized source")
            if run_id in observed and observed[run_id] != check:
                raise AuthorizationError("conflicting observations of the same source check-run ID")
            observed[run_id] = check
        if len(runs) < 100:
            break
    else:
        raise AuthorizationError("source check-run enumeration exceeded its bound")

    selected = []
    for context, required_app in bindings:
        eligible = [check for check in observed.values() if check["name"] == context
                    and (required_app is None or check["app"]["id"] == required_app)]
        if not eligible:
            raise AuthorizationError(f"required source check is missing: {context} (app {required_app})")
        latest = max(eligible, key=lambda check: check["id"])
        if latest.get("status") != "completed":
            raise AuthorizationError(f"required source check is pending: {context} (app {required_app})")
        if latest.get("conclusion") != "success":
            raise AuthorizationError(f"required source check failed: {context} (app {required_app})")
        selected.append({"name": context, "required_app_id": required_app, "app_id": latest["app"]["id"],
                         "run_id": latest["id"], "status": latest["status"], "conclusion": latest["conclusion"],
                         "details_url": latest.get("details_url")})
    return selected


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

    verified_checks = _verified_checks(getter, source_revision, check_enforcement)

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
            "checks": verified_checks,
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
