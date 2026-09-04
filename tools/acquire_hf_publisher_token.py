#!/usr/bin/env python3
"""Acquire one valid Hugging Face publisher token without exposing credentials.

The selector prefers a repo-scoped Trusted Publisher token, then validates each
configured fallback independently. An expired earlier secret therefore cannot
mask a later valid organization write token. Reports contain source labels,
status codes, request IDs, and hashes only; token bytes are never persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence


TOKEN_ENV_ORDER: tuple[tuple[str, str], ...] = (
    ("HF_ORG_TOKEN", "HF_ORG_TOKEN_CANDIDATE"),
    ("HF_WRITE_TOKEN", "HF_WRITE_TOKEN_CANDIDATE"),
    ("HF_TOKEN", "HF_TOKEN_CANDIDATE"),
    ("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN_CANDIDATE"),
    ("HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN_CANDIDATE"),
)
REQUEST_ID = re.compile(r"Request ID:\s*([^\n)]+)", re.IGNORECASE)
TOKEN_LINE = re.compile(r"^hf_[A-Za-z0-9._-]+$")


class CredentialSelectionError(RuntimeError):
    """No configured credential passed active Hub validation."""


class OidcExchangeError(RuntimeError):
    """The GitHub OIDC to Hugging Face token exchange failed."""


@dataclass(frozen=True)
class ValidationResult:
    source: str
    identity_sha256: str
    target_access: str


@dataclass(frozen=True)
class Attempt:
    source: str
    present: bool
    valid: bool
    failure_type: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    failure_sha256: str | None = None
    target_access: str | None = None


def _failure_evidence(source: str, error: BaseException) -> Attempt:
    text = str(error)
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    request_match = REQUEST_ID.search(text)
    return Attempt(
        source=source,
        present=True,
        valid=False,
        failure_type=type(error).__name__,
        status_code=status_code if isinstance(status_code, int) else None,
        request_id=request_match.group(1).strip() if request_match else None,
        failure_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _normalize_token(raw: str) -> str:
    token = str(raw or "").strip()
    if not token or any(character.isspace() for character in token):
        raise CredentialSelectionError("credential is empty or contains whitespace")
    if not TOKEN_LINE.fullmatch(token):
        raise CredentialSelectionError("credential does not match a Hugging Face token shape")
    return token


def acquire_oidc_token(resource: str) -> str:
    """Request a fresh repo-scoped token through the official ``hf`` CLI."""

    environment = os.environ.copy()
    for key in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        environment.pop(key, None)
    environment["HF_OIDC_RESOURCE"] = resource
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    process = subprocess.run(
        ["hf", "auth", "token"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if process.returncode != 0:
        raise OidcExchangeError(process.stderr.strip() or "OIDC exchange failed")
    candidates = [
        line.strip()
        for line in process.stdout.splitlines()
        if TOKEN_LINE.fullmatch(line.strip())
    ]
    if len(candidates) != 1:
        raise OidcExchangeError("OIDC exchange did not return exactly one token")
    return _normalize_token(candidates[0])


def validate_token(
    token: str,
    *,
    source: str,
    target_repo: str,
    target_type: str,
    allow_create: bool,
) -> ValidationResult:
    """Actively validate identity and target write access without caching a token."""

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    normalized = _normalize_token(token)
    api = HfApi(token=normalized)
    identity = api.whoami()
    identity_name = str((identity or {}).get("name") or "").strip()
    if not identity_name:
        raise CredentialSelectionError("Hub identity response did not contain a name")

    target_access = "EXISTING_WRITE_CONFIRMED"
    try:
        api.auth_check(repo_id=target_repo, repo_type=target_type, write=True)
    except RepositoryNotFoundError:
        if source == "TRUSTED_PUBLISHER" or not allow_create:
            raise
        target_access = "CREATE_OR_RECOVER_REQUIRED"

    return ValidationResult(
        source=source,
        identity_sha256=hashlib.sha256(identity_name.encode("utf-8")).hexdigest(),
        target_access=target_access,
    )


def select_credential(
    *,
    resource: str | None,
    target_repo: str,
    target_type: str,
    allow_create: bool,
    environment: Mapping[str, str],
    oidc_supplier: Callable[[str], str] = acquire_oidc_token,
    validator: Callable[..., ValidationResult] = validate_token,
) -> tuple[str, ValidationResult, list[Attempt]]:
    """Return the first actively validated credential in deterministic order."""

    attempts: list[Attempt] = []
    if resource:
        try:
            oidc = oidc_supplier(resource)
            result = validator(
                oidc,
                source="TRUSTED_PUBLISHER",
                target_repo=target_repo,
                target_type=target_type,
                allow_create=False,
            )
            attempts.append(
                Attempt(
                    source=result.source,
                    present=True,
                    valid=True,
                    target_access=result.target_access,
                )
            )
            return oidc, result, attempts
        except Exception as error:  # diagnostic only; fallbacks remain eligible
            attempts.append(_failure_evidence("TRUSTED_PUBLISHER", error))

    for source, variable in TOKEN_ENV_ORDER:
        raw = str(environment.get(variable) or "").strip()
        if not raw:
            attempts.append(Attempt(source=source, present=False, valid=False))
            continue
        try:
            token = _normalize_token(raw)
            result = validator(
                token,
                source=source,
                target_repo=target_repo,
                target_type=target_type,
                allow_create=allow_create,
            )
            attempts.append(
                Attempt(
                    source=result.source,
                    present=True,
                    valid=True,
                    target_access=result.target_access,
                )
            )
            return token, result, attempts
        except Exception as error:
            attempts.append(_failure_evidence(source, error))

    raise CredentialSelectionError("no valid Hugging Face publisher credential")


def _write_report(
    path: Path,
    *,
    target_repo: str,
    target_type: str,
    resource: str | None,
    selected: ValidationResult | None,
    attempts: Sequence[Attempt],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "szl.hf-publisher-credential/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": {"repo_id": target_repo, "repo_type": target_type},
        "oidc": {
            "requested": resource is not None,
            "resource": resource,
            "issuer": "https://token.actions.githubusercontent.com",
            "audience": "https://huggingface.co",
        },
        "selected": asdict(selected) if selected is not None else None,
        "attempts": [asdict(attempt) for attempt in attempts],
        "token_persisted": False,
        "token_logged": False,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_github_environment(path: Path, token: str, source: str) -> None:
    if "\n" in token or "\r" in token:
        raise CredentialSelectionError("credential contains a newline")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"HF_TOKEN={token}\n")
        handle.write(f"HF_TOKEN_SOURCE={source}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument(
        "--target-type",
        required=True,
        choices=("model", "dataset", "space"),
    )
    parser.add_argument("--oidc-resource")
    parser.add_argument("--allow-create", action="store_true")
    parser.add_argument("--github-env", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    attempts: list[Attempt] = []
    selected: ValidationResult | None = None
    try:
        token, selected, attempts = select_credential(
            resource=args.oidc_resource,
            target_repo=args.target_repo,
            target_type=args.target_type,
            allow_create=args.allow_create,
            environment=os.environ,
        )
        print(f"::add-mask::{token}")
        _append_github_environment(args.github_env, token, selected.source)
        _write_report(
            args.report,
            target_repo=args.target_repo,
            target_type=args.target_type,
            resource=args.oidc_resource,
            selected=selected,
            attempts=attempts,
        )
        print(
            "Hugging Face credential validated: "
            f"source={selected.source} target_access={selected.target_access}"
        )
        return 0
    except Exception as error:
        _write_report(
            args.report,
            target_repo=args.target_repo,
            target_type=args.target_type,
            resource=args.oidc_resource,
            selected=selected,
            attempts=attempts,
        )
        print(
            "::error::Hugging Face publisher credential validation failed: "
            f"{type(error).__name__}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
