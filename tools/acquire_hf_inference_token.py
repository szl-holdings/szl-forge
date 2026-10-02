#!/usr/bin/env python3
"""Select an authenticated Hugging Face credential for bounded inference.

Each configured secret is tested independently. Token bytes are masked and only
written to the GitHub Actions environment file; reports contain hashes and safe
status metadata, never credentials. Identity and declared token permissions are
checked without model execution; the real evaluation proves target capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

ROUTER_MODELS_URL = "https://router.huggingface.co/v1/models"
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
INFERENCE_PERMISSION = "inference.serverless.write"
TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9._-]+$")
TOKEN_ENV_ORDER: tuple[tuple[str, str], ...] = (
    ("HF_INFERENCE_TOKEN", "HF_INFERENCE_TOKEN_CANDIDATE"),
    ("HF_ORG_TOKEN", "HF_ORG_TOKEN_CANDIDATE"),
    ("HF_ORG_TOKEN1", "HF_ORG_TOKEN1_CANDIDATE"),
    ("HF_WRITE_TOKEN", "HF_WRITE_TOKEN_CANDIDATE"),
    ("HF_TOKEN", "HF_TOKEN_CANDIDATE"),
    ("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN_CANDIDATE"),
    ("HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN_CANDIDATE"),
)


@dataclass(frozen=True)
class Attempt:
    source: str
    present: bool
    valid: bool
    status_code: int | None = None
    target_model_listed: bool | None = None
    response_sha256: str | None = None
    failure_type: str | None = None
    failure_sha256: str | None = None
    identity_verified: bool = False
    identity_sha256: str | None = None
    identity_status_code: int | None = None
    token_role: str | None = None
    inference_scope: str | None = None


class TokenSelectionError(RuntimeError):
    def __init__(self, message: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


def normalize_token(value: str) -> str:
    token = str(value or "").strip()
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("token is missing or has an invalid shape")
    return token


def identity_permission(identity: object) -> tuple[str, str, str]:
    """Accept only authenticated user-token roles with declared inference scope.

    HF documents legacy read/write roles for inference and the fine-grained
    permission at https://huggingface.co/docs/inference-providers/index.
    Repository-scoped permissions alone do not grant serverless inference.
    """
    if not isinstance(identity, dict) or identity.get("type") != "user":
        raise ValueError("identity response does not identify a user")
    name = identity.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("identity response has no user name")
    auth = identity.get("auth")
    if not isinstance(auth, dict) or auth.get("type") != "access_token":
        raise ValueError("identity response does not authenticate an access token")
    access = auth.get("accessToken")
    if not isinstance(access, dict):
        raise ValueError("identity response has no access-token permissions")
    role = access.get("role")
    if role in ("read", "write"):
        scope = "LEGACY_READ_WRITE_ROLE"
    elif role in ("fineGrained", "fine-grained"):
        fine = access.get("fineGrained")
        permissions = fine.get("global") if isinstance(fine, dict) else None
        if (
            not isinstance(permissions, list)
            or not all(isinstance(item, str) for item in permissions)
            or INFERENCE_PERMISSION not in permissions
        ):
            raise ValueError("token lacks the global Inference Providers permission")
        scope = "FINE_GRAINED_INFERENCE_PROVIDERS"
    else:
        raise ValueError("identity response has an unrecognized token role")
    return hashlib.sha256(name.encode()).hexdigest(), role, scope


def validate_token(token: str, target_model: str, timeout: float) -> Attempt:
    evidence: dict[str, object] = {}
    try:
        token = normalize_token(token)
        headers = {"authorization": f"Bearer {token}", "accept": "application/json"}
        identity_request = urllib.request.Request(WHOAMI_URL, headers=headers, method="GET")
        with urllib.request.urlopen(identity_request, timeout=timeout) as response:
            status = int(response.status)
            evidence["identity_status_code"] = status
            if not 200 <= status < 300:
                raise ValueError("identity endpoint did not return a successful status")
            raw_identity = response.read(1_000_001)
            if len(raw_identity) > 1_000_000:
                raise ValueError("identity response exceeds size limit")
            identity_hash, role, scope = identity_permission(json.loads(raw_identity))
            evidence.update(
                identity_verified=True,
                identity_sha256=identity_hash,
                token_role=role,
                inference_scope=scope,
            )
        # The catalog is public, including when an invalid bearer is supplied.
        # Keep it as a reachability/model-listing diagnostic after authentication.
        request = urllib.request.Request(ROUTER_MODELS_URL, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(8_000_000)
            text = raw.decode("utf-8", "replace")
            return Attempt(
                source="",
                present=True,
                valid=200 <= int(response.status) < 300,
                status_code=int(response.status),
                target_model_listed=target_model in text,
                response_sha256=hashlib.sha256(raw).hexdigest(),
                **evidence,
            )
    except urllib.error.HTTPError as error:
        try:
            raw = error.read(64_000)
        except Exception:
            raw = b""
        safe = f"HTTPError:{error.code}:{hashlib.sha256(raw).hexdigest()}"
        return Attempt(
            source="",
            present=True,
            valid=False,
            status_code=int(error.code),
            failure_type="HTTPError",
            failure_sha256=hashlib.sha256(safe.encode()).hexdigest(),
            **evidence,
        )
    except Exception as error:
        safe = f"{type(error).__name__}:{str(error)[:300]}"
        return Attempt(
            source="",
            present=True,
            valid=False,
            failure_type=type(error).__name__,
            failure_sha256=hashlib.sha256(safe.encode()).hexdigest(),
            **evidence,
        )


def select(
    environment: Mapping[str, str],
    *,
    target_model: str,
    timeout: float,
) -> tuple[str, str, list[Attempt]]:
    attempts: list[Attempt] = []
    for source, variable in TOKEN_ENV_ORDER:
        raw = str(environment.get(variable) or "").strip()
        if not raw:
            attempts.append(Attempt(source=source, present=False, valid=False))
            continue
        try:
            token = normalize_token(raw)
        except Exception as error:
            safe = f"{type(error).__name__}:{error}"
            attempts.append(
                Attempt(
                    source=source,
                    present=True,
                    valid=False,
                    failure_type=type(error).__name__,
                    failure_sha256=hashlib.sha256(safe.encode()).hexdigest(),
                )
            )
            continue
        observed = validate_token(token, target_model, timeout)
        attempt = replace(observed, source=source)
        attempts.append(attempt)
        # Authentication and declared permissions precede catalog access. The
        # real evaluation remains the target-model/credit/route capability check.
        if attempt.valid:
            return token, source, attempts
    raise TokenSelectionError(
        "no configured token passed identity and inference permission checks",
        attempts,
    )


def write_report(
    path: Path,
    *,
    target_model: str,
    selected_source: str | None,
    attempts: Sequence[Attempt],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "szl.hf-inference-credential-selection.v1",
                "router": ROUTER_MODELS_URL,
                "identity_endpoint": WHOAMI_URL,
                "validation_boundary": "authenticated identity and declared permissions; no inference executed",
                "catalog_is_authentication_evidence": False,
                "target_inference_verified": False,
                "target_model": target_model,
                "selected_source": selected_source,
                "attempts": [asdict(attempt) for attempt in attempts],
                "token_logged": False,
                "token_persisted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def append_github_env(path: Path, token: str, source: str) -> None:
    if "\n" in token or "\r" in token:
        raise ValueError("token contains a newline")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"HF_INFERENCE_TOKEN={token}\n")
        handle.write(f"HF_INFERENCE_TOKEN_SOURCE={source}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--github-env", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    attempts: list[Attempt] = []
    source: str | None = None
    try:
        token, source, attempts = select(
            os.environ,
            target_model=args.target_model,
            timeout=args.timeout_seconds,
        )
        print(f"::add-mask::{token}")
        append_github_env(args.github_env, token, source)
        write_report(
            args.report,
            target_model=args.target_model,
            selected_source=source,
            attempts=attempts,
        )
        print(
            f"Hugging Face identity and declared permissions validated: source={source}; "
            "target inference remains unverified"
        )
        return 0
    except TokenSelectionError as error:
        attempts = list(error.attempts)
        write_report(
            args.report,
            target_model=args.target_model,
            selected_source=None,
            attempts=attempts,
        )
        print("::error::No configured token passed identity, permission and catalog checks")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
