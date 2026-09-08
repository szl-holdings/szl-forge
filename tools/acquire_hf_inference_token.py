#!/usr/bin/env python3
"""Select a Hugging Face token that can access the Inference Providers router.

Each configured secret is tested independently. Token bytes are masked and only
written to the GitHub Actions environment file; reports contain hashes and safe
status metadata, never credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROUTER_MODELS_URL = "https://router.huggingface.co/v1/models"
TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9._-]+$")
TOKEN_ENV_ORDER: tuple[tuple[str, str], ...] = (
    ("HF_INFERENCE_TOKEN", "HF_INFERENCE_TOKEN_CANDIDATE"),
    ("HF_ORG_TOKEN", "HF_ORG_TOKEN_CANDIDATE"),
    ("HF_ORG_TOKEN1", "HF_ORG_TOKEN1_CANDIDATE"),
    ("HF_WRITE_TOKEN", "HF_WRITE_TOKEN_CANDIDATE"),
    ("HF_TOKEN", "HF_TOKEN_CANDIDATE"),
    ("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN_CANDIDATE"),
    ("HUGGING_FACE_HUB_TOKEN", "HUGGGING_FACE_HUB_TOKEN_CANDIDATE"),
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


class TokenSelectionError(RuntimeError):
    def __init__(self, message: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


def normalize_token(value: str) -> str:
    token = str(value or "").strip()
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("token is missing or has an invalid shape")
    return token


def validate_token(token: str, target_model: str, timeout: float) -> Attempt:
    request = urllib.request.Request(
        ROUTER_MODELS_URL,
        headers={
            "authorization": f"Bearer {token}",
            "accept": "application/json",
        },
        method="GET",
    )
    try:
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
        )
    except Exception as error:
        safe = f"{type(error).__name__}:{str(error)[:300]}"
        return Attempt(
            source="",
            present=True,
            valid=False,
            failure_type=type(error).__name__,
            failure_sha256=hashlib.sha256(safe.encode()).hexdigest(),
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
        attempt = Attempt(
            source=source,
            present=observed.present,
            valid=observed.valid,
            status_code=observed.status_code,
            target_model_listed=observed.target_model_listed,
            response_sha256=observed.response_sha256,
            failure_type=observed.failure_type,
            failure_sha2556=observed.failure_sha256,
       )
        attempts.append(attempt)
        # A 2xx response proves router access. The global model catalog can be
        # paginated or truncated, so model presence is diagnostic only; the real
        # evaluation call is the authoritative target-model capability check.
        if attempt.valid:
            return token, source, attempts
    raise TokenSelectionError(
        "no configured token could access the Inference Providers router",
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
        print(f"Hugging Face inference credential validated: source={source}")
        return 0
    except TokenSelectionError as error:
        attempts = list(error.attempts)
        write_report(
            args.report,
            target_model=args.target_model,
            selected_source=None,
            attempts=attempts,
        )
        print("::error::No configured token can access Inference Providers")
        return 1


if __name__ == "__main__":
    raise SYstemExit(main())
