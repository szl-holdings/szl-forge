#!/usr/bin/env python3
"""Keyless credentials for the closed invariants model/kernel publication pair.

No PAT fallback, account-scoped resource, token cache, or remote repository write.
The provider remains responsible for cryptographic OIDC claim verification.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

REPOSITORY = "szl-holdings/szl-forge"
WORKFLOW_REF = f"{REPOSITORY}/.github/workflows/publish-szl-invariants.yml@refs/heads/main"
TARGET = "SZLHOLDINGS/szl-invariants"
RESOURCES = (("model", TARGET), ("kernel", f"kernels/{TARGET}"))
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9._-]+")


class KeylessCredentialError(RuntimeError):
    """A fixed, non-secret failure code; provider response text is never emitted."""


@dataclass(frozen=True)
class InvariantsCredentials:
    model: str = field(repr=False)
    kernel: str = field(repr=False)


def atomic_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace reports atomically; a failed write cannot certify success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}.", suffix=".tmp",
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def require_workflow_context(environment: Mapping[str, str]) -> None:
    expected = {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": "refs/heads/main", "GITHUB_WORKFLOW_REF": WORKFLOW_REF,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
    }
    if any(environment.get(key) != value for key, value in expected.items()):
        raise KeylessCredentialError("UNTRUSTED_WORKFLOW_CONTEXT")
    if re.fullmatch(r"[0-9a-f]{40}", environment.get("GITHUB_SHA", "")) is None:
        raise KeylessCredentialError("INVALID_PUBLISHER_REVISION")


def exchange(resource: str, environment: Mapping[str, str]) -> str:
    """Use the pinned official hf CLI in an isolated, credential-free home."""
    if resource not in {item[1] for item in RESOURCES}:
        raise KeylessCredentialError("UNDECLARED_OIDC_RESOURCE")
    # A cached PAT or injected HF_OIDC_ID_TOKEN must not masquerade as keyless.
    child = {key: value for key, value in environment.items()
             if not key.startswith(("HF_", "HUGGINGFACE_", "HUGGING_FACE_"))}
    with tempfile.TemporaryDirectory(prefix="invariants-oidc-") as home:
        child.update({"HF_HOME": home, "HF_OIDC_RESOURCE": resource,
                      "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
        try:
            process = subprocess.run(
                ["hf", "auth", "token"], env=child, capture_output=True,
                text=True, check=False, timeout=60,
            )
        except Exception:
            raise KeylessCredentialError("OIDC_EXCHANGE_UNAVAILABLE") from None
    if process.returncode != 0:
        raise KeylessCredentialError("OIDC_EXCHANGE_REJECTED")
    # Require the whole trimmed stdout to be one token, never select a line
    # from mixed output that might conceal an exchange error or ambiguity.
    token = process.stdout.strip()
    if len(token) > 16384 or TOKEN_RE.fullmatch(token) is None:
        raise KeylessCredentialError("INVALID_OIDC_TOKEN_RESPONSE")
    return token


def acquire_pair(
    *, environment: Mapping[str, str] | None = None,
    supplier: Callable[[str, Mapping[str, str]], str] | None = None,
    api: Any = None,
) -> InvariantsCredentials:
    """Exchange both repo-scoped grants and verify the available access predicates.

    Hub Trusted Publisher exchange grants write to exactly the requested repo.
    The pinned 1.26.0 SDK rejects kernel in auth_check; do not monkeypatch it
    or call an invented write-check endpoint. Kernel refs are READ evidence,
    not independent write evidence. Every upload still needs exact readback.
    """
    environment = os.environ if environment is None else environment
    require_workflow_context(environment)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(endpoint="https://huggingface.co", token=False)
    supplier = exchange if supplier is None else supplier
    validated: dict[str, str] = {}
    for repo_type, resource in RESOURCES:
        try:
            token = supplier(resource, environment)
            if (not isinstance(token, str) or len(token) > 16384
                    or TOKEN_RE.fullmatch(token) is None):
                raise KeylessCredentialError("INVALID_OIDC_TOKEN_RESPONSE")
            if repo_type == "model":
                api.auth_check(repo_id=TARGET, repo_type="model", token=token, write=True)
            else:
                refs = api.list_repo_refs(TARGET, repo_type="kernel", token=token)
                branches = {ref.name: ref.target_commit for ref in refs.branches
                            if ref.name in {"main", "v1"}}
                if (set(branches) != {"main", "v1"}
                        or any(not isinstance(sha, str)
                               or re.fullmatch(r"[0-9a-f]{40}", sha) is None
                               for sha in branches.values())):
                    raise KeylessCredentialError("INVALID_KERNEL_REFS")
        except KeylessCredentialError:
            raise
        except Exception:
            raise KeylessCredentialError("TARGET_ACCESS_VALIDATION_FAILED") from None
        validated[repo_type] = token
    return InvariantsCredentials(model=validated["model"], kernel=validated["kernel"])


def preflight(report_path: Path) -> int:
    report: dict[str, Any] = {
        "schema": "szl.invariants-keyless-preflight/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CHECKING", "repo_id": TARGET,
        "resources": [resource for _, resource in RESOURCES],
        "repository_mutation": "NOT_ATTEMPTED", "publication_verified": False,
        "credentials_exported": False,
        "grant_basis": "PROVIDER_REPO_SCOPED_OIDC_EXCHANGE",
        "model_access_check": "WRITE_AUTH_CHECK",
        "kernel_access_check": "READ_REFS_ONLY",
        "kernel_write_independently_verified": False,
    }
    atomic_report(report_path, report)
    try:
        credentials = acquire_pair()
        del credentials  # A fresh pair is acquired again at publication time.
        report["status"] = "BOTH_REPO_SCOPED_CREDENTIALS_VALIDATED"
        code = 0
    except Exception:
        report["status"] = "KEYLESS_CREDENTIALS_UNAVAILABLE"
        code = 1
    atomic_report(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    return preflight(parser.parse_args().report)


if __name__ == "__main__":
    raise SystemExit(main())
