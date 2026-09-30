#!/usr/bin/env python3
"""Acquire two closed, short-lived Hub grants for the SZL kernel release.

The model mirror and the first-class Kernel repository are different Hub
resources even though they share the same visible repository ID.  A single
credential is therefore never accepted for both targets.  The official
``hf auth token`` exchange performs GitHub OIDC verification at the provider;
this module only admits the one protected workflow context and never prints,
caches, exports, or serializes either returned token.
"""
from __future__ import annotations

import hmac
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


REPOSITORY = "szl-holdings/szl-forge"
WORKFLOW_PATH = ".github/workflows/publish-szl-kernels.yml"
WORKFLOW_FILENAME = "publish-szl-kernels.yml"
WORKFLOW_REF = f"{REPOSITORY}/{WORKFLOW_PATH}@refs/heads/main"
TARGET = "SZLHOLDINGS/szl-kernels"
MODEL_RESOURCE = TARGET
KERNEL_RESOURCE = f"kernels/{TARGET}"
RESOURCES = (("model", MODEL_RESOURCE), ("kernel", KERNEL_RESOURCE))
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
OIDC_AUDIENCE = "https://huggingface.co"
OIDC_EXCHANGE_ENDPOINT = "https://huggingface.co/oauth/token"
TOKEN_RE = re.compile(r"hf_jwt_[A-Za-z0-9._-]+")
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
SAFE_CHILD_ENV = frozenset({
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "RUNNER_TEMP",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "GITHUB_ACTIONS",
    "ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
})
FORBIDDEN_AMBIENT_CREDENTIALS = frozenset({
    "HF_TOKEN", "HF_ORG_TOKEN", "HF_ORG_TOKEN1", "HF_WRITE_TOKEN",
    "HF_TOKEN_PATH", "HF_OIDC_ID_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
})


class KeylessCredentialError(RuntimeError):
    """A fixed non-secret failure code; provider response text is discarded."""


def _access_validation_failure(code: str, cause: Exception) -> KeylessCredentialError:
    """Retain only a bounded HTTP status from the pinned client's response."""
    from httpx import Response
    from huggingface_hub.errors import HfHubHTTPError

    failure = KeylessCredentialError(code)
    failure.http_status = None
    if isinstance(cause, HfHubHTTPError):
        # Do not consult arbitrary exception properties, headers, URLs, or
        # bodies. Only the pinned Hub client's concrete response is admitted.
        response = vars(cause).get("response")
        if type(response) is Response:
            status = vars(response).get("status_code")
            if type(status) is int and 400 <= status <= 599:
                failure.http_status = status
    return failure


@dataclass(frozen=True)
class KernelPublisherCredentials:
    model: str = field(repr=False)
    kernel: str = field(repr=False)


def _require_actions_oidc_endpoint(environment: Mapping[str, str]) -> None:
    request_url = environment.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
    parsed = urlparse(request_url)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".actions.githubusercontent.com")
        or not parsed.path
    ):
        raise KeylessCredentialError("INVALID_GITHUB_OIDC_ENDPOINT")
    request_token = environment.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
    if not request_token or len(request_token) > 16384:
        raise KeylessCredentialError("MISSING_GITHUB_OIDC_REQUEST_TOKEN")


def require_workflow_context(environment: Mapping[str, str]) -> None:
    """Deny forks, PRs, mutable refs, and any other workflow identity."""
    expected = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_WORKFLOW_REF": WORKFLOW_REF,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
    }
    if any(environment.get(key) != value for key, value in expected.items()):
        raise KeylessCredentialError("UNTRUSTED_WORKFLOW_CONTEXT")
    if environment.get("GITHUB_HEAD_REF") not in (None, ""):
        raise KeylessCredentialError("PULL_REQUEST_CONTEXT_REJECTED")
    if FULL_SHA_RE.fullmatch(environment.get("GITHUB_SHA", "")) is None:
        raise KeylessCredentialError("INVALID_PUBLISHER_REVISION")
    _require_actions_oidc_endpoint(environment)


def authority_evidence(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the closed public grant description, never credential material."""
    environment = os.environ if environment is None else environment
    require_workflow_context(environment)
    return authority_policy(environment["GITHUB_SHA"])


def authority_policy(publisher_revision: str) -> dict[str, Any]:
    """Describe the grants required by a release without acquiring authority."""
    if FULL_SHA_RE.fullmatch(publisher_revision) is None:
        raise KeylessCredentialError("INVALID_PUBLISHER_REVISION")
    return {
        "schema": "szl.hf-trusted-publisher-authority/v1",
        "mode": "GITHUB_OIDC_REPO_SCOPED",
        "issuer": OIDC_ISSUER,
        "audience": OIDC_AUDIENCE,
        "exchange_endpoint": OIDC_EXCHANGE_ENDPOINT,
        "github_repository": REPOSITORY,
        "github_ref": "refs/heads/main",
        "github_workflow_path": WORKFLOW_PATH,
        "github_workflow_ref": WORKFLOW_REF,
        "publisher_revision": publisher_revision,
        "trusted_publisher_claims": {
            "repository": REPOSITORY,
            "branch": "main",
            "workflow": WORKFLOW_FILENAME,
        },
        "resources": {
            "legacy_model": MODEL_RESOURCE,
            "first_class_kernel": KERNEL_RESOURCE,
        },
        "distinct_target_credentials": True,
        "persistent_hub_secret_used": False,
    }


def exchange(resource: str, environment: Mapping[str, str]) -> str:
    """Exchange GitHub OIDC through the pinned official CLI in an empty home."""
    if resource not in {item[1] for item in RESOURCES}:
        raise KeylessCredentialError("UNDECLARED_OIDC_RESOURCE")
    require_workflow_context(environment)
    child = {
        key: value for key, value in environment.items() if key in SAFE_CHILD_ENV
    }
    with tempfile.TemporaryDirectory(prefix="szl-kernels-oidc-") as home:
        child.update({
            "HOME": home,
            "USERPROFILE": home,
            "XDG_CACHE_HOME": os.path.join(home, ".cache"),
            "HF_HOME": os.path.join(home, "huggingface"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_OIDC_RESOURCE": resource,
        })
        try:
            process = subprocess.run(
                ["hf", "auth", "token"],
                env=child,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except Exception:
            raise KeylessCredentialError("OIDC_EXCHANGE_UNAVAILABLE") from None
    if process.returncode != 0:
        raise KeylessCredentialError("OIDC_EXCHANGE_REJECTED")
    token = process.stdout.strip()
    if len(token) > 16384 or TOKEN_RE.fullmatch(token) is None:
        raise KeylessCredentialError("INVALID_OIDC_TOKEN_RESPONSE")
    return token


def acquire_pair(
    *,
    environment: Mapping[str, str] | None = None,
    supplier: Callable[[str, Mapping[str, str]], str] | None = None,
    api: Any = None,
) -> KernelPublisherCredentials:
    """Acquire and validate both target grants without attempting a write."""
    environment = os.environ if environment is None else environment
    require_workflow_context(environment)
    if any(environment.get(key) for key in FORBIDDEN_AMBIENT_CREDENTIALS):
        raise KeylessCredentialError("AMBIENT_HUB_CREDENTIAL_REJECTED")
    supplier = exchange if supplier is None else supplier
    try:
        model = supplier(MODEL_RESOURCE, environment)
        kernel = supplier(KERNEL_RESOURCE, environment)
    except KeylessCredentialError:
        raise
    except Exception:
        raise KeylessCredentialError("OIDC_EXCHANGE_UNAVAILABLE") from None
    for token in (model, kernel):
        if (
            not isinstance(token, str)
            or len(token) > 16384
            or TOKEN_RE.fullmatch(token) is None
        ):
            raise KeylessCredentialError("INVALID_OIDC_TOKEN_RESPONSE")
    if hmac.compare_digest(model, kernel):
        raise KeylessCredentialError("CROSS_TARGET_TOKEN_REUSE_REJECTED")

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(endpoint="https://huggingface.co", token=False)
    try:
        api.auth_check(repo_id=TARGET, repo_type="model", token=model, write=True)
    except Exception as exc:
        raise _access_validation_failure("MODEL_WRITE_ACCESS_VALIDATION_FAILED", exc) from None
    try:
        refs = api.list_repo_refs(TARGET, repo_type="kernel", token=kernel)
        branches = {
            ref.name: ref.target_commit
            for ref in refs.branches
            if ref.name in {"main", "v1"}
        }
        if (
            set(branches) != {"main", "v1"}
            or any(
                not isinstance(sha, str) or FULL_SHA_RE.fullmatch(sha) is None
                for sha in branches.values()
            )
        ):
            raise KeylessCredentialError("INVALID_KERNEL_REFS")
    except KeylessCredentialError:
        raise
    except Exception as exc:
        raise _access_validation_failure("KERNEL_REFS_VALIDATION_FAILED", exc) from None
    return KernelPublisherCredentials(model=model, kernel=kernel)

