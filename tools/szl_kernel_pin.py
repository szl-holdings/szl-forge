#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel Hub pin contract. Offline. Not a loader, publisher, or authorizer.

kernels>=0.15 requires version XOR a 40-character revision.
SZLHOLDINGS is not a trusted publisher.
trust_remote_code is EVALUATION, not admission.
This module never calls get_kernel or the network.
production_authorization is always false.
"""
from __future__ import annotations

import re
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
TRUSTED_PUBLISHERS = frozenset()
SCHEMA = "szl.kernel-pin/v1"


class PinError(ValueError):
    """Fixed diagnostic. Never a private path."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise PinError(code)


def pin(spec: Any) -> dict[str, Any]:
    need(type(spec) is dict, "PIN_OBJECT")
    if spec.get("production_authorization") is True:
        raise PinError("FORBIDDEN_PROMOTION_AUTHORITY")
    if spec.get("runtime_loaded") is True:
        raise PinError("FORBIDDEN_PROMOTION_RUNTIME")
    repo = spec.get("repo_id")
    need(type(repo) is str and 0 < len(repo) <= 256 and "/" in repo, "PIN_REPO")
    org = repo.split("/", 1)[0]
    version = spec.get("version")
    revision = spec.get("revision")
    has_version = version is not None
    has_revision = revision is not None
    need(has_version or has_revision, "PIN_MISSING")
    need(not (has_version and has_revision), "PIN_BOTH")
    if has_version:
        need(type(version) is int and version >= 1, "PIN_VERSION")
    else:
        need(type(revision) is str and SHA40.fullmatch(revision) is not None, "PIN_REVISION")
    trusted = org in TRUSTED_PUBLISHERS
    remote = spec.get("trust_remote_code")
    if trusted:
        floor = "PINNED"
    else:
        need(remote is True, "PIN_UNTRUSTED_REQUIRES_FLAG")
        floor = "EVALUATION"
    return {
        "schema": SCHEMA,
        "repo_id": repo,
        "org": org,
        "version": version if has_version else None,
        "revision": revision if has_revision else None,
        "trusted_publisher": trusted,
        "trust_remote_code": remote is True,
        "floor": floor,
        "runtime_loaded": False,
        "signature_verified": False,
        "production_authorization": False,
        "meaning": "PIN_IS_NOT_LOAD_NOT_SIGNATURE_NOT_AUTHORIZATION",
    }
