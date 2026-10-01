#!/usr/bin/env python3
"""Offline Kernel Hub consumer checks; never provider admission or live collection.

This module reads no credentials, loads no kernels and makes no network requests.
A valid consumer description is only PINNED: provider resolution, trust, artifact
integrity and numerical/runtime witnesses remain separate gates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
TAKEDOWN_DATE = "2026-09-13"  # Upstream gives a date, not an exact UTC instant.
HISTORICAL_SOURCE = (
    "https://github.com/szl-holdings/szl-forge/blob/"
    "4ed693c2b7ea43ec1c23cdbb31bba08601802b30/docs/KERNEL_HUB_STATUS.md"
)


def _repo_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and REPO_ID.fullmatch(value) is not None
        and ".." not in value
        and "--" not in value
        and not value.endswith((".git", "."))
    )


def classify_consumer_call(call: dict[str, Any]) -> dict[str, Any]:
    """Check a declaration's syntax, not a Hub artifact or execution permission.

    Missing/invalid repo types are never silently treated as observed kernels.
    Moving major-version refs remain valid upstream API inputs, but do not meet
    SZL's immutable-revision policy until separately resolved to a commit.
    """
    if not isinstance(call, dict):
        raise TypeError("consumer declaration must be a dictionary")
    blockers: list[str] = []
    repo_id = call.get("repo_id")
    repo_type = call.get("repo_type")
    revision = call.get("revision")
    if not _repo_id(repo_id):
        blockers.append("VALID_REPO_ID_REQUIRED")
    if repo_type == "model":
        blockers.append("LEGACY_MODEL_TYPE_FORBIDDEN")
    elif repo_type != "kernel":
        blockers.append("EXPLICIT_KERNEL_TYPE_REQUIRED")
    if revision is None or revision == "":
        blockers.append("REVISION_REQUIRED")
    elif revision == "main":
        blockers.append("FLOATING_MAIN_NOT_ADMITTED_AT_RUNTIME")
    elif not isinstance(revision, str) or SHA.fullmatch(revision) is None:
        blockers.append("IMMUTABLE_REVISION_REQUIRED")

    trust = call.get("trust_remote_code", False)
    if trust is True:
        blockers.append("TRUST_REMOTE_CODE_REQUIRES_REVIEW")
    elif trust is False:
        pass  # Upstream's default policy; publisher trust is NOT verified here.
    elif isinstance(trust, list):
        if not _repo_id(repo_id) or trust != [repo_id]:
            blockers.append("EXACT_REPOSITORY_ALLOWLIST_REQUIRED")
    else:
        blockers.append("INVALID_TRUST_REMOTE_CODE")

    pinned = not blockers
    return {
        "ok": pinned,
        "state": "PINNED" if pinned else "HOLD",
        "validation_scope": "STATIC_DECLARATION_ONLY",
        "executed": False,
        "provider_resolved": False,
        "publisher_trust_verified": False,
        "blockers": blockers,
        "repo_id": repo_id if _repo_id(repo_id) else None,
        "revision": revision if isinstance(revision, str) and SHA.fullmatch(revision) else None,
        "repo_type": "kernel" if repo_type == "kernel" else None,
        "production_authorization": False,
    }


def historical_snapshot(*, generated_at: str | None = None) -> dict[str, Any]:
    """Describe old, source-recorded claims without restamping them as live data."""
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": "szl.kernel-hub-ops-counts/v2",
        "generated_at": stamp,
        "observed_at": None,
        "collection_performed": False,
        "evidence_kind": "HISTORICAL_SOURCE_CLAIMS_ONLY",
        "takedown_starts_date": TAKEDOWN_DATE,
        "state": "HOLD",
        "blockers": ["LIVE_KERNEL_OBSERVATION_UNAVAILABLE"],
        "counts": None,
        "historical_claims": {
            "source": HISTORICAL_SOURCE,
            "independently_verified": False,
            "list_and_tags": {
                "claimed_observed_at": "2026-09-11T23:07Z",
                "scope": "unauthenticated kernels API list, author=SZLHOLDINGS",
                "counts": {"kernels_api_list": 14, "tagged_kernel": 9, "untagged_kernel": 5},
                "untagged_repo_ids": [
                    "SZLHOLDINGS/szl-governed-norm",
                    "SZLHOLDINGS/governed-inference-meter",
                    "SZLHOLDINGS/szl-maskmod",
                    "SZLHOLDINGS/szl-block-kv",
                    "SZLHOLDINGS/szl-receipt-attn",
                ],
            },
            "version_branches": {
                "claimed_observed_date": "2026-09-11",
                "claimed_observed_at": None,
                "scope": "separate earlier observation; not the 23:07Z list sweep",
                "missing_v1_repo_ids": [
                    "SZLHOLDINGS/szl-maskmod",
                    "SZLHOLDINGS/szl-block-kv",
                    "SZLHOLDINGS/szl-receipt-attn",
                    "SZLHOLDINGS/YARQA-ATTN",
                ],
            },
        },
        "mirrors_deleted": False,
        "runtime_loaded": False,
        "production_authorization": False,
        "note": "Historical counts are not current inventory. No Hub observation was performed.",
    }


def live_snapshot(observed_at: str | None = None) -> dict[str, Any]:
    """Compatibility name only: refuse synthetic observation times; never collect.

    Callers must inspect evidence_kind. New code should use historical_snapshot.
    """
    if observed_at is not None:
        raise ValueError("offline source claims cannot accept a live observation timestamp")
    return historical_snapshot()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new report file; default: stdout")
    args = parser.parse_args(argv)
    payload = historical_snapshot()
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        try:
            # Exclusive creation preserves existing evidence and refuses a final symlink.
            # This is an operator-selected output path, not an untrusted task input.
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except OSError as exc:
            parser.error(f"cannot create report ({type(exc).__name__}); use a new file in an existing directory")
    # Writing a report is not success at live collection or fleet qualification.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
