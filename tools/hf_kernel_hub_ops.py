#!/usr/bin/env python3
"""Kernel Hub fleet consumer contract. No Hub writes. No weight loads."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
TAKEDOWN = "2026-09-13T00:00:00Z"


def classify_consumer_call(call: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    repo_type = call.get("repo_type") or "kernel"
    revision = call.get("revision")
    if repo_type != "kernel":
        blockers.append("LEGACY_MODEL_TYPE_FORBIDDEN")
    if not revision:
        blockers.append("REVISION_REQUIRED")
    elif revision == "main":
        blockers.append("FLOATING_MAIN_NOT_ADMITTED_AT_RUNTIME")
    if call.get("trust_remote_code") is True:
        blockers.append("TRUST_REMOTE_CODE_REQUIRES_REVIEW")
    admitted = not blockers
    return {
        "ok": admitted,
        "state": "ADMITTED" if admitted else "HOLD",
        "executed": False,
        "blockers": blockers,
        "repo_id": call.get("repo_id"),
        "revision": revision,
        "repo_type": repo_type,
        "production_authorization": False,
    }


def live_snapshot(observed_at: str | None = None) -> dict[str, Any]:
    return {
        "schema": "szl.kernel-hub-ops-counts/v1",
        "observed_at": observed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "takedown_starts": TAKEDOWN,
        "state": "HOLD",
        "counts": {
            "kernels_api_list": 14,
            "tagged_kernel": 9,
            "kernels_api_untagged": 5,
            "model_type_mirrors": "UNAVAILABLE",
            "missing_v1_claimed_prior": 4,
        },
        "kernels_api_untagged": [
            "SZLHOLDINGS/szl-governed-norm",
            "SZLHOLDINGS/governed-inference-meter",
            "SZLHOLDINGS/szl-maskmod",
            "SZLHOLDINGS/szl-block-kv",
            "SZLHOLDINGS/szl-receipt-attn",
        ],
        "missing_v1_prior_observation": [
            "SZLHOLDINGS/szl-maskmod",
            "SZLHOLDINGS/szl-block-kv",
            "SZLHOLDINGS/szl-receipt-attn",
            "SZLHOLDINGS/YARQA-ATTN",
        ],
        "predicates_incomparable": True,
        "mirrors_deleted": False,
        "runtime_loaded": False,
        "production_authorization": False,
        "note": "API list 14 and tag=kernel 9 are different predicates. Do not mix with membership 46/35/21.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/KERNEL_HUB_FLEET_LIVE.json")
    args = parser.parse_args()
    payload = live_snapshot()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
