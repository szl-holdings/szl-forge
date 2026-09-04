#!/usr/bin/env python3
"""Wave 6 — collection rebuild dry-run.

Investor read:
    Flagship is empty on purpose. Nothing is promoted. This script
    never writes the Hub. A publisher job with an owner token is a
    later, reviewed step.

Developer run:
    python3 -m publishing.collection_rebuild
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from publishing.collection_quarantine import assert_policy_sound, load_policy, may_join_collection

ROOT = Path(__file__).resolve().parents[1]
REBUILD = Path(__file__).with_name("collection-rebuild.json")
TOKEN_KEYS = (
    "HF_TOKEN",
    "HF_ORG_TOKEN1",
    "HF_ORG_TOKEN",
    "HF_WRITE_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _token() -> str:
    for key in TOKEN_KEYS:
        raw = str(os.environ.get(key) or "").strip()
        if raw:
            return raw
    return ""


def load_rebuild() -> dict:
    raw = json.loads(REBUILD.read_text(encoding="utf-8"))
    if raw.get("schema") != "szl.collection-rebuild/v1":
        raise SystemExit(f"unexpected rebuild schema {raw.get('schema')!r}")
    if raw.get("hub_write") != "DENIED_IN_THIS_CHANGE":
        raise SystemExit("rebuild plan must not imply a silent Hub write")
    if raw["collections"].get("flagship"):
        raise SystemExit("REFUSED: flagship is not empty")
    return raw


def dry_run() -> dict:
    policy = load_policy()
    assert_policy_sound(policy)
    plan = load_rebuild()
    token = _token()
    # Even with a token this function does not PUT. It only reports intent.
    research = plan["collections"]["research"]
    for hub_id in research:
        item = next((row for row in policy["items"] if row["hub_id"] == hub_id), None)
        if item is None:
            continue
        if may_join_collection(item, "flagship"):
            raise SystemExit(f"REFUSED: {hub_id} may join flagship")
    return {
        "schema": "szl.wave6-collection-dry-run/v1",
        "wave": 6,
        "source": "COMPLETE",
        "hub_write": "DENIED_NO_TOKEN" if not token else "DENIED_IN_THIS_CHANGE",
        "token_present": bool(token),
        "flagship": [],
        "research": research,
        "candidates": plan["collections"]["candidates"],
        "kernels": plan["collections"]["kernels"],
        "winner": None,
        "ready": False,
    }


def main() -> int:
    print(json.dumps(dry_run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
