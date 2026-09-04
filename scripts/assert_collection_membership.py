#!/usr/bin/env python3
"""Refuse a proposed Hub collection membership that violates quarantine.

Stdin or --proposal JSON: {"collection": "flagship", "members": ["SZLHOLDINGS/..."]}
Does not write the Hub. Exit 0 only when every member may join that slug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from publishing.collection_quarantine import item_by_id, load_policy, may_join_collection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, help="JSON file; default stdin")
    args = parser.parse_args(argv)
    raw = args.proposal.read_text(encoding="utf-8") if args.proposal else sys.stdin.read()
    proposal = json.loads(raw)
    collection = proposal.get("collection")
    members = proposal.get("members")
    if not isinstance(collection, str) or not isinstance(members, list):
        print("REFUSED: proposal must have collection string and members list", file=sys.stderr)
        return 1
    policy = load_policy()
    refused: list[str] = []
    for hub_id in members:
        item = item_by_id(policy, hub_id)
        if item is None:
            continue
        if not may_join_collection(item, collection):
            refused.append(hub_id)
    if refused:
        print("REFUSED: " + ", ".join(refused) + f" cannot join {collection}", file=sys.stderr)
        return 1
    print(f"OK collection={collection} members={len(members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
