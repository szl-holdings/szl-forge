"""GitHub-side collection quarantine for CTO-HF-FRONTIER-2026-09-04.

Does not talk to the Hub. Publishers must load this policy before any
collection membership change. Fail closed: unknown promotion is HOLD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "szl.collection-quarantine/v1"
POLICY_PATH = Path(__file__).with_name("collection-quarantine.json")

REQUIRED_ITEM_KEYS = (
    "hub_id",
    "cluster",
    "code",
    "promotion",
    "allowed_collections",
    "denied_collections",
    "reason",
)

REQUIRED_QUARANTINE_IDS = (
    "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3",
    "SZLHOLDINGS/SZL-Khipu-1.5B-abstain",
    "SZLHOLDINGS/chaski",
    "SZLHOLDINGS/chaski-5050",
    "SZLHOLDINGS/A11OY-MINI",
)


def load_policy(path: Path | None = None) -> dict[str, Any]:
    raw = json.loads((path or POLICY_PATH).read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema: {raw.get('schema')!r}")
    return raw


def item_by_id(policy: dict[str, Any], hub_id: str) -> dict[str, Any] | None:
    for item in policy["items"]:
        if item["hub_id"] == hub_id:
            return item
    return None


def may_join_collection(item: dict[str, Any], collection_slug: str) -> bool:
    slug = collection_slug.strip().split("/")[-1]
    if slug in item["denied_collections"]:
        return False
    if item["promotion"] != "HOLD":
        raise ValueError(f"{item['hub_id']} promotion is {item['promotion']!r}; only HOLD is legal here")
    return slug in item["allowed_collections"]


def assert_policy_sound(policy: dict[str, Any]) -> None:
    if policy.get("hub_write") != "DENIED_IN_THIS_CHANGE":
        raise AssertionError("policy must not imply a silent Hub write")
    flagship = set(policy["flagship_collection_slugs"])
    seen: set[str] = set()
    for item in policy["items"]:
        missing = [k for k in REQUIRED_ITEM_KEYS if k not in item]
        if missing:
            raise AssertionError(f"{item.get('hub_id')} missing {missing}")
        if item["promotion"] != "HOLD":
            raise AssertionError(f"{item['hub_id']} promotion must stay HOLD until Hub write + bakeoff")
        if item["hub_id"] in seen:
            raise AssertionError(f"duplicate hub_id {item['hub_id']}")
        seen.add(item["hub_id"])
        denied = set(item["denied_collections"])
        if not flagship <= denied:
            raise AssertionError(f"{item['hub_id']} does not deny every flagship slug")
        overlap = set(item["allowed_collections"]) & denied
        if overlap:
            raise AssertionError(f"{item['hub_id']} allow/deny overlap: {sorted(overlap)}")
        for slug in flagship:
            if may_join_collection(item, slug):
                raise AssertionError(f"{item['hub_id']} may not join flagship slug {slug}")
    missing_ids = [hid for hid in REQUIRED_QUARANTINE_IDS if hid not in seen]
    if missing_ids:
        raise AssertionError(f"required quarantine IDs missing: {missing_ids}")


def main() -> int:
    policy = load_policy()
    assert_policy_sound(policy)
    print(f"collection-quarantine ok items={len(policy['items'])} schema={SCHEMA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
