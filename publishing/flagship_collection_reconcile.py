"""Validate the source-owned plan for the public SZL flagship collection.

This command is read-only. It never changes collection metadata, membership,
repository visibility, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from publishing.collection_quarantine import assert_policy_sound, load_policy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("flagship-collection-reconciliation.v1.json")
REBUILD = Path(__file__).with_name("collection-rebuild.json")
SCHEMA = "szl.flagship-collection-reconciliation/v1"
COLLECTION = "SZLHOLDINGS/szl-flagship-models-6a9315c1c853da528726dd8d"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX24 = re.compile(r"[0-9a-f]{24}\Z")


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on an incomplete or promotion-implying source plan."""
    if manifest.get("schema") != SCHEMA or manifest.get("collection_slug") != COLLECTION:
        raise ValueError("unexpected reconciliation schema or collection")
    if manifest.get("source_repository") != "szl-holdings/szl-forge":
        raise ValueError("unexpected source repository")
    if manifest.get("hub_write") != "DENIED_IN_THIS_SOURCE_CHANGE":
        raise ValueError("source change must not imply a Hub write")
    target = manifest.get("target", {})
    if target.get("membership") != "EMPTY_UNTIL_SOURCE_BOUND_QUALIFICATION":
        raise ValueError("flagship target must remain empty")
    if target.get("repository_visibility_change") is not False or target.get("model_repository_delete") is not False:
        raise ValueError("reconciliation cannot change repository visibility or delete a model")
    if not target.get("public_catalog_url", "").startswith("https://holdings.a-11-oy.com/frontier/"):
        raise ValueError("public catalog location missing")

    policy = load_policy()
    assert_policy_sound(policy)
    denied = {item["hub_id"] for item in policy["items"]}
    rebuild = json.loads(REBUILD.read_text(encoding="utf-8"))
    if rebuild.get("collections", {}).get("flagship") != []:
        raise ValueError("canonical rebuild policy no longer has an empty flagship")
    planned_shelves = {
        hub_id: shelf
        for shelf, ids in rebuild["collections"].items()
        for hub_id in ids
    }

    items = manifest.get("items", [])
    observed = manifest.get("observed_collection", {})
    if len(items) != 26 or observed.get("item_count") != 26 or observed.get("private") is not False:
        raise ValueError("the 26-item public observation is incomplete")
    if observed.get("title") != "SZL Flagship Models" or not observed.get("last_updated"):
        raise ValueError("collection observation is incomplete")
    if not observed.get("description", "").startswith("Trained weights only."):
        raise ValueError("the inaccurate observed description changed; reobserve before editing")
    target_description = target.get("description", "")
    if "No model is promoted" not in target_description:
        raise ValueError("target description must state the promotion boundary")
    if len(target_description) > 150:
        raise ValueError("target description exceeds the Hub collection limit of 150 characters")

    seen_ids: set[str] = set()
    seen_objects: set[str] = set()
    for item in items:
        hub_id = item.get("hub_id", "")
        object_id = item.get("collection_item_object_id", "")
        if not hub_id.startswith("SZLHOLDINGS/") or hub_id in seen_ids:
            raise ValueError(f"invalid or duplicate Hub ID: {hub_id}")
        if not HEX24.fullmatch(object_id) or object_id in seen_objects:
            raise ValueError(f"invalid or duplicate collection object ID: {hub_id}")
        if not HEX40.fullmatch(item.get("model_revision", "")):
            raise ValueError(f"unbound model revision: {hub_id}")
        if not item.get("artifact_class") or not item.get("promotion_declaration"):
            raise ValueError(f"missing artifact or evidence class: {hub_id}")
        if item.get("decision") != "REMOVE_FROM_FLAGSHIP_KEEP_PUBLIC_REPOSITORY":
            raise ValueError(f"unsafe membership or visibility decision: {hub_id}")
        if item.get("quarantine_denies_flagship") is not (hub_id in denied):
            raise ValueError(f"quarantine policy mismatch: {hub_id}")
        if item.get("source_planned_shelf") != planned_shelves.get(hub_id, "public_catalog_only"):
            raise ValueError(f"source shelf mismatch: {hub_id}")
        seen_ids.add(hub_id)
        seen_objects.add(object_id)
    if seen_ids & denied != denied:
        raise ValueError("one or more denied assets are absent from the observed collection")


def fetch_public_collection() -> dict[str, Any]:
    url = "https://huggingface.co/api/collections/" + COLLECTION
    request = urllib.request.Request(url, headers={"User-Agent": "szl-flagship-readonly/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def plan_live(manifest: dict[str, Any], collection: dict[str, Any]) -> dict[str, Any]:
    """Compare live membership with the approved baseline without changing it."""
    validate_manifest(manifest)
    if collection.get("slug") != COLLECTION or collection.get("private") is not False:
        raise ValueError("collection identity or visibility drift")
    by_id = {item["hub_id"]: item for item in manifest["items"]}
    remaining: list[str] = []
    seen: set[str] = set()
    for item in collection.get("items", []):
        hub_id = item.get("id", "")
        baseline = by_id.get(hub_id)
        if baseline is None or hub_id in seen or item.get("type") != "model" or item.get("private") is not False:
            raise ValueError(f"unknown, duplicate, non-model, or private collection item: {hub_id}")
        if item.get("_id") != baseline["collection_item_object_id"]:
            raise ValueError(f"collection object ID changed: {hub_id}")
        remaining.append(hub_id)
        seen.add(hub_id)
    metadata_aligned = (
        collection.get("title") == manifest["target"]["title"]
        and collection.get("description") == manifest["target"]["description"]
    )
    return {
        "schema": "szl.flagship-collection-plan/v1",
        "source_plan": "VALID",
        "hub_write": "NONE",
        "collection_slug": COLLECTION,
        "provider_last_updated": collection.get("lastUpdated"),
        "observed_baseline_items": len(by_id),
        "remaining_flagship_items": len(remaining),
        "removed_since_baseline": len(by_id) - len(remaining),
        "metadata_aligned": metadata_aligned,
        "next_item_for_separate_review": remaining[0] if remaining else None,
        "state": "ALIGNED" if not remaining and metadata_aligned else "RECONCILIATION_PENDING",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="read public collection metadata and plan; never write")
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        validate_manifest(manifest)
        if args.live:
            print(json.dumps(plan_live(manifest, fetch_public_collection()), indent=2))
        else:
            print("flagship source plan valid: 26 observed; 6 quarantined; target empty; Hub write denied")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"flagship reconciliation HOLD: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
