"""The public flagship correction must fail closed on membership drift."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from publishing.flagship_collection_apply import UnknownAfterAttempt, apply_one
from publishing.flagship_collection_reconcile import load_manifest, plan_live, validate_manifest


def observed_collection_from_manifest() -> dict:
    manifest = load_manifest()
    return {
        "slug": manifest["collection_slug"],
        "private": False,
        "title": manifest["observed_collection"]["title"],
        "description": manifest["observed_collection"]["description"],
        "lastUpdated": manifest["observed_collection"]["last_updated"],
        "items": [
            {
                "id": item["hub_id"],
                "_id": item["collection_item_object_id"],
                "type": "model",
                "private": False,
            }
            for item in manifest["items"]
        ],
    }


def test_source_plan_preserves_all_public_repositories_and_quarantine() -> None:
    manifest = load_manifest()
    validate_manifest(manifest)
    assert len(manifest["items"]) == 26
    assert sum(item["quarantine_denies_flagship"] for item in manifest["items"]) == 6
    assert all(item["decision"] == "REMOVE_FROM_FLAGSHIP_KEEP_PUBLIC_REPOSITORY" for item in manifest["items"])
    assert manifest["target"]["membership"] == "EMPTY_UNTIL_SOURCE_BOUND_QUALIFICATION"


def test_live_plan_reports_current_misplacement_without_writing() -> None:
    plan = plan_live(load_manifest(), observed_collection_from_manifest())
    assert plan["remaining_flagship_items"] == 26
    assert plan["removed_since_baseline"] == 0
    assert plan["state"] == "RECONCILIATION_PENDING"
    assert plan["hub_write"] == "NONE"


def test_live_plan_accepts_a_verified_single_removal() -> None:
    live = observed_collection_from_manifest()
    live["items"].pop(0)
    plan = plan_live(load_manifest(), live)
    assert plan["remaining_flagship_items"] == 25
    assert plan["removed_since_baseline"] == 1


def test_live_plan_rejects_unknown_or_replaced_item() -> None:
    live = observed_collection_from_manifest()
    live["items"][0]["id"] = "SZLHOLDINGS/unreviewed-new-item"
    with pytest.raises(ValueError, match="unknown"):
        plan_live(load_manifest(), live)

    live = observed_collection_from_manifest()
    live["items"][0]["_id"] = "000000000000000000000000"
    with pytest.raises(ValueError, match="object ID changed"):
        plan_live(load_manifest(), live)


def test_manifest_rejects_quarantine_or_visibility_downgrade() -> None:
    manifest = load_manifest()
    bad = copy.deepcopy(manifest)
    denied = next(item for item in bad["items"] if item["quarantine_denies_flagship"])
    denied["quarantine_denies_flagship"] = False
    with pytest.raises(ValueError, match="quarantine policy mismatch"):
        validate_manifest(bad)

    bad = copy.deepcopy(manifest)
    bad["target"]["repository_visibility_change"] = True
    with pytest.raises(ValueError, match="visibility"):
        validate_manifest(bad)


def test_manifest_rejects_description_the_hub_will_refuse() -> None:
    manifest = load_manifest()
    assert len(manifest["target"]["description"]) <= 150
    bad = copy.deepcopy(manifest)
    bad["target"]["description"] = "No model is promoted. " + "x" * 150
    with pytest.raises(ValueError, match="150 characters"):
        validate_manifest(bad)


class FakeHub:
    def __init__(self, collection: dict) -> None:
        self.collection = collection
        self.writes: list[str] = []

    def whoami(self) -> dict:
        return {"name": "betterwithage", "orgs": [{"name": "SZLHOLDINGS", "roleInOrg": "admin"}]}

    def model_info(self, *, repo_id: str) -> SimpleNamespace:
        revision = next(item["model_revision"] for item in load_manifest()["items"] if item["hub_id"] == repo_id)
        return SimpleNamespace(sha=revision)

    def delete_collection_item(self, _slug: str, object_id: str) -> None:
        self.writes.append(object_id)
        self.collection["items"] = [item for item in self.collection["items"] if item["_id"] != object_id]
        self.collection["lastUpdated"] = "2026-10-03T01:00:00.000Z"

    def update_collection_metadata(self, _slug: str, *, title: str, description: str) -> None:
        self.writes.append("metadata")
        self.collection["title"] = title
        self.collection["description"] = description
        self.collection["lastUpdated"] = "2026-10-03T01:00:00.000Z"


def test_one_target_apply_reads_back_exact_removal() -> None:
    manifest = load_manifest()
    hub = FakeHub(observed_collection_from_manifest())
    target = hub.collection["items"][0]
    before = hub.collection["lastUpdated"]
    receipt = apply_one(
        manifest,
        mode="remove",
        target_id=target["id"],
        expected_last_updated=before,
        api=hub,
        read_collection=lambda: copy.deepcopy(hub.collection),
    )
    assert hub.writes == [target["_id"]]
    assert receipt["status"] == "READ_BACK"
    assert receipt["remaining_flagship_items"] == 25


def test_one_target_apply_refuses_stale_plan_before_write() -> None:
    hub = FakeHub(observed_collection_from_manifest())
    with pytest.raises(ValueError, match="changed since the operator plan"):
        apply_one(
            load_manifest(),
            mode="remove",
            target_id=hub.collection["items"][0]["id"],
            expected_last_updated="2026-01-01T00:00:00Z",
            api=hub,
            read_collection=lambda: copy.deepcopy(hub.collection),
        )
    assert hub.writes == []


def test_one_target_apply_refuses_model_revision_drift() -> None:
    hub = FakeHub(observed_collection_from_manifest())
    hub.model_info = lambda *, repo_id: SimpleNamespace(sha="0" * 40)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="model revision changed"):
        apply_one(
            load_manifest(),
            mode="remove",
            target_id=hub.collection["items"][0]["id"],
            expected_last_updated=hub.collection["lastUpdated"],
            api=hub,
            read_collection=lambda: copy.deepcopy(hub.collection),
        )
    assert hub.writes == []


def test_one_target_apply_marks_ambiguous_write_without_retry() -> None:
    hub = FakeHub(observed_collection_from_manifest())
    calls: list[str] = []

    def fail_once(_slug: str, object_id: str) -> None:
        calls.append(object_id)
        raise RuntimeError("uncertain transport")

    hub.delete_collection_item = fail_once  # type: ignore[method-assign]
    with pytest.raises(UnknownAfterAttempt, match="inspect Hub before retry"):
        apply_one(
            load_manifest(),
            mode="remove",
            target_id=hub.collection["items"][0]["id"],
            expected_last_updated=hub.collection["lastUpdated"],
            api=hub,
            read_collection=lambda: copy.deepcopy(hub.collection),
        )
    assert len(calls) == 1


def test_metadata_write_requires_empty_shelf_and_reads_back() -> None:
    manifest = load_manifest()
    hub = FakeHub(observed_collection_from_manifest())
    with pytest.raises(ValueError, match="empty flagship"):
        apply_one(
            manifest,
            mode="metadata",
            target_id=None,
            expected_last_updated=hub.collection["lastUpdated"],
            api=hub,
            read_collection=lambda: copy.deepcopy(hub.collection),
        )
    assert hub.writes == []
    hub.collection["items"] = []
    receipt = apply_one(
        manifest,
        mode="metadata",
        target_id=None,
        expected_last_updated=hub.collection["lastUpdated"],
        api=hub,
        read_collection=lambda: copy.deepcopy(hub.collection),
    )
    assert hub.writes == ["metadata"]
    assert receipt["remaining_flagship_items"] == 0
    assert receipt["metadata_aligned"] is True
