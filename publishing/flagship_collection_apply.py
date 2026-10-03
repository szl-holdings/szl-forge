"""One-target, source-bound Hugging Face flagship reconciliation operator.

Use only after the source PR is merged, exact-head CI is green, and competing
collection writers have been checked. Each invocation makes at most one Hub
write and immediately reads the public collection back. Never retry an
ambiguous attempt without an independent provider readback.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from typing import Any, Callable

from publishing.flagship_collection_reconcile import (
    COLLECTION,
    ROOT,
    fetch_public_collection,
    load_manifest,
    plan_live,
    validate_manifest,
)


class UnknownAfterAttempt(RuntimeError):
    """A provider write may have escaped, but its exact effect is unverified."""


def assert_protected_source(expected_sha: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_sha) is None:
        raise ValueError("an exact 40-hex protected source SHA is required")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=30
        ).strip()

    if git("branch", "--show-current") != "main":
        raise ValueError("operator must run from protected main")
    if git("rev-parse", "HEAD") != expected_sha or git("status", "--porcelain=v1"):
        raise ValueError("local source is dirty or does not match the expected SHA")
    remote = git("ls-remote", "origin", "refs/heads/main").split()
    if len(remote) != 2 or remote[0] != expected_sha:
        raise ValueError("remote protected main moved or is unavailable")
    subprocess.check_output(
        ["git", "verify-commit", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, timeout=30
    )


def assert_org_admin(api: Any) -> None:
    identity = api.whoami()
    if identity.get("name") != "betterwithage":
        raise ValueError("unexpected Hugging Face operator identity")
    memberships = identity.get("orgs", [])
    if not any(org.get("name") == "SZLHOLDINGS" and org.get("roleInOrg") == "admin" for org in memberships):
        raise ValueError("SZLHOLDINGS admin authority was not observed")


def apply_one(
    manifest: dict[str, Any],
    *,
    mode: str,
    target_id: str | None,
    expected_last_updated: str,
    api: Any,
    read_collection: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Attempt one membership removal or one metadata correction, then read back."""
    validate_manifest(manifest)
    if mode not in {"remove", "metadata"} or not expected_last_updated:
        raise ValueError("one exact transition and provider timestamp are required")
    baseline = read_collection()
    plan = plan_live(manifest, baseline)
    if baseline.get("lastUpdated") != expected_last_updated:
        raise ValueError("collection changed since the operator plan")
    if mode == "remove":
        by_id = {item["hub_id"]: item for item in manifest["items"]}
        if target_id not in by_id:
            raise ValueError("target is not in the source-owned 26-item baseline")
        if api.model_info(repo_id=target_id).sha != by_id[target_id]["model_revision"]:
            raise ValueError("target model revision changed since the source review")
        selected = [item for item in baseline["items"] if item["id"] == target_id]
        if len(selected) != 1 or selected[0]["_id"] != by_id[target_id]["collection_item_object_id"]:
            raise ValueError("target membership or object identity changed")
    elif target_id is not None or plan["remaining_flagship_items"] != 0:
        raise ValueError("metadata correction requires the empty flagship shelf")

    assert_org_admin(api)
    before_write = read_collection()
    if before_write != baseline:
        raise ValueError("collection changed before the write")

    try:
        if mode == "remove":
            api.delete_collection_item(COLLECTION, selected[0]["_id"])
        else:
            api.update_collection_metadata(
                COLLECTION,
                title=manifest["target"]["title"],
                description=manifest["target"]["description"],
            )
    except Exception as exc:
        raise UnknownAfterAttempt(f"provider write returned {type(exc).__name__}; inspect Hub before retry") from None

    try:
        after = read_collection()
        after_plan = plan_live(manifest, after)
        before_ids = {item["id"] for item in baseline["items"]}
        after_ids = {item["id"] for item in after["items"]}
        if mode == "remove":
            if before_ids - after_ids != {target_id} or after_ids - before_ids:
                raise ValueError("membership readback did not show exactly the selected removal")
            if after.get("title") != baseline.get("title") or after.get("description") != baseline.get("description"):
                raise ValueError("metadata changed during membership removal")
        else:
            if before_ids != after_ids or not after_plan["metadata_aligned"]:
                raise ValueError("metadata readback did not match the approved source")
        return {
            "schema": "szl.flagship-collection-apply-receipt/v1",
            "mode": mode,
            "target_id": target_id,
            "status": "READ_BACK",
            "before_last_updated": expected_last_updated,
            "after_last_updated": after.get("lastUpdated"),
            "remaining_flagship_items": after_plan["remaining_flagship_items"],
            "metadata_aligned": after_plan["metadata_aligned"],
        }
    except Exception as exc:
        raise UnknownAfterAttempt(f"provider readback failed {type(exc).__name__}; inspect Hub before retry") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("remove", "metadata"), required=True)
    parser.add_argument("--target-id", help="one exact Hub model ID for remove mode")
    parser.add_argument("--expected-last-updated", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--execute", action="store_true", help="required to authorize one Hub write")
    args = parser.parse_args()
    if not args.execute:
        parser.exit(2, "HOLD: --execute is required; use the read-only plan first\n")
    try:
        assert_protected_source(args.expected_source_sha)
        from huggingface_hub import HfApi

        receipt = apply_one(
            load_manifest(),
            mode=args.mode,
            target_id=args.target_id,
            expected_last_updated=args.expected_last_updated,
            api=HfApi(),
            read_collection=fetch_public_collection,
        )
        print(json.dumps(receipt, indent=2))
    except UnknownAfterAttempt as exc:
        parser.exit(3, f"UNKNOWN_AFTER_ATTEMPT: {exc}\n")
    except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"HOLD: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
