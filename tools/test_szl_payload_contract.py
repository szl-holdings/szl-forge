# SPDX-License-Identifier: Apache-2.0
"""Synthetic offline controls; no runtime, remote source or model qualification."""
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import szl_payload_contract as M

NOW = datetime(2026, 9, 19, 13, tzinfo=timezone.utc)
START = "2026-09-17T10:00:00Z"
END = "2026-09-17T10:01:00Z"


def git_item():
    # Deliberately exercise Git's directory-vs-file sort: a.c sorts before a/.
    entries = [
        {"path": "a.c", "mode": "100644", "type": "blob", "sha": M.git_object("blob", b"x"), "size": 1},
        {"path": "a/z", "mode": "100755", "type": "blob", "sha": M.git_object("blob", b"z"), "size": 1},
        {"path": "link", "mode": "120000", "type": "blob", "sha": M.git_object("blob", b"a.c"), "size": 3},
        {"path": "vendor", "mode": "160000", "type": "commit", "sha": "d" * 40, "size": None},
    ]
    subtree = M.git_object("tree", M.git_tree_body([entries[1]]))
    entries.append({"path": "a", "mode": "040000", "type": "tree", "sha": subtree, "size": None})
    root = M.git_object("tree", M.git_tree_body([e for e in entries if "/" not in e["path"]]))
    return {"repository": "szl-holdings/fixture", "revision": "a" * 40,
            "revision_after": "a" * 40, "tree_sha": root, "default_branch": "main",
            "entries": entries, "file_count": 3, "symlink_count": 1, "submodule_count": 1,
            "tree_complete": True, "tree_merkle_verified": True, "complete": True,
            "content_bytes_verified": False, "runtime_verified": False, "blockers": [],
            "started_at": START, "finished_at": END}


def hf_item(family="models", redacted=False):
    row = {"path": "weights.safetensors", "type": "file", "oid": "b" * 40, "size": 42,
           "lfs_identity_state": "REDACTED" if redacted else "OBSERVED",
           "lfs_oid": None if redacted else "c" * 64}
    return {"repo_id": "SZLHOLDINGS/fixture", "repo_type_endpoint": family,
            "revision": "a" * 40, "revision_after": "a" * 40, "entries": [row],
            "file_count": 1, "tree_complete": True, "file_metadata_identity_complete": not redacted,
            "redacted_lfs_file_count": int(redacted), "complete": not redacted,
            "content_bytes_verified": False, "runtime_verified": False,
            "runtime_stage_reported": "RUNNING" if family == "spaces" else None,
            "blockers": ["HF_LFS_IDENTITY_REDACTED"] if redacted else [],
            "started_at": START, "finished_at": END}


def population(items):
    count = sum(item["complete"] for item in items)
    known = sum(item["file_count"] for item in items if item["file_count"] is not None)
    complete = count == len(items)
    return {"items": items, "items_observed": len(items), "items_complete": count,
            "known_file_subtotal": known, "complete_scope_file_count": known if complete else None,
            "complete": complete, "membership_stable": True, "blockers": []}


def receipt(lane="github", partial=False):
    populations = {"github": population([git_item()])} if lane == "github" else {
        f: population([hf_item(f, partial and f == "models")]) for f in ("models", "datasets", "spaces", "kernels")}
    complete = all(p["complete"] for p in populations.values())
    return {"schema": M.SCHEMA, "scope": M.SCOPE, "lane": lane, "populations": populations,
            "source_revision": "f" * 40, "observer_sha256": "e" * 64,
            "started_at": START, "finished_at": END, "source_content_files_read": 0,
            "complete": complete, "blockers": [], "production_authorization": False,
            "semantic_review_complete": False, "runtime_verified": False,
            "status": "FILE_METADATA_OBSERVED_NOT_QUALIFIED" if complete else "PARTIAL_OR_UNAVAILABLE"}


def archive(path, value, member="receipt.json", second=False, special=False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        if special:
            info = zipfile.ZipInfo(member)
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            out.writestr(info, M.canonical(value))
        else:
            out.writestr(member, M.canonical(value))
        if second:
            out.writestr("extra.json", b"{}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
