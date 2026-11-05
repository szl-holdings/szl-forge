"""The attested Space trees match later main, including this checkout.

Hosted pytest checks out the pull-request merge ref at fetch-depth 1. The
named commits are not in that clone, and Git then reports a path "exists on
disk, but not in" the revision. Each test fetches the named commit at depth 1
and compares trees. A depth-1 fetch does not include the commits between the
two revisions, so the recorded commit count is not re-derived here.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "space-release-readback-2026-11-05.json"
_FETCHED: set[str] = set()


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30)


def fetch(revision: str) -> None:
    if revision in _FETCHED:
        return
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", revision],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL, timeout=60,
    )
    _FETCHED.add(revision)


def document() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_later_main_does_not_change_the_attested_space_trees():
    record = document()
    assert record["hf_writes"] is False
    attested = record["attested_source_revision"]
    main = record["observed_main"]
    fetch(attested)
    fetch(main)
    for space in record["spaces"]:
        assert space["runtime_readback"]["revision"] == attested
        for revision in (main, "HEAD"):
            names = _git("diff", "--name-only", attested, revision, "--", space["source_dir"])
            assert names == b""
        assert "Do not dispatch" in " ".join(space["acceptance"])


def test_inference_release_manifest_matches_the_attested_tree():
    record = document()
    attested = record["attested_source_revision"]
    fetch(attested)
    manifest = json.loads(
        (ROOT / "spaces/szl-model-inference-lab/release.json").read_text(encoding="utf-8")
    )
    for path, digest in manifest["source_files"].items():
        relative = f"spaces/szl-model-inference-lab/{path}"
        attested_blob = _git("show", f"{attested}:{relative}")
        head_blob = _git("show", f"HEAD:{relative}")
        assert attested_blob == head_blob
        assert hashlib.sha256(attested_blob).hexdigest() == digest


def test_foundation_archive_pointer_is_the_readback_digest():
    record = document()
    space = next(
        item for item in record["spaces"] if item["repo_id"].endswith("szl-foundation-confirmation")
    )
    attested = record["attested_source_revision"]
    fetch(attested)
    relative = "spaces/szl-foundation-confirmation/release.zip"
    pointer = _git("show", f"{attested}:{relative}").decode()
    head = _git("show", f"HEAD:{relative}").decode()
    assert pointer == head
    assert f"oid sha256:{space['archive_sha256']}" in pointer
    assert f"size {space['archive_bytes']}" in pointer
