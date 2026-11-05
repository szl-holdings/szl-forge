"""The attested Space release is 65e5e4c0, and later main does not change those trees."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "space-release-readback-2026-11-05.json"


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def test_later_main_does_not_change_the_attested_space_trees():
    document = json.loads(RECORD.read_text(encoding="utf-8"))
    assert document["hf_writes"] is False
    attested = document["attested_source_revision"]
    main = document["observed_main"]
    count = int(_git("rev-list", "--count", f"{attested}..{main}").decode().strip())
    assert count == document["commits_after_attested_release"]
    for space in document["spaces"]:
        assert space["runtime_readback"]["revision"] == attested
        names = _git("diff", "--name-only", attested, main, "--", space["source_dir"])
        assert names == b""
        assert "dispatch" in " ".join(space["acceptance"]).lower() or "Do not dispatch" in " ".join(space["acceptance"])


def test_inference_release_manifest_matches_the_attested_tree():
    document = json.loads(RECORD.read_text(encoding="utf-8"))
    attested = document["attested_source_revision"]
    manifest = json.loads((ROOT / "spaces/szl-model-inference-lab/release.json").read_text(encoding="utf-8"))
    for path, digest in manifest["source_files"].items():
        blob = _git("show", f"{attested}:spaces/szl-model-inference-lab/{path}")
        assert hashlib.sha256(blob).hexdigest() == digest


def test_foundation_archive_pointer_is_the_readback_digest():
    document = json.loads(RECORD.read_text(encoding="utf-8"))
    space = next(item for item in document["spaces"] if item["repo_id"].endswith("szl-foundation-confirmation"))
    pointer = _git(
        "show",
        f"{document['attested_source_revision']}:spaces/szl-foundation-confirmation/release.zip",
    ).decode()
    assert f"oid sha256:{space['archive_sha256']}" in pointer
    assert f"size {space['archive_bytes']}" in pointer
