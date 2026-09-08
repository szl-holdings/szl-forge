#!/usr/bin/env python3
"""Bind an existing local, public HF snapshot to immutable provider file hashes.

Metadata-only network reads. No downloads, remote code, credentials or uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def file_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def verify(repo_id: str, revision: str, directory: Path, output: Path) -> dict:
    from huggingface_hub import HfApi

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("an immutable 40-character revision is required")
    if directory.is_symlink() or not directory.is_dir() or output.exists():
        raise ValueError("existing directory and a new manifest path are required")
    root = directory.resolve(strict=True)
    if output.resolve().is_relative_to(root):
        raise ValueError("manifest must be outside snapshot")
    info = HfApi(token=False).model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision or info.private:
        raise ValueError("only exact public snapshots are admitted")
    admitted = {entry.rfilename: entry for entry in info.siblings}
    rows = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix().casefold()):
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise ValueError("symlinks are not admitted")
        if not path.is_file():
            continue
        name = relative.as_posix()
        item = admitted.get(name)
        if item is None or item.size != path.stat().st_size:
            raise ValueError(f"provider manifest mismatch: {name}")
        sha256, git_blob = file_hashes(path)
        if item.lfs is not None:
            expected = item.lfs.sha256
            actual = sha256
            algorithm = "sha256"
        else:
            expected = item.blob_id
            actual = git_blob
            algorithm = "git-sha1"
        if not expected or actual != expected:
            raise ValueError(f"provider content hash mismatch: {name}")
        rows.append({"path": name, "bytes": item.size, "sha256": sha256,
                     "provider_hash_algorithm": algorithm, "provider_hash": expected})
    if not rows:
        raise ValueError("snapshot is empty")
    report = {"schema": "szl.local-artifact/v1", "repo_id": repo_id,
              "revision": revision, "files": rows,
              "verified_from": "PINNED_HUB_METADATA_AND_LOCAL_BYTES"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo, args.revision, args.directory, args.output)
    print(json.dumps({"repo_id": result["repo_id"], "revision": result["revision"],
                      "verified_files": len(result["files"])}))
