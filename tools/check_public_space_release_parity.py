#!/usr/bin/env python3
"""Fail-closed public HF Space / source release-manifest byte witness.

This tool does *not* deploy, authenticate, edit providers, mint evidence,
inspect model weights, or claim running-image identity or production readiness.
It compares a locally supplied source-release manifest against an exact
immutable public HF Space repository revision and every manifest-listed file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SPACE_ID = re.compile(r"SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9_.-]{0,94}\Z")
MAX_MANIFEST_BYTES = 128 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_FILES = 120
MAX_WORKERS = 4


class ValidationError(ValueError):
    """The provided source or immutable revision cannot be safely compared."""


class RemoteUnavailable(Exception):
    """Anonymous provider read could not establish the exact bytes."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def digest(data: bytes) -> str:
    """Use the same Git-canonical CRLF->LF text normalization as the runtime."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def unique_json(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError("DUPLICATE_MANIFEST_KEY")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("INVALID_MANIFEST_JSON") from exc


def source_paths(manifest: Any) -> dict[str, str]:
    if not isinstance(manifest, dict) or manifest.get("schema") != "szl.space-source-release/v1":
        raise ValidationError("MANIFEST_SCHEMA")
    items = manifest.get("source_files")
    if not isinstance(items, dict) or not 0 < len(items) <= MAX_FILES:
        raise ValidationError("SOURCE_FILE_SET")
    for name, expected in items.items():
        if not isinstance(name, str) or not name or "\\" in name or "//" in name:
            raise ValidationError("SOURCE_PATH_INVALID")
        p = PurePosixPath(name)
        if p.is_absolute() or any(x in {"", ".", ".."} for x in name.split("/")):
            raise ValidationError("SOURCE_PATH_INVALID")
        if name.startswith(".git/") or name == ".git" or name == "release.json":
            raise ValidationError("SOURCE_PATH_RESERVED")
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValidationError("SOURCE_DIGEST_INVALID")
    return items


def read_local_manifest(source_dir: Path) -> tuple[bytes, dict[str, str], dict[str, str]]:
    root = source_dir.resolve()
    if not root.is_dir():
        raise ValidationError("SOURCE_DIRECTORY_UNAVAILABLE")
    marker = root / "release.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValidationError("LOCAL_RELEASE_MANIFEST_UNAVAILABLE")
    if marker.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValidationError("MANIFEST_TOO_LARGE")
    raw = marker.read_bytes()
    entries = source_paths(unique_json(raw))
    local: dict[str, str] = {}
    for name, expected in entries.items():
        file = root.joinpath(*name.split("/"))
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            raise ValidationError("LOCAL_SOURCE_UNAVAILABLE")
        if file.stat().st_size > MAX_SOURCE_BYTES:
            raise ValidationError("LOCAL_SOURCE_TOO_LARGE")
        actual = digest(file.read_bytes())
        if actual != expected:
            local[name] = actual
    return raw, entries, local


def fetch_public_bytes(url: str, limit: int = MAX_SOURCE_BYTES) -> bytes:
    """Fetch only from a caller-composed immutable Hugging Face URL, without auth."""
    if limit < 1 or limit > MAX_SOURCE_BYTES:
        raise ValidationError("REMOTE_BODY_LIMIT_INVALID")
    req = urllib.request.Request(
        url, headers={"User-Agent": "SZL-exact-source-readonly/1"}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=11) as response:
            if response.status != 200:
                raise RemoteUnavailable(f"HTTP_{response.status}")
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        raise RemoteUnavailable(f"HTTP_{exc.code}") from exc
    except (TimeoutError, OSError, ValueError) as exc:
        raise RemoteUnavailable(type(exc).__name__) from exc
    if len(data) > limit:
        raise RemoteUnavailable("BODY_LIMIT")
    return data


def immutable_url(space_id: str, revision: str, name: str) -> str:
    if not isinstance(space_id, str) or not SPACE_ID.fullmatch(space_id):
        raise ValidationError("SPACE_ID_INVALID")
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        raise ValidationError("SPACE_REVISION_INVALID")
    if name != "release.json":
        source_paths({"schema": "szl.space-source-release/v1", "source_files": {name: "a"*64}})
    encoded_path = urllib.parse.quote(name, safe="/")
    return f"https://huggingface.co/spaces/{space_id}/resolve/{revision}/{encoded_path}"


def verify(
    source_dir: Path,
    space_id: str,
    revision: str,
    fetch: Callable[[str, int], bytes] = fetch_public_bytes,
) -> dict[str, Any]:
    """Return a bounded report; all conclusions are restricted to repository bytes."""
    result: dict[str, Any] = {
        "schema": "szl.public-space-manifest-parity/v1",
        "space_id": space_id,
        "hf_repository_revision": revision,
        "state": "UNAVAILABLE",
        "reason": None,
        "source_file_count": None,
        "matched_count": 0,
        "mismatches": [],
        "unavailable": [],
        "local_source_integrity": "UNAVAILABLE",
        "space_repository_parity": False,
        "served_image_verified": False,
        "model_inference_verified": False,
        "receipt_signed_by_this_tool": False,
        "owner_authority_verified": False,
        "release_admitted": False,
        "provider_writes": False,
    }
    try:
        immutable_url(space_id, revision, "release.json")
        raw, entries, local_mismatch = read_local_manifest(source_dir)
    except (ValidationError, OSError) as exc:
        result["reason"] = str(exc) if isinstance(exc, ValidationError) else "LOCAL_READ_UNAVAILABLE"
        return result
    result["source_file_count"] = len(entries)
    if local_mismatch:
        result["state"] = "LOCAL_SOURCE_CONFLICT"
        result["reason"] = "LOCAL_FILE_DIGEST_DRIFT"
        result["local_mismatches"] = sorted(local_mismatch)
        return result
    result["local_source_integrity"] = "MANIFEST_HASHES_MATCH_LOCAL_FILES"
    try:
        remote_manifest = fetch(immutable_url(space_id, revision, "release.json"), MAX_MANIFEST_BYTES)
    except RemoteUnavailable as exc:
        result["reason"] = "RELEASE_MANIFEST_" + exc.code
        return result
    if remote_manifest != raw:
        result["state"] = "DRIFT"
        result["reason"] = "RELEASE_MANIFEST_BYTES_DIFFER"
        result["mismatches"] = ["release.json"]
        return result

    def check(item: tuple[str, str]) -> tuple[str, str]:
        name, expected = item
        try:
            actual = fetch(immutable_url(space_id, revision, name), MAX_SOURCE_BYTES)
        except RemoteUnavailable as exc:
            return name, "UNAVAILABLE_" + exc.code
        return name, "MATCH" if digest(actual) == expected else "MISMATCH"

    # Fixed worker cap; no retries, no secrets, no mutable ref or POST.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        observations = list(executor.map(check, entries.items()))
    result["matched_count"] = sum(state == "MATCH" for _, state in observations)
    result["mismatches"] = sorted(name for name, state in observations if state == "MISMATCH")
    result["unavailable"] = sorted(
        [{"path": name, "reason": state} for name, state in observations
         if state.startswith("UNAVAILABLE")],
        key=lambda entry: entry["path"],
    )
    if result["unavailable"]:
        result["state"] = "UNAVAILABLE"
        result["reason"] = "REMOTE_SOURCE_UNAVAILABLE"
    elif result["mismatches"]:
        result["state"] = "DRIFT"
        result["reason"] = "REMOTE_SOURCE_DIGEST_DRIFT"
    else:
        result["state"] = "REPOSITORY_BYTES_MATCH_ONLY"
        result["reason"] = "MANIFEST_BYTES_AND_LISTED_FILES_MATCH"
        result["space_repository_parity"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.source_dir, args.space_id, args.hf_revision)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"{result['state']} matched={result['matched_count']} report={args.output}")
    else:
        sys.stdout.write(payload)
    return 0 if result["space_repository_parity"] else 1 if result["state"] == "DRIFT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
