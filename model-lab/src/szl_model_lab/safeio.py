"""Bounded regular-file reads and exclusive writes; never execute serialization.

The configured artifact root and all ancestors must be owner-controlled. These
checks defend accidental/symlink substitution, not a hostile same-user process.
"""
from __future__ import annotations
import json
import os
import stat
from pathlib import Path

def _unsafe(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)

def read_regular(path: Path, limit: int) -> bytes:
    path = path.absolute()
    for p in (path, *path.parents):
        info = p.lstat()
        if _unsafe(info):
            raise ValueError("symlink_or_reparse_path_rejected")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError("file_size_or_type_rejected")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev, before.st_ino, before.st_size
        ):
            raise ValueError("file_changed_before_read")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) != before.st_size or (opened.st_mtime_ns, opened.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError("file_changed_during_read")
    return data

def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def write_new(path: Path, data: bytes) -> None:
    # x mode is deliberate: neither training nor evaluation overwrites an artifact.
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def strict_json(raw: str | bytes) -> object:
    """Reject duplicate keys and non-finite JSON tokens at all trust boundaries."""
    def pairs(items: list[tuple[str, object]]) -> dict:
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out

    def constant(_: str) -> None:
        raise ValueError("nonfinite_json_constant")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid_json_encoding_or_depth") from exc
