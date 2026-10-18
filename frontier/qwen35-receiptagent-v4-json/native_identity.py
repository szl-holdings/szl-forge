# SPDX-License-Identifier: Apache-2.0
"""Data-only comparison of a native directory with externally pinned file bytes.

This distinct native-file profile is NOT the v1 admission's container identity.
The caller supplies an expected manifest SHA-256 obtained through its separate
review/trust procedure. Neither the manifest nor this verifier establishes that
procedure, owner approval, installed-wheel provenance, dependency completeness,
the identity of the executing interpreter, standard library or native libraries,
or permission to import, train, serve or launch anything. A successful comparison
only observes the named directory's exact bytes at the time of these reads.

Only regular files with one link and plain directories are supported. Names are
portable ASCII, case-distinct, slash-separated relative paths. Empty additional
directories, symbolic links, junctions, hardlinks, special files and bytecode not
explicitly committed in the manifest are rejected. The manifest lives outside
the tree. Contents such as .pth files are opaque bytes: no site initialization,
package import, RECORD interpretation, executable discovery or file execution
is performed by this module. Import this verifier from an already trusted
stdlib-only bootstrap; importing it cannot undo Python startup code that already
ran. A future launcher must separately qualify its bootstrap and dependencies.

Reads and inventory snapshots detect ordinary changes and refuse partial results.
They assume a trusted single writer and are not a hostile same-UID sandbox or an
immutable filesystem snapshot. The directory must be separately held read-only
before consumption. This module contains no installation, network, signature,
nonce, GPU, model or process-launch interface, and no production runtime hookup.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MANIFEST_SCHEMA = "szl.receiptagent.native-file-manifest/v1"
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_FILES = 200000
MAX_TREE_ENTRIES = 400000
MAX_SINGLE_FILE_BYTES = 8 * 1024**3
MAX_TOTAL_FILE_BYTES = 64 * 1024**3
MAX_PATH_BYTES = 1024
MAX_DEPTH = 32
HASH_CHUNK_BYTES = 1024**2
_HEX = re.compile(r"[0-9a-f]{64}")
_SEGMENT = re.compile(r"[A-Za-z0-9_.+@-]{1,255}")
_WINDOWS_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", re.I)


class NativeIdentityError(ValueError):
    """Value-free normal error display, not secrecy against Python introspection."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if condition is not True:
        raise NativeIdentityError(code)


def _plain(meta) -> bool:
    return not stat.S_ISLNK(meta.st_mode) and not getattr(meta, "st_file_attributes", 0) & 0x400


def _checked_path(value: Path) -> Path:
    _require(isinstance(value, Path) and value.is_absolute(), "ABSOLUTE_PATH_REQUIRED")
    _require(".." not in value.parts, "PATH_TRAVERSAL_FORBIDDEN")
    for ancestor in reversed((value, *value.parents)):
        _require(_plain(ancestor.lstat()), "PATH_ALIAS_FORBIDDEN")
    return value


def _stable(meta) -> tuple:
    # Windows fstat and path lstat may use different ctime semantics. Cross-check
    # stable fields, but compare each API's own ctime before/after separately.
    return (meta.st_dev, meta.st_ino, meta.st_mode, meta.st_size, meta.st_mtime_ns, meta.st_nlink)


def _stamp(meta) -> tuple:
    return (*_stable(meta), meta.st_ctime_ns)


def _capture_file(path: Path, *, maximum: int, expected_size: int | None = None,
                  retain: bool = False) -> tuple[str, bytes | None, tuple]:
    before_path = path.lstat()
    _require(_plain(before_path) and stat.S_ISREG(before_path.st_mode)
             and before_path.st_nlink == 1, "REGULAR_UNALIASED_FILE_REQUIRED")
    _require(0 <= before_path.st_size <= maximum, "FILE_BYTE_CAP")
    if expected_size is not None:
        _require(before_path.st_size == expected_size, "FILE_SIZE_MISMATCH")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_BINARY", 0))
    try:
        before_fd = os.fstat(descriptor)
        _require(stat.S_ISREG(before_fd.st_mode) and before_fd.st_nlink == 1
                 and _stable(before_path) == _stable(before_fd), "FILE_CHANGED_DURING_READ")
        hasher = hashlib.sha256()
        chunks = [] if retain else None
        size = 0
        while True:
            block = os.read(descriptor, min(HASH_CHUNK_BYTES, maximum - size + 1))
            if not block:
                break
            size += len(block)
            _require(size <= maximum, "FILE_BYTE_CAP")
            hasher.update(block)
            if chunks is not None:
                chunks.append(block)
        after_fd = os.fstat(descriptor)
        after_path = path.lstat()
        _require(_stamp(before_fd) == _stamp(after_fd)
                 and _stamp(before_path) == _stamp(after_path)
                 and _stable(after_fd) == _stable(after_path)
                 and size == before_fd.st_size, "FILE_CHANGED_DURING_READ")
        return hasher.hexdigest(), None if chunks is None else b"".join(chunks), _stamp(after_path)
    finally:
        os.close(descriptor)


def _pairs(pairs):
    value = {}
    for key, child in pairs:
        _require(key not in value, "DUPLICATE_JSON_FIELD")
        value[key] = child
    return value


def _forbidden_number(_value):
    raise NativeIdentityError("INTEGER_ONLY_MANIFEST")


def _relative_name(value: str) -> str:
    _require(type(value) is str and 0 < len(value) <= MAX_PATH_BYTES, "RELATIVE_PATH_INVALID")
    parts = value.split("/")
    _require(len(parts) <= MAX_DEPTH and all(
        part not in (".", "..") and _SEGMENT.fullmatch(part) is not None
        and not part.endswith(".") and _WINDOWS_RESERVED.fullmatch(part) is None
        for part in parts), "RELATIVE_PATH_INVALID")
    return value


def _manifest(raw: bytes) -> tuple[dict, set[str], int]:
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs,
                           parse_float=_forbidden_number, parse_constant=_forbidden_number)
    except NativeIdentityError:
        raise
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise NativeIdentityError("MANIFEST_JSON_INVALID") from exc
    _require(type(value) is dict and set(value) == {"schema", "files"}
             and value["schema"] == MANIFEST_SCHEMA, "MANIFEST_FIELDS_INVALID")
    files = value["files"]
    _require(type(files) is list and 1 <= len(files) <= MAX_FILES, "MANIFEST_FILE_COUNT_INVALID")
    entries, directories, folded = {}, set(), set()
    total = 0
    for entry in files:
        _require(type(entry) is dict and set(entry) == {"path", "size_bytes", "sha256"},
                 "MANIFEST_FILE_FIELDS_INVALID")
        name = _relative_name(entry["path"])
        _require(name.casefold() not in folded, "DUPLICATE_OR_CASE_ALIAS_PATH")
        folded.add(name.casefold())
        size = entry["size_bytes"]
        _require(type(size) is int and 0 <= size <= MAX_SINGLE_FILE_BYTES, "MANIFEST_FILE_SIZE_INVALID")
        digest = entry["sha256"]
        _require(type(digest) is str and _HEX.fullmatch(digest) is not None, "MANIFEST_DIGEST_INVALID")
        total += size
        _require(total <= MAX_TOTAL_FILE_BYTES, "TREE_BYTE_CAP")
        entries[name] = (size, digest)
        parts = name.split("/")
        directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    _require(list(entries) == sorted(entries), "MANIFEST_FILE_ORDER_INVALID")
    _require(not set(entries) & directories, "FILE_DIRECTORY_COLLISION")
    all_names = list(entries) + list(directories)
    _require(len({name.casefold() for name in all_names}) == len(all_names), "DUPLICATE_OR_CASE_ALIAS_PATH")
    _require(len(all_names) <= MAX_TREE_ENTRIES, "TREE_ENTRY_CAP")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii")
    _require(raw == canonical, "NONCANONICAL_MANIFEST")
    return entries, directories, total


def _inventory(root: Path) -> tuple[dict, dict]:
    files, directories = {}, {"": _stamp(root.lstat())}
    pending = [(root, "")]
    observed_bytes = 0
    while pending:
        path, relative = pending.pop()
        before = path.lstat()
        _require(_plain(before) and stat.S_ISDIR(before.st_mode), "PLAIN_DIRECTORY_REQUIRED")
        _require(_stamp(before) == directories[relative], "TREE_CHANGED_DURING_READ")
        with os.scandir(path) as children:
            for child in children:
                name = _relative_name(relative + "/" + child.name if relative else child.name)
                # Windows DirEntry.stat may omit inode/link-count fields. Use
                # the same pathname API as the independent later snapshots.
                meta = Path(child.path).lstat()
                _require(_plain(meta), "PATH_ALIAS_FORBIDDEN")
                if stat.S_ISDIR(meta.st_mode):
                    directories[name] = _stamp(meta)
                    pending.append((Path(child.path), name))
                else:
                    _require(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1,
                             "REGULAR_UNALIASED_FILE_REQUIRED")
                    _require(0 <= meta.st_size <= MAX_SINGLE_FILE_BYTES, "FILE_BYTE_CAP")
                    observed_bytes += meta.st_size
                    _require(observed_bytes <= MAX_TOTAL_FILE_BYTES, "TREE_BYTE_CAP")
                    files[name] = _stamp(meta)
                    _require(len(files) <= MAX_FILES, "TREE_FILE_COUNT_CAP")
                _require(len(files) + len(directories) - 1 <= MAX_TREE_ENTRIES, "TREE_ENTRY_CAP")
        _require(_stamp(before) == _stamp(path.lstat()), "TREE_CHANGED_DURING_READ")
    return files, directories


@dataclass(frozen=True)
class NativeFileIdentity:
    """Readback counts only, not a launch capability or authenticity verdict."""
    manifest_sha256: str
    file_count: int
    total_bytes: int


def verify_native_file_manifest(root: Path, manifest_path: Path, *,
                                expected_manifest_sha256: str) -> NativeFileIdentity:
    """Compare pinned manifest bytes and an exact directory; never execute input.

    The external expected hash is mandatory and never read from the manifest.
    Unsupported files or any mismatch raise, with no partial identity result.
    Caps apply before streaming and again while reading; multi-GiB package files
    are supported without retaining their bytes in memory.
    """
    _require(type(expected_manifest_sha256) is str
             and _HEX.fullmatch(expected_manifest_sha256) is not None, "EXTERNAL_MANIFEST_PIN_REQUIRED")
    try:
        _checked_path(root)
        _checked_path(manifest_path)
        _require(stat.S_ISDIR(root.lstat().st_mode), "PLAIN_DIRECTORY_REQUIRED")
        _require(root != manifest_path and root not in manifest_path.parents, "MANIFEST_MUST_BE_OUTSIDE_TREE")
        digest, raw, manifest_stamp = _capture_file(manifest_path, maximum=MAX_MANIFEST_BYTES, retain=True)
        _require(digest == expected_manifest_sha256, "MANIFEST_PIN_MISMATCH")
        entries, expected_directories, total = _manifest(raw)
        initial = _inventory(root)
        _require(set(initial[0]) == set(entries)
                 and set(initial[1]) == expected_directories | {""}, "TREE_INVENTORY_MISMATCH")
        for name, (size, expected_digest) in entries.items():
            actual_digest, _, observed_stamp = _capture_file(root.joinpath(*name.split("/")),
                                                            maximum=MAX_SINGLE_FILE_BYTES,
                                                            expected_size=size)
            _require(observed_stamp == initial[0][name], "TREE_CHANGED_DURING_READ")
            _require(actual_digest == expected_digest, "FILE_DIGEST_MISMATCH")
        _require(_inventory(root) == initial, "TREE_CHANGED_DURING_READ")
        _checked_path(root)
        _checked_path(manifest_path)
        _require(_stamp(manifest_path.lstat()) == manifest_stamp, "MANIFEST_CHANGED_DURING_READ")
        return NativeFileIdentity(digest, len(entries), total)
    except OSError:
        # Normal tracebacks must not print the filesystem exception's paths.
        # Trusted Python callers can still introspect exception context.
        raise NativeIdentityError("FILESYSTEM_UNAVAILABLE") from None
