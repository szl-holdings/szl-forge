#!/usr/bin/env python3
"""Bind Git LFS-tracked frozen archives to their frozen SHA-256 digest.

Some sealed artifacts are committed as Git LFS pointers. For those files the
pointer's ``oid sha256:`` IS the integrity binding: it must equal the frozen
digest declared by the consuming source, and the working tree must hold
either that exact pointer (LFS not materialized) or bytes whose SHA-256 and
size equal the pointer (LFS materialized). Everything else fails closed:

* MISSING                    the archive is absent or not a regular file
* UNTRACKED                  the archive is not tracked by Git
* STAGED_CHANGE              the index entry differs from HEAD
* LFS_FILTER_DISABLED        ``.gitattributes`` does not route it through LFS
* COMMITTED_NOT_LFS_POINTER  HEAD holds raw bytes instead of a pointer
* POINTER_OID_MISMATCH       the committed pointer oid is not the frozen digest
* WORKTREE_POINTER_MISMATCH  the checked-out pointer differs from HEAD
* WORKTREE_DIGEST_MISMATCH   materialized bytes do not hash to the digest
* FROZEN_DIGEST_UNREADABLE   the declaring source has no exact digest constant

Only Git plumbing is used (no clean/smudge filter is executed), so the result
does not depend on whether git-lfs is installed on the machine. Stdlib only.
This is an integrity check; it does not change any registered result.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
import re
import subprocess

# Repository path -> (declaring source file, module-level constant name).
# The digest is read from the consuming source so there is one source of truth.
BOUND_ARCHIVES: dict[str, tuple[str, str]] = {
    "spaces/szl-foundation-confirmation/release.zip": (
        "spaces/szl-foundation-confirmation/app.py",
        "ARCHIVE_SHA256",
    ),
}

LFS_POINTER = re.compile(
    rb"version https://git-lfs\.github\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size (0|[1-9][0-9]{0,15})\n"
)
MAX_POINTER_BYTES = 1024
SHA256 = re.compile(r"[0-9a-f]{64}")


class BindingError(ValueError):
    """The LFS archive binding failed closed."""

    def __init__(self, code: str, path: str, detail: str = ""):
        self.code = code
        self.path = path
        super().__init__(f"{code}: {path}" + (f" ({detail})" if detail else ""))


def parse_pointer(data: bytes) -> tuple[str, int] | None:
    """Return (oid, size) for an exact canonical LFS pointer, else None."""
    if len(data) > MAX_POINTER_BYTES:
        return None
    match = LFS_POINTER.fullmatch(data)
    if match is None:
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def frozen_digest(root: pathlib.Path, source: str, name: str) -> str:
    """Read one exact lowercase SHA-256 string constant without importing."""
    try:
        tree = ast.parse((root / source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        raise BindingError("FROZEN_DIGEST_UNREADABLE", source, type(exc).__name__) from None
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(values) != 1 or SHA256.fullmatch(values[0]) is None:
        raise BindingError("FROZEN_DIGEST_UNREADABLE", source, name)
    return values[0]


def _git(root: pathlib.Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, timeout=30, check=False
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, args)
    return completed.stdout


def lfs_attributes(root: pathlib.Path, path: str) -> dict[str, str]:
    raw = _git(root, "check-attr", "-z", "filter", "diff", "merge", "--", path)
    fields = raw.split(b"\0")
    result = {}
    for index in range(0, len(fields) - 2, 3):
        result[fields[index + 1].decode()] = fields[index + 2].decode()
    return result


def verify_archive(root: pathlib.Path, path: str, expected: str) -> dict:
    """Verify one LFS-bound archive; return evidence or raise BindingError."""
    root = pathlib.Path(root)
    if SHA256.fullmatch(expected or "") is None:
        raise BindingError("FROZEN_DIGEST_UNREADABLE", path, "expected digest")
    target = root / path
    if target.is_symlink() or not target.is_file():
        raise BindingError("MISSING", path)
    try:
        staged = _git(root, "ls-files", "-s", "-z", "--", path).split(b"\0")
        staged = [entry for entry in staged if entry]
        if len(staged) != 1:
            raise BindingError("UNTRACKED", path)
        meta, _, name = staged[0].partition(b"\t")
        mode, index_blob, stage = meta.split(b" ")
        if name.decode() != path or mode != b"100644" or stage != b"0":
            raise BindingError("UNTRACKED", path, "unexpected index entry")
        try:
            head_blob = _git(root, "rev-parse", "--verify", "--quiet", f"HEAD:{path}").strip()
        except subprocess.CalledProcessError:
            raise BindingError("UNTRACKED", path, "absent from HEAD") from None
        if head_blob != index_blob:
            raise BindingError("STAGED_CHANGE", path)
        attributes = lfs_attributes(root, path)
        committed = _git(root, "cat-file", "blob", head_blob.decode())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        if isinstance(exc, BindingError):
            raise
        raise BindingError("UNTRACKED", path, type(exc).__name__) from None
    if any(attributes.get(key) != "lfs" for key in ("filter", "diff", "merge")):
        raise BindingError("LFS_FILTER_DISABLED", path, repr(attributes))
    pointer = parse_pointer(committed)
    if pointer is None:
        raise BindingError("COMMITTED_NOT_LFS_POINTER", path)
    oid, size = pointer
    if oid != expected:
        raise BindingError("POINTER_OID_MISMATCH", path, oid)
    with target.open("rb") as stream:
        head = stream.read(MAX_POINTER_BYTES + 1)
    if parse_pointer(head) is not None:
        if head != committed:
            raise BindingError("WORKTREE_POINTER_MISMATCH", path)
        state = "LFS_POINTER_NOT_MATERIALIZED"
    else:
        digest = hashlib.sha256()
        observed = 0
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
                observed += len(block)
        if digest.hexdigest() != expected or observed != size:
            raise BindingError("WORKTREE_DIGEST_MISMATCH", path)
        state = "LFS_MATERIALIZED_BYTES_MATCH"
    return {
        "path": path,
        "frozen_sha256": expected,
        "pointer_oid": oid,
        "pointer_size": size,
        "worktree": state,
        "lfs_attributes": attributes,
    }


def verify_registered(root: pathlib.Path, registry: dict | None = None) -> list[dict]:
    """Verify every registered archive against its declaring source digest."""
    root = pathlib.Path(root)
    evidence = []
    for path, (source, name) in sorted((registry or BOUND_ARCHIVES).items()):
        evidence.append(verify_archive(root, path, frozen_digest(root, source, name)))
    return evidence


if __name__ == "__main__":
    import json
    import sys

    try:
        print(json.dumps(verify_registered(pathlib.Path(__file__).resolve().parents[1]),
                         indent=2, sort_keys=True))
    except BindingError as error:
        print(f"::error::lfs_archive_binding: {error}", file=sys.stderr)
        raise SystemExit(1)
