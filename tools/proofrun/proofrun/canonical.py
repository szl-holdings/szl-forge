"""Canonical JSON, hashing and strict parsing (pattern carried over from the OAC kernel:
duplicate keys rejected, non-finite numbers rejected, hashes over canonical bytes)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

HEX64 = "0123456789abcdef"


class ProofError(Exception):
    """Raised when evidence is malformed. Callers turn it into a FAIL, never a pass."""


def _check_finite(v: Any, where: str = "$") -> None:
    if isinstance(v, float) and not math.isfinite(v):
        raise ProofError(f"non-finite number at {where}")
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                raise ProofError(f"non-string key at {where}")
            _check_finite(x, f"{where}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _check_finite(x, f"{where}[{i}]")


def canonical_bytes(obj: Any) -> bytes:
    _check_finite(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_bytes(obj))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_hex64(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in HEX64 for c in s)


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ProofError(f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def _reject_constant(name):
    raise ProofError(f"non-finite JSON constant {name}")


def loads_strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_no_dupes, parse_constant=_reject_constant)


def load_json(path: Path | str) -> Any:
    try:
        return loads_strict(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProofError(f"missing file: {path}")
    except json.JSONDecodeError as e:
        raise ProofError(f"invalid JSON in {path}: {e}")


def write_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def tree_entries(root: Path | str, exclude: Iterable[str] = ()) -> list[dict]:
    """[{path, size, sha256}] for every regular file under root, sorted by path."""
    root = Path(root)
    ex = set(exclude)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = p.relative_to(root).as_posix()
            if rel in ex or rel.split("/")[0] in ex:
                continue
            rows.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    return rows


def tree_sha256(entries: list[dict]) -> str:
    """Order-independent digest of a file tree: sha256 of canonical sorted (path, sha256, size)."""
    rows = sorted(({"path": e["path"], "sha256": e["sha256"], "size": e["size"]} for e in entries),
                  key=lambda r: r["path"])
    return sha256_obj(rows)
