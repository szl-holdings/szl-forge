#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned Lean eval slices: deterministic selection and digest.

A slice names a source repo, a path at a pinned commit, a format and a filter.
Its digest is reproducible from a checkout of that commit:

    sha256("\n".join(sorted(sha256(canonical_json(row)) for row in selected)))

For ``format=file`` the digest is the sha256 of the file bytes and rows is 1.
This module only reads local files. It proves nothing about a model; a slice
is an evaluation target list, not a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

FORMATS = ("jsonl", "atlas", "file")
LANES = ("core", "mathlib")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _match(row: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in flt.items())


def select(root: Path, spec: Dict[str, Any]) -> List[Any]:
    path = root / spec["path"]
    fmt = spec["format"]
    flt = spec.get("filter") or {}
    if fmt == "jsonl":
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [r for r in rows if _match(r, flt)]
    if fmt == "atlas":
        rows = json.loads(path.read_text(encoding="utf-8"))["atlas"]
        return [r for r in rows if _match(r, flt)]
    if fmt == "file":
        return [path.read_bytes()]
    raise ValueError("unknown format")


def digest(root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    rows = select(root, spec)
    if spec["format"] == "file":
        return {"rows": 1, "sha256": hashlib.sha256(rows[0]).hexdigest()}
    leaves = sorted(hashlib.sha256(canonical(r).encode("utf-8")).hexdigest() for r in rows)
    return {"rows": len(rows), "sha256": hashlib.sha256("\n".join(leaves).encode("utf-8")).hexdigest()}


def verify_map(map_path: Path, sources: Path) -> List[str]:
    """Recompute every slice from ``sources/<repo>`` checkouts; report mismatches."""
    data = json.loads(map_path.read_text(encoding="utf-8"))
    errors: List[str] = []
    for f in data["flagships"]:
        for sl in f["lean"]["eval_slices"]:
            root = sources / sl["repo"]
            if not root.is_dir():
                errors.append(f"{f['line']}:{sl['id']}: source checkout missing")
                continue
            got = digest(root, sl)
            if got != {"rows": sl["rows"], "sha256": sl["sha256"]}:
                errors.append(f"{f['line']}:{sl['id']}: expected {sl['rows']}/{sl['sha256'][:12]} got {got['rows']}/{got['sha256'][:12]}")
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, default=Path(__file__).resolve().parents[1] / "portfolio" / "flagship_training_map.v1.json")
    ap.add_argument("--sources", type=Path, required=True, help="directory holding <repo>/ checkouts at the pinned commits")
    args = ap.parse_args(argv)
    errors = verify_map(args.map, args.sources)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
