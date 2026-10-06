"""Materialize ops-health/hub/oac-v1/ (v1's six published files) from v1's staged model directory.

SYNTHETIC.  The v2 research code reads v1's published kernel, model and receipt from
``hub/oac-v1/`` under its root (``registry.V1_HUB_DIR``): the final analysis re-hashes them, the
legacy-source record cross-checks v1's receipt, and some tests load v1's kernel.  v1's files are
already staged in this repository, in the one directory matching
``*/huggingface/model/oac-system-health-v1/`` (it mirrors the Hub model at
``registry.V1_HUB_REVISION``), so they are copied, not committed a second time.

Before copying, every file that v2's committed records bind is checked against those records:
``v2/results/final/MANIFEST.json`` (``contenders.v1.files``) binds v1's kernel, model and receipt.
A missing or different file is refused; nothing is written then.  An existing destination file
is accepted only if it is byte-identical (the tool is idempotent and never overwrites).

Usage (from the szl-forge repository root):
    python -B ops-health/tools/materialize_v1_hub.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

OPS_HEALTH = Path(__file__).resolve().parents[1]
REPO = OPS_HEALTH.parent
V1_STAGED_GLOB = "*/huggingface/model/oac-system-health-v1"
DEFAULT_DEST = OPS_HEALTH / "hub" / "oac-v1"
FILES = ("LICENSE", "README.md", "artifact_receipt.json", "example_input.json", "model.json",
         "oac_operational_health.py")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bound_hashes(root: Path) -> dict[str, str]:
    """File name -> sha256 that v2's committed score manifest binds for hub/oac-v1/<name>."""
    manifest = json.loads((root / "v2" / "results" / "final" / "MANIFEST.json").read_text("utf-8"))
    files = manifest["contenders"]["v1"]["files"]
    out = {}
    for rel, digest in files.items():
        if not rel.startswith("hub/oac-v1/"):
            raise SystemExit(f"unexpected v1 file binding {rel!r}; refusing")
        out[rel[len("hub/oac-v1/"):]] = digest
    if not out:
        raise SystemExit("no v1 file binding in v2/results/final/MANIFEST.json; refusing")
    return out


def staged_v1_dir() -> Path:
    hits = sorted(p for p in REPO.glob(V1_STAGED_GLOB) if p.is_dir())
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one staged v1 model directory ({V1_STAGED_GLOB}), "
                         f"found {len(hits)}; refusing")
    return hits[0]


def materialize(source: Path, dest: Path, root: Path) -> dict:
    bound = bound_hashes(root)
    problems, files = [], {}
    payload = {}
    for name in FILES:
        path = source / name
        if not path.is_file():
            problems.append(f"{name}: missing in {source}")
            continue
        data = path.read_bytes()
        digest = _sha256(data)
        files[name] = {"sha256": digest, "bytes": len(data), "bound": bound.get(name)}
        if name in bound and digest != bound[name]:
            problems.append(f"{name}: sha256 {digest} differs from the bound {bound[name]}")
        if b"\r" in data and name in bound:
            problems.append(f"{name}: carries CR bytes")
        payload[name] = data
    unknown = sorted(set(bound) - set(FILES))
    if unknown:
        problems.append(f"bound files not materialized: {unknown}")
    written, identical = [], []
    if not problems:
        for name, data in payload.items():
            target = dest / name
            if target.exists():
                if target.read_bytes() != data:
                    problems.append(f"{target} exists with other bytes; refusing to overwrite")
                else:
                    identical.append(name)
        if not problems:
            dest.mkdir(parents=True, exist_ok=True)
            for name, data in payload.items():
                if name in identical:
                    continue
                with (dest / name).open("xb") as handle:
                    handle.write(data)
                written.append(name)
    return {"schema": "szl-oac/ops-health-v1-hub-materialize/v1", "ok": not problems,
            "problems": problems, "files": files, "written": written, "already_identical": identical}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="v1's staged model directory (default: discovered)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--root", type=Path, default=OPS_HEALTH,
                        help="the directory holding v2/ (default: this tool's ops-health/)")
    args = parser.parse_args(argv)
    source = args.source if args.source is not None else staged_v1_dir()
    report = materialize(source, args.dest, args.root)
    report["source"] = source.resolve().relative_to(REPO.resolve()).as_posix()         if source.resolve().is_relative_to(REPO.resolve()) else source.as_posix()
    report["dest"] = args.dest.as_posix()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
