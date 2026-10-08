"""Generate and seal the registered v2 splits (PREREGISTRATION sections 3-4). SYNTHETIC only.

Usage (from the repository root):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.make_data --write    # one time; refuses to overwrite
    PYTHONUTF8=1 py -3.12 -B -m v2.research.make_data --verify   # hash-only reproducibility check

--write generates every split with its own derived seed, writes canonical JSONL (open splits
under v2/data/, test and shift splits under v2/data/sealed/), then writes v2/data/MANIFEST.json.
No label statistics or row contents of any split are printed or recorded.

--verify regenerates every split in memory and compares sha256 digests against MANIFEST.json and
against the bytes on disk.  It hashes bytes only; it never parses a sealed file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from . import generator as gen
from . import registry

MANIFEST_SCHEMA = "szl-oac/ops-health-data-manifest/v2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _split_bytes(name: str) -> bytes:
    regime, rows, _sealed, _use = registry.SPLIT_TABLE[name]
    return gen.jsonl_bytes(gen.generate_rows(name, rows, regime))


def _check_preregistration() -> str:
    digest = _sha256(registry.PREREG_PATH.read_bytes())
    if digest != registry.PREREG_SHA256:
        raise SystemExit(
            f"PREREGISTRATION.md sha256 {digest} != registered {registry.PREREG_SHA256}; refusing"
        )
    return digest


def write_all(data_dir: Path = registry.DATA_DIR) -> dict:
    prereg = _check_preregistration()
    manifest_path = data_dir / "MANIFEST.json"
    targets = [registry.split_path(name, data_dir) for name, *_ in registry.SPLITS]
    existing = [p for p in [manifest_path, *targets] if p.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite sealed/registered data: {existing}")
    splits = {}
    for name, regime, rows, sealed, use in registry.SPLITS:
        data = _split_bytes(name)
        path = registry.split_path(name, data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:  # exclusive create
            handle.write(data)
        splits[name] = {
            "path": registry.relpath(path),
            "regime": regime,
            "rows": rows,
            "sealed": sealed,
            "use": use,
            "seed": gen.derive_seed(name),
            "seed_material": f"{gen.SEED_NAMESPACE}:{gen.MASTER_SEED}:{name}",
            "sha256": _sha256(data),
            "bytes": len(data),
        }
    legacy = {}
    for split in ("train", "validation", "test"):
        path = registry.LEGACY_DIR / f"{split}.jsonl"
        legacy[f"legacy_v1_{split}"] = {
            "path": registry.relpath(path),
            "sha256": _sha256(path.read_bytes()),
            "use": "secondary only (REPORTED comparability)" if split == "test" else "reference",
        }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "synthetic": True,
        "generator_version": gen.GENERATOR_VERSION,
        "generator_path": registry.relpath(registry.GENERATOR_PATH),
        "generator_sha256": _sha256(registry.GENERATOR_PATH.read_bytes()),
        "make_data_sha256": _sha256(Path(__file__).read_bytes()),
        "master_seed": gen.MASTER_SEED,
        "seed_derivation": (
            'int.from_bytes(sha256(f"oac-ops-health-v2:{M}:{split}".encode()).digest()[:8], "big")'
        ),
        "row_schema": gen.ROW_SCHEMA,
        "jsonl_encoding": "json.dumps(row, sort_keys=True, separators=(',', ':')) + '\\n', utf-8",
        "preregistration_path": registry.relpath(registry.PREREG_PATH),
        "preregistration_sha256": prereg,
        "python_version": platform.python_version(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": splits,
        "legacy_v1": legacy,
        "pooled_shift_suite": [n for n in registry.SEALED_SPLITS if n.startswith("shift_")],
        "sealing_rule": (
            "files under v2/data/sealed/ are parsed only by v2/research/sealed_guard.py with "
            "purpose FINAL_TEST_OPENING, exactly once"
        ),
    }
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
    return manifest


def verify_all(data_dir: Path = registry.DATA_DIR) -> dict:
    manifest = json.loads((data_dir / "MANIFEST.json").read_text("utf-8"))
    problems = []
    if manifest["generator_sha256"] != _sha256(registry.GENERATOR_PATH.read_bytes()):
        problems.append("generator.py changed since the data were generated")
    if manifest["preregistration_sha256"] != _sha256(registry.PREREG_PATH.read_bytes()):
        problems.append("PREREGISTRATION.md changed")
    checked = {}
    for name, *_ in registry.SPLITS:
        expected = manifest["splits"][name]["sha256"]
        regenerated = _sha256(_split_bytes(name))
        on_disk = _sha256(registry.split_path(name, data_dir).read_bytes())  # hash only
        checked[name] = {"regenerated_matches": regenerated == expected, "disk_matches": on_disk == expected}
        if regenerated != expected:
            problems.append(f"{name}: regenerated sha256 differs from MANIFEST")
        if on_disk != expected:
            problems.append(f"{name}: on-disk sha256 differs from MANIFEST")
    return {"ok": not problems, "problems": problems, "checked": checked}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        started = time.perf_counter()
        manifest = write_all()
        summary = {
            name: {k: info[k] for k in ("path", "regime", "rows", "sealed", "seed", "sha256")}
            for name, info in manifest["splits"].items()
        }
        print(json.dumps({"ok": True, "generator_sha256": manifest["generator_sha256"],
                          "created_utc": manifest["created_utc"], "splits": summary,
                          "seconds": round(time.perf_counter() - started, 3)}, indent=2, sort_keys=True))
        return 0
    result = verify_all()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
