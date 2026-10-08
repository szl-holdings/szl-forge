"""Record (and re-verify) the byte-exact legacy v1 split copies under v2/data/legacy_v1/.

Truth label of what this script establishes when run: MEASURED (sha256 of bytes on disk,
git blob ids read from the read-only szl-forge clone at the pinned commit).

Usage (from the repository root):
    py -3.12 -B -m v2.research.legacy_v1_source --write    # writes SOURCE.json
    py -3.12 -B -m v2.research.legacy_v1_source --verify   # re-checks copies against SOURCE.json

Naming rule: v2 files never repeat the legacy upstream directory name of the v1 gateway
package.  Upstream paths are therefore recorded relative to that package directory and are
made verifiable by their git blob ids at the pinned commit
(`git -C src/szl-forge ls-tree -r <commit> | grep <blob id>` resolves the full path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORGE = ROOT / "src" / "szl-forge"
FORGE_COMMIT = "9a6fdd2c7bdb1660de009cef974d7df46866960a"
V1_HUB_REVISION = "dd7d109813abcd90c5250106dc2eabd2804e7ac3"
LEGACY_DIR = ROOT / "v2" / "data" / "legacy_v1"
SPLIT_ROWS = {"train": 768, "validation": 192, "test": 240}
PKG_PLACEHOLDER = "{V1_GATEWAY_PACKAGE_DIR}"
_ANCHOR = "/operational-model/data/train.jsonl"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(FORGE), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _package_dir() -> str:
    """Discover the v1 gateway package directory at the pinned commit (not hard-coded)."""
    hits = [
        path[: -len(_ANCHOR)]
        for path in _git("ls-tree", "-r", "--name-only", FORGE_COMMIT).splitlines()
        if path.endswith(_ANCHOR)
    ]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one v1 package directory, found {len(hits)}")
    return hits[0]


def _blob_id(relative: str) -> str:
    return _git("rev-parse", f"{FORGE_COMMIT}:{_package_dir()}/{relative}")


def _committed_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(FORGE), "cat-file", "-p", f"{FORGE_COMMIT}:{_package_dir()}/{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def build_source_record() -> dict:
    receipt_rel = "operational-model/artifacts/dataset-receipt.json"
    receipt_bytes = _committed_bytes(receipt_rel)
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    hub_receipt = json.loads((ROOT / "hub" / "oac-v1" / "artifact_receipt.json").read_text("utf-8"))
    files = {}
    for split, rows in SPLIT_ROWS.items():
        rel = f"operational-model/data/{split}.jsonl"
        copied = (LEGACY_DIR / f"{split}.jsonl").read_bytes()
        committed = _committed_bytes(rel)
        if copied != committed:
            raise SystemExit(f"legacy copy {split}.jsonl differs from the committed upstream blob")
        digest = sha256_bytes(copied)
        if receipt["files"][f"data/{split}.jsonl"] != digest:
            raise SystemExit(f"{split}.jsonl sha256 does not match the v1 dataset receipt")
        line_count = copied.count(b"\n")
        if line_count != rows:
            raise SystemExit(f"{split}.jsonl has {line_count} rows, expected {rows}")
        files[f"{split}.jsonl"] = {
            "source_path": f"{PKG_PLACEHOLDER}/{rel}",
            "git_blob_sha1": _blob_id(rel),
            "sha256": digest,
            "bytes": len(copied),
            "rows": rows,
            "matches_dataset_receipt": True,
            "byte_identical_to_committed_blob": True,
        }
    receipt_sha = sha256_bytes(receipt_bytes)
    return {
        "schema": "szl-oac/ops-health-legacy-source/v2",
        "what": "byte-exact copies of the v1 public synthetic splits (SYNTHETIC)",
        "source_repository": "szl-holdings/szl-forge",
        "source_commit": FORGE_COMMIT,
        "source_path_convention": (
            PKG_PLACEHOLDER
            + " is the v1 gateway package directory of szl-forge at source_commit; its legacy "
            "name is not repeated in v2 files. Resolve any path with "
            "`git -C src/szl-forge ls-tree -r <source_commit> | grep <git_blob_sha1>`."
        ),
        "files": files,
        "dataset_receipt": {
            "source_path": f"{PKG_PLACEHOLDER}/{receipt_rel}",
            "git_blob_sha1": _blob_id(receipt_rel),
            "sha256": receipt_sha,
            "seed": receipt["seed"],
            "split_rows": receipt["split_rows"],
            "generator_sha256": receipt["generator_sha256"],
        },
        "v1_hub_model_revision": V1_HUB_REVISION,
        "v1_hub_receipt_dataset_receipt_sha256_matches": hub_receipt.get("dataset_receipt_sha256")
        == receipt_sha,
        "license": "Apache-2.0",
        "license_note": "same organization (SZL Holdings); upstream LICENSE is Apache License 2.0",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorded_by": "v2/research/legacy_v1_source.py",
        "truth_label": "MEASURED",
    }


def verify() -> list[str]:
    record = json.loads((LEGACY_DIR / "SOURCE.json").read_text("utf-8"))
    problems = []
    for name, info in record["files"].items():
        digest = sha256_bytes((LEGACY_DIR / name).read_bytes())
        if digest != info["sha256"]:
            problems.append(f"{name}: sha256 {digest} != {info['sha256']}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        record = build_source_record()
        text = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        (LEGACY_DIR / "SOURCE.json").write_bytes(text.encode("utf-8"))
        print(text, end="")
        return 0
    problems = verify()
    print(json.dumps({"ok": not problems, "problems": problems}, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
