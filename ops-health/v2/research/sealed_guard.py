"""Single gate for opening registered splits (PREREGISTRATION sections 3, 4, 7).

Rules enforced here:
  * Every split's bytes are sha256-verified against v2/data/MANIFEST.json before parsing.
  * Open splits need a registered purpose:
        train       -> TRAIN, ABLATION
        calibration -> CALIBRATION, ABLATION
        conformal   -> CONFORMAL, ABLATION
        validation  -> REGISTERED_SEARCH, ABLATION, GATES
  * Sealed splits (test and the four shift splits) are refused for every purpose except
    FINAL_TEST_OPENING.  The final opening is atomic over the whole sealed set: it requires
    runs/TEST_OPENED.json to be absent, verifies every sealed file's sha256, hashes the frozen
    contenders, creates runs/TEST_OPENED.json exclusively (utc, git HEAD, contenders and their
    sha256s, sealed split digests), and only then parses and returns all sealed splits as a
    dict {split: rows}.  Any later opening is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from . import registry
from .dataio import parse_jsonl_bytes

FINAL_TEST_OPENING = "FINAL_TEST_OPENING"
OPEN_PURPOSES = {
    "train": frozenset({"TRAIN", "ABLATION"}),
    "calibration": frozenset({"CALIBRATION", "ABLATION"}),
    "conformal": frozenset({"CONFORMAL", "ABLATION"}),
    "validation": frozenset({"REGISTERED_SEARCH", "ABLATION", "GATES"}),
}
SEALED_SET_NAME = "sealed"


class SealedAccessError(PermissionError):
    """Refusal to open a split (wrong purpose, second opening, or integrity failure)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SealedGuard:
    def __init__(
        self,
        data_dir: Path = registry.DATA_DIR,
        runs_dir: Path = registry.RUNS_DIR,
        repo_root: Path = registry.ROOT,
    ):
        self.data_dir = Path(data_dir)
        self.runs_dir = Path(runs_dir)
        self.repo_root = Path(repo_root)

    # -- manifest and hashing -------------------------------------------------------------
    def manifest(self) -> dict:
        path = self.data_dir / "MANIFEST.json"
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise SealedAccessError(f"unable to read data manifest: {exc}") from exc

    def _verified_bytes(self, name: str, manifest: dict) -> bytes:
        entry = manifest.get("splits", {}).get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise SealedAccessError(f"split {name!r} is not in the data manifest")
        data = registry.split_path(name, self.data_dir).read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise SealedAccessError(f"split {name!r} sha256 does not match MANIFEST; refusing")
        return data

    # -- public API ------------------------------------------------------------------------
    def open_split(
        self,
        name: str,
        purpose: str,
        *,
        contenders: Mapping[str, Sequence[Path | str]] | None = None,
    ):
        if name not in registry.SPLIT_TABLE and name != SEALED_SET_NAME:
            raise SealedAccessError(f"unknown split {name!r}")
        if name == SEALED_SET_NAME or name in registry.SEALED_SPLITS:
            if purpose != FINAL_TEST_OPENING:
                raise SealedAccessError(
                    f"split {name!r} is sealed; only purpose {FINAL_TEST_OPENING} may open it"
                )
            return self._final_opening(contenders)
        allowed = OPEN_PURPOSES[name]
        if purpose not in allowed:
            raise SealedAccessError(
                f"purpose {purpose!r} is not registered for split {name!r} (allowed: {sorted(allowed)})"
            )
        return parse_jsonl_bytes(self._verified_bytes(name, self.manifest()))

    def _git_head(self) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def _final_opening(self, contenders) -> dict[str, list[dict]]:
        marker = self.runs_dir / registry.TEST_OPENED_NAME
        if marker.exists():
            raise SealedAccessError(
                f"the sealed set was already opened ({marker}); a second opening is refused"
            )
        if not contenders:
            raise SealedAccessError("the final opening requires the frozen contenders and their files")
        frozen = {}
        for contender, files in sorted(contenders.items()):
            if not files:
                raise SealedAccessError(f"contender {contender!r} lists no frozen files")
            frozen[contender] = {}
            for file in files:
                path = Path(file)
                try:
                    frozen[contender][path.as_posix()] = _sha256(path.read_bytes())
                except OSError as exc:
                    raise SealedAccessError(f"contender file unreadable: {exc}") from exc
        manifest = self.manifest()
        verified = {name: self._verified_bytes(name, manifest) for name in registry.SEALED_SPLITS}
        record = {
            "schema": "szl-oac/ops-health-test-opening/v2",
            "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": self._git_head(),
            "purpose": FINAL_TEST_OPENING,
            "contenders": frozen,
            "sealed_splits_sha256": {n: _sha256(b) for n, b in verified.items()},
            "data_manifest_sha256": _sha256((self.data_dir / "MANIFEST.json").read_bytes()),
            "rule": "single opening; any later opening is refused",
        }
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
        except FileExistsError as exc:
            raise SealedAccessError("the sealed set was opened concurrently; refusing") from exc
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        return {name: parse_jsonl_bytes(data) for name, data in verified.items()}


_DEFAULT = SealedGuard()


def open_split(name: str, purpose: str, **kwargs):
    """Module-level entry point bound to the repository's v2/data and runs/ directories."""
    return _DEFAULT.open_split(name, purpose, **kwargs)
