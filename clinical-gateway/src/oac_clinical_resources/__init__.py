"""Explicit, non-overwriting preparation of a synthetic-only local workspace.

The installed wheel and its generated integrity module are the trust anchor.
Hashes detect bundle corruption; they do not authenticate a maliciously replaced
Python installation. Parent directories must remain trusted and non-shared.
Preparation never starts a listener, initializes a key, or connects a device.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence


ASSET_PATHS = (
    "clinical-gateway/fixtures/assay_map.json",
    "clinical-gateway/frontend/index.html",
    "clinical-gateway/operational-model/artifacts/model-receipt.json",
    "clinical-gateway/operational-model/artifacts/model.json",
    "clinical-gateway/operational-model/example-input.json",
)
MANIFEST_SCHEMA = "szl-oac/portable-workspace-assets/v1"
PACKAGE_VERSION = "2.5.1"
MAX_ASSET_BYTES = 1024 * 1024


class WorkspacePreparationError(ValueError):
    """The asset or destination contract was not satisfied."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkspacePreparationError("duplicate JSON member in portable manifest")
        result[key] = value
    return result


def _verified_assets(bundle: Any, expected_manifest_sha256: str) -> tuple[bytes, dict[str, bytes]]:
    """Read every closed-set byte before creating the destination directory."""
    try:
        with bundle.joinpath("manifest.json").open("rb") as handle:
            encoded = handle.read(16385)
        if len(encoded) > 16384 or hashlib.sha256(encoded).hexdigest() != expected_manifest_sha256:
            raise WorkspacePreparationError("portable asset manifest integrity mismatch")
        manifest = json.loads(encoded, object_pairs_hook=_unique_object)
        if not isinstance(manifest, dict) or set(manifest) != {"schema", "package_version", "files"}:
            raise WorkspacePreparationError("portable asset manifest fields mismatch")
        if manifest["schema"] != MANIFEST_SCHEMA or manifest["package_version"] != PACKAGE_VERSION:
            raise WorkspacePreparationError("portable asset manifest identity mismatch")
        entries = manifest["files"]
        if not isinstance(entries, list) or len(entries) != len(ASSET_PATHS):
            raise WorkspacePreparationError("portable asset manifest must use the closed file set")
        payloads: dict[str, bytes] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
                raise WorkspacePreparationError("portable asset entry fields mismatch")
            relative = entry["path"]
            if not isinstance(relative, str) or relative not in ASSET_PATHS or relative in payloads:
                raise WorkspacePreparationError("portable asset path is undeclared or duplicated")
            size = entry["size_bytes"]
            if type(size) is not int or not 0 < size <= MAX_ASSET_BYTES:
                raise WorkspacePreparationError("portable asset size is invalid")
            digest = entry["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise WorkspacePreparationError("portable asset digest is invalid")
            with bundle.joinpath(*relative.split("/")).open("rb") as handle:
                payload = handle.read(MAX_ASSET_BYTES + 1)
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                raise WorkspacePreparationError(f"portable asset integrity mismatch: {relative}")
            payloads[relative] = payload
        return encoded, payloads
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspacePreparationError("portable asset bundle is unreadable") from exc


def load_verified_assets() -> tuple[bytes, dict[str, bytes]]:
    try:
        from ._asset_integrity import MANIFEST_SHA256
    except ImportError as exc:
        raise WorkspacePreparationError("portable assets require a built and installed wheel") from exc
    return _verified_assets(resources.files(__package__).joinpath("assets"), MANIFEST_SHA256)


def _reject_link(path: Path) -> os.stat_result:
    observed = path.lstat()
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & reparse:
        raise WorkspacePreparationError("workspace path must not contain symlinks or reparse points")
    return observed


def _destination(value: Path | str) -> Path:
    raw = os.fspath(value)
    if not raw or any(part in {".", ".."} for part in raw.replace("\\", "/").split("/")):
        raise WorkspacePreparationError("workspace directory must be a new explicit path without dot traversal")
    target = Path(os.path.abspath(raw))
    if target == Path(target.anchor):
        raise WorkspacePreparationError("workspace directory must not be a filesystem root")
    if os.name == "nt":
        from owned_agent_clinical_control import windows_fixed_drive

        if target.drive.startswith("\\\\") or len(target.drive) != 2:
            raise WorkspacePreparationError("workspace must use a local drive, not a UNC or device path")
        if not windows_fixed_drive(target):
            raise WorkspacePreparationError("workspace must use a fixed local drive, not a mapped network or removable drive")
        if any(part.rstrip(" .") != part or ":" in part or Path(part).is_reserved()
               for part in target.parts[1:]):
            raise WorkspacePreparationError("workspace directory name is invalid")
    # Validate the unreconciled path first: resolve() must not erase link evidence.
    for ancestor in reversed(target.parents):
        observed = _reject_link(ancestor)
        if not stat.S_ISDIR(observed.st_mode):
            raise WorkspacePreparationError("workspace parent must be an existing directory")
    if os.path.lexists(target):
        raise WorkspacePreparationError("workspace directory already exists; nothing was overwritten")
    if target.parent.resolve(strict=True) != target.parent:
        raise WorkspacePreparationError("workspace parent has an unexpected resolved identity")
    return target


def prepare_workspace(directory: Path | str) -> dict[str, Any]:
    manifest, payloads = load_verified_assets()
    try:
        target = _destination(directory)
        # mkdir is the exclusive claim: even an empty existing directory is refused.
        target.mkdir(mode=0o700, exist_ok=False)
        _reject_link(target)
        for relative, payload in payloads.items():
            destination = target.joinpath(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            for ancestor in (destination.parent, *destination.parent.parents):
                _reject_link(ancestor)
                if ancestor == target:
                    break
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(destination, flags, 0o600), "wb") as handle:
                handle.write(payload)
        # This receipt is emitted last; it is not a clinical or release receipt.
        with (target / "portable-assets.json").open("xb") as handle:
            handle.write(manifest)
    except (OSError, WorkspacePreparationError) as exc:
        # Never recursively remove a path after an ambiguous filesystem failure.
        raise WorkspacePreparationError(
            f"workspace preparation failed; any partial new directory is retained: {exc}"
        ) from exc
    return {"ok": True, "state": "PREPARED_SYNTHETIC_SHADOW_WORKSPACE", "data_root": str(target),
            "files": list(ASSET_PATHS), "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "clinical_use_authorized": False, "real_phi_authorized": False, "site_validated": False,
            "device_connection_started": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a NEW synthetic-only workspace from installed wheel assets.")
    parser.add_argument("--directory", required=True, help="New directory under an existing trusted parent; never overwritten")
    args = parser.parse_args(argv)
    try:
        result = prepare_workspace(args.directory)
    except WorkspacePreparationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0
