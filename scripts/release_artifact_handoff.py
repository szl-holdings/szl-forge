#!/usr/bin/env python3
"""Bind small release evidence artifacts to one source/run/attempt, not approval.

No network, model loading, signing or publishing. GitHub's immutable producer
artifact ID is selected separately by the workflow. This unsigned envelope
detects missing, stale, swapped or changed transport bytes; it is NOT a trusted
signature, model evaluation, or authorization to publish. A compromised runner
or a malicious producer can forge an envelope and is outside this boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

CHECKOUT = Path(__file__).resolve().parents[1]
RECEIPT = "handoff.json"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024
PROFILES = {
    "heldout-eval": ("heldout-eval.json",),
    "model-bom": ("model-bom.cdx.json",),
    "signed-release": (
        "run-manifest.json", "run-manifest.sig", "run-manifest.crt",
        "weights.sha256", "weights.sha256.sig", "weights.sha256.crt",
        "model-bom.cdx.json.sig", "model-bom.cdx.json.crt",
    ),
}


class HandoffError(ValueError):
    """A fail-closed transport boundary failure; never includes input bytes."""


def validate_artifact_id(value: str) -> str:
    # A missing artifact-ids input makes download-artifact select ALL artifacts.
    if type(value) is not str or re.fullmatch(r"[1-9][0-9]{0,19}", value) is None:
        raise HandoffError("POSITIVE_DECIMAL_ID_REQUIRED")
    return value


def make_context(repository: str, git_sha: str, run_id: str, run_attempt: str) -> dict:
    if repository != "szl-holdings/szl-forge":
        raise HandoffError("CANONICAL_REPOSITORY_REQUIRED")
    if type(git_sha) is not str or re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise HandoffError("EXACT_SOURCE_SHA_REQUIRED")
    return {
        "repository": repository,
        "git_sha": git_sha,
        "run_id": validate_artifact_id(run_id),
        "run_attempt": validate_artifact_id(run_attempt),
        "workflow": "model-publish-gate",
    }


def checked_context(context: dict) -> dict:
    if type(context) is not dict or set(context) != {
        "repository", "git_sha", "run_id", "run_attempt", "workflow",
    }:
        raise HandoffError("INVALID_CONTEXT")
    expected = make_context(context["repository"], context["git_sha"],
                            context["run_id"], context["run_attempt"])
    if context != expected:
        raise HandoffError("INVALID_CONTEXT")
    return expected


def no_links(path: Path) -> None:
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise HandoffError("LINK_OR_REPARSE_POINT_REJECTED")


def stage_path(stage: Path) -> Path:
    stage = Path(stage)
    if not stage.is_absolute():
        raise HandoffError("ABSOLUTE_STAGE_REQUIRED")
    no_links(stage)
    stage = stage.resolve()
    if stage.is_relative_to(CHECKOUT.resolve()):
        raise HandoffError("STAGE_MUST_BE_OUTSIDE_CHECKOUT")
    return stage


def initialize(stage: Path) -> None:
    stage = stage_path(stage)
    # Do not remove or reuse an old stage, including an empty one.
    try:
        stage.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise HandoffError("FRESH_STAGE_REQUIRED") from exc


def read_regular(path: Path, limit: int) -> bytes:
    no_links(path)
    try:
        # Reject FIFOs/devices before open, which could otherwise block forever.
        if not stat.S_ISREG(path.lstat().st_mode):
            raise HandoffError("REGULAR_FILE_REQUIRED")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise HandoffError("NONEMPTY_BOUNDED_REGULAR_FILE_REQUIRED")
            raw = stream.read(limit + 1)
        if not 0 < len(raw) <= limit or len(raw) != info.st_size:
            raise HandoffError("FILE_SIZE_CHANGED_OR_OUT_OF_BOUNDS")
        return raw
    except OSError as exc:
        raise HandoffError("UNREADABLE_FILE") from exc


def inventory(stage: Path, profile: str, *, sealed: bool) -> dict:
    if type(profile) is not str or profile not in PROFILES:
        raise HandoffError("UNKNOWN_PROFILE")
    stage = stage_path(stage)
    try:
        names = {p.name for p in stage.iterdir()}
    except OSError as exc:
        raise HandoffError("STAGE_UNAVAILABLE") from exc
    expected = set(PROFILES[profile]) | ({RECEIPT} if sealed else set())
    if names != expected:
        raise HandoffError("EXACT_FILE_SET_REQUIRED")
    result = {}
    for name in PROFILES[profile]:
        raw = read_regular(stage / name, MAX_FILE_BYTES)
        result[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    return result


def envelope(profile: str, context: dict, files: dict) -> dict:
    return {
        "schema": "szl.release-artifact-handoff/v1",
        "context": checked_context(context),
        "profile": profile,
        "evidence_class": "MEASURED",
        "signature_verification": "UNAVAILABLE",
        "publication_eligible": False,
        "files": files,
    }


def seal(stage: Path, profile: str, context: dict) -> dict:
    expected = envelope(profile, context, inventory(stage, profile, sealed=False))
    raw = (json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > MAX_RECEIPT_BYTES:
        raise HandoffError("RECEIPT_SIZE_EXCEEDED")
    try:
        with (stage_path(stage) / RECEIPT).open("xb") as stream:
            stream.write(raw)
    except OSError as exc:
        raise HandoffError("NEW_RECEIPT_REQUIRED") from exc
    return expected


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_number(_):
        raise ValueError("non-integer number")

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=reject_number, parse_float=reject_number)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise HandoffError("INVALID_RECEIPT_JSON") from exc
    if type(result) is not dict:
        raise HandoffError("RECEIPT_OBJECT_REQUIRED")
    return result


def verify(stage: Path, profile: str, context: dict) -> dict:
    expected = envelope(profile, context, inventory(stage, profile, sealed=True))
    actual = strict_json(read_regular(stage_path(stage) / RECEIPT, MAX_RECEIPT_BYTES))
    files = actual.get("files")
    if type(files) is not dict or set(files) != set(expected["files"]):
        raise HandoffError("INVALID_FILE_BINDINGS")
    for record in files.values():
        if type(record) is not dict or set(record) != {"sha256", "size_bytes"}:
            raise HandoffError("INVALID_FILE_BINDINGS")
        if type(record["size_bytes"]) is not int or type(record["sha256"]) is not str:
            raise HandoffError("INVALID_FILE_BINDING_TYPES")
    # bool/int equality must never promote an authorization field.
    if actual.get("publication_eligible") is not False or actual != expected:
        raise HandoffError("HANDOFF_BINDING_MISMATCH")
    return expected


def github_context() -> dict:
    context = make_context(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_SHA"),
                           os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_RUN_ATTEMPT"))
    try:
        head = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=CHECKOUT,
                                       text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise HandoffError("CHECKOUT_UNAVAILABLE") from exc
    if head != context["git_sha"]:
        raise HandoffError("CHECKOUT_REVISION_MISMATCH")
    return context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifact = commands.add_parser("check-artifact-id")
    artifact.add_argument("--artifact-id", required=True)
    for name in ("init", "seal", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--stage", required=True, type=Path)
        if name != "init":
            command.add_argument("--profile", required=True, choices=tuple(PROFILES))
    args = parser.parse_args(argv)
    try:
        if args.command == "check-artifact-id":
            validate_artifact_id(args.artifact_id)
        else:
            context = github_context()
            if args.command == "init":
                initialize(args.stage)
            elif args.command == "seal":
                seal(args.stage, args.profile, context)
            else:
                verify(args.stage, args.profile, context)
    except (HandoffError, OSError) as exc:
        code = str(exc) if isinstance(exc, HandoffError) else "FILESYSTEM_ERROR"
        print(f"::error::release-artifact-handoff: {code}", file=sys.stderr)
        return 1
    print(f"release-artifact-handoff {args.command}: OK (transport only; no publication authorization)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
