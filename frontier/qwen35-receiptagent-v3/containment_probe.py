#!/usr/bin/env python3
"""Prove the exact training worker namespace hides non-training host data."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence


EXPECTED_INPUT_FILES = {
    "candidate.json",
    "containment_probe.py",
    "curriculum-manifest.json",
    "train.jsonl",
    "train_candidate.py",
    "training-bundle.json",
}
HEX_40 = re.compile(r"[0-9a-f]{40}")
RUN_ID = re.compile(r"[0-9a-f]{32}")
EXPECTED_UNREADABLE_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.ENOENT,
        errno.ENOTDIR,
    }
)
EXPECTED_UNWRITABLE_ERRNOS = frozenset(
    {
        *EXPECTED_UNREADABLE_ERRNOS,
        errno.EROFS,
    }
)


class ProbeError(RuntimeError):
    """The effective worker mount namespace did not meet the fixed contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def assert_readable(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.read(descriptor, 1)
    finally:
        os.close(descriptor)


def assert_unreadable(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        if exc.errno in EXPECTED_UNREADABLE_ERRNOS:
            return
        raise ProbeError("forbidden read check failed unexpectedly") from exc
    os.close(descriptor)
    raise ProbeError("a forbidden host path remained readable")


def assert_unwritable(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        if exc.errno in EXPECTED_UNWRITABLE_ERRNOS:
            return
        raise ProbeError("forbidden write check failed unexpectedly") from exc
    os.close(descriptor)
    raise ProbeError("a read-only or inaccessible path remained writable")


def publish_report(path: Path, report: dict[str, object]) -> None:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    report["reportSha256"] = hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    data = (canonical_json(report) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("containment report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def perform_probe(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_dir.resolve(strict=True)
    cache_dir = args.cache_dir.resolve(strict=True)
    venv_dir = args.venv_dir.resolve(strict=True)
    model_repository = args.model_repository.resolve(strict=True)
    if {path.name for path in input_dir.iterdir()} != EXPECTED_INPUT_FILES:
        raise ProbeError("training input mount contains missing or unapproved files")
    for filename in (
        "candidate.json",
        "curriculum-manifest.json",
        "train.jsonl",
        "train_candidate.py",
        "training-bundle.json",
    ):
        assert_readable(input_dir / filename)
    if not (venv_dir / "bin" / "python").exists():
        raise ProbeError("exact runtime venv is unavailable")
    if not HEX_40.fullmatch(args.model_revision):
        raise ProbeError("model revision is malformed")
    if not (model_repository / "snapshots" / args.model_revision).is_dir():
        raise ProbeError("exact model snapshot is unavailable")
    for forbidden in args.forbidden_read:
        assert_unreadable(forbidden)
    for forbidden in (
        input_dir / ".probe-write",
        venv_dir / ".probe-write",
        model_repository / ".probe-write",
        Path("/etc/.szl-ra3-probe-write"),
        Path("/mnt/c/.szl-ra3-probe-write"),
    ):
        assert_unwritable(forbidden)
    if args.report.parent.resolve(strict=True) != cache_dir:
        raise ProbeError("containment report is outside the writable cache mount")
    return {
        "schema": "szl.receiptagent-v3-containment-probe/v1",
        "state": "PASS",
        "trainingOnlyInputSetExact": True,
        "heldOutContentAbsent": True,
        "forbiddenHostReadsFailed": True,
        "nonOutputWritesFailed": True,
        "runtimeAndModelInputsReadable": True,
        "secretContentRead": False,
    }


def fixed_trainer_argv(args: argparse.Namespace) -> list[str]:
    if not HEX_40.fullmatch(args.source_commit):
        raise ProbeError("source commit is malformed")
    if not RUN_ID.fullmatch(args.supervisor_run_id):
        raise ProbeError("supervisor run ID is malformed")
    input_dir = args.input_dir.resolve(strict=True)
    output_dir = (input_dir.parent / "output").resolve(strict=True)
    trainer = input_dir / "train_candidate.py"
    if not trainer.is_file() or not output_dir.is_dir():
        raise ProbeError("fixed trainer inputs are unavailable")
    return [
        sys.executable,
        "-I",
        "-B",
        str(trainer),
        "--source-commit",
        args.source_commit,
        "--run-kind",
        args.run_kind,
        "--output-dir",
        str(output_dir),
        "--source-bundle-dir",
        str(input_dir),
        "--supervisor-run-id",
        args.supervisor_run_id,
    ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--venv-dir", type=Path, required=True)
    result.add_argument("--model-repository", type=Path, required=True)
    result.add_argument("--model-revision", required=True)
    result.add_argument("--forbidden-read", type=Path, action="append", required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    result.add_argument("--supervisor-run-id", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        worker_argv = fixed_trainer_argv(args)
        report = perform_probe(args)
        report["trainerExecBound"] = True
        publish_report(args.report, report)
        print(canonical_json(report), flush=True)
        os.execv(worker_argv[0], worker_argv)
    except Exception as exc:  # noqa: BLE001 - fail closed without path details
        print(
            canonical_json(
                {
                    "schema": "szl.receiptagent-v3-containment-probe/v1",
                    "state": "FAIL",
                    "errorType": type(exc).__name__,
                    "secretContentRead": False,
                }
            )
        )
        return 1
    raise AssertionError("successful worker exec does not return")


if __name__ == "__main__":
    sys.exit(main())
