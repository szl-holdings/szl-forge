#!/usr/bin/env python3
"""Verify an exact public SZL Kernel revision without publisher credentials."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from publish_szl_kernels import canonical_json, verify_stable_kernel_runtime

FAILURE_DETAIL_LIMIT = 2000


def _write_evidence(output: Path | None, evidence: dict[str, object]) -> None:
    serialized = canonical_json(evidence)
    if output is None:
        print(serialized, end="")
    else:
        output.write_text(serialized, encoding="utf-8")


def _bounded_printable(value: object, *, limit: int) -> str:
    try:
        rendered = str(value)
    except BaseException:
        rendered = "<unprintable>"
    return "".join(
        character if 32 <= ord(character) <= 126 else "?"
        for character in rendered
    )[:limit]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        evidence = verify_stable_kernel_runtime(revision=args.revision)
    except (Exception, SystemExit) as exc:
        _write_evidence(
            args.output,
            {
                "status": "FAILED",
                "error_type": _bounded_printable(type(exc).__name__, limit=128),
                "error": _bounded_printable(exc, limit=FAILURE_DETAIL_LIMIT),
            },
        )
        return 1
    _write_evidence(args.output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
