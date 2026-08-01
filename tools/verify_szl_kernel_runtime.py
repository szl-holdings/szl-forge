#!/usr/bin/env python3
"""Verify an exact public SZL Kernel revision without publisher credentials."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

FAILURE_DETAIL_LIMIT = 2000


def _write_evidence(output: Path | None, evidence: dict[str, object]) -> None:
    serialized = json.dumps(
        evidence,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    if output is None:
        print(serialized, end="")
    else:
        output.write_text(serialized, encoding="utf-8")


def verify_stable_kernel_runtime(*, revision: str) -> dict[str, object]:
    """Import the trusted verifier lazily so bootstrap errors become evidence."""
    from publish_szl_kernels import verify_stable_kernel_runtime as verify

    return verify(revision=revision)


def _bounded_printable(value: object, *, limit: int) -> str:
    try:
        rendered = str(value)
        return "".join(
            character if 32 <= ord(character) <= 126 else "?"
            for character in rendered
        )[:limit]
    except BaseException:
        return "<unprintable>"


def _bounded_error_type(value: object, *, limit: int) -> str:
    try:
        rendered = str(value)
        normalized = "".join(
            character
            if character.isascii()
            and (character.isalnum() or character in "_.")
            else "_"
            for character in rendered
        )
    except BaseException:
        return "UnprintableError"
    if not normalized:
        return "UnprintableError"
    if not (normalized[0].isalpha() or normalized[0] == "_"):
        normalized = f"_{normalized}"
    return normalized[:limit]


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
                "error_type": _bounded_error_type(type(exc).__name__, limit=128),
                "error": _bounded_printable(exc, limit=FAILURE_DETAIL_LIMIT),
            },
        )
        return 1
    _write_evidence(args.output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
