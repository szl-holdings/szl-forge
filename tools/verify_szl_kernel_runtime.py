#!/usr/bin/env python3
"""Verify an exact public SZL Kernel revision without publisher credentials."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from publish_szl_kernels import canonical_json, verify_stable_kernel_runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    evidence = verify_stable_kernel_runtime(revision=args.revision)
    print(canonical_json(evidence), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
