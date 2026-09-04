#!/usr/bin/env python3
"""Load a Python verifier completely before invoking its ``main`` function.

This prevents script-mode call-order bugs when a verifier's ``if __name__ ==
'__main__'`` guard appears before helper definitions. The target is loaded with
a non-main run name, so every definition is installed before ``main(argv)`` is
called. The wrapper does not alter the verifier's network, hashing, or failure
semantics.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class VerifierEntrypointError(RuntimeError):
    """The verifier file could not be loaded as a callable program."""


def load_main(script: Path) -> Callable[[list[str] | None], int]:
    script = Path(script).resolve()
    if not script.is_file():
        raise VerifierEntrypointError(f"verifier script is missing: {script}")
    namespace: dict[str, Any] = runpy.run_path(
        str(script),
        run_name="szl_governed_live_verifier_loaded",
    )
    entrypoint = namespace.get("main")
    if not callable(entrypoint):
        raise VerifierEntrypointError("verifier does not expose callable main(argv)")
    return entrypoint


def invoke(script: Path, argv: Sequence[str]) -> int:
    result = load_main(script)(list(argv))
    if isinstance(result, bool) or not isinstance(result, int):
        raise VerifierEntrypointError("verifier main(argv) must return an integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    args, forwarded = parser.parse_known_args(argv)
    return invoke(args.script, forwarded)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
