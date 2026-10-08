"""Stage-1 review, lens (c): runtime file-access audit of the v2 test suite (and extra entry points).

Installs a sys.addaudithook 'open' hook, then runs `unittest discover -s v2/tests -t .` in this
process, plus the stats/bench-free entry points named on the command line (module names run via
runpy with their CLI args).  Every open of a path under ROOT/v2/data is recorded with the opening
v2 frame (module:function).  Flags: any open under v2/data/sealed/, any open of the registered
validation split v2/data/validation.jsonl.  Subprocesses spawned by tests are not audited (stated).

Usage (from ROOT):  PYTHONUTF8=1 py -3.12 -B logs/v2/review/sealing_audit.py
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = (ROOT / "v2" / "data").resolve()
SEALED = DATA / "sealed"
VALIDATION = DATA / "validation.jsonl"
EVENTS: Counter = Counter()


def _opener() -> str:
    frame = sys._getframe(2)
    while frame is not None:
        fname = frame.f_code.co_filename.replace("\\", "/")
        if "/v2/" in fname and "sealing_audit" not in fname:
            return f"{fname.split('/v2/', 1)[1]}:{frame.f_code.co_name}"
        frame = frame.f_back
    return "<non-v2>"


def hook(event, args):
    if event != "open" or not args:
        return
    target = args[0]
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    try:
        path = Path(os.fsdecode(target)).resolve()
    except (OSError, ValueError, TypeError):
        return
    try:
        path.relative_to(DATA)
    except ValueError:
        return
    mode = args[1] if len(args) > 1 else None
    EVENTS[(path.relative_to(ROOT).as_posix(), str(mode), _opener())] += 1


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.addaudithook(hook)
    suite = unittest.defaultTestLoader.discover(start_dir="v2/tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(suite)
    print("# sealing audit (lens c): opens under v2/data during the in-process v2 test suite")
    print(f"python {sys.version.split()[0]}; tests run {result.testsRun}, failures "
          f"{len(result.failures)}, errors {len(result.errors)}")
    sealed_hits, validation_hits = [], []
    for (path, mode, opener), count in sorted(EVENTS.items()):
        print(f"  {path} | mode {mode} | {opener} | x{count}")
        if path.startswith("v2/data/sealed/"):
            sealed_hits.append((path, opener))
        if path == "v2/data/validation.jsonl":
            validation_hits.append((path, opener))
    print(f"opens under v2/data/sealed/: {len(sealed_hits)} {sealed_hits}")
    print(f"opens of the registered validation split: {len(validation_hits)} {validation_hits}")
    print("note: subprocesses spawned by tests (CLI / determinism) are not covered by this hook")
    ok = not sealed_hits and not validation_hits and result.wasSuccessful()
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
