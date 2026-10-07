"""Run the v2 unit suite and fail unless the only skips are the declared origin-layer skips.

SYNTHETIC; DEV data only (the suite never reads a registered split or a sealed file).

The frozen v2 receipt names source commit 92872f88 of the unpublished research repository.  Where
that commit is absent (any szl-forge checkout), exactly seven tests cannot run their origin layer
and skip with ``v2.tests._support.ORIGIN_SKIP_REASON``.  This runner accepts that and nothing
else: the suite must be successful (no failure, no error, no unexpected success), every skip must
be one of the seven declared origin skips with that exact reason, and all seven must be present.
Where the commit is present (the research repository), no test may skip.

Usage (from the szl-forge repository root):
    python -B ops-health/tools/run_v2_tests.py [--report PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path

OPS_HEALTH = Path(__file__).resolve().parents[1]

ORIGIN_SKIPS = (
    "setUpClass (v2.tests.test_final_analyze.FinalAnalyzeDevTest)",
    "setUpClass (v2.tests.test_final_score.FinalScoreDryRunTest)",
    "v2.tests.test_mutation_suite.CatalogueTest.test_receipt_catalogue",
    "v2.tests.test_mutation_suite.IntegrityLayerTest.test_frozen_v2_pair_passes_the_integrity_layer",
    "v2.tests.test_mutation_suite.IntegrityLayerTest.test_registered_receipt_mutants_are_caught",
    "v2.tests.test_mutation_suite.IntegrityLayerTest.test_artifact_mutant_pass1_refused_pass2_loads",
    "v2.tests.test_mutation_suite.BehaviorAndBaselineTest."
    "test_broken_baselines_are_invalid_without_per_class_rates (fixture='constant_alert')",
)


def _test_id(test) -> str:
    # class-level skips are reported through an _ErrorHolder whose id() is its description
    return test.id()


def _readiness_differences(limit: int = 64) -> dict:
    """Diagnose an exact recorded-readiness failure without changing the test verdict."""
    from v2.research import mutation  # noqa: PLC0415

    recorded = json.loads(mutation.RESULTS_PATH.read_text(encoding="utf-8"))
    good = {fid: fx["gates"] for fid, fx in recorded["known_good"].items()}
    broken = {m["id"]: m["pass2_behavior"]["gates"] for m in recorded["artifact_mutants"]}
    actual = mutation.readiness(good, broken)
    expected = recorded["readiness"]
    items: list[dict] = []
    count = 0

    def show(value):
        if isinstance(value, float):
            return {"decimal": repr(value), "hex": value.hex()}
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, str) and value in ("KEEP", "REJECTED"):
            return value
        if isinstance(value, str):
            return {"type": "str", "chars": len(value),
                    "sha256_prefix": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}
        return {"type": type(value).__name__}

    def note(path: str, recorded_value, recomputed_value) -> None:
        nonlocal count
        count += 1
        if len(items) < limit:
            items.append({"path": path[:160], "recorded": show(recorded_value),
                          "recomputed": show(recomputed_value)})

    def walk(path: str, left, right) -> None:
        if type(left) is not type(right):
            note(path, left, right)
        elif isinstance(left, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{str(key)[:64]}"
                if key not in left or key not in right:
                    note(child, left.get(key, "<missing>"), right.get(key, "<missing>"))
                else:
                    walk(child, left[key], right[key])
        elif isinstance(left, list):
            if len(left) != len(right):
                note(f"{path}.length", len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                walk(f"{path}[{index}]", a, b)
        elif left != right:
            note(path, left, right)

    walk("readiness", expected, actual)
    return {"difference_count": count, "shown": items, "limit": limit,
            "truncated": count > limit}


def run(root: Path) -> dict:
    os.chdir(root)
    sys.path.insert(0, str(root))
    from v2.research import registry  # noqa: E402
    from v2.tests import _support  # noqa: E402

    if registry.ROOT.resolve() != root.resolve():
        raise SystemExit(f"imported v2 from {registry.ROOT}, expected {root}; refusing")
    origin = _support.origin_commit_present()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover("v2/tests", top_level_dir=".")
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    skips = sorted((_test_id(t), reason) for t, reason in result.skipped)
    expected = [] if origin else sorted((name, _support.ORIGIN_SKIP_REASON) for name in ORIGIN_SKIPS)
    problems = []
    if not result.wasSuccessful():
        problems.append(f"suite failed: {len(result.failures)} failures, {len(result.errors)} errors, "
                        f"{len(result.unexpectedSuccesses)} unexpected successes")
    if skips != expected:
        unexpected = [s for s in skips if s not in expected]
        missing = [s for s in expected if s not in skips]
        problems.append(f"skips differ from the declared origin skips: unexpected {unexpected}, "
                        f"missing {missing}")
    report = {
        "schema": "szl-oac/ops-health-v2-tests/v1",
        "ok": not problems,
        "problems": problems,
        "origin_commit_present": origin,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": [{"test": name, "reason": reason} for name, reason in skips],
        "python": sys.version.split()[0],
        "seconds": round(time.perf_counter() - started, 3),
    }
    if any(_test_id(test).endswith("RecordedRunTest.test_aggregation_recomputes")
           for test, _traceback in result.failures):
        try:
            report["readiness_diagnostics"] = _readiness_differences()
        except Exception as exc:  # diagnosis must not mask the original failing test
            report["readiness_diagnostics"] = {"error_type": type(exc).__name__}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=OPS_HEALTH,
                        help="the directory holding v2/ (default: this tool's ops-health/)")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report_path = args.report.resolve() if args.report is not None else None
    report = run(args.root.resolve())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(text.encode("utf-8"))
    sys.stdout.write(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
