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
    return {
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
