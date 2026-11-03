"""Run the offline contracts and expected-refusal CLI sample in one CPU process."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import offline_ci_intake as intake  # noqa: E402

EXPECTED_SAMPLE_SHA256 = "068141c3d027b75f4ddd3c871180eda95111fd19f61f3e2a6635f919604dd351"


def run():
    report = {
        "schema": "szl.offline-ci-intake-checks/v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "LOCAL_SYNTHETIC_CONTRACTS_ONLY",
        "complete": False,
        "status": "FAIL",
        "execution_permission": False,
        "release_permission": False,
        "checks": [],
    }
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    tests_ok = result.testsRun >= 53 and result.wasSuccessful() and not result.skipped
    report["checks"].append({"name": "unit_contracts", "passed": tests_ok,
                             "tests_run": result.testsRun, "failures": len(result.failures),
                             "errors": len(result.errors), "skipped": len(result.skipped)})
    # A refusal is the expected sample result, so exit 2 does not skip later checks.
    try:
        output = io.StringIO()
        code = intake.main(io.StringIO((ROOT / "sample_input.json").read_text(encoding="utf-8")), output)
        raw = output.getvalue().encode("utf-8")
        saved = (ROOT / "sample_result.json").read_bytes()
        value = json.loads(raw)
        sample_ok = (code == 2 and raw == saved and
                     hashlib.sha256(saved).hexdigest() == EXPECTED_SAMPLE_SHA256 and
                     value["proposed_pilot_label"] == "REQUIRES_REVIEW" and
                     value["candidate_ci_layer"]["state"] == "FAIL" and
                     all(value[key] is False for key in
                         ("authorized", "execution_permission", "release_permission")))
        report["checks"].append({"name": "expected_refusal_sample", "passed": sample_ok,
                                 "cli_exit_code": code, "expected_cli_exit_code": 2,
                                 "output_sha256": hashlib.sha256(raw).hexdigest(),
                                 "saved_output_identical": raw == saved})
    except Exception as exc:
        sample_ok = False
        report["checks"].append({"name": "expected_refusal_sample", "passed": False,
                                 "error_type": type(exc).__name__})
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(ROOT.iterdir()) if p.suffix in (".py", ".json")}
    report["complete"] = tests_ok and sample_ok
    report["status"] = "PASS" if report["complete"] else "FAIL"
    sys.stderr.write(log.getvalue())
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    raise SystemExit(0 if report["complete"] else 1)
