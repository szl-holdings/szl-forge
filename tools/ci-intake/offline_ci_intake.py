"""Original synthetic CI intake prototype; not an SZL authority/readiness kernel.

Only standard-library parsing and deterministic in-memory normalization occur.
No network, subprocess, credentials, filesystem reads, signatures, or model calls.
The CLI reads JSON on stdin and emits JSON on stdout; policy must be caller-reviewed.
"""

import datetime as dt
import json
import re
import sys
from urllib.parse import urlsplit


STATUSES = ("passed", "failed", "absent", "skipped", "error", "xfail", "xpass")
MAX_JSON_CHARS = 1_048_576
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}\Z")
OBSERVER = "offline-ci-normalizer-prototype"


class InvalidInput(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise InvalidInput(message)


def _keys(value, required, optional=()):
    _require(isinstance(value, dict), "Expected an object.")
    _require(set(required) <= set(value), "Required fields are absent.")
    _require(set(value) <= set(required) | set(optional), "Unknown fields are not accepted.")


def _text(value, limit=1000):
    _require(isinstance(value, str) and 0 < len(value) <= limit,
             "Expected a bounded nonempty string.")
    _require(not any(ord(c) < 32 for c in value), "Control characters are not accepted.")
    return value


def _identifier(value):
    _require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
             "Invalid identifier.")
    return value


def _revision(value):
    _require(isinstance(value, str) and REVISION.fullmatch(value) is not None,
             "Expected a full lowercase 40- or 64-character hexadecimal revision.")
    return value


def _time(value):
    _require(isinstance(value, str) and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is not None,
        "Expected a UTC timestamp with second precision.")
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise InvalidInput("Invalid UTC timestamp.") from exc


def _integer(value, low, high):
    _require(type(value) is int and low <= value <= high, "Invalid integer bound.")


def _labels(value):
    _require(isinstance(value, list) and len(value) <= 100, "Expected a bounded label list.")
    for item in value:
        _identifier(item)
    _require(len(set(value)) == len(value), "Duplicate labels are not accepted.")


def _locator(value):
    _text(value, 2000)
    try:
        parsed = urlsplit(value)
        _require(parsed.scheme in {"synthetic", "https"} and bool(parsed.netloc),
                 "Expected a synthetic or HTTPS artifact URI.")
        _require(parsed.username is None and parsed.password is None,
                 "User information in artifact URIs is forbidden.")
        _require(not parsed.query and not parsed.fragment,
                 "Query strings and fragments are forbidden in artifact URIs.")
    except ValueError as exc:
        raise InvalidInput("Invalid artifact URI.") from exc
    return value


def _validate_policy(policy):
    _keys(policy, ("schema", "policy_id", "source_revision", "max_age_seconds", "required_checks"))
    _require(policy["schema"] == "qa-ci-policy/v1", "Unsupported policy schema.")
    _identifier(policy["policy_id"])
    _revision(policy["source_revision"])
    _integer(policy["max_age_seconds"], 1, 604800)
    checks = policy["required_checks"]
    _require(isinstance(checks, list) and 1 <= len(checks) <= 100,
             "Policy must declare 1 to 100 required checks.")
    ids = []
    for check in checks:
        _keys(check, ("check_id", "critical", "minimum_passed_tests", "required_coverage"))
        ids.append(_identifier(check["check_id"]))
        _require(type(check["critical"]) is bool, "Critical must be a boolean.")
        _integer(check["minimum_passed_tests"], 1, 10000)
        _labels(check["required_coverage"])
    _require(len(set(ids)) == len(ids), "Duplicate required checks are not accepted.")


def _validate_header(intake):
    _keys(intake, ("schema", "source_revision", "data_class", "rights_basis", "retention_note", "results"),
          ("model_narrative",))
    _require(intake["schema"] == "qa-ci-intake/v1", "Unsupported intake schema.")
    _revision(intake["source_revision"])
    _require(intake["data_class"] == "synthetic" and intake["rights_basis"] == "original_synthetic",
             "This prototype accepts original synthetic fixtures only.")
    _text(intake["retention_note"])
    if "model_narrative" in intake:
        _require(isinstance(intake["model_narrative"], str) and len(intake["model_narrative"]) <= 10000,
                 "Narrative must be a bounded string.")
    _require(isinstance(intake["results"], list) and len(intake["results"]) <= 1000,
             "Expected at most 1000 result records.")


def _validate_record(record):
    _keys(record, ("evidence_id", "check_id", "locator", "source_revision", "observed_at", "tests"),
          ("narrative",))
    _identifier(record["evidence_id"])
    _identifier(record["check_id"])
    _locator(record["locator"])
    _revision(record["source_revision"])
    _time(record["observed_at"])
    if "narrative" in record:
        _require(isinstance(record["narrative"], str) and len(record["narrative"]) <= 10000,
                 "Narrative must be a bounded string.")
    _require(isinstance(record["tests"], list) and len(record["tests"]) <= 10000,
             "Expected a bounded test list.")
    ids = []
    for test in record["tests"]:
        _keys(test, ("test_id", "attempt", "status", "coverage"))
        _identifier(test["test_id"])
        _integer(test["attempt"], 1, 10000)
        ids.append((test["test_id"], test["attempt"]))
        _require(isinstance(test["status"], str) and test["status"] in STATUSES,
                 "Unknown test status; no implicit conversion to passed.")
        _labels(test["coverage"])
    _require(len(set(ids)) == len(ids), "Duplicate test/attempt identifiers are not accepted.")


def _base_report():
    return {
        "schema": "qa-ci-intake-report/v1",
        "proposed_pilot_label": "INSUFFICIENT_EVIDENCE",
        "mapping_status": "DRAFT_NOT_INTEGRATION_TESTED",
        "authorized": False,
        "execution_permission": False,
        "release_permission": False,
        "authenticity": "NOT_ASSERTED",
        "availability": "NOT_CHECKED",
        "trust": "UNVERIFIED_INPUT_ASSERTIONS",
        "reference_semantics": "Input JSON pointers and declared artifact URIs only; no artifact was fetched.",
        "reference_catalog": [{"reference_id": "input", "kind": "input_document", "locator": "/"}],
        "findings": [],
        "checks": [],
        "candidate_ci_layer": {"state": "BLOCKED", "evidence": []},
        "limitations": [
            "Pilot labels are human-review annotations, not existing SZL gate decisions or release approval.",
            "DERIVED observations do not qualify as independently observed CI evidence in the existing adapter.",
            "Only CI is mapped; source, provenance, controller, deployment and runtime remain unassessed.",
            "Rights and retention are caller assertions, not legal review or a retention enforcement mechanism.",
            "No policy origin, artifact content, test-run execution, availability or authenticity is verified.",
            "Narratives cannot set labels, evidence states, authorization or actions.",
        ],
    }


def _finding(report, code, message, refs, severity="blocker"):
    report["findings"].append({"code": code, "severity": severity,
                               "message": message, "reference_ids": list(refs)})


def invalid_json_report():
    report = _base_report()
    _finding(report, "INVALID_JSON", "Input is not accepted strict bounded JSON.", ["input"])
    return report


def normalize_ci_intake(policy, intake, assessment_time):
    """Pure mapping: caller-reviewed policy + supplied synthetic results -> draft CI layer.

    Priority: invalid policy/time -> insufficient; valid failure or xpass -> review;
    any intake gap -> insufficient; otherwise ready
    for HUMAN review. No result grants release or execution permission.
    """
    report = _base_report()
    catalog = report["reference_catalog"]
    catalog.extend([
        {"reference_id": "policy", "kind": "input_document", "locator": "/policy"},
        {"reference_id": "intake", "kind": "input_document", "locator": "/intake"},
        {"reference_id": "assessment_time", "kind": "input_document", "locator": "/assessment_time"},
    ])
    try:
        _validate_policy(policy)
    except InvalidInput as exc:
        _finding(report, "INVALID_POLICY", str(exc), ["policy"])
        return report
    try:
        now = _time(assessment_time)
    except InvalidInput as exc:
        _finding(report, "INVALID_ASSESSMENT_TIME", str(exc), ["assessment_time"])
        return report
    report["policy_id"] = policy["policy_id"]
    report["source_revision"] = policy["source_revision"]
    report["assessment_time"] = assessment_time
    report["candidate_ci_layer"]["subject_revision"] = policy["source_revision"]
    try:
        _validate_header(intake)
    except InvalidInput as exc:
        _finding(report, "INVALID_INTAKE", str(exc), ["intake"])
    if not isinstance(intake, dict) or not isinstance(intake.get("results"), list):
        return report
    if len(intake["results"]) > 1000:
        return report
    header_bound = intake.get("source_revision") == policy["source_revision"]
    if not header_bound:
        _finding(report, "INTAKE_REVISION_MISMATCH", "Intake is not bound to the policy revision.", ["policy", "intake"])
    required = {}
    for i, check in enumerate(policy["required_checks"]):
        ref = f"policy-check-{i}"
        catalog.append({"reference_id": ref, "kind": "input_document", "locator": f"/policy/required_checks/{i}"})
        required[check["check_id"]] = (check, ref)
    grouped = {check_id: [] for check_id in required}
    seen_ids = set()
    known_critical_failure = False
    known_other_failure = False
    known_xpass = False
    for index, record in enumerate(intake["results"]):
        ref = f"result-{index}"
        entry = {"reference_id": ref, "kind": "input_document", "locator": f"/intake/results/{index}"}
        catalog.append(entry)
        try:
            _validate_record(record)
        except InvalidInput as exc:
            _finding(report, "MALFORMED_RESULT", str(exc), [ref])
            continue
        entry.update(evidence_id=record["evidence_id"], declared_artifact_uri=record["locator"])
        if record["evidence_id"] in seen_ids:
            _finding(report, "DUPLICATE_EVIDENCE_ID", "Evidence identity is ambiguous.", [ref])
        seen_ids.add(record["evidence_id"])
        if record["check_id"] not in required:
            _finding(report, "UNCONFIGURED_CHECK", "Check is not in the caller-reviewed required-check policy.", [ref, "policy"])
            continue
        check, policy_ref = required[record["check_id"]]
        bound = header_bound and record["source_revision"] == policy["source_revision"]
        if not bound:
            _finding(report, "RESULT_REVISION_MISMATCH", "Result does not match the exact policy source revision.", [ref, policy_ref])
        age = (now - _time(record["observed_at"])).total_seconds()
        fresh = 0 <= age <= policy["max_age_seconds"]
        if not fresh:
            _finding(report, "FUTURE_RESULT" if age < 0 else "STALE_RESULT",
                     "Result timestamp is outside the accepted assessment window.", [ref, "assessment_time", "policy"])
        counts = {status: sum(t["status"] == status for t in record["tests"]) for status in STATUSES}
        distinct_passed = len({t["test_id"] for t in record["tests"] if t["status"] == "passed"})
        passed_coverage = sorted({label for t in record["tests"] if t["status"] == "passed" for label in t["coverage"]})
        normalized = {
            "check_id": record["check_id"], "evidence_id": record["evidence_id"],
            "reference_id": ref, "status_counts": counts,
            "distinct_passed_tests": distinct_passed,
            "passed_coverage": passed_coverage, "revision_matches": bound,
            "timestamp_in_window": fresh,
            "tests": [{"test_id": t["test_id"], "attempt": t["attempt"], "status": t["status"], "coverage": sorted(t["coverage"])}
                      for t in sorted(record["tests"], key=lambda t: (t["test_id"], t["attempt"]))],
        }
        grouped[record["check_id"]].append(normalized)
        report["candidate_ci_layer"]["evidence"].append({
            "uri": record["locator"], "observation": "DERIVED",
            "observed_at": record["observed_at"], "observer": OBSERVER,
            "subject_revision": record["source_revision"],
            "note": "Normalized supplied synthetic CI JSON; authenticity and availability not verified.",
        })
        if bound and fresh and counts["failed"]:
            known_critical_failure |= check["critical"]
            known_other_failure |= not check["critical"]
            _finding(report, "CRITICAL_TEST_FAILURE" if check["critical"] else "TEST_FAILURE",
                     "At least one test is explicitly failed for the expected revision.", [ref, policy_ref],
                     "critical" if check["critical"] else "failure")
        if bound and fresh and counts["xpass"]:
            known_xpass = True
            _finding(report, "UNEXPECTED_PASS", "An expected-failure test unexpectedly passed; this is not an ordinary pass.",
                     [ref, policy_ref], "failure")
        for status in ("absent", "skipped", "error", "xfail"):
            if counts[status]:
                _finding(report, f"TESTS_{status.upper()}", f"Required check includes {status} tests; these are not passed tests.", [ref, policy_ref])
        if not record["tests"]:
            _finding(report, "NO_TESTS", "The supplied artifact contains no test cases.", [ref, policy_ref])
        if distinct_passed < check["minimum_passed_tests"]:
            _finding(report, "INSUFFICIENT_PASSED_TESTS", "The required passed-test count is not met.", [ref, policy_ref])
        if not set(check["required_coverage"]) <= set(passed_coverage):
            _finding(report, "COVERAGE_GAP", "Required coverage labels are absent from passed tests.", [ref, policy_ref])
    for check_id, (check, policy_ref) in required.items():
        records = grouped[check_id]
        report["checks"].append({"check_id": check_id, "critical": check["critical"],
                                  "artifact_presence": "DECLARED" if records else "ABSENT_OR_MALFORMED",
                                  "results": records})
        if not records:
            _finding(report, "MISSING_CHECK_ARTIFACT", "Required check artifact is absent or malformed; no supporting evidence ID is asserted.",
                     [policy_ref, "intake"])
        elif len(records) > 1:
            _finding(report, "AMBIGUOUS_CHECK_RUNS", "Multiple runs for one required check require explicit human selection; no implicit latest/pass selection.",
                     [policy_ref] + [record["reference_id"] for record in records])
        qualified = [record for record in records if record["revision_matches"] and record["timestamp_in_window"]]
        attempts = {}
        for record in qualified:
            for test in record["tests"]:
                attempts.setdefault(test["test_id"], []).append(test)
        for test_id, history in attempts.items():
            attempt_numbers = sorted({test["attempt"] for test in history})
            if attempt_numbers != list(range(1, attempt_numbers[-1] + 1)):
                _finding(report, "MISSING_ATTEMPT_HISTORY", "Test attempt history must start at 1 and be contiguous; omitted attempts cannot be assumed passed.",
                         [policy_ref] + [record["reference_id"] for record in qualified])
            failed_attempts = [test["attempt"] for test in history if test["status"] == "failed"]
            passed_attempts = [test["attempt"] for test in history if test["status"] == "passed"]
            if failed_attempts and passed_attempts and min(failed_attempts) < max(passed_attempts):
                _finding(report, "FLAKY_RETRY", "A later passing retry does not erase a prior failed attempt.",
                         [policy_ref] + [record["reference_id"] for record in qualified], "failure")
    blocked = any(f["severity"] == "blocker" for f in report["findings"])
    if known_critical_failure or known_other_failure or known_xpass:
        label, state = "REQUIRES_REVIEW", "FAIL"
    elif blocked:
        label, state = "INSUFFICIENT_EVIDENCE", "BLOCKED"
    else:
        label, state = "READY_FOR_REVIEW", "PASS"
    report["proposed_pilot_label"] = label
    report["candidate_ci_layer"]["state"] = state
    return report


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidInput("Duplicate JSON keys are not accepted.")
        result[key] = value
    return result


def parse_and_normalize(text):
    try:
        _require(isinstance(text, str) and len(text) <= MAX_JSON_CHARS, "Input is too large.")
        def reject_constant(value):
            raise InvalidInput("Non-finite JSON numbers are not accepted.")
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)
        _keys(value, ("policy", "intake", "assessment_time"))
    except (InvalidInput, ValueError, RecursionError):
        return invalid_json_report()
    return normalize_ci_intake(value["policy"], value["intake"], value["assessment_time"])


def main(stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    report = parse_and_normalize(stdin.read(MAX_JSON_CHARS + 1))
    stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"READY_FOR_REVIEW": 0, "REQUIRES_REVIEW": 2, "INSUFFICIENT_EVIDENCE": 3}[report["proposed_pilot_label"]]


if __name__ == "__main__":
    raise SystemExit(main())
