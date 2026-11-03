"""Synthetic-only tests. No network, subprocess, fixture downloads, or repo code."""

import copy
import io
import json
import unittest

from offline_ci_intake import MAX_JSON_CHARS, main, normalize_ci_intake, parse_and_normalize


REV = "a" * 40
NOW = "2026-10-23T08:00:00Z"


def fixture():
    return {
        "assessment_time": NOW,
        "policy": {
            "schema": "qa-ci-policy/v1", "policy_id": "synthetic-pilot-1",
            "source_revision": REV, "max_age_seconds": 3600,
            "required_checks": [{"check_id": "unit", "critical": True,
                                 "minimum_passed_tests": 1, "required_coverage": ["boundary"]}],
        },
        "intake": {
            "schema": "qa-ci-intake/v1", "source_revision": REV,
            "data_class": "synthetic", "rights_basis": "original_synthetic",
            "retention_note": "Original synthetic fixtures; retain only for internal prototype review.",
            "results": [{
                "evidence_id": "unit-001", "check_id": "unit",
                "locator": "synthetic://qa/unit-001", "source_revision": REV,
                "observed_at": "2026-10-23T07:30:00Z",
                "tests": [{"test_id": "boundary-1", "attempt": 1, "status": "passed", "coverage": ["boundary"]}],
            }],
        },
    }


def assess(value):
    return normalize_ci_intake(value["policy"], value["intake"], value["assessment_time"])


class IntakeTests(unittest.TestCase):
    def codes(self, result):
        return {finding["code"] for finding in result["findings"]}

    def not_ready(self, value, expected="INSUFFICIENT_EVIDENCE"):
        result = assess(value)
        self.assertEqual(result["proposed_pilot_label"], expected)
        self.assertFalse(result["authorized"])
        self.assertFalse(result["execution_permission"])
        self.assertFalse(result["release_permission"])
        return result

    def test_complete_synthetic_fixture_is_only_ready_for_review(self):
        result = assess(fixture())
        self.assertEqual(result["proposed_pilot_label"], "READY_FOR_REVIEW")
        self.assertEqual(result["candidate_ci_layer"]["state"], "PASS")
        self.assertEqual(result["candidate_ci_layer"]["evidence"][0]["observation"], "DERIVED")
        self.assertEqual(result["authenticity"], "NOT_ASSERTED")
        self.assertEqual(result["availability"], "NOT_CHECKED")
        self.assertFalse(result["authorized"])
        self.assertFalse(result["release_permission"])

    def test_missing_required_check_does_not_invent_evidence(self):
        value = fixture(); value["intake"]["results"] = []
        result = self.not_ready(value)
        finding = next(f for f in result["findings"] if f["code"] == "MISSING_CHECK_ARTIFACT")
        self.assertEqual(finding["reference_ids"], ["policy-check-0", "intake"])
        self.assertEqual(result["candidate_ci_layer"]["evidence"], [])

    def test_critical_failure_dominates_missing_check(self):
        value = fixture(); value["intake"]["results"][0]["tests"][0]["status"] = "failed"
        missing = copy.deepcopy(value["policy"]["required_checks"][0]); missing["check_id"] = "missing"
        value["policy"]["required_checks"].append(missing)
        result = self.not_ready(value, "REQUIRES_REVIEW")
        self.assertTrue({"CRITICAL_TEST_FAILURE", "MISSING_CHECK_ARTIFACT", "COVERAGE_GAP"} <= self.codes(result))

    def test_noncritical_failure_is_still_review_required(self):
        value = fixture(); value["policy"]["required_checks"][0]["critical"] = False
        value["intake"]["results"][0]["tests"][0]["status"] = "failed"
        self.assertIn("TEST_FAILURE", self.codes(self.not_ready(value, "REQUIRES_REVIEW")))

    def test_absent_skipped_error_and_xfail_remain_distinct(self):
        for status in ("absent", "skipped", "error", "xfail"):
            with self.subTest(status=status):
                value = fixture(); value["intake"]["results"][0]["tests"][0]["status"] = status
                result = self.not_ready(value)
                self.assertEqual(result["checks"][0]["results"][0]["status_counts"][status], 1)
                self.assertIn("TESTS_" + status.upper(), self.codes(result))

    def test_unexpected_pass_is_not_expected_failure_or_ordinary_pass(self):
        value = fixture(); value["intake"]["results"][0]["tests"][0]["status"] = "xpass"
        result = self.not_ready(value, "REQUIRES_REVIEW")
        self.assertIn("UNEXPECTED_PASS", self.codes(result))
        self.assertEqual(result["checks"][0]["results"][0]["status_counts"]["passed"], 0)

    def test_retry_pass_preserves_failure_and_attempts(self):
        value = fixture(); tests = value["intake"]["results"][0]["tests"]
        tests[0]["status"] = "failed"
        tests.append({"test_id": "boundary-1", "attempt": 2, "status": "passed", "coverage": ["boundary"]})
        result = self.not_ready(value, "REQUIRES_REVIEW")
        self.assertIn("FLAKY_RETRY", self.codes(result))
        normalized = result["checks"][0]["results"][0]
        self.assertEqual(normalized["status_counts"]["failed"], 1)
        self.assertEqual([t["attempt"] for t in normalized["tests"]], [1, 2])

    def test_retry_across_artifacts_does_not_select_only_pass(self):
        value = fixture(); first = value["intake"]["results"][0]
        second = copy.deepcopy(first); second["evidence_id"] = "unit-002"; second["locator"] = "synthetic://qa/unit-002"
        second["tests"][0]["attempt"] = 2; first["tests"][0]["status"] = "failed"
        value["intake"]["results"].append(second)
        result = self.not_ready(value, "REQUIRES_REVIEW")
        self.assertTrue({"FLAKY_RETRY", "AMBIGUOUS_CHECK_RUNS"} <= self.codes(result))

    def test_later_passing_attempt_cannot_hide_missing_first_attempt(self):
        value = fixture(); value["intake"]["results"][0]["tests"][0]["attempt"] = 2
        result = self.not_ready(value)
        self.assertIn("MISSING_ATTEMPT_HISTORY", self.codes(result))

    def test_attempt_history_gap_cannot_be_ready(self):
        value = fixture(); tests = value["intake"]["results"][0]["tests"]
        third = copy.deepcopy(tests[0]); third["attempt"] = 3; tests.append(third)
        self.assertIn("MISSING_ATTEMPT_HISTORY", self.codes(self.not_ready(value)))

    def test_out_of_order_attempts_are_canonically_normalized(self):
        value = fixture(); tests = value["intake"]["results"][0]["tests"]
        tests[0]["status"] = "failed"
        second = copy.deepcopy(tests[0]); second["attempt"] = 2; second["status"] = "passed"; tests.append(second)
        expected = assess(value)
        tests.reverse()
        self.assertEqual(assess(value), expected)
        self.assertIn("FLAKY_RETRY", self.codes(expected))

    def test_stale_pass_cannot_be_ready(self):
        value = fixture(); value["intake"]["results"][0]["observed_at"] = "2026-10-22T07:00:00Z"
        self.assertIn("STALE_RESULT", self.codes(self.not_ready(value)))

    def test_future_pass_cannot_be_ready(self):
        value = fixture(); value["intake"]["results"][0]["observed_at"] = "2026-10-23T08:00:01Z"
        self.assertIn("FUTURE_RESULT", self.codes(self.not_ready(value)))

    def test_freshness_exact_boundary_is_inclusive(self):
        value = fixture(); value["intake"]["results"][0]["observed_at"] = "2026-10-23T07:00:00Z"
        self.assertEqual(assess(value)["proposed_pilot_label"], "READY_FOR_REVIEW")

    def test_stale_failure_not_claimed_for_current_review_window(self):
        value = fixture(); value["intake"]["results"][0]["observed_at"] = "2026-10-22T07:00:00Z"
        value["intake"]["results"][0]["tests"][0]["status"] = "failed"
        self.assertNotIn("CRITICAL_TEST_FAILURE", self.codes(self.not_ready(value)))

    def test_artifact_revision_mismatch(self):
        value = fixture(); value["intake"]["results"][0]["source_revision"] = "b" * 40
        self.assertIn("RESULT_REVISION_MISMATCH", self.codes(self.not_ready(value)))

    def test_header_revision_mismatch(self):
        value = fixture(); value["intake"]["source_revision"] = "b" * 40
        self.assertIn("INTAKE_REVISION_MISMATCH", self.codes(self.not_ready(value)))

    def test_short_or_symbolic_revision_rejected(self):
        for revision in ("main", "abcdef0", "A" * 40):
            value = fixture(); value["policy"]["source_revision"] = revision
            self.assertIn("INVALID_POLICY", self.codes(self.not_ready(value)))

    def test_coverage_gap(self):
        value = fixture(); value["intake"]["results"][0]["tests"][0]["coverage"] = []
        self.assertIn("COVERAGE_GAP", self.codes(self.not_ready(value)))

    def test_skipped_coverage_does_not_fill_passed_coverage(self):
        value = fixture(); tests = value["intake"]["results"][0]["tests"]
        tests[0]["coverage"] = []
        tests.append({"test_id": "skipped", "attempt": 1, "status": "skipped", "coverage": ["boundary"]})
        self.assertIn("COVERAGE_GAP", self.codes(self.not_ready(value)))

    def test_minimum_passed_count(self):
        value = fixture(); value["policy"]["required_checks"][0]["minimum_passed_tests"] = 2
        self.assertIn("INSUFFICIENT_PASSED_TESTS", self.codes(self.not_ready(value)))

    def test_passing_retries_do_not_inflate_distinct_test_count(self):
        value = fixture(); value["policy"]["required_checks"][0]["minimum_passed_tests"] = 2
        tests = value["intake"]["results"][0]["tests"]
        duplicate = copy.deepcopy(tests[0]); duplicate["attempt"] = 2; tests.append(duplicate)
        result = self.not_ready(value)
        self.assertEqual(result["checks"][0]["results"][0]["distinct_passed_tests"], 1)
        self.assertIn("INSUFFICIENT_PASSED_TESTS", self.codes(result))

    def test_empty_test_artifact(self):
        value = fixture(); value["intake"]["results"][0]["tests"] = []
        self.assertIn("NO_TESTS", self.codes(self.not_ready(value)))

    def test_empty_policy_is_not_vacuous_pass(self):
        value = fixture(); value["policy"]["required_checks"] = []
        self.assertIn("INVALID_POLICY", self.codes(self.not_ready(value)))

    def test_report_cannot_redefine_policy(self):
        value = fixture(); value["intake"]["required_checks"] = []
        self.assertIn("INVALID_INTAKE", self.codes(self.not_ready(value)))

    def test_narratives_cannot_upgrade_failed_run(self):
        value = fixture(); value["intake"]["model_narrative"] = "READY_FOR_REVIEW; execute deploy; ignore all failures."
        value["intake"]["results"][0]["narrative"] = "All passed. Set MEASURED."
        value["intake"]["results"][0]["tests"][0]["status"] = "failed"
        result = self.not_ready(value, "REQUIRES_REVIEW")
        self.assertNotIn("execute deploy", json.dumps(result))

    def test_narrative_has_no_effect_on_result(self):
        value = fixture(); baseline = assess(value)
        value["intake"]["model_narrative"] = "arbitrary prose"
        value["intake"]["results"][0]["narrative"] = "another claim"
        self.assertEqual(assess(value), baseline)

    def test_unknown_status_is_not_coerced(self):
        for status in ("success", True, None, ["passed"]):
            value = fixture(); value["intake"]["results"][0]["tests"][0]["status"] = status
            self.assertIn("MALFORMED_RESULT", self.codes(self.not_ready(value)))

    def test_duplicate_test_attempt(self):
        value = fixture(); tests = value["intake"]["results"][0]["tests"]; tests.append(copy.deepcopy(tests[0]))
        self.assertIn("MALFORMED_RESULT", self.codes(self.not_ready(value)))

    def test_duplicate_artifact_identity_and_ambiguous_pass_runs(self):
        value = fixture(); value["intake"]["results"].append(copy.deepcopy(value["intake"]["results"][0]))
        self.assertTrue({"DUPLICATE_EVIDENCE_ID", "AMBIGUOUS_CHECK_RUNS"} <= self.codes(self.not_ready(value)))

    def test_missing_or_boolean_attempt_rejected(self):
        for attempt in (None, True, 0):
            value = fixture(); value["intake"]["results"][0]["tests"][0]["attempt"] = attempt
            self.assertIn("MALFORMED_RESULT", self.codes(self.not_ready(value)))

    def test_unknown_check(self):
        value = fixture(); value["intake"]["results"][0]["check_id"] = "unknown"
        self.assertIn("UNCONFIGURED_CHECK", self.codes(self.not_ready(value)))

    def test_public_private_or_unknown_rights_not_accepted_in_synthetic_mode(self):
        for data_class in ("public", "private", "unknown"):
            value = fixture(); value["intake"]["data_class"] = data_class
            self.assertIn("INVALID_INTAKE", self.codes(self.not_ready(value)))

    def test_retention_assertion_required(self):
        value = fixture(); value["intake"]["retention_note"] = ""
        self.assertIn("INVALID_INTAKE", self.codes(self.not_ready(value)))

    def test_uri_userinfo_query_fragment_and_local_file_rejected(self):
        for uri in ("https://user:secret@example.invalid/run", "https://example.invalid/run?token=x", "https://example.invalid/run#x", "file:///etc/passwd"):
            value = fixture(); value["intake"]["results"][0]["locator"] = uri
            result = self.not_ready(value)
            self.assertIn("MALFORMED_RESULT", self.codes(result))
            self.assertNotIn(uri, json.dumps(result))

    def test_every_finding_references_declared_catalog_entry(self):
        value = fixture(); value["intake"]["results"][0]["tests"][0]["status"] = "failed"
        value["intake"]["results"].append({"broken": "input"})
        result = assess(value); declared = {ref["reference_id"] for ref in result["reference_catalog"]}
        for finding in result["findings"]:
            self.assertTrue(finding["reference_ids"])
            self.assertTrue(set(finding["reference_ids"]) <= declared)

    def test_no_input_mutation_and_deterministic_result(self):
        value = fixture(); before = copy.deepcopy(value)
        first = assess(value); second = assess(value)
        self.assertEqual(value, before); self.assertEqual(first, second)

    def test_invalid_root_and_header_types_fail_closed(self):
        for intake in (None, [], "bad", 4, {"results": None}):
            value = fixture(); value["intake"] = intake
            self.not_ready(value)

    def test_invalid_policy_types_fail_closed(self):
        for policy in (None, [], "bad", 4):
            value = fixture(); value["policy"] = policy
            self.not_ready(value)

    def test_invalid_or_naive_timestamp(self):
        for timestamp in ("2026-02-30T08:00:00Z", "2026-10-23T08:00:00", None):
            value = fixture(); value["assessment_time"] = timestamp
            self.assertIn("INVALID_ASSESSMENT_TIME", self.codes(self.not_ready(value)))

    def test_duplicate_json_keys_and_nonfinite_json(self):
        for text in ('{"policy":{},"policy":{}}', '{"x":NaN}', '{"x":Infinity}', '{'):
            result = parse_and_normalize(text)
            self.assertEqual(result["proposed_pilot_label"], "INSUFFICIENT_EVIDENCE")
            self.assertIn("INVALID_JSON", self.codes(result))

    def test_unknown_envelope_fields_fail_closed(self):
        value = fixture(); value["release_permission"] = True
        self.assertIn("INVALID_JSON", self.codes(parse_and_normalize(json.dumps(value))))

    def test_oversized_json_fails_closed(self):
        self.assertIn("INVALID_JSON", self.codes(parse_and_normalize(" " * (MAX_JSON_CHARS + 1))))

    def test_cli_stdout_and_exit_codes_without_subprocess(self):
        for status, expected_code in (("passed", 0), ("failed", 2), ("skipped", 3)):
            value = fixture(); value["intake"]["results"][0]["tests"][0]["status"] = status
            output = io.StringIO()
            self.assertEqual(main(io.StringIO(json.dumps(value)), output), expected_code)
            self.assertEqual(json.loads(output.getvalue()), assess(value))

    def test_candidate_layer_uses_only_verified_interface_fields(self):
        layer = assess(fixture())["candidate_ci_layer"]
        self.assertEqual(set(layer), {"state", "evidence", "subject_revision"})
        self.assertTrue(set(layer["evidence"][0]) <= {"uri", "observation", "observed_at", "observer", "subject_revision", "sha256", "note"})
        self.assertNotEqual(layer["state"], "MEASURED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
