# SPDX-License-Identifier: Apache-2.0
"""Synthetic controls for predicate-AND / dream-gate. No runtime qualification."""
import contextlib
import io
import json
import unittest

import szl_predicate_and as M


def predicates(values):
    rows = []
    for ident, value in values:
        rows.append({"id": ident, "value": value, "population": "fixture-" + ident})
    return rows


class DreamTests(unittest.TestCase):
    def test_forbidden_promotions(self):
        cases = {
            "make it all AGI": "FORBIDDEN_PROMOTION_AGI",
            "status ALL_DONE": "FORBIDDEN_PROMOTION_ALL_DONE",
            "everything fully operational": "FORBIDDEN_PROMOTION_OPERATIONAL",
            "paint it all green": "FORBIDDEN_PROMOTION_ALL_GREEN",
            "production_authorized=true": "FORBIDDEN_PROMOTION_AUTHORITY",
            "runtime_verified: true": "FORBIDDEN_PROMOTION_RUNTIME",
            "make it all work": "FORBIDDEN_PROMOTION_SWEEP",
            "HTTP 200 means LIVE": "FORBIDDEN_PROMOTION_HTTP_LIVE",
            "RUNNING therefore runtime verified": "FORBIDDEN_PROMOTION_RUNNING",
        }
        for claim, code in cases.items():
            with self.subTest(claim=claim):
                with self.assertRaises(M.GateError) as raised:
                    M.classify({"claim": claim})
                self.assertEqual(str(raised.exception), code)

    def test_honest_partial_is_not_a_dream(self):
        result = M.classify({"claim": "The estate is PARTIAL. HTTP 200 is OBSERVED."})
        self.assertEqual(result["dreams"], [])
        self.assertFalse(result["production_authorization"])

    def test_authority_flag_in_payload(self):
        with self.assertRaises(M.GateError) as raised:
            M.classify({"production_authorization": True})
        self.assertEqual(str(raised.exception), "FORBIDDEN_PROMOTION_AUTHORITY")


class AndFoldTests(unittest.TestCase):
    def test_and_not_or(self):
        result = M.and_fold(predicates([("a", True), ("b", False), ("c", True)]))
        self.assertEqual(result["fold"], "BLOCKED")
        self.assertEqual(result["blocked"], ["b"])

    def test_unknown_is_hold_not_fail(self):
        result = M.and_fold(predicates([("a", True), ("b", None)]))
        self.assertEqual(result["fold"], "HOLD")
        self.assertEqual(result["holds"], ["b"])

    def test_all_true_is_still_not_authorization(self):
        classified = M.classify({"predicates": predicates([("a", True), ("b", True)])})
        self.assertEqual(classified["state"], "ADMIT_PREDICATES_ONLY")
        self.assertFalse(classified["production_authorization"])
        self.assertFalse(classified["runtime_verified"])
        self.assertFalse(classified["agi_claim"])

    def test_boolean_one_is_not_true(self):
        with self.assertRaises(M.GateError):
            M.and_fold([{"id": "a", "value": 1, "population": "x"}])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(M.GateError):
            M.and_fold(predicates([("a", True), ("a", False)]))

    def test_default_estate_is_blocked(self):
        result = M.default_estate()
        self.assertEqual(result["state"], "BLOCKED")
        self.assertIn("lambda_admit", result["predicates"]["blocked"])
        self.assertFalse(result["production_authorization"])


class IncomparabilityTests(unittest.TestCase):
    def test_same_population_may_add(self):
        result = M.sum_counts([
            {"population": "github-named-active", "count": 40},
            {"population": "github-named-active", "count": 55},
        ])
        self.assertEqual(result["count"], 95)

    def test_github_plus_hf_refused(self):
        with self.assertRaises(M.GateError) as raised:
            M.sum_counts([
                {"population": "github-org-search", "count": 124},
                {"population": "hf-models-likes-sort", "count": 47},
            ])
        self.assertEqual(str(raised.exception), "INCOMPARABLE_POPULATIONS")

    def test_boolean_count_rejected(self):
        with self.assertRaises(M.GateError):
            M.sum_counts([{"population": "x", "count": True}])


class CollapseTests(unittest.TestCase):
    def test_one_score_refused(self):
        with self.assertRaises(M.GateError) as raised:
            M.refuse_collapse({"mean_at_k": 0.75, "pass_power_k": 0.5,
                               "pass_at_k_empirical": 1.0, "as_one_score": True})
        self.assertEqual(str(raised.exception), "DISTINCT_METRICS")

    def test_keeping_metrics_distinct_is_ok(self):
        M.refuse_collapse({"mean_at_k": 0.75, "pass_power_k": 0.5,
                           "pass_at_k_empirical": 1.0, "as_one_score": False})


class SequenceTests(unittest.TestCase):
    def closed(self, **overrides):
        row = {name: False for name in M.RELEASE_GATES}
        row.update(overrides)
        return row

    def test_skip_detected(self):
        row = self.closed(source_admitted=True, release_authorized=True)
        with self.assertRaises(M.GateError) as raised:
            M.sequence_check(row)
        self.assertEqual(str(raised.exception), "SEQUENCE_SKIP")

    def test_first_open_recorded(self):
        result = M.sequence_check(self.closed())
        self.assertEqual(result["first_open"], "source_admitted")
        self.assertFalse(any(item["value"] is True for item in result["gates"]))

    def test_release_authorized_true_forbidden(self):
        row = {name: True for name in M.RELEASE_GATES}
        with self.assertRaises(M.GateError):
            M.sequence_check(row)


class CliTests(unittest.TestCase):
    def run_main(self, action, raw=b""):
        stdin = io.BytesIO(raw)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            old = sys_stdin(stdin)
            try:
                code = M.main([action])
            finally:
                old()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_estate_exit_two(self):
        code, out, _ = self.run_main("estate")
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertFalse(payload["production_authorization"])
        self.assertEqual(payload["state"], "BLOCKED")

    def test_classify_forbidden_exits_one(self):
        code, _, err = self.run_main("classify", b'{"claim":"make it all AGI"}')
        self.assertEqual(code, 1)
        self.assertIn("FORBIDDEN_PROMOTION_AGI", err)
        self.assertIn('"production_authorization":false', err.replace(" ", ""))

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(M.GateError):
            M.strict_json(b'{"x":1,"x":2}')


def sys_stdin(stream):
    import sys
    original = sys.stdin
    sys.stdin = stream
    if not hasattr(sys.stdin, "buffer"):
        sys.stdin.buffer = stream

    def restore():
        sys.stdin = original
    return restore


if __name__ == "__main__":
    unittest.main()
