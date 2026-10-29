"""CPU-only additive diagnostics regression tests; no optional dependencies.

Run directly with ``python -I -B tests/test_willay_diagnostics_unittest.py``.
The frozen reference functions preserve scoring/comparison behavior from main
a76b0eb5bbb4afd8bf108d6d33ffb0058bbfc4ee, independently of evaluator helpers.
Diagnostic type reasons describe a field's TOP-LEVEL JSON type. Differences
inside a JSON object or array are value mismatches, including nested types.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "willay_diagnostics_eval",
    Path(__file__).resolve().parents[1] / "operational" / "evaluate_willay.py",
)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def legacy_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def legacy_parse(raw):
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=reject_constant)


def legacy_score(raw, expected):
    """Frozen score oracle: no calls to the evaluator under test."""
    checks = {"json_object": False}
    try:
        parsed = legacy_parse(raw)
        legacy_canonical(parsed)
        checks["json_object"] = isinstance(parsed, dict)
    except (ValueError, TypeError, RecursionError):
        parsed = None
    for field, value in expected.items():
        checks["field:" + field] = (
            isinstance(parsed, dict) and field in parsed
            and legacy_canonical(parsed[field]) == legacy_canonical(value)
        )
    return {"checks": checks, "score": sum(checks.values()) / len(checks),
            "passed": all(checks.values())}


def legacy_comparison(cases, base, candidate, pass_rate=1.0, improvement=0.0):
    """Frozen valid-input comparison oracle; original threshold tests persist."""
    lanes = {}
    for name, outputs in (("base", base), ("candidate", candidate)):
        rows = []
        for case, raw in zip(cases, outputs):
            rows.append({"case_id": case["id"], "raw_output": raw,
                         "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                         **legacy_score(raw, case["expected"])})
        lanes[name] = {
            "cases": rows,
            "mean_score": sum(row["score"] for row in rows) / len(rows),
            "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        }
    delta = lanes["candidate"]["mean_score"] - lanes["base"]["mean_score"]
    pass_delta = lanes["candidate"]["pass_rate"] - lanes["base"]["pass_rate"]
    passed = (lanes["candidate"]["pass_rate"] >= pass_rate
              and delta >= improvement and pass_delta >= 0)
    return {
        "lanes": lanes, "delta_mean_score": delta, "delta_pass_rate": pass_delta,
        "thresholds": {"minimum_pass_rate": pass_rate,
                       "minimum_improvement": improvement},
        "result": "PASS" if passed else "FAIL",
    }


def without_diagnostics(value):
    """Remove only the explicitly additive lane/row diagnostic locations."""
    result = dict(value)
    result["lanes"] = {}
    for name, lane in value["lanes"].items():
        clean_lane = {key: item for key, item in lane.items() if key != "diagnostics"}
        clean_lane["cases"] = [
            {key: item for key, item in row.items() if key != "diagnostics"}
            for row in lane["cases"]
        ]
        result["lanes"][name] = clean_lane
    return result


EXPECTED_VARIANTS = [
    {"state": "HOLD", "approved": False},
    {"value": 1}, {"value": 1.0}, {"value": True}, {"value": None},
    {"value": [1, False, {"nested": "\u2603"}]},
    {"value": {"n": 1, "enabled": False}},
    {"diagnostics": "ordinary expected field"},
]
RAW_VARIANTS = [
    "", "not json", "{", "{}", "[]", "null", "true", "1", "1.0", '"{}"',
    '{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}',
    '{"value":1e999}', '{"value":-1e999}',
    '{"value":1,"value":2}', '{"value":{"n":1,"n":2}}',
    '{"value":0}', '{"value":false}', '{"value":"1"}',
    '{"value":[]}', '{"value":{}}',
    '{"value":[1,0,{"nested":"\u2603"}]}',
    '{"value":{"enabled":0,"n":1}}',
    '{"approved":false,"state":"HOLD","extra":"diagnostic only"}',
    ' {"state":"HOLD","approved":false} \n',
    '{"state":"GO","approved":true}',
] + [legacy_canonical(expected).decode() for expected in EXPECTED_VARIANTS]


class ScoreDiagnosticsTests(unittest.TestCase):
    def assert_diagnostics(self, raw, expected, parse, fields, extras=()):
        actual = ev.score_output(raw, expected)
        self.assertEqual(actual["diagnostics"], {
            "parse": parse, "fields": fields, "unexpected_fields": list(extras),
        })
        self.assertEqual({key: value for key, value in actual.items()
                          if key != "diagnostics"}, legacy_score(raw, expected))

    def test_invalid_json_duplicate_keys_and_nonfinite_are_unavailable(self):
        invalid = ["", "not json", "{", '{"state":"HOLD",}',
                   '{"state":"HOLD","state":"GO"}',
                   '{"nested":{"x":0,"x":1}}',
                   "NaN", "Infinity", "-Infinity", "1e999",
                   '{"approved":NaN}', '{"approved":Infinity}',
                   '{"approved":-Infinity}', '{"approved":1e999}',
                   '{"approved":[{"nested":-1e999}]}']
        expected = {"state": "HOLD", "approved": False}
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assert_diagnostics(raw, expected, "invalid_json",
                                        {"state": "unavailable", "approved": "unavailable"})

    def test_valid_nonobjects_are_distinct_from_invalid_json(self):
        for raw in ["null", "true", "false", "0", "1.0", '"text"', "[]", '[1,{"x":2}]']:
            with self.subTest(raw=raw):
                self.assert_diagnostics(raw, {"value": None}, "non_object",
                                        {"value": "unavailable"})

    def test_explicit_top_level_type_and_nested_value_reasons(self):
        samples = [
            (None, None, "match"), (False, False, "match"),
            (1, 1, "match"), (1.0, 1.0, "match"),
            (1, 1.0, "value_mismatch"), (1.0, 1, "value_mismatch"),
            (0, -0.0, "value_mismatch"), ("one", "two", "value_mismatch"),
            (True, 1, "type_mismatch"), (1, True, "type_mismatch"),
            (False, 0.0, "type_mismatch"), (0.0, False, "type_mismatch"),
            (None, False, "type_mismatch"), ("1", 1, "type_mismatch"),
            ({}, [], "type_mismatch"), ([], {}, "type_mismatch"),
            ([], None, "type_mismatch"), ({}, "{}", "type_mismatch"),
            ({"flag": True}, {"flag": 1}, "value_mismatch"),
            ([False], [0], "value_mismatch"),
            ({"items": []}, {"items": {}}, "value_mismatch"),
            ({"n": 1}, {"n": 1.0}, "value_mismatch"),
            ({"x": None}, {}, "value_mismatch"),
            ({}, {"extra": None}, "value_mismatch"),
            ([1], [1, 2], "value_mismatch"),
            ({"a": 1, "b": [False, "\u2603"]},
             {"b": [False, "\u2603"], "a": 1}, "match"),
        ]
        for expected, observed, reason in samples:
            with self.subTest(expected=expected, observed=observed):
                self.assert_diagnostics(legacy_canonical({"value": observed}).decode(),
                                        {"value": expected}, "valid", {"value": reason})

    def test_missing_is_not_null_and_extra_fields_do_not_change_pass(self):
        self.assert_diagnostics("{}", {"value": None}, "valid", {"value": "missing"})
        self.assert_diagnostics('{"value":null}', {"value": None}, "valid", {"value": "match"})
        raw = '{"z":9,"state":"HOLD","approved":false,"a":1}'
        expected = {"state": "HOLD", "approved": False}
        self.assert_diagnostics(raw, expected, "valid",
                                {"state": "match", "approved": "match"}, ["a", "z"])
        self.assertTrue(ev.score_output(raw, expected)["passed"])
        self.assertEqual(ev.score_output(raw, expected)["score"], 1.0)

    def test_diagnostic_field_names_are_not_reserved_in_expected_objects(self):
        expected = {"diagnostics": "ordinary expected field", "parse": "data"}
        self.assert_diagnostics(legacy_canonical(expected).decode(), expected, "valid",
                                {"diagnostics": "match", "parse": "match"})

    def test_broad_scoring_equivalence_to_frozen_reference(self):
        for expected in EXPECTED_VARIANTS:
            for raw in RAW_VARIANTS:
                with self.subTest(expected=expected, raw=raw):
                    actual = ev.score_output(raw, expected)
                    self.assertIn("diagnostics", actual)
                    self.assertEqual({key: value for key, value in actual.items()
                                      if key != "diagnostics"}, legacy_score(raw, expected))


class ComparisonDiagnosticsTests(unittest.TestCase):
    def test_lane_counts_are_sparse_case_counts_and_remain_separate(self):
        expected = {"flag": False, "count": 1, "meta": {"ok": True}}
        cases = [{"id": str(index), "expected": expected} for index in range(4)]
        base = [
            '{"flag":false,"count":1,"meta":{"ok":true},"x":[],"z":0}',
            '{"flag":0,"count":1.0,"meta":{"ok":1},"x":{"x":1}}',
            '{"count":1,"count":2}', "[]",
        ]
        candidate = [
            '{"flag":false,"count":1,"meta":{"ok":true}}',
            '{"flag":true,"count":"1","meta":[]}',
            '{"flag":false}', "{}",
        ]
        actual = ev.compare_outputs(cases, base, candidate)
        self.assertEqual(actual["lanes"]["base"]["diagnostics"], {
            "case_count": 4,
            "parse_counts": {"valid": 2, "invalid_json": 1, "non_object": 1},
            "field_counts": {
                "flag": {"match": 1, "type_mismatch": 1, "unavailable": 2},
                "count": {"match": 1, "value_mismatch": 1, "unavailable": 2},
                "meta": {"match": 1, "value_mismatch": 1, "unavailable": 2},
            },
            "unexpected_field_counts": {"x": 2, "z": 1},
        })
        self.assertEqual(actual["lanes"]["candidate"]["diagnostics"], {
            "case_count": 4, "parse_counts": {"valid": 4},
            "field_counts": {
                "flag": {"match": 2, "value_mismatch": 1, "missing": 1},
                "count": {"match": 1, "type_mismatch": 1, "missing": 2},
                "meta": {"match": 1, "type_mismatch": 1, "missing": 2},
            },
            "unexpected_field_counts": {},
        })
        self.assertEqual(without_diagnostics(actual), legacy_comparison(cases, base, candidate))

    def test_heterogeneous_expected_fields_count_only_applicable_cases(self):
        cases = [{"id": "a", "expected": {"alpha": None}},
                 {"id": "b", "expected": {"beta": False}}]
        outputs = ['{"alpha":null,"beta":false}', "{}"]
        actual = ev.compare_outputs(cases, outputs, outputs)
        expected_counts = {
            "case_count": 2, "parse_counts": {"valid": 2},
            "field_counts": {"alpha": {"match": 1}, "beta": {"missing": 1}},
            "unexpected_field_counts": {"beta": 1},
        }
        for lane in actual["lanes"].values():
            self.assertEqual(lane["diagnostics"], expected_counts)

    def test_broad_comparison_scores_hashes_thresholds_and_decisions_are_unchanged(self):
        cases = [{"id": "case-" + str(index),
                  "expected": EXPECTED_VARIANTS[index % len(EXPECTED_VARIANTS)]}
                 for index in range(len(RAW_VARIANTS))]
        good = [legacy_canonical(case["expected"]).decode() for case in cases]
        mixed = list(RAW_VARIANTS)
        pairs = [(good, good), (mixed, good), (good, mixed),
                 (mixed, list(reversed(mixed))), (mixed, mixed)]
        for base, candidate in pairs:
            for pass_rate, improvement in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                                            (0.0, 0.01), (1.0, 1.0)]:
                with self.subTest(base=base is good, candidate=candidate is good,
                                  pass_rate=pass_rate, improvement=improvement):
                    actual = ev.compare_outputs(cases, base, candidate, pass_rate, improvement)
                    self.assertEqual(without_diagnostics(actual),
                                     legacy_comparison(cases, base, candidate, pass_rate, improvement))
                    for lane in actual["lanes"].values():
                        self.assertEqual(lane["diagnostics"]["case_count"], len(cases))
                        self.assertEqual(sum(lane["diagnostics"]["parse_counts"].values()), len(cases))


class ReceiptDiagnosticsTests(unittest.TestCase):
    def run_receipt(self, directory, generator):
        root = Path(directory)
        suite = {"schema": "szl.willay-suite/v1", "cases": [{
            "id": "hold", "messages": [{"role": "user", "content": "Return HOLD."}],
            "expected": {"state": "HOLD", "approved": False},
        }]}
        raw_suite = legacy_canonical(suite)
        suite_path = root / "suite.json"
        suite_path.write_bytes(raw_suite)
        output = root / "receipt.json"
        output.write_text('{"status":"MEASURED","result":"PASS","stale":true}', encoding="utf-8")
        result = ev.main([
            "--suite", str(suite_path), "--suite-sha256", hashlib.sha256(raw_suite).hexdigest(),
            "--model-revision", "1" * 40, "--base-revision", "2" * 40,
            "--output", str(output),
        ], generator)
        raw_receipt = output.read_text(encoding="utf-8")
        receipt = json.loads(raw_receipt)
        claimed = receipt.pop("receipt_sha256")
        self.assertEqual(claimed, hashlib.sha256(legacy_canonical(receipt)).hexdigest())
        self.assertNotIn("stale", receipt)
        self.assertFalse(receipt["publication_eligible"])
        self.assertEqual(receipt["exit_code"], result)
        self.assertEqual(receipt["suite_sha256"], hashlib.sha256(raw_suite).hexdigest())
        self.assertEqual(receipt["model"]["revision"], "1" * 40)
        self.assertEqual(receipt["base"]["revision"], "2" * 40)
        return result, receipt, raw_receipt

    def test_success_and_regression_receipts_keep_legacy_decision_and_raw_hashes(self):
        good = '{"state":"HOLD","approved":false}'
        bad = '{"state":"GO","approved":true}'
        for base, candidate, code, decision in [("{}", good, 0, "PASS"),
                                                 (good, bad, 1, "FAIL")]:
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                def generate(args, cases):
                    return [base], [candidate], {"test_double": True}
                result, receipt, _ = self.run_receipt(directory, generate)
                self.assertEqual(result, code)
                self.assertEqual(receipt["status"], "MEASURED")
                self.assertEqual(receipt["result"], decision)
                keys = ("lanes", "delta_mean_score", "delta_pass_rate", "thresholds", "result")
                actual = {key: receipt[key] for key in keys}
                self.assertEqual(without_diagnostics(actual),
                                 legacy_comparison(receipt["suite"]["cases"], [base], [candidate]))
                for lane in receipt["lanes"].values():
                    self.assertEqual(lane["diagnostics"]["case_count"], 1)
                    self.assertIn("diagnostics", lane["cases"][0])

    def test_unavailable_generator_replaces_stale_success_without_secret_or_fake_counts(self):
        def unavailable(args, cases):
            raise RuntimeError("SYNTHETIC_SECRET_MUST_NOT_APPEAR")
        with tempfile.TemporaryDirectory() as directory:
            result, receipt, raw = self.run_receipt(directory, unavailable)
        self.assertEqual(result, 2)
        self.assertEqual(receipt["status"], "UNKNOWN")
        self.assertEqual(receipt["result"], "UNAVAILABLE")
        self.assertEqual(receipt["failure"], {"phase": "local_generation", "type": "RuntimeError"})
        self.assertNotIn("SYNTHETIC_SECRET_MUST_NOT_APPEAR", raw)
        self.assertNotIn("lanes", receipt)


if __name__ == "__main__":
    unittest.main()
