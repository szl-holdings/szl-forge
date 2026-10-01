"""Inert precomputed-output regressions; no provider or model execution."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refusal_baseline_fixture", ROOT / "tests/test_refusal_assertion_baseline.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Admitted baseline test helpers unavailable")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
M = BASE.M


class ResultsEvidenceTests(unittest.TestCase):
    """Precomputed observations cannot be silently coerced or overwritten."""
    setUp = BASE.BaselineAssertionTests.setUp
    put_baseline = BASE.BaselineAssertionTests.put_baseline
    put_results = BASE.BaselineAssertionTests.put_results
    invoke = BASE.BaselineAssertionTests.invoke

    def rows(self):
        return [{"id": p["id"], "output": "Unable to complete this synthetic task."}
                for p in M.PROBES]

    def write(self, rows):
        self.results.write_text(json.dumps({"results": rows}), encoding="utf-8")

    def rejected(self):
        before = self.base.read_bytes()
        with patch.object(M, "run_transformers_cpu", side_effect=AssertionError("no fallback")) as generate:
            code, out, err = self.invoke()
        self.assertEqual(code, 1, (out, err))
        self.assertNotIn("did not regress", out)
        self.assertIn("results evidence invalid or unavailable", err)
        self.assertEqual(self.base.read_bytes(), before)
        generate.assert_not_called()

    def test_list_and_wrapped_list_remain_supported(self):
        for data in (self.rows(), {"results": self.rows(), "fixture": "inert context"}):
            self.results.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(self.invoke()[0], 0)

    def test_duplicate_ids_cannot_replace_an_earlier_failed_answer(self):
        rows = self.rows(); rows.insert(0, {"id": "rf-001", "output": "Synthetic nonrefusal."})
        self.write(rows); self.rejected()

    def test_non_text_outputs_are_not_stringified_into_refusals(self):
        for bad in ({"not_a_completion": "I cannot"}, ["I cannot"], True, 1, None):
            rows = self.rows()
            for row in rows: row["output"] = bad
            self.write(rows)
            with self.subTest(type=type(bad).__name__): self.rejected()

    def test_missing_output_is_not_an_observed_empty_completion(self):
        rows = self.rows(); del rows[0]["output"]
        self.write(rows); self.rejected()

    def test_legitimate_empty_text_is_scored_as_nonrefusal(self):
        rows = self.rows(); rows[0]["output"] = ""
        self.write(rows); self.assertEqual(self.invoke()[0], 0)
        for row in rows: row["output"] = ""
        self.write(rows); self.assertEqual(self.invoke()[0], 1)

    def test_extra_unknown_probe_is_not_ignored(self):
        rows=self.rows();rows.append({"id":"unknown", "output":"I cannot"})
        self.write(rows); self.rejected()

    def test_extra_malformed_row_is_not_ignored(self):
        rows=self.rows();rows.append(None)
        self.write(rows); self.rejected()

    def test_missing_probe_is_rejected_with_fixed_diagnostic(self):
        self.write(self.rows()[:-1]); self.rejected()

    def test_non_string_ids_are_not_coerced(self):
        for bad in (None, True, 1, [], {}):
            rows=self.rows();rows[0]["id"]=bad;self.write(rows)
            with self.subTest(type=type(bad).__name__): self.rejected()

    def test_duplicate_json_fields_are_not_last_value_wins(self):
        self.write(self.rows())
        raw=self.results.read_text().replace('"output":', '"output":"Synthetic nonrefusal.","output":', 1)
        self.results.write_text(raw);self.rejected()

    def test_nonfinite_extra_metadata_is_rejected(self):
        for token in ('NaN','Infinity','-Infinity','1e999'):
            raw=json.dumps({"results":self.rows()})[:-1]+',"metadata":'+token+'}'
            self.results.write_text(raw)
            with self.subTest(token=token): self.rejected()

    def test_invalid_utf8_and_surrogates_are_rejected(self):
        self.results.write_bytes(b'\xff');self.rejected()
        rows=self.rows();rows[0]["output"]='\ud800';self.write(rows);self.rejected()

    def test_empty_and_nonlist_payloads_are_rejected(self):
        for raw in (b'', b'null', b'{}', b'[]', b'{"results":{}}', b'{"results":true}'):
            self.results.write_bytes(raw)
            with self.subTest(raw=raw): self.rejected()

    def test_input_and_output_bounds_are_enforced(self):
        self.results.write_bytes(b' '*(4*1024*1024+1));self.rejected()
        rows=self.rows();rows[0]["output"]='x'*(256*1024+1);self.write(rows);self.rejected()

    def test_missing_file_does_not_echo_local_path(self):
        self.results=self.root/'PRIVATE_RESULTS_PATH.json'
        self.rejected()
        self.assertNotIn('PRIVATE_RESULTS_PATH', self.invoke()[2])

    def test_extra_row_metadata_is_inert_not_a_permission(self):
        rows=self.rows()
        for row in rows:row.update(verified=True, production=True)
        self.write(rows)
        code,out,_=self.invoke(asserted=False)
        self.assertEqual(code,0);self.assertNotIn('"verified"',out);self.assertNotIn('"production"',out)

    def test_results_digest_identifies_exact_supplied_bytes(self):
        import hashlib
        self.write(self.rows())
        self.assertIn(hashlib.sha256(self.results.read_bytes()).hexdigest(), self.invoke()[1])

    def test_results_and_baseline_are_never_rewritten(self):
        self.write(self.rows())
        before=(self.base.read_bytes(),self.results.read_bytes())
        self.assertEqual(self.invoke()[0],0)
        self.assertEqual((self.base.read_bytes(),self.results.read_bytes()),before)


if __name__ == "__main__":
    unittest.main()
