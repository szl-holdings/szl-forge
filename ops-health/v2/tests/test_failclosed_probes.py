"""M6: fail-closed probes against the v2 kernel (CONTRACT section 2.6 classes plus v2 variants).

Probe payloads use operational field names and prohibited field NAMES with dummy numeric
values only.  Every probe must be refused with the expected message.
"""

from __future__ import annotations

import copy
import io
import json
import math
import unittest
from contextlib import redirect_stderr, redirect_stdout

from v2.research import gates
from v2.tests import _support as S

K = S.kernel()
CONTRACT_2_6_NAME_FRAGMENTS = ("patient", "hl7", "fhir", "specimen", "order", "result", "mrn")


class FailClosedProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = K.ArtifactScorer(S.candidate())

    def test_valid_input_is_accepted_with_all_false_authority(self):
        out = self.scorer.advise({"features": dict(gates.VALID_FEATURES)})
        self.assertIn(out["advisory"], ("ALERT", "NO_ALERT", "ABSTAIN"))
        self.assertEqual(
            out["authority"],
            {
                "acknowledgement": False,
                "clinical_decision": False,
                "device_control": False,
                "result_interpretation": False,
                "result_release": False,
            },
        )
        self.assertEqual(
            set(out),
            {
                "schema", "advisory", "abstain_reason", "prediction_set",
                "operator_attention_score", "threshold_decision", "decision_threshold",
                "score_semantics", "purpose", "authority", "normalized_feature_contributions",
                "coverage_statement",
            },
        )
        self.assertEqual(list(out["normalized_feature_contributions"]), list(K.FEATURE_NAMES))

    def test_exactly_eight_fields(self):
        self.assertEqual(len(K.FEATURE_NAMES), 8)
        self.assertEqual(K.FEATURE_NAMES, tuple(gates.VALID_FEATURES))
        for name in K.FEATURE_NAMES:
            payload = {"features": {k: v for k, v in gates.VALID_FEATURES.items() if k != name}}
            with self.subTest(missing=name):
                with self.assertRaises(K.ModelInputError) as ctx:
                    self.scorer.advise(payload)
                self.assertIn(f"missing=['{name}']", str(ctx.exception))

    def test_every_probe_is_refused_with_expected_message(self):
        self.assertGreaterEqual(len(gates.FAILCLOSED_PROBES), 60)
        for probe_id, payload, fragment in gates.FAILCLOSED_PROBES:
            with self.subTest(probe=probe_id):
                with self.assertRaises(K.ModelInputError) as ctx:
                    self.scorer.advise(copy.deepcopy(payload))
                self.assertIn(fragment, str(ctx.exception))

    def test_contract_2_6_probe_classes_are_present(self):
        ids = {p[0] for p in gates.FAILCLOSED_PROBES}
        for required in ("missing_field", "extra_field", "nan_continuous", "inf_continuous",
                         "out_of_range_high", "out_of_range_low", "text_instead_of_number"):
            self.assertIn(required, ids)
        extra_names = set()
        for _pid, payload, _frag in gates.FAILCLOSED_PROBES:
            if isinstance(payload, dict) and isinstance(payload.get("features"), dict):
                extra_names |= set(map(str, payload["features"])) - set(K.FEATURE_NAMES)
        for fragment in CONTRACT_2_6_NAME_FRAGMENTS:
            with self.subTest(fragment=fragment):
                hits = [n for n in extra_names if fragment in K.normalized_key(n)]
                self.assertTrue(hits, f"no probe field name contains {fragment!r}")
        # case, hyphen, space and substring variants are all exercised
        self.assertTrue(any(n != n.lower() for n in extra_names))
        self.assertTrue(any("-" in n for n in extra_names))
        self.assertTrue(any(" " in n.strip() for n in extra_names))
        self.assertTrue(any(n != n.strip() for n in extra_names))
        kinds = {K.prohibited_match(n) for n in extra_names}
        self.assertTrue({"exact", "token", "substring"} <= kinds)
        self.assertIn("prohibited_replaces_field", ids)

    def test_prohibited_screen_runs_before_schema_check(self):
        # A prohibited name that replaces a required field is reported as prohibited, not as a
        # schema mismatch (the screen runs first).
        for replaced, name in (("queue_utilization", "patient_id"), ("tls_enabled", "HL7"),
                               ("consecutive_failures", "Order-ID"), ("listener_running", "result value"),
                               ("configuration_valid", "xMRNx"), ("ledger_integrity_ok", "FhirPort")):
            features = {k: v for k, v in gates.VALID_FEATURES.items() if k != replaced}
            features[name] = 0
            with self.subTest(name=name):
                with self.assertRaises(K.ModelInputError) as ctx:
                    self.scorer.advise({"features": features})
                self.assertIn("prohibited non-operational field", str(ctx.exception))
                self.assertNotIn("schema mismatch", str(ctx.exception))

    def test_match_kinds(self):
        cases = {
            "patient": "exact", "PATIENT": "exact", " Patient-ID ": "exact", "order id": "exact",
            "Result-Value": "exact", "source_hl7_port": "token", "last-order-count": "token",
            "queue name": "token", "dob_year": "token", "operator_email": "token",
            "PatientCount": "substring", "xhl7x": "substring", "fhirendpoint": "substring",
            "specimens": "substring", "mrnhash": "substring",
        }
        for key, kind in cases.items():
            with self.subTest(key=key):
                self.assertEqual(K.prohibited_match(key), kind)
        for key in K.FEATURE_NAMES + ("sortorder", "resultant", "uptime_seconds"):
            with self.subTest(allowed=key):
                self.assertIsNone(K.prohibited_match(key))

    def test_gate_counts_every_probe(self):
        value, report = gates.g_failclosed(self.scorer.advise)
        self.assertEqual(value, 1.0)
        self.assertEqual(report["escaped"], [])
        self.assertEqual(report["refused"], len(gates.FAILCLOSED_PROBES))

    def test_cli_refuses_with_exit_code_2(self):
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "art")
            cases = {
                "nan_constant": '{"features": {"listener_running": 1, "tls_enabled": 1, '
                '"peer_allowlist_configured": 1, "queue_utilization": NaN, "consecutive_failures": 0, '
                '"seconds_since_last_success": 14.0, "ledger_integrity_ok": 1, "configuration_valid": 1}}',
                "duplicate_key": '{"features": {"listener_running": 1, "listener_running": 1}}',
                "prohibited": json.dumps({"features": {**gates.VALID_FEATURES, "specimen_id": 0}}),
                "not_json": "{",
            }
            for name, text in cases.items():
                path = tmp / f"{name}.json"
                path.write_text(text, encoding="utf-8")
                err, out = io.StringIO(), io.StringIO()
                with self.subTest(case=name), redirect_stderr(err), redirect_stdout(out):
                    code = K.main(["--model", info["model_path"], "--receipt", info["receipt_path"],
                                   "--input", str(path)])
                self.assertEqual(code, 2, name)
                self.assertEqual(out.getvalue(), "")
                self.assertFalse(json.loads(err.getvalue())["ok"])
            ok_path = tmp / "ok.json"
            ok_path.write_text(json.dumps({"features": gates.VALID_FEATURES}), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = K.main(["--model", info["model_path"], "--receipt", info["receipt_path"],
                               "--input", str(ok_path)])
            self.assertEqual(code, 0)
            result = json.loads(out.getvalue())
            self.assertTrue(math.isfinite(result["operator_attention_score"]))
            self.assertTrue(all(v is False for v in result["authority"].values()))


if __name__ == "__main__":
    unittest.main()
