import unittest
from tools.szl_math_edge_admission import evaluate, canonical_hash, PINNED_SOURCES


def fixture():
    n = 1000
    base = {
        "model_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "errors": 70,
        "safety_violations": 0,
        "flash_bytes": 8000,
        "ram_bytes": 900,
        "latency_p99_ms": 90.0,
        "energy_j": 0.04,
    }
    cand = dict(
        base,
        model_sha256="d" * 64,
        receipt_sha256="e" * 64,
        errors=50,
        flash_bytes=6000,
        ram_bytes=700,
        latency_p99_ms=70.0,
        energy_j=0.03,
    )
    return {
        "schema": "szl.edge-math-evidence/v1",
        "source_pins": dict(PINNED_SOURCES),
        "scenario_set": {
            "sha256": "c" * 64,
            "heldout_count": n,
            "iid_declared": True,
            "pre_registered": True,
            "independent_evaluator": True,
        },
        "baseline": base,
        "candidate": cand,
        "constraints": {
            "flash_max_bytes": 12000,
            "ram_max_bytes": 1200,
            "latency_p99_max_ms": 100,
            "energy_max_j": 0.06,
            "risk_upper_max": 0.15,
            "delta": 0.05,
        },
        "governance": {
            "rights_approved": True,
            "source_verified": True,
            "weights_verified": True,
            "measured_on_target": True,
            "independent_receipt_verified": True,
            "zero_observed_safety_violations": True,
        },
    }


class SZLMathTests(unittest.TestCase):
    def test_positive_fixture_not_production_proof(self):
        x = evaluate(fixture())
        self.assertEqual(x["state"], "STRUCTURALLY_ADMISSIBLE_UNVERIFIED")
        self.assertFalse(x["production_authorized"])
        self.assertFalse(x["lambda_uniqueness_proven"])
        self.assertFalse(x["proof_refinement"])
        self.assertGreater(x["candidate_safety_rate_upper_hoeffding_conditional"], 0)

    def test_pinned_math_required(self):
        x = fixture()
        x["source_pins"]["szl-papers"] = "f" * 40
        self.assertEqual(evaluate(x)["reasons"], ["SOURCE_PIN_MISMATCH"])

    def test_denominator_matters(self):
        x = fixture()
        x["scenario_set"]["heldout_count"] = 10
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_data_identity_matters(self):
        x = fixture()
        x["candidate"]["dataset_sha256"] = "f" * 64
        self.assertEqual(evaluate(x)["reasons"], ["UNPAIRED_DATASET"])

    def test_rights_veto(self):
        x = fixture()
        x["governance"]["rights_approved"] = False
        self.assertEqual(evaluate(x)["reasons"], ["GOVERNANCE_HOLD"])

    def test_energy_regression_veto(self):
        x = fixture()
        x["candidate"]["energy_j"] = 0.05
        self.assertEqual(evaluate(x)["reasons"], ["NOT_PARETO_NONREGRESSIVE"])

    def test_latency_budget_veto(self):
        x = fixture()
        x["candidate"]["latency_p99_ms"] = 110
        self.assertEqual(evaluate(x)["reasons"], ["PERFORMANCE_BUDGET_EXCEEDED"])

    def test_flash_budget_veto(self):
        x = fixture()
        x["candidate"]["flash_bytes"] = 13000
        self.assertEqual(evaluate(x)["reasons"], ["MEMORY_BUDGET_EXCEEDED"])

    def test_safety_veto(self):
        x = fixture()
        x["candidate"]["safety_violations"] = 1
        self.assertEqual(evaluate(x)["reasons"], ["OBSERVED_SAFETY_VIOLATION"])

    def test_risk_upper_veto(self):
        x = fixture()
        x["candidate"]["errors"] = 150
        self.assertEqual(evaluate(x)["reasons"], ["RISK_UPPER_BOUND_EXCEEDED"])

    def test_non_iid_hold(self):
        x = fixture()
        x["scenario_set"]["iid_declared"] = False
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_bool_not_a_count(self):
        x = fixture()
        x["candidate"]["errors"] = True
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_nan_hold(self):
        x = fixture()
        x["candidate"]["energy_j"] = float("nan")
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_nonmapping_hold(self):
        self.assertEqual(evaluate(None)["state"], "HOLD")

    def test_missing_field_hold(self):
        x = fixture()
        del x["baseline"]["energy_j"]
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_deterministic_hash_key_order(self):
        x = fixture()
        self.assertEqual(
            canonical_hash(x), canonical_hash(dict(reversed(list(x.items()))))
        )

    def test_no_improvement_hold(self):
        x = fixture()
        x["candidate"].update(x["baseline"])
        self.assertEqual(evaluate(x)["reasons"], ["NO_MEASURED_IMPROVEMENT"])

    def test_min_subnormal_delta_is_held_not_raised(self):
        x = fixture()
        x["constraints"]["delta"] = 5e-324  # delta / 2 underflows to 0.0
        self.assertEqual(evaluate(x)["reasons"], ["RISK_UPPER_BOUND_EXCEEDED"])

    def test_risk_bounds_match_documented_formula(self):
        # Wanted values are U = errors/n + sqrt(log(2/delta)/(2n)) evaluated in
        # 60-digit decimal arithmetic with errors=50 (0 for safety), n=10**6.
        # For delta = 2**-k the log is exactly (k+1)*ln 2, so subnormal delta has
        # an exact reference: the bound must not saturate to 1.0 there.
        table = (
            # delta, risk upper, safety-rate upper
            (5e-324, 0.01935198460135565, 0.01930198460135565),
            (1e-323, 0.01934300484529796, 0.019293004845297963),
            (2.2250738585072014e-308, 0.01887935959761808, 0.018829359597618077),
            (0.5, 0.0008825546111576977, 0.0008325546111576978),
            (0.05, 0.0014081015157406194, 0.0013581015157406195),
        )
        for delta, want_risk, want_safety in table:
            with self.subTest(delta=delta):
                x = fixture()
                x["scenario_set"]["heldout_count"] = 1_000_000
                x["constraints"]["delta"] = delta
                got = evaluate(x)
                self.assertEqual(got["state"], "STRUCTURALLY_ADMISSIBLE_UNVERIFIED")
                self.assertAlmostEqual(
                    got["candidate_risk_upper_hoeffding_conditional"],
                    want_risk,
                    places=15,
                )
                self.assertAlmostEqual(
                    got["candidate_safety_rate_upper_hoeffding_conditional"],
                    want_safety,
                    places=15,
                )

    def test_int_beyond_float_range_is_held_not_raised(self):
        huge = 10**400  # a valid JSON integer that cannot convert to a float
        table = (
            ("constraints", "flash_max_bytes", huge, "CONSTRAINTS_INVALID"),
            ("constraints", "delta", huge, "CONSTRAINTS_INVALID"),
            ("constraints", "energy_max_j", -huge, "CONSTRAINTS_INVALID"),
            ("candidate", "energy_j", huge, "INVALID_MEASUREMENT_CANDIDATE"),
            ("baseline", "latency_p99_ms", huge, "INVALID_MEASUREMENT_BASELINE"),
        )
        for section, key, value, want in table:
            with self.subTest(section=section, key=key):
                x = fixture()
                x[section][key] = value
                self.assertEqual(evaluate(x)["reasons"], [want])

    def test_every_field_value_yields_a_decision(self):
        # evaluate() is a gate: whatever JSON-shaped value lands in any field,
        # it must return a decision, never raise, and never authorize production.
        hostile = (
            None, True, False, "", "x", [], {}, 0, -1, 1, 2**63, 2**1024,
            10**400, -(10**400), 0.0, -0.0, 5e-324, 1e-323,
            2.2250738585072014e-308, 0.5, 1.0, 1.7976931348623157e308,
            float("inf"), float("-inf"), float("nan"),
        )  # fmt: skip
        states = ("HOLD", "STRUCTURALLY_ADMISSIBLE_UNVERIFIED")
        for section, body in fixture().items():
            # None targets the whole section; the rest target each field in it.
            for key in [None] + (list(body) if type(body) is dict else []):
                for value in hostile:
                    x = fixture()
                    if key is None:
                        x[section] = value
                    else:
                        x[section][key] = value
                    with self.subTest(section=section, key=key, value=repr(value)[:24]):
                        got = evaluate(x)
                        self.assertIn(got["state"], states)
                        self.assertIs(got["production_authorized"], False)


if __name__ == "__main__":
    unittest.main()
