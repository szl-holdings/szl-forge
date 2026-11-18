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


if __name__ == "__main__":
    unittest.main()
