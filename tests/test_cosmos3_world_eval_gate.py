import copy
import unittest

from tools.cosmos3_world_eval_gate import evaluate

S40 = "a" * 40
S64 = "b" * 64


def sample():
    return {
        "schema": "szl.world-eval-input/v1",
        "model": {"id": "nvidia/Cosmos3-Edge", "revision": S40, "sha256": S64},
        "source": {"id": "szl-holdings/szl-forge", "revision": S40, "sha256": S64},
        "dataset": {"id": "owner-approved-synthetic", "revision": S40, "sha256": S64},
        "rights": {
            "approved": True,
            "scope": "EVALUATION_ONLY",
            "evidence_sha256": S64,
        },
        "baseline": {"model_revision": S40, "adapter_sha256": S64},
        "candidate": {"model_revision": S40, "adapter_sha256": S64},
        "evaluation": {
            "protocol_sha256": S64,
            "heldout": True,
            "independent": True,
            "synthetic_labeled": True,
            "hardware_observed": True,
        },
        "scenarios": [
            {
                "id": f"scenario-{i}",
                "seed": i,
                "split": "heldout",
                "baseline_physics_error": 0.4,
                "candidate_physics_error": 0.3,
                "baseline_success": False,
                "candidate_success": True,
                "safety_violation": False,
            }
            for i in range(30)
        ],
    }


class GateTests(unittest.TestCase):
    def test_synthetic_paired_fixture_only_candidate_not_qualified(self):
        r = evaluate(sample())
        self.assertEqual(r["state"], "STRUCTURALLY_ADMISSIBLE_UNVERIFIED")
        self.assertFalse(r["production_qualified"])
        self.assertFalse(r["robot_deployment_authorized"])

    def test_unpinned_model(self):
        x = sample()
        x["model"]["revision"] = "main"
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_unapproved_rights(self):
        x = sample()
        x["rights"]["approved"] = False
        self.assertIn("RIGHTS_UNVERIFIED", evaluate(x)["reasons"])

    def test_nonindependent(self):
        x = sample()
        x["evaluation"]["independent"] = False
        self.assertEqual(evaluate(x)["state"], "HOLD")

    def test_too_few(self):
        x = sample()
        x["scenarios"] = x["scenarios"][:2]
        self.assertIn("INSUFFICIENT_SAMPLE_SIZE", evaluate(x)["reasons"])

    def test_duplicate_pair(self):
        x = sample()
        x["scenarios"][1] = copy.deepcopy(x["scenarios"][0])
        self.assertIn("INVALID_OR_DUPLICATE_PAIR", evaluate(x)["reasons"])

    def test_safety_failure(self):
        x = sample()
        x["scenarios"][0]["safety_violation"] = True
        self.assertIn("SAFETY_VIOLATION", evaluate(x)["reasons"])

    def test_physics_regression(self):
        x = sample()
        for row in x["scenarios"]:
            row["candidate_physics_error"] = 1
        self.assertIn("PHYSICAL_CONSISTENCY_REGRESSION", evaluate(x)["reasons"])

    def test_success_regression(self):
        x = sample()
        for row in x["scenarios"]:
            row["baseline_success"] = True
            row["candidate_success"] = False
        self.assertIn("TASK_SUCCESS_REGRESSION", evaluate(x)["reasons"])

    def test_nan_not_admitted(self):
        x = sample()
        x["scenarios"][0]["candidate_physics_error"] = float("nan")
        self.assertIn("INVALID_MEASUREMENTS", evaluate(x)["reasons"])

    def test_extra_field_rejected(self):
        x = sample()
        x["secret"] = "should-not-be-here"
        self.assertEqual(evaluate(x)["reasons"], ["INVALID_SCHEMA"])

    def test_malformed_model_holds_without_crash(self):
        x = sample()
        x["model"] = {"id": "nvidia/Cosmos3-Edge"}
        self.assertIn("INVALID_MODEL", evaluate(x)["reasons"])

    def test_malformed_candidate_holds_without_crash(self):
        x = sample()
        x["candidate"] = {}
        self.assertIn("INVALID_CANDIDATE", evaluate(x)["reasons"])

    def test_nonmapping_record_holds(self):
        self.assertEqual(evaluate(None)["state"], "HOLD")

    def test_bool_not_numeric(self):
        x = sample()
        x["scenarios"][0]["candidate_physics_error"] = True
        self.assertIn("INVALID_MEASUREMENTS", evaluate(x)["reasons"])

    def test_not_heldout(self):
        x = sample()
        x["scenarios"][0]["split"] = "train"
        self.assertIn("INVALID_OR_DUPLICATE_PAIR", evaluate(x)["reasons"])


if __name__ == "__main__":
    unittest.main()
