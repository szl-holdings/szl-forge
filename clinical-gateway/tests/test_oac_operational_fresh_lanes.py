"""Fresh-seed and distribution-shift lanes: small, fast versions of the CLI lanes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "evaluate_operational_health_model.py"
SPEC = importlib.util.spec_from_file_location("oac_operational_evaluation_lanes", TOOL)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
SOURCE = (ROOT / evaluation.TRAINER_FILE).read_bytes()


def kernel():
    return evaluation.OperationalHealthKernel(ROOT / evaluation.MODEL_FILE, ROOT / evaluation.MODEL_RECEIPT_FILE)


class FreshSeedLaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluation.fresh_seed_lanes(SOURCE, kernel(), 0.16, rows=600, replicates=40)
        cls.lanes = cls.report["lanes"]

    def test_all_lanes_present_with_declared_shift_kinds(self) -> None:
        self.assertEqual({"fresh_seed_in_distribution": "none", "covariate_queue_saturation": "covariate",
                          "covariate_long_outage_tail": "covariate", "covariate_fault_heavy": "covariate",
                          "concept_tls_weight_tripled": "concept"},
                         {name: lane["shift_kind"] for name, lane in self.lanes.items()})

    def test_fresh_seeds_never_reuse_the_development_seed(self) -> None:
        self.assertNotIn(2500, {lane["seed"] for lane in self.lanes.values()})

    def test_queue_saturation_flags_every_row_and_so_does_the_oracle(self) -> None:
        # Every saturated row has true risk above 0.16, so flagging all of them is the
        # correct decision at this threshold. Balanced accuracy 0.5 here is a property of
        # the metric under this shift, not a model or threshold defect.
        lane = self.lanes["covariate_queue_saturation"]
        self.assertEqual(1.0, lane["flag_rate"])
        self.assertEqual(1.0, lane["lane_oracle_at_threshold"]["flag_rate"])
        self.assertEqual(lane["lane_oracle_at_threshold"]["balanced_accuracy"], lane["balanced_accuracy"]["value"])
        self.assertGreater(lane["roc_auc"]["value"], 0.7)

    def test_long_outage_lane_underflags_relative_to_the_oracle(self) -> None:
        lane = self.lanes["covariate_long_outage_tail"]
        self.assertLess(lane["flag_rate"], lane["lane_oracle_at_threshold"]["flag_rate"])

    def test_headroom_is_measured_against_each_lanes_own_rule(self) -> None:
        for name, lane in self.lanes.items():
            with self.subTest(lane=name):
                self.assertAlmostEqual(lane["lane_oracle_roc_auc"] - lane["roc_auc"]["value"],
                                       lane["roc_auc_headroom"]["value"], places=11)

    def test_lanes_are_deterministic(self) -> None:
        self.assertEqual(self.report, evaluation.fresh_seed_lanes(SOURCE, kernel(), 0.16, rows=600, replicates=40))

    def test_default_report_does_not_run_lanes(self) -> None:
        self.assertNotIn("fresh_seed_lanes", evaluation.evaluate())


class FreshLaneMutationTests(unittest.TestCase):
    def test_generator_without_feature_sampler_fails_closed(self) -> None:
        mutant = SOURCE.replace(b"def _generate_features", b"def _renamed_features", 1)
        with self.assertRaisesRegex(evaluation.EvaluationError, "GENERATOR_LABEL_RULE_UNAVAILABLE"):
            evaluation.fresh_seed_lanes(mutant, kernel(), 0.16, rows=50, replicates=2)

    def test_lanes_follow_the_generator_label_rule(self) -> None:
        mutant = SOURCE.replace(b"-3.6\n", b"-9.6\n", 1)
        self.assertNotEqual(SOURCE, mutant)
        base = evaluation.fresh_seed_lanes(SOURCE, kernel(), 0.16, rows=300, replicates=2)
        with self.assertRaisesRegex(evaluation.EvaluationError, "FRESH_LANE_SINGLE_CLASS"):
            evaluation.fresh_seed_lanes(mutant, kernel(), 0.16, rows=300, replicates=2)
        self.assertGreater(base["lanes"]["fresh_seed_in_distribution"]["positive_rows"], 0)


if __name__ == "__main__":
    unittest.main()
