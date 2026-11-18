import unittest

from tools.edge_nn_budget_gate import assess


def example():
    return {
        "schema": "szl.edge-nn-budget-input/v1",
        "layers": [32, 32, 16, 8, 4],
        "device": {
            "flash_bytes": 100000,
            "ram_bytes": 4096,
            "clock_hz": 16000000,
            "cycles_per_mac_upper": 400,
        },
        "timing": {"decision_deadline_ms": 200, "non_mac_cycles_upper": 10000},
        "artifact": {
            "weight_bits": 8,
            "code_bytes": 3000,
            "activation_bytes_per_value": 1,
            "scratch_bytes": 512,
            "weight_sha256": "a" * 64,
        },
        "accuracy": {
            "baseline_correct": 450,
            "candidate_correct": 460,
            "heldout_count": 512,
            "paired_and_independent": True,
        },
    }


class EdgeBudgetTests(unittest.TestCase):
    def test_exact_reference_topology(self):
        r = assess(example())
        self.assertEqual(r["macs"], 1696)
        self.assertEqual(r["parameters"], 1756)
        self.assertEqual(r["packed_weights_bytes"], 1756)
        self.assertEqual(r["state"], "STRUCTURALLY_FEASIBLE_UNVERIFIED")
        self.assertFalse(r["physical_hardware_tested"])

    def test_float32_reference_storage(self):
        x = example()
        x["artifact"]["weight_bits"] = 32
        self.assertEqual(assess(x)["packed_weights_bytes"], 7024)

    def test_flash_hold(self):
        x = example()
        x["device"]["flash_bytes"] = 100
        self.assertIn("FLASH_BUDGET_EXCEEDED", assess(x)["reasons"])

    def test_ram_hold(self):
        x = example()
        x["device"]["ram_bytes"] = 1
        self.assertIn("RAM_BUDGET_EXCEEDED", assess(x)["reasons"])

    def test_deadline_hold(self):
        x = example()
        x["timing"]["decision_deadline_ms"] = 1
        self.assertIn("LATENCY_BUDGET_EXCEEDED", assess(x)["reasons"])

    def test_accuracy_regression_hold(self):
        x = example()
        x["accuracy"]["candidate_correct"] = 400
        self.assertIn("ACCURACY_REGRESSION", assess(x)["reasons"])

    def test_unpaired_hold(self):
        x = example()
        x["accuracy"]["paired_and_independent"] = False
        self.assertIn("ACCURACY_EVIDENCE_INSUFFICIENT", assess(x)["reasons"])

    def test_missing_weight_hash(self):
        x = example()
        x["artifact"]["weight_sha256"] = "unknown"
        self.assertIn("INVALID_ARTIFACT", assess(x)["reasons"])

    def test_nan_deadline(self):
        x = example()
        x["timing"]["decision_deadline_ms"] = float("nan")
        self.assertIn("INVALID_TIMING", assess(x)["reasons"])

    def test_boolean_device_rejected(self):
        x = example()
        x["device"]["clock_hz"] = True
        self.assertIn("INVALID_DEVICE_BUDGET", assess(x)["reasons"])

    def test_extra_key_rejected(self):
        x = example()
        x["secret"] = "do not accept"
        self.assertEqual(assess(x)["reasons"], ["INVALID_SCHEMA"])

    def test_invalid_layers(self):
        x = example()
        x["layers"] = [32]
        self.assertIn("INVALID_TOPOLOGY", assess(x)["reasons"])

    def test_zero_samples_hold(self):
        x = example()
        x["accuracy"]["heldout_count"] = 0
        self.assertIn("ACCURACY_EVIDENCE_INSUFFICIENT", assess(x)["reasons"])


if __name__ == "__main__":
    unittest.main()
