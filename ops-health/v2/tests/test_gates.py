"""M6: behavioral gates (section 10) on DEV candidates; a sign-flipped artifact must fail
g_truth_sign and g_monotone."""

from __future__ import annotations

import unittest
from pathlib import Path

from v2.research import gates, truth
from v2.tests import _support as S

K = S.kernel()


def monotone_of(artifact):
    scorer = K.ArtifactScorer(artifact)
    return gates.g_monotone(lambda f: scorer.advise({"features": f})["operator_attention_score"])


class GateTest(unittest.TestCase):
    def test_grid_is_the_declared_one(self):
        self.assertEqual(len(gates.monotone_points()), 32 * 5 * 5 * 5)
        checks = gates.monotone_checks()
        self.assertEqual(len(checks), 19600)
        kinds = {kind for kind, _b, _c in checks}
        self.assertEqual(len(kinds), 3 + 5)

    def test_good_candidates_pass(self):
        for kind in ("none", "platt", "isotonic"):
            art = S.candidate(calibrator=kind)
            value, report = monotone_of(art)
            with self.subTest(calibrator=kind):
                self.assertEqual(value, 1.0, report)
                self.assertEqual(gates.g_truth_sign(art["weights"]), 1.0)

    def test_sign_flipped_artifact_fails_truth_sign_and_monotone(self):
        for feature in ("consecutive_failures", "queue_utilization", "ledger_integrity_ok",
                        "listener_running"):
            art = S.candidate(calibrator="platt")
            art["weights"][feature] = -art["weights"][feature]
            value, report = monotone_of(art)
            with self.subTest(flipped=feature):
                self.assertEqual(gates.g_truth_sign(art["weights"]), 7 / 8)
                self.assertEqual(truth.sign_agreement(art["weights"])["disagreeing"], [feature])
                self.assertLess(value, 1.0)
                expected_kind = (f"flag_off:{feature}" if feature in gates.HEALTH_FLAGS
                                 else f"raise:{feature}")
                self.assertIn(expected_kind, report["failed_by_kind"])

    def test_decreasing_calibrator_fails_monotone(self):
        art = S.candidate(calibrator="platt")
        art["calibrator"]["params"]["a"] = -abs(art["calibrator"]["params"]["a"])
        value, _report = monotone_of(art)
        self.assertLess(value, 0.5)
        self.assertEqual(gates.g_truth_sign(art["weights"]), 1.0)  # weights untouched

    def test_scalar_gate_formulas(self):
        self.assertEqual(gates.g_validation(0.35), 0.5)
        self.assertEqual(gates.g_validation(0.9), 1.0)
        self.assertEqual(gates.g_validation(None), 0.0)
        self.assertEqual(gates.g_calibration(0.05), 1.0)
        self.assertAlmostEqual(gates.g_calibration(0.10), 0.5, places=15)
        self.assertEqual(gates.g_calibration(0.30), 0.0)
        self.assertEqual(gates.g_calibration(None), 0.0)
        self.assertEqual(gates.g_baseline_valid({"valid": True}), 1.0)
        self.assertEqual(gates.g_baseline_valid({"valid": "yes"}), 0.0)

    def test_gate_vector_is_strict(self):
        full = {name: 1.0 for name in gates.GATE_NAMES}
        self.assertEqual(gates.gate_vector(full), full)
        with self.assertRaises(ValueError):
            gates.gate_vector({k: v for k, v in full.items() if k != "g_monotone"})
        with self.assertRaises(ValueError):
            gates.gate_vector(dict(full, g_integrity=1.5))

    def test_integrity_gate(self):
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "a")
            self.assertEqual(gates.g_integrity(info["model_path"], info["receipt_path"], K), (1.0, None))
            data = bytearray(Path(info["model_path"]).read_bytes())
            data[5] ^= 1
            bad = tmp / "bad.json"
            bad.write_bytes(bytes(data))
            value, message = gates.g_integrity(bad, info["receipt_path"], K)
            self.assertEqual(value, 0.0)
            self.assertIn("hash does not match", message)


if __name__ == "__main__":
    unittest.main()
