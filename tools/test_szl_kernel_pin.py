# SPDX-License-Identifier: Apache-2.0
"""Synthetic controls for the kernel pin contract. Not a load or release."""
import unittest

import szl_kernel_pin as M


class PinTests(unittest.TestCase):
    def test_untrusted_org_with_revision_is_evaluation(self):
        out = M.pin({
            "repo_id": "SZLHOLDINGS/szl-invariants",
            "revision": "a" * 40,
            "trust_remote_code": True,
        })
        self.assertEqual(out["floor"], "EVALUATION")
        self.assertFalse(out["trusted_publisher"])
        self.assertFalse(out["production_authorization"])
        self.assertFalse(out["runtime_loaded"])
        self.assertFalse(out["signature_verified"])

    def test_untrusted_without_flag_rejected(self):
        with self.assertRaises(M.PinError) as raised:
            M.pin({"repo_id": "SZLHOLDINGS/szl-invariants", "revision": "a" * 40})
        self.assertEqual(str(raised.exception), "PIN_UNTRUSTED_REQUIRES_FLAG")

    def test_missing_pin_rejected(self):
        with self.assertRaises(M.PinError) as raised:
            M.pin({"repo_id": "SZLHOLDINGS/szl-invariants", "trust_remote_code": True})
        self.assertEqual(str(raised.exception), "PIN_MISSING")

    def test_version_and_revision_together_rejected(self):
        with self.assertRaises(M.PinError) as raised:
            M.pin({
                "repo_id": "SZLHOLDINGS/szl-invariants",
                "version": 1,
                "revision": "a" * 40,
                "trust_remote_code": True,
            })
        self.assertEqual(str(raised.exception), "PIN_BOTH")

    def test_short_revision_rejected(self):
        with self.assertRaises(M.PinError):
            M.pin({
                "repo_id": "SZLHOLDINGS/szl-invariants",
                "revision": "abc",
                "trust_remote_code": True,
            })

    def test_version_zero_rejected(self):
        with self.assertRaises(M.PinError):
            M.pin({
                "repo_id": "SZLHOLDINGS/szl-invariants",
                "version": 0,
                "trust_remote_code": True,
            })

    def test_boolean_version_rejected(self):
        with self.assertRaises(M.PinError):
            M.pin({
                "repo_id": "SZLHOLDINGS/szl-invariants",
                "version": True,
                "trust_remote_code": True,
            })

    def test_authority_flag_forbidden(self):
        with self.assertRaises(M.PinError) as raised:
            M.pin({
                "repo_id": "SZLHOLDINGS/szl-invariants",
                "revision": "a" * 40,
                "trust_remote_code": True,
                "production_authorization": True,
            })
        self.assertEqual(str(raised.exception), "FORBIDDEN_PROMOTION_AUTHORITY")

    def test_runtime_loaded_forbidden(self):
        with self.assertRaises(M.PinError) as raised:
            M.pin({
                "repo_id": "SZLHOLDINGS/szl-invariants",
                "revision": "a" * 40,
                "trust_remote_code": True,
                "runtime_loaded": True,
            })
        self.assertEqual(str(raised.exception), "FORBIDDEN_PROMOTION_RUNTIME")

    def test_version_only_untrusted_is_evaluation(self):
        out = M.pin({
            "repo_id": "SZLHOLDINGS/szl-blocked",
            "version": 1,
            "trust_remote_code": True,
        })
        self.assertEqual(out["version"], 1)
        self.assertIsNone(out["revision"])
        self.assertEqual(out["floor"], "EVALUATION")
        self.assertFalse(out["production_authorization"])


if __name__ == "__main__":
    unittest.main()
