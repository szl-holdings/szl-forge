#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic controls for the field-leader bridge. Not estate qualification."""
from __future__ import annotations

import unittest

from szl_field_leader_bridge import (
    BridgeError,
    classify,
    default_bridge,
    refuse_altk_collapse,
    refuse_outrank,
)


class RankTests(unittest.TestCase):
    def test_same_layer_ok(self) -> None:
        row = refuse_outrank("ARTIFACT", "ARTIFACT")
        self.assertTrue(row["admitted"])

    def test_weaker_claim_ok(self) -> None:
        row = refuse_outrank("NARRATIVE", "STATE")
        self.assertTrue(row["admitted"])

    def test_outrank_blocked(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            refuse_outrank("STATE", "NARRATIVE")
        self.assertEqual(str(ctx.exception), "CLAIM_OUTRANKS_EVIDENCE")


class AltkTests(unittest.TestCase):
    def test_gap_kept(self) -> None:
        row = refuse_altk_collapse({
            "mean_at_k": 0.75, "pass_power_k": 0.5, "pass_at_k": 1.0, "as_one_score": False,
        })
        self.assertEqual(row["consistency_gap"], 0.5)
        self.assertFalse(row["collapsed"])

    def test_collapse_forbidden(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            refuse_altk_collapse({
                "mean_at_k": 0.75, "pass_power_k": 0.5, "pass_at_k": 1.0, "as_one_score": True,
            })
        self.assertEqual(str(ctx.exception), "ALTK_COLLAPSE")


class ClassifyTests(unittest.TestCase):
    def test_agi_refused(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            classify({"claim": "make it all AGI"})
        self.assertEqual(str(ctx.exception), "FORBIDDEN_PROMOTION_AGI")

    def test_operational_refused(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            classify({"claim": "everything is fully operational"})
        self.assertEqual(str(ctx.exception), "FORBIDDEN_PROMOTION_OPERATIONAL")

    def test_authorization_flag_refused(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            classify({"production_authorization": True})
        self.assertEqual(str(ctx.exception), "FORBIDDEN_PROMOTION_AUTHORITY")

    def test_live_chip_refused(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            classify({"chip": {"fold": "LIVE", "evidence_layer": "STATE"}})
        self.assertEqual(str(ctx.exception), "FORBIDDEN_PROMOTION_CHIP")

    def test_admit_without_state_is_eval_chip(self) -> None:
        row = classify({"chip": {"fold": "ADMIT_PREDICATES_ONLY", "evidence_layer": "BEHAVIOR"}})
        self.assertEqual(row["chip"]["chip"], "eval")
        self.assertFalse(row["chip"]["live"])
        self.assertFalse(row["production_authorization"])

    def test_admit_with_state_still_not_live(self) -> None:
        row = classify({"chip": {"fold": "ADMIT_PREDICATES_ONLY", "evidence_layer": "STATE"}})
        self.assertEqual(row["chip"]["chip"], "admit")
        self.assertFalse(row["chip"]["production_authorized"])

    def test_evaluator_same_writer_conditional(self) -> None:
        row = classify({
            "evaluator": {
                "collector_id": "same.py",
                "second_reader_id": "same.py",
                "second_reader_network": False,
                "second_reader_can_authorize": False,
            }
        })
        self.assertFalse(row["evaluator"]["independent"])
        self.assertEqual(row["evaluator"]["floor"], "CONDITIONAL")

    def test_evaluator_network_refused(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            classify({
                "evaluator": {
                    "collector_id": "a",
                    "second_reader_id": "b",
                    "second_reader_network": True,
                    "second_reader_can_authorize": False,
                }
            })
        self.assertEqual(str(ctx.exception), "EVAL_NETWORK")

    def test_default_fixture(self) -> None:
        row = default_bridge()
        self.assertEqual(row["schema"], "szl.field-leader-bridge/v1")
        self.assertFalse(row["production_authorization"])
        self.assertFalse(row["agi_claim"])
        self.assertEqual(row["state"], "PARTIAL")
        self.assertEqual(row["chip"]["chip"], "blocked")
        self.assertTrue(row["evaluator"]["independent"])
        self.assertEqual(row["altk"]["consistency_gap"], 0.5)
        self.assertIn("hub_module_present", row["pacing"]["holds"])
        today = row["pacing"]["reportable_today"]
        self.assertEqual(len(today), 6)
        self.assertEqual(today[5]["question"], "Is HAL (2p-1)^2 kept off the ALTK and Λ axes?")
        self.assertEqual(today[5]["answer"], "YES")
        self.assertFalse(row["pacing"]["production_authorized"])
        self.assertFalse(row["pacing"]["agi_claim"])


if __name__ == "__main__":
    unittest.main()
