# SPDX-License-Identifier: Apache-2.0
"""Khipu training-sanity and held-out-eval cross-field parity, offline only."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "khipu"))
import eval_khipu  # noqa: E402
import sanity_gate  # noqa: E402

O1 = "node://khipu-synthetic/1"
O2 = "node://khipu-synthetic/2"
OFFERED = {O1, O2}


def valid_plan():
    return {
        "decision": "NAVIGATE",
        "candidates": [{"nodeId": O1}, {"nodeId": O2}],
        "steps": [{"action": "RETRIEVE", "nodeId": O1},
                  {"action": "CITE", "nodeId": O1}],
        "citedNodeIds": [O1],
        "abstainReason": None,
    }


class TrainingEvaluationParity(unittest.TestCase):
    def assert_both(self, plan, expected, label):
        eval_ok = eval_khipu.cross_field_ok(plan, OFFERED)
        sanity_ok, reason = sanity_gate.cross_field_ok(plan, OFFERED)
        self.assertEqual(eval_ok, expected, "eval/" + label)
        self.assertEqual(sanity_ok, expected, "sanity/" + label + "/" + reason)

    def test_valid_navigation_and_abstention_match(self):
        p = valid_plan()
        self.assert_both(p, True, "good navigate")
        p.update({"decision": "ABSTAIN", "steps": [], "citedNodeIds": [],
                  "abstainReason": "no offered handle supports the query"})
        self.assert_both(p, True, "good abstain")

    def test_adversarial_subcases_are_rejected_by_both_gates(self):
        cases = [
            ("ABSTAIN_WITH_RETRIEVE_STEP",
             lambda p: p.update(decision="ABSTAIN", citedNodeIds=[],
                                steps=[{"action": "RETRIEVE", "nodeId": O1}],
                                abstainReason="no evidence")),
            ("ABSTAIN_WITH_EXPAND_STEP",
             lambda p: p.update(decision="ABSTAIN", citedNodeIds=[],
                                steps=[{"action": "EXPAND", "nodeId": O1}],
                                abstainReason="no evidence")),
            ("ABSTAIN_WITH_WHITESPACE_REASON",
             lambda p: p.update(decision="ABSTAIN", citedNodeIds=[], steps=[],
                                abstainReason="   ")),
            ("DUPLICATE_CITED_ID", lambda p: p.update(citedNodeIds=[O1, O1])),
            ("DUPLICATE_CITE_ACTION", lambda p: p["steps"].append(
                {"action": "CITE", "nodeId": O1})),
            ("DUPLICATE_CANDIDATE_ID", lambda p: p["candidates"].append(
                {"nodeId": O1})),
            ("OFFERED_SET_BYPASS", lambda p: p["candidates"].append(
                {"nodeId": "node://fabricated"})),
            ("NAVIGATE_WITH_ABSTENTION_REASON",
             lambda p: p.update(abstainReason="No source")),
            ("NAVIGATE_WITHOUT_CITATIONS", lambda p: p.update(citedNodeIds=[])),
        ]
        for name, mutation in cases:
            with self.subTest(case=name):
                p = valid_plan()
                mutation(p)
                self.assert_both(p, False, name)

    def test_permuted_valid_candidate_order_does_not_change_admission(self):
        p = valid_plan()
        p["candidates"].reverse()
        self.assert_both(p, True, "permutation-invariant")


if __name__ == "__main__":
    unittest.main()
