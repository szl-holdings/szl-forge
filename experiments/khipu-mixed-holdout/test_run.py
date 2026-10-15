"""Small contract checks for the frozen Khipu diagnostic scorer."""

import json
import unittest

import run


def proposal(case):
    cited = case["expected_cited_node_ids"]
    return {
        "planId": "plan-synthetic-test",
        "capabilityProfile": "SZL-Khipu-1.5B-BrainNavigator",
        "provenance": "MODEL_PROPOSED",
        "query": case["query"],
        "contentAccess": "HANDLES_ONLY",
        "candidates": case["candidates"],
        "decision": case["expected_decision"],
        "steps": ([
            {"action": "RETRIEVE", "nodeId": cited[0], "rationale": "Matching metadata"},
            {"action": "CITE", "nodeId": cited[0], "rationale": "Matching metadata"},
        ] if cited else []),
        "citedNodeIds": cited,
        "groundedOnly": True,
        "brainBinding": {"protocol": "khipu-retrieval", "status": "NOT_RESOLVED", "note": "The controller resolves content."},
        "controllerBoundary": "A11oy validates and resolves handles outside the model.",
        "abstainReason": None if cited else "No offered handle supports the query.",
    }


class GradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = run.load_json(run.CASES)["cases"]
        cls.schema = run.load_json(run.SCHEMA)

    def test_navigation_and_abstention_reference_plans_pass(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                result = run.grade(json.dumps(proposal(case)), case, self.schema)
                self.assertTrue(result["case_pass"])

    def test_fabricated_citation_fails(self):
        case = self.cases[0]
        plan = proposal(case)
        plan["citedNodeIds"] = ["node://khipu-synthetic/fabricated"]
        self.assertFalse(run.grade(json.dumps(plan), case, self.schema)["case_pass"])

    def test_mutated_candidates_fail(self):
        case = self.cases[0]
        plan = proposal(case)
        plan["candidates"] = plan["candidates"][::-1]
        self.assertFalse(run.grade(json.dumps(plan), case, self.schema)["contract_valid"])

    def test_over_abstention_fails_navigation(self):
        case = self.cases[0]
        plan = proposal(case)
        plan["decision"] = "ABSTAIN"
        plan["steps"] = []
        plan["citedNodeIds"] = []
        plan["abstainReason"] = "I decline despite a matching handle."
        result = run.grade(json.dumps(plan), case, self.schema)
        self.assertTrue(result["contract_valid"])
        self.assertFalse(result["case_pass"])


if __name__ == "__main__":
    unittest.main()
