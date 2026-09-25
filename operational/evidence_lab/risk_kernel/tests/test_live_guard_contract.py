"""Live guard comparison must reject behavior or rejected-span drift."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unresolved_identifier_gate_v3 import gate_decision, validate_live_gate


class LiveGuardContractTests(unittest.TestCase):
    def fixture(self, question):
        base = {"status": "ANSWER", "answer": "Paris", "evidence": {"start": 0, "end": 5}}
        expected = gate_decision(question, base, set())
        result = {**base, "status": expected["status"], "answer": expected["answer"],
                  "identifier_guard": {"enabled": True, "reason": expected["reason"],
                                       "unresolved_identifiers": expected["unresolved_identifiers"],
                                       "base_status": base["status"]}}
        if expected["status"] == "ABSTAIN":
            result["evidence"] = None
        return base, expected, result

    def test_accepts_true_live_abstention(self):
        base, expected, result = self.fixture("Who is UnknownSubjectIdentifier012345?")
        validate_live_gate(result, base, expected)

    def test_rejects_answer_reason_or_detection_drift(self):
        base, expected, result = self.fixture("Who is UnknownSubjectIdentifier012345?")
        for key, value in (("status", "ANSWER"), ("answer", "Paris"), ("evidence", base["evidence"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_live_gate({**result, key: value}, base, expected)
        for key, value in (("enabled", False), ("reason", "different"), ("unresolved_identifiers", []), ("base_status", "ABSTAIN")):
            broken = copy.deepcopy(result)
            broken["identifier_guard"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_live_gate(broken, base, expected)

    def test_preserves_accepted_evidence(self):
        base, expected, result = self.fixture("Which city?")
        validate_live_gate(result, base, expected)
        with self.assertRaises(ValueError):
            validate_live_gate({**result, "evidence": {"start": 4, "end": 9}}, base, expected)


if __name__ == "__main__":
    unittest.main()
