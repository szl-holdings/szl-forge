from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "unresolved_identifier_gate_v3.py"
SPEC = importlib.util.spec_from_file_location("unresolved_identifier_gate_v3", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IdentifierGateTests(unittest.TestCase):
    def test_identifier_shape_is_narrow(self) -> None:
        self.assertTrue(MODULE.identifier_like("SzlNqSubjectabc123def456"))
        self.assertTrue(MODULE.identifier_like("LongCamelCaseIdentifier"))
        self.assertFalse(MODULE.identifier_like("ratification"))
        self.assertFalse(MODULE.identifier_like("Beyoncé"))

    def test_known_identifier_is_not_unresolved(self) -> None:
        token = "SzlNqSubjectabc123def456"
        vocabulary = {MODULE.normalized_token(token)}
        self.assertEqual(MODULE.unresolved_identifiers(token, vocabulary), [])

    def test_unknown_identifier_forces_abstention(self) -> None:
        question = "What did SzlNqSubjectabc123def456 sign?"
        response = {"status": "ANSWER", "answer": "a treaty"}
        decision = MODULE.gate_decision(question, response, set())
        self.assertEqual(decision["status"], "ABSTAIN")
        self.assertEqual(decision["reason"], "UNRESOLVED_IDENTIFIER")

    def test_base_abstention_stays_abstained(self) -> None:
        decision = MODULE.gate_decision(
            "Ordinary question", {"status": "ABSTAIN", "answer": None}, set()
        )
        self.assertEqual(decision["status"], "ABSTAIN")
        self.assertEqual(decision["reason"], "BASE_POLICY_ABSTAINED")

    def test_corpus_vocabulary_uses_title_and_text(self) -> None:
        vocabulary = MODULE.corpus_vocabulary(
            {
                "documents": [
                    {"title": "KnownIdentifier123456", "text": "ordinary words"}
                ]
            }
        )
        self.assertIn("knownidentifier123456", vocabulary)


if __name__ == "__main__":
    unittest.main()
