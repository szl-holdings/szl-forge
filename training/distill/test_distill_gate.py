"""Unit tests for the governed-distillation grounding gate (offline, no GPU)."""
import sys, unittest
sys.path.insert(0, ".")
from distill_wave import grounding_gate, repetition_ratio


CTX = ("The estate doctrine is defined in szl.lambda/v1 and the gate contract "
       "lives in HIRAG-ADMISSION. Receipts are hash-chained under Khipu.")


class TestGroundingGate(unittest.TestCase):
    def test_grounded_answer_accepted(self):
        r = "The estate gate contract lives in HIRAG-ADMISSION and receipts are hash-chained under Khipu per szl.lambda/v1."
        g = grounding_gate(CTX, r)
        self.assertEqual(g["verdict"], "ACCEPT")
        self.assertEqual(g["mode"], "ANSWER")

    def test_explicit_abstain_accepted(self):
        g = grounding_gate(CTX, "ABSTAIN — I don't know the answer to this.")
        self.assertEqual(g["verdict"], "ACCEPT")
        self.assertEqual(g["mode"], "ABSTAIN")

    def test_fabricated_identifier_rejected(self):
        r = "The gate contract lives in HIRAG-ADMISSION-V9-NEVER-EXISTS per doctrine."
        g = grounding_gate(CTX, r)
        self.assertEqual(g["verdict"], "REJECT")
        self.assertTrue(any(x.startswith("FABRICATED_IDENTIFIERS") for x in g["reasons"]))

    def test_empty_rejected(self):
        g = grounding_gate(CTX, "   ")
        self.assertEqual(g["verdict"], "REJECT")
        self.assertIn("EMPTY_RESPONSE", g["reasons"])

    def test_repetition_loop_rejected(self):
        loop = " ".join(["word"] * 60)
        g = grounding_gate(CTX, loop)
        self.assertEqual(g["verdict"], "REJECT")
        self.assertIn("REPETITION_LOOP", g["reasons"])

    def test_evasion_rejected(self):
        g = grounding_gate(CTX, "Maybe.")
        self.assertEqual(g["verdict"], "REJECT")
        self.assertIn("NEITHER_ANSWERS_NOR_ABSTAINS", g["reasons"])

    def test_code_fence_identifiers_not_fabrication(self):
        r = ("The contract is as follows: ```\ndef f(): return HIRAG_ADMISSION\n``` "
             "and it lives in HIRAG-ADMISSION per szl.lambda/v1.")
        g = grounding_gate(CTX, r)
        self.assertEqual(g["verdict"], "ACCEPT")

    def test_repetition_ratio_bounds(self):
        self.assertEqual(repetition_ratio("one two three"), 0.0)
        self.assertGreater(repetition_ratio(" ".join(["x"] * 50)), 0.9)


if __name__ == "__main__":
    unittest.main()
