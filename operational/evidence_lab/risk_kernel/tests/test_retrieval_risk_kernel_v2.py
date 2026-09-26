from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "retrieval_risk_kernel_v2.py"
SPEC = importlib.util.spec_from_file_location("retrieval_risk_kernel_v2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def payload(count: int = 8) -> dict[str, object]:
    documents = []
    for index in range(count):
        text = (
            f"Document {index} records alpine cedar observatory measurements and "
            f"historical navigation details for a known public benchmark passage."
        )
        documents.append(
            {
                "id": f"doc-{index}",
                "title": f"Title {index}",
                "citation": "test",
                "text": text,
            }
        )
    return {"documents": documents}


class AnchoredKernelTests(unittest.TestCase):
    def test_generation_is_reproducible_and_disjoint(self) -> None:
        frozen = payload(8)
        first = MODULE.generate_anchored_cases("03" * 32, 3, 3, frozen)
        second = MODULE.generate_anchored_cases("03" * 32, 3, 3, frozen)
        self.assertEqual(first, second)
        self.assertEqual(
            len({case["anchor"]["document_id"] for case in first}), 6
        )
        proof = MODULE.verify_anchors(first, frozen)
        self.assertEqual(proof["unique_anchor_documents"], 6)
        self.assertEqual(proof["occurrences_found"], 0)

    def test_anchor_tampering_is_rejected(self) -> None:
        frozen = payload(4)
        cases = MODULE.generate_anchored_cases("04" * 32, 1, 1, frozen)
        cases[0]["anchor"]["phrase"] = "tampered phrase"
        with self.assertRaises(ValueError):
            MODULE.verify_anchors(cases, frozen)

    def test_anchor_recall_reports_rank(self) -> None:
        records = [
            {
                "anchor": {"document_id": "doc-b"},
                "response": {
                    "passages": [{"id": "doc-a"}, {"id": "doc-b"}]
                },
            }
        ]
        result = MODULE.anchor_recall(records)
        self.assertEqual(result["anchor_recall_at_3"], 1.0)
        self.assertEqual(result["mean_anchor_rank_when_retrieved"], 2.0)


if __name__ == "__main__":
    unittest.main()
