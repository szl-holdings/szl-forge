from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "retrieval_risk_kernel.py"
SPEC = importlib.util.spec_from_file_location("retrieval_risk_kernel", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def response(
    margin: float, threshold: float = 1.0, answer: str | None = "span"
) -> dict[str, object]:
    return {
        "status": "ANSWER"
        if answer is not None and margin > threshold
        else "ABSTAIN",
        "answer": answer if answer is not None and margin > threshold else None,
        "margin": margin,
        "threshold": threshold,
    }


class KernelTests(unittest.TestCase):
    def test_cases_are_deterministic_disjoint_and_absent(self) -> None:
        seed = "01" * 32
        first = MODULE.generate_cases(seed, 3, 3)
        second = MODULE.generate_cases(seed, 3, 3)
        self.assertEqual(first, second)
        nonces = {
            str(case[field])
            for case in first
            for field in ("subject_nonce", "event_nonce")
        }
        self.assertEqual(len(nonces), 12)
        proof = MODULE.verify_nonce_absence(
            first, {"documents": [{"text": "ordinary corpus"}]}
        )
        self.assertEqual(proof["occurrences_found"], 0)

    def test_absence_is_nfkc_casefolded(self) -> None:
        cases = MODULE.generate_cases("02" * 32, 1, 1)
        nonce = str(cases[0]["subject_nonce"])
        with self.assertRaises(ValueError):
            MODULE.verify_nonce_absence(
                cases, {"documents": [{"text": nonce.swapcase()}]}
            )

    def test_wilson_zero_errors_needs_sufficient_sample(self) -> None:
        self.assertGreater(MODULE.wilson_interval(0, 60)[1], 0.05)
        self.assertLess(MODULE.wilson_interval(0, 80)[1], 0.05)

    def test_threshold_selection_uses_strict_heldout_rule(self) -> None:
        rows = [response(2.0) for _ in range(80)]
        selected = MODULE.select_threshold(rows, 1.0, 0.05)
        self.assertEqual(selected["threshold"], 2.0)
        self.assertEqual(selected["false_answers"], 0)
        self.assertTrue(selected["target_met"])

    def test_threshold_does_not_reanimate_base_abstention(self) -> None:
        row = response(0.5)
        self.assertFalse(MODULE.is_answer(row, 0.1))

    def test_positive_replay_exposes_coverage_tradeoff(self) -> None:
        records = [
            {
                "gold": ["justice and prosperity"],
                "response": response(3.0, answer="justice and prosperity"),
            },
            {"gold": ["elsewhere"], "response": response(2.0, answer="wrong")},
        ]
        scored = MODULE.score_positive_replay(records, 2.5)
        self.assertEqual(scored["base"]["coverage"], 1.0)
        self.assertEqual(
            scored["retrieval_conditioned_kernel"]["coverage"], 0.5
        )
        self.assertEqual(
            scored["retrieval_conditioned_kernel"]["exact_match"], 0.5
        )

    def test_seed_validation_rejects_non_hex(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_seed("z" * 64)


if __name__ == "__main__":
    unittest.main()
