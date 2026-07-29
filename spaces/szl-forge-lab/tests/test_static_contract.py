from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVIDENCE = (
    "run_manifest.json",
    "eval_receipt.json",
    "training_summary.json",
    "thesis_formula_index.json",
    "science_source_ledger.json",
    "curriculum.json",
    "model_portfolio.json",
)


class StaticForgeContractTests(unittest.TestCase):
    def test_space_is_an_explicit_static_evidence_console(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("sdk: static\n", readme)
        self.assertIn("app_file: index.html\n", readme)

    def test_mobile_console_loads_every_packaged_evidence_file(self) -> None:
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            index,
        )
        self.assertIn('rel="icon" href="data:image/svg+xml,', index)
        self.assertIn("MODEL / KERNEL PORTFOLIO", index)
        for filename in REQUIRED_EVIDENCE:
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / filename).is_file())
                self.assertIn(filename, index)
                json.loads((ROOT / filename).read_text(encoding="utf-8"))

    def test_model_portfolio_does_not_conflate_cards_with_weights(self) -> None:
        portfolio = json.loads(
            (ROOT / "model_portfolio.json").read_text(encoding="utf-8")
        )
        kinds = [item["kind"] for item in portfolio["artifacts"]]
        self.assertEqual(3, kinds.count("trained_model"))
        self.assertEqual(1, kinds.count("quantized_model"))
        self.assertEqual(1, kinds.count("learned_kernel"))
        self.assertEqual(11, kinds.count("software_kernel"))
        self.assertTrue(
            all(item["autonomy_eligible"] is False for item in portfolio["artifacts"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
