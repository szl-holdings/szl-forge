#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import verify_model_portfolio as verifier


class PortfolioContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(
            verifier.DEFAULT_PORTFOLIO.read_text(encoding="utf-8")
        )

    def test_portfolio_names_every_public_model_repository_once(self) -> None:
        repo_ids = verifier.validate_portfolio(self.document)
        self.assertEqual(15, len(repo_ids))
        self.assertEqual(15, len(set(repo_ids)))

    def test_forge_lab_renders_the_exact_canonical_portfolio(self) -> None:
        self.assertEqual(
            verifier.DEFAULT_PORTFOLIO.read_bytes(),
            (
                verifier.ROOT
                / "spaces"
                / "szl-forge-lab"
                / "model_portfolio.json"
            ).read_bytes(),
        )
        index = (
            verifier.ROOT / "spaces" / "szl-forge-lab" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("MODEL / KERNEL PORTFOLIO", index)
        self.assertIn("model_portfolio.json", index)

    def test_only_true_artifacts_are_typed_as_weighted(self) -> None:
        weighted = {
            item["repo_id"]
            for item in self.document["artifacts"]
            if item["kind"] != "software_kernel"
        }
        self.assertEqual(
            {
                "SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent",
                "SZLHOLDINGS/SZL-Khipu-1.5B",
                "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF",
                "SZLHOLDINGS/szl-lambda-gate",
            },
            weighted,
        )

    def test_no_artifact_is_marked_autonomy_eligible(self) -> None:
        self.assertTrue(
            all(item["autonomy_eligible"] is False for item in self.document["artifacts"])
        )

    def test_validator_refuses_duplicate_repositories(self) -> None:
        changed = copy.deepcopy(self.document)
        changed["artifacts"].append(copy.deepcopy(changed["artifacts"][0]))
        with self.assertRaises(verifier.PortfolioError):
            verifier.validate_portfolio(changed)

    def test_validator_refuses_unreviewed_autonomy_promotion(self) -> None:
        changed = copy.deepcopy(self.document)
        changed["artifacts"][0]["autonomy_eligible"] = True
        with self.assertRaises(verifier.PortfolioError):
            verifier.validate_portfolio(changed)

    def test_local_signed_receipts_and_dataset_hashes_verify(self) -> None:
        for relative in ("receiptagent", "khipu"):
            with self.subTest(relative=relative):
                evidence = verifier.verify_signed_receipts(verifier.ROOT / relative)
                self.assertEqual(
                    "DECLARED_KEY_SIGNATURES_VALID", evidence["status"]
                )
                self.assertFalse(evidence["weights_hash_recomputed"])

    def test_receipt_hashes_use_committed_bytes_on_windows(self) -> None:
        path = verifier.ROOT / "receiptagent" / "train.jsonl"
        self.assertEqual(
            "775e25b526a96d1486e80aae048f731bdad02a0d94ea76144593e115802fa24f",
            verifier.sha256_source(path),
        )

    def test_khipu_limit_is_encoded_from_signed_counts(self) -> None:
        evidence = verifier.verify_signed_receipts(verifier.ROOT / "khipu")
        self.assertEqual(2, evidence["abstainCorrect"])
        self.assertEqual(6, evidence["abstainTotal"])
        item = next(
            entry
            for entry in self.document["artifacts"]
            if entry["repo_id"] == "SZLHOLDINGS/SZL-Khipu-1.5B"
        )
        self.assertFalse(item["autonomy_eligible"])
        self.assertIn("2/6", " ".join(item["limitations"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
