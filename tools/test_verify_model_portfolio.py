#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import verify_model_portfolio as verifier


class PortfolioContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(
            verifier.DEFAULT_PORTFOLIO.read_text(encoding="utf-8")
        )

    def test_portfolio_names_every_public_model_repository_once(self) -> None:
        repo_ids = verifier.validate_portfolio(self.document)
        self.assertEqual(16, len(repo_ids))
        self.assertEqual(16, len(set(repo_ids)))

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
                "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2",
                "SZLHOLDINGS/SZL-Khipu-1.5B",
                "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF",
                "SZLHOLDINGS/szl-lambda-gate",
            },
            weighted,
        )

    def test_qwen35_release_pins_hub_revision_and_weight_digest(self) -> None:
        item = next(
            entry
            for entry in self.document["artifacts"]
            if entry["repo_id"]
            == "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2"
        )
        self.assertEqual(40, len(item["hub_revision"]))
        self.assertEqual(
            {
                "adapter_model.safetensors": (
                    "885fc29fcb4cf55c280dc085fdb0a40f40d6b946"
                    "fee400dd5e4ed3459fe6334f"
                )
            },
            item["expected_weight_sha256"],
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

    def test_live_audit_fails_closed_on_revision_or_weight_drift(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/pinned-test",
            "kind": "trained_model",
            "maturity": "MEASURED_PUBLISHED_LIMITED",
            "autonomy_eligible": False,
            "hub_revision": "a" * 40,
            "expected_weight_sha256": {
                "adapter_model.safetensors": "b" * 64,
            },
            "required_files": ["adapter_model.safetensors"],
            "github_source": "https://github.com/szl-holdings/szl-forge",
        }
        info = SimpleNamespace(
            sha="c" * 40,
            card_data=SimpleNamespace(license="apache-2.0"),
            downloads=0,
            siblings=[
                SimpleNamespace(
                    rfilename="adapter_model.safetensors",
                    size=100,
                    lfs={"sha256": "d" * 64},
                )
            ],
        )
        api = mock.Mock()
        api.model_info.return_value = info
        result = verifier.audit_live_artifact(
            artifact,
            api=api,
            weight_extensions=(".safetensors",),
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("differs from pin" in error for error in result["errors"])
        )
        api.model_info.assert_called_once_with(
            artifact["repo_id"],
            files_metadata=True,
            revision=artifact["hub_revision"],
        )

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

    def test_live_receipt_parity_uses_committed_bytes(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/test-software-kernel",
            "kind": "software_kernel",
            "maturity": "SOFTWARE_ARTIFACT",
            "local_receipt_dir": "receiptagent",
            "required_files": [],
            "github_source": "https://github.com/szl-holdings/szl-forge",
        }
        info = SimpleNamespace(
            sha="a" * 40,
            card_data=SimpleNamespace(license="apache-2.0"),
            downloads=0,
            siblings=[],
        )
        api = mock.Mock()
        api.model_info.return_value = info
        with tempfile.TemporaryDirectory() as directory:
            remote_files = {}
            for name in verifier.RECEIPT_FILES:
                data = subprocess.check_output(
                    ["git", "show", f"HEAD:receiptagent/{name}"],
                    cwd=verifier.ROOT,
                )
                path = Path(directory) / name
                path.write_bytes(data)
                remote_files[name] = str(path)

            def download(*, filename, **_):
                return remote_files[filename]

            with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=download,
            ):
                result = verifier.audit_live_artifact(
                    artifact,
                    api=api,
                    weight_extensions=(".safetensors",),
                )
        self.assertTrue(result["ok"])
        self.assertTrue(
            all(item["matched"] for item in result["receipt_parity"].values())
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

    def test_khipu_quickstart_is_pinned_and_read_only(self) -> None:
        card = (verifier.ROOT / "khipu" / "HF_MODEL_CARD.md").read_text(
            encoding="utf-8"
        )
        requirements = (
            verifier.ROOT / "khipu" / "requirements-verify.txt"
        ).read_text(encoding="utf-8")
        self.assertEqual(requirements.splitlines(), ["cryptography==49.0.0"])
        self.assertIn("python tools/verify_model_portfolio.py --offline", card)
        self.assertNotIn("python khipu/eval_khipu.py --help", card)
        self.assertNotIn("python khipu/sanity_gate.py", card)


if __name__ == "__main__":
    unittest.main(verbosity=2)
