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
        self.assertEqual(requirements.splitlines(), ["cryptography==50.0.0"])
        self.assertIn("python tools/verify_model_portfolio.py --offline", card)
        self.assertNotIn("python khipu/eval_khipu.py --help", card)
        self.assertNotIn("python khipu/sanity_gate.py", card)


class PortfolioArtifactMetadataTests(unittest.TestCase):
    def audit_sizes(self, sizes, *, maximum=None, kind="trained_model"):
        artifact = {
            "repo_id": "SZLHOLDINGS/synthetic-metadata",
            "kind": kind,
            "maturity": "RESEARCH_ONLY",
            "hub_revision": "a" * 40,
            "required_files": [],
            "github_source": "https://github.com/szl-holdings/szl-forge",
        }
        if maximum is not None:
            artifact["max_total_weight_bytes"] = maximum
        info = SimpleNamespace(
            sha="a" * 40,
            card_data=SimpleNamespace(license="apache-2.0"),
            downloads=0,
            siblings=[
                SimpleNamespace(rfilename=f"weight-{index}.safetensors", size=size, lfs=None)
                for index, size in enumerate(sizes)
            ],
        )
        api = mock.Mock()
        api.model_info.return_value = info
        with (
            mock.patch("socket.create_connection", side_effect=AssertionError("network forbidden")),
            mock.patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
            mock.patch("huggingface_hub.hf_hub_download") as download,
        ):
            result = verifier.audit_live_artifact(
                artifact, api=api, weight_extensions=(".safetensors",)
            )
        download.assert_not_called()
        api.model_info.assert_called_once_with(
            artifact["repo_id"], files_metadata=True, revision="a" * 40
        )
        return result

    def test_unknown_weight_size_remains_unknown_and_fails_closed(self):
        for sizes in ([None], [10, None]):
            with self.subTest(sizes=sizes):
                result = self.audit_sizes(sizes, maximum=100)
                self.assertFalse(result["ok"])
                self.assertIsNone(result["total_weight_bytes"])
                self.assertTrue(any("size is unknown" in error for error in result["errors"]))
                self.assertIsNone(result["weight_files"][-1]["size"])

    def test_empty_weight_payload_cannot_pass_the_artifact_gate(self):
        for sizes in ([0], [0, 10]):
            with self.subTest(sizes=sizes):
                result = self.audit_sizes(sizes)
                self.assertFalse(result["ok"])
                self.assertTrue(any("empty weight artifact" in error for error in result["errors"]))

    def test_invalid_weight_sizes_do_not_become_byte_counts(self):
        for size in (-1, True, "100", 0.0):
            with self.subTest(size=size):
                result = self.audit_sizes([size])
                self.assertFalse(result["ok"])
                self.assertIsNone(result["total_weight_bytes"])
                self.assertTrue(any("invalid weight size" in error for error in result["errors"]))

    def test_known_sizes_keep_the_existing_total_budget_check(self):
        within = self.audit_sizes([3, 7], maximum=10)
        self.assertTrue(within["ok"])
        self.assertEqual(10, within["total_weight_bytes"])
        over = self.audit_sizes([3, 7], maximum=9)
        self.assertFalse(over["ok"])
        self.assertTrue(any("exceed declared maximum" in error for error in over["errors"]))

    def test_small_positive_file_is_not_declared_a_placeholder(self):
        result = self.audit_sizes([1], maximum=10)
        self.assertTrue(result["ok"])
        self.assertEqual(1, result["total_weight_bytes"])
        self.assertEqual([], result["warnings"])

    def test_software_kernel_classification_is_unchanged(self):
        empty = self.audit_sizes([], kind="software_kernel")
        self.assertTrue(empty["ok"])
        self.assertEqual("software_kernel", empty["kind"])
        self.assertEqual("RESEARCH_ONLY", empty["maturity"])
        self.assertEqual(0, empty["total_weight_bytes"])
        weighted = self.audit_sizes([10], kind="software_kernel")
        self.assertFalse(weighted["ok"])
        self.assertTrue(any("unexpectedly contains" in error for error in weighted["errors"]))

    def audit_receipts(self, *, resolved_revision, requested_pin=None):
        artifact = {
            "repo_id": "SZLHOLDINGS/synthetic-receipts",
            "kind": "software_kernel",
            "maturity": "RESEARCH_ONLY",
            "local_receipt_dir": "receipts",
            "required_files": [],
            "github_source": "https://github.com/szl-holdings/szl-forge",
        }
        if requested_pin is not None:
            artifact["hub_revision"] = requested_pin
        info = SimpleNamespace(
            sha=resolved_revision,
            card_data=SimpleNamespace(license="apache-2.0"),
            downloads=0,
            siblings=[],
        )
        api = mock.Mock()
        api.model_info.return_value = info
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir()
            remote_files = {}
            for name in verifier.RECEIPT_FILES:
                path = receipts / name
                path.write_bytes(b"synthetic receipt parity fixture\n")
                remote_files[name] = str(path)

            def download(*, filename, revision, **kwargs):
                self.assertIn(filename, verifier.RECEIPT_FILES)
                self.assertEqual(resolved_revision, revision)
                self.assertEqual("model", kwargs["repo_type"])
                self.assertTrue(kwargs["force_download"])
                return remote_files[filename]

            with (
                mock.patch("socket.create_connection", side_effect=AssertionError("network forbidden")),
                mock.patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
                mock.patch.object(verifier, "ROOT", root),
                mock.patch.object(verifier, "sha256_source", side_effect=verifier.sha256_path),
                mock.patch("huggingface_hub.hf_hub_download", side_effect=download) as downloader,
            ):
                result = verifier.audit_live_artifact(
                    artifact, api=api, weight_extensions=(".safetensors",)
                )
                calls = list(downloader.call_args_list)
        expected_arguments = {"files_metadata": True}
        if requested_pin is not None:
            expected_arguments["revision"] = requested_pin
        api.model_info.assert_called_once_with(artifact["repo_id"], **expected_arguments)
        return result, calls

    def test_receipt_downloads_bind_the_valid_metadata_revision(self):
        for pin in (None, "a" * 40):
            with self.subTest(pin=pin):
                result, calls = self.audit_receipts(
                    resolved_revision="a" * 40, requested_pin=pin
                )
                self.assertTrue(result["ok"])
                self.assertEqual(len(verifier.RECEIPT_FILES), len(calls))
                for call in calls:
                    self.assertEqual("a" * 40, call.kwargs["revision"])
                    self.assertEqual("model", call.kwargs["repo_type"])
                self.assertTrue(all(item["matched"] for item in result["receipt_parity"].values()))

    def test_missing_or_invalid_revision_never_downloads_receipts(self):
        for revision in (None, "main", "a" * 39, "A" * 40, "g" * 40):
            with self.subTest(revision=revision):
                result, calls = self.audit_receipts(resolved_revision=revision)
                self.assertFalse(result["ok"])
                self.assertIsNone(result["receipt_parity"])
                self.assertEqual([], calls)
                self.assertTrue(any("exact resolved Hub revision" in error for error in result["errors"]))

    def test_stale_resolved_head_never_downloads_receipts(self):
        result, calls = self.audit_receipts(
            resolved_revision="b" * 40, requested_pin="a" * 40
        )
        self.assertFalse(result["ok"])
        self.assertIsNone(result["receipt_parity"])
        self.assertEqual([], calls)
        self.assertTrue(any("differs from pin" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
