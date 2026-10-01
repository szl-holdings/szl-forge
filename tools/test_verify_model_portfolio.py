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


class ArtifactCompletenessTests(unittest.TestCase):
    """Synthetic same-commit observations; no providers, payloads or loaders."""

    REVISION = "a" * 40

    def classify(self, paths, *, documents=None, headers=None, complete=True, sizes=None, revision=None):
        revision = self.REVISION if revision is None else revision
        files = [
            {"path": path, "revision": revision, "size": (sizes or {}).get(path, 8)}
            for path in paths
        ]
        return verifier.classify_artifact_completeness(
            revision=revision, inventory_complete=complete, files=files,
            json_observations=documents, gguf_header_observations=headers,
        )

    def parsed(self, data, *, revision=None):
        return {"revision": revision or self.REVISION, "state": "PARSED", "data": data}

    def model_documents(self, index=None):
        result = {"config.json": self.parsed({"model_type": "synthetic"})}
        if index is not None:
            result["model.safetensors.index.json"] = self.parsed({"weight_map": index})
        return result

    def test_complete_index_requires_every_referenced_shard(self):
        paths = ["config.json", "model.safetensors.index.json", "model-1.safetensors", "model-2.safetensors"]
        documents = self.model_documents({"a": "model-1.safetensors", "b": "model-2.safetensors", "c": "model-1.safetensors"})
        result = self.classify(paths, documents=documents)
        self.assertEqual("COMPLETE_STRUCTURE", result["status"])
        self.assertEqual(["model-1.safetensors", "model-2.safetensors"], result["packages"][0]["referenced_shards"])
        self.assertEqual("NOT_EVALUATED", result["artifact_validity"])
        missing = self.classify(paths[:-1], documents=documents)
        self.assertEqual("INCOMPLETE_STRUCTURE", missing["status"])
        self.assertIn({"code": "ABSENT_AT_COMPLETE_INVENTORY", "path": "model-2.safetensors"}, missing["packages"][0]["issues"])

    def test_incomplete_inventory_cannot_establish_absence(self):
        documents = self.model_documents({"a": "missing.safetensors"})
        for complete in (None, False):
            with self.subTest(complete=complete):
                result = self.classify(["config.json", "model.safetensors.index.json"], documents=documents, complete=complete)
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn({"code": "PRESENCE_UNKNOWN", "path": "missing.safetensors"}, result["packages"][0]["issues"])
                self.assertNotIn("ABSENT_AT_COMPLETE_INVENTORY", str(result))

    def test_nested_referenced_shards_do_not_create_an_unconfigured_variant(self):
        paths = ["config.json", "model.safetensors.index.json", "shards/model-1.safetensors"]
        documents = self.model_documents({"a": "shards/model-1.safetensors"})
        result = self.classify(paths, documents=documents)
        self.assertEqual("COMPLETE_STRUCTURE", result["status"])
        self.assertEqual(1, len(result["packages"]))
        self.assertEqual(["shards/model-1.safetensors"], result["packages"][0]["referenced_shards"])
        # A directory with independent payload or config remains its own variant.
        result = self.classify(paths + ["shards/independent.safetensors"], documents=documents)
        self.assertEqual("INCOMPLETE_STRUCTURE", result["status"])
        self.assertEqual(2, len(result["packages"]))
        result = self.classify(paths + ["shards/config.json"], documents=documents)
        self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
        self.assertEqual(2, len(result["packages"]))

    def test_parsed_zero_byte_sidecars_are_conflicting_evidence(self):
        paths = ["config.json", "model.safetensors.index.json", "model-1.safetensors"]
        documents = self.model_documents({"a": "model-1.safetensors"})
        for path in ("config.json", "model.safetensors.index.json"):
            with self.subTest(path=path):
                result = self.classify(paths, documents=documents, sizes={path: 0})
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn({"code": "PARSED_EMPTY_METADATA_CONFLICT", "path": path}, result["packages"][0]["issues"])

    def test_index_alone_is_metadata_not_weight_payload(self):
        documents = self.model_documents({"a": "missing.safetensors"})
        result = self.classify(["config.json", "model.safetensors.index.json"], documents=documents)
        self.assertEqual("INCOMPLETE_STRUCTURE", result["status"])
        self.assertEqual([], result["packages"][0]["payloads"])
        self.assertEqual("WEIGHT_INDEX_METADATA", result["file_roles"][1]["role"])

    def test_binary_index_and_variant_config_use_their_own_directory(self):
        documents = {
            "config.json": self.parsed({"model_type": "synthetic"}),
            "variant/pytorch_model.bin.index.json": self.parsed({"weight_map": {"a": "pytorch_model-1.bin"}}),
        }
        paths = ["config.json", "variant/pytorch_model.bin.index.json", "variant/pytorch_model-1.bin"]
        result = self.classify(paths, documents=documents)
        group = next(item for item in result["packages"] if item.get("directory") == "variant")
        self.assertEqual("INCOMPLETE_STRUCTURE", group["status"])
        self.assertIn({"code": "ABSENT_AT_COMPLETE_INVENTORY", "path": "variant/config.json"}, group["issues"])
        paths.append("variant/config.json")
        documents["variant/config.json"] = self.parsed({"architectures": ["SyntheticModel"]})
        result = self.classify(paths, documents=documents)
        self.assertEqual("COMPLETE_STRUCTURE", next(item for item in result["packages"] if item.get("directory") == "variant")["status"])

    def test_unsafe_or_unsupported_index_references_stay_unknown(self):
        for shard in ("../outside.safetensors", "/absolute.safetensors", "C:/outside.safetensors", "dir\\part.safetensors", "bad\x00.safetensors", "model.safetensors.index.json", "", None):
            with self.subTest(shard=shard):
                result = self.classify(["config.json", "model.safetensors.index.json"], documents=self.model_documents({"a": shard}))
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn("UNSAFE_OR_UNSUPPORTED_SHARD_REFERENCE", str(result))
                self.assertEqual([], result["packages"][0]["referenced_shards"])

    def test_empty_or_malformed_indexes_are_unknown_schema(self):
        for data in ({}, {"weight_map": {}}, {"weight_map": []}, {"weight_map": "not a map"}, []):
            with self.subTest(data=data):
                documents = self.model_documents()
                documents["model.safetensors.index.json"] = self.parsed(data)
                result = self.classify(["config.json", "model.safetensors.index.json"], documents=documents)
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn("SCHEMA", str(result))

    def test_config_only_and_missing_config_are_reported(self):
        config_only = self.classify(["config.json"], documents=self.model_documents())
        self.assertEqual("INCOMPLETE_STRUCTURE", config_only["status"])
        self.assertIn("NO_PRINCIPAL_PAYLOAD", str(config_only))
        missing_config = self.classify(["model.safetensors"])
        self.assertEqual("INCOMPLETE_STRUCTURE", missing_config["status"])
        self.assertIn({"code": "ABSENT_AT_COMPLETE_INVENTORY", "path": "config.json"}, missing_config["packages"][0]["issues"])

    def test_empty_and_unsupported_config_objects_are_not_valid_configs(self):
        for data in ({}, [], {"unrecognized": True}, {"model_type": ""}, {"architectures": []}):
            with self.subTest(data=data):
                result = self.classify(["config.json", "model.safetensors"], documents={"config.json": self.parsed(data)})
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn("SCHEMA", str(result))

    def test_metadata_read_failures_remain_distinct_from_absence(self):
        for state in ("ACCESS_DENIED", "FETCH_FAILED", "PARSE_FAILED", "NOT_OBSERVED", "TRUNCATED"):
            with self.subTest(state=state):
                result = self.classify(["config.json", "model.safetensors"], documents={"config.json": {"revision": self.REVISION, "state": state}})
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn({"code": state, "path": "config.json"}, result["packages"][0]["issues"])
                self.assertNotIn("ABSENT_AT_COMPLETE_INVENTORY", str(result))

    def test_every_file_and_document_requires_the_same_immutable_revision(self):
        for revision in (None, "main", "A" * 40, "a" * 39):
            with self.subTest(revision=revision):
                result = verifier.classify_artifact_completeness(revision=revision, inventory_complete=True, files=[])
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn("INVALID_REVISION", str(result))
        rows = [{"path": "model.gguf", "revision": "b" * 40, "size": 8}]
        result = verifier.classify_artifact_completeness(revision=self.REVISION, inventory_complete=True, files=rows)
        self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
        self.assertIn("FILE_REVISION_MISMATCH", str(result))
        documents = {"config.json": self.parsed({"model_type": "synthetic"}, revision="b" * 40)}
        result = self.classify(["config.json", "model.safetensors"], documents=documents)
        self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
        self.assertIn("DOCUMENT_REVISION_MISMATCH", str(result))
        result = self.classify(["model.gguf"], headers={"model.gguf": self.parsed({"quantization": "QUANTIZED"}, revision="b" * 40)})
        self.assertEqual("UNKNOWN_STRUCTURE", result["status"])

    def test_unknown_empty_and_small_payloads_keep_distinct_meanings(self):
        for size, expected, code in ((None, "UNKNOWN_STRUCTURE", "PAYLOAD_SIZE_UNKNOWN"), (0, "INCOMPLETE_STRUCTURE", "EMPTY_PAYLOAD"), (0.0, "UNKNOWN_STRUCTURE", "INVALID_PAYLOAD_SIZE"), (-1, "UNKNOWN_STRUCTURE", "INVALID_PAYLOAD_SIZE"), (True, "UNKNOWN_STRUCTURE", "INVALID_PAYLOAD_SIZE"), (1, "COMPLETE_STRUCTURE", None)):
            with self.subTest(size=size):
                result = self.classify(["config.json", "model.safetensors"], documents=self.model_documents(), sizes={"model.safetensors": size})
                self.assertEqual(expected, result["status"])
                if code:
                    self.assertIn(code, str(result))
                self.assertNotIn("placeholder", str(result).lower())

    def test_mmproj_is_auxiliary_and_gguf_needs_no_external_config(self):
        for path in ("mmproj.gguf", "vision/mmproj-model-f16.gguf", "MMProj.gguf"):
            with self.subTest(path=path):
                result = self.classify([path])
                self.assertEqual("INCOMPLETE_STRUCTURE", result["status"])
                self.assertEqual([], result["packages"])
                self.assertEqual("AUXILIARY_GGUF_FILENAME", result["file_roles"][0]["role"])
        result = self.classify(["model-Q4_K_M.gguf", "mmproj-f16.gguf"])
        self.assertEqual("COMPLETE_STRUCTURE", result["status"])
        self.assertEqual(None, result["packages"][0]["config"])
        self.assertEqual("UNKNOWN", result["gguf_quantization"][0]["status"])

    def test_filenames_never_prove_gguf_quantization(self):
        for path in ("model-F16.gguf", "model-BF16.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"):
            with self.subTest(path=path):
                result = self.classify([path])
                self.assertEqual("UNKNOWN", result["gguf_quantization"][0]["status"])
                self.assertEqual(None, result["gguf_quantization"][0]["precision"])

    def test_supplied_headers_separate_floating_quantized_and_conflicting_evidence(self):
        for precision in ("F16", "BF16"):
            with self.subTest(precision=precision):
                result = self.classify(["model.gguf"], headers={"model.gguf": self.parsed({"quantization": "UNKNOWN", "precision": precision})})
                self.assertEqual("FLOATING", result["gguf_quantization"][0]["status"])
                result = self.classify(["model.gguf"], headers={"model.gguf": self.parsed({"quantization": "QUANTIZED", "precision": precision})})
                self.assertEqual("UNKNOWN", result["gguf_quantization"][0]["status"])
                self.assertIn("CONFLICTING_HEADER_OBSERVATION", str(result))
        result = self.classify(["model.gguf"], headers={"model.gguf": self.parsed({"quantization": "QUANTIZED"})})
        self.assertEqual("QUANTIZED", result["gguf_quantization"][0]["status"])

    def test_adapter_variants_preserve_exact_mixed_bases_and_unknown_revisions(self):
        paths = ["adapter_model.safetensors", "adapter_config.json", "alt/adapter_model.safetensors", "alt/adapter_config.json"]
        documents = {
            "adapter_config.json": self.parsed({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "revision": None}),
            "alt/adapter_config.json": self.parsed({"base_model_name_or_path": "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit", "revision": None}),
        }
        result = self.classify(paths, documents=documents)
        self.assertEqual("COMPLETE_STRUCTURE", result["status"])
        self.assertTrue(result["mixed_adapter_bases"])
        self.assertEqual(2, len(result["adapter_groups"]))
        self.assertEqual({"Qwen/Qwen2.5-0.5B-Instruct", "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit"}, {item["base_model_name_or_path"] for item in result["adapter_groups"]})
        for item in result["adapter_groups"]:
            self.assertIsNone(item["base_revision"])
            self.assertEqual("UNKNOWN", item["lineage_status"])
            self.assertEqual("NOT_EVALUATED", item["compatibility"])

    def test_adapter_unknown_base_and_index_without_payload_remain_explicit(self):
        for base in (None, "", " "):
            with self.subTest(base=base):
                documents = {"adapter_config.json": self.parsed({"peft_type": "LORA", "base_model_name_or_path": base})}
                result = self.classify(["adapter_config.json", "adapter_model.safetensors"], documents=documents)
                self.assertEqual("UNKNOWN", result["adapter_groups"][0]["base_status"])
                self.assertIsNone(result["mixed_adapter_bases"])
        documents = {
            "adapter_config.json": self.parsed({"peft_type": "LORA", "base_model_name_or_path": "exact/base", "revision": "b" * 40}),
            "adapter_model.safetensors.index.json": self.parsed({"weight_map": {"a": "adapter_model-1.safetensors"}}),
        }
        result = self.classify(["adapter_config.json", "adapter_model.safetensors.index.json"], documents=documents)
        self.assertEqual("INCOMPLETE_STRUCTURE", result["status"])
        self.assertEqual("RECORDED", result["adapter_groups"][0]["lineage_status"])
        self.assertEqual([], result["packages"][0]["payloads"])

    def test_adapter_index_owns_arbitrarily_named_safe_shards(self):
        for shard in ("part-00001.safetensors", "parts/part-00001.safetensors"):
            with self.subTest(shard=shard):
                documents = {
                    "adapter_config.json": self.parsed({"peft_type": "LORA", "base_model_name_or_path": "exact/base"}),
                    "adapter_model.safetensors.index.json": self.parsed({"weight_map": {"a": shard}}),
                }
                result = self.classify(["adapter_config.json", "adapter_model.safetensors.index.json", shard], documents=documents)
                self.assertEqual("COMPLETE_STRUCTURE", result["status"])
                self.assertEqual(1, len(result["packages"]))
                self.assertEqual("ADAPTER", result["packages"][0]["kind"])
                self.assertEqual([shard], result["adapter_groups"][0]["payloads"])

    def test_classifier_is_pure_and_does_not_modify_supplied_observations(self):
        files = [{"path": "model.gguf", "revision": self.REVISION, "size": 1}]
        documents = {}
        headers = {"model.gguf": self.parsed({"quantization": "UNKNOWN"})}
        before = copy.deepcopy((files, documents, headers))
        import socket
        with mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")), mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), mock.patch.object(subprocess, "run", side_effect=AssertionError("process forbidden")), mock.patch.object(Path, "open", side_effect=AssertionError("file I/O forbidden")):
            result = verifier.classify_artifact_completeness(revision=self.REVISION, inventory_complete=True, files=files, json_observations=documents, gguf_header_observations=headers)
        self.assertEqual(before, (files, documents, headers))
        self.assertEqual("COMPLETE_STRUCTURE", result["status"])

    def test_malformed_observation_envelopes_fail_closed_without_crashing(self):
        for state in (None, [], {}, 7):
            with self.subTest(state=state):
                result = self.classify(["config.json", "model.safetensors"], documents={"config.json": {"revision": self.REVISION, "state": state}})
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertIn("UNKNOWN_OBSERVATION_STATE", str(result))
        for data in ({"quantization": []}, {"quantization": [], "precision": "F16"}, {"precision": "BF16"}, {"quantization": "QUANTIZED", "precision": []}, {}):
            with self.subTest(data=data):
                result = self.classify(["model.gguf"], headers={"model.gguf": self.parsed(data)})
                self.assertEqual("UNKNOWN", result["gguf_quantization"][0]["status"])
                self.assertIn("SCHEMA", str(result))
        result = self.classify(["model.gguf"], documents=[])
        self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
        self.assertIn("INVALID_DOCUMENT_COLLECTION", str(result))

    def test_unusable_inventory_cannot_assert_absence_even_when_claimed_complete(self):
        for files in ({}, None, [{"path": "bad/../path", "revision": self.REVISION, "size": 8}], [{"path": "model.safetensors", "revision": "b" * 40, "size": 8}]):
            with self.subTest(files=files):
                result = verifier.classify_artifact_completeness(revision=self.REVISION, inventory_complete=True, files=files)
                self.assertEqual("UNKNOWN_STRUCTURE", result["status"])
                self.assertFalse(result["absence_supported"])
                self.assertNotIn("ABSENT_AT_COMPLETE_INVENTORY", str(result))
                self.assertNotIn("NO_PRINCIPAL_PAYLOAD", str(result))

    def test_live_audit_keeps_old_ok_scope_and_makes_no_new_reads(self):
        artifact = {"repo_id": "example/synthetic", "kind": "quantized_model", "maturity": "HELD_TEST_ONLY", "required_files": [], "github_source": "https://example.invalid/source"}
        info = SimpleNamespace(sha=self.REVISION, siblings=[SimpleNamespace(rfilename="model.gguf", size=8, lfs=None)], card_data=SimpleNamespace(license="apache-2.0"))
        api = SimpleNamespace(model_info=mock.Mock(return_value=info))
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download forbidden")):
            result = verifier.audit_live_artifact(artifact, api=api, weight_extensions=(".gguf", ".safetensors"))
        api.model_info.assert_called_once_with(artifact["repo_id"], files_metadata=True)
        self.assertTrue(result["ok"])
        self.assertEqual("UNKNOWN_STRUCTURE", result["metadata_structure"]["status"])
        self.assertIsNone(result["metadata_structure"]["inventory_complete"])
        self.assertEqual("existing catalog suffix/size/pin/license/receipt checks only", result["ok_scope"])
        self.assertEqual("NOT_EVALUATED", result["artifact_validity"])
        self.assertEqual("NOT_EVALUATED", result["runtime_validity"])
        self.assertEqual("HELD_TEST_ONLY", result["maturity"])

    def test_live_mmproj_alone_cannot_satisfy_principal_weight_gate(self):
        artifact = {"repo_id": "example/synthetic", "kind": "quantized_model", "maturity": "HELD_TEST_ONLY", "required_files": [], "github_source": "https://example.invalid/source"}
        info = SimpleNamespace(sha=self.REVISION, siblings=[SimpleNamespace(rfilename="mmproj-f16.gguf", size=8, lfs=None)], card_data=SimpleNamespace(license="apache-2.0"))
        api = SimpleNamespace(model_info=mock.Mock(return_value=info))
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("download forbidden")):
            result = verifier.audit_live_artifact(artifact, api=api, weight_extensions=(".gguf",))
        self.assertFalse(result["ok"])
        self.assertEqual([], result["weight_files"])
        self.assertIn("quantized_model has no weight artifact", result["errors"])
        self.assertEqual("quantized_model", result["kind"])
        self.assertEqual("HELD_TEST_ONLY", result["maturity"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
