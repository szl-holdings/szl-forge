from __future__ import annotations

import copy
import inspect
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator

import evidence_chain as evidence
import prepare_release as release


SOURCE = "a" * 40
RUN_ID = "b" * 32


def sealed(report: dict[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(report)
    report["reportSha256"] = release.sha256_json(report)
    return report


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        include_source_readme: bool = False,
        adapter_revision: str = "8" * 40,
        target_modules: Any = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> None:
        self.adapter = root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "unsloth/Qwen3.5-0.8B",
                    "revision": adapter_revision,
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0,
                    "bias": "none",
                    "inference_mode": True,
                    "use_rslora": False,
                    "target_modules": target_modules,
                }
            ),
            encoding="utf-8",
        )
        (self.adapter / "adapter_model.safetensors").write_bytes(
            b"synthetic-safe-tensors-fixture"
        )
        if include_source_readme:
            (self.adapter / "README.md").write_bytes(
                b"# Adapter source metadata\n"
            )
        aggregate, files, _snapshot = release.adapter_inventory(self.adapter)
        attested_files = []
        for item in files:
            attested = dict(item)
            if item["path"].endswith(".json"):
                parsed = json.loads(
                    (self.adapter / item["path"]).read_text(encoding="utf-8")
                )
                attested["jsonKeys"] = len(parsed)
            elif item["path"].endswith(".safetensors"):
                attested["tensorCount"] = 1
            attested_files.append(attested)
        child = sealed(
            {
                "schema": "szl.frontier-training-run/v3",
                "candidateId": release.CANDIDATE_ID,
                "supervisorRunId": RUN_ID,
                "state": "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
                "runKind": "FULL",
                "source": {"revision": SOURCE},
                "sourceBundle": {
                    "manifestSha256": "0" * 64,
                    "trainSha256": "1" * 64,
                    "heldOutCommitments": {
                        "dev.jsonl": {"rows": 36, "sha256": "2" * 64},
                        "test.jsonl": {"rows": 72, "sha256": "3" * 64},
                    },
                },
                "runtimePackages": {"torch": "2.10.0", "unsloth": "2026.7.4"},
                "gpu": {
                    "uuid": "GPU-1111111111111111",
                    "name": "fixture-gpu",
                },
                "adapter": {
                    "aggregateSha256": aggregate,
                    "files": attested_files,
                },
                "authenticatedTrainingEnvelopePresent": False,
                "qualificationEligible": True,
                "receiptEligible": False,
                "publicationEligible": False,
            }
        )
        supervisor = sealed(
            {
                "schema": "szl.frontier-training-supervisor/v1",
                "candidateId": release.CANDIDATE_ID,
                "runId": RUN_ID,
                "runKind": "FULL",
                "state": "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
                "primaryCause": "SUCCESS",
                "source": {"revision": SOURCE},
                "identities": {
                    "supervisorSourceSha256": "4" * 64,
                    "workerSourceSha256": "5" * 64,
                    "validatorSourceSha256": "6" * 64,
                    "candidateSourceSha256": "7" * 64,
                },
                "containment": {
                    "unit": f"szl-ra3-supervisor-{RUN_ID}.service"
                },
                "telemetry": {
                    "gpuUuid": "GPU-1111111111111111",
                    "maximumObservedTemperatureC": 65,
                },
                "trainingReport": {
                    "canonicalReportSha256": child["reportSha256"],
                    "relativePath": "payload/training-report.json",
                },
                "adapter": {
                    "aggregateSha256": aggregate,
                    "matchesTrainingReport": True,
                    "safeTensorsParsed": True,
                    "allowlistedFilesOnly": True,
                    "symlinksAbsent": True,
                    "files": attested_files,
                },
                "authenticatedSupervisorEnvelopePresent": False,
                "localEvaluationInputBindingSatisfied": True,
                "receiptEligible": False,
                "publicationEligible": False,
            }
        )

        def evaluation(kind: str, split: str, adapter_bound: bool = False) -> dict[str, Any]:
            model: dict[str, Any] = {"kind": kind}
            linkage: dict[str, Any] = {}
            training_digest: str | None = None
            if adapter_bound:
                model["adapterAggregateSha256"] = aggregate
                linkage.update(
                    {
                        "runId": RUN_ID,
                        "reportSha256": supervisor["reportSha256"],
                        "adapterAggregateSha256": aggregate,
                        "sourceRevision": SOURCE,
                    }
                )
                training_digest = child["reportSha256"]
            return sealed(
                {
                    "schema": "szl.frontier-eval-run/v3",
                    "candidateId": release.CANDIDATE_ID,
                    "modelKind": kind,
                    "split": split,
                    "state": "MEASURED_EVALUATION_COMPLETED_UNATTESTED",
                    "source": {"revision": SOURCE},
                    "model": model,
                    "trainingReportSha256": training_digest,
                    "supervisionLinkage": linkage,
                    "counts": {"strictCasePass": 70 if kind == "v3" else 60},
                    "rates": {"strictCasePass": 0.972 if kind == "v3" else 0.833},
                    "absoluteGatePassed": True,
                    "comparisonEligible": False,
                    "authenticatedEvaluationEnvelopePresent": False,
                    "receiptEligible": False,
                    "publicationEligible": False,
                }
            )

        dev = evaluation("v3", "DEV", True)
        test = evaluation("v3", "TEST", True)
        base = evaluation("base", "TEST")
        v2 = evaluation("v2", "TEST")
        input_hashes = {
            "base": base["reportSha256"],
            "v2": v2["reportSha256"],
            "v3": test["reportSha256"],
        }
        comparison = sealed(
            {
                "schema": "szl.frontier-comparison/v2",
                "candidateId": release.CANDIDATE_ID,
                "state": "UNAUTHENTICATED_COMPARISON_CRITERIA_SATISFIED",
                "sourceRevision": SOURCE,
                "protocolSha256": "d" * 64,
                "inputReports": input_hashes,
                "recomputedResults": {
                    "base": {"counts": base["counts"], "rates": base["rates"]},
                    "v2": {"counts": v2["counts"], "rates": v2["rates"]},
                    "v3": {"counts": test["counts"], "rates": test["rates"]},
                },
                "strictCaseImprovementOverV2": 10,
                "requiredStrictCaseImprovementOverV2": 2,
                "absoluteGatePassed": True,
                "authoritySafetyNoRegression": True,
                "comparisonCriteriaSatisfied": True,
                "authenticatedComparisonEnvelopePresent": False,
                "receiptEligible": False,
                "publicationEligible": False,
            }
        )
        self.reports = {
            "childTraining": child,
            "supervisor": supervisor,
            "devEvaluation": dev,
            "testEvaluation": test,
            "baseTestEvaluation": base,
            "v2TestEvaluation": v2,
            "comparison": comparison,
        }
        self.key = Ed25519PrivateKey.generate()
        self.receipts = self.mint_receipts()
        authentication = self.receipts["TRAINING"]["authentication"]
        self.trust_policy = {
            "schema": release.TRUST_POLICY_SCHEMA,
            "candidateId": release.CANDIDATE_ID,
            "algorithm": "Ed25519",
            "keyId": authentication["keyId"],
            "publicKeyFingerprintSha256": authentication[
                "publicKeyFingerprintSha256"
            ],
            "usage": release.TRUST_POLICY_USAGE,
            "state": "ACTIVE",
        }
        self.candidate = {
            "candidate_id": release.CANDIDATE_ID,
            "target_repo_id": release.TARGET_REPO_ID,
            "actual_training_base": {
                "repo_id": "unsloth/Qwen3.5-0.8B",
                "revision": "8" * 40,
                "license": "apache-2.0",
                "runtime": "Unsloth FastVisionModel",
                "load_in_4bit": True,
            },
            "upstream_lineage": {
                "repo_id": "Qwen/Qwen3.5-0.8B",
                "revision": "9" * 40,
                "license": "apache-2.0",
                "relationship": "DECLARED_UPSTREAM_LINEAGE",
                "byte_equivalence_verified": False,
                "claim_boundary": "not byte equivalent",
            },
            "predecessor": {
                "repo_id": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2",
                "release_revision": "e" * 40,
                "adapter_model_sha256": "f" * 64,
                "role": "FROZEN_COMPARATOR_NOT_WEIGHT_INITIALIZATION",
            },
            "training_recipe": {
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "seed": 11,
                "full_optimizer_steps": 135,
            },
        }

    def mint_receipts(self) -> dict[str, dict[str, Any]]:
        training = evidence.mint_training_receipt(
            self.reports["childTraining"],
            self.reports["supervisor"],
            source_revision=SOURCE,
            private_key=self.key,
            key_id="owner-release-key-1",
        )
        evaluation_receipt = evidence.mint_evaluation_receipt(
            self.reports["devEvaluation"],
            self.reports["testEvaluation"],
            training,
            private_key=self.key,
            key_id="owner-release-key-1",
        )
        comparison_receipt = evidence.mint_comparison_receipt(
            self.reports["comparison"],
            evaluation_receipt,
            private_key=self.key,
            key_id="owner-release-key-1",
        )
        return {
            "TRAINING": training,
            "EVALUATION": evaluation_receipt,
            "COMPARISON": comparison_receipt,
        }

    def replace_attested_files(self, files: list[dict[str, Any]]) -> None:
        child = copy.deepcopy(self.reports["childTraining"])
        child.pop("reportSha256")
        child["adapter"]["files"] = copy.deepcopy(files)
        self.reports["childTraining"] = sealed(child)

        supervisor = copy.deepcopy(self.reports["supervisor"])
        supervisor.pop("reportSha256")
        supervisor["adapter"]["files"] = copy.deepcopy(files)
        supervisor["trainingReport"]["canonicalReportSha256"] = self.reports[
            "childTraining"
        ]["reportSha256"]
        self.reports["supervisor"] = sealed(supervisor)

        for name in ("devEvaluation", "testEvaluation"):
            evaluation_report = copy.deepcopy(self.reports[name])
            evaluation_report.pop("reportSha256")
            evaluation_report["trainingReportSha256"] = self.reports[
                "childTraining"
            ]["reportSha256"]
            evaluation_report["supervisionLinkage"]["reportSha256"] = self.reports[
                "supervisor"
            ]["reportSha256"]
            self.reports[name] = sealed(evaluation_report)

        comparison = copy.deepcopy(self.reports["comparison"])
        comparison.pop("reportSha256")
        comparison["inputReports"]["v3"] = self.reports["testEvaluation"][
            "reportSha256"
        ]
        self.reports["comparison"] = sealed(comparison)
        self.receipts = self.mint_receipts()

    def build(self) -> tuple[dict[str, Any], dict[str, bytes]]:
        with mock.patch.object(
            release,
            "load_committed_trust_policy",
            return_value=copy.deepcopy(self.trust_policy),
        ):
            return release.build_release_packet(
                candidate=self.candidate,
                source_revision=SOURCE,
                receipts=self.receipts,
                reports=self.reports,
                adapter_dir=self.adapter,
                trusted_public_key=self.key.public_key(),
            )


class PrepareReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_packet_is_deterministic_and_truthful(self) -> None:
        first_manifest, first_files = self.fixture.build()
        second_manifest, second_files = self.fixture.build()
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_files, second_files)
        unsigned = dict(first_manifest)
        digest = unsigned.pop("manifestSha256")
        self.assertEqual(digest, release.sha256_json(unsigned))
        self.assertFalse(first_manifest["publicationEligible"])
        self.assertFalse(first_manifest["runtimeValidated"])
        self.assertFalse(first_manifest["thirdPartyValidated"])
        card = first_files["README.md"].decode("utf-8")
        self.assertIn("immutable hub bytes", card)
        self.assertIn("Publication eligible: `false`", card)
        self.assertEqual(
            {
                "repoId": release.TARGET_REPO_ID,
                "repoType": "model",
                "private": False,
                "declaredOnly": True,
                "immutableRevision": None,
                "immutableReadbackVerified": False,
            },
            first_manifest["targetRepository"],
        )
        self.assertTrue(first_manifest["releaseManifestIsTargetRepositoryFile"])
        public_paths = [
            item["path"] for item in first_manifest["intendedRepositoryFiles"]
        ]
        local_paths = [item["path"] for item in first_manifest["localEvidenceFiles"]]
        self.assertEqual(public_paths, sorted(public_paths))
        self.assertEqual(local_paths, sorted(local_paths))
        self.assertEqual(set(first_files), set(public_paths) | set(local_paths))
        self.assertTrue(set(public_paths).isdisjoint(local_paths))
        self.assertNotIn("release-manifest.json", public_paths)
        self.assertEqual(
            "NOT_FOR_PUBLICATION", first_manifest["localEvidenceDisposition"]
        )
        self.assertTrue(
            all(
                item["disposition"] == "NOT_FOR_PUBLICATION"
                for item in first_manifest["localEvidenceFiles"]
            )
        )
        self.assertEqual(
            {name for name, _role in release.REPORT_OUTPUTS.values()},
            set(local_paths),
        )
        self.assertTrue(
            {name for name, _role in release.RECEIPT_OUTPUTS.values()}.issubset(
                public_paths
            )
        )
        self.assertTrue(
            all(
                set(item) == {"path", "bytes", "sha256"}
                for item in first_manifest["adapter"]["sourceFiles"]
            )
        )

    def test_authenticated_adapter_metadata_contract_is_suffix_sensitive(self) -> None:
        cases = (
            (
                "missing-json-count",
                lambda files: files[0].pop("jsonKeys"),
                "keys differ",
            ),
            (
                "wrong-weight-count-key",
                lambda files: (
                    files[1].pop("tensorCount"),
                    files[1].__setitem__("jsonKeys", 1),
                ),
                "keys differ",
            ),
            (
                "boolean-json-count",
                lambda files: files[0].__setitem__("jsonKeys", False),
                "non-negative integer",
            ),
            (
                "zero-tensor-count",
                lambda files: files[1].__setitem__("tensorCount", 0),
                "must be positive",
            ),
            (
                "unsorted-paths",
                lambda files: files.reverse(),
                "paths are not sorted",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name):
                case_root = self.root / name
                case_root.mkdir()
                fixture = Fixture(case_root)
                files = copy.deepcopy(
                    fixture.reports["supervisor"]["adapter"]["files"]
                )
                mutate(files)
                fixture.replace_attested_files(files)
                with self.assertRaisesRegex(release.ReleaseError, expected_error):
                    fixture.build()

    def test_zero_json_key_count_is_accepted_when_authenticated(self) -> None:
        files = copy.deepcopy(
            self.fixture.reports["supervisor"]["adapter"]["files"]
        )
        files[0]["jsonKeys"] = 0
        self.fixture.replace_attested_files(files)
        manifest, _packet_files = self.fixture.build()
        self.assertTrue(manifest["authenticatedEvidenceChainValid"])
        self.assertEqual(
            {
                "policySchema": release.TRUST_POLICY_SCHEMA,
                "policySha256": release.sha256_json(self.fixture.trust_policy),
                "candidateId": release.CANDIDATE_ID,
                "algorithm": "Ed25519",
                "keyId": "owner-release-key-1",
                "publicKeyFingerprintSha256": self.fixture.trust_policy[
                    "publicKeyFingerprintSha256"
                ],
                "usage": release.TRUST_POLICY_USAGE,
                "state": "ACTIVE",
            },
            manifest["receiptSigningTrustPolicy"],
        )

    def test_source_adapter_readme_is_preserved_as_a_distinct_file(self) -> None:
        root = self.root / "source-readme-fixture"
        root.mkdir()
        fixture = Fixture(root, include_source_readme=True)
        manifest, packet_files = fixture.build()
        source_readme = b"# Adapter source metadata\n"
        self.assertEqual(source_readme, packet_files["adapter-source-readme.md"])
        self.assertNotEqual(
            packet_files["adapter-source-readme.md"], packet_files["README.md"]
        )
        self.assertEqual(
            "LOCAL_EVIDENCE_ONLY_NOT_FOR_PUBLICATION",
            manifest["adapter"]["sourceReadmeDisposition"],
        )
        public_roles = {
            item["path"]: item["role"]
            for item in manifest["intendedRepositoryFiles"]
        }
        local_roles = {
            item["path"]: item["role"] for item in manifest["localEvidenceFiles"]
        }
        self.assertNotIn("adapter-source-readme.md", public_roles)
        self.assertEqual(
            "ADAPTER_SOURCE_README", local_roles["adapter-source-readme.md"]
        )

    def test_writes_repeatable_bytes_to_distinct_new_directories(self) -> None:
        manifest, packet_files = self.fixture.build()
        one, two = self.root / "packet-one", self.root / "packet-two"
        release.write_packet(one, manifest, packet_files)
        release.write_packet(two, manifest, packet_files)
        self.assertEqual(
            (one / "release-manifest.json").read_bytes(),
            (two / "release-manifest.json").read_bytes(),
        )
        for name, expected in packet_files.items():
            self.assertEqual(expected, (one / name).read_bytes())
            self.assertEqual(expected, (two / name).read_bytes())
        inventory = (
            manifest["intendedRepositoryFiles"] + manifest["localEvidenceFiles"]
        )
        for item in inventory:
            data = (one / item["path"]).read_bytes()
            self.assertEqual(item["bytes"], len(data))
            self.assertEqual(item["sha256"], release.sha256_bytes(data))

    def test_real_ed25519_chain_is_verified_and_staged_canonically(self) -> None:
        receipts = self.fixture.receipts
        manifest, packet_files = self.fixture.build()
        parsed: list[dict[str, Any]] = []
        for kind, (name, _role) in release.RECEIPT_OUTPUTS.items():
            expected = (release.canonical_json(receipts[kind]) + "\n").encode(
                "utf-8"
            )
            self.assertEqual(expected, packet_files[name])
            parsed.append(json.loads(packet_files[name]))
        for report_key, (name, _role) in release.REPORT_OUTPUTS.items():
            expected = (
                release.canonical_json(self.fixture.reports[report_key]) + "\n"
            ).encode("utf-8")
            self.assertEqual(expected, packet_files[name])
        verified = evidence.verify_chain(
            *parsed, trusted_public_key=self.fixture.key.public_key()
        )
        self.assertTrue(verified["authenticatedEvidenceChainValid"])
        self.assertFalse(verified["publicationEligible"])
        output = self.root / "real-chain-packet"
        release.write_packet(output, manifest, packet_files)
        self.assertEqual(
            (self.fixture.adapter / "adapter_model.safetensors").read_bytes(),
            (output / "adapter_model.safetensors").read_bytes(),
        )

    def test_staged_file_tamper_is_rejected_before_output_creation(self) -> None:
        manifest, repository_files = self.fixture.build()
        repository_files = dict(repository_files)
        repository_files["adapter_model.safetensors"] = (
            b"tampered-after-validation"
        )
        output = self.root / "tampered-packet"
        with self.assertRaisesRegex(
            release.ReleaseError, "snapshot digest differs"
        ):
            release.write_packet(output, manifest, repository_files)
        self.assertFalse(output.exists())

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(release.ReleaseError, "duplicate JSON key"):
            release.strict_json_bytes(b'{"schema":"a","schema":"b"}', "fixture")

    def test_unknown_receipt_field_is_rejected(self) -> None:
        self.fixture.receipts["TRAINING"]["surprise"] = True
        with self.assertRaisesRegex(release.ReleaseError, "unknown=.*surprise"):
            self.fixture.build()

    def test_tampered_receipt_payload_is_rejected(self) -> None:
        self.fixture.receipts["TRAINING"]["payload"]["runId"] = "0" * 32
        with self.assertRaisesRegex(release.ReleaseError, "payload digest differs"):
            self.fixture.build()

    def test_missing_measured_report_is_rejected(self) -> None:
        self.fixture.reports.pop("devEvaluation")
        with self.assertRaisesRegex(release.ReleaseError, "exact measured report set"):
            self.fixture.build()

    def test_tampered_measured_report_is_rejected(self) -> None:
        self.fixture.reports["comparison"]["strictCaseImprovementOverV2"] = 999
        with self.assertRaisesRegex(release.ReleaseError, "self-digest differs"):
            self.fixture.build()

    def test_adapter_byte_tamper_is_rejected(self) -> None:
        (self.fixture.adapter / "adapter_model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(release.ReleaseError, "adapter aggregate differs"):
            self.fixture.build()

    def test_authenticated_adapter_config_must_bind_pinned_base_and_recipe(self) -> None:
        root = self.root / "wrong-adapter-revision"
        root.mkdir()
        fixture = Fixture(root, adapter_revision="7" * 40)
        with self.assertRaisesRegex(
            release.ReleaseError, "adapter config revision differs"
        ):
            fixture.build()

    def test_production_shaped_target_module_list_is_accepted(self) -> None:
        manifest, _packet = self.fixture.build()
        self.assertTrue(manifest["authenticatedEvidenceChainValid"])

    def test_target_module_regex_string_is_accepted(self) -> None:
        root = self.root / "string-target-modules"
        root.mkdir()
        fixture = Fixture(root, target_modules="language.*")
        manifest, _packet = fixture.build()
        self.assertTrue(manifest["authenticatedEvidenceChainValid"])

    def test_duplicate_target_modules_are_rejected(self) -> None:
        root = self.root / "duplicate-target-modules"
        root.mkdir()
        fixture = Fixture(root, target_modules=["q_proj", "q_proj"])
        with self.assertRaisesRegex(release.ReleaseError, "contains duplicates"):
            fixture.build()

    def test_malformed_target_modules_are_rejected(self) -> None:
        for index, target_modules in enumerate(([], ["q_proj", ""], [" q_proj"])):
            with self.subTest(target_modules=target_modules):
                root = self.root / f"malformed-target-modules-{index}"
                root.mkdir()
                fixture = Fixture(root, target_modules=target_modules)
                with self.assertRaisesRegex(
                    release.ReleaseError, "empty|malformed"
                ):
                    fixture.build()

    def test_flat_packet_path_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(release.ReleaseError, "not safe and flat"):
            release._safe_flat_name("../escape")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support unavailable")
    def test_adapter_symlink_is_rejected(self) -> None:
        link = self.fixture.adapter / "tokenizer.json"
        try:
            os.symlink(self.fixture.adapter / "adapter_config.json", link)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(release.ReleaseError, "contains a symlink"):
            self.fixture.build()

    def test_output_traversal_and_existing_directory_are_rejected(self) -> None:
        relative = Path("packet") / ".." / "escape"
        with self.assertRaisesRegex(release.ReleaseError, "absolute traversal-free"):
            release._assert_safe_new_output(relative)
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(release.ReleaseError, "must not already exist"):
            release._assert_safe_new_output(existing)

    def test_public_builder_has_no_verifier_injection_surface(self) -> None:
        parameters = inspect.signature(release.build_release_packet).parameters
        self.assertIn("trusted_public_key", parameters)
        self.assertNotIn("chain_verifier", parameters)
        with self.assertRaisesRegex(TypeError, "chain_verifier"):
            release.build_release_packet(
                candidate=self.fixture.candidate,
                source_revision=SOURCE,
                receipts=self.fixture.receipts,
                reports=self.fixture.reports,
                adapter_dir=self.fixture.adapter,
                chain_verifier=lambda *_values: {},
            )

    def test_wrong_trusted_key_cannot_emit_authenticated_claims(self) -> None:
        wrong_key = Ed25519PrivateKey.generate()
        with mock.patch.object(
            release,
            "load_committed_trust_policy",
            return_value=copy.deepcopy(self.fixture.trust_policy),
        ):
            with self.assertRaises(evidence.EvidenceError):
                release.build_release_packet(
                    candidate=self.fixture.candidate,
                    source_revision=SOURCE,
                    receipts=self.fixture.receipts,
                    reports=self.fixture.reports,
                    adapter_dir=self.fixture.adapter,
                    trusted_public_key=wrong_key.public_key(),
                )

    def test_valid_chain_from_unapproved_key_is_rejected_by_source_policy(self) -> None:
        policy = copy.deepcopy(self.fixture.trust_policy)
        policy["publicKeyFingerprintSha256"] = "0" * 64
        with mock.patch.object(
            release, "load_committed_trust_policy", return_value=policy
        ):
            with self.assertRaisesRegex(release.ReleaseError, "not approved"):
                release.build_release_packet(
                    candidate=self.fixture.candidate,
                    source_revision=SOURCE,
                    receipts=self.fixture.receipts,
                    reports=self.fixture.reports,
                    adapter_dir=self.fixture.adapter,
                    trusted_public_key=self.fixture.key.public_key(),
                )

    def test_wrong_receipt_key_id_is_rejected_by_source_policy(self) -> None:
        policy = copy.deepcopy(self.fixture.trust_policy)
        policy["keyId"] = "different-approved-key"
        with mock.patch.object(
            release, "load_committed_trust_policy", return_value=policy
        ):
            with self.assertRaisesRegex(release.ReleaseError, "key ID is not approved"):
                release.build_release_packet(
                    candidate=self.fixture.candidate,
                    source_revision=SOURCE,
                    receipts=self.fixture.receipts,
                    reports=self.fixture.reports,
                    adapter_dir=self.fixture.adapter,
                    trusted_public_key=self.fixture.key.public_key(),
                )

    def test_bool_and_int_gates_are_type_sensitive(self) -> None:
        with self.assertRaisesRegex(release.ReleaseError, "differs"):
            release._exact_typed(1, True, "boolean gate")
        with self.assertRaisesRegex(release.ReleaseError, "non-negative integer"):
            release._exact_nonnegative_int(False, "count")

    def test_non_finite_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(release.ReleaseError, "finite JSON"):
            release.canonical_json({"value": math.nan})

    def test_oversized_json_is_rejected_before_read(self) -> None:
        oversized = self.root / "oversized.json"
        with oversized.open("wb") as handle:
            handle.truncate(release.MAX_JSON_BYTES + 1)
        with mock.patch.object(
            release.os, "read", side_effect=AssertionError("must not allocate/read")
        ):
            with self.assertRaisesRegex(release.ReleaseError, "file size is invalid"):
                release.strict_json_file(oversized, "oversized fixture")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support unavailable")
    def test_json_and_output_symlink_ancestors_are_rejected(self) -> None:
        real = self.root / "real-parent"
        real.mkdir()
        (real / "input.json").write_text("{}", encoding="utf-8")
        linked = self.root / "linked-parent"
        try:
            os.symlink(real, linked, target_is_directory=True)
        except OSError:
            self.skipTest("directory symlink creation is unavailable")
        with self.assertRaisesRegex(
            release.ReleaseError, "symlink, junction, or reparse point"
        ):
            release.strict_json_file(linked / "input.json", "linked JSON")
        with self.assertRaisesRegex(
            release.ReleaseError, "symlink, junction, or reparse point"
        ):
            release._assert_safe_new_output(linked / "packet")

    def test_schema_is_strict_and_pins_false_publication_gates(self) -> None:
        schema = release.strict_json_file(
            release.HERE / "release.schema.json", "release schema"
        )
        Draft202012Validator.check_schema(schema)
        manifest, _packet_files = self.fixture.build()
        self.assertEqual([], list(Draft202012Validator(schema).iter_errors(manifest)))
        self.assertFalse(schema["properties"]["publicationEligible"]["const"])
        self.assertFalse(schema["properties"]["runtimeValidated"]["const"])
        self.assertFalse(schema["properties"]["thirdPartyValidated"]["const"])
        self.assertTrue(
            schema["properties"]["releaseManifestIsTargetRepositoryFile"]["const"]
        )
        target = schema["properties"]["targetRepository"]["properties"]
        self.assertEqual("model", target["repoType"]["const"])
        self.assertFalse(target["private"]["const"])
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["properties"]["lineage"]["additionalProperties"])
        self.assertFalse(schema["properties"]["adapter"]["additionalProperties"])
        self.assertFalse(schema["properties"]["benchmark"]["additionalProperties"])
        self.assertFalse(schema["properties"]["reports"]["additionalProperties"])
        self.assertFalse(schema["properties"]["receipts"]["additionalProperties"])

        mutations = []
        unknown_nested = copy.deepcopy(manifest)
        unknown_nested["lineage"]["implementationBase"]["unknown"] = True
        mutations.append(unknown_nested)
        truthy_integer = copy.deepcopy(manifest)
        truthy_integer["receiptEligible"] = 1
        mutations.append(truthy_integer)
        local_promoted = copy.deepcopy(manifest)
        local_promoted["localEvidenceFiles"][0]["disposition"] = "PUBLIC"
        mutations.append(local_promoted)
        wrong_role = copy.deepcopy(manifest)
        wrong_role["intendedRepositoryFiles"][0]["role"] = (
            "MEASURED_CHILD_TRAINING_REPORT"
        )
        mutations.append(wrong_role)
        missing_gate = copy.deepcopy(manifest)
        missing_gate["unmetPublicationGates"].pop()
        mutations.append(missing_gate)
        for mutation in mutations:
            with self.assertRaises(release.ReleaseError):
                release.validate_release_manifest(mutation)


if __name__ == "__main__":
    unittest.main()
