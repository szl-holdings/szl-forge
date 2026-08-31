from __future__ import annotations

import base64
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema.validators import validator_for

import evidence_chain as evidence


SOURCE = "1" * 40
OTHER_SOURCE = "2" * 40
RUN_ID = "a" * 32
ADAPTER_SHA = "b" * 64
TRAIN_SHA = "c" * 64
DEV_SHA = "d" * 64
TEST_SHA = "e" * 64
BASE_REPORT_SHA = "3" * 64
V2_REPORT_SHA = "4" * 64


def self_digest(report: dict) -> dict:
    report = copy.deepcopy(report)
    report.pop("reportSha256", None)
    report["reportSha256"] = evidence.sha256_json(report)
    return report


def resign_receipt(receipt: dict, key: Ed25519PrivateKey) -> dict:
    """Re-sign an adversarial wrapper so structure checks see a valid signature."""

    wrapper = copy.deepcopy(receipt)
    wrapper["payloadSha256"] = evidence.sha256_json(wrapper["payload"])
    signature = key.sign(
        evidence.canonical_json(evidence._signature_document(wrapper)).encode("utf-8")
    )
    wrapper["authentication"]["signatureBase64"] = base64.b64encode(signature).decode(
        "ascii"
    )
    unsigned = dict(wrapper)
    unsigned.pop("receiptSha256", None)
    wrapper["receiptSha256"] = evidence.sha256_json(unsigned)
    return wrapper


def child_report() -> dict:
    return self_digest(
        {
            "schema": "szl.frontier-training-run/v3",
            "candidateId": evidence.CANDIDATE_ID,
            "supervisorRunId": RUN_ID,
            "state": "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
            "runKind": "FULL",
            "source": {"revision": SOURCE},
            "sourceBundle": {
                "manifestSha256": "5" * 64,
                "trainSha256": TRAIN_SHA,
                "trainBytes": 100,
                "uniqueTrainingRows": 180,
                "kindCounts": {"DRAFT": 60, "RECOVERY": 60, "REFUSAL": 60},
                "heldOutCommitments": {
                    "dev.jsonl": {"rows": 36, "sha256": DEV_SHA},
                    "test.jsonl": {"rows": 72, "sha256": TEST_SHA},
                },
                "trainerOpenedSplitContent": ["TRAIN"],
                "bundleSha256": "6" * 64,
            },
            "runtimePackages": {"torch": "2.10.0", "unsloth": "2026.7.4"},
            "gpu": {
                "uuid": "GPU-1111111111111111",
                "name": "synthetic-test-gpu",
            },
            "adapter": {
                "aggregateSha256": ADAPTER_SHA,
                "files": [
                    {
                        "path": "adapter_model.safetensors",
                        "sha256": "7" * 64,
                        "bytes": 8,
                    }
                ],
            },
            "authenticatedTrainingEnvelopePresent": False,
            "receiptEligible": False,
            "publicationEligible": False,
        }
    )


def supervisor_report(child: dict) -> dict:
    return self_digest(
        {
            "schema": "szl.frontier-training-supervisor/v1",
            "candidateId": evidence.CANDIDATE_ID,
            "runId": RUN_ID,
            "runKind": "FULL",
            "state": "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
            "primaryCause": "SUCCESS",
            "localEvaluationInputBindingSatisfied": True,
            "source": {"revision": SOURCE},
            "identities": {
                "supervisorSourceSha256": "8" * 64,
                "workerSourceSha256": "9" * 64,
                "validatorSourceSha256": "a" * 64,
                "candidateSourceSha256": "b" * 64,
            },
            "containment": {
                "unit": f"szl-ra3-supervisor-{RUN_ID}.service",
                "controlGroup": f"/user.slice/{RUN_ID}",
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
                "aggregateSha256": ADAPTER_SHA,
                "matchesTrainingReport": True,
                "safeTensorsParsed": True,
                "allowlistedFilesOnly": True,
                "symlinksAbsent": True,
                "files": child["adapter"]["files"],
            },
            "authenticatedSupervisorEnvelopePresent": False,
            "receiptEligible": False,
            "publicationEligible": False,
        }
    )


def evaluation_report(split: str, child_sha: str, supervisor_sha: str) -> dict:
    return self_digest(
        {
            "schema": "szl.frontier-eval-run/v3",
            "candidateId": evidence.CANDIDATE_ID,
            "modelKind": "v3",
            "split": split,
            "state": "MEASURED_EVALUATION_COMPLETED_UNATTESTED",
            "source": {"revision": SOURCE},
            "trainingReportSha256": child_sha,
            "model": {"adapterAggregateSha256": ADAPTER_SHA},
            "supervisionLinkage": {
                "runId": RUN_ID,
                "reportSha256": supervisor_sha,
                "adapterAggregateSha256": ADAPTER_SHA,
                "sourceRevision": SOURCE,
            },
            "absoluteGatePassed": True,
            "comparisonEligible": False,
            "authenticatedEvaluationEnvelopePresent": False,
            "receiptEligible": False,
            "publicationEligible": False,
        }
    )


def comparison_report(test_report_sha: str) -> dict:
    return self_digest(
        {
            "schema": "szl.frontier-comparison/v2",
            "candidateId": evidence.CANDIDATE_ID,
            "sourceRevision": SOURCE,
            "state": "UNAUTHENTICATED_COMPARISON_CRITERIA_SATISFIED",
            "inputReports": {
                "base": BASE_REPORT_SHA,
                "v2": V2_REPORT_SHA,
                "v3": test_report_sha,
            },
            "comparisonCriteriaSatisfied": True,
            "absoluteGatePassed": True,
            "authoritySafetyNoRegression": True,
            "authenticatedComparisonEnvelopePresent": False,
            "receiptEligible": False,
            "publicationEligible": False,
        }
    )


class EvidenceChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.child = child_report()
        self.supervisor = supervisor_report(self.child)
        self.dev = evaluation_report(
            "DEV", self.child["reportSha256"], self.supervisor["reportSha256"]
        )
        self.test = evaluation_report(
            "TEST", self.child["reportSha256"], self.supervisor["reportSha256"]
        )
        self.training = evidence.mint_training_receipt(
            self.child,
            self.supervisor,
            source_revision=SOURCE,
            private_key=self.key,
            key_id="owner-test-key",
        )
        self.evaluation = evidence.mint_evaluation_receipt(
            self.dev,
            self.test,
            self.training,
            private_key=self.key,
            key_id="owner-test-key",
        )
        self.comparison_report = comparison_report(self.test["reportSha256"])
        self.comparison = evidence.mint_comparison_receipt(
            self.comparison_report,
            self.evaluation,
            private_key=self.key,
            key_id="owner-test-key",
        )

    def test_complete_chain_verifies_but_remains_publication_ineligible(self) -> None:
        result = evidence.verify_chain(
            self.training,
            self.evaluation,
            self.comparison,
            trusted_public_key=self.key.public_key(),
        )
        self.assertTrue(result["authenticatedEvidenceChainValid"])
        self.assertTrue(result["receiptEligible"])
        self.assertFalse(result["publicationEligible"])
        self.assertEqual(
            [
                self.training["kind"],
                self.evaluation["kind"],
                self.comparison["kind"],
            ],
            ["TRAINING", "EVALUATION", "COMPARISON"],
        )
        self.assertIsNone(self.training["payload"]["previousReceiptSha256"])
        self.assertEqual(
            self.evaluation["payload"]["previousReceiptSha256"],
            self.training["receiptSha256"],
        )
        self.assertEqual(
            self.comparison["payload"]["previousReceiptSha256"],
            self.evaluation["receiptSha256"],
        )

    def test_receipts_match_strict_json_schema(self) -> None:
        schema_path = Path(__file__).parent / "schemas" / "authenticated-receipt.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
        validator = validator_type(schema)
        for receipt in (self.training, self.evaluation, self.comparison):
            validator.validate(receipt)

    def test_payload_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.training)
        tampered["payload"]["childReportSha256"] = "0" * 64
        with self.assertRaisesRegex(evidence.EvidenceError, "payload digest"):
            evidence.verify_receipt(
                tampered, trusted_public_key=self.key.public_key()
            )

    def test_signature_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.training)
        tampered["authentication"]["signatureBase64"] = "A" * 88
        with self.assertRaisesRegex(evidence.EvidenceError, "signature"):
            evidence.verify_receipt(
                tampered, trusted_public_key=self.key.public_key()
            )

    def test_wrong_trusted_key_is_rejected(self) -> None:
        wrong = Ed25519PrivateKey.generate().public_key()
        with self.assertRaisesRegex(evidence.EvidenceError, "trusted public key"):
            evidence.verify_chain(
                self.training,
                self.evaluation,
                self.comparison,
                trusted_public_key=wrong,
            )

    def test_chain_verification_requires_a_trust_anchor(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "trusted Ed25519"):
            evidence.verify_chain(
                self.training,
                self.evaluation,
                self.comparison,
            )
        inspected = evidence.verify_receipt(self.training)
        self.assertTrue(inspected["signatureValid"])
        self.assertFalse(inspected["trustedKeyMatched"])

    def test_wrong_source_is_rejected_before_signing(self) -> None:
        wrong_source = copy.deepcopy(self.dev)
        wrong_source["source"]["revision"] = OTHER_SOURCE
        wrong_source = self_digest(wrong_source)
        with self.assertRaisesRegex(evidence.EvidenceError, "evaluation source"):
            evidence.mint_evaluation_receipt(
                wrong_source,
                self.test,
                self.training,
                private_key=self.key,
                key_id="owner-test-key",
            )

    def test_wrong_adapter_is_rejected_before_signing(self) -> None:
        wrong_adapter = copy.deepcopy(self.test)
        wrong_adapter["model"]["adapterAggregateSha256"] = "f" * 64
        wrong_adapter = self_digest(wrong_adapter)
        with self.assertRaisesRegex(evidence.EvidenceError, "adapter binding"):
            evidence.mint_evaluation_receipt(
                self.dev,
                wrong_adapter,
                self.training,
                private_key=self.key,
                key_id="owner-test-key",
            )

    def test_child_supervisor_adapter_disagreement_is_rejected(self) -> None:
        wrong_child = copy.deepcopy(self.child)
        wrong_child["adapter"]["aggregateSha256"] = "f" * 64
        wrong_child = self_digest(wrong_child)
        wrong_supervisor = copy.deepcopy(self.supervisor)
        wrong_supervisor["trainingReport"]["canonicalReportSha256"] = wrong_child[
            "reportSha256"
        ]
        wrong_supervisor = self_digest(wrong_supervisor)
        with self.assertRaisesRegex(evidence.EvidenceError, "adapter aggregate"):
            evidence.mint_training_receipt(
                wrong_child,
                wrong_supervisor,
                source_revision=SOURCE,
                private_key=self.key,
                key_id="owner-test-key",
            )

    def test_replayed_comparison_after_new_evaluation_is_rejected(self) -> None:
        new_dev = copy.deepcopy(self.dev)
        new_dev["measuredAt"] = "synthetic-rerun"
        new_dev = self_digest(new_dev)
        new_evaluation = evidence.mint_evaluation_receipt(
            new_dev,
            self.test,
            self.training,
            private_key=self.key,
            key_id="owner-test-key",
        )
        with self.assertRaisesRegex(evidence.EvidenceError, "comparison chain link"):
            evidence.verify_chain(
                self.training,
                new_evaluation,
                self.comparison,
                trusted_public_key=self.key.public_key(),
            )

    def test_out_of_order_receipts_are_rejected(self) -> None:
        with self.assertRaisesRegex(evidence.EvidenceError, "receipt kind"):
            evidence.verify_chain(
                self.evaluation,
                self.training,
                self.comparison,
                trusted_public_key=self.key.public_key(),
            )

    def test_malformed_and_extra_fields_are_rejected(self) -> None:
        malformed = copy.deepcopy(self.training)
        malformed["unexpected"] = True
        with self.assertRaisesRegex(evidence.EvidenceError, "receipt keys differ"):
            evidence.verify_receipt(
                malformed, trusted_public_key=self.key.public_key()
            )

    def test_signed_integer_boolean_gates_are_rejected(self) -> None:
        false_as_zero = copy.deepcopy(self.training)
        false_as_zero["payload"]["publicationEligible"] = 0
        false_as_zero = resign_receipt(false_as_zero, self.key)
        with self.assertRaisesRegex(evidence.EvidenceError, "publication flag"):
            evidence.verify_receipt(
                false_as_zero,
                trusted_public_key=self.key.public_key(),
            )

        true_as_one = copy.deepcopy(self.comparison)
        true_as_one["payload"]["comparisonCriteriaSatisfied"] = 1
        true_as_one = resign_receipt(true_as_one, self.key)
        with self.assertRaisesRegex(evidence.EvidenceError, "comparison criteria"):
            evidence.verify_receipt(
                true_as_one,
                trusted_public_key=self.key.public_key(),
            )

    def test_signed_malformed_comparison_input_digests_are_rejected(self) -> None:
        malformed_values = {
            "base": "A" * 64,
            "v2": "4" * 63,
            "v3": 1,
        }
        for name, malformed_value in malformed_values.items():
            with self.subTest(name=name):
                malformed = copy.deepcopy(self.comparison)
                malformed["payload"]["inputReportSha256s"][name] = malformed_value
                malformed = resign_receipt(malformed, self.key)
                with self.assertRaisesRegex(
                    evidence.EvidenceError,
                    f"comparison input {name}",
                ):
                    evidence.verify_receipt(
                        malformed,
                        trusted_public_key=self.key.public_key(),
                    )

    def test_oversized_json_and_key_inputs_are_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized_json = root / "oversized.json"
            with oversized_json.open("wb") as handle:
                handle.truncate(evidence.MAX_JSON_BYTES + 1)
            with self.assertRaisesRegex(evidence.EvidenceError, "exceeds"):
                evidence.load_json(oversized_json)

            oversized_key = root / "oversized.key"
            with oversized_key.open("wb") as handle:
                handle.truncate(evidence.MAX_KEY_BYTES + 1)
            with self.assertRaisesRegex(evidence.EvidenceError, "exceeds"):
                evidence.verify_receipt(
                    self.training,
                    trusted_public_key=oversized_key,
                )

    def test_symlink_or_reparse_json_and_key_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text('{"ok":true}', encoding="utf-8")
            alias = root / "alias.json"
            try:
                os.symlink(target, alias)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"filesystem symlinks unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(
                evidence.EvidenceError,
                "symlink or reparse point",
            ):
                evidence.load_json(alias)
            with self.assertRaisesRegex(
                evidence.EvidenceError,
                "symlink or reparse point",
            ):
                evidence.verify_receipt(
                    self.training,
                    trusted_public_key=alias,
                )

    def test_symlink_or_reparse_path_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            target_dir.mkdir()
            (target_dir / "report.json").write_text('{"ok":true}', encoding="utf-8")
            alias_dir = root / "alias"
            try:
                os.symlink(target_dir, alias_dir, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(
                    f"directory symlinks/reparse points unavailable: {type(exc).__name__}"
                )
            with self.assertRaisesRegex(
                evidence.EvidenceError,
                "symlink or reparse point",
            ):
                evidence.load_json(alias_dir / "report.json")

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
            with self.assertRaisesRegex(evidence.EvidenceError, "duplicate JSON key"):
                evidence.load_json(path)


if __name__ == "__main__":
    unittest.main()
