from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "publish_receiptagent_v3", HERE / "publish_receiptagent_v3.py"
)
publisher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)

evidence, release = publisher._release_modules()
SOURCE = "a" * 40
RUN_ID = "b" * 32


def test_private_key():
    return Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))


TEST_PUBLIC_DER = test_private_key().public_key().public_bytes(
    serialization.Encoding.DER,
    serialization.PublicFormat.SubjectPublicKeyInfo,
)
TEST_TRUST_POLICY = {
    "schema": "szl.receiptagent-v3-receipt-signing-trust-policy/v1",
    "candidateId": publisher.CANDIDATE_ID,
    "algorithm": "Ed25519",
    "keyId": "publisher-integration-key",
    "publicKeyFingerprintSha256": publisher.sha256_bytes(TEST_PUBLIC_DER),
    "usage": "AUTHENTICATED_TRAINING_EVALUATION_COMPARISON_RECEIPTS",
    "state": "ACTIVE",
}
TEST_TRUST_POLICY_SHA256 = publisher.sha256_bytes(
    publisher.canonical_json(TEST_TRUST_POLICY)
)


def sealed(report):
    value = copy.deepcopy(report)
    value.pop("reportSha256", None)
    value["reportSha256"] = release.sha256_json(value)
    return value


def candidate_document():
    return {
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


def make_packet(root: Path) -> tuple[Path, Path]:
    adapter = root / "adapter-source"
    adapter.mkdir()
    (adapter / "README.md").write_text("# original adapter card\n", encoding="utf-8")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unsloth/Qwen3.5-0.8B",
                "revision": "8" * 40,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0,
                "bias": "none",
                "inference_mode": True,
                "use_rslora": False,
                "target_modules": "language.*",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"deterministic-adapter")
    aggregate, adapter_files, _adapter_bytes = release.adapter_inventory(adapter)
    attested_adapter_files = []
    for item in adapter_files:
        attested = dict(item)
        if item["path"].endswith(".json"):
            parsed = json.loads(
                (adapter / item["path"]).read_text(encoding="utf-8")
            )
            attested["jsonKeys"] = len(parsed)
        elif item["path"].endswith(".safetensors"):
            attested["tensorCount"] = 1
        attested_adapter_files.append(attested)

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
                "trainBytes": 100,
                "uniqueTrainingRows": 180,
                "kindCounts": {"DRAFT": 60, "RECOVERY": 60, "REFUSAL": 60},
                "heldOutCommitments": {
                    "dev.jsonl": {"rows": 36, "sha256": "2" * 64},
                    "test.jsonl": {"rows": 72, "sha256": "3" * 64},
                },
                "trainerOpenedSplitContent": ["TRAIN"],
                "bundleSha256": "4" * 64,
            },
            "runtimePackages": {"torch": "2.10.0", "unsloth": "2026.7.4"},
            "gpu": {"uuid": "GPU-1111111111111111", "name": "synthetic-test-gpu"},
            "adapter": {
                "aggregateSha256": aggregate,
                "files": attested_adapter_files,
            },
            "qualificationEligible": True,
            "authenticatedTrainingEnvelopePresent": False,
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
            "localEvaluationInputBindingSatisfied": True,
            "source": {"revision": SOURCE},
            "identities": {
                "supervisorSourceSha256": "5" * 64,
                "workerSourceSha256": "6" * 64,
                "validatorSourceSha256": "7" * 64,
                "candidateSourceSha256": "8" * 64,
            },
            "containment": {"unit": f"szl-ra3-supervisor-{RUN_ID}.service"},
            "telemetry": {"gpuUuid": "GPU-1111111111111111"},
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
                "files": attested_adapter_files,
            },
            "authenticatedSupervisorEnvelopePresent": False,
            "receiptEligible": False,
            "publicationEligible": False,
        }
    )

    def evaluation(kind: str, split: str, authenticated_input: bool = False):
        model = {"kind": kind}
        linkage = {}
        training_sha = None
        if authenticated_input:
            model["adapterAggregateSha256"] = aggregate
            linkage = {
                "runId": RUN_ID,
                "reportSha256": supervisor["reportSha256"],
                "adapterAggregateSha256": aggregate,
                "sourceRevision": SOURCE,
            }
            training_sha = child["reportSha256"]
        return sealed(
            {
                "schema": "szl.frontier-eval-run/v3",
                "candidateId": release.CANDIDATE_ID,
                "modelKind": kind,
                "split": split,
                "state": "MEASURED_EVALUATION_COMPLETED_UNATTESTED",
                "source": {"revision": SOURCE},
                "model": model,
                "trainingReportSha256": training_sha,
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
            "protocolSha256": "9" * 64,
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
    reports = {
        "childTraining": child,
        "supervisor": supervisor,
        "devEvaluation": dev,
        "testEvaluation": test,
        "baseTestEvaluation": base,
        "v2TestEvaluation": v2,
        "comparison": comparison,
    }

    key = test_private_key()
    training = evidence.mint_training_receipt(
        child,
        supervisor,
        source_revision=SOURCE,
        private_key=key,
        key_id="publisher-integration-key",
    )
    evaluation_receipt = evidence.mint_evaluation_receipt(
        dev,
        test,
        training,
        private_key=key,
        key_id="publisher-integration-key",
    )
    comparison_receipt = evidence.mint_comparison_receipt(
        comparison,
        evaluation_receipt,
        private_key=key,
        key_id="publisher-integration-key",
    )
    receipts = {
        "TRAINING": training,
        "EVALUATION": evaluation_receipt,
        "COMPARISON": comparison_receipt,
    }

    with mock.patch.object(
        release,
        "load_committed_trust_policy",
        return_value=copy.deepcopy(TEST_TRUST_POLICY),
    ):
        manifest_document, packet_files = release.build_release_packet(
            candidate=candidate_document(),
            source_revision=SOURCE,
            receipts=receipts,
            reports=reports,
            adapter_dir=adapter,
            trusted_public_key=key.public_key(),
        )
    packet = root / "packet"
    release.write_packet(packet, manifest_document, packet_files)
    trusted_key = root / "trusted-public-key.pem"
    trusted_key.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return packet / "release-manifest.json", root / "publication-receipt.json"


def candidate_loader(source_revision):
    if source_revision != SOURCE:
        raise AssertionError(source_revision)
    return candidate_document()


def source_blob(source_revision, relative_path):
    if publisher.FULL_SHA_RE.fullmatch(source_revision or "") is None:
        raise AssertionError(source_revision)
    return (publisher.REPOSITORY / relative_path).read_bytes()


def packet_key(manifest):
    return manifest.parent.parent / "trusted-public-key.pem"


def prepare_packet(manifest, receipt):
    return publisher.prepare_release(
        manifest,
        receipt_path=receipt,
        trusted_public_key=packet_key(manifest),
        expected_source_revision=SOURCE,
        candidate_loader=candidate_loader,
    )


def run_packet(*, manifest_path, receipt_path, publish, **kwargs):
    return publisher.run(
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        publish=publish,
        trusted_public_key=packet_key(manifest_path),
        expected_source_revision=SOURCE,
        candidate_loader=candidate_loader,
        **kwargs,
    )


def reseal_manifest(path, document):
    document.pop("manifestSha256", None)
    document["manifestSha256"] = publisher.sha256_bytes(
        publisher.compact_json(document)
    )
    path.write_bytes(publisher.canonical_json(document))


class FakeApi:
    def __init__(
        self,
        root: Path,
        *,
        exists: bool = True,
        wrong_type: str | None = None,
        partial: bool = False,
        private: bool = False,
        raise_after_commit: bool = False,
        post_main: str | None = None,
        post_private: bool | None = None,
    ) -> None:
        self.root = root
        self.download_temporary = tempfile.TemporaryDirectory()
        self.download_root = Path(self.download_temporary.name)
        self.exists = exists
        self.wrong_type = wrong_type
        self.partial = partial
        self.private = private
        self.raise_after_commit = raise_after_commit
        self.post_main = post_main
        self.post_private = post_private
        self.current = "a" * 40
        self.created = 0
        self.commits = 0
        self.downloads = 0
        self.repo_info_calls = 0
        self.metadata_size_overrides: dict[tuple[str, str], int] = {}
        self.operation_paths: list[str] = []
        self.remote: dict[str, dict[str, bytes]] = {
            self.current: {".gitattributes": b"*.safetensors filter=lfs\n"}
        }

    def __del__(self):
        temporary = getattr(self, "download_temporary", None)
        if temporary is not None:
            temporary.cleanup()

    def repo_exists(self, *, repo_id, repo_type, token):
        self._common(repo_id, token)
        if repo_type == "model":
            return self.exists
        return not self.exists and repo_type == self.wrong_type

    def create_repo(self, *, repo_id, repo_type, private, exist_ok, token):
        self._common(repo_id, token)
        if repo_type != "model" or exist_ok or not isinstance(private, bool):
            raise AssertionError("unsafe create_repo contract")
        self.exists = True
        self.private = private
        self.created += 1

    def repo_info(self, *, repo_id, repo_type, revision, token):
        self._common(repo_id, token)
        if repo_type != "model" or revision != "main" or not self.exists:
            raise AssertionError("missing model")
        self.repo_info_calls += 1
        postcondition = self.repo_info_calls > 1
        return SimpleNamespace(
            id=publisher.TARGET_REPO_ID,
            sha=(
                self.post_main
                if postcondition and self.post_main is not None
                else self.current
            ),
            private=(
                self.post_private
                if postcondition and self.post_private is not None
                else self.private
            ),
        )

    def list_repo_files(self, *, repo_id, repo_type, revision, token):
        self._common(repo_id, token)
        if repo_type != "model":
            raise AssertionError(repo_type)
        return sorted(self.remote[revision])

    def get_paths_info(self, *, repo_id, paths, repo_type, revision, token):
        self._common(repo_id, token)
        if repo_type != "model":
            raise AssertionError(repo_type)
        return [
            SimpleNamespace(
                path=path,
                type="file",
                size=self.metadata_size_overrides.get(
                    (revision, path), len(self.remote[revision][path])
                ),
            )
            for path in paths
            if path in self.remote[revision]
        ]

    def create_commit(
        self,
        *,
        repo_id,
        repo_type,
        revision,
        parent_commit,
        operations,
        token,
        commit_message,
    ):
        self._common(repo_id, token)
        if repo_type != "model" or revision != "main" or parent_commit != self.current:
            raise AssertionError("CAS contract weakened")
        if "authenticated ReceiptAgent v3" not in commit_message:
            raise AssertionError(commit_message)
        updated = dict(self.remote[parent_commit])
        selected = list(operations)
        self.operation_paths = [operation.path_in_repo for operation in selected]
        if self.partial:
            selected = selected[:-1]
        for operation in selected:
            source = operation.path_or_fileobj
            source.seek(0)
            updated[operation.path_in_repo] = source.read()
        self.current = "b" * 40
        self.remote[self.current] = updated
        self.commits += 1
        if self.raise_after_commit:
            raise OSError("injected ambiguous commit response")
        return SimpleNamespace(oid=self.current)

    def downloader(self, *, repo_id, filename, repo_type, revision, token):
        self._common(repo_id, token)
        if repo_type != "model":
            raise AssertionError(repo_type)
        self.downloads += 1
        destination = self.download_root / revision / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.remote[revision][filename])
        return str(destination)

    @staticmethod
    def _common(repo_id, token):
        if repo_id != publisher.TARGET_REPO_ID or token != "scoped-token":
            raise AssertionError("target or token mismatch")


class PublishReceiptAgentV3Tests(unittest.TestCase):
    def setUp(self):
        self.source_binding_patch = mock.patch.object(
            publisher, "_git_blob", side_effect=source_blob
        )
        self.source_binding_patch.start()
        self.addCleanup(self.source_binding_patch.stop)
        self.trust_policy_patch = mock.patch.object(
            publisher,
            "_load_trust_policy",
            return_value=(copy.deepcopy(TEST_TRUST_POLICY), TEST_TRUST_POLICY_SHA256),
        )
        self.trust_policy_patch.start()
        self.addCleanup(self.trust_policy_patch.stop)

    def assert_failure_receipt(
        self,
        receipt: Path,
        *,
        status: str,
        phase: str,
        published_revision: str | None,
    ) -> dict:
        document = json.loads(receipt.read_bytes())
        self.assertEqual(document["schema"], publisher.FAILURE_RECEIPT_SCHEMA)
        self.assertEqual(document["status"], status)
        self.assertEqual(document["failurePhase"], phase)
        self.assertEqual(document["publishedRevision"], published_revision)
        digest = document.pop("receiptSha256")
        self.assertEqual(
            digest, publisher.sha256_bytes(publisher.canonical_json(document))
        )
        return document

    def test_isolated_workflow_import_restores_exact_module_registration(self):
        script = "\n".join(
            (
                "import sys",
                "from types import ModuleType",
                "import publish_receiptagent_v3 as publisher",
                "sentinel = ModuleType('evidence_chain')",
                "sys.modules['evidence_chain'] = sentinel",
                "evidence_module, release_module = publisher._release_modules()",
                "assert sys.modules['evidence_chain'] is sentinel",
                "assert release_module.evidence is evidence_module",
            )
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "tools"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=HERE.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_dry_run_is_local_only_and_validates_exact_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            result = run_packet(
                manifest_path=manifest,
                receipt_path=receipt,
                publish=False,
            )
            self.assertEqual(result["status"], "LOCAL_RELEASE_PACKET_VERIFIED")
            self.assertEqual(result["networkAccess"], "NOT_PERFORMED")
            self.assertFalse(receipt.exists())
            document = json.loads(manifest.read_bytes())
            expected_public = {
                item["path"] for item in document["intendedRepositoryFiles"]
            } | {"release-manifest.json"}
            expected_local = {
                item["path"] for item in document["localEvidenceFiles"]
            }
            self.assertEqual(
                {item["path"] for item in result["files"]}, expected_public
            )
            self.assertEqual(result["localEvidenceDisposition"], "NOT_FOR_PUBLICATION")
            self.assertEqual(
                {item["path"] for item in result["localEvidenceFiles"]}, expected_local
            )
            self.assertTrue(
                all(
                    item["publicationDisposition"] == "NOT_FOR_PUBLICATION"
                    for item in result["localEvidenceFiles"]
                )
            )
            self.assertTrue(expected_public.isdisjoint(expected_local))
            self.assertEqual(result["sourceRevision"], SOURCE)
            self.assertEqual(
                [item["path"] for item in result["sourceByteBindings"]],
                list(publisher.SOURCE_BOUND_PATHS),
            )
            self.assertEqual(
                result["receiptSigningTrustPolicy"]["keyId"],
                "publisher-integration-key",
            )

    def test_committed_trust_policy_authorizes_key_and_fingerprint(self):
        body = publisher.canonical_json(TEST_TRUST_POLICY)
        with mock.patch.object(publisher, "_git_blob", return_value=body):
            policy, digest = publisher._load_trust_policy(SOURCE)
        self.assertEqual(policy, TEST_TRUST_POLICY)
        self.assertEqual(digest, publisher.sha256_bytes(body))

        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            unauthorized = copy.deepcopy(TEST_TRUST_POLICY)
            unauthorized["publicKeyFingerprintSha256"] = "f" * 64
            document = json.loads(manifest.read_bytes())
            document["receiptSigningTrustPolicy"] = {
                "policySchema": unauthorized["schema"],
                "policySha256": publisher.sha256_bytes(
                    publisher.compact_json(unauthorized)
                ),
                "candidateId": unauthorized["candidateId"],
                "algorithm": unauthorized["algorithm"],
                "keyId": unauthorized["keyId"],
                "publicKeyFingerprintSha256": unauthorized[
                    "publicKeyFingerprintSha256"
                ],
                "usage": unauthorized["usage"],
                "state": unauthorized["state"],
            }
            reseal_manifest(manifest, document)
            with mock.patch.object(
                publisher,
                "_load_trust_policy",
                return_value=(
                    unauthorized,
                    publisher.sha256_bytes(publisher.canonical_json(unauthorized)),
                ),
            ):
                with self.assertRaisesRegex(
                    publisher.PublicationError, "not authorized by source policy"
                ):
                    prepare_packet(manifest, receipt)

    def test_manifest_trust_policy_identity_must_match_exact_source_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            document = json.loads(manifest.read_bytes())
            document["receiptSigningTrustPolicy"]["policySha256"] = "f" * 64
            reseal_manifest(manifest, document)

            with self.assertRaisesRegex(
                publisher.PublicationError, "differs from exact source policy"
            ):
                prepare_packet(manifest, receipt)

    def test_manifest_filename_and_exact_source_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            renamed = manifest.with_name("Release-Manifest.json")
            manifest.replace(renamed)
            with self.assertRaisesRegex(publisher.PublicationError, "filename"):
                publisher.prepare_release(
                    renamed,
                    receipt_path=receipt,
                    trusted_public_key=packet_key(renamed),
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )

        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            with mock.patch.object(
                publisher, "_git_blob", return_value=b"one-byte-drift"
            ):
                with self.assertRaisesRegex(
                    publisher.PublicationError, "local source differs"
                ):
                    prepare_packet(manifest, receipt)

    def test_target_and_remote_repo_type_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            document = json.loads(manifest.read_bytes())
            document["targetRepository"]["repoType"] = "dataset"
            reseal_manifest(manifest, document)
            with self.assertRaisesRegex(publisher.PublicationError, "repoType"):
                prepare_packet(manifest, receipt)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, exists=False, wrong_type="dataset")
            with self.assertRaisesRegex(publisher.PublicationError, "dataset"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.created, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, exists=False)
            with self.assertRaisesRegex(
                publisher.PublicationError, "initialize it in a separately authorized"
            ):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.created, 0)

    def test_publish_requires_token_and_explicit_expected_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            with self.assertRaisesRegex(publisher.PublicationError, "expected-parent"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    token="scoped-token",
                    api=FakeApi(root),
                )
            with self.assertRaisesRegex(publisher.PublicationError, "HF_TOKEN"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token=None,
                    api=FakeApi(root),
                )

    def test_existing_repository_visibility_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, private=True)
            with self.assertRaisesRegex(publisher.PublicationError, "visibility differs"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.commits, 0)
            self.assertFalse(receipt.exists())
            self.assertFalse(publisher._info_private({"private": False}))
            self.assertTrue(
                publisher._info_private(SimpleNamespace(private=True))
            )

    def test_local_and_parent_drift_fail_before_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            (manifest.parent / "README.md").write_bytes(b"drift")
            with self.assertRaisesRegex(publisher.PublicationError, "drift"):
                prepare_packet(manifest, receipt)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)
            with self.assertRaisesRegex(publisher.PublicationError, "parent drift"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="9" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.commits, 0)

    def test_partial_upload_writes_post_commit_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, partial=True)
            with self.assertRaisesRegex(publisher.PublicationError, "missing path"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.commits, 1)
            failure = self.assert_failure_receipt(
                receipt,
                status="REMOTE_COMMIT_UNVERIFIED",
                phase="IMMUTABLE_READBACK",
                published_revision="b" * 40,
            )
            self.assertTrue(failure["committed"])
            self.assertTrue(failure["mutationAttempted"])

    def test_success_accounts_for_gitattributes_and_reads_every_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)
            result = run_packet(
                manifest_path=manifest,
                receipt_path=receipt,
                publish=True,
                expected_parent_revision="a" * 40,
                token="scoped-token",
                api=api,
                downloader=api.downloader,
            )
            self.assertEqual(result["publishedRevision"], "b" * 40)
            self.assertEqual(
                result["gitattributes"]["accounting"],
                "GITATTRIBUTES_READBACK_VERIFIED_UNATTRIBUTED",
            )
            document = json.loads(manifest.read_bytes())
            expected_public = {
                item["path"] for item in document["intendedRepositoryFiles"]
            } | {"release-manifest.json"}
            expected_local = {
                item["path"] for item in document["localEvidenceFiles"]
            }
            remote_paths = set(api.remote[api.current]) - {".gitattributes"}
            self.assertEqual({item["path"] for item in result["files"]}, expected_public)
            self.assertEqual(set(api.operation_paths), expected_public)
            self.assertEqual(remote_paths, expected_public)
            self.assertTrue(expected_local.isdisjoint(api.operation_paths))
            self.assertTrue(expected_local.isdisjoint(remote_paths))
            self.assertTrue(
                {
                    "README.md",
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "training-receipt.json",
                    "evaluation-receipt.json",
                    "comparison-receipt.json",
                    "release-manifest.json",
                }.issubset(remote_paths)
            )
            self.assertIn("adapter-source-readme.md", expected_local)
            self.assertIn("training-report.json", expected_local)
            self.assertEqual(json.loads(receipt.read_bytes()), result)
            self.assertEqual(result["portfolioMutation"], "NOT_PERFORMED")

    def test_immutable_readback_mismatch_writes_post_commit_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)

            def corrupt(**kwargs):
                downloaded = Path(api.downloader(**kwargs))
                if kwargs["filename"] == "README.md" and kwargs["revision"] == "b" * 40:
                    downloaded.write_bytes(b"corrupt")
                return str(downloaded)

            with self.assertRaisesRegex(
                publisher.PublicationError, "readback mismatch|size drift"
            ):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=corrupt,
                )
            self.assertEqual(api.commits, 1)
            self.assert_failure_receipt(
                receipt,
                status="REMOTE_COMMIT_UNVERIFIED",
                phase="IMMUTABLE_READBACK",
                published_revision="b" * 40,
            )

    def test_ambiguous_create_outcome_writes_unknown_mutation_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, raise_after_commit=True)
            with self.assertRaisesRegex(OSError, "ambiguous commit response"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.commits, 1)
            failure = self.assert_failure_receipt(
                receipt,
                status="REMOTE_MUTATION_RESULT_UNKNOWN",
                phase="COMMIT_INVOCATION",
                published_revision=None,
            )
            self.assertIsNone(failure["committed"])

    def test_post_commit_main_drift_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, post_main="c" * 40)
            with self.assertRaisesRegex(publisher.PublicationError, "main advanced"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            failure = self.assert_failure_receipt(
                receipt,
                status="REMOTE_COMMIT_UNVERIFIED",
                phase="MAIN_REVISION_POSTCONDITION",
                published_revision="b" * 40,
            )
            self.assertEqual(failure["observedMainRevision"], "c" * 40)
            self.assertFalse(failure["defaultRevisionVerified"])

    def test_post_commit_visibility_drift_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root, post_private=True)
            with self.assertRaisesRegex(publisher.PublicationError, "visibility changed"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            failure = self.assert_failure_receipt(
                receipt,
                status="REMOTE_COMMIT_UNVERIFIED",
                phase="VISIBILITY_POSTCONDITION",
                published_revision="b" * 40,
            )
            self.assertTrue(failure["defaultRevisionVerified"])
            self.assertFalse(failure["visibilityVerified"])

    def test_parent_metadata_size_drift_fails_before_download_or_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            prepared = prepare_packet(manifest, receipt)
            api = FakeApi(root)
            api.remote[api.current].update(
                {item.path: item.body for item in prepared.files}
            )
            api.metadata_size_overrides[(api.current, "README.md")] = (
                len(api.remote[api.current]["README.md"]) + 1
            )
            with self.assertRaisesRegex(publisher.PublicationError, "metadata size drift"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.downloads, 0)
            self.assertEqual(api.commits, 0)
            self.assertFalse(receipt.exists())

    def test_post_commit_metadata_size_drift_fails_before_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)
            api.metadata_size_overrides[("b" * 40, "README.md")] = 1
            with self.assertRaisesRegex(publisher.PublicationError, "metadata size drift"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.downloads, 0)
            self.assertEqual(api.commits, 1)
            self.assert_failure_receipt(
                receipt,
                status="REMOTE_COMMIT_UNVERIFIED",
                phase="IMMUTABLE_READBACK",
                published_revision="b" * 40,
            )

    def test_exact_replay_is_noop_and_rewrites_local_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)
            first = run_packet(
                manifest_path=manifest,
                receipt_path=receipt,
                publish=True,
                expected_parent_revision="a" * 40,
                token="scoped-token",
                api=api,
                downloader=api.downloader,
            )
            receipt.unlink()
            second = run_packet(
                manifest_path=manifest,
                receipt_path=receipt,
                publish=True,
                expected_parent_revision=first["publishedRevision"],
                token="scoped-token",
                api=api,
                downloader=api.downloader,
            )
            self.assertEqual(api.commits, 1)
            self.assertEqual(second["status"], "REPLAY_READBACK_VERIFIED")
            self.assertEqual(second["publishedRevision"], first["publishedRevision"])

    def test_strict_manifest_rejects_duplicate_nonfinite_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            original = manifest.read_text(encoding="utf-8")
            duplicate = original.replace(
                "{\n", f'{{\n  "schema": "{publisher.MANIFEST_SCHEMA}",\n', 1
            )
            manifest.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(publisher.PublicationError, "duplicate JSON key"):
                prepare_packet(manifest, receipt)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            body = manifest.read_text(encoding="utf-8").replace(
                '"thirdPartyValidated": false', '"thirdPartyValidated": NaN'
            )
            manifest.write_text(body, encoding="utf-8")
            with self.assertRaisesRegex(publisher.PublicationError, "non-finite"):
                prepare_packet(manifest, receipt)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt = make_packet(Path(temporary))
            document = json.loads(manifest.read_bytes())
            document["unexpected"] = True
            reseal_manifest(manifest, document)
            with self.assertRaisesRegex(publisher.PublicationError, "unknown=.*unexpected"):
                prepare_packet(manifest, receipt)

    def test_manifest_and_trust_anchor_symlinks_fail_before_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            manifest_link = root / publisher.RELEASE_MANIFEST_FILENAME
            key_link = root / "trusted-key-link.pem"
            try:
                manifest_link.symlink_to(manifest)
                key_link.symlink_to(packet_key(manifest))
            except OSError as exc:
                self.skipTest(f"file symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(
                publisher.PublicationError, "symlink|junction|reparse"
            ):
                publisher.prepare_release(
                    manifest_link,
                    receipt_path=receipt,
                    trusted_public_key=packet_key(manifest),
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )
            with self.assertRaisesRegex(
                publisher.PublicationError, "symlink|junction|reparse"
            ):
                publisher.prepare_release(
                    manifest,
                    receipt_path=receipt,
                    trusted_public_key=key_link,
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )

    def test_receipt_parent_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _receipt = make_packet(root)
            real_parent = root / "real-output"
            real_parent.mkdir()
            linked_parent = root / "linked-output"
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(
                publisher.PublicationError, "symlink|junction|reparse"
            ):
                publisher.prepare_release(
                    manifest,
                    receipt_path=linked_parent / "publication-receipt.json",
                    trusted_public_key=packet_key(manifest),
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )

    def test_oversized_manifest_and_inventory_claim_fail_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            with manifest.open("r+b") as handle:
                handle.truncate(publisher.MAX_MANIFEST_BYTES + 1)
            with self.assertRaisesRegex(publisher.PublicationError, "oversized"):
                prepare_packet(manifest, receipt)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            document = json.loads(manifest.read_bytes())
            document["intendedRepositoryFiles"][0]["bytes"] = (
                publisher.MAX_INVENTORY_FILE_BYTES + 1
            )
            reseal_manifest(manifest, document)
            with self.assertRaisesRegex(publisher.PublicationError, "publisher limit"):
                prepare_packet(manifest, receipt)

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_windows_junction_ancestors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            fixture.mkdir()
            manifest, _receipt = make_packet(fixture)
            key_target = root / "key-target"
            key_target.mkdir()
            trusted_key = packet_key(manifest)
            relocated_key = key_target / trusted_key.name
            trusted_key.replace(relocated_key)
            key_junction = root / "key-junction"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(key_junction), str(key_target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                self.skipTest(f"junction creation is unavailable: {completed.stderr}")
            with self.assertRaisesRegex(
                publisher.PublicationError, "junction|reparse"
            ):
                publisher.prepare_release(
                    manifest,
                    receipt_path=fixture / "publication-receipt.json",
                    trusted_public_key=key_junction / relocated_key.name,
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )

            output_target = root / "output-target"
            output_target.mkdir()
            output_junction = root / "output-junction"
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(output_junction), str(output_target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                self.skipTest(f"junction creation is unavailable: {completed.stderr}")
            with self.assertRaisesRegex(
                publisher.PublicationError, "junction|reparse"
            ):
                publisher.prepare_release(
                    manifest,
                    receipt_path=output_junction / "publication-receipt.json",
                    trusted_public_key=relocated_key,
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )

    def test_wrong_trust_anchor_blocks_real_crypto_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            wrong = root / "wrong-public-key.pem"
            wrong.write_bytes(
                Ed25519PrivateKey.generate().public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            with self.assertRaisesRegex(publisher.PublicationError, "trusted public key"):
                publisher.run(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=False,
                    trusted_public_key=wrong,
                    expected_source_revision=SOURCE,
                    candidate_loader=candidate_loader,
                )

    def test_stale_expected_source_blocks_before_any_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)
            with self.assertRaisesRegex(publisher.PublicationError, "source differs"):
                publisher.run(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    trusted_public_key=packet_key(manifest),
                    expected_source_revision="f" * 40,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                    candidate_loader=candidate_loader,
                )
            self.assertEqual(api.commits, 0)

    def test_receipt_destination_is_create_only_and_outside_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, receipt = make_packet(root)
            api = FakeApi(root)
            inside = manifest.parent / "publication-receipt.json"
            with self.assertRaisesRegex(publisher.PublicationError, "outside"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=inside,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.commits, 0)

            receipt.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(publisher.PublicationError, "must not already exist"):
                run_packet(
                    manifest_path=manifest,
                    receipt_path=receipt,
                    publish=True,
                    expected_parent_revision="a" * 40,
                    token="scoped-token",
                    api=api,
                    downloader=api.downloader,
                )
            self.assertEqual(api.commits, 0)


if __name__ == "__main__":
    unittest.main()
