from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import publish_model_source_bindings as bindings
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


class FakeApi:
    def model_info(
        self,
        repo_id: str,
        files_metadata: bool = False,
        token: str | None = None,
        revision: str | None = None,
    ):
        del repo_id, files_metadata, token
        if revision is not None:
            return SimpleNamespace(
                sha=revision,
                card_data=SimpleNamespace(license="apache-2.0"),
                tags=["license:apache-2.0"],
            )
        return SimpleNamespace(
            sha="a" * 40,
            siblings=[
                SimpleNamespace(
                    rfilename="model.safetensors",
                    size=7,
                    blob_id="blob-model",
                    lfs=SimpleNamespace(sha256="b" * 64),
                ),
                SimpleNamespace(
                    rfilename="training_receipt.signed.json",
                    size=5,
                    blob_id="blob-receipt",
                    lfs=None,
                ),
                SimpleNamespace(
                    rfilename="eval_receipt.signed.json",
                    size=5,
                    blob_id="blob-eval-receipt",
                    lfs=None,
                ),
                SimpleNamespace(
                    rfilename="owner_pubkey.json",
                    size=5,
                    blob_id="blob-public-key",
                    lfs=None,
                ),
            ],
        )


def _receipt_downloader(root: Path):
    private_key = Ed25519PrivateKey.generate()
    public_der = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    key_b64 = base64.b64encode(public_der).decode("ascii")
    key_id = hashlib.sha256(public_der).hexdigest()[:16]

    def signed(payload: dict) -> dict:
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return {
            "payload": payload,
            "canonical": canonical,
            "signatureBase64": base64.b64encode(
                private_key.sign(canonical.encode("utf-8"))
            ).decode("ascii"),
            "publicKeySpkiBase64": key_b64,
            "keyId": key_id,
        }

    training = signed({"kind": "training", "baseModel": "Qwen/example"})
    training_sha = hashlib.sha256(training["canonical"].encode("utf-8")).hexdigest()
    evaluation = signed(
        {
            "kind": "evaluation",
            "trainingReceiptSha256": training_sha,
            "abstainTotal": 2,
            "abstainCorrect": 2,
        }
    )
    files = {
        "owner_pubkey.json": {
            "algo": "ed25519",
            "publicKeySpkiBase64": key_b64,
            "keyId": key_id,
        },
        "training_receipt.signed.json": training,
        "eval_receipt.signed.json": evaluation,
    }
    for name, payload in files.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")

    def downloader(**kwargs):
        return str(root / kwargs["filename"])

    return downloader


class PublishModelSourceBindingsTests(unittest.TestCase):
    def test_runtime_probe_matches_packaged_model_lab_release(self) -> None:
        repository_root = Path(__file__).parents[1]
        contract = json.loads(
            (repository_root / "publishing/model-source-bindings.json").read_text(
                encoding="utf-8"
            )
        )
        release = json.loads(
            (
                repository_root
                / "spaces/szl-model-inference-lab/release.json"
            ).read_text(encoding="utf-8")
        )
        runtime_artifacts = [
            artifact
            for artifact in contract["artifacts"]
            if artifact.get("runtime_probe") is not None
        ]
        self.assertEqual(len(runtime_artifacts), 1)
        self.assertEqual(
            runtime_artifacts[0]["runtime_probe"]["release_id"],
            release["release_id"],
        )

    def test_dry_run_verifies_source_and_weight_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text("source\n", encoding="utf-8")
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "schema": "szl.model-source-bindings/v2",
                        "source_repository": "szl-holdings/szl-forge",
                        "policy": {
                            "artifact_equivalence": "NOT_CLAIMED",
                            "reproducible_build": "NOT_CLAIMED",
                            "statement": "bounded source claim",
                        },
                        "artifacts": [
                            {
                                "repo_id": "SZLHOLDINGS/example",
                                "artifact_class": "fine_tuned_model",
                                "promotion_state": "NOT_PROMOTED_LIMITED_EVIDENCE",
                                "source_path": ".",
                                "maturity": "MEASURED_LIMITED",
                                "role": "test",
                                "required_hub_files": [
                                    "model.safetensors",
                                    "training_receipt.signed.json",
                                    "eval_receipt.signed.json",
                                    "owner_pubkey.json",
                                ],
                                "expected_weight_sha256": {
                                    "model.safetensors": "b" * 64
                                },
                                "lineage": [
                                    {
                                        "relation": "finetune",
                                        "repo_id": "Qwen/example",
                                        "revision": "d" * 40,
                                        "license": "apache-2.0",
                                    }
                                ],
                                "signed_receipts": {
                                    "public_key": "owner_pubkey.json",
                                    "training": "training_receipt.signed.json",
                                    "evaluation": "eval_receipt.signed.json",
                                    "claim_scope": "REPOSITORY_DECLARED_KEY_CONTINUITY_ONLY",
                                },
                                "source_files": ["source.txt"],
                                "limitations": ["test only"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = root / "report.json"
            with mock.patch.object(bindings, "ROOT", root):
                payload = bindings.run(
                    contract_path=contract,
                    report_path=report,
                    source_revision="c" * 40,
                    publish=False,
                    token=None,
                    api=FakeApi(),
                    downloader=_receipt_downloader(root),
                )
            self.assertEqual(payload["status"], "VERIFIED_DRY_RUN")
            self.assertEqual(
                payload["results"][0]["status"],
                "VERIFIED_DRY_RUN",
            )
            self.assertTrue(report.is_file())

    def test_weight_drift_fails_closed(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/example",
            "artifact_class": "fine_tuned_model",
            "promotion_state": "NOT_PROMOTED_LIMITED_EVIDENCE",
            "required_hub_files": ["model.safetensors"],
            "expected_weight_sha256": {"model.safetensors": "f" * 64},
        }
        with self.assertRaisesRegex(bindings.BindingError, "drifted"):
            bindings.hub_evidence(FakeApi(), artifact)

    def test_publish_requires_token(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/example",
            "source_path": ".",
            "maturity": "MEASURED_LIMITED",
            "role": "test",
            "required_hub_files": [
                "model.safetensors",
                "training_receipt.signed.json",
            ],
            "expected_weight_sha256": {"model.safetensors": "b" * 64},
            "lineage": [
                {
                    "relation": "finetune",
                    "repo_id": "Qwen/example",
                    "revision": "d" * 40,
                    "license": "apache-2.0",
                }
            ],
            "signed_receipts": {
                "public_key": "owner_pubkey.json",
                "training": "training_receipt.signed.json",
                "evaluation": "eval_receipt.signed.json",
                "claim_scope": "REPOSITORY_DECLARED_KEY_CONTINUITY_ONLY",
            },
            "source_files": ["README.md"],
            "limitations": ["test"],
        }
        contract = {
            "source_repository": "szl-holdings/szl-forge",
            "policy": {
                "artifact_equivalence": "NOT_CLAIMED",
                "reproducible_build": "NOT_CLAIMED",
                "statement": "bounded source claim",
            },
        }
        with self.assertRaisesRegex(bindings.BindingError, "HF_TOKEN"):
            bindings.publish_one(
                FakeApi(),
                contract,
                artifact,
                source_revision="c" * 40,
                publish=True,
                token=None,
            )

    def test_runtime_probe_requires_exact_identity_and_reproducible_output(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/example",
            "runtime_probe": {
                "base_url": "https://example.invalid",
                "model_revision": "1" * 40,
                "model_file": "model.gguf",
                "model_sha256": "2" * 64,
                "release_id": "release-1",
                "prompt": "abstain",
                "max_new_tokens": 8,
                "repeat_count": 2,
                "required_output_substring": "abstain",
            },
        }

        def request(url: str, payload=None):
            if url.endswith("/health"):
                return {
                    "status": "READY",
                    "model_sha256_verified": True,
                    "source_integrity": True,
                    "receipt_status": "DECLARED_KEY_SIGNATURES_VALID",
                    "failure_code": None,
                }
            if url.endswith("/api/build-info"):
                return {
                    "build": {"revision": "3" * 40},
                    "runtime": {"state": "READY", "model_sha256_verified": True},
                }
            if url.endswith("/api/v1/identity"):
                return {
                    "space": {"release_id": "release-1", "source_integrity": True},
                    "hardware": {"memory_observed": "16G"},
                    "model": {
                        "repo": "SZLHOLDINGS/example",
                        "revision": "1" * 40,
                        "file": "model.gguf",
                        "sha256_loaded": "2" * 64,
                        "sha256_expected": "2" * 64,
                    },
                }
            self.assertEqual(payload["prompt"], "abstain")
            return {
                "output": "I abstain.",
                "elapsed_ms": 7,
                "model": {"revision": "1" * 40, "sha256": "2" * 64},
            }

        evidence = bindings.runtime_evidence(artifact, request)
        self.assertEqual(evidence["status"], "VERIFIED_OPERATIONAL_LIMITED")
        self.assertEqual(evidence["inference"]["runs"], 2)
        self.assertTrue(evidence["inference"]["deterministic_outputs_equal"])
        self.assertEqual(evidence["energy"]["state"], "UNAVAILABLE")

        matching = bindings.runtime_evidence(
            artifact,
            request,
            expected_source_revision="3" * 40,
        )
        self.assertEqual(matching["runtime_source_revision"], "3" * 40)
        with self.assertRaisesRegex(bindings.BindingError, "does not match expected"):
            bindings.runtime_evidence(
                artifact,
                request,
                expected_source_revision="4" * 40,
            )

    def test_transient_runtime_probe_is_explicitly_not_qualified_in_dry_run(
        self,
    ) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/example",
            "runtime_probe": {"base_url": "https://example.invalid"},
        }
        requested: list[str] = []

        def unavailable(url: str, payload=None):
            del payload
            requested.append(url)
            raise bindings.TransientBindingError("runtime probe returned 503")

        evidence = bindings.runtime_evidence(artifact, unavailable)

        self.assertEqual(
            evidence,
            {
                "status": "NOT_QUALIFIED_NO_RUNTIME_PROBE",
                "failure_reason": "runtime probe returned 503",
                "failure_code": "RUNTIME_SERVICE_UNAVAILABLE",
            },
        )
        self.assertEqual(requested, ["https://example.invalid/health"])

    def test_expected_source_rejects_transient_runtime_probe(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/example",
            "runtime_probe": {"base_url": "https://example.invalid"},
        }

        def unavailable(_url: str, payload=None):
            del payload
            raise bindings.TransientBindingError("runtime probe returned 503")

        with self.assertRaisesRegex(
            bindings.BindingError,
            "exact runtime evidence is required but unavailable",
        ):
            bindings.runtime_evidence(
                artifact,
                unavailable,
                expected_source_revision="3" * 40,
            )

    def test_publish_paths_require_runtime_evidence(self) -> None:
        artifact = {"repo_id": "SZLHOLDINGS/example"}
        prepared = (
            {"repo_id": artifact["repo_id"], "hub_revision_before": "a" * 40},
            b"{}\n",
        )

        with (
            mock.patch.object(
                bindings, "prepare_one", return_value=prepared
            ) as prepare,
            mock.patch.object(
                bindings,
                "publish_prepared",
                return_value={"status": "PUBLISHED_AND_READBACK_VERIFIED"},
            ),
        ):
            bindings.publish_one(
                FakeApi(),
                {"source_repository": "szl-holdings/szl-forge"},
                artifact,
                source_revision="3" * 40,
                publish=True,
                token="test-token",
            )

        self.assertTrue(prepare.call_args.kwargs["require_runtime_evidence"])

    def test_publish_rejects_transient_runtime_probe_before_upload(self) -> None:
        artifact = {
            "repo_id": "SZLHOLDINGS/example",
            "artifact_class": "fine_tuned_model",
            "promotion_state": "NOT_PROMOTED_LIMITED_EVIDENCE",
            "source_path": ".",
            "maturity": "MEASURED_LIMITED",
            "role": "test",
            "limitations": ["test only"],
            "runtime_probe": {"base_url": "https://example.invalid"},
        }
        contract = {
            "source_repository": "szl-holdings/szl-forge",
            "policy": {
                "artifact_equivalence": "NOT_CLAIMED",
                "reproducible_build": "NOT_CLAIMED",
                "statement": "bounded source claim",
            },
        }
        api = mock.Mock()

        def unavailable(_url: str, payload=None):
            del payload
            raise bindings.TransientBindingError("runtime probe returned 503")

        receipts = {
            "status": "DECLARED_KEY_SIGNATURES_VALID",
            "held_out_evaluation": {},
        }
        with (
            mock.patch.object(bindings, "source_evidence", return_value=[]),
            mock.patch.object(
                bindings,
                "hub_evidence",
                return_value=("a" * 40, []),
            ),
            mock.patch.object(bindings, "lineage_evidence", return_value=[]),
            mock.patch.object(
                bindings,
                "signed_receipt_evidence",
                return_value=receipts,
            ),
            mock.patch.object(bindings, "publish_prepared") as publish_prepared,
        ):
            with self.assertRaisesRegex(
                bindings.BindingError,
                "exact runtime evidence is required but unavailable",
            ):
                bindings.publish_one(
                    api,
                    contract,
                    artifact,
                    source_revision="3" * 40,
                    publish=True,
                    token="test-token",
                    requester=unavailable,
                )

        publish_prepared.assert_not_called()
        api.upload_file.assert_not_called()

    def test_run_wires_publish_mode_to_runtime_evidence_requirement(self) -> None:
        artifact = {"repo_id": "SZLHOLDINGS/example"}
        contract = {
            "source_repository": "szl-holdings/szl-forge",
            "artifacts": [artifact],
        }
        prepared = (
            {"repo_id": artifact["repo_id"], "hub_revision_before": "a" * 40},
            b"{}\n",
        )

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            with (
                mock.patch.object(bindings, "load_contract", return_value=contract),
                mock.patch.object(
                    bindings, "prepare_one", return_value=prepared
                ) as prepare,
                mock.patch.object(
                    bindings,
                    "publish_prepared",
                    return_value={"status": "PUBLISHED_AND_READBACK_VERIFIED"},
                ),
            ):
                bindings.run(
                    contract_path=Path("contract.json"),
                    report_path=report,
                    source_revision="3" * 40,
                    publish=True,
                    token="test-token",
                    api=FakeApi(),
                )

        self.assertTrue(prepare.call_args.kwargs["require_runtime_evidence"])

    def test_publish_preflights_every_artifact_before_the_first_upload(self) -> None:
        contract = {
            "source_repository": "szl-holdings/szl-forge",
            "artifacts": [
                {"repo_id": "SZLHOLDINGS/first"},
                {"repo_id": "SZLHOLDINGS/runtime", "runtime_probe": {}},
            ],
        }

        def prepare(_api, _contract, artifact, **_kwargs):
            if artifact.get("runtime_probe") is not None:
                raise bindings.BindingError("live runtime source revision mismatch")
            return ({"repo_id": artifact["repo_id"]}, b"{}\n")

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            with (
                mock.patch.object(bindings, "load_contract", return_value=contract),
                mock.patch.object(bindings, "prepare_one", side_effect=prepare),
                mock.patch.object(bindings, "publish_prepared") as publish_prepared,
            ):
                with self.assertRaisesRegex(bindings.BindingError, "revision mismatch"):
                    bindings.run(
                        contract_path=Path("contract.json"),
                        report_path=report,
                        source_revision="3" * 40,
                        expected_runtime_source_revision="3" * 40,
                        publish=True,
                        token="test-token",
                        api=FakeApi(),
                    )
                publish_prepared.assert_not_called()
                self.assertFalse(report.exists())

    def test_upload_is_bound_to_the_verified_hub_parent(self) -> None:
        artifact = {"repo_id": "SZLHOLDINGS/example"}
        source_revision = "3" * 40
        hub_revision = "a" * 40
        body = b'{"schema":"test"}\n'

        with tempfile.TemporaryDirectory() as temporary:
            readback = Path(temporary) / "publication.json"
            readback.write_bytes(body)
            api = mock.Mock()
            api.upload_file.return_value = SimpleNamespace(oid="d" * 40)
            result = {
                "status": "VERIFIED_DRY_RUN",
                "hub_revision_before": hub_revision,
            }
            with mock.patch.object(
                bindings,
                "hf_hub_download",
                return_value=str(readback),
            ):
                published = bindings.publish_prepared(
                    api,
                    artifact,
                    result,
                    body,
                    source_revision=source_revision,
                    token="test-token",
                )

            upload = api.upload_file.call_args.kwargs
            self.assertEqual(upload["revision"], "main")
            self.assertEqual(upload["parent_commit"], hub_revision)
            self.assertEqual(published["status"], "PUBLISHED_AND_READBACK_VERIFIED")

            conflicting_api = mock.Mock()
            conflicting_api.upload_file.side_effect = RuntimeError("parent conflict")
            with self.assertRaisesRegex(RuntimeError, "parent conflict"):
                bindings.publish_prepared(
                    conflicting_api,
                    artifact,
                    {
                        "status": "VERIFIED_DRY_RUN",
                        "hub_revision_before": hub_revision,
                    },
                    body,
                    source_revision=source_revision,
                    token="test-token",
                )


if __name__ == "__main__":
    unittest.main()
