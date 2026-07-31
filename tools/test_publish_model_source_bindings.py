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


if __name__ == "__main__":
    unittest.main()
