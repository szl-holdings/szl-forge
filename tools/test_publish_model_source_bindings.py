from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import publish_model_source_bindings as bindings


class FakeApi:
    def model_info(self, repo_id: str, files_metadata: bool = False, token: str | None = None):
        del repo_id, files_metadata, token
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
            ],
        )


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
                        "schema": "szl.model-source-bindings/v1",
                        "source_repository": "szl-holdings/szl-forge",
                        "policy": {
                            "artifact_equivalence": "NOT_CLAIMED",
                            "reproducible_build": "NOT_CLAIMED",
                            "statement": "bounded source claim",
                        },
                        "artifacts": [
                            {
                                "repo_id": "SZLHOLDINGS/example",
                                "source_path": ".",
                                "maturity": "MEASURED_LIMITED",
                                "role": "test",
                                "required_hub_files": [
                                    "model.safetensors",
                                    "training_receipt.signed.json",
                                ],
                                "expected_weight_sha256": {
                                    "model.safetensors": "b" * 64
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


if __name__ == "__main__":
    unittest.main()
