from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app
import app_governed as governed
import formula_atlas_contract as formula
import verify_governed_live as witness


class FormulaAtlasContractTests(unittest.TestCase):
    def test_live_verifier_pins_match_asset_validator(self) -> None:
        self.assertEqual(witness.ATLAS_REPOSITORY, formula.ATLAS_REPOSITORY)
        self.assertEqual(witness.ATLAS_REVISION, formula.ATLAS_REVISION)
        self.assertEqual(witness.ATLAS_GIT_BLOB_SHA, formula.ATLAS_GIT_BLOB_SHA)
        self.assertEqual(witness.ATLAS_SHA256, formula.ATLAS_SHA256)
        self.assertEqual(
            witness.ATLAS_PAYLOAD_SHA256, formula.ATLAS_PAYLOAD_SHA256
        )
        self.assertEqual(witness.ATTRIBUTED_SOURCE, formula.ATTRIBUTED_SOURCE)

    def test_materialized_files_match_exact_immutable_sources(self) -> None:
        atlas_raw = formula._canonical_text_bytes(formula.ATLAS_PATH)
        source_raw = formula._canonical_text_bytes(formula.ATTRIBUTED_SOURCE_PATH)

        self.assertEqual(hashlib.sha256(atlas_raw).hexdigest(), formula.ATLAS_SHA256)
        self.assertEqual(formula._git_blob_sha(atlas_raw), formula.ATLAS_GIT_BLOB_SHA)
        self.assertEqual(
            hashlib.sha256(source_raw).hexdigest(),
            formula.ATTRIBUTED_SOURCE["sha256"],
        )
        self.assertEqual(
            formula._git_blob_sha(source_raw),
            formula.ATTRIBUTED_SOURCE["git_blob_sha"],
        )

    def test_counts_domains_and_authority_are_fail_closed(self) -> None:
        atlas = formula.validate_formula_assets()
        self.assertEqual(atlas["summary"]["attributed_formula_count"], 30)
        self.assertEqual(atlas["summary"]["executable_formula_count"], 21)
        self.assertEqual(len(atlas["quant_domains"]), 9)
        self.assertEqual(atlas["authority"]["locked_proven_count"], 8)
        self.assertEqual(
            set(atlas["authority"]["locked_proven_ids"]),
            formula.LOCKED_PROVEN_IDS,
        )
        self.assertEqual(
            atlas["authority"]["lambda_status"],
            "CONJECTURE_1_OPEN_ADVISORY_ONLY",
        )
        self.assertEqual(
            atlas["authority"]["f_number_to_executable_registry_mapping"],
            "UNKNOWN_NOT_INFERRED",
        )
        self.assertEqual(atlas["payload_sha256"], formula.ATLAS_PAYLOAD_SHA256)

    def test_public_projection_exposes_handles_without_private_content(self) -> None:
        payload = formula.public_formula_atlas()
        serialized = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["schema"], "szl.formula-atlas.public/v2")
        self.assertEqual(len(payload["formula_handles"]), 30)
        self.assertEqual(len(payload["executable_formula_handles"]), 21)
        self.assertNotIn('"statement"', serialized)
        self.assertNotIn(
            "Weighted geometric mean is positively homogeneous", serialized
        )
        self.assertFalse(payload["privacy"]["formula_statements_included"])
        self.assertFalse(payload["privacy"]["second_brain_handles_included"])
        self.assertFalse(payload["privacy"]["private_graph_content_included"])
        self.assertFalse(payload["privacy"]["private_retrieval_performed"])
        self.assertFalse(payload["authority"]["formula_may_authorize"])
        self.assertFalse(payload["authority"]["model_may_authorize"])
        self.assertFalse(payload["authority"]["public_effectors_enabled"])
        self.assertTrue(payload["authority"]["human_binding_required"])
        self.assertEqual(payload["authority"]["public_tools"], [])
        self.assertIsNone(formula.SENSITIVE_VALUE.search(serialized))

    def test_corrupt_atlas_and_source_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = root / "atlas.json"
            source = root / "source.json"
            shutil.copyfile(formula.ATLAS_PATH, atlas)
            shutil.copyfile(formula.ATTRIBUTED_SOURCE_PATH, source)

            atlas.write_bytes(atlas.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                formula.FormulaAtlasIntegrityError, "formula atlas SHA-256 drift"
            ):
                formula.validate_formula_assets(atlas, source)

            shutil.copyfile(formula.ATLAS_PATH, atlas)
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                formula.FormulaAtlasIntegrityError,
                "attributed formula source SHA-256 drift",
            ):
                formula.validate_formula_assets(atlas, source)

    def test_source_revision_drift_is_rejected_before_serving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas = root / "atlas.json"
            source = root / "source.json"
            value = json.loads(formula.ATLAS_PATH.read_text(encoding="utf-8"))
            value["source"]["revision"] = "0" * 40
            atlas.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            shutil.copyfile(formula.ATTRIBUTED_SOURCE_PATH, source)
            with self.assertRaisesRegex(
                formula.FormulaAtlasIntegrityError, "formula atlas SHA-256 drift"
            ):
                formula.validate_formula_assets(atlas, source)

    def test_formula_and_source_routes_return_verified_contracts(self) -> None:
        response = governed.formula_atlas()
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["counts"]["attributed_formula_count"], 30)
        self.assertEqual(payload["source"]["atlas"]["revision"], formula.ATLAS_REVISION)

        with mock.patch.dict(
            "os.environ", {app.SOURCE_REVISION_ENV: "f" * 40}, clear=False
        ):
            source_response = governed.source_contract()
        source_payload = json.loads(source_response.body)
        self.assertEqual(source_response.status_code, 200)
        self.assertEqual(source_payload["state"], "OBSERVED")
        self.assertEqual(
            source_payload["service"]["repository"], "szl-holdings/szl-forge"
        )
        self.assertEqual(source_payload["service"]["revision"], "f" * 40)
        self.assertEqual(
            source_payload["components"]["formula_atlas"]["git_blob_sha"],
            formula.ATLAS_GIT_BLOB_SHA,
        )
        witness.verify_source_contract(source_payload, "f" * 40)
        evidence = witness.verify_formula_atlas(payload)
        self.assertEqual(evidence["atlas_revision"], formula.ATLAS_REVISION)
        route_paths = {route.path for route in governed.app.routes}
        self.assertIn("/api/v2/formula-atlas", route_paths)
        self.assertIn("/api/source", route_paths)
        self.assertIn("/.well-known/szl-source.json", route_paths)

    def test_live_verifier_rejects_count_and_authority_drift(self) -> None:
        atlas = formula.public_formula_atlas()
        atlas["counts"]["locked_proven_count"] = 9
        with self.assertRaisesRegex(witness.VerificationError, "locked_proven_count"):
            witness.verify_formula_atlas(atlas)

        atlas = formula.public_formula_atlas()
        atlas["authority"]["public_effectors_enabled"] = True
        with self.assertRaisesRegex(witness.VerificationError, "public effectors"):
            witness.verify_formula_atlas(atlas)

    def test_integrity_failure_blocks_dependency_readiness(self) -> None:
        with mock.patch.object(
            governed,
            "public_formula_atlas",
            side_effect=formula.FormulaAtlasIntegrityError("corrupt bytes"),
        ):
            dependency = governed._dependency_status()
        self.assertFalse(dependency["formula_atlas_ready"])
        self.assertFalse(dependency["ready"])
        self.assertEqual(
            dependency["formula_atlas_error"], "FormulaAtlasIntegrityError"
        )

    def test_integrity_failure_is_a_sanitized_503(self) -> None:
        with mock.patch.object(
            governed,
            "public_formula_atlas",
            side_effect=formula.FormulaAtlasIntegrityError("sensitive path"),
        ):
            response = governed.formula_atlas()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            json.loads(response.body),
            {
                "schema": "szl.formula-atlas.public/v2",
                "state": "UNAVAILABLE",
                "error": "FORMULA_ATLAS_INTEGRITY_FAILURE",
            },
        )
        self.assertNotIn(b"sensitive path", response.body)

        with mock.patch.object(
            app,
            "load_release_manifest",
            side_effect=RuntimeError("sensitive manifest path"),
        ):
            source_response = governed.source_contract()
        self.assertEqual(source_response.status_code, 503)
        self.assertEqual(
            json.loads(source_response.body),
            {
                "schema": "szl.model-inference-lab.source/v1",
                "state": "UNAVAILABLE",
                "error": "SOURCE_CONTRACT_INTEGRITY_FAILURE",
            },
        )
        self.assertNotIn(b"sensitive manifest path", source_response.body)

    def test_docker_and_release_manifest_close_over_formula_assets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        manifest = app.load_release_manifest()

        self.assertIn("FROM runtime-base AS formula-contract-validator", dockerfile)
        self.assertIn(
            "COPY --chown=user:user formula_atlas/ ./formula_atlas/", dockerfile
        )
        self.assertIn("python formula_atlas_contract.py --verify", dockerfile)
        self.assertIn("/opt/szl/formula-contract-verified", dockerfile)
        self.assertNotIn("formula_atlas", dockerignore)
        for path in (
            "formula_atlas_contract.py",
            "formula_atlas/formula-atlas.v1.json",
            "formula_atlas/source-formula-ledger-corpus.json",
            "tests/test_formula_atlas_contract.py",
        ):
            self.assertIn(path, manifest["source_files"])


if __name__ == "__main__":
    unittest.main()
