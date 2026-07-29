import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

import forge_runtime_contract as contract
from forge_lab import get_status


MEASUREMENT = {
    "hf_revision": "b" * 40,
    "revision_state": "MEASURED_REPOSITORY_HEAD",
    "running_process_revision_state": "NOT_EXPOSED_IN_PROCESS",
    "measurement_method": "HUGGINGFACE_HUB_API",
    "resolver": "https://huggingface.co/api/spaces/SZLHOLDINGS/szl-forge-lab",
    "observed_at": "2026-07-12T00:00:00Z",
    "provider_last_modified": "2026-07-12T00:00:00Z",
}


class RuntimeContractTests(unittest.TestCase):
    def test_source_attestation_separates_head_evidence_from_process_identity(self):
        with patch.object(contract, "measure_hf_repository_head", return_value=MEASUREMENT):
            payload, status_code = contract.build_source_attestation()
        self.assertEqual(200, status_code)
        self.assertEqual("szl.deployment-source/v1", payload["schema"])
        self.assertEqual("b" * 40, payload["deployment"]["hf_revision"])
        self.assertEqual("PENDING_GITHUB_SYNC", payload["alignment_state"])
        evidence = payload["extensions"]["deployment_revision_evidence"]
        self.assertEqual("NOT_EXPOSED_IN_PROCESS", evidence["running_process_revision_state"])
        self.assertIn(
            "Provider-reported lastModified",
            payload["extensions"]["build_metadata"]["built_at_semantics"],
        )
        self.assertEqual("NOT_CLAIMED", payload["extensions"]["claims"]["github_parity"])

    def test_source_attestation_fails_closed_when_hub_measurement_is_unavailable(self):
        unavailable = dict(MEASUREMENT, hf_revision=None, revision_state="UNAVAILABLE")
        with patch.object(contract, "measure_hf_repository_head", return_value=unavailable):
            payload, status_code = contract.build_source_attestation()
        self.assertEqual(503, status_code)
        self.assertEqual("szl.deployment-source-unavailable/v1", payload["schema"])
        self.assertEqual("UNAVAILABLE", payload["evidence_state"])

    def test_health_does_not_conflate_reachability_with_quality_or_parity(self):
        payload = contract.build_health(get_status)
        self.assertEqual("OK", payload["status"])
        self.assertEqual("REACHABLE", payload["transport_state"])
        self.assertEqual("NOT_MEASURED", payload["checks"]["model_quality"])
        self.assertEqual("PENDING_GITHUB_SYNC", payload["checks"]["source_parity"])

    def test_runtime_status_wraps_existing_packaged_contract(self):
        with patch.object(contract, "measure_hf_repository_head", return_value=MEASUREMENT):
            payload = contract.build_runtime_status(get_status)
        self.assertEqual("szl.forge.runtime-status/v1", payload["schema"])
        self.assertEqual("szl.forge.lab-status/v1", payload["application"]["schema"])
        self.assertEqual("READ_ONLY", payload["authority_state"])
        self.assertEqual("NOT_CLAIMED", payload["source"]["github_parity"])

    def test_exact_routes_precede_html_fallback_and_return_json(self):
        app = FastAPI()

        @app.get("/{full_path:path}")
        async def fallback(full_path: str):  # noqa: ARG001
            return HTMLResponse("<html>fallback</html>")

        result = contract.register_contract_routes(app, get_status)
        self.assertTrue(result["ok"])
        paths = [getattr(route, "path", None) for route in app.router.routes]
        for path in result["routes"]:
            self.assertLess(paths.index(path), paths.index("/{full_path:path}"))

        with patch.object(contract, "measure_hf_repository_head", return_value=MEASUREMENT):
            client = TestClient(app)
            for path, schema in (
                ("/.well-known/szl-source.json", "szl.deployment-source/v1"),
                ("/healthz", "szl.forge.health/v1"),
                ("/api/status", "szl.forge.runtime-status/v1"),
            ):
                response = client.get(path)
                self.assertEqual(200, response.status_code)
                self.assertTrue(response.headers["content-type"].startswith("application/json"))
                self.assertEqual("no-store", response.headers["cache-control"])
                self.assertEqual(schema, response.json()["schema"])

    def test_registration_refuses_route_collisions(self):
        app = FastAPI()

        @app.get("/healthz")
        async def existing_health():
            return {"status": "existing"}

        with self.assertRaisesRegex(RuntimeError, "Refusing to shadow"):
            contract.register_contract_routes(app, get_status)

    def test_space_id_is_built_from_documented_environment_variable(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, *_args):
                return (
                    b'{"sha":"cccccccccccccccccccccccccccccccccccccccc",'
                    b'"lastModified":"2026-07-12T00:00:00Z"}'
                )

        with patch.dict(os.environ, {"SPACE_ID": "SZLHOLDINGS/szl-forge-lab"}):
            with patch("urllib.request.urlopen", return_value=Response()) as mocked:
                measurement = contract.measure_hf_repository_head(force=True)
        request = mocked.call_args.args[0]
        self.assertIn("SZLHOLDINGS/szl-forge-lab", request.full_url)
        self.assertEqual("c" * 40, measurement["hf_revision"])


if __name__ == "__main__":
    unittest.main()
