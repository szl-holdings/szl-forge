"""Opt-in synthetic v2 contracts; the established v1 route remains the default."""

from __future__ import annotations

import http.client
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
SPEC = importlib.util.spec_from_file_location("oac_v2_test_app", ROOT / "app.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "oac_v2_test_release", ROOT / "verify_release.py"
)
release = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(release)
sys.path.insert(0, str(ROOT))
import verify_live as live  # noqa: E402
SOURCE = "a" * 40
DEGRADED = {
    "listener_running": 0,
    "tls_enabled": 0,
    "peer_allowlist_configured": 0,
    "queue_utilization": 0.95,
    "consecutive_failures": 15,
    "seconds_since_last_success": 7200,
    "ledger_integrity_ok": 0,
    "configuration_valid": 0,
}
AMBIGUOUS = {
    "listener_running": 1,
    "tls_enabled": 1,
    "peer_allowlist_configured": 1,
    "queue_utilization": 0.1,
    "consecutive_failures": 5,
    "seconds_since_last_success": 1,
    "ledger_integrity_ok": 1,
    "configuration_valid": 1,
}


class V2Contracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="szl-oac-v2-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(ROOT / "artifacts", self.root / "artifacts")
        for name in ("release.json", "release_v2.json"):
            shutil.copyfile(ROOT / name, self.root / name)
        self.example = (ROOT / "artifacts" / "v2" / "example_input.json").read_bytes()

    def application(self, source=SOURCE):
        return app.Application(self.root, source)

    def request(self, server, path, method="GET", body=None):
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        self.addCleanup(connection.close)
        headers = {"Content-Type": "application/json", "X-SZL-Preview": "1"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def server(self, application):
        server = app.create_server(("127.0.0.1", 0), application)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(1)

        self.addCleanup(close)
        return server

    def test_pinned_v2_identity_and_canonical_advisory(self):
        application = self.application()
        self.assertTrue(application.ready)
        self.assertTrue(application.v2_ready)
        self.assertEqual(application.identity()["schema"], "szl.oac-health-space-identity/v1")
        identity = application.v2_identity()
        self.assertEqual(identity["hub_model"]["revision"], app.V2_MODEL_REVISION)
        self.assertEqual(identity["artifacts"], app.V2_ARTIFACTS)
        self.assertEqual(identity["authority"], app.V2_AUTHORITY)
        self.assertFalse(identity["clinical_use_authorized"])
        self.assertFalse(identity["production_promotion_allowed"])
        for label, body, expected in (
            ("healthy", self.example, ("NO_ALERT", None, [0])),
            ("degraded", app.canonical({"features": DEGRADED}), ("ALERT", None, [1])),
            ("ambiguous", app.canonical({"features": AMBIGUOUS}), ("ABSTAIN", "BOTH", [0, 1])),
        ):
            with self.subTest(label=label):
                response = application.score_v2(body)
                advisory = response["advisory"]
                self.assertEqual(
                    (advisory["advisory"], advisory["abstain_reason"], advisory["prediction_set"]),
                    expected,
                )
                self.assertEqual(advisory["authority"], app.V2_AUTHORITY)
                self.assertFalse(response["receipt_minted"])
                canonical = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(ROOT / "artifacts" / "v2" / "ops_health.py"),
                        "--model",
                        str(ROOT / "artifacts" / "v2" / "model.json"),
                        "--receipt",
                        str(ROOT / "artifacts" / "v2" / "artifact_receipt.json"),
                        "--input",
                        "-",
                    ],
                    input=body,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(canonical.returncode, 0, canonical.stderr)
                self.assertEqual(advisory, json.loads(canonical.stdout))

    def test_v2_loader_does_not_reopen_verified_model_or_receipt(self):
        real_open = Path.open
        observed = {name: 0 for name in app.V2_ARTIFACTS}

        def count_artifact_reads(path, *args, **kwargs):
            if path.parent == self.root / "artifacts" / "v2":
                observed[path.name] += 1
                if observed[path.name] > 1:
                    raise AssertionError("second v2 artifact read: " + path.name)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", count_artifact_reads):
            application = self.application()
            response = application.score_v2(self.example)
        self.assertTrue(application.ready)
        self.assertTrue(application.v2_ready)
        self.assertEqual(response["advisory"]["advisory"], "NO_ALERT")
        self.assertEqual(observed, dict.fromkeys(app.V2_ARTIFACTS, 1))

    def test_v2_http_is_distinct_and_rejects_extra_fields_without_echo(self):
        server = self.server(self.application())
        status, v1 = self.request(server, "/api/v1/identity")
        self.assertEqual((status, v1["schema"]), (200, "szl.oac-health-space-identity/v1"))
        status, ready = self.request(server, "/api/v2/readyz")
        self.assertEqual((status, ready["state"]), (200, "READY"))
        status, identity = self.request(server, "/api/v2/identity")
        self.assertEqual((status, identity["hub_model"]["revision"]), (200, app.V2_MODEL_REVISION))
        status, score = self.request(server, "/api/v2/score", "POST", self.example)
        self.assertEqual((status, score["advisory"]["advisory"]), (200, "NO_ALERT"))
        features = json.loads(self.example)["features"]
        features["patient_id"] = "SYNTHETIC_REJECTION_MARKER"
        status, refused = self.request(
            server, "/api/v2/score", "POST", app.canonical({"features": features})
        )
        self.assertEqual((status, refused), (400, {"ok": False, "error": "INVALID_SCORE_INPUT"}))
        self.assertNotIn("SYNTHETIC_REJECTION_MARKER", json.dumps(refused))
        status, refused = self.request(server, "/api/v2/score", "POST", b'{"features":{"foo":"bar"}}')
        self.assertEqual(status, 400)
        self.assertEqual(refused["error"], "INVALID_SCORE_INPUT")
        status, v1_score = self.request(server, "/api/score", "POST", self.example)
        self.assertEqual((status, v1_score["identity"]["schema"]), (200, "szl.oac-health-space-identity/v1"))

    def test_v2_artifact_or_manifest_drift_cannot_break_v1(self):
        path = self.root / "artifacts" / "v2" / "model.json"
        path.write_bytes(path.read_bytes() + b"\n")
        application = self.application()
        self.assertTrue(application.ready)
        self.assertFalse(application.v2_ready)
        self.assertIsNone(application.v2_identity()["artifacts"])
        server = self.server(application)
        status, refused = self.request(server, "/api/v2/score", "POST", self.example)
        self.assertEqual((status, refused["error"]), (503, "SERVICE_UNAVAILABLE"))
        status, v1 = self.request(server, "/api/v1/identity")
        self.assertEqual((status, v1["state"]), (200, "READY"))
        shutil.copyfile(ROOT / "artifacts" / "v2" / "model.json", path)
        manifest = json.loads((self.root / "release_v2.json").read_text(encoding="utf-8"))
        manifest["hub_model"]["revision"] = "b" * 40
        (self.root / "release_v2.json").write_bytes(app.canonical(manifest))
        self.assertFalse(self.application().v2_ready)
        self.assertTrue(self.application().ready)

    def test_v2_manifest_requires_exact_boolean_types(self):
        original = json.loads((ROOT / "release_v2.json").read_text(encoding="utf-8"))
        for key, numeric in (
            ("synthetic_training_data", 1),
            ("clinical_use_authorized", 0),
            ("production_promotion_allowed", 0),
        ):
            with self.subTest(key=key):
                altered = dict(original, **{key: numeric})
                (self.root / "release_v2.json").write_bytes(app.canonical(altered))
                application = self.application()
                self.assertTrue(application.ready)
                self.assertFalse(application.v2_ready)
                with self.assertRaises(release.ReleaseVerificationError):
                    release.verify_v2_release(self.root, REPOSITORY)

    def test_source_binding_and_immutable_git_artifacts(self):
        application = self.application("")
        self.assertFalse(application.ready)
        self.assertFalse(application.v2_ready)
        self.assertTrue(application.v2_artifacts_verified)
        self.assertEqual(application.v2_failure_code, "SOURCE_BINDING_UNAVAILABLE")
        evidence = release.verify_v2_release(self.root, REPOSITORY)
        self.assertEqual(evidence["source_parity_count"], 4)
        self.assertFalse(evidence["provider_parity_claimed"])
        manifest = json.loads((self.root / "release_v2.json").read_text(encoding="utf-8"))
        manifest["artifacts"]["model.json"] = "0" * 64
        (self.root / "release_v2.json").write_bytes(app.canonical(manifest))
        with self.assertRaises(release.ReleaseVerificationError):
            release.verify_v2_release(self.root, REPOSITORY)

    def test_v1_and_v2_source_bound_loopback_witness(self):
        server = self.server(self.application())
        origin = "http://127.0.0.1:" + str(server.server_address[1])
        report = live.verify_live(
            origin=origin,
            expected_source_sha=SOURCE,
            space_root=self.root,
            repository_root=REPOSITORY,
            allow_localhost=True,
            verify_v2=True,
        )
        self.assertEqual(report["status"], "PASS", report.get("failure"))
        self.assertEqual(report["observation_scope"], "LOCAL_TEST_ONLY")
        self.assertEqual(report["v2_release_verification"]["source_parity_count"], 4)
        names = {item["name"] for item in report["checks"]}
        self.assertIn("/api/v2/identity", names)
        self.assertIn("v2_authored_healthy_synthetic", names)
        self.assertIn("v2_authored_ambiguous_synthetic", names)
        self.assertIn("v2_clinical_like_extra_field_refused_without_echo", names)


if __name__ == "__main__":
    unittest.main()
