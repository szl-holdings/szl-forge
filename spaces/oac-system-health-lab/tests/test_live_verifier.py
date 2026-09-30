"""Hermetic fake-observation regressions for the source-bound live verifier."""

import copy
import json
import io
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import Mock, patch

SPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPACE_ROOT))
import verify_live as live  # noqa: E402
import verify_release as release  # noqa: E402

APPLICATION_SHA = "1234567890abcdef1234567890abcdef12345678"


class FakeTransport:
    def __init__(self, manifest, kernel, transform=None):
        self.identity = live.expected_identity(manifest, APPLICATION_SHA)
        self.kernel = kernel
        self.transform = transform
        self.calls = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        status = 200
        if path == "/healthz":
            value = {"alive": True, "state": "READY", "receipt_minted": False}
        elif path == "/readyz":
            value = {"state": "READY", "error": None, "receipt_minted": False}
        elif path == "/api/build-info":
            value = {
                "service": "oac-system-health-lab",
                "build": {
                    "state": "OBSERVED",
                    "revision": APPLICATION_SHA,
                    "revision_source": "env:SZL_GITHUB_SOURCE_REVISION",
                },
                "runtime": {"state": "READY"},
                "receipt_minted": False,
            }
        elif path == "/api/v1/identity":
            value = copy.deepcopy(self.identity)
        elif path == "/api/score" and "patient_id" in payload["features"]:
            value = {"ok": False, "error": "INVALID_INPUT"}
            status = 400
        elif path == "/api/score":
            advisory = self.kernel.score(payload["features"])
            value = {
                "ok": True,
                "advisory": advisory,
                "identity": copy.deepcopy(self.identity),
                "receipt_minted": False,
                "input_sha256": live.digest(payload["features"]),
                "output_sha256": live.digest(advisory),
            }
        else:
            raise AssertionError("unexpected request")
        if self.transform:
            changed = self.transform(method, path, payload, status, value)
            if changed is not None:
                return changed
        return live.Observation(
            status, "application/json; charset=utf-8", live.canonical(value)
        )


class LiveVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = release.load_manifest(SPACE_ROOT)
        cls.artifact_bytes = {
            name: (SPACE_ROOT / "artifacts" / name).read_bytes()
            for name in release.ARTIFACT_NAMES
        }

        def immutable_fixture(_root, *arguments):
            prefix = (
                cls.manifest["artifact_source"]["revision"]
                + ":"
                + release.SOURCE_DIRECTORY
                + "/"
            )
            if arguments[0] != "show" or not arguments[-1].startswith(prefix):
                raise AssertionError("not an immutable artifact read")
            name = arguments[-1][len(prefix) :]
            if name not in release.ARTIFACT_NAMES:
                raise AssertionError("unexpected artifact")
            return cls.artifact_bytes[name]

        with patch.object(live, "_git", side_effect=immutable_fixture):
            cls.kernel = live._load_kernel(
                SPACE_ROOT, Path("hermetic-fixture"), cls.manifest
            )

    def verify(self, transform=None, **overrides):
        transport = FakeTransport(self.manifest, self.kernel, transform)
        arguments = {
            "origin": live.DEFAULT_ORIGIN,
            "expected_source_sha": APPLICATION_SHA,
            "space_root": SPACE_ROOT,
            "repository_root": Path("hermetic-fixture"),
            "transport": transport,
        }
        arguments.update(overrides)
        with (
            patch.object(
                live,
                "verify_release",
                return_value={
                    "complete": True,
                    "status": "PASS",
                    "source_parity_count": 3,
                },
            ),
            patch.object(live, "_load_kernel", return_value=self.kernel),
        ):
            report = live.verify_live(**arguments)
        return report, transport

    def test_seven_checks_include_real_scoring_contract_and_rejection(self):
        report, transport = self.verify()
        self.assertTrue(report["complete"])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["synthetic_case_count"], 2)
        self.assertEqual(len(report["checks"]), 7)
        self.assertEqual(len(transport.calls), 7)
        self.assertEqual(
            [method for method, _, _ in transport.calls], ["GET"] * 4 + ["POST"] * 3
        )
        self.assertFalse(report["clinical_use_authorized"])
        self.assertFalse(report["production_promotion_allowed"])
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["observation_scope"], "PUBLIC_SYNTHETIC_PROBE")
        self.assertEqual(set(transport.calls[4][2]), {"features"})
        self.assertEqual(len(transport.calls[4][2]["features"]), 8)
        self.assertEqual(
            transport.calls[-1][2]["features"]["patient_id"], live.REJECTION_MARKER
        )

    def test_expected_cases_are_independent_authored_extremes(self):
        healthy = self.kernel.score(live.CASES[0][1])
        degraded = self.kernel.score(live.CASES[1][1])
        self.assertFalse(healthy["operator_attention_required"])
        self.assertTrue(degraded["operator_attention_required"])
        self.assertLess(
            healthy["operator_attention_score"], degraded["operator_attention_score"]
        )

    def test_stale_build_revision_fails_before_scoring(self):
        def transform(_method, path, _payload, _status, value):
            if path == "/api/build-info":
                value["build"]["revision"] = "a" * 40

        report, transport = self.verify(transform)
        self.assertFalse(report["complete"])
        self.assertEqual(report["failure"], "BUILD_SOURCE_CONTRACT_MISMATCH")
        self.assertEqual(len(transport.calls), 3)

    def test_identity_rejects_each_pin_and_authority_change(self):
        for field in (
            "application",
            "artifact_source",
            "hub_model",
            "hub_dataset",
            "artifacts",
            "synthetic_training_data",
            "clinical_use_authorized",
            "production_promotion_allowed",
        ):
            with self.subTest(field=field):

                def transform(_method, path, _payload, _status, value):
                    if path == "/api/v1/identity":
                        value[field] = {"unexpected": "changed"}

                report, _ = self.verify(transform)
                self.assertFalse(report["complete"])
                self.assertEqual(
                    report["failure"], "IDENTITY_OR_SOURCE_BINDING_MISMATCH"
                )

    def test_advisory_score_tamper_fails_even_with_consistent_output_hash(self):
        def transform(_method, path, payload, _status, value):
            if path == "/api/score" and "patient_id" not in payload["features"]:
                value["advisory"]["operator_attention_score"] = 0.5
                value["output_sha256"] = live.digest(value["advisory"])

        report, _ = self.verify(transform)
        self.assertFalse(report["complete"])
        self.assertEqual(report["failure"], "CANONICAL_SCORE_MISMATCH")

    def test_all_five_authority_flags_must_be_boolean_false(self):
        for flag in live.AUTHORITY_KEYS:
            for alteration in (True, 0, None):
                with self.subTest(flag=flag, alteration=alteration):

                    def transform(_method, path, payload, _status, value):
                        if (
                            path == "/api/score"
                            and "patient_id" not in payload["features"]
                        ):
                            value["advisory"]["authority"][flag] = alteration
                            value["output_sha256"] = live.digest(value["advisory"])

                    report, _ = self.verify(transform)
                    self.assertFalse(report["complete"])
                    self.assertEqual(report["failure"], "AUTHORITY_BOUNDARY_MISMATCH")

    def test_authority_keys_are_closed(self):
        def transform(_method, path, payload, _status, value):
            if path == "/api/score" and "patient_id" not in payload["features"]:
                value["advisory"]["authority"]["extra_authority"] = False

        report, _ = self.verify(transform)
        self.assertEqual(report["failure"], "AUTHORITY_BOUNDARY_MISMATCH")

    def test_scoring_input_and_output_digest_domains_are_exact(self):
        for field in ("input_sha256", "output_sha256"):
            with self.subTest(field=field):

                def transform(_method, path, payload, _status, value):
                    if path == "/api/score" and "patient_id" not in payload["features"]:
                        value[field] = live.digest(
                            payload if field == "input_sha256" else value
                        )

                report, _ = self.verify(transform)
                self.assertFalse(report["complete"])
                self.assertEqual(report["failure"], "SCORING_DIGEST_MISMATCH")

    def test_score_receipt_and_success_are_exact_booleans(self):
        for field, altered in (
            ("receipt_minted", True),
            ("receipt_minted", 0),
            ("ok", 1),
            ("ok", False),
        ):
            with self.subTest(field=field, altered=altered):

                def transform(_method, path, payload, _status, value):
                    if path == "/api/score" and "patient_id" not in payload["features"]:
                        value[field] = altered

                report, _ = self.verify(transform)
                self.assertEqual(
                    report["failure"], "SCORING_OR_RECEIPT_CONTRACT_MISMATCH"
                )

    def test_liveness_is_not_readiness(self):
        def transform(_method, path, _payload, _status, value):
            if path == "/readyz":
                value["state"] = "UNAVAILABLE"

        report, _ = self.verify(transform)
        self.assertFalse(report["complete"])
        self.assertEqual(report["failure"], "READINESS_CONTRACT_MISMATCH")
        self.assertEqual(len(report["checks"]), 1)

    def test_json_200_is_not_application_contract(self):
        for body, content_type, status in (
            (b"<html>SPA</html>", "text/html", 200),
            (b"{}", "application/json", 302),
            (b"[]", "application/json", 200),
            (b'{"alive":true,"alive":false}', "application/json", 200),
            (b'{"x":1e999}', "application/json", 200),
        ):
            with self.subTest(body=body):
                report, transport = self.verify(
                    lambda *_: live.Observation(status, content_type, body)
                )
                self.assertFalse(report["complete"])
                self.assertEqual(len(transport.calls), 1)

    def test_no_retries_on_transport_failure(self):
        def fail(*_args):
            raise live.LiveVerificationError("HTTP_TRANSPORT_FAILURE")

        report, transport = self.verify(fail)
        self.assertFalse(report["complete"])
        self.assertEqual(report["failure"], "HTTP_TRANSPORT_FAILURE")
        self.assertEqual(len(transport.calls), 1)

    def test_clinical_extra_field_must_not_succeed_or_echo(self):
        for mode in ("accepted", "echo", "extra_field"):
            with self.subTest(mode=mode):

                def transform(_method, path, payload, _status, value):
                    if path == "/api/score" and "patient_id" in payload["features"]:
                        if mode == "accepted":
                            return live.Observation(
                                200, "application/json", b'{"ok":true}'
                            )
                        if mode == "echo":
                            value["error"] = live.REJECTION_MARKER
                        else:
                            value["submitted"] = payload

                report, _ = self.verify(transform)
                self.assertFalse(report["complete"])
                self.assertEqual(len(report["checks"]), 6)

    def test_expected_source_is_an_exact_sha_not_main_or_unknown(self):
        for source in ("main", "HEAD", "a" * 39, "A" * 40, "a" * 41):
            with self.subTest(source=source):
                report, transport = self.verify(expected_source_sha=source)
                self.assertFalse(report["complete"])
                self.assertEqual(
                    report["failure"], "EXACT_APPLICATION_SOURCE_SHA_REQUIRED"
                )
                self.assertEqual(transport.calls, [])

    def test_unicode_escaped_rejected_input_cannot_hide_an_echo(self):
        def transform(_method, path, payload, _status, _value):
            if path == "/api/score" and "patient_id" in payload["features"]:
                encoded_marker = live.REJECTION_MARKER.replace("_", "\\u005f")
                body = ('{"ok":false,"error":"' + encoded_marker + '"}').encode("ascii")
                return live.Observation(400, "application/json", body)

        report, _ = self.verify(transform)
        self.assertFalse(report["complete"])
        self.assertEqual(report["failure"], "REJECTED_INPUT_ECHOED")

    def test_origin_is_closed_and_no_redirect_target_or_credentials_allowed(self):
        for origin in (
            "http://" + live.PUBLIC_HOST,
            "https://" + live.PUBLIC_HOST + ".example",
            "https://example.com",
            "https://user:password@" + live.PUBLIC_HOST,
            live.DEFAULT_ORIGIN + "/other",
            live.DEFAULT_ORIGIN + "?query=1",
            live.DEFAULT_ORIGIN + "#fragment",
            "http://127.0.0.1:8080",
            live.DEFAULT_ORIGIN + ":444",
        ):
            with self.subTest(origin=origin):
                report, transport = self.verify(origin=origin)
                self.assertEqual(report["failure"], "TARGET_NOT_ALLOWED")
                self.assertEqual(transport.calls, [])

    def test_explicit_loopback_is_labelled_test_only(self):
        report, _ = self.verify(origin="http://127.0.0.1:12345", allow_localhost=True)
        self.assertTrue(report["complete"])
        self.assertEqual(report["observation_scope"], "LOCAL_TEST_ONLY")
        with self.assertRaises(live.LiveVerificationError):
            live.target("http://127.0.0.2:12345", allow_localhost=True)

    def test_release_failure_prevents_any_network_call(self):
        transport = FakeTransport(self.manifest, self.kernel)
        with patch.object(
            live,
            "verify_release",
            side_effect=release.ReleaseVerificationError(
                "immutable Git artifact mismatch"
            ),
        ):
            report = live.verify_live(
                origin=live.DEFAULT_ORIGIN,
                expected_source_sha=APPLICATION_SHA,
                space_root=SPACE_ROOT,
                repository_root=Path("unused"),
                transport=transport,
            )
        self.assertFalse(report["complete"])
        self.assertEqual(transport.calls, [])

    def test_canonical_kernel_is_not_loaded_from_application_module(self):
        self.assertNotIn("app", self.kernel.score.__func__.__module__)
        self.assertEqual(
            self.kernel.score.__func__.__module__, "_szl_oac_live_verifier_canonical"
        )

    def test_canonical_constructor_reads_only_verified_memory_snapshots(self):
        def immutable_fixture(_root, *arguments):
            return self.artifact_bytes[arguments[-1].rsplit("/", 1)[1]]

        with (
            patch.object(live, "_git", side_effect=immutable_fixture),
            patch.object(
                Path, "open", side_effect=AssertionError("mutable file opened")
            ),
        ):
            kernel = live._load_kernel(SPACE_ROOT, Path("unused"), self.manifest)
            advisory = kernel.score(live.CASES[0][1])
        self.assertEqual(advisory, self.kernel.score(live.CASES[0][1]))
        self.assertEqual(kernel.model_path.content, self.artifact_bytes["model.json"])
        self.assertEqual(
            kernel.receipt_path.content, self.artifact_bytes["artifact_receipt.json"]
        )
        for field in ("model_path", "receipt_path"):
            with self.subTest(field=field):
                with self.assertRaises(live.LiveVerificationError):
                    setattr(kernel, field, Path("replacement.json"))

    def test_canonical_kernel_rejects_tampered_git_bytes_before_execution(self):
        for changed in release.ARTIFACT_NAMES:
            with self.subTest(artifact=changed):

                def immutable_fixture(_root, *arguments):
                    name = arguments[-1].rsplit("/", 1)[1]
                    raw = self.artifact_bytes[name]
                    return raw + b" " if name == changed else raw

                with (
                    patch.object(live, "_git", side_effect=immutable_fixture),
                    patch.object(
                        live,
                        "exec",
                        side_effect=AssertionError("unverified kernel executed"),
                        create=True,
                    ),
                ):
                    with self.assertRaisesRegex(
                        live.LiveVerificationError, "CANONICAL_KERNEL_ARTIFACT_MISMATCH"
                    ):
                        live._load_kernel(SPACE_ROOT, Path("unused"), self.manifest)

    def test_cli_exclusive_receipt_preserves_first_observation(self):
        report, _ = self.verify()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "live.json"
            with patch.object(live, "verify_live", return_value=report):
                self.assertEqual(
                    live.main(
                        [
                            "--expected-source-sha",
                            APPLICATION_SHA,
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
                first = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    live.main(
                        [
                            "--expected-source-sha",
                            APPLICATION_SHA,
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(first, output.read_bytes())
            retained = json.loads(first)
            self.assertEqual(
                retained["expected_application_source_sha"], APPLICATION_SHA
            )
            self.assertIn("completed_at", retained)

    def test_real_transport_sets_opt_in_without_auth_or_redirect_handling(self):
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.getheader.return_value = "application/json"
        response.read.return_value = b"{}"
        with patch.object(
            live.http.client, "HTTPSConnection", return_value=connection
        ) as constructor:
            observation = live.Transport(live.DEFAULT_ORIGIN).request(
                "POST", "/api/score", {"features": live.CASES[0][1]}
            )
        self.assertEqual(observation.status, 200)
        constructor.assert_called_once_with(
            live.PUBLIC_HOST, 443, timeout=live.REQUEST_SECONDS
        )
        arguments = connection.request.call_args
        self.assertEqual(arguments.args, ("POST", "/api/score"))
        self.assertEqual(
            arguments.kwargs["body"], live.canonical({"features": live.CASES[0][1]})
        )
        headers = arguments.kwargs["headers"]
        self.assertEqual(headers["X-SZL-Preview"], "1")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertFalse(
            any(
                name.lower() in {"authorization", "cookie", "proxy-authorization"}
                for name in headers
            )
        )
        response.read.assert_called_once_with(live.MAX_RESPONSE_BYTES + 1)
        connection.close.assert_called_once()

    def test_real_transport_closes_on_failure_and_enforces_response_size(self):
        for outcome in (
            OSError("transport unavailable"),
            b"x" * (live.MAX_RESPONSE_BYTES + 1),
        ):
            with self.subTest(outcome=type(outcome).__name__):
                connection = Mock()
                if isinstance(outcome, BaseException):
                    connection.getresponse.side_effect = outcome
                else:
                    connection.getresponse.return_value.read.return_value = outcome
                with patch.object(
                    live.http.client, "HTTPSConnection", return_value=connection
                ):
                    with self.assertRaises(live.LiveVerificationError):
                        live.Transport(live.DEFAULT_ORIGIN).request("GET", "/healthz")
                connection.close.assert_called_once()

    def test_absolute_deadline_is_applied_below_buffered_reads(self):
        connection = Mock()
        reader = live._DeadlineReader(io.BytesIO(b"ab"), connection, 0.2)
        with patch.object(live.time, "monotonic", side_effect=[0.05, 0.3]):
            self.assertEqual(reader.readinto(bytearray(1)), 1)
            with self.assertRaises(TimeoutError):
                reader.readinto(bytearray(1))
        self.assertAlmostEqual(connection.settimeout.call_args.args[0], 0.15)
        reader.close()
        self.assertTrue(reader.raw.closed)

    def test_raw_http_header_and_body_budget_cannot_grow_unbounded(self):
        reader = live._DeadlineReader(
            io.BytesIO(b"x" * (live.MAX_RESPONSE_BYTES + 16_385)),
            Mock(),
            live.time.monotonic() + 5,
        )
        try:
            with self.assertRaisesRegex(
                live.LiveVerificationError, "HTTP_RECEIVE_BYTE_LIMIT"
            ):
                reader.readinto(bytearray(live.MAX_RESPONSE_BYTES + 16_385))
        finally:
            reader.close()

    def test_transport_path_allowlist_rejects_arbitrary_effects_before_connection(self):
        with patch.object(live.http.client, "HTTPSConnection") as constructor:
            with self.assertRaises(live.LiveVerificationError):
                live.Transport(live.DEFAULT_ORIGIN).request(
                    "POST", "/admin/restart", {}
                )
        constructor.assert_not_called()

    def test_cli_help_imports_trusted_sibling_under_isolated_python(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(SPACE_ROOT / "verify_live.py"),
                    "--help",
                ],
                cwd=directory,
                capture_output=True,
                check=False,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn(b"--expected-source-sha", result.stdout)
        self.assertIn(b"--origin", result.stdout)


if __name__ == "__main__":
    unittest.main()
