from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("szl_live_mesh_v4_test", ROOT / "tools" / "szl_live_mesh.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("test subject unavailable")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
A, L, N, F, R = (c * 40 for c in "abcde")
NOW = "2026-09-11T23:10:00Z"


class Fixture:
    """Synthetic responses only: no real network or deployment evidence."""
    def __init__(self):
        gh = M.ORIGINS["github"]
        self.responses = {
            f"{gh}/repos/{repo}/commits/main": (200, {"sha": sha})
            for repo, sha in zip(M.REPOS.values(), (A, L, N, F, R))
        }
        raw, product = M.ORIGINS["raw"], M.ORIGINS["product"]
        self.pin_url = f"{raw}/szl-holdings/a11oy/{A}/scripts/hf_publish_lyte_enterprise.py"
        self.proof_source = f"{raw}/szl-holdings/a11oy-net/{N}/health.json"
        self.proof_live = f"{M.ORIGINS['proof']}/health.json"
        proof = json.dumps({"probe_contract": "STATIC_DOCUMENT", "sha": "f" * 40}) + "\n"
        self.responses.update({
            self.pin_url: (200, f'SOURCE_REVISION = "{L}"\n'),
            f"{product}/api/build-info": (200, {"build": {"revision": A}}),
            f"{product}/api/a11oy/v1/honest": (200, {"git_sha": A}),
            f"{product}/api/a11oy/healthz": (200, {"status": "ok"}),
            f"{product}/api/a11oy/v1/frontier/surfaces": (200, {"surfaces": []}),
            f"{M.ORIGINS['hf_a11oy']}/api/build-info": (200, {"build": {"revision": A}}),
            f"{M.ORIGINS['hf_lyte']}/api/build-info": (200, {"source_revision": L, "source_binding": {"bindings_agree": True}}),
            f"{M.ORIGINS['hf_lyte']}/api/lyte/v2/metrics": (200, {"metrics": []}),
            self.proof_source: (200, proof), self.proof_live: (200, proof),
        })
        self.calls = []
        self.second = {}
        self.counts = {}

    def __call__(self, url, *, json_ok=True):
        self.calls.append((url, json_ok))
        self.counts[url] = self.counts.get(url, 0) + 1
        if self.counts[url] > 1 and url in self.second:
            return copy.deepcopy(self.second[url])
        return copy.deepcopy(self.responses.get(url, (404, {})))

    def report(self):
        return M.probe(self, lambda: NOW)


class MeshContractTests(unittest.TestCase):
    def test_all_metadata_consistent_still_never_authorizes(self):
        result = Fixture().report()
        self.assertEqual(result["observation_state"], "METADATA_CONSISTENT")
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["state"], "HOLD")
        self.assertTrue(result["product_source_parity"])
        self.assertTrue(result["lyte_source_parity"])
        for field in ("p01_close", "product_aligned", "production_authorization", "mutations", "signed_off_done"):
            self.assertIs(result[field], False)
        self.assertEqual(len(result["qualification_blockers"]), 5)

    def test_original_false_green_shape_is_rejected(self):
        fixture = Fixture()
        for url in list(fixture.responses):
            if ("a11oy/commits" in url or "/api/build-info" in url and "lyte" not in url
                    or "a11oy-net" in url or url == fixture.proof_live):
                fixture.responses[url] = (503, {})
        result = fixture.report()
        self.assertEqual(result["state"], "HOLD")
        self.assertFalse(result["product_source_parity"])
        self.assertNotEqual(result["blockers"], [])

    def test_every_required_endpoint_cannot_silently_disappear(self):
        for url in Fixture().responses:
            with self.subTest(url=url):
                fixture = Fixture()
                fixture.responses[url] = (503, {})
                result = fixture.report()
                self.assertEqual(result["state"], "HOLD")
                self.assertNotEqual(result["blockers"], [])

    def test_no_network_returns_no_empty_green(self):
        result = M.probe(lambda *a, **k: (0, {}), lambda: NOW)
        self.assertEqual(result["state"], "HOLD")
        self.assertFalse(result["p01_close"])
        self.assertGreater(len(result["blockers"]), 10)

    def test_non200_with_valid_looking_identity_is_not_used(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["product"] + "/api/build-info"] = (500, {"build": {"revision": A}})
        result = fixture.report()
        self.assertIsNone(result["planes"]["a11oy_product"])
        self.assertFalse(result["product_source_parity"])

    def test_honest_identity_is_actually_compared(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["product"] + "/api/a11oy/v1/honest"] = (200, {"git_sha": F})
        self.assertIn("product_source_parity:SOURCE_MISMATCH", fixture.report()["blockers"])

    def test_conflicting_runtime_aliases_do_not_prefer_first(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["hf_lyte"] + "/api/build-info"] = (200, {"source_revision": L, "build": {"revision": F}, "source_binding": {"bindings_agree": True}})
        result = fixture.report()
        self.assertEqual(result["identity_states"]["lyte_live_space"], "CONFLICT")
        self.assertFalse(result["lyte_source_parity"])

    def test_malformed_alias_cannot_be_masked_by_valid_alias(self):
        value, state = M._identity({"source_revision": "main", "build": {"revision": A}}, (("source_revision",), ("build", "revision")))
        self.assertIsNone(value)
        self.assertEqual(state, "INVALID")

    def test_sha_types_strict(self):
        for value in (None, True, 1, [], {}, "main", "A" * 40, "a" * 39, "a" * 40 + "\n"):
            with self.subTest(value=value):
                self.assertIsNone(M._sha({"sha": value}, "sha"))

    def test_binding_agreement_must_be_literal_true(self):
        for value in (None, False, "true", "false", 1, [], {}):
            fixture = Fixture()
            fixture.responses[M.ORIGINS["hf_lyte"] + "/api/build-info"][1]["source_binding"]["bindings_agree"] = value
            self.assertIn("lyte_live_space:BINDING_AGREEMENT_NOT_OBSERVED", fixture.report()["blockers"])

    def test_static_proof_historical_sha_not_misread_as_deployment(self):
        result = Fixture().report()
        self.assertEqual(result["proof_document"]["historical_sha"], "f" * 40)
        self.assertNotEqual(result["proof_document"]["historical_sha"], N)
        self.assertEqual(result["proof_document"]["state"], "BYTE_PARITY")
        self.assertIs(result["proof_document"]["deployment_verified"], False)
        self.assertNotIn("PROOF_PUBLISHED_NE_GITHUB", result["blockers"])

    def test_proof_byte_change_detected_even_when_json_equivalent(self):
        fixture = Fixture()
        status, text = fixture.responses[fixture.proof_live]
        fixture.responses[fixture.proof_live] = (status, text + " ")
        self.assertEqual(fixture.report()["proof_document"]["state"], "DOCUMENT_MISMATCH")

    def test_proof_requires_both_reads(self):
        for target in ("proof_live", "proof_source"):
            fixture = Fixture()
            fixture.responses[getattr(fixture, target)] = (404, "")
            self.assertEqual(fixture.report()["proof_document"]["state"], "UNAVAILABLE")

    def test_proof_unknown_contract_rejected(self):
        fixture = Fixture()
        fixture.responses[fixture.proof_live] = (200, '{"probe_contract":"LIVE"}')
        self.assertEqual(fixture.report()["proof_document"]["state"], "INVALID")

    def test_pin_fetched_by_exact_observed_source_not_main(self):
        fixture = Fixture()
        fixture.report()
        urls = [url for url, _ in fixture.calls]
        self.assertIn(fixture.pin_url, urls)
        self.assertFalse(any("raw.githubusercontent.com" in url and "/main/" in url for url in urls))

    def test_duplicate_pin_assignment_rejected(self):
        fixture = Fixture()
        fixture.responses[fixture.pin_url] = (200, f'SOURCE_REVISION = "{L}"\nSOURCE_REVISION = "{L}"\n')
        self.assertIn("lyte_publisher_pin:UNAVAILABLE_OR_AMBIGUOUS", fixture.report()["blockers"])

    def test_all_end_source_changes_detected(self):
        for name, repo in M.REPOS.items():
            fixture = Fixture()
            fixture.second[M.ORIGINS["github"] + f"/repos/{repo}/commits/main"] = (200, {"sha": "f" * 40})
            self.assertIn(f"{name}:SOURCE_CHANGED_DURING_OBSERVATION", fixture.report()["blockers"])

    def test_missing_end_read_detected(self):
        fixture = Fixture()
        fixture.second[M.ORIGINS["github"] + "/repos/szl-holdings/a11oy/commits/main"] = (0, {})
        self.assertIn("a11oy:END_SOURCE_UNAVAILABLE", fixture.report()["blockers"])

    def test_health_degraded_status_not_ok(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["product"] + "/api/a11oy/healthz"] = (200, {"status": "degraded"})
        self.assertIn("product_healthz:STATUS_NOT_OK", fixture.report()["blockers"])

    def test_empty_metadata_object_does_not_become_identity(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["product"] + "/api/build-info"] = (200, {})
        self.assertIn("a11oy_product:UNAVAILABLE", fixture.report()["blockers"])

    def test_object_endpoint_malformed_body_is_unavailable(self):
        for body in ([], None, "ok", True, {"error": "failure", "sha": A}):
            fixture = Fixture()
            fixture.responses[M.ORIGINS["github"] + "/repos/szl-holdings/a11oy/commits/main"] = (200, body)
            self.assertFalse(fixture.report()["product_source_parity"])

    def test_fetch_adapter_exception_does_not_echo_secret(self):
        def bad(*args, **kwargs):
            raise RuntimeError("synthetic-sensitive-value")
        result = M.probe(bad, lambda: NOW)
        self.assertNotIn("synthetic-sensitive-value", json.dumps(result))
        self.assertEqual(result["state"], "HOLD")

    def test_invalid_http_status_is_rejected(self):
        for status in (True, "200", -1, 600):
            self.assertEqual(M._observe(lambda *a, **k: (status, {}), "unused")[0], 0)

    def test_time_window_invalid_expired_or_naive_fails(self):
        for finish in ("2026-09-11T23:09:59Z", "2026-09-11T23:20:01Z", "2026-09-11T23:10:00", "bad"):
            stamps = iter((NOW, finish))
            result = M.probe(Fixture(), lambda: next(stamps))
            self.assertIn("OBSERVATION_WINDOW_INVALID_OR_EXPIRED", result["blockers"])

    def test_output_order_deterministic_for_same_inputs_and_time(self):
        self.assertEqual(Fixture().report(), Fixture().report())


class TransportAndOutputTests(unittest.TestCase):
    def test_json_duplicate_keys_rejected(self):
        with self.assertRaises(M.MeshError):
            M._decode(b'{"sha":"a","sha":"b"}', True)

    def test_json_non_finite_rejected(self):
        for text in (b'{"x": NaN}', b'{"x": Infinity}', b'{"x": -Infinity}', b'{"x":1e999}'):
            with self.assertRaises(M.MeshError):
                M._decode(text, True)

    def test_oversize_rejected(self):
        with self.assertRaises(M.MeshError):
            M._decode(b"a" * (M.MAX_BODY_BYTES + 1), False)

    def test_malformed_utf8_rejected(self):
        with self.assertRaises(UnicodeError):
            M._decode(b"\xff", False)

    def test_untrusted_urls_refused_without_network(self):
        for url in ("http://a11oy.net/health.json", "https://evil.example/", "https://a11oy.net.evil.example/", "https://user:secret@a11oy.net/", "https://a11oy.net:8443/", "https://a11oy.net/../private", "https://a11oy.net/%2e%2e/", "https://a11oy.net/health.json?token=abc", "https://a11oy.net/#a", "https://a11oy.net/\\private", "https://a11oy.net/\n"):
            with self.subTest(url=url):
                self.assertFalse(M._allowed_url(url))
                self.assertEqual(M._get(url), (0, {"error": "URL_REFUSED"}))

    def test_declared_https_origin_accepted(self):
        self.assertTrue(M._allowed_url("https://a11oy.net/health.json"))

    def test_redirects_never_followed(self):
        with self.assertRaises(M.MeshError):
            M._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example/")

    def test_timeout_must_be_bounded_real_number(self):
        for timeout in (True, "12", 0, -1, float("inf"), float("nan"), 31):
            self.assertEqual(M._get("https://a11oy.net/health.json", timeout=timeout)[1]["error"], "TIMEOUT_REFUSED")

    def test_new_output_roundtrip_and_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            M._write_new(path, "{}\n")
            self.assertEqual(path.read_bytes(), b"{}\n")
            with self.assertRaises(FileExistsError):
                M._write_new(path, "different")
            self.assertEqual(path.read_bytes(), b"{}\n")

    def test_symlink_target_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original"
            original.write_text("keep")
            link = Path(tmp) / "link"
            link.symlink_to(original)
            with self.assertRaises(OSError):
                M._write_new(link, "bad")
            self.assertEqual(original.read_text(), "keep")

    def test_symlink_parent_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "real"
            actual.mkdir()
            link = root / "link"
            link.symlink_to(actual, target_is_directory=True)
            with self.assertRaises(M.MeshError):
                M._write_new(link / "receipt.json", "{}")
            self.assertFalse((actual / "receipt.json").exists())

    def test_metadata_only_cli_always_hold_exit_code(self):
        with patch.object(M, "probe", return_value=Fixture().report()), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(M.main([]), 2)
            self.assertEqual(json.loads(output.getvalue())["state"], "HOLD")


class FalseAlignedDoublesTests(unittest.TestCase):
    """Seven V5 doubles that must not mint ALIGNED. HTML-metrics is the merge blocker."""

    def _assert_divergent(self, result):
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["observation_state"], "INCOMPLETE_OR_DIVERGENT")
        self.assertFalse(result["production_authorization"])
        self.assertFalse(result["product_aligned"])

    def test_html_200_metrics_is_not_aligned(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["hf_lyte"] + "/api/lyte/v2/metrics"] = (
            200, "<!doctype html><html><body>metrics ok</body></html>",
        )
        result = fixture.report()
        self._assert_divergent(result)
        self.assertFalse(result["lyte_source_parity"])
        self.assertTrue(any("lyte_metrics_alias:HTML_200_NOT_METRICS" in item for item in result["blockers"]))
        self.assertIn("lyte_source_parity:INCOMPLETE", result["blockers"])

    def test_404_with_a_sha_is_not_identity(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["product"] + "/api/build-info"] = (404, {"sha": A, "build": {"revision": A}})
        result = fixture.report()
        self._assert_divergent(result)
        self.assertTrue(any("IDENTITY_FROM_ERROR_REFUSED" in item for item in result["blockers"]))
        self.assertFalse(result["product_source_parity"])

    def test_missing_proof_is_incomplete(self):
        fixture = Fixture()
        fixture.responses[fixture.proof_live] = (404, "")
        result = fixture.report()
        self._assert_divergent(result)
        self.assertEqual(result["proof_document"]["state"], "UNAVAILABLE")

    def test_skipped_metrics_sibling_cannot_ride_green_parent(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["hf_lyte"] + "/api/lyte/v2/metrics"] = (503, {})
        result = fixture.report()
        self._assert_divergent(result)
        self.assertFalse(result["lyte_source_parity"])
        self.assertIn("lyte_source_parity:INCOMPLETE", result["blockers"])

    def test_string_true_binding_is_not_observed(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["hf_lyte"] + "/api/build-info"][1]["source_binding"]["bindings_agree"] = "true"
        result = fixture.report()
        self._assert_divergent(result)
        self.assertIn("lyte_live_space:BINDING_AGREEMENT_NOT_OBSERVED", result["blockers"])

    def test_conflicting_runtime_aliases_divergent(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["hf_lyte"] + "/api/build-info"] = (
            200, {"source_revision": L, "build": {"revision": F}, "source_binding": {"bindings_agree": True}},
        )
        result = fixture.report()
        self._assert_divergent(result)
        self.assertEqual(result["identity_states"]["lyte_live_space"], "CONFLICT")

    def test_non200_identity_body_is_divergent(self):
        fixture = Fixture()
        fixture.responses[M.ORIGINS["product"] + "/api/a11oy/v1/honest"] = (500, {"git_sha": A})
        result = fixture.report()
        self._assert_divergent(result)
        self.assertFalse(result["product_source_parity"])


if __name__ == "__main__":
    unittest.main()
