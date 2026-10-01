"""Synthetic, network-blocked contract tests; these are not release evidence."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/verify_invariants_hub_bytes.py"
SPEC = importlib.util.spec_from_file_location("szl_public_byte_observer_test", MODULE_PATH)
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)

SRC, MODEL, KERNEL = "a" * 40, "b" * 40, "c" * 40


def fixture():
    source = {path: (b'PROVENANCE = {"trained_weights_present": False}\n'
                     if path.endswith(".py") else b'{"trained_weights_present": false}\n')
              for path in observer.SOURCE_FILES}
    contract = {
        "schema": "szl.invariants-source-binding/v1",
        "repo_id": observer.HUB_REPO, "source_repository": observer.SOURCE_REPO,
        "artifact_files": sorted(source),
        "expected_artifact_sha256": {path: observer.digest(raw) for path, raw in source.items()},
        "publication_targets": [dict(zip(("repo_type", "source_path", "path_in_repo"), row))
                                for row in sorted(observer.TARGETS)],
        "claims": {"trained_weights_present": False},
    }
    blobs = {("source", path): raw for path, raw in source.items()}
    blobs[("source", observer.CONTRACT)] = json.dumps(contract).encode()
    blobs.update(observer.expected_hub_files(source))
    return contract, source, blobs


class FakeResponse(io.BytesIO):
    def __init__(self, raw=b"fixture", url=None, status=200, headers=None):
        super().__init__(raw)
        self.status = status
        self.headers = {} if headers is None else headers
        self.url = url

    def geturl(self):
        return self.url


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report = Path(self.tmp.name) / "report.json"
        self.contract, self.source, self.blobs = fixture()
        self.requests = []
        blocked = mock.patch("socket.socket.connect", side_effect=AssertionError("network prohibited"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def reader(self, resource):
        self.requests.append(resource)
        expected_pin = {"source": SRC, "model": MODEL, "kernel": KERNEL}[resource.kind]
        self.assertEqual(resource.revision, expected_pin)
        return self.blobs[(resource.kind, resource.path)]

    def run_observation(self, **kwargs):
        args = dict(source_revision=SRC, model_revision=MODEL, kernel_revision=KERNEL,
                    report_path=self.report, enabled=True, reader=self.reader)
        args.update(kwargs)
        return observer.observe(**args)

    def update_contract(self):
        self.blobs[("source", observer.CONTRACT)] = json.dumps(self.contract).encode()

    def test_complete_comparison_is_not_publication_authority(self):
        report = self.run_observation()
        self.assertEqual(report["status"], "BYTE_ALIGNMENT_VERIFIED_REVIEW_REQUIRED")
        self.assertTrue(report["all_bytes_equal"])
        self.assertEqual(len(self.requests), 13)
        self.assertEqual(len(report["comparisons"]), 8)
        self.assertEqual({r["repo_type"] for r in report["comparisons"]}, {"model", "kernel"})
        for key in ("publication_verified", "signature_independently_verified",
                    "write_authorization_verified", "branch_membership_verified", "freshness_verified"):
            self.assertIs(report[key], False)
        self.assertEqual(report["repository_mutation"], "NOT_ATTEMPTED")
        self.assertEqual(json.loads(self.report.read_text()), report)

    def test_default_requires_explicit_read_opt_in(self):
        report = self.run_observation(enabled=False)
        self.assertEqual(self.requests, [])
        self.assertEqual(report["status"], "NOT_OBSERVED")
        self.assertIsNone(report["all_bytes_equal"])

    def test_mutable_revisions_make_no_request(self):
        for field in ("source_revision", "model_revision", "kernel_revision"):
            for bad in ("main", "v1", "a" * 39, "A" * 40, "a" * 40 + "\n", True):
                with self.subTest(field=field, bad=bad):
                    report = self.run_observation(**{field: bad})
                    self.assertEqual(report["reason"], "REVISION_NOT_IMMUTABLE")
        self.assertEqual(self.requests, [])

    def test_stale_success_is_replaced_before_first_read(self):
        self.report.write_text('{"publication_verified":true}')
        def read(resource):
            saved = json.loads(self.report.read_text())
            self.assertIs(saved["publication_verified"], False)
            self.assertIn(saved["status"], ("OBSERVING",))
            return self.reader(resource)
        self.run_observation(reader=read)

    def test_marker_write_failure_prevents_reads(self):
        with mock.patch.object(observer, "atomic_report", side_effect=OSError("private-path")):
            with self.assertRaises(OSError):
                self.run_observation()
        self.assertEqual(self.requests, [])

    def test_source_contract_hash_mismatch_prevents_hub_reads(self):
        self.blobs[("source", sorted(self.source)[0])] += b"unexpected"
        report = self.run_observation()
        self.assertEqual(report["reason"], "SOURCE_BYTES_DO_NOT_MATCH_CONTRACT")
        self.assertTrue(all(r.kind == "source" for r in self.requests))

    def test_line_endings_are_not_normalized(self):
        key = ("model", "build/torch-universal/szl_invariants/__init__.py")
        self.blobs[key] = self.blobs[key].replace(b"\n", b"\r\n")
        report = self.run_observation()
        self.assertEqual(report["status"], "BYTE_DRIFT_OBSERVED")
        self.assertIs(report["all_bytes_equal"], False)
        self.assertEqual(sum(r["status"] == "DRIFT" for r in report["comparisons"]), 1)

    def test_both_generated_metadata_files_are_compared(self):
        for variant in ("torch-cpu", "torch-universal"):
            self.blobs[("kernel", f"build/{variant}/metadata.json")] = b"{}"
        report = self.run_observation()
        drift = [r for r in report["comparisons"] if r["status"] == "DRIFT"]
        self.assertEqual(len(drift), 2)

    def test_unavailable_is_unknown_not_missing_or_zero(self):
        def read(resource):
            if resource.kind == "kernel" and resource.path.endswith("__init__.py"):
                raise observer.ObservationError("HTTP_404_UNAVAILABLE")
            return self.reader(resource)
        report = self.run_observation(reader=read)
        self.assertIsNone(report["all_bytes_equal"])
        self.assertFalse(report["observation_complete"])
        unavailable = [r for r in report["comparisons"] if r["status"] == "UNAVAILABLE"]
        self.assertEqual(len(unavailable), 2)
        self.assertTrue(all(r["observed_bytes"] is None and r["equal"] is None for r in unavailable))
        self.assertEqual(sum(r["status"] == "MATCH" for r in report["comparisons"]), 6)

    def test_partial_drift_and_unavailability_both_survive(self):
        self.blobs[("model", "build/torch-universal/szl_invariants/metadata.json")] = b"changed"
        def read(resource):
            if resource.kind == "kernel":
                raise observer.ObservationError("HTTP_403_UNAVAILABLE")
            return self.reader(resource)
        report = self.run_observation(reader=read)
        self.assertEqual(report["status"], "OBSERVATION_INCOMPLETE")
        self.assertEqual(sum(r["status"] == "DRIFT" for r in report["comparisons"]), 1)
        self.assertIsNone(report["all_bytes_equal"])

    def test_provider_exception_text_is_not_exported(self):
        secret = "hf_private_token-example https://evil.example/private-path"
        for error in (RuntimeError(secret), observer.ObservationError(secret)):
            def read(resource):
                if resource.kind != "source":
                    raise error
                return self.reader(resource)
            report = self.run_observation(reader=read)
            self.assertNotIn(secret, json.dumps(report))
            self.assertNotIn(secret, self.report.read_text())

    def test_source_exception_text_is_not_exported(self):
        def read(_):
            raise OSError("private-path-token")
        report = self.run_observation(reader=read)
        self.assertNotIn("private-path-token", json.dumps(report))
        self.assertIsNone(report["all_bytes_equal"])

    def test_source_byte_limit_is_enforced_with_custom_transport(self):
        self.blobs[("source", sorted(self.source)[0])] = b"x" * (observer.MAX_FILE_BYTES + 1)
        report = self.run_observation()
        self.assertEqual(report["reason"], "FILE_SIZE_INVALID")

    def test_empty_provider_file_is_not_equal(self):
        self.blobs[("model", "build/torch-universal/szl_invariants/metadata.json")] = b""
        report = self.run_observation()
        self.assertFalse(report["observation_complete"])
        self.assertTrue(any(r["reason"] == "FILE_SIZE_INVALID" for r in report["comparisons"]))

    def test_duplicate_contract_json_keys_are_refused(self):
        raw = self.blobs[("source", observer.CONTRACT)]
        self.blobs[("source", observer.CONTRACT)] = raw[:-1] + b',"schema":"other"}'
        report = self.run_observation()
        self.assertEqual(report["reason"], "DUPLICATE_JSON_KEY")
        self.assertEqual(len(self.requests), 1)

    def test_nonfinite_contract_values_are_refused(self):
        for value in (b"NaN", b"Infinity", b"1e999"):
            raw = self.blobs[("source", observer.CONTRACT)]
            with self.subTest(value=value):
                self.blobs[("source", observer.CONTRACT)] = raw[:-1] + b',"unknown":' + value + b'}'
                self.assertEqual(self.run_observation()["reason"], "NONFINITE_JSON_VALUE")
            self.blobs[("source", observer.CONTRACT)] = raw

    def test_malformed_json_is_refused(self):
        for raw in (b"[]", b"\xff", b'{"x":', b"{" * 2000):
            with self.subTest(raw=raw[:10]):
                self.blobs[("source", observer.CONTRACT)] = raw
                report = self.run_observation()
                self.assertEqual(report["reason"], "INVALID_SOURCE_CONTRACT")

    def test_target_remapping_and_new_destinations_are_refused(self):
        self.contract["publication_targets"][0]["path_in_repo"] = "README.md"
        self.update_contract()
        self.assertEqual(self.run_observation()["reason"], "TARGET_SET_MISMATCH")

    def test_target_json_url_is_never_used(self):
        self.contract["publication_targets"][0]["url"] = "http://localhost:8000/secret"
        self.update_contract()
        report = self.run_observation()
        self.assertEqual(report["reason"], "TARGET_SET_MISMATCH")
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("localhost", json.dumps(report))

    def test_wrong_source_repository_is_refused(self):
        self.contract["source_repository"] = "other/repo"
        self.update_contract()
        self.assertEqual(self.run_observation()["reason"], "SOURCE_CONTRACT_IDENTITY_MISMATCH")

    def test_malformed_source_file_or_hash_sets_are_refused(self):
        self.contract["artifact_files"].append("model.joblib")
        self.update_contract()
        self.assertEqual(self.run_observation()["reason"], "SOURCE_FILE_SET_MISMATCH")
        self.contract["artifact_files"].pop()
        self.contract["expected_artifact_sha256"][sorted(self.source)[0]] = True
        self.update_contract()
        self.assertEqual(self.run_observation()["reason"], "SOURCE_HASH_SET_MISMATCH")

    def test_trained_weights_claim_must_be_false_boolean(self):
        for value in (True, 0, None, "false"):
            with self.subTest(value=value):
                self.contract["claims"]["trained_weights_present"] = value
                self.update_contract()
                self.assertEqual(self.run_observation()["reason"], "UNSUPPORTED_SOURCE_CLAIMS")

    def test_source_variants_must_match_even_with_updated_hashes(self):
        path = "torch-ext/szl_invariants/__init__.py"
        self.blobs[("source", path)] += b"variant drift"
        self.contract["expected_artifact_sha256"][path] = observer.digest(self.blobs[("source", path)])
        self.update_contract()
        self.assertEqual(self.run_observation()["reason"], "SOURCE_VARIANT_MISMATCH")

    def test_metadata_uses_original_algorithm_and_exact_serialization(self):
        files = observer.expected_hub_files(self.source)
        raw = files[("kernel", "build/torch-cpu/metadata.json")]
        meta = json.loads(raw)
        d = hashlib.sha256()
        for name in ("__init__.py", "metadata.json"):
            relative = f"szl_invariants/{name}"
            payload = self.source[f"torch-ext/{relative}"]
            d.update(relative.encode())
            d.update(b"\0")
            d.update(hashlib.sha256(payload).digest())
        self.assertEqual(meta["id"], f"_szl_invariants_torch_cpu_{d.hexdigest()[:8]}")
        self.assertEqual(raw, (json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())
        self.assertEqual(meta["backend"], {"type": "cpu"})
        self.assertEqual(meta["version"], 1)

    def test_report_replacement_failure_does_not_report_success(self):
        original = observer.atomic_report
        def persist(path, report):
            if report["status"] == "BYTE_ALIGNMENT_VERIFIED_REVIEW_REQUIRED":
                raise OSError("storage-failed")
            original(path, report)
        with mock.patch.object(observer, "atomic_report", side_effect=persist):
            with self.assertRaises(OSError):
                self.run_observation()
        self.assertFalse(json.loads(self.report.read_text())["observation_complete"])

    def test_cli_without_opt_in_is_nonzero_and_network_free(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = observer.main(["--source-revision", SRC, "--model-revision", MODEL,
                                    "--kernel-revision", KERNEL, "--report", str(self.report)])
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "NOT_OBSERVED")

    def test_cli_filesystem_error_does_not_leak_path(self):
        with mock.patch.object(observer, "atomic_report", side_effect=OSError("private-path")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            status = observer.main(["--source-revision", SRC, "--model-revision", MODEL,
                                    "--kernel-revision", KERNEL, "--report", str(self.report)])
        self.assertEqual(status, 2)
        self.assertNotIn("private-path", output.getvalue())


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.resource = observer.Resource("kernel", KERNEL, "build/torch-cpu/metadata.json")
        blocked = mock.patch("socket.socket.connect", side_effect=AssertionError("network prohibited"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def test_typed_canonical_urls_have_distinct_namespaces(self):
        model = observer.Resource("model", MODEL, "build/torch-universal/szl_invariants/metadata.json")
        self.assertTrue(self.resource.url.startswith("https://huggingface.co/kernels/"))
        self.assertTrue(model.url.startswith("https://huggingface.co/SZLHOLDINGS/"))
        self.assertFalse(model.allows(self.resource.url))
        self.assertFalse(self.resource.allows(model.url))

    def test_arbitrary_paths_are_impossible(self):
        for kind, path in (("source", "../secret"), ("kernel", "model.joblib"), ("other", "README.md")):
            with self.subTest(kind=kind, path=path), self.assertRaises(observer.ObservationError):
                observer.Resource(kind, SRC, path)

    def test_redirection_remains_same_typed_repo_revision_and_path(self):
        good = f"https://huggingface.co/api/resolve-cache/kernels/{observer.HUB_REPO}/{KERNEL}/{self.resource.path}?etag=abc"
        self.assertTrue(self.resource.allows(good))
        bad_urls = [good.replace("https:", "http:"), good.replace("huggingface.co", "evil.example"),
                    good.replace("huggingface.co", "user@huggingface.co"),
                    good.replace("huggingface.co", "huggingface.co:444"),
                    good.replace("kernels/", "models/"), good.replace(KERNEL, "main"),
                    good.replace(observer.HUB_REPO, "SZLHOLDINGS/other"), good + "#fragment",
                    good + "\n", good.replace("metadata.json", "../README.md"),
                    good.replace("metadata.json", "%2e%2e/README.md"),
                    good.replace("huggingface.co/", "huggingface.co\\/")]
        for url in bad_urls:
            with self.subTest(url=url):
                self.assertFalse(self.resource.allows(url))

    def test_get_does_not_consume_ambient_tokens_or_proxies(self):
        fake = mock.Mock()
        fake.open.return_value = FakeResponse(url=self.resource.url)
        with mock.patch.dict(observer.os.environ, {"HF_TOKEN": "private", "HTTPS_PROXY": "http://evil"}), \
                mock.patch.object(observer.urllib.request, "build_opener", return_value=fake) as build:
            self.assertEqual(observer.public_read(self.resource), b"fixture")
        request = fake.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertNotIn("authorization", {k.lower() for k in request.headers})
        self.assertIsNone(request.data)
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertEqual(fake.open.call_count, 1)

    def test_valid_cache_redirect_is_followed_without_credentials(self):
        cache = f"/api/resolve-cache/kernels/{observer.HUB_REPO}/{KERNEL}/{self.resource.path}"
        fake = mock.Mock()
        fake.open.side_effect = [urllib.error.HTTPError(self.resource.url, 307, "redirect",
                                                       {"Location": cache}, None),
                                 FakeResponse(url="https://huggingface.co" + cache)]
        with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
            self.assertEqual(observer.public_read(self.resource), b"fixture")
        self.assertEqual(fake.open.call_count, 2)

    def test_cross_origin_redirect_is_not_contacted(self):
        fake = mock.Mock()
        fake.open.side_effect = urllib.error.HTTPError(self.resource.url, 302, "redirect",
                                                       {"Location": "https://evil.example/"}, None)
        with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
            with self.assertRaisesRegex(observer.ObservationError, "UNSAFE_REDIRECT"):
                observer.public_read(self.resource)
        self.assertEqual(fake.open.call_count, 1)

    def test_redirect_cycle_is_rejected(self):
        fake = mock.Mock()
        fake.open.side_effect = urllib.error.HTTPError(self.resource.url, 302, "redirect",
                                                       {"Location": self.resource.url}, None)
        with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
            with self.assertRaisesRegex(observer.ObservationError, "REDIRECT_CYCLE"):
                observer.public_read(self.resource)
        self.assertEqual(fake.open.call_count, 1)

    def test_failure_statuses_are_unavailable_not_missing(self):
        for status in (401, 403, 404, 429, 500):
            fake = mock.Mock()
            fake.open.side_effect = urllib.error.HTTPError(self.resource.url, status, "private-token", {}, None)
            with self.subTest(status=status), \
                    mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
                with self.assertRaises(observer.ObservationError) as caught:
                    observer.public_read(self.resource)
                self.assertIn("UNAVAILABLE", str(caught.exception))
                self.assertNotIn("private-token", str(caught.exception))
            self.assertEqual(fake.open.call_count, 1)

    def test_oversize_and_empty_responses_are_refused(self):
        for body in (b"", b"x" * (observer.MAX_FILE_BYTES + 1)):
            fake = mock.Mock()
            fake.open.return_value = FakeResponse(body, self.resource.url)
            with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
                with self.assertRaisesRegex(observer.ObservationError, "FILE_SIZE_INVALID"):
                    observer.public_read(self.resource)

    def test_response_origin_is_validated_even_if_opener_redirects(self):
        fake = mock.Mock()
        fake.open.return_value = FakeResponse(url="https://evil.example")
        with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
            with self.assertRaisesRegex(observer.ObservationError, "UNSAFE_RESPONSE_LOCATION"):
                observer.public_read(self.resource)

    def test_compressed_responses_are_rejected(self):
        fake = mock.Mock()
        fake.open.return_value = FakeResponse(url=self.resource.url, headers={"Content-Encoding": "gzip"})
        with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
            with self.assertRaisesRegex(observer.ObservationError, "UNSUPPORTED_CONTENT_ENCODING"):
                observer.public_read(self.resource)

    def test_network_error_text_is_redacted(self):
        fake = mock.Mock()
        fake.open.side_effect = OSError("private-path-token")
        with mock.patch.object(observer.urllib.request, "build_opener", return_value=fake):
            with self.assertRaisesRegex(observer.ObservationError, "TRANSPORT_UNAVAILABLE"):
                observer.public_read(self.resource)

    def test_source_redirect_cannot_escape_pin_or_add_query(self):
        source = observer.Resource("source", SRC, observer.CONTRACT)
        self.assertTrue(source.allows(source.url))
        self.assertFalse(source.allows(source.url + "?token=abc"))
        self.assertFalse(source.allows(source.url.replace(SRC, "main")))


if __name__ == "__main__":
    unittest.main()
