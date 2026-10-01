"""Offline tests for a bounded, credentialless metadata collector."""
from __future__ import annotations
import base64
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("observer", Path(__file__).with_name("observe_kernel_fleet.py"))
assert SPEC and SPEC.loader
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
A, B = "a" * 40, "b" * 40
ID = "SZLHOLDINGS/szl-kernels"


def reply(value, link="", status=200):
    return status, {"content-type": "application/json", "link": link}, json.dumps(value).encode()


def list_item(identity=ID, sha=A, tags=None):
    return {"id": identity, "sha": sha, "tags": ["kernel"] if tags is None else tags}


def transport(url):
    if "/refs" in url:
        return reply({"branches": [{"name": "main", "targetCommit": A}, {"name": "v1", "targetCommit": B}], "tags": []})
    if "/revision/" in url:
        return reply({"id": ID, "sha": url.rsplit("/", 1)[1], "repoType": "kernel"})
    return reply([list_item()])


class UrlTests(unittest.TestCase):
    def test_fixed_endpoints(self):
        for u in (m.LIST_URL, m.LIST_URL + "&cursor=opaque%3D", m.ORIGIN + "/api/kernels/" + ID + "/refs", m.ORIGIN + "/api/kernels/" + ID + "/revision/" + A):
            self.assertEqual(m.validate_url(u), u)

    def test_bad_urls(self):
        for u in ("http://huggingface.co/api/kernels?author=SZLHOLDINGS&limit=100", m.LIST_URL + "#x", m.LIST_URL.replace("huggingface.co", "huggingface.co:443"), m.LIST_URL.replace("huggingface.co", "huggingface.co@evil.example"), m.LIST_URL + "&author=OTHER", m.LIST_URL.replace("SZLHOLDINGS", "OTHER"), m.LIST_URL + "&token=secret", m.LIST_URL + "&cursor=", m.LIST_URL + "\n", m.ORIGIN + "/api/models/" + ID, m.ORIGIN + "/api/kernels/" + ID + "/revision/main", m.ORIGIN + "/api/kernels/SZLHOLDINGS/../refs", m.ORIGIN + "/api/kernels/" + ID + "/refs?token=bad"):
            with self.subTest(url=u), self.assertRaises(ValueError):
                m.validate_url(u)

    def test_next_scope(self):
        nxt = m.LIST_URL + "&cursor=x"
        self.assertEqual(m.next_link('<' + nxt + '>; rel="next"'), nxt)
        self.assertIsNone(m.next_link(""))
        for header in ('<https://evil.example/>; rel="next"', '<' + nxt + '>; rel="next", <' + nxt + '>; rel="next"', 'garbage', '<' + m.ORIGIN + '/api/kernels/' + ID + '/refs>; rel="next"'):
            with self.assertRaises(ValueError):
                m.next_link(header)

    def test_redirect_denied(self):
        with self.assertRaises(m.ObservationError):
            m.NoRedirect().redirect_request(None, None, 302, None, {}, "https://example.org/")

    def test_bad_ids_and_revisions(self):
        for value in (None, 0, True, "OTHER/a", "SZLHOLDINGS/a..b", "SZLHOLDINGS/a.git", "SZLHOLDINGS/a\n"):
            with self.assertRaises(m.ObservationError): m.repo_id(value)
        for value in (None, True, "main", "v1", "a" * 39, "A" * 40, A + "\n"):
            with self.assertRaises(m.ObservationError): m.revision(value)


class TransportTests(unittest.TestCase):
    def test_raw_bytes_bound_to_digest(self):
        reader = m.PublicReader(transport=transport)
        reader.read(m.LIST_URL)
        record = reader.records[0]
        self.assertEqual(m.digest(base64.b64decode(record["body_base64"])), record["body_sha256"])
        self.assertIsNotNone(record["completed_at"])

    def test_unknown_http_not_empty(self):
        for status in (401, 403, 404, 429, 500):
            with self.subTest(status=status):
                reader = m.PublicReader(transport=lambda u: reply([], status=status))
                result = m.collect(reader, source_revision=A)
                self.assertFalse(result["observation_complete"])
                self.assertIsNone(result["counts"])
                self.assertEqual(reader.records[0]["http_status"], status)

    def test_no_network_error_strings(self):
        def fail(_url): raise OSError("secret-free simulation, must not be reproduced")
        reader = m.PublicReader(transport=fail)
        with self.assertRaises(m.ObservationError): reader.read(m.LIST_URL)
        self.assertEqual(reader.records[0]["error"], "TRANSPORT_UNAVAILABLE")

    def test_content_type_and_json(self):
        for raw in (b'[{"id": "a", "id": "b"}]', b'{', b'NaN', b'1e9999', b'\xff'):
            with self.assertRaises(m.ObservationError): m.strict_json(raw)
        reader = m.PublicReader(transport=lambda u: (200, {"content-type": "text/html"}, b'[]'))
        with self.assertRaises(m.ObservationError): reader.read(m.LIST_URL)

    def test_budgets(self):
        reader = m.PublicReader(transport=lambda u: (200, {"content-type": "application/json"}, b'x' * (m.MAX_BODY + 1)))
        with self.assertRaises(m.ObservationError): reader.read(m.LIST_URL)
        reader = m.PublicReader(transport=transport)
        reader.deadline = 0
        with self.assertRaises(m.ObservationError): reader.read(m.LIST_URL)
        with mock.patch.object(m, "MAX_REQUESTS", 0):
            with self.assertRaises(m.ObservationError): m.PublicReader(transport=transport).read(m.LIST_URL)

    def test_invalid_url_never_reaches_transport(self):
        fn = mock.Mock(side_effect=AssertionError("called"))
        with self.assertRaises(m.ObservationError): m.PublicReader(transport=fn).read("https://evil.example/")
        fn.assert_not_called()


class ListingTests(unittest.TestCase):
    def test_two_pages(self):
        nxt = m.LIST_URL + "&cursor=two"
        def pages(url):
            return reply([list_item()],["", '<' + nxt + '>; rel="next"'][url == m.LIST_URL]) if url == m.LIST_URL else reply([list_item("SZLHOLDINGS/second")])
        result = m.listing(m.PublicReader(transport=pages))
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["items"]), 2)

    def test_bad_shapes_and_duplicate_items(self):
        for data in ({"items": []}, [1], [list_item(), list_item()], [{"id": ID, "sha": "bad"}], [list_item("OTHER/repo")], [dict(list_item(), tags=True)], [dict(list_item(), repoType="model")]):
            with self.subTest(data=data):
                result = m.listing(m.PublicReader(transport=lambda u: reply(data)))
                self.assertFalse(result["complete"])

    def test_loop_and_pagination_failure(self):
        for header in ('<' + m.LIST_URL + '>; rel="next"', '<https://evil.example/>; rel="next"'):
            result = m.listing(m.PublicReader(transport=lambda u: reply([list_item()], header)))
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["items"]), 1)

    def test_optional_list_revision(self):
        result = m.listing(m.PublicReader(transport=lambda u: reply([{"id": ID}])))
        self.assertTrue(result["complete"])
        self.assertIsNone(result["items"][0]["listed_revision"])

    def test_missing_tags_unknown(self):
        result = m.listing(m.PublicReader(transport=lambda u: reply([{"id": ID, "sha": A}])))
        self.assertTrue(result["complete"])
        self.assertIsNone(result["items"][0]["tagged_kernel"])


class CollectionTests(unittest.TestCase):
    def test_complete_is_only_metadata(self):
        report = m.collect(m.PublicReader(transport=transport), source_revision=A)
        self.assertTrue(report["observation_complete"])
        self.assertEqual(report["counts"]["kernel_api_items"], 1)
        self.assertEqual(report["counts"]["v1_present"], 1)
        self.assertEqual(report["status"], "OBSERVED_METADATA_ONLY")
        self.assertEqual(report["fleet_qualification"], "HOLD")
        self.assertFalse(report["atomic_snapshot"])
        self.assertFalse(report["production_authorization"])
        self.assertFalse(report["runtime_loaded"])

    def test_refs_verify_when_list_omits_sha(self):
        def minimal(url): return reply([{"id": ID, "tags": ["kernel"]}]) if url == m.LIST_URL else transport(url)
        report = m.collect(m.PublicReader(transport=minimal), source_revision=A)
        self.assertTrue(report["observation_complete"])
        self.assertEqual(report["items"][0]["refs"]["main"], A)

    def test_empty_observed_list_is_zero(self):
        report = m.collect(m.PublicReader(transport=lambda u: reply([])), source_revision=A)
        self.assertTrue(report["observation_complete"])
        self.assertEqual(report["counts"]["kernel_api_items"], 0)

    def test_missing_v1_is_observed_not_assumed(self):
        def no_v1(url):
            if url.endswith("/refs"):
                return reply({"branches": [{"name": "main", "targetCommit": A}]})
            return transport(url)
        report = m.collect(m.PublicReader(transport=no_v1), source_revision=A)
        self.assertTrue(report["observation_complete"])
        self.assertEqual(report["counts"]["v1_missing_in_observed_refs"], 1)

    def test_failed_refs_preserve_unknown(self):
        def fail_refs(url): return reply({}, status=403) if url.endswith("/refs") else transport(url)
        report = m.collect(m.PublicReader(transport=fail_refs), source_revision=A)
        self.assertFalse(report["observation_complete"])
        self.assertIsNone(report["counts"])
        self.assertIsNone(report["items"][0]["v1_present"])

    def test_revision_mismatch(self):
        def wrong(url): return reply({"id": ID, "sha": "c" * 40}) if "/revision/" in url else transport(url)
        report = m.collect(m.PublicReader(transport=wrong), source_revision=A)
        self.assertFalse(report["observation_complete"])
        self.assertEqual(report["items"][0]["error"], "IMMUTABLE_METADATA_MISMATCH")

    def test_list_drift(self):
        count = 0
        def drift(url):
            nonlocal count
            if url == m.LIST_URL:
                count += 1
                return reply([list_item(sha=A if count == 1 else B)])
            return transport(url)
        report = m.collect(m.PublicReader(transport=drift), source_revision=A)
        self.assertFalse(report["list_stable_across_two_passes"])
        self.assertIsNone(report["counts"])

    def test_refs_drift(self):
        count = 0
        def drift(url):
            nonlocal count
            if url.endswith("/refs"):
                count += 1
                return reply({"branches": [{"name": "main", "targetCommit": A if count == 1 else B}]})
            return transport(url)
        report = m.collect(m.PublicReader(transport=drift), source_revision=A)
        self.assertFalse(report["observation_complete"])
        self.assertEqual(report["items"][0]["error"], "REFS_MOVED_DURING_OBSERVATION")

    def test_invalid_and_paginated_refs(self):
        for obj, link in (({"branches": []}, ""), ({"branches": [{"name": "main", "targetCommit": A}] * 2}, ""), ({"branches": "bad"}, ""), ({"branches": [{"name": "main", "targetCommit": A}]}, "anything")):
            with self.assertRaises(m.ObservationError):
                m.refs(m.PublicReader(transport=lambda u: reply(obj, link)), ID)

    def test_no_tags_never_counted_as_untagged(self):
        def no_tags(url): return reply([{"id": ID, "sha": A}]) if url == m.LIST_URL else transport(url)
        report = m.collect(m.PublicReader(transport=no_tags), source_revision=A)
        self.assertTrue(report["observation_complete"])
        self.assertIsNone(report["counts"]["tagged_kernel"])
        self.assertIsNone(report["counts"]["untagged_kernel"])


if __name__ == "__main__":
    unittest.main()
