"""Offline regressions; no live Hub, provider, or model calls.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import copy
import io
import itertools
import json
import tempfile
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

from inference import hf_frontier as hf

AT = "2026-09-09T12:00:00+00:00"
SOURCE = "1" * 40


def raw(repo_id: str) -> dict:
    return {"id": repo_id, "modelId": repo_id, "sha": "a" * 40,
            "tags": ["license:apache-2.0"], "cardData": {"license": "apache-2.0"},
            "gated": False, "private": False, "disabled": False,
            "pipeline_tag": "text-generation", "library_name": "pytorch"}


class Client:
    def fetch(self, repo_id):
        return raw(repo_id)


def manifest():
    return hf.refresh(client=Client(), observed_at=AT, source_revision=SOURCE)


def rehash(value):
    for row in value["receipts"]:
        row["receiptSha256"] = hf.sha256({k: v for k, v in row.items() if k != "receiptSha256"})
    value["semanticSha256"] = hf._semantic(value)
    value["manifestSha256"] = hf.sha256({k: v for k, v in value.items() if k != "manifestSha256"})


class FrontierV2Tests(unittest.TestCase):
    def test_explicit_hold_and_reject_never_become_evaluate(self):
        for disposition in (hf.Disposition.HOLD, hf.Disposition.REJECT, hf.Disposition.WATCH):
            with self.subTest(disposition=disposition):
                candidate = hf.Candidate("owner/model", "test", disposition, "test")
                decision = hf.gate(hf.normalize_metadata(candidate, raw(candidate.repo_id)))
                self.assertEqual(decision["effectiveDisposition"], disposition.value)
                self.assertIs(decision["evaluationMetadataAdmitted"], disposition != hf.Disposition.REJECT)

    def test_evaluate_requires_affirmative_ungated_metadata(self):
        candidate = hf.CANDIDATES[0]
        for value in (None, 0, 1, "false", "unknown", [], {}, True, "manual", "auto"):
            with self.subTest(value=value):
                data = raw(candidate.repo_id)
                data["gated"] = value
                self.assertEqual(hf.gate(hf.normalize_metadata(candidate, data))["effectiveDisposition"], "HOLD")
        data.pop("gated")
        self.assertIn("gating_unknown", hf.gate(hf.normalize_metadata(candidate, data))["reasonCodes"])

    def test_clean_metadata_evaluates_but_grants_no_execution(self):
        candidate = hf.CANDIDATES[0]
        decision = hf.gate(hf.normalize_metadata(candidate, raw(candidate.repo_id)))
        self.assertEqual(decision["effectiveDisposition"], "EVALUATE")
        for key in hf.DENIED_AUTHORITY:
            self.assertIs(decision[key], False)

    def test_license_conflict_does_not_select_permissive_declaration(self):
        data = raw(hf.CANDIDATES[0].repo_id)
        data["cardData"]["license"] = "other"
        decision = hf.gate(hf.normalize_metadata(hf.CANDIDATES[0], data))
        self.assertIn("license_conflict", decision["reasonCodes"])
        self.assertEqual(decision["effectiveDisposition"], "HOLD")

    def test_missing_license_is_held(self):
        data = raw(hf.CANDIDATES[0].repo_id)
        data["tags"], data["cardData"] = [], {}
        self.assertIn("license_unknown", hf.gate(hf.normalize_metadata(hf.CANDIDATES[0], data))["reasonCodes"])

    def test_forged_license_admitted_flag_cannot_override_license(self):
        data = hf.normalize_metadata(hf.CANDIDATES[0], raw(hf.CANDIDATES[0].repo_id))
        data.update(license="other", licenseDeclarations=["other"], licenseAdmitted=True)
        self.assertEqual(hf.gate(data)["effectiveDisposition"], "HOLD")

    def test_conflicting_identity_fields_are_rejected(self):
        data = raw(hf.CANDIDATES[0].repo_id)
        data["id"] = "different/repo"
        with self.assertRaises(hf.FrontierError):
            hf.normalize_metadata(hf.CANDIDATES[0], data)

    def test_repository_id_path_confusion_is_rejected(self):
        for name in ("../model", "owner/..", "owner/../model", "owner/a%2Fb", "https://hf.co/a/b", "owner/model.git", "owner/a?x=1", "owner/a#x", "owner/--model", "owner/a\\b"):
            with self.subTest(name=name), self.assertRaises(hf.FrontierError):
                hf.Candidate(name, "test", hf.Disposition.WATCH, "test")

    def test_candidate_types_cannot_coerce_authority(self):
        for kwargs in ({"disposition": "HOLD"}, {"production_eligible": 0}, {"category": 2}):
            fields = dict(repo_id="owner/model", category="test", disposition=hf.Disposition.WATCH, rationale="test")
            fields.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(hf.FrontierError):
                hf.Candidate(**fields)

    def test_remote_code_detected_without_custom_code_tag(self):
        for field, value in (("config", {"auto_map": {"AutoModel": "model.Model"}}),
                             ("transformersInfo", {"auto_map": {"AutoModel": "model.Model"}}),
                             ("siblings", [{"rfilename": "m2r/model.py"}])):
            data = raw(hf.CANDIDATES[0].repo_id)
            data[field] = value
            decision = hf.gate(hf.normalize_metadata(hf.CANDIDATES[0], data))
            self.assertIn("remote_code_signal", decision["reasonCodes"])
            self.assertEqual(decision["effectiveDisposition"], "HOLD")

    def test_private_or_disabled_repository_held(self):
        for field in ("private", "disabled"):
            data = raw(hf.CANDIDATES[0].repo_id)
            data[field] = True
            self.assertIn(field + "_repository", hf.gate(hf.normalize_metadata(hf.CANDIDATES[0], data))["reasonCodes"])

    def test_malformed_metadata_fails_closed(self):
        for field, value in (("tags", "license:mit"), ("tags", [0]), ("cardData", []),
                             ("config", "safe"), ("siblings", {}), ("siblings", ["model.py"]),
                             ("private", 0), ("pipeline_tag", {}), ("sha", "main")):
            data = raw(hf.CANDIDATES[0].repo_id)
            data[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(hf.FrontierError):
                hf.normalize_metadata(hf.CANDIDATES[0], data)

    def test_strict_json_rejects_duplicates_nonfinite_and_invalid_unicode(self):
        for body in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                     b'[]', b'null', b'\xff', b'{"x":"\\ud800"}', b'{"x":'):
            with self.subTest(body=body), self.assertRaises(hf.FrontierError):
                hf.strict_json(body)

    def test_json_size_is_bounded(self):
        with self.assertRaises(hf.FrontierError):
            hf.strict_json(b" " * (hf.MAX_JSON_BYTES + 1))
        self.assertEqual(hf.strict_json('{"word":"café"}'), {"word": "café"})

    def test_timestamp_requires_timezone(self):
        for value in ("", "invalid", "2026-09-09", "2026-09-09T12:00:00"):
            with self.subTest(value=value), self.assertRaises(hf.FrontierError):
                hf.build_receipt(hf.CANDIDATES[0], raw(hf.CANDIDATES[0].repo_id), observed_at=value)

    def test_receipt_does_not_mutate_input(self):
        data = raw(hf.CANDIDATES[0].repo_id)
        original = copy.deepcopy(data)
        receipt = hf.build_receipt(hf.CANDIDATES[0], data, observed_at=AT)
        self.assertEqual(data, original)
        self.assertEqual(receipt["receiptSha256"], hf.sha256({k: v for k, v in receipt.items() if k != "receiptSha256"}))

    def test_registry_empty_duplicate_and_unbounded_rejected(self):
        for candidates in ([], [hf.CANDIDATES[0]] * 2, itertools.repeat(hf.CANDIDATES[0])):
            with self.assertRaises(hf.FrontierError):
                hf.refresh(candidates=candidates, client=Client())

    def test_manifest_is_deterministic_for_same_observation(self):
        self.assertEqual(manifest(), manifest())
        hf.verify_manifest(manifest(), expected_source_revision=SOURCE)

    def test_semantic_digest_ignores_scan_time_and_download_count(self):
        first = manifest()
        client = Client()
        client.fetch = lambda repo_id: {**raw(repo_id), "downloads": 999}
        second = hf.refresh(client=client, observed_at="2026-09-09T13:00:00+00:00", source_revision=SOURCE)
        self.assertEqual(first["semanticSha256"], second["semanticSha256"])
        self.assertNotEqual(first["manifestSha256"], second["manifestSha256"])

    def test_registry_iteration_order_does_not_change_result(self):
        self.assertEqual(manifest(), hf.refresh(candidates=reversed(hf.CANDIDATES), client=Client(), observed_at=AT, source_revision=SOURCE))

    def test_one_upstream_failure_preserves_others_and_redacts_error(self):
        client = Client()
        def fetch(repo_id):
            if repo_id == hf.CANDIDATES[0].repo_id:
                raise hf.FrontierError("sensitive upstream response")
            return raw(repo_id)
        client.fetch = fetch
        value = hf.refresh(client=client, observed_at=AT, source_revision=SOURCE)
        self.assertEqual(value["scanStatus"], "PARTIAL")
        self.assertEqual(len(value["receipts"]), 2)
        self.assertNotIn("sensitive", json.dumps(value))
        hf.verify_manifest(value)

    def test_all_upstreams_unavailable_not_reported_complete(self):
        client = MagicMock()
        client.fetch.side_effect = hf.FrontierError("offline")
        value = hf.refresh(client=client, observed_at=AT, source_revision=SOURCE)
        self.assertEqual(value["scanStatus"], "UNAVAILABLE")
        self.assertEqual(len(value["errors"]), 3)
        hf.verify_manifest(value)

    def test_programmer_errors_are_not_silently_swallowed(self):
        client = MagicMock()
        client.fetch.side_effect = RuntimeError("bug")
        with self.assertRaises(RuntimeError):
            hf.refresh(client=client)

    def test_tampered_receipt_or_manifest_rejected(self):
        for target in ("manifest", "receipt"):
            value = manifest()
            if target == "manifest":
                value["productionDisposition"] = "LIVE"
            else:
                value["receipts"][0]["candidate"]["license"] = "other"
            with self.subTest(target=target), self.assertRaises(hf.FrontierError):
                hf.verify_manifest(value)

    def test_rehashed_authority_escalation_still_rejected(self):
        for key in hf.DENIED_AUTHORITY:
            for location in ("manifest", "gate"):
                value = manifest()
                node = value if location == "manifest" else value["receipts"][0]["gate"]
                node[key] = True
                rehash(value)
                with self.subTest(key=key, location=location), self.assertRaises(hf.FrontierError):
                    hf.verify_manifest(value)

    def test_rehashed_extra_authority_keys_rejected(self):
        for location in ("manifest", "receipt", "candidate"):
            value = manifest()
            node = value if location == "manifest" else value["receipts"][0]
            if location == "candidate":
                node = node["candidate"]
            node["executeAnything"] = True
            rehash(value)
            with self.subTest(location=location), self.assertRaises(hf.FrontierError):
                hf.verify_manifest(value)

    def test_rehashed_malformed_normalized_fields_rejected(self):
        for key, item in (("gated", 0), ("private", 0), ("productionEligible", 0),
                          ("licenseAdmitted", 1), ("remoteCodeSignal", 0),
                          ("codeSignals", ["invented_signal"]), ("tags", "apache-2.0")):
            value = manifest()
            value["receipts"][0]["candidate"][key] = item
            rehash(value)
            with self.subTest(key=key), self.assertRaises(hf.FrontierError):
                hf.verify_manifest(value)

    def test_rehashed_coverage_and_registry_changes_rejected(self):
        for mutate in (lambda value: value["receipts"].pop(),
                       lambda value: value["receipts"].append(copy.deepcopy(value["receipts"][0])),
                       lambda value: value["registry"][0].update(requestedDisposition="WATCH"),
                       lambda value: value.update(candidateCount=True),
                       lambda value: value.update(observedAt=None)):
            value = manifest()
            mutate(value)
            rehash(value)
            with self.assertRaises(hf.FrontierError):
                hf.verify_manifest(value)

    def test_source_revision_mismatch_rejected(self):
        with self.assertRaises(hf.FrontierError):
            hf.verify_manifest(manifest(), expected_source_revision="2" * 40)

    def test_projection_is_source_bound_non_serving_and_non_mutating(self):
        value = manifest()
        original = copy.deepcopy(value)
        projection = hf.project_manifest(value, now=AT)
        self.assertEqual(value, original)
        self.assertEqual(projection["modelEvaluation"], "NOT_PERFORMED")
        self.assertEqual(projection["runtimeVerification"], "NOT_PERFORMED")
        self.assertEqual(projection["sourceRevision"], SOURCE)
        self.assertEqual(projection["projectionSha256"], hf.sha256({k: v for k, v in projection.items() if k != "projectionSha256"}))
        for key in hf.DENIED_AUTHORITY:
            self.assertIs(projection[key], False)

    def test_projection_rejects_unbound_stale_and_future_evidence(self):
        value = hf.refresh(client=Client(), observed_at=AT)
        with self.assertRaises(hf.FrontierError):
            hf.project_manifest(value, now=AT)
        for now in ("2026-09-11T12:00:00Z", "2026-09-09T11:58:00Z"):
            with self.subTest(now=now), self.assertRaises(hf.FrontierError):
                hf.project_manifest(manifest(), now=now)

    def test_projection_cannot_weaken_freshness_budget(self):
        for budget in (True, 0, -1, 86401, float("inf")):
            with self.subTest(budget=budget), self.assertRaises(hf.FrontierError):
                hf.project_manifest(manifest(), now=AT, max_age_seconds=budget)

    def test_concurrent_atomic_writers_leave_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.json"
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: hf.write_manifest(destination, {"sequence": i}), range(20)))
            self.assertIn(hf.strict_json(destination.read_bytes())["sequence"], range(20))
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_failed_replacement_retains_previous_evidence_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.json"
            hf.write_manifest(destination, {"previous": True})
            with patch.object(hf.os, "replace", side_effect=OSError("read-only")):
                with self.assertRaises(OSError):
                    hf.write_manifest(destination, {"next": True})
            self.assertEqual(hf.strict_json(destination.read_bytes()), {"previous": True})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_timeout_and_header_validation(self):
        for timeout in (True, 0, -1, 61, float("nan"), float("inf"), "15"):
            with self.subTest(timeout=timeout), self.assertRaises(hf.FrontierError):
                hf.HuggingFaceMetadataClient(timeout_seconds=timeout)
        with self.assertRaises(hf.FrontierError):
            hf.HuggingFaceMetadataClient(user_agent="agent\r\nAuthorization: secret")

    def response(self, data=None, content_type="application/json", url=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.geturl.return_value = url or hf.HF_API + hf.CANDIDATES[0].repo_id
        response.headers = Message()
        response.headers["Content-Type"] = content_type
        response.read.return_value = json.dumps(data or raw(hf.CANDIDATES[0].repo_id)).encode()
        return response

    def test_client_get_is_unauthenticated_and_bounded(self):
        client = hf.HuggingFaceMetadataClient()
        response = self.response()
        with patch.object(client._opener, "open", return_value=response) as opened:
            self.assertEqual(client.fetch(hf.CANDIDATES[0].repo_id), raw(hf.CANDIDATES[0].repo_id))
        request = opened.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.get_header("Authorization"))
        response.read.assert_called_once_with(hf.MAX_JSON_BYTES + 1)

    def test_client_rejects_redirects_and_wrong_content_type(self):
        with self.assertRaises(hf.FrontierError):
            hf._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")
        for response in (self.response(content_type="text/html"), self.response(url="https://example.com")):
            client = hf.HuggingFaceMetadataClient()
            with patch.object(client._opener, "open", return_value=response), self.assertRaises(hf.FrontierError):
                client.fetch(hf.CANDIDATES[0].repo_id)

    def test_client_retries_transients_and_honors_retry_after(self):
        client = hf.HuggingFaceMetadataClient()
        headers = Message()
        headers["Retry-After"] = "3"
        error = urllib.error.HTTPError(hf.HF_API, 429, "limited", headers, None)
        with patch.object(client._opener, "open", side_effect=[error, self.response()]) as opened, patch.object(hf.time, "sleep") as sleep:
            client.fetch(hf.CANDIDATES[0].repo_id)
        self.assertEqual(opened.call_count, 2)
        sleep.assert_called_once_with(3.0)

    def test_client_does_not_retry_auth_or_unbounded_retry_after(self):
        for code, retry in ((401, None), (403, None), (429, "120"), (429, "Wed, 09 Sep 2026 12:00:00 GMT")):
            client = hf.HuggingFaceMetadataClient()
            headers = Message()
            if retry:
                headers["Retry-After"] = retry
            error = urllib.error.HTTPError(hf.HF_API, code, "failure", headers, None)
            with patch.object(client._opener, "open", side_effect=error) as opened, patch.object(hf.time, "sleep") as sleep, self.assertRaises(hf.FrontierError):
                client.fetch(hf.CANDIDATES[0].repo_id)
            self.assertEqual(opened.call_count, 1)
            sleep.assert_not_called()

    def test_client_transport_retry_budget_is_three(self):
        client = hf.HuggingFaceMetadataClient()
        with patch.object(client._opener, "open", side_effect=urllib.error.URLError("offline")) as opened, patch.object(hf.time, "sleep"), self.assertRaises(hf.FrontierError):
            client.fetch(hf.CANDIDATES[0].repo_id)
        self.assertEqual(opened.call_count, 3)

    def test_cli_plan_is_offline(self):
        with patch.object(hf.HuggingFaceMetadataClient, "fetch", side_effect=AssertionError("network")), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(hf.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())["networkAction"], "metadata-only")

    def test_cli_partial_scan_writes_evidence_and_exits_two(self):
        value = manifest()
        missing = value["receipts"].pop(0)
        value["errors"].append({"repoId": missing["candidate"]["repoId"], "reasonCode": "metadata_unavailable_or_invalid"})
        value["scanStatus"] = "PARTIAL"
        rehash(value)
        with tempfile.TemporaryDirectory() as directory, patch.object(hf, "refresh", return_value=value), patch("sys.stdout", new_callable=io.StringIO):
            output = Path(directory) / "partial.json"
            self.assertEqual(hf.main(["--refresh", "--output", str(output)]), 2)
            hf.verify_manifest(hf.strict_json(output.read_bytes()))

    def test_cli_verify_and_tampered_input_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            output = Path(directory) / "manifest.json"
            hf.write_manifest(output, manifest())
            self.assertEqual(hf.main(["--verify", str(output), "--source-revision", SOURCE]), 0)
            self.assertEqual(hf.main(["--verify", str(output), "--source-revision", "2" * 40]), 1)
            output.write_text('{"schema":"forged"}')
            self.assertEqual(hf.main(["--verify", str(output)]), 1)

    def test_cli_projection_cannot_overwrite_source_manifest(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            hf.main(["--refresh", "--output", "same.json", "--projection-output", "same.json"])


if __name__ == "__main__":
    unittest.main()
