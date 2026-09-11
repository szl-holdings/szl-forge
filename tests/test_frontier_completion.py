"""Synthetic/offline unit tests. None are model benchmarks or live-site witnesses."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from frontier_completion import core, metrics, closure, scan
from frontier_completion.__main__ import main
from frontier_completion.registry import CANDIDATES

SOURCE = "a" * 40
NOW = "2026-09-11T16:00:00+00:00"
C = scan.Candidate("test-model", "owner/model", "model", "testing")


def metadata():
    return {"id": C.repo, "sha": "b" * 40, "gated": False, "private": False, "disabled": False,
            "tags": ["license:apache-2.0"], "cardData": {"license": "apache-2.0"},
            "config": {"architectures": ["FixtureModel"]}, "library_name": "transformers",
            "siblings": [{"rfilename": "README.md", "blobId": "c" * 40, "size": 10},
                         {"rfilename": "model.safetensors", "lfs": {"sha256": "d" * 64, "size": 123}}]}


def plan():
    return {"schema": "szl.evaluation-plan.v1", "runId": "fixture", "candidateId": "owner/model",
            "sourceRevision": SOURCE, "modelRevision": "b" * 40, "datasetSha256": "c" * 64,
            "runtimeImageDigest": "sha256:" + "d" * 64, "maxWallSeconds": 60, "maxCostUSD": 0,
            "maxDownloadBytes": 0, "maxInputTokens": 32, "maxOutputTokens": 16,
            "requestedGPUCount": 0, "approvedBudgetUSD": 0, "approvalReceiptId": "fixture-only-untrusted",
            "privateData": False, "rightsReviewed": True}


class CoreTests(unittest.TestCase):
    def test_canonical_key_order(self):
        self.assertEqual(core.sha256({"a": 1, "b": 2}), core.sha256({"b": 2, "a": 1}))

    def test_strict_json_bad_inputs(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'null', b'\xff', b'{"x":"\\ud800"}'):
            with self.subTest(raw=raw), self.assertRaises(core.EvidenceError):
                core.parse_json(raw)

    def test_json_size_bound(self):
        with self.assertRaises(core.EvidenceError):
            core.parse_json(b" " * (core.MAX_JSON + 1))

    def test_finite_rejects_bool_nan_infinity(self):
        for value in (True, "1", None, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(core.EvidenceError):
                core.finite(value)

    def test_digest_requires_lowercase_immutable_hash(self):
        for value in ("main", "A" * 40, "a" * 39, 0):
            with self.subTest(value=value), self.assertRaises(core.EvidenceError):
                core.digest(value, 40)

    def test_timezone_required(self):
        with self.assertRaises(core.EvidenceError):
            core.timestamp("2026-09-11T16:00:00")
        self.assertEqual(core.age_seconds(NOW, now=NOW), 0)

    def test_receipt_tamper_and_external_pin(self):
        value = core.receipt({"x": 1})
        core.verify_receipt(value, value["receiptSha256"])
        with self.assertRaises(core.EvidenceError):
            core.verify_receipt(value, "0" * 64)
        value["x"] = 2
        with self.assertRaises(core.EvidenceError):
            core.verify_receipt(value)

    def test_receipt_copy_not_shared(self):
        data = {"x": [1]}
        sealed = core.receipt(data)
        data["x"].append(2)
        self.assertEqual(sealed["x"], [1])

    def test_atomic_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x.json"
            value = core.receipt({"x": 1})
            core.write_json(path, value)
            self.assertEqual(core.load_json(path), value)
            self.assertEqual(core.file_sha256(path, 10000), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_failed_atomic_replace_retains_old_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x.json"
            core.write_json(path, {"old": True})
            with patch.object(core.os, "replace", side_effect=OSError("fixture")), self.assertRaises(OSError):
                core.write_json(path, {"new": True})
            self.assertEqual(core.load_json(path), {"old": True})
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_streaming_hash_bound(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x"
            path.write_bytes(b"abcd")
            with self.assertRaises(core.EvidenceError):
                core.file_sha256(path, 3)


class MetricTests(unittest.TestCase):
    def test_forecast_known_values(self):
        value = metrics.forecast_metrics([3, 5], [2, 5], [0, 1, 2])
        self.assertEqual(value["mae"], .5)
        self.assertEqual(value["mase"], .5)
        self.assertEqual(value["wape"], .125)

    def test_zero_scale_is_undefined_not_zero_error(self):
        value = metrics.forecast_metrics([0, 0], [1, 1], [4, 4, 4])
        self.assertIsNone(value["mase"])
        self.assertIsNone(value["wape"])
        self.assertEqual(value["mae"], 1)

    def test_forecast_length_and_nonfinite(self):
        for actual, forecast in (([1], [1, 2]), ([math.nan], [1]), ([True], [1])):
            with self.assertRaises(core.EvidenceError):
                metrics.forecast_metrics(actual, forecast, [0, 1])

    def test_quantile_coverage_is_not_calibration(self):
        value = metrics.quantile_metrics([2, 9], [.1, .5, .9], [[1, 2, 3], [7, 8, 10]])
        self.assertEqual(value["intervalCoverage"], 1)
        self.assertFalse(value["calibrationEstablished"])
        self.assertIsNone(value["exactCRPS"])

    def test_crossing_and_duplicate_quantiles_rejected(self):
        for qs, rows in (([.9, .1], [[1, 2]]), ([.1, .1], [[1, 2]]), ([.1, .9], [[2, 1]])):
            with self.assertRaises(core.EvidenceError):
                metrics.quantile_metrics([1], qs, rows)

    def test_rolling_origins_have_no_target_leakage(self):
        rows = metrics.rolling_origins(20, 10, 4, 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["trainStop"] == row["testStart"] and row["testStop"] <= 20 for row in rows))

    def test_no_full_fold_returns_empty(self):
        self.assertEqual(metrics.rolling_origins(10, 8, 4, 1), [])

    def test_wer_known_edits(self):
        value = metrics.speech_error([("one two", "one three")])
        self.assertEqual(value["edits"], 1)
        self.assertEqual(value["value"], .5)

    def test_cer_is_explicitly_codepoints(self):
        value = metrics.speech_error([("ab", "ac")], character_level=True)
        self.assertEqual(value["metric"], "codepoint_CER")
        self.assertEqual(value["value"], .5)

    def test_empty_reference_stays_undefined(self):
        value = metrics.speech_error([("", "hallucinated")])
        self.assertEqual(value["edits"], 1)
        self.assertIsNone(value["value"])

    def test_edit_distance_work_bound(self):
        with self.assertRaises(core.EvidenceError):
            metrics.edit_distance(["a"] * 2001, ["b"] * 2001)

    def test_ranking_perfect_and_absent_qrels(self):
        result = metrics.ranking_metrics(["a", "b"], {"a": 2, "b": 1}, 2)
        self.assertEqual(result["ndcg"], 1)
        self.assertEqual(result["recall"], 1)
        self.assertIsNone(metrics.ranking_metrics(["a"], {}, 1)["ndcg"])

    def test_duplicate_ranked_ids_rejected(self):
        with self.assertRaises(core.EvidenceError):
            metrics.ranking_metrics(["a", "a"], {"a": 1}, 2)

    def test_equal_totals_not_parity(self):
        value = metrics.paired_cases([{"id": "a", "pass": True}, {"id": "b", "pass": False}],
                                     [{"id": "a", "pass": False}, {"id": "b", "pass": True}])
        self.assertEqual(value["baselinePassed"], value["candidatePassed"])
        self.assertEqual(value["regressedIds"], ["a"])
        self.assertFalse(value["sameCaseOutcomes"])
        self.assertFalse(value["productionPromotion"])

    def test_missing_case_cannot_improve_denominator(self):
        with self.assertRaises(core.EvidenceError):
            metrics.paired_cases([{"id": "a", "pass": True}], [{"id": "b", "pass": True}])

    def test_unknown_case_result_rejected(self):
        with self.assertRaises(core.EvidenceError):
            metrics.paired_cases([{"id": "a", "pass": None}], [{"id": "a", "pass": True}])

    def test_kl_exact_and_infinite(self):
        self.assertEqual(metrics.kl_divergence([.5, .5], [.5, .5])["value"], 0)
        self.assertEqual(metrics.kl_divergence([1, 0], [0, 1])["state"], "INFINITE_DIVERGENCE")

    def test_kl_requires_normalized_probabilities(self):
        with self.assertRaises(core.EvidenceError):
            metrics.kl_divergence([1, 1], [.5, .5])


class MaterialityTests(unittest.TestCase):
    def test_registry_unique(self):
        self.assertEqual(len({row.key for row in CANDIDATES}), len(CANDIDATES))

    def test_model_fact_extraction(self):
        result = scan.materiality(C, metadata())
        self.assertTrue(result["materialIdentityComplete"])
        self.assertEqual(result["effectiveDisposition"], "HOLD")
        self.assertFalse(result["downloadedModelBytes"])

    def test_revision_popularity_readme_do_not_notify(self):
        old = metadata(); new = copy.deepcopy(old)
        new.update(sha="e" * 40, downloads=10**9, likes=1000, lastModified=NOW)
        new["siblings"][0]["blobId"] = "f" * 40
        result = scan.compare_observations(scan.materiality(C, old), scan.materiality(C, new))
        self.assertFalse(result["notify"])

    def test_weight_change_is_material(self):
        old = metadata(); new = copy.deepcopy(old)
        new["siblings"][1]["lfs"]["sha256"] = "e" * 64
        result = scan.compare_observations(scan.materiality(C, old), scan.materiality(C, new))
        self.assertTrue(result["notify"])

    def test_license_change_is_material(self):
        old = metadata(); new = copy.deepcopy(old)
        new["cardData"]["license"] = "other"
        self.assertTrue(scan.compare_observations(scan.materiality(C, old), scan.materiality(C, new))["notify"])

    def test_license_document_bytes_are_material(self):
        old = metadata(); old["siblings"].append({"rfilename": "LICENSE.md", "size": 5, "blobId": "a" * 40})
        new = copy.deepcopy(old); new["siblings"][-1]["blobId"] = "f" * 40
        self.assertTrue(scan.compare_observations(scan.materiality(C, old), scan.materiality(C, new))["notify"])

    def test_missing_file_identity_is_not_silence_claim(self):
        old = scan.materiality(C, metadata())
        raw = metadata(); raw["siblings"][1].pop("lfs")
        new = scan.materiality(C, raw)
        self.assertFalse(new["materialIdentityComplete"])
        self.assertEqual(scan.compare_observations(old, new)["state"], "INDETERMINATE")

    def test_repo_identity_disagreement(self):
        raw = metadata(); raw["modelId"] = "other/model"
        with self.assertRaises(core.EvidenceError):
            scan.materiality(C, raw)

    def test_duplicate_and_traversal_files_rejected(self):
        for name in ("../config.json", "a/../config.json", "/config.json", "a\\config.json", "model.safetensors"):
            raw = metadata(); raw["siblings"].append({"rfilename": name, "size": 1, "blobId": "a" * 40})
            with self.subTest(name=name), self.assertRaises(core.EvidenceError):
                scan.materiality(C, raw)

    def test_hold_and_reject_remain_absorbing(self):
        for state in ("HOLD", "REJECT", "WATCH"):
            candidate = scan.Candidate(C.key, C.repo, "model", C.lane, state)
            self.assertEqual(scan.materiality(candidate, metadata())["effectiveDisposition"], state)

    def test_pinned_revision_lookup(self):
        client = MagicMock(); client.json.return_value = metadata()
        scan.observe(C, client)
        self.assertIn("/revision/" + "b" * 40, client.json.call_args.args[0])

    def test_pinned_revision_mismatch_is_blocked(self):
        second = metadata(); second["sha"] = "c" * 40
        client = MagicMock(); client.json.side_effect = [metadata(), second]
        with self.assertRaises(core.EvidenceError):
            scan.observe(C, client)

    def test_partial_failure_records_failure_without_baseline_alert(self):
        client = MagicMock(); client.json.side_effect = core.EvidenceError("http_403")
        report = scan.scan([C], SOURCE, client)
        self.assertEqual(report["scanStatus"], "UNAVAILABLE")
        self.assertEqual(report["notifications"], [])
        self.assertFalse(report["completeMaterialityCoverage"])

    def test_first_scan_is_baseline_not_new_release(self):
        client = MagicMock(); client.json.return_value = metadata()
        report = scan.scan([C], SOURCE, client)
        self.assertEqual(report["notifications"], [])
        self.assertTrue(report["completeMaterialityCoverage"])

    def test_unknown_hosts_and_redirects_are_blocked(self):
        client = scan.PublicClient()
        for url in ("http://huggingface.co/api/models/x", "https://example.com", "https://token@huggingface.co/x"):
            with self.assertRaises(core.EvidenceError):
                client.get(url)
        with self.assertRaises(core.EvidenceError):
            scan.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other")

    def test_release_annotated_tag_resolves_to_commit(self):
        client = MagicMock(); client.json.side_effect = [
            {"tag_name": "v1.13.0", "draft": False, "prerelease": False},
            {"ref": "refs/tags/v1.13.0", "object": {"sha": "a" * 40, "type": "tag"}},
            {"object": {"sha": "b" * 40, "type": "commit"}}]
        value = scan.observe_release("huggingface/trl", "v1.13.0", "b" * 40, client)
        self.assertEqual(value["commit"], "b" * 40)
        self.assertEqual(len(value["gitObjectChain"]), 2)

    def test_release_wrong_commit_is_not_silently_updated(self):
        client = MagicMock(); client.json.side_effect = [
            {"tag_name": "v1.13.0", "draft": False, "prerelease": False},
            {"ref": "refs/tags/v1.13.0", "object": {"sha": "b" * 40, "type": "commit"}}]
        with self.assertRaises(core.EvidenceError):
            scan.observe_release("huggingface/trl", "v1.13.0", "c" * 40, client)

    def test_plan_cli_is_offline(self):
        with patch.object(scan.PublicClient, "get", side_effect=AssertionError("network")), patch("sys.stdout", new_callable=io.StringIO) as stream:
            self.assertEqual(main(["plan"]), 0)
            self.assertEqual(json.loads(stream.getvalue())["execution"], "NONE")


class GateTests(unittest.TestCase):
    def test_plan_never_self_authorizes_launch(self):
        report = closure.plan_preflight(plan())
        self.assertTrue(report["structureValid"])
        self.assertFalse(report["schedulerMayLaunch"])
        self.assertFalse(report["approvalAuthenticityVerified"])

    def test_private_data_budget_and_mutable_pins_rejected(self):
        for key, value in (("privateData", True), ("maxCostUSD", 1), ("sourceRevision", "main"),
                           ("rightsReviewed", False), ("maxWallSeconds", True), ("requestedGPUCount", 65)):
            p = plan(); p[key] = value
            with self.subTest(key=key), self.assertRaises(core.EvidenceError):
                closure.plan_preflight(p)

    def stages(self):
        return [{"stage": stage, "state": "PASS", "observedAt": NOW,
                 "productSourceRevision": SOURCE, "recipeSha256": "b" * 64,
                 "evidenceSha256": "c" * 64} for stage in closure.STAGES]

    def close(self, rows):
        return closure.summarize_closure(rows, expected_product_source=SOURCE,
                                         expected_recipe_sha256="b" * 64, now=NOW)

    def test_reported_closure_is_not_authentication_or_promotion(self):
        result = self.close(self.stages())
        self.assertEqual(result["reportedStageClosure"], "COMPLETE")
        self.assertFalse(result["outerEvidenceAuthenticityVerified"])
        self.assertEqual(result["productionDisposition"], "HOLD")

    def test_one_failed_stage_never_becomes_green(self):
        rows = self.stages(); rows[-1]["state"] = "FAIL"
        self.assertEqual(self.close(rows)["reportedStageClosure"], "PARTIAL")

    def test_missing_stage_stays_unavailable(self):
        self.assertEqual(self.close(self.stages()[:-1])["stages"]["proof"]["state"], "UNAVAILABLE")

    def test_stale_and_wrong_source_remain_separate(self):
        rows = self.stages(); rows[0]["productSourceRevision"] = "f" * 40
        rows[1]["observedAt"] = "2026-09-09T00:00:00Z"
        result = self.close(rows)
        self.assertEqual(result["stages"]["source"]["state"], "FAIL")
        self.assertEqual(result["stages"]["artifact"]["state"], "UNAVAILABLE")

    def test_duplicate_stage_rejected(self):
        with self.assertRaises(core.EvidenceError):
            self.close([self.stages()[0]] * 2)

    def quant(self):
        fields = {"modelRevision", "tokenizerSha256", "templateSha256", "calibrationSha256", "holdoutSha256",
                  "runnerSha256", "hardwareProfileSha256", "tensorSchemaSha256", "generatorSha256", "weightSha256"}
        row = {key: "a" * (40 if key == "modelRevision" else 64) for key in fields}
        row.update(holdoutSha256="b" * 64, bytes=100, kl=.1, caseRegressions=0)
        return row

    def test_quant_canary_keeps_incumbent_on_regression(self):
        old = self.quant(); new = copy.deepcopy(old); new["caseRegressions"] = 1
        report = closure.quantization_canary(old, new, max_absolute_kl_increase=.001)
        self.assertEqual(report["selection"], "KEEP_INCUMBENT")

    def test_quant_canary_no_promotion_on_pass(self):
        report = closure.quantization_canary(self.quant(), self.quant(), max_absolute_kl_increase=0)
        self.assertTrue(report["localRulePassed"])
        self.assertFalse(report["productionPromotion"])

    def test_uncontrolled_quant_comparison_rejected(self):
        old = self.quant(); new = copy.deepcopy(old); new["bytes"] = 99
        with self.assertRaises(core.EvidenceError):
            closure.quantization_canary(old, new, max_absolute_kl_increase=0)

    def test_calibration_is_not_holdout(self):
        row = self.quant(); row["holdoutSha256"] = row["calibrationSha256"]
        with self.assertRaises(core.EvidenceError):
            closure.quantization_canary(row, row, max_absolute_kl_increase=0)


class MalformedMetadataTests(unittest.TestCase):
    def test_unhashable_lfs_identity_is_controlled_rejection(self):
        raw = metadata()
        raw["siblings"][1]["lfs"]["sha256"] = {"not": "a hash"}
        with self.assertRaises(core.EvidenceError):
            scan.materiality(C, raw)

    def test_falsey_malformed_card_and_config_are_not_missing(self):
        for field in ("cardData", "config"):
            for value in (0, False, [], ""):
                raw = metadata(); raw[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(core.EvidenceError):
                    scan.materiality(C, raw)

    def test_falsey_malformed_license_rejected(self):
        for value in (0, False, {}):
            raw = metadata(); raw["cardData"]["license"] = value
            with self.subTest(value=value), self.assertRaises(core.EvidenceError):
                scan.materiality(C, raw)

    def test_nonstring_candidate_key_rejected(self):
        with self.assertRaises(core.EvidenceError):
            scan.Candidate(42, "owner/model", "model", "testing")

    def test_invalid_candidate_registry_rejected_before_network(self):
        client = MagicMock()
        with self.assertRaises(core.EvidenceError):
            scan.scan([None], SOURCE, client)
        client.json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
