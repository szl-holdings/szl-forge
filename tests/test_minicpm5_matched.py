"""Same-worker comparison regressions. All generated evidence is OFFLINE_FIXTURE."""
import copy
import hashlib
import unittest
from unittest.mock import patch

from inference import minicpm5_gguf as g
from inference import minicpm5_matched as m

REV = "1" * 40
HOST = {"system": "Linux", "machine": "x86_64", "python": "3.12", "cpuCount": 8}


def seal(value):
    value.pop("recordSha256", None)
    value["recordSha256"] = g.base.digest(value)
    return value


def fixtures():
    reports = []
    source = seal({"modelId": g.MODEL, "revision": g.REVISION, "files": g.plan()["files"],
        "license": "apache-2.0", "evidenceClass": "METADATA_ONLY", "weightsDownloaded": False,
        "productionDisposition": "HOLD"})
    for variant in m.ORDER:
        rows = []
        for i, case in enumerate(g.base.suite()):
            rows.append({"id": case["id"], "category": case["category"], "passed": True,
                "reasonCodes": [], "outputSha256": hashlib.sha256(g.base.canonical(case["expected"])).hexdigest(),
                "roundtripMs": i + 1.0, "generatedTokens": 20, "inputTokens": 200,
                "reasoningProduced": False})
        reports.append(g.summarize({"schema": "szl.forge.minicpm5-gguf-result.v1", "plan": g.plan(),
            "variant": variant, "sourceRepository": "szl-holdings/szl-forge", "sourceRevision": REV,
            "runnerSha256": m.GGUF_RUNNER_SHA, "modelSha256": g.FILES[variant][2],
            "runtimeArchiveSha256": g.RUNTIME["sha256"], "serverBinarySha256": "a" * 64,
            "phase": "complete", "artifactBytesVerified": True, "cases": rows,
            "sourceObservation": copy.deepcopy(source), "peakResidentBytes": 10000,
            "host": dict(HOST), "jobId": "offline-fixture-job"}, g.baseline()))
    worker = dict(HOST, jobId="offline-fixture-job", workerHash="b" * 64)
    context = {"evidenceClass": "OFFLINE_FIXTURE", "workerObservations": [dict(worker), dict(worker)],
        "comparisonId": "c" * 64, "protocolSha256": g.base.digest([g.request_body(c) for c in g.base.suite()])}
    return reports, context


class MatchedTests(unittest.TestCase):
    def test_fixture_cannot_claim_live_execution_or_production(self):
        rows, context = fixtures()
        result = m.summarize_pair(rows, context, REV)
        self.assertEqual(result["status"], "OFFLINE_FIXTURE")
        self.assertFalse(result["matchedExecutionObserved"])
        for key, value in m.BOUNDARIES.items():
            self.assertEqual(result[key], value)
        self.assertFalse(result["quantizationIsolationVerified"])
        self.assertIsNone(result["speedupClaim"])
        m.validate_pair(result)

    def test_plan_is_bounded_and_never_automatic(self):
        value = m.plan()
        self.assertEqual(value["variants"], ["F16", "Q4_K_M"])
        self.assertEqual(value["maximumEvaluations"], 2)
        self.assertFalse(value["automaticRetry"])
        self.assertFalse(value["automaticExecution"])
        self.assertLessEqual(value["maxWallSeconds"], 900)

    def test_equal_totals_are_not_verdict_parity(self):
        rows, context = fixtures()
        for row, index in zip(rows, (0, 6)):
            row["cases"][index].update(passed=False, reasonCodes=["decision_or_grounding_mismatch"], outputSha256="e" * 64)
            row.update(g.summarize(row, g.baseline()))
        result = m.summarize_pair(rows, context, REV)
        self.assertEqual(result["f16PassedCases"], result["q4PassedCases"])
        self.assertFalse(result["verdictParity"])
        self.assertFalse(result["outputHashParity"])
        self.assertEqual(result["q4RegressedCaseIds"], ["probe_06"])
        self.assertEqual(result["q4RecoveredCaseIds"], ["probe_00"])

    def test_same_verdicts_different_outputs_are_not_hash_parity(self):
        rows, context = fixtures()
        rows[1]["cases"][0]["outputSha256"] = "f" * 64
        rows[1] = g.summarize(rows[1], g.baseline())
        result = m.summarize_pair(rows, context, REV)
        self.assertTrue(result["verdictParity"])
        self.assertFalse(result["outputHashParity"])

    def test_different_worker_or_job_is_rejected(self):
        for key in ("workerHash", "jobId", "cpuCount", "python"):
            with self.subTest(key=key):
                rows, context = fixtures()
                context["workerObservations"][1][key] = "changed"
                with self.assertRaises(m.MatchedError):
                    m.summarize_pair(rows, context, REV)

    def test_variant_cannot_substitute_a_different_job(self):
        rows, context = fixtures()
        rows[1]["jobId"] = "other"
        seal(rows[1])
        with self.assertRaises(m.MatchedError):
            m.summarize_pair(rows, context, REV)

    def test_runtime_source_artifact_flags_cannot_drift(self):
        for key, bad in (("modelSha256", "a" * 64), ("sourceRevision", "2" * 40),
                         ("runnerSha256", "b" * 64), ("artifactBytesVerified", False),
                         ("productionDisposition", "PROMOTE"), ("sealed", True),
                         ("modelOperational", 0), ("phase", "startup")):
            with self.subTest(key=key):
                rows, context = fixtures()
                rows[1][key] = bad
                seal(rows[1])
                with self.assertRaises(m.MatchedError):
                    m.summarize_pair(rows, context, REV)

    def test_native_binary_and_host_must_match(self):
        for key, bad in (("serverBinarySha256", "e" * 64), ("host", dict(HOST, cpuCount=4))):
            with self.subTest(key=key):
                rows, context = fixtures()
                rows[1][key] = bad
                seal(rows[1])
                with self.assertRaises(m.MatchedError):
                    m.summarize_pair(rows, context, REV)

    def test_incomplete_cases_are_rejected(self):
        rows, context = fixtures()
        rows[1]["cases"].pop()
        rows[1] = g.summarize(rows[1], g.baseline())
        with self.assertRaises(m.MatchedError):
            m.summarize_pair(rows, context, REV)

    def test_wrong_order_and_duplicate_variant_rejected(self):
        for indices in ((1, 0), (0, 0)):
            with self.subTest(indices=indices):
                rows, context = fixtures()
                with self.assertRaises(m.MatchedError):
                    m.summarize_pair([rows[i] for i in indices], context, REV)

    def test_protocol_and_digest_tamper_rejected(self):
        rows, context = fixtures()
        context["protocolSha256"] = "e" * 64
        with self.assertRaises(m.MatchedError):
            m.summarize_pair(rows, context, REV)
        rows, context = fixtures()
        rows[1]["passedCases"] = 0
        with self.assertRaises(m.MatchedError):
            m.summarize_pair(rows, context, REV)

    def test_forged_summary_still_rejected_after_resealing(self):
        rows, context = fixtures()
        rows[1]["p50RoundtripMs"] = 900
        seal(rows[1])
        with self.assertRaises(m.MatchedError):
            m.summarize_pair(rows, context, REV)

    def test_nested_source_digest_cannot_be_swapped(self):
        rows, context = fixtures()
        rows[1]["sourceObservation"]["revision"] = "9" * 40
        seal(rows[1]["sourceObservation"])
        seal(rows[1])
        with self.assertRaises(m.MatchedError):
            m.summarize_pair(rows, context, REV)

    def test_pair_claims_recomputed_after_resealing(self):
        rows, context = fixtures()
        record = m.summarize_pair(rows, context, REV)
        record["quantizationIsolationVerified"] = True
        seal(record)
        with self.assertRaises(m.MatchedError):
            m.validate_pair(record)

    def test_preflight_failure_replaces_stale_success_without_network(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text('{"status":"EXECUTED"}')
            with patch.object(m.subprocess, "run") as child, contextlib_redirect():
                self.assertEqual(m.run("not-a-sha", output), 1)
                child.assert_not_called()
            self.assertEqual(g.loads(output.read_bytes())["status"], "INCOMPLETE")


def contextlib_redirect():
    import contextlib
    import io
    return contextlib.redirect_stdout(io.StringIO())


if __name__ == "__main__":
    unittest.main()
