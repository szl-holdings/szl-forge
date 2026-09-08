"""Offline synthetic contracts; no model download or inference occurs in tests."""
import copy
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference import minicpm5_gguf as g


def inventory():
    return {"id": g.MODEL, "sha": g.REVISION, "private": False, "gated": False,
        "cardData": {"license": "apache-2.0"}, "siblings": [
            {"rfilename": name, "size": size, "lfs": {"sha256": sha, "size": size}}
            for name, size, sha in g.FILES.values()]}


def response(case):
    return {"model": g.ALIAS, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": g.base.canonical(case["expected"]).decode()}}],
        "usage": {"completion_tokens": 20, "prompt_tokens": 200}}


def cases():
    rows = []
    for case in g.base.suite():
        row = g.response_grade(response(case), case)
        row["roundtripMs"] = 100.0
        rows.append(row)
    return rows


class InventoryTests(unittest.TestCase):
    def test_exact_pins_have_one_source_record_not_model_qualification(self):
        record = g.validate_inventory(inventory())
        self.assertEqual(record["revision"], g.REVISION)
        self.assertEqual(record["evidenceClass"], "METADATA_ONLY")
        self.assertFalse(record["weightsDownloaded"])
        self.assertEqual(record["productionDisposition"], "HOLD")

    def test_same_size_changed_weight_is_rejected(self):
        data = inventory()
        data["siblings"][0]["lfs"]["sha256"] = "a" * 64
        with self.assertRaises(g.GGUFError):
            g.validate_inventory(data)

    def test_wrong_size_is_rejected(self):
        data = inventory()
        data["siblings"][0]["size"] += 1
        with self.assertRaises(g.GGUFError):
            g.validate_inventory(data)

    def test_missing_variant_cannot_claim_complete_coverage(self):
        data = inventory()
        data["siblings"].pop()
        with self.assertRaises(g.GGUFError):
            g.validate_inventory(data)

    def test_duplicate_variant_is_rejected(self):
        data = inventory()
        data["siblings"].append(copy.deepcopy(data["siblings"][0]))
        with self.assertRaises(g.GGUFError):
            g.validate_inventory(data)

    def test_unknown_private_license_or_revision_fails(self):
        for fields in [{"private": True}, {"gated": "auto"}, {"sha": "main"},
                       {"id": "other/repo"}, {"cardData": {"license": "other"}}]:
            with self.subTest(fields=fields):
                data = inventory()
                data.update(fields)
                with self.assertRaises(g.GGUFError):
                    g.validate_inventory(data)

    def test_readme_artwork_and_popularity_do_not_change_file_identity(self):
        data = inventory()
        data.update(likes=9000, lastModified="2099-01-01")
        data["siblings"].append({"rfilename": "README.md", "blobId": "a" * 40})
        self.assertEqual(g.validate_inventory(data)["files"], g.plan()["files"])

    def test_preflight_cannot_invent_bf16_execution_or_conversion_proof(self):
        plan = g.plan()
        self.assertEqual(plan["baselineExecutionDtype"], "float16")
        self.assertFalse(plan["conversionLineageVerified"])
        self.assertFalse(plan["matchedQuantizationExperiment"])
        self.assertFalse(plan["runtimeQualified"])
        self.assertEqual(plan["runtime"]["sha256"], g.RUNTIME["sha256"])


class BoundaryTests(unittest.TestCase):
    def test_foreign_url_or_credentials_or_http_are_refused(self):
        for url in ["http://huggingface.co/x", "https://huggingface.co.evil.org/x",
                    "https://u:p@huggingface.co/x", "https://127.0.0.1/x", "https://github.com:444/x"]:
            with self.subTest(url=url), self.assertRaises(g.GGUFError):
                g.remote_url(url)

    def test_official_download_origins_are_allowed(self):
        for url in ["https://huggingface.co/x", "https://github.com/x", "https://release-assets.githubusercontent.com/x", "https://cas-bridge.xethub.hf.co/x"]:
            g.remote_url(url)

    def test_redirect_is_checked_before_network(self):
        with self.assertRaises(g.GGUFError):
            g.Redirects().redirect_request(None, None, 302, "", None, "https://evil.org")
        with self.assertRaises(g.GGUFError):
            g.NoRedirect().redirect_request(None, None, 302, "", None, "http://127.0.0.1")

    def test_json_rejects_duplicates_nonfinite_and_oversized(self):
        for raw in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b" " * (g.MAX_JSON + 1)]:
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                g.loads(raw)

    def test_archive_escape_or_device_is_rejected(self):
        for name, type_ in [("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE), ("device", tarfile.CHRTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                archive = Path(folder) / "runtime.tar.gz"
                with tarfile.open(archive, "w:gz") as bundle:
                    item = tarfile.TarInfo(name)
                    item.type = type_
                    bundle.addfile(item, io.BytesIO())
                with patch.object(g.base, "file_digest", return_value=g.RUNTIME["sha256"]):
                    with self.assertRaises(g.GGUFError):
                        g.unpack(archive, Path(folder) / "extracted")

    def test_model_download_rejects_unpinned_digest_without_network(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(g, "build_opener") as network:
            with self.assertRaises(g.GGUFError):
                g.download("https://huggingface.co/x", Path(folder) / "x", 1, "wrong")
            network.assert_not_called()

    def test_default_command_does_not_download_or_start_process(self):
        with patch.object(sys, "argv", ["gguf"]), patch.object(g, "download") as download, patch.object(g.subprocess, "Popen") as launch, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(g.main(), 0)
            download.assert_not_called()
            launch.assert_not_called()


class BehaviorTests(unittest.TestCase):
    def test_original_system_case_suite_and_grader_are_reused(self):
        self.assertEqual(g.base.digest(g.base.suite()), g.SUITE_SHA)
        self.assertEqual(g.base.file_digest(Path(g.base.__file__)), g.BASE_RUNNER_SHA)
        body = g.request_body(g.base.suite()[0])
        self.assertEqual(body["messages"][0]["content"], g.base.SYSTEM)
        self.assertEqual(body["max_tokens"], 96)
        self.assertNotIn("response_format", body)
        self.assertNotIn("tools", body)

    def test_expected_proposal_passes_but_never_executes(self):
        case = g.base.suite()[0]
        result = g.response_grade(response(case), case)
        self.assertTrue(result["passed"])
        self.assertNotIn("content", result)
        self.assertNotIn("messages", result)

    def test_historical_overabstention_stays_failure(self):
        case = g.base.suite()[0]
        value = response(case)
        value["choices"][0]["message"]["content"] = '{"decision":"ABSTAIN","evidence_id":null,"value":null}'
        self.assertFalse(g.response_grade(value, case)["passed"])

    def test_duplicate_output_keys_remain_failure(self):
        case = g.base.suite()[0]
        value = response(case)
        value["choices"][0]["message"]["content"] = '{"decision":"LOOKUP","decision":"ABSTAIN"}'
        self.assertIn("invalid_json", g.response_grade(value, case)["reasonCodes"])

    def test_truncated_expected_json_is_not_a_pass(self):
        case = g.base.suite()[0]
        value = response(case)
        value["choices"][0]["finish_reason"] = "length"
        self.assertFalse(g.response_grade(value, case)["passed"])

    def test_identity_tools_or_invalid_usage_fail_closed(self):
        case = g.base.suite()[0]
        for mode in ["model", "tool", "tokens", "empty", "role"]:
            with self.subTest(mode=mode):
                value = response(case)
                if mode == "model":
                    value["model"] = "unknown"
                elif mode == "tool":
                    value["choices"][0]["message"]["tool_calls"] = ["shell"]
                elif mode == "tokens":
                    value["usage"]["completion_tokens"] = True
                elif mode == "role":
                    value["choices"][0]["message"]["role"] = "system"
                else:
                    value["choices"] = []
                with self.assertRaises(g.GGUFError):
                    g.response_grade(value, case)

    def test_full_pass_cannot_erase_baseline_failure_or_promote(self):
        result = g.summarize({"cases": cases()}, g.baseline())
        self.assertEqual(result["status"], "SMOKE_PASS")
        self.assertEqual(result["recoveredCaseIds"], ["probe_00", "probe_04", "probe_10"])
        self.assertEqual(result["productionDisposition"], "HOLD")
        self.assertFalse(result["modelOperational"])
        self.assertIsNone(result["speedupClaim"])
        self.assertEqual(g.baseline()["passedCases"], 9)

    def test_partial_comparison_is_incomplete(self):
        result = g.summarize({"cases": cases()[:-1]}, g.baseline())
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["comparisonComplete"])

    def test_missing_duplicate_or_reordered_case_is_invalid(self):
        for rows in [cases()[1:], list(reversed(cases())), cases() + cases()[:1]]:
            with self.assertRaises(g.GGUFError):
                g.summarize({"cases": rows}, g.baseline())

    def test_nonfinite_bool_latency_or_inconsistent_verdict_is_invalid(self):
        for change in [{"roundtripMs": float("nan")}, {"roundtripMs": True}, {"passed": 1},
                       {"reasonCodes": ["invalid_json"]}, {"category": "made-up"}]:
            with self.subTest(change=change):
                rows = cases()
                rows[0].update(change)
                with self.assertRaises(g.GGUFError):
                    g.summarize({"cases": rows}, g.baseline())

    def test_baseline_mutation_is_rejected(self):
        data = g.baseline()
        data["passedCases"] = 12
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(g.GGUFError):
                g.baseline(path)

    def test_failed_cli_replaces_stale_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "record.json"
            out.write_text('{"status":"SMOKE_PASS"}')
            result = subprocess.run([sys.executable, "-m", "inference.minicpm5_gguf", "run", "--source-revision", "main", "--output", str(out)], capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(out.read_text())["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
