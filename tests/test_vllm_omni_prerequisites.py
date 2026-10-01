"""Offline negative contracts; these fixtures never qualify an upstream runtime."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import evaluate_vllm_omni_prerequisites as target


def plan():
    return {"version": "1", "pip_version": target.PIP_VERSION,
            "environment": {"python_version": "3.12", "sys_platform": "linux"},
            "install": [{"is_yanked": False,
                         "metadata": {"name": "openai-whisper", "version": "20250625"},
                         "download_info": {"url": "https://files.pythonhosted.org/packages/test.whl",
                                           "archive_info": {"hashes": {"sha256": "a" * 64}}}}]}


class PrerequisiteTests(unittest.TestCase):
    def test_report_is_metadata_only(self):
        value = target.validate_resolution(plan(), "openai-whisper")
        self.assertEqual(value["evidenceClass"], "RESOLVER_REPORTED_NOT_BYTE_VERIFIED")
        self.assertFalse(value["packages"][0]["independentlyVerified"])

    def test_no_runtime_authority(self):
        self.assertTrue(target.DENIED)
        self.assertTrue(all(x is False for x in target.DENIED.values()))

    def test_json_duplicate(self):
        with self.assertRaises(target.PrerequisiteError):
            target.decode_report(b'{"version":"1","version":"2"}')

    def test_json_size_and_shape(self):
        for body in (b"[]", b" " * (target.MAX_REPORT + 1)):
            with self.subTest(size=len(body)), self.assertRaises(target.PrerequisiteError):
                target.decode_report(body)

    def test_schema_tool_and_environment(self):
        for key, bad in (("version", "2"), ("pip_version", "0"), ("environment", None)):
            with self.subTest(key=key), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution({**plan(), key: bad}, "openai-whisper")

    def test_inventory_bounds(self):
        for bad in ([], None, plan()["install"] * 513):
            with self.subTest(count=0 if bad is None else len(bad)), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution({**plan(), "install": bad}, "openai-whisper")

    def test_root_missing(self):
        with self.assertRaises(target.PrerequisiteError):
            target.validate_resolution(plan(), "vllm-omni")

    def test_duplicate_normalized_name(self):
        data = plan()
        second = copy.deepcopy(data["install"][0])
        second["metadata"]["name"] = "OpenAI_Whisper"
        data["install"].append(second)
        with self.assertRaises(target.PrerequisiteError):
            target.validate_resolution(data, "openai-whisper")

    def test_yanked_and_missing_metadata(self):
        for key, bad in (("is_yanked", True), ("is_yanked", None), ("metadata", None), ("download_info", None)):
            data = plan()
            data["install"][0][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution(data, "openai-whisper")

    def test_sdist_vcs_local_and_credential_urls_rejected(self):
        for url in ("https://files.pythonhosted.org/p.tar.gz", "file:///tmp/test.whl",
                    "https://evil.invalid/test.whl", "http://files.pythonhosted.org/test.whl",
                    "https://x:y@files.pythonhosted.org/test.whl",
                    "https://files.pythonhosted.org:444/test.whl",
                    "https://files.pythonhosted.org/test.whl?token=no",
                    "https://files.pythonhosted.org/test.whl#x"):
            data = plan()
            data["install"][0]["download_info"]["url"] = url
            with self.subTest(url=url), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution(data, "openai-whisper")

    def test_missing_or_invalid_hash(self):
        for value in (None, {}, {"hashes": {}}, {"hashes": {"sha256": "A" * 64}},
                      {"hashes": {"sha256": "a" * 63}}):
            data = plan()
            data["install"][0]["download_info"]["archive_info"] = value
            with self.subTest(value=value), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution(data, "openai-whisper")

    def test_fixed_command_does_not_disable_dependencies_or_build(self):
        reference = "https://files.pythonhosted.org/fixed.whl#sha256=" + "b" * 64
        for root in (reference, target.PREREQUISITE):
            argv = target.command(root, Path("/tmp/report.json"), reference)
            self.assertIn("--dry-run", argv)
            self.assertIn("--only-binary=:all:", argv)
            self.assertIn("--ignore-installed", argv)
            self.assertIn("--isolated", argv)
            self.assertNotIn("--no-deps", argv)
            self.assertNotIn("--extra-index-url", argv)
            self.assertEqual(argv[-1], root)

    def test_unapproved_input_rejected(self):
        with self.assertRaises(target.PrerequisiteError):
            target.command("--help", Path("/tmp/output.json"), "fixed")

    def test_no_inherited_secret_proxy_or_pip_environment(self):
        with patch.dict(os.environ, {"GH_TOKEN": "test-only", "HTTPS_PROXY": "test-only",
                                     "PIP_EXTRA_INDEX_URL": "test-only", "PYTHONPATH": "test-only"}):
            result = target.child_environment(Path("/tmp/owned"))
        for key in ("GH_TOKEN", "HTTPS_PROXY", "PIP_EXTRA_INDEX_URL", "PYTHONPATH"):
            self.assertNotIn(key, result)
        self.assertEqual(result["PIP_CONFIG_FILE"], os.devnull)

    def test_nonzero_is_not_a_pass_or_a_specific_root_cause(self):
        self.assertEqual(target.classify({"returnCode": 1, "stoppedBy": None}, False), "BLOCKED_OR_UNAVAILABLE")

    def test_timeout_is_incomplete(self):
        self.assertEqual(target.classify({"returnCode": -9, "stoppedBy": "deadline"}, False), "INCOMPLETE")

    def test_success_requires_report(self):
        with self.assertRaises(target.PrerequisiteError):
            target.classify({"returnCode": 0, "stoppedBy": None}, False)
        self.assertEqual(target.classify({"returnCode": 0, "stoppedBy": None}, True), "RESOLVER_COMPLETED")

    def test_invalid_exit_code(self):
        for bad in (True, None, "0"):
            with self.subTest(bad=bad), self.assertRaises(target.PrerequisiteError):
                target.classify({"returnCode": bad}, False)

    def test_source_failure_does_not_start_subprocess_or_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(target, "source_identity", side_effect=target.PrerequisiteError("dirty source")):
                with patch.object(target, "execute") as execute:
                    result = target.run(Path(temporary))
        execute.assert_not_called()
        self.assertEqual(result["state"], "INCOMPLETE")
        self.assertEqual(result["phases"], [])
        for key in target.DENIED:
            self.assertIs(result[key], False)

    def test_wrong_whisper_version_rejected(self):
        for version in ("20200101", "future", "20250625.dev1"):
            value = plan()
            value["install"][0]["metadata"]["version"] = version
            with self.subTest(version=version), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution(value, "openai-whisper")

    def test_omni_root_identity_is_exact(self):
        value = plan()
        value["install"][0]["metadata"] = {"name": "vllm-omni", "version": "0.29.0rc1"}
        identity = {"version": "0.29.0rc1", "url": value["install"][0]["download_info"]["url"],
                    "reportedSha256": "a" * 64}
        target.validate_resolution(value, "vllm-omni", identity)
        for key in identity:
            altered = {**identity, key: "changed"}
            with self.subTest(key=key), self.assertRaises(target.PrerequisiteError):
                target.validate_resolution(value, "vllm-omni", altered)
        with self.assertRaises(target.PrerequisiteError):
            target.validate_resolution(value, "vllm-omni")

    @unittest.skipUnless(os.name == "posix", "Linux-owned process-group execution only")
    def test_actual_bounded_child_exit_and_timeout(self):
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = target.execute([sys.executable, "-c", "print('fixture-only')"], root, root / "ok.log")
            self.assertEqual(value["returnCode"], 0)
            self.assertEqual(value["logSha256"], target.digest(b"fixture-only\n"))
            with patch.object(target, "TIMEOUT", 0.01):
                value = target.execute([sys.executable, "-c", "import time; time.sleep(30)"], root, root / "timeout.log")
            self.assertEqual(value["stoppedBy"], "deadline")
            self.assertNotEqual(value["returnCode"], 0)

    def test_fixture_round_trip(self):
        data = plan()
        self.assertEqual(target.decode_report(json.dumps(data).encode()), data)


if __name__ == "__main__":
    unittest.main()
