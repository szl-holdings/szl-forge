"""Offline control tests only: no download, installation, or kernel runtime claim."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_kernel_0161 as q


def blocked_report():
    return {"schema": q.SCHEMA, "lane": "provider", "source_repository": q.SOURCE_REPOSITORY,
            "source_revision": q.SOURCE_REVISION, "hub_repository": q.HUB_REPOSITORY,
            "hub_revision": q.HUB_REVISION, "client_version": q.CLIENT_VERSION,
            "installed_packages": {"kernels": q.CLIENT_VERSION}, "production_authorization": False,
            "gpu_qualified": False, "state": "SOURCE_PROVIDER_DRIFT", "runtime_qualified": False,
            "smoke": None, "missing_files": [q.PREFIX + "retrieval.py"]}


class ControlTests(unittest.TestCase):
    def test_source_and_hub_identity_are_distinct(self):
        self.assertNotEqual(q.SOURCE_REVISION, q.HUB_REVISION)
        self.assertEqual(q.CLIENT_VERSION, "0.16.1")
        self.assertEqual(len(q.SOURCE_BLOBS), 4)

    def test_source_bytes_must_match_git_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in q.SOURCE_BLOBS:
                (root / name).write_text("not admitted code", encoding="utf-8")
            with self.assertRaisesRegex(q.QualificationError, "Git blob mismatch"):
                q.read_source(root)

    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.write_bytes(b"x")
            try:
                (root / "_kernel_api.py").symlink_to(real)
            except OSError:
                self.skipTest("symlink creation unavailable on this host")
            with self.assertRaisesRegex(q.QualificationError, "invalid admitted"):
                q.read_source(root)

    def test_source_mapping_preserves_both_loader_layouts(self):
        source = {name: name.encode() for name in q.SOURCE_BLOBS}
        mapped = q.executable_map(source)
        self.assertEqual(len(mapped), 8)
        self.assertEqual(mapped[q.PREFIX + "__init__.py"], source["_kernel_api.py"])
        self.assertEqual(mapped[q.PREFIX + "szl_kernels/__init__.py"], source["_kernel_api.py"])

    def test_partial_source_cannot_be_staged(self):
        with self.assertRaises(q.QualificationError):
            q.executable_map({"_kernel_api.py": b"x"})

    def test_comparison_hashes_actual_bytes(self):
        measured = q.compare_bytes({"a": b"old"}, {"a": b"new"})
        self.assertFalse(measured["all_equal"])
        self.assertEqual(measured["files"][0]["source_sha256"], q.sha256(b"old"))
        self.assertEqual(measured["files"][0]["provider_sha256"], q.sha256(b"new"))

    def test_missing_observations_are_not_equal(self):
        for expected, observed in (({}, {}), ({"a": b"x"}, {}), ({"a": b"x"}, {"b": b"x"})):
            with self.subTest(observed=observed), self.assertRaises(q.QualificationError):
                q.compare_bytes(expected, observed)

    def test_equal_comparison_is_only_byte_equivalence(self):
        measured = q.compare_bytes({"a": b"same"}, {"a": b"same"})
        self.assertTrue(measured["all_equal"])
        self.assertNotIn("runtime_qualified", measured)

    def test_staging_only_copies_known_inputs_and_metadata(self):
        source = {name: b"# fixture, not executed\n" for name in q.SOURCE_BLOBS}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q.stage_local(root, source)
            files = [path for path in root.rglob("*") if path.is_file()]
            self.assertEqual(len(files), 9)
            meta = q.strict_json((root / q.PREFIX / "metadata.json").read_bytes())
            self.assertEqual(meta["backend"], {"type": "cpu"})
            self.assertEqual(meta["python-depends"], [])
            self.assertEqual(len(meta["digest"]["files"]), 8)

    def test_strict_json_rejects_duplicate_nonfinite_and_oversized(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', ' ' * (q.MAX_BYTES + 1)):
            with self.subTest(raw=raw[:40]), self.assertRaises(q.QualificationError):
                q.strict_json(raw)

    def test_completed_drift_observation_is_not_qualification(self):
        report = blocked_report()
        q.validate_report(report, "provider")
        self.assertFalse(report["runtime_qualified"])

    def test_unexplained_drift_rejected(self):
        report = blocked_report()
        report["missing_files"] = []
        with self.assertRaisesRegex(q.QualificationError, "observed differences"):
            q.validate_report(report, "provider")

    def test_no_production_or_gpu_authority(self):
        for field in ("production_authorization", "gpu_qualified"):
            for value in (True, 0, None):
                report = blocked_report()
                report[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(q.QualificationError):
                    q.validate_report(report, "provider")

    def test_blocker_cannot_have_runtime_or_smoke(self):
        for changes in ({"runtime_qualified": True}, {"smoke": {}}, {"runtime_qualified": 0}):
            report = blocked_report() | changes
            with self.assertRaises(q.QualificationError):
                q.validate_report(report, "provider")

    def test_source_cannot_pass_on_provider_drift(self):
        with self.assertRaises(q.QualificationError):
            q.validate_report(blocked_report(), "source")

    def test_wrong_source_or_client_rejected(self):
        for changes in ({"source_revision": "a" * 40}, {"client_version": "0.16.0"},
                        {"installed_packages": {"kernels": "0.16.0"}}):
            with self.subTest(changes=changes), self.assertRaises(q.QualificationError):
                q.validate_report(blocked_report() | changes, "provider")

    def test_host_checks_source_fingerprints(self):
        with self.assertRaisesRegex(q.QualificationError, "source fingerprint"):
            q.validate_report(blocked_report(), "provider", {"a": b"expected"})

    def test_trust_blocker_needs_prior_byte_comparison(self):
        report = blocked_report() | {"state": "PUBLISHER_UNTRUSTED"}
        with self.assertRaises(q.QualificationError):
            q.validate_report(report, "provider")
        report["byte_comparison"] = {"all_equal": True}
        q.validate_report(report, "provider")

    def test_actual_version_is_checked_before_loading(self):
        with patch.dict("os.environ", {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}, clear=True), \
             patch.object(q.importlib.metadata, "version", return_value="0.16.0"), \
             patch.object(q, "read_source") as reader:
            with self.assertRaisesRegex(q.QualificationError, "installed client"):
                q.observe("source")
            reader.assert_not_called()

    def test_credential_or_trust_overrides_fail_before_import(self):
        for name in ("HF_TOKEN", "LOCAL_KERNELS", "HF_HUB_OFFLINE", "HF_ENDPOINT"):
            with patch.dict("os.environ", {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", name: "fixture"}, clear=True):
                with self.subTest(name=name), self.assertRaisesRegex(q.QualificationError, "override"):
                    q.observe("provider")

    def test_implicit_credential_policy_required(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(q.QualificationError, "implicit credentials"):
                q.observe("source")

    def test_real_loader_arguments_do_not_disable_trust(self):
        # Static check supplements, not substitutes for, the hosted actual loader.
        import ast
        tree = ast.parse(Path(q.__file__).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "get_kernel"]
        self.assertEqual(len(calls), 1)
        trust = next(item.value for item in calls[0].keywords if item.arg == "trust_remote_code")
        self.assertIs(trust.value, False)


if __name__ == "__main__":
    unittest.main()
