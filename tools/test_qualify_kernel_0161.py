"""Offline control tests only: no download, installation, or kernel runtime claim."""
from __future__ import annotations

from pathlib import Path
import base64
import hashlib
import json
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
        # A summary flag alone cannot substitute for retained validated controls.
        with self.assertRaises(q.QualificationError):
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


class ProviderTreeTests(unittest.TestCase):
    def file(self, path="build/torch-cpu/_ops.py", size=7):
        from huggingface_hub import RepoFile
        return RepoFile(path=path, size=size, oid="a" * 40)

    def folder(self, path="build"):
        from huggingface_hub import RepoFolder
        return RepoFolder(path=path, oid="b" * 40)

    def api(self, entries):
        return SimpleNamespace(list_repo_tree=Mock(return_value=iter(entries)))

    def test_real_sdk_tree_types_and_pinned_call(self):
        api = self.api([self.folder(), self.file()])
        self.assertEqual(q.provider_files(api), {"build/torch-cpu/_ops.py": 7})
        api.list_repo_tree.assert_called_once_with(
            q.HUB_REPOSITORY, repo_type="kernel", revision=q.HUB_REVISION,
            recursive=True, token=False)

    def test_malformed_tree_records_rejected(self):
        for entries in ([{}], [SimpleNamespace(path="x", size=5)],
                        [self.file(size=True)], [self.file(size=-1)],
                        [self.file(path="../x")], [self.file(path="a//b")],
                        [self.file(path="a/b\\c")]):
            with self.subTest(entries=entries), self.assertRaises(q.QualificationError):
                q.provider_files(self.api(entries))

    def test_duplicate_file_or_folder_is_not_silently_collapsed(self):
        for entries in ([self.file(), self.file()], [self.folder(), self.folder()],
                        [self.file(path="build"), self.folder()]):
            with self.subTest(entries=entries), self.assertRaisesRegex(q.QualificationError, "duplicate"):
                q.provider_files(self.api(entries))

    def test_empty_or_folder_only_tree_is_not_missing_everything(self):
        for entries in ([], [self.folder()]):
            with self.assertRaisesRegex(q.QualificationError, "empty"):
                q.provider_files(self.api(entries))

    def test_entry_limit_stops_iteration(self):
        with self.assertRaisesRegex(q.QualificationError, "bound"):
            q.provider_files(self.api(self.file(path=f"file{i}") for i in range(257)))

    def test_partial_iteration_error_is_not_a_complete_tree(self):
        def interrupted():
            yield self.file()
            raise OSError("fixture interrupted page")
        with self.assertRaises(OSError):
            q.provider_files(self.api(interrupted()))

    def test_kernel_info_without_siblings_reaches_real_tree_method(self):
        # Actual SDK RepoFile objects; API and installed-version responses are fixtures.
        # No downloaded/executed kernel is claimed by this unit test.
        api = self.api([self.file(path="README.md")])
        api.repo_info = Mock(return_value=SimpleNamespace(sha=q.HUB_REVISION))
        self.assertFalse(hasattr(api.repo_info.return_value, "siblings"))
        source = {name: b"fixture" for name in q.SOURCE_BLOBS}
        with patch.dict("os.environ", {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}, clear=True), \
             patch.object(q.importlib.metadata, "version", return_value=q.CLIENT_VERSION), \
             patch.object(q, "read_source", return_value=source), \
             patch.dict("sys.modules", {"kernels": ModuleType("kernels")}), \
             patch("huggingface_hub.HfApi", return_value=api), \
             patch("huggingface_hub.hf_hub_download") as download:
            report = q.observe("provider")
        self.assertEqual(report["state"], "SOURCE_PROVIDER_DRIFT")
        self.assertIs(report["runtime_qualified"], False)
        self.assertEqual(len(report["missing_files"]), 8)
        download.assert_not_called()
        api.list_repo_tree.assert_called_once()


class ProviderControlTests(unittest.TestCase):
    def fixture(self):
        source = {name: b"fixture" for name in q.SOURCE_BLOBS}
        binding = {"schema": "szl.hf-first-class-kernel-binding/v1",
                   "source_repository": q.SOURCE_REPOSITORY, "source_revision": q.SOURCE_REVISION,
                   "artifact": {"repo_id": q.HUB_REPOSITORY, "repo_type": "kernel",
                                "backend": "torch-cpu", "package": "szl_kernels", "version": 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q.stage_local(root, source)
            metadata = q.strict_json((root / q.PREFIX / "metadata.json").read_bytes())
        raw_binding = json.dumps(binding).encode()
        metadata["digest"]["files"]["source-binding.json"] = base64.b64encode(hashlib.sha256(raw_binding).digest()).decode()
        controls = {"metadata.json": json.dumps(metadata).encode(), "source-binding.json": raw_binding}
        return source, controls, metadata, binding

    def test_complete_control_bytes_and_source_binding(self):
        source, controls, _, _ = self.fixture()
        q.validate_control_bytes(source, controls)

    def test_metadata_cannot_redirect_execution_or_dependencies(self):
        for changes in ({"name": "another"}, {"version": True}, {"python-depends": ["os"]},
                        {"backend": {"type": "cuda"}}, {"id": "../escape"}, {"extra": "field"}):
            source, controls, metadata, _ = self.fixture()
            controls["metadata.json"] = json.dumps(metadata | changes).encode()
            with self.subTest(changes=changes), self.assertRaises(q.QualificationError):
                q.validate_control_bytes(source, controls)

    def test_digest_manifest_must_cover_only_verified_bytes(self):
        source, controls, metadata, _ = self.fixture()
        metadata["digest"]["files"]["../extra.py"] = "a" * 44
        controls["metadata.json"] = json.dumps(metadata).encode()
        with self.assertRaisesRegex(q.QualificationError, "digest"):
            q.validate_control_bytes(source, controls)

    def test_source_binding_must_match_even_with_recomputed_digest(self):
        for changes in ({"source_revision": "a" * 40}, {"source_repository": "OTHER/repo"},
                        {"artifact": {"repo_type": "model"}}):
            source, controls, metadata, binding = self.fixture()
            controls["source-binding.json"] = json.dumps(binding | changes).encode()
            metadata["digest"]["files"]["source-binding.json"] = base64.b64encode(
                hashlib.sha256(controls["source-binding.json"]).digest()).decode()
            controls["metadata.json"] = json.dumps(metadata).encode()
            with self.subTest(changes=changes), self.assertRaises(q.QualificationError):
                q.validate_control_bytes(source, controls)

    def test_host_revalidates_controls_for_trust_blocker(self):
        source, controls, _, _ = self.fixture()
        report = blocked_report() | {"state": "PUBLISHER_UNTRUSTED",
                    "source_sha256": {key: q.sha256(value) for key, value in source.items()},
                    "byte_comparison": {"all_equal": True},
                    "provider_control_bytes": {key: base64.b64encode(value).decode() for key, value in controls.items()}}
        q.validate_report(report, "provider", source)
        report["provider_control_bytes"]["metadata.json"] = "invalid=base64!"
        with self.assertRaises(q.QualificationError):
            q.validate_report(report, "provider", source)


if __name__ == "__main__":
    unittest.main()
