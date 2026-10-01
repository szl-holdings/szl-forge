from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import oac_clinical_resources as workspace


class PortableWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oac-portable-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.payloads = {name: (name + "\n").encode() for name in workspace.ASSET_PATHS}
        self.manifest = {"schema": workspace.MANIFEST_SCHEMA, "package_version": workspace.PACKAGE_VERSION,
                         "files": [{"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                   for name, data in self.payloads.items()]}
        for name, data in self.payloads.items():
            path = self.bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.encoded, self.digest = self.write_manifest()

    def write_manifest(self):
        encoded = json.dumps(self.manifest).encode()
        (self.bundle / "manifest.json").write_bytes(encoded)
        return encoded, hashlib.sha256(encoded).hexdigest()

    def prepare(self, directory):
        with patch.object(workspace, "load_verified_assets", return_value=(self.encoded, self.payloads)):
            return workspace.prepare_workspace(directory)

    def test_verified_closed_bundle(self):
        encoded, payloads = workspace._verified_assets(self.bundle, self.digest)
        self.assertEqual(encoded, self.encoded)
        self.assertEqual(payloads, self.payloads)

    def test_new_workspace_only_contains_declared_assets_and_receipt(self):
        target = self.root / "prepared"
        result = self.prepare(target)
        files = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
        self.assertEqual(files, set(workspace.ASSET_PATHS) | {"portable-assets.json"})
        for key in ("clinical_use_authorized", "real_phi_authorized", "site_validated", "device_connection_started"):
            self.assertIs(result[key], False)
        for name, data in self.payloads.items():
            self.assertEqual((target / name).read_bytes(), data)

    def test_existing_directory_or_file_is_never_modified(self):
        for kind in ("directory", "file"):
            with self.subTest(kind=kind):
                target = self.root / kind
                if kind == "directory":
                    target.mkdir()
                    sentinel = target / "keep.txt"
                else:
                    sentinel = target
                sentinel.write_bytes(b"owner data")
                with self.assertRaises(workspace.WorkspacePreparationError):
                    self.prepare(target)
                self.assertEqual(sentinel.read_bytes(), b"owner data")

    def test_parent_must_exist_and_dot_traversal_is_rejected(self):
        for target in (str(self.root / "missing" / "new"), str(self.root) + "/../escaped", str(self.root) + "/./new"):
            with self.subTest(target=target), self.assertRaises(workspace.WorkspacePreparationError):
                self.prepare(target)
        self.assertFalse((self.root / "missing").exists())

    def test_root_is_rejected(self):
        with self.assertRaises(workspace.WorkspacePreparationError):
            self.prepare(self.root.anchor)

    def test_symlink_parent_is_rejected(self):
        link = self.root / "link"
        try:
            link.symlink_to(self.bundle, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaises(workspace.WorkspacePreparationError):
            self.prepare(link / "new")
        self.assertFalse((self.bundle / "new").exists())

    def test_reparse_point_is_rejected_even_when_not_a_symlink(self):
        observed = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        with patch.object(Path, "lstat", return_value=observed):
            with self.assertRaises(workspace.WorkspacePreparationError):
                workspace._reject_link(self.root)

    @unittest.skipUnless(os.name == "nt", "Windows fixed-drive policy")
    def test_mapped_or_removable_drive_is_rejected_before_parent_io(self):
        with patch("owned_agent_clinical_control.windows_fixed_drive", return_value=False):
            with patch.object(Path, "lstat", side_effect=AssertionError("must not probe remote path")):
                with self.assertRaises(workspace.WorkspacePreparationError):
                    self.prepare(self.root / "new")

    def test_bundle_failure_occurs_before_destination_creation(self):
        target = self.root / "never-created"
        with patch.object(workspace, "load_verified_assets", side_effect=workspace.WorkspacePreparationError("bad bundle")):
            with self.assertRaises(workspace.WorkspacePreparationError):
                workspace.prepare_workspace(target)
        self.assertFalse(target.exists())

    def test_each_modified_asset_is_rejected(self):
        for name, original in self.payloads.items():
            with self.subTest(name=name):
                (self.bundle / name).write_bytes(original + b"tampered")
                with self.assertRaises(workspace.WorkspacePreparationError):
                    workspace._verified_assets(self.bundle, self.digest)
                (self.bundle / name).write_bytes(original)

    def test_manifest_and_asset_replacement_does_not_replace_trusted_digest(self):
        self.manifest["files"][0]["sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaises(workspace.WorkspacePreparationError):
            workspace._verified_assets(self.bundle, self.digest)

    def test_missing_extra_duplicate_and_traversal_assets_are_rejected(self):
        for replacement in ("../escape", "site-key.pem", self.manifest["files"][1]["path"]):
            with self.subTest(replacement=replacement):
                self.manifest["files"][0]["path"] = replacement
                _, digest = self.write_manifest()
                with self.assertRaises(workspace.WorkspacePreparationError):
                    workspace._verified_assets(self.bundle, digest)
        self.manifest["files"].pop()
        _, digest = self.write_manifest()
        with self.assertRaises(workspace.WorkspacePreparationError):
            workspace._verified_assets(self.bundle, digest)

    def test_boolean_and_oversize_lengths_are_rejected(self):
        for size in (True, 0, -1, workspace.MAX_ASSET_BYTES + 1):
            with self.subTest(size=size):
                self.manifest["files"][0]["size_bytes"] = size
                _, digest = self.write_manifest()
                with self.assertRaises(workspace.WorkspacePreparationError):
                    workspace._verified_assets(self.bundle, digest)

    def test_partial_failure_retains_only_new_work_and_no_success_receipt(self):
        target = self.root / "partial"
        with patch.object(os, "open", side_effect=OSError("simulated write failure")):
            with self.assertRaises(workspace.WorkspacePreparationError):
                self.prepare(target)
        self.assertTrue(target.is_dir())
        self.assertFalse((target / "portable-assets.json").exists())


if __name__ == "__main__":
    unittest.main()
