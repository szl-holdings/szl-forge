# SPDX-License-Identifier: Apache-2.0
"""Offline unit checks for the fail-closed tokenizer precheck boundaries."""
from __future__ import annotations

from pathlib import Path
import os
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_base_tokenizer as probe


class ProbeBaseTokenizerTests(unittest.TestCase):
    def test_only_committed_train_bytes_are_accepted(self):
        rows = probe._read_train(Path(__file__).resolve().parent)
        self.assertEqual(len(rows), 360)
        self.assertTrue(all(row["split"] == "train" for row in rows))

    def test_exact_asset_set_and_bytes_required(self):
        contents = {"tokenizer.json": b"tokenizer", "chat_template.jinja": b"template"}
        digests = {name: probe._sha(raw) for name, raw in contents.items()}
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, "ASSET_SHA256", digests):
            root = Path(directory)
            for name, raw in contents.items():
                (root / name).write_bytes(raw)
            self.assertEqual(probe._read_assets(root), contents)
            (root / "tokenizer.json").write_bytes(b"mutated")
            with self.assertRaisesRegex(probe.ProbeError, "HASH_MISMATCH"):
                probe._read_assets(root)
            (root / "tokenizer.json").write_bytes(contents["tokenizer.json"])
            (root / "unbound.json").write_bytes(b"extra")
            with self.assertRaisesRegex(probe.ProbeError, "SET_MISMATCH"):
                probe._read_assets(root)

    def test_reads_are_bounded_before_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_bytes(b"1234")
            with patch.object(probe, "ASSET_SHA256", {"tokenizer.json": probe._sha(b"1234")}), \
                    patch.object(probe, "MAX_ASSET_BYTES", 3):
                with self.assertRaisesRegex(probe.ProbeError, "TOO_LARGE"):
                    probe._read_assets(root)

    def test_redirected_asset_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "assets"
            root.mkdir()
            secret = parent / "heldout.jsonl"
            secret.write_bytes(b"heldout")
            try:
                (root / "tokenizer.json").symlink_to(secret)
            except OSError:
                self.skipTest("symlink creation unavailable on this host")
            with patch.object(probe, "ASSET_SHA256", {"tokenizer.json": probe._sha(b"heldout")}):
                with self.assertRaisesRegex(probe.ProbeError, "PATH_REDIRECTED"):
                    probe._read_assets(root)

    def test_unc_directory_rejected_without_open(self):
        if sys.platform != "win32":
            self.skipTest("UNC path semantics require Windows")
        with self.assertRaisesRegex(probe.ProbeError, "LOCAL_PATH_REQUIRED"):
            probe._read_assets(Path(r"\\example.invalid\share\tokenizer"))

    def test_reparse_attribute_rejected_before_open(self):
        if not hasattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT"):
            self.skipTest("Windows reparse attribute unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "tokenizer.json"
            asset.write_bytes(b"tokenizer")
            real_lstat = os.lstat

            def marked_lstat(path):
                info = real_lstat(path)
                if Path(path) == asset:
                    return SimpleNamespace(st_mode=info.st_mode, st_dev=info.st_dev,
                                           st_ino=info.st_ino,
                                           st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
                return info

            with patch.object(probe, "ASSET_SHA256", {"tokenizer.json": probe._sha(b"tokenizer")}), \
                    patch.object(probe.os, "lstat", side_effect=marked_lstat):
                with self.assertRaisesRegex(probe.ProbeError, "PATH_REDIRECTED"):
                    probe._read_assets(root)


if __name__ == "__main__":
    unittest.main()
