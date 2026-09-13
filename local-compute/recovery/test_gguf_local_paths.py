# SPDX-License-Identifier: Apache-2.0
"""Windows storage-boundary regressions; no remote drive is contacted or mounted."""
import contextlib
import ctypes
import io
import os
from pathlib import Path, PureWindowsPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gguf_forensics as m


class LocalPathTests(unittest.TestCase):
    def test_fixed_drive_admitted_with_trailing_root_separator(self):
        with patch.object(m, "windows_drive_type", return_value=3) as query:
            m.require_local_windows_path(PureWindowsPath(r"C:\models\blobs"))
        query.assert_called_once_with("C:\\")

    def test_forward_slashes_normalize_without_filesystem_access(self):
        with patch.object(m, "windows_drive_type", return_value=3) as query:
            m.require_local_windows_path("D:/models/blobs")
        query.assert_called_once_with("D:\\")

    def test_remote_unknown_and_nonfixed_drive_types_refused(self):
        for kind in (0, 1, 2, 4, 5, 6, 99):
            with self.subTest(kind=kind), patch.object(m, "windows_drive_type", return_value=kind):
                with self.assertRaisesRegex(m.ForensicsError, "FIXED_LOCAL_DRIVE_REQUIRED"):
                    m.require_local_windows_path(r"Z:\models\blobs")

    def test_unc_device_relative_and_stream_paths_refused_before_native_query(self):
        bad = [r"\\server\share\blobs", "//server/share/blobs",
               r"\\?\C:\models", r"\\.\C:\models", r"\??\C:\models",
               r"C:models", r"\models", "models", "", r"C:\models\..\other",
               r"C:\models:alternate", "C:\\models\x00other"]
        for name in bad:
            with self.subTest(name=name), patch.object(m, "windows_drive_type") as query:
                with self.assertRaisesRegex(m.ForensicsError, "LOCAL_DRIVE_PATH_REQUIRED"):
                    m.require_local_windows_path(name)
                query.assert_not_called()

    def assert_main_refuses_before_filesystem(self, home, root):
        fake_os = SimpleNamespace(name="nt", environ={"COMPUTERNAME": "BETTERWITHAGE"}, fspath=os.fspath)
        factory = Mock(side_effect=PureWindowsPath)
        factory.home.return_value = PureWindowsPath(home)
        def drive_type(anchor):
            return 4 if anchor in ("Z:\\", "H:\\") else 3
        with patch.object(m, "os", fake_os), patch.object(m, "Path", factory), \
             patch.object(m, "windows_drive_type", side_effect=drive_type, create=True), \
             patch.object(m, "plain_path", side_effect=AssertionError("FILESYSTEM_BEFORE_DRIVE_ADMISSION")) as fs, \
             patch.object(m, "inspect_file") as read:
            with self.assertRaisesRegex(m.ForensicsError, "FIXED_LOCAL_DRIVE_REQUIRED"):
                m.main(["--inspect-blobs", "--blob-root", root])
            fs.assert_not_called()
            read.assert_not_called()

    def test_mapped_blob_root_refused_before_any_filesystem_inspection(self):
        self.assert_main_refuses_before_filesystem(r"C:\Users\fixture", r"Z:\models\blobs")

    def test_mapped_report_home_refused_before_any_filesystem_inspection(self):
        self.assert_main_refuses_before_filesystem(r"H:\Users\fixture", r"C:\models\blobs")

    def test_plan_mode_never_queries_drives_or_paths(self):
        with patch.object(m, "windows_drive_type") as query, patch.object(m, "plain_path") as fs, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(["--blob-root", r"Z:\models\blobs"]), 0)
        query.assert_not_called()
        fs.assert_not_called()

    def test_native_api_signature_and_system32_loader(self):
        query = Mock(return_value=3)
        kernel = SimpleNamespace(GetDriveTypeW=query)
        with patch.object(m, "os", SimpleNamespace(name="nt")), \
             patch.object(ctypes, "WinDLL", return_value=kernel, create=True) as loader:
            self.assertEqual(m.windows_drive_type("C:\\"), 3)
        loader.assert_called_once_with("kernel32.dll", use_last_error=True, winmode=0x800)
        self.assertEqual(query.argtypes, [ctypes.c_wchar_p])
        self.assertIs(query.restype, ctypes.c_uint)
        query.assert_called_once_with("C:\\")

    def test_native_api_failure_is_redacted_and_fails_closed(self):
        with patch.object(m, "os", SimpleNamespace(name="nt")), \
             patch.object(ctypes, "WinDLL", side_effect=OSError("PRIVATE_DETAIL"), create=True):
            with self.assertRaisesRegex(m.ForensicsError, "LOCAL_DRIVE_TYPE_UNAVAILABLE") as caught:
                m.windows_drive_type("C:\\")
        self.assertNotIn("PRIVATE_DETAIL", str(caught.exception))

    def test_native_api_cannot_be_used_as_nonwindows_fallback(self):
        with patch.object(m, "os", SimpleNamespace(name="posix")), \
             patch.object(ctypes, "WinDLL", create=True) as loader:
            with self.assertRaisesRegex(m.ForensicsError, "WINDOWS_DRIVE_QUERY_REQUIRED"):
                m.windows_drive_type("C:\\")
        loader.assert_not_called()

    def test_unknown_drive_does_not_become_local_by_exception_handling(self):
        with patch.object(m, "windows_drive_type", side_effect=m.ForensicsError("LOCAL_DRIVE_TYPE_UNAVAILABLE")):
            with self.assertRaisesRegex(m.ForensicsError, "LOCAL_DRIVE_TYPE_UNAVAILABLE"):
                m.require_local_windows_path(r"C:\models")

    @unittest.skipUnless(os.name == "nt", "Native GetDriveTypeW smoke requires Windows")
    def test_native_windows_temp_drive_is_fixed(self):
        # Hosted Windows evidence only; this never implies owner-laptop execution.
        root = PureWindowsPath(tempfile.gettempdir()).anchor
        self.assertEqual(m.windows_drive_type(root), 3)
        m.require_local_windows_path(Path(tempfile.gettempdir()))


if __name__ == "__main__":
    unittest.main()
