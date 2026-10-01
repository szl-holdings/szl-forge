# SPDX-License-Identifier: Apache-2.0
"""Regression for path-stat/fstat ctime disagreement; no owner model fixtures."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gguf_forensics as m


def tiny_gguf():
    """Independent one-tensor Qwen2 GGUF fixture, assembled from the format."""
    def text(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
    header += text('general.architecture') + struct.pack('<I', 8) + text('qwen2')
    header += text('weight') + struct.pack('<IQIQ', 1, 2, 1, 0)
    return header + b'\x00' * ((-len(header)) % 32) + struct.pack('<2e', 1, 2)


def info(**changes):
    values = dict(st_dev=5, st_ino=123, st_mode=stat.S_IFREG | 0o600,
                  st_size=len(tiny_gguf()), st_mtime_ns=300,
                  st_birthtime_ns=100, st_ctime_ns=100)
    values.update(changes)
    return SimpleNamespace(**values)


class FileIdentityTests(unittest.TestCase):
    def test_old_cross_api_predicate_reproduces_false_change(self):
        path, fd = info(), info(st_ctime_ns=200)
        self.assertNotEqual(m.snapshot(path), m.snapshot(fd))
        with patch.object(m.os, 'name', 'nt'):
            m.require_stat_match(path, fd, 'INPUT_CHANGED_AT_OPEN', same_api=False)

    def test_different_birthtime_or_identity_size_mtime_still_rejected(self):
        for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_birthtime_ns'):
            with self.subTest(field=field), patch.object(m.os, 'name', 'nt'):
                with self.assertRaisesRegex(m.ForensicsError, 'INPUT_CHANGED_AT_OPEN:' + field):
                    m.require_stat_match(info(), info(**{field: 777}),
                                         'INPUT_CHANGED_AT_OPEN', same_api=False)

    def test_ctime_change_on_descriptor_still_rejected(self):
        with patch.object(m.os, 'name', 'nt'), self.assertRaisesRegex(m.ForensicsError, 'DURING_READ:st_ctime_ns'):
            m.require_stat_match(info(st_ctime_ns=200), info(st_ctime_ns=201),
                                 'INPUT_CHANGED_DURING_READ', same_api=True)

    def test_ctime_change_on_path_still_rejected(self):
        with patch.object(m.os, 'name', 'nt'), self.assertRaisesRegex(m.ForensicsError, 'AFTER_READ:st_ctime_ns'):
            m.require_stat_match(info(), info(st_ctime_ns=201),
                                 'INPUT_CHANGED_AFTER_READ', same_api=True)

    def test_posix_cross_api_ctime_still_rejected(self):
        with patch.object(m.os, 'name', 'posix'), self.assertRaisesRegex(m.ForensicsError, 'AT_OPEN:st_ctime_ns'):
            m.require_stat_match(info(), info(st_ctime_ns=201), 'INPUT_CHANGED_AT_OPEN', same_api=False)

    def test_birthtime_missing_is_not_a_fallback(self):
        item = info(); del item.st_birthtime_ns
        with patch.object(m.os, 'name', 'nt'), self.assertRaisesRegex(m.ForensicsError, 'UNAVAILABLE:st_birthtime_ns'):
            m.require_stat_match(item, item, 'INPUT_CHANGED_AT_OPEN', same_api=False)

    def test_missing_or_coerced_fields_and_zero_file_id_refused(self):
        for field, value in (('st_ino', 0), ('st_dev', None), ('st_size', True), ('st_mtime_ns', 1.0)):
            with self.subTest(field=field), patch.object(m.os, 'name', 'nt'):
                item = info(**{field: value})
                with self.assertRaises(m.ForensicsError):
                    m.require_stat_match(item, item, 'INPUT_CHANGED_AT_OPEN', same_api=False)

    def test_non_regular_descriptor_refused(self):
        with self.assertRaisesRegex(m.ForensicsError, 'REGULAR_FILE_REQUIRED'):
            m.require_stat_match(info(), info(st_mode=stat.S_IFDIR), 'INPUT_CHANGED_AT_OPEN', same_api=False)

    def simulated_inspect(self, *, fd_after=None, path_after=None, expected=None):
        """Real read-only fixture descriptor with independently supplied stat channels."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.gguf'; raw = tiny_gguf(); path.write_bytes(raw)
            first_path = info(); first_fd = info(st_ctime_ns=200)
            with patch.object(m.os, 'name', 'nt'), patch.object(m, 'plain_path', return_value=True), \
                 patch.object(type(path), 'stat', side_effect=[first_path, path_after or first_path]), \
                 patch.object(m.os, 'fstat', side_effect=[first_fd, fd_after or first_fd]):
                result = m.inspect_file(path, expected or m.digest(raw))
            self.assertEqual(path.read_bytes(), raw)
            return result

    def test_full_read_uses_separate_path_and_descriptor_baselines(self):
        result = self.simulated_inspect()
        self.assertTrue(result['matches_historical_from_digest'])
        self.assertTrue(result['file_identity_observation']['windows_cross_api_ctime_difference'])
        self.assertFalse(result['tensor_values_or_health_checked'])

    def test_full_read_descriptor_change_fails(self):
        with self.assertRaisesRegex(m.ForensicsError, 'DURING_READ:st_ctime_ns'):
            self.simulated_inspect(fd_after=info(st_ctime_ns=201))

    def test_full_read_path_replacement_fails(self):
        with self.assertRaisesRegex(m.ForensicsError, 'AFTER_READ:st_ino'):
            self.simulated_inspect(path_after=info(st_ino=999))

    def test_full_read_hash_still_required(self):
        with self.assertRaisesRegex(m.ForensicsError, 'BLOB_DIGEST_MISMATCH'):
            self.simulated_inspect(expected='0' * 64)

    def test_native_fixture_identity_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.gguf'; raw = tiny_gguf(); path.write_bytes(raw)
            result = m.inspect_file(path, m.digest(raw))
            self.assertTrue(result['matches_historical_from_digest'])
            self.assertEqual(path.read_bytes(), raw)

    @unittest.skipUnless(os.name == 'nt', 'Native Windows API regression')
    def test_native_windows_distinct_creation_and_change_time(self):
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel = ctypes.WinDLL('kernel32.dll', use_last_error=True, winmode=0x800)
        set_time = kernel.SetFileTime
        set_time.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                             ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
        set_time.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.gguf'; raw = tiny_gguf(); path.write_bytes(raw)
            # On THIS disposable fixture only, make creation time much older than
            # change time. The production inspector has no SetFileTime calls.
            ticks = 132223104000000000  # 2020-01-01 UTC, 100ns since 1601.
            creation = wintypes.FILETIME(ticks & 0xffffffff, ticks >> 32)
            with path.open('r+b') as stream:
                handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
                if not set_time(handle, ctypes.byref(creation), None, None):
                    raise ctypes.WinError(ctypes.get_last_error())
            by_path = path.stat()
            with path.open('rb') as stream:
                by_fd = os.fstat(stream.fileno())
            mismatch = by_path.st_ctime_ns != by_fd.st_ctime_ns
            if sys.version_info[:3] == (3, 12, 10):
                self.assertTrue(mismatch, '3.12.10 fixture must reproduce the original false mismatch')
                with self.assertRaises(AssertionError):
                    self.assertEqual(m.snapshot(by_path), m.snapshot(by_fd))
            result = m.inspect_file(path, m.digest(raw))
            self.assertTrue(result['matches_historical_from_digest'])
            self.assertEqual(result['file_identity_observation']['windows_cross_api_ctime_difference'], mismatch)
            print(json.dumps({'native_windows_ctime_mismatch_reproduced': mismatch,
                              'fixed_inspector_fixture_hash_verified': True}), flush=True)

    def test_console_lists_per_model_errors_without_raw_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp); root = home / '.ollama' / 'models' / 'blobs'; root.mkdir(parents=True)
            with patch.object(m.os, 'name', 'nt'), patch.dict(os.environ, {'COMPUTERNAME': 'BETTERWITHAGE'}), \
                 patch.object(m, 'Path') as path_factory, \
                 patch.object(m, 'require_local_windows_path'), \
                 patch.object(m, 'inspect_file', side_effect=m.ForensicsError('INPUT_CHANGED_AT_OPEN:st_ino')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                path_factory.side_effect = type(home)
                path_factory.home.return_value = home
                self.assertEqual(m.main(['--inspect-blobs']), 1)
            for model in m.BLOBS:
                self.assertIn(model, output.getvalue())
            self.assertIn('INPUT_CHANGED_AT_OPEN:st_ino', output.getvalue())
            report = json.loads(next(home.glob('szl-gguf-forensics-*/gguf-forensics.json')).read_text())
            self.assertIsNone(report['comparison'])
            self.assertFalse(report['model_weights_changed'])


if __name__ == '__main__':
    unittest.main()
