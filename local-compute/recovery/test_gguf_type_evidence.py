# SPDX-License-Identifier: Apache-2.0
"""Offline diagnostic regressions. Numeric fixture types are not owner evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import gguf_forensics as m


def fixture(kinds=(1,), *, name='weight', misleading_file_type=False):
    """Assemble GGUF bytes independently; opaque unsupported payload is not parsed."""
    def text(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b
    metadata = text('general.architecture') + struct.pack('<I', 8) + text('qwen2')
    if misleading_file_type:
        metadata += text('general.file_type') + struct.pack('<II', 4, 1)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(kinds), 2 if misleading_file_type else 1) + metadata
    offsets = []
    for i, kind in enumerate(kinds):
        descriptor = text(name + str(i)) + struct.pack('<IQ', 1, 2)
        offsets.append(len(header) + len(descriptor))
        header += descriptor + struct.pack('<IQ', kind, i * 32)
    header += b'\x00' * ((-len(header)) % 32)
    # Each tensor occupies two values; spacing exists solely for fixture layout.
    body = b''.join((struct.pack('<2f', 1, 2) if kind == 0 else b'\x80\x3f\x00\x40').ljust(32, b'\0')
                    for kind in kinds)
    return header + body, offsets


class TypeEvidenceTests(unittest.TestCase):
    def rejected(self, kind=30, **kw):
        raw, offsets = fixture((kind,), **kw)
        with self.assertRaises(m.UnsupportedTensorType) as raised:
            m.fingerprint_stream(io.BytesIO(raw), len(raw))
        return raw, offsets, raised.exception

    def test_exact_numeric_type_is_preserved(self):
        for kind in (2, 12, 24, 30, 31, 42, 0xffffffff):
            with self.subTest(kind=kind):
                _, offsets, error = self.rejected(kind)
                self.assertEqual(str(error), f'UNSUPPORTED_TENSOR_TYPE:ggml_type_id={kind}')
                self.assertEqual(error.observation['ggml_type_id'], kind)
                self.assertEqual(error.observation['type_field_byte_offset'], offsets[0])

    def test_allowlist_is_not_widened(self):
        self.assertEqual(m.TENSOR_TYPES, {0: ('F32', 4), 1: ('F16', 2)})
        for kind in (0, 1):
            raw, _ = fixture((kind,))
            result = m.fingerprint_stream(io.BytesIO(raw), len(raw))
            self.assertEqual(result['container_sha256'], m.digest(raw))
            self.assertNotIn('tensor_type_observation', result)

    def test_model_level_f16_label_does_not_override_tensor_type(self):
        _, _, error = self.rejected(30, misleading_file_type=True)
        self.assertEqual(error.observation['ggml_type_id'], 30)
        self.assertFalse(error.observation['inferred_from_model_label'])

    def test_second_tensor_index_is_exact(self):
        raw, offsets = fixture((1, 30))
        with self.assertRaises(m.UnsupportedTensorType) as raised:
            m.fingerprint_stream(io.BytesIO(raw), len(raw))
        self.assertEqual(raised.exception.observation['tensor_index_zero_based'], 1)
        self.assertEqual(raised.exception.observation['type_field_byte_offset'], offsets[1])

    def test_rejects_before_tensor_offset_or_payload_read(self):
        raw, offsets = fixture((30,))
        class Bounded(io.BytesIO):
            def read(self, n=-1):
                if self.tell() + n > offsets[0] + 4:
                    raise AssertionError('Read past rejected type field')
                return super().read(n)
        stream = Bounded(raw)
        with self.assertRaises(m.UnsupportedTensorType):
            m.fingerprint_stream(stream, len(raw))
        self.assertEqual(stream.tell(), offsets[0] + 4)

    def test_no_partial_digest_is_claimed_authenticated(self):
        _, _, error = self.rejected()
        record = m.rejected_model(error)
        self.assertEqual(record['state'], 'UNAVAILABLE_OR_REJECTED')
        self.assertEqual(record['tensor_type_observation']['historical_blob_sha256_verification'], 'NOT_COMPLETED')
        self.assertEqual(record['tensor_type_observation']['scope'], 'PARTIAL_HEADER_NOT_FULL_FILE_AUTHENTICATED')
        self.assertNotIn('container_sha256', record)
        self.assertNotIn('matches_historical_from_digest', record)

    def test_raw_tensor_names_never_exported(self):
        _, _, error = self.rejected(name='PRIVATE_FIXTURE_NAME_DO_NOT_EXPORT')
        serialized = json.dumps(m.rejected_model(error))
        self.assertNotIn('PRIVATE_FIXTURE_NAME', serialized)
        self.assertNotIn('tensor_name', serialized)

    def test_invalid_observation_fields_are_refused(self):
        for args in ((-1, 0, 0), (2**32, 0, 0), (True, 0, 0), ('30', 0, 0),
                     (30, -1, 0), (30, m.MAX_TENSORS, 0), (30, False, 0),
                     (30, 0, -1), (30, 0, m.MAX_HEADER), (30, 0, 1.0)):
            with self.subTest(args=args), self.assertRaisesRegex(m.ForensicsError, 'INVALID_TENSOR_TYPE_OBSERVATION'):
                m.UnsupportedTensorType(*args)

    def test_non_type_failures_do_not_fabricate_a_type(self):
        for error in (m.ForensicsError('BAD_MAGIC'), OSError('PRIVATE_PATH_OR_SECRET')):
            record = m.rejected_model(error)
            self.assertNotIn('tensor_type_observation', record)
            self.assertNotIn('PRIVATE_PATH', json.dumps(record))

    def test_incomplete_type_field_is_not_a_type_observation(self):
        raw, offsets = fixture((30,))
        for size in range(offsets[0], offsets[0]+4):
            with self.assertRaises(m.ForensicsError) as raised:
                m.fingerprint_stream(io.BytesIO(raw[:size]), size)
            self.assertNotIsInstance(raised.exception, m.UnsupportedTensorType)

    def test_file_rejection_does_not_change_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.gguf'; raw, _ = fixture((30,)); path.write_bytes(raw)
            with self.assertRaises(m.UnsupportedTensorType) as raised:
                m.inspect_file(path, m.digest(raw))
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(raised.exception.observation['ggml_type_id'], 30)

    def test_failed_reports_checkpoints_and_console_preserve_numeric_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp); (home / '.ollama/models/blobs').mkdir(parents=True)
            with patch.object(m.os, 'name', 'nt'), patch.dict(os.environ, {'COMPUTERNAME': 'BETTERWITHAGE'}), \
                 patch.object(m, 'Path') as paths, patch.object(m, 'require_local_windows_path'), \
                 patch.object(m, 'inspect_file', side_effect=[m.UnsupportedTensorType(30, 0, 100),
                                                            m.UnsupportedTensorType(12, 3, 200)]), \
                 patch.object(m, 'compare') as compare, contextlib.redirect_stdout(io.StringIO()) as console:
                paths.side_effect = type(home); paths.home.return_value = home
                self.assertEqual(m.main(['--inspect-blobs']), 1)
            compare.assert_not_called()
            directory = next(home.glob('szl-gguf-forensics-*'))
            report = json.loads((directory / 'gguf-forensics.json').read_text())
            self.assertEqual(report['state'], 'INCOMPLETE'); self.assertIsNone(report['comparison'])
            self.assertEqual([x['tensor_type_observation']['ggml_type_id'] for x in report['models'].values()], [30, 12])
            for flag in ('inference', 'training', 'model_weights_changed', 'paired_attempts_touched',
                         'publication_eligible', 'automatic_resume_authorized'):
                self.assertIs(report[flag], False)
            self.assertEqual(len(list(directory.glob('checkpoint-*.json'))), 5)
            saved = json.loads((directory / 'checkpoint-004.json').read_text())
            self.assertEqual(saved['report']['models'], report['models'])
            self.assertEqual(saved['report']['state'], 'INCOMPLETE')
            self.assertIn('ggml_type_id=30', console.getvalue())
            self.assertIn('ggml_type_id=12', console.getvalue())
            self.assertFalse((home / '.szl-recovery-attempts').exists())

    def test_checkpoint_failure_still_prevents_blob_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp); (home / '.ollama/models/blobs').mkdir(parents=True)
            with patch.object(m.os, 'name', 'nt'), patch.dict(os.environ, {'COMPUTERNAME': 'BETTERWITHAGE'}), \
                 patch.object(m, 'Path') as paths, patch.object(m, 'require_local_windows_path'), \
                 patch.object(m, 'checkpoint', side_effect=OSError('fsync fixture')), patch.object(m, 'inspect_file') as inspect:
                paths.side_effect = type(home); paths.home.return_value = home
                with self.assertRaises(OSError): m.main(['--inspect-blobs'])
                inspect.assert_not_called()

    def test_default_plan_does_not_read_blobs(self):
        with patch.object(m, 'inspect_file') as read, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main([]), 0)
        read.assert_not_called()

    def test_supported_fixture_fingerprint_unchanged_from_parent(self):
        # Fixed expected bytes derived with the unmodified pinned parent, not a new score.
        raw, _ = fixture((0, 1))
        result = m.fingerprint_stream(io.BytesIO(raw), len(raw))
        self.assertEqual(result['tensor_set_sha256'], '9a341696ba24e4448de9a5aca316ef9113edb6ded84483a3691de38a5d7f8781')
        self.assertEqual(result['tensor_count'], 2)
        self.assertEqual(result['tensor_bytes'], 12)
        self.assertFalse(result['tensor_values_or_health_checked'])
        self.assertEqual(m.compare(result, result)['same_tensor_set_bytes'], True)


if __name__ == '__main__':
    unittest.main()
