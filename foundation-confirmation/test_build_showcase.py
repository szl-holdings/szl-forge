import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import build_showcase as build

HERE = Path(__file__).resolve().parent
ARCHIVES = (
    HERE.parent / 'spaces/szl-forge-lab/foundation/data/release.zip',
    HERE.parent.parent / 'outputs/SZL-foundation-confirmation-v0.4-20260925.zip',
)
ARCHIVE = next((p for p in ARCHIVES if p.is_file()), ARCHIVES[0])


class ReplayBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.files = build.read_release(ARCHIVE)

    def test_complete_source_and_deterministic_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'a', Path(temp) / 'b'
            catalog = build.build(ARCHIVE, first)
            build.build(ARCHIVE, second)
            self.assertEqual(catalog['row_count'], 5184)
            self.assertEqual(catalog['overall_registered_gate'], 'FAILED')
            for path in (first / 'data').rglob('*'):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
            exported = []
            for family in build.FAMILIES:
                path = first / catalog['episodes'][family]['path']
                self.assertEqual(build.sha(path.read_bytes()), catalog['episodes'][family]['sha256'])
                exported.extend(json.loads(path.read_bytes()))
            original = [json.loads(line) for line in self.files['evaluation.jsonl'].splitlines()]
            self.assertEqual({r['row_sha256'] for r in exported}, {r['row_sha256'] for r in original})
            self.assertEqual(len(exported), len(original))

    def test_changed_archive_is_refused_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'bad.zip'
            p.write_bytes(self.raw + b'changed')
            with self.assertRaisesRegex(ValueError, 'archive hash'):
                build.build(p, Path(temp) / 'public')
            self.assertFalse((Path(temp) / 'public').exists())

    def test_source_directory_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            for name, data in self.files.items():
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            (source / 'core.py').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Source directory drift'):
                build.read_release(ARCHIVE, source)

    def test_row_tamper_rejected_even_if_journal_binding_rewritten(self):
        files = copy.copy(self.files)
        rows = [json.loads(line) for line in files['evaluation.jsonl'].splitlines()]
        rows[0]['accuracy'] = 0.25
        files['evaluation.jsonl'] = b''.join(build.encode(r) for r in rows)
        summary = json.loads(files['benchmark-summary.json'])
        summary['evaluation_sha256'] = hashlib.sha256(files['evaluation.jsonl']).hexdigest()
        files['benchmark-summary.json'] = build.encode(summary)
        with self.assertRaisesRegex(ValueError, 'row hash'):
            build.validate_rows(files)

    def test_incomplete_population_rejected(self):
        files = copy.copy(self.files)
        files['evaluation.jsonl'] = b'\n'.join(files['evaluation.jsonl'].splitlines()[:-1]) + b'\n'
        summary = json.loads(files['benchmark-summary.json'])
        summary['evaluation_sha256'] = build.sha(files['evaluation.jsonl'])
        files['benchmark-summary.json'] = build.encode(summary)
        with self.assertRaisesRegex(ValueError, 'Incomplete evaluation'):
            build.validate_rows(files)

    def test_summary_rewrite_rejected(self):
        files = copy.copy(self.files)
        summary = json.loads(files['benchmark-summary.json'])
        summary['summaries']['clean']['learned']['net_utility'] = 1
        files['benchmark-summary.json'] = build.encode(summary)
        with self.assertRaisesRegex(ValueError, 'Aggregate metric'):
            build.validate_rows(files)


if __name__ == '__main__':
    unittest.main()
