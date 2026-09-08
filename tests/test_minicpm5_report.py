"""Offline report tests; never a measured-model witness."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('report_verifier', ROOT/'inference/verify_minicpm5_report.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
SOURCE, RUNNER = 'a'*40, 'b'*64


def fixture():
    report = {'schema': 'szl.forge.minicpm5-qualification.v1', 'cases': [], 'plan': v.q.plan(),
              'sourceRepository': v.q.SOURCE, 'sourceRevision': SOURCE, 'runnerSha256': RUNNER,
              'modelLoaded': True, 'imageDigestVerified': False}
    for case in v.q.suite():
        result = v.q.grade(json.dumps(case['expected']), case['expected'])
        result.update(id=case['id'], category=case['category'], generationMs=1.0,
                      generatedTokens=20, peakCudaAllocatedBytes=100, peakCudaReservedBytes=200)
        report['cases'].append(result)
    return v.q.finalize(report)


def rehash(record):
    record.pop('recordSha256', None)
    record['recordSha256'] = v.q.digest(record)
    return record


class ReportTests(unittest.TestCase):
    def test_fixture_integrity_is_not_certification(self):
        result = v.projections(fixture(), SOURCE, RUNNER)
        for surface in result.values():
            self.assertFalse(surface['modelOperational'])
            self.assertFalse(surface['independentlyCertified'])
            self.assertIn('UNSIGNED', surface['trust'])

    def test_unhashed_change_is_rejected(self):
        report = fixture()
        report['passedCases'] = 0
        with self.assertRaises(v.ReportError):
            v.verify(report, SOURCE, RUNNER)

    def test_rehashed_false_summary_is_rejected(self):
        report = fixture()
        report['passedCases'] = 0
        with self.assertRaises(v.ReportError):
            v.verify(rehash(report), SOURCE, RUNNER)

    def test_duplicate_case_cannot_fill_missing_case(self):
        report = fixture()
        report['cases'][1] = report['cases'][0]
        with self.assertRaises(v.ReportError):
            v.verify(rehash(report), SOURCE, RUNNER)

    def test_wrong_expected_source_or_runner_rejected(self):
        for source, runner in [('c'*40, RUNNER), (SOURCE, 'c'*64), ('main', RUNNER)]:
            with self.subTest(source=source), self.assertRaises(v.ReportError):
                v.verify(fixture(), source, runner)

    def test_changed_model_or_workload_rejected(self):
        report = fixture()
        report['plan']['modelRevision'] = 'c'*40
        with self.assertRaises(v.ReportError):
            v.verify(rehash(report), SOURCE, RUNNER)

    def test_unsupported_promotion_or_signature_rejected(self):
        for key, val in [('productionDisposition','PROMOTE'), ('sealed',True), ('runtimeQualified',True), ('modelLoaded',False), ('costUsd',1)]:
            report = fixture()
            report[key] = val
            with self.subTest(key=key), self.assertRaises(v.ReportError):
                v.verify(rehash(report), SOURCE, RUNNER)

    def test_boolean_latency_and_excess_tokens_rejected(self):
        for key, val in [('generationMs',True), ('generatedTokens',97)]:
            report = fixture()
            report['cases'][0][key] = val
            with self.subTest(key=key), self.assertRaises(v.ReportError):
                v.verify(rehash(report), SOURCE, RUNNER)

    def test_reason_and_verdict_must_agree(self):
        report = fixture()
        report['cases'][0]['reasonCodes'] = ['invalid_json']
        with self.assertRaises(v.ReportError):
            v.verify(rehash(report), SOURCE, RUNNER)

    def test_negative_model_result_is_kept(self):
        report = fixture()
        report['cases'][0].update(passed=False, reasonCodes=['invalid_json'])
        report = v.q.finalize(report)
        result = v.projections(report, SOURCE, RUNNER)
        self.assertEqual(result['product']['executionState'], 'SMOKE_FAIL')
        self.assertEqual(result['proof']['passedCases'], 11)

    def test_cli_failure_replaces_previous_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder)/'in.json', Path(folder)/'out.json'
            src.write_text('{}')
            dst.write_text('{"modelOperational":true}')
            proc = subprocess.run([sys.executable, str(ROOT/'inference/verify_minicpm5_report.py'), '--report', str(src),
                '--expected-source', SOURCE, '--expected-runner-sha256', RUNNER, '--output', str(dst)], capture_output=True)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(json.loads(dst.read_text())['status'], 'INVALID')
            self.assertFalse(json.loads(dst.read_text())['modelOperational'])


if __name__ == '__main__':
    unittest.main()
