"""Replay the observed negative result; never rerun inference in CI."""
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('minicpm_evidence_verifier', ROOT / 'inference/verify_minicpm5_report.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
EVIDENCE = ROOT / 'frontier/minicpm5/evidence'
SOURCE = '8edf8c1ee72e5ae3cc59857097363e571a36d75f'
RUNNER = 'b1f750253004d43fd44ead3581245920867ee775b9388ff3eec002af80a29af7'


class ObservedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((EVIDENCE / '2026-09-07-a10g-smoke.json').read_text())

    def test_actual_record_integrity_and_negative_result_are_retained(self):
        report = v.verify(self.report, SOURCE, RUNNER)
        self.assertEqual(report['jobId'], '6a9f61fde686246ca69a9ad6')
        self.assertEqual(report['recordSha256'], '177c49709d7cc6f981bdda195bcea73febd8062335ecd9d3be4393b8095ed6b7')
        self.assertEqual(report['status'], 'SMOKE_FAIL')
        self.assertEqual((report['passedCases'], report['completedCases']), (9, 12))
        self.assertEqual([c['id'] for c in report['cases'] if not c['passed']], ['probe_00', 'probe_04', 'probe_10'])

    def test_failed_outputs_match_abstention_not_an_executed_tool(self):
        abstain = hashlib.sha256(b'{"decision":"ABSTAIN","evidence_id":null,"value":null}').hexdigest()
        for row in self.report['cases']:
            if not row['passed']:
                self.assertEqual(row['outputSha256'], abstain)
        self.assertFalse(self.report['toolExecuted'])

    def test_product_and_proof_projections_match_without_live_claim(self):
        expected = v.projections(self.report, SOURCE, RUNNER)
        stored = json.loads((EVIDENCE / '2026-09-07-projections.json').read_text())
        self.assertEqual(stored, expected)
        for surface in stored.values():
            self.assertFalse(surface['modelOperational'])
            self.assertEqual(surface['productionDisposition'], 'HOLD')


if __name__ == '__main__':
    unittest.main()
