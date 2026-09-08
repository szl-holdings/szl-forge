"""Replay the recorded CPU result without rerunning or repairing the experiment."""
import unittest
from pathlib import Path

from inference import minicpm5_gguf as g

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'frontier/minicpm5/evidence/2026-09-08-gguf-q4-cpu.json'
DIGEST = 'd5354fadbfbaab0f865079ce54d866ff5874b2a2530e846f8152536be07cb986'


class GGUFMeasuredEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.report = g.loads(RECORD.read_bytes())

    def test_original_log_record_and_nested_source_integrity(self):
        value = dict(self.report)
        self.assertEqual(value.pop('recordSha256'), DIGEST)
        self.assertEqual(g.base.digest(value), DIGEST)
        observation = dict(value['sourceObservation'])
        expected = observation.pop('recordSha256')
        self.assertEqual(g.base.digest(observation), expected)

    def test_fixed_negative_outcome_and_behavioral_differences(self):
        self.assertEqual(g.summarize(self.report, g.baseline()), self.report)
        self.assertEqual(self.report['status'], 'SMOKE_FAIL')
        self.assertEqual(self.report['passedCases'], 9)
        self.assertEqual(self.report['recoveredCaseIds'], ['probe_00'])
        self.assertEqual(self.report['newlyFailedCaseIds'], ['probe_06'])
        self.assertEqual(self.report['sameOutputHashCount'], 10)
        self.assertEqual([r['id'] for r in self.report['cases'] if not r['passed']], ['probe_04', 'probe_06', 'probe_10'])

    def test_executed_source_model_and_binary_remain_exact(self):
        self.assertEqual(self.report['sourceRevision'], '3b749712244c1d54cf218500d908be53854d912b')
        self.assertEqual(self.report['runnerSha256'], 'e94f91230c25e712ae77b076a3bfabe60704ae70b244c8c8b12fb781dfa9a573')
        self.assertEqual(self.report['modelSha256'], g.FILES['Q4_K_M'][2])
        self.assertEqual(self.report['runtimeArchiveSha256'], g.RUNTIME['sha256'])
        self.assertEqual(self.report['serverBinarySha256'], '07723ae07835bdf11cf0d00d68eb4df8308df0603f689c64c9f841a626cad01d')
        self.assertTrue(self.report['artifactBytesVerified'])
        self.assertEqual(self.report['plan'], g.plan())

    def test_no_parity_speedup_or_production_claim_can_follow(self):
        self.assertEqual(self.report['comparisonClass'], 'HISTORICAL_BEHAVIOR_REFERENCE')
        self.assertFalse(self.report['matchedQuantizationExperiment'])
        self.assertIsNone(self.report['speedupClaim'])
        self.assertEqual(self.report['productionDisposition'], 'HOLD')
        for key in ['modelOperational', 'publicationEligible', 'runtimeQualified', 'sealed', 'trainingAuthorized', 'toolExecuted']:
            self.assertFalse(self.report[key])
        self.assertEqual(g.baseline()['status'], 'SMOKE_FAIL')


if __name__ == '__main__':
    unittest.main()
