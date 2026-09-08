"""Replay one immutable measured failure; these tests never execute a model."""
import copy
import gzip
import hashlib
import unittest
from pathlib import Path

from inference import minicpm5_matched as m

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'frontier/minicpm5/evidence/2026-09-08-matched-f16-q4.json.gz'
RECORD_SHA = 'f514192cf45013e55ad75d748554bce1769dbcac9ffdfbdacea00c065543fef0'


def load_record():
    archive = PATH.read_bytes()
    if hashlib.sha256(archive).hexdigest() != '324037fd2548507460e94890c866123c6606a2032432bf20e0d049c9f22da7a4':
        raise ValueError('archive bytes changed')
    with gzip.open(PATH, 'rb') as stream:
        raw = stream.read(200001)
    if len(raw) > 200000 or hashlib.sha256(raw).hexdigest() != 'e4fb24662541b61491939f9c1fccd848105256a37bf323c31b1ed8f9caeac9c1':
        raise ValueError('decompressed bytes changed')
    return m.g.loads(raw)


class MeasuredPairReplayTests(unittest.TestCase):
    def test_full_and_nested_measured_integrity(self):
        record = load_record()
        self.assertEqual(record['recordSha256'], RECORD_SHA)
        m.validate_pair(record)
        self.assertEqual(record['sourceRevision'], '04c68889c996b74c3b131190abd5f87b157a520c')
        self.assertEqual(record['context']['workerObservations'][0]['jobId'], '6a9ff695b012ba1d5b8f2b37')
        self.assertEqual(record['context']['workerObservations'][0], record['context']['workerObservations'][1])

    def test_equal_totals_preserve_measured_regression(self):
        record = load_record()
        self.assertEqual((record['f16PassedCases'], record['q4PassedCases']), (9, 9))
        self.assertFalse(record['verdictParity'])
        self.assertFalse(record['outputHashParity'])
        self.assertEqual(record['q4RecoveredCaseIds'], ['probe_00'])
        self.assertEqual(record['q4RegressedCaseIds'], ['probe_06'])
        self.assertEqual(sum(c['sameOutputHash'] for c in record['caseComparison']), 10)
        for variant in record['variantReports']:
            self.assertEqual(variant['status'], 'SMOKE_FAIL')
            self.assertEqual(variant['completedCases'], 12)
            self.assertFalse(variant['modelOperational'])
            self.assertEqual(variant['productionDisposition'], 'HOLD')

    def test_resealed_production_upgrade_is_rejected(self):
        record = copy.deepcopy(load_record())
        record['modelOperational'] = True
        record.pop('recordSha256')
        record['recordSha256'] = m.g.base.digest(record)
        with self.assertRaises(m.MatchedError):
            m.validate_pair(record)

    def test_runtime_and_statistical_claims_remain_unverified(self):
        record = load_record()
        self.assertFalse(record['runtimeQualified'])
        self.assertFalse(record['quantizationIsolationVerified'])
        self.assertIsNone(record['speedupClaim'])
        self.assertFalse(record['sealed'])
        self.assertFalse(record['publicationEligible'])
        self.assertEqual([r['p50RoundtripMs'] for r in record['variantReports']], [6403.305, 2657.402])
        self.assertEqual([r['peakResidentBytes'] for r in record['variantReports']], [5302505472, 2777247744])


if __name__ == '__main__':
    unittest.main()
