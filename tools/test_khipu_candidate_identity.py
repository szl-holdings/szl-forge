# SPDX-License-Identifier: Apache-2.0
"""Source-evidence tests and synthetic refusal tests; no model execution."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch
import verify_khipu_candidate_identity as identity


class KhipuCandidateIdentityTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(identity.MANIFEST.read_text(encoding='utf-8'))

    def test_distinct_signed_receipts_and_dataset_closures_verify(self):
        result = identity.verify(self.document)
        self.assertEqual(result['state'], 'SOURCE_RECEIPT_INTEGRITY_VALID')
        self.assertEqual(result['bindings']['original']['evidence']['abstainCorrect'], 2)
        self.assertEqual(result['bindings']['candidate']['evidence']['abstainCorrect'], 3)
        expected = result.pop('record_sha256')
        raw = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
        for key in ('production_authorization', 'weights_downloaded', 'model_executed',
                    'weights_recomputed', 'hidden_l2_acceptance', 'quantized_metrics_inherited'):
            self.assertIs(result[key], False)

    def test_repository_alias_refused(self):
        self.document['bindings']['candidate']['repo_id'] = self.document['bindings']['original']['repo_id']
        with self.assertRaises(identity.IdentityError): identity.verify(self.document)

    def test_revision_swap_refused(self):
        self.document['bindings']['candidate']['hub_revision'] = self.document['bindings']['original']['hub_revision']
        with self.assertRaises(identity.IdentityError): identity.verify(self.document)

    def test_receipt_directory_escape_or_alias_refused(self):
        for path in ('../../khipu', 'khipu'):
            doc = copy.deepcopy(self.document); doc['bindings']['candidate']['local_receipt_dir'] = path
            with self.assertRaises(identity.IdentityError): identity.verify(doc)

    def test_missing_or_true_authority_refused(self):
        for key in ('production_authorization', 'autonomy_eligible', 'hidden_l2_acceptance',
                    'weights_recomputed', 'quantized_metrics_inherited'):
            for value in (None, True, 0):
                doc = copy.deepcopy(self.document); doc[key] = value
                with self.assertRaises(identity.IdentityError): identity.verify(doc)

    def test_changed_committed_receipt_bytes_refused(self):
        with patch.object(identity.portfolio, 'sha256_source', return_value='0'*64):
            with self.assertRaises(identity.IdentityError): identity.verify(self.document)

    def test_signature_failure_is_not_swallowed_as_success(self):
        with patch.object(identity.portfolio, 'verify_signed_receipts', side_effect=ValueError('bad signature')):
            with self.assertRaises(ValueError): identity.verify(self.document)

    def test_source_signed_aggregate_cannot_be_gguf_lfs_hash(self):
        doc = copy.deepcopy(self.document)
        doc['bindings']['candidate']['weights_artifact_sha256'] = doc['candidate_weight_metadata'][1]['lfs_sha256']
        with self.assertRaises(identity.IdentityError): identity.verify(doc)

    def test_candidate_does_not_inherit_original_public_portfolio(self):
        current = json.loads(identity.portfolio.DEFAULT_PORTFOLIO.read_text())
        original = next(r for r in current['artifacts'] if r['repo_id'] == identity.ANCHORS['original']['repo_id'])
        self.assertEqual(original['local_receipt_dir'], 'khipu')
        self.assertIn('2/6', ' '.join(original['limitations']))
        self.assertFalse(any(r['repo_id'] == identity.ANCHORS['candidate']['repo_id'] for r in current['artifacts']))

    def test_candidate_five_synthetic_inputs_are_unchanged_source_blobs(self):
        candidate = identity.portfolio.ROOT / identity.CANDIDATE
        training = json.loads((candidate/'training_receipt.signed.json').read_text())
        self.assertEqual(len(training['payload']['datasets']), 5)
        for name, expected in training['payload']['datasets'].items():
            self.assertEqual(identity.portfolio.sha256_source(candidate/name), expected)
            self.assertEqual(identity.portfolio.sha256_source(identity.portfolio.ROOT/'khipu'/name), expected)


if __name__ == '__main__': unittest.main(verbosity=2)
