# SPDX-License-Identifier: Apache-2.0
"""Offline identity separation: original Khipu 2/6 is not its abstain candidate 3/6.

Reuse the existing signature/training-link/dataset verifier. This verifies source
receipts and pinned observation identity, never weights, fresh inference, hidden
benchmark acceptance, model serving, or permission to execute model proposals.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
import verify_model_portfolio as portfolio

CANDIDATE = 'frontier/khipu-abstain-20260908'
MANIFEST = portfolio.ROOT / CANDIDATE / 'identity.json'
KEY_ID = '89540347a69b789e'
ANCHORS = {
    'original': {
        'repo_id': 'SZLHOLDINGS/SZL-Khipu-1.5B',
        'hub_revision': '8d3032dcf954f347b98f80050de107169d69eb7e',
        'local_receipt_dir': 'khipu', 'abstain_correct': 2,
        'weights_artifact_sha256': 'ea91ef6aee4e147f5ae5b3cafc4615059749549c735b1078c3c7fc146ca6791d',
        'receipt_sha256': {
            'owner_pubkey.json': '843d0958392b4ee11ad8e36519261bebf841ee20caec479cbbc4bb9e8c991031',
            'training_receipt.signed.json': '7af76dd4f26dcd122012bfd1e47a0f55481a952b86aee28956cf7cfaaf59bd04',
            'eval_receipt.signed.json': '32edd2d862fd5abac390bee3d30950f4718afedc41f4da4e24f3d0dfe67f8450',
        },
    },
    'candidate': {
        'repo_id': 'SZLHOLDINGS/SZL-Khipu-1.5B-abstain',
        'hub_revision': '7f93c1b53d19a174e831af61425f101c8cbd41de',
        'local_receipt_dir': CANDIDATE, 'abstain_correct': 3,
        'weights_artifact_sha256': '86c33222c44349c31c3db9a42429a4bfe7b79acc1b68b26645645e881e553476',
        'receipt_sha256': {
            'owner_pubkey.json': '843d0958392b4ee11ad8e36519261bebf841ee20caec479cbbc4bb9e8c991031',
            'training_receipt.signed.json': '63b76e044c23e3fb5d2c759a0c7027aa0d3ad55d682da658d2d260420cc0c870',
            'eval_receipt.signed.json': 'd1114da7fc8fec9d591234c7812af31002bf8a9bfc1f542503f0cdb32a9d3cd5',
        },
    },
}


class IdentityError(ValueError):
    """Signed evidence must stay attached to the artifact actually observed."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise IdentityError(code)


def verify(document: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(document, dict), 'MANIFEST_SHAPE')
    require(document.get('schema') == 'szl.khipu.distinct-candidate/v1', 'MANIFEST_SCHEMA')
    require(document.get('bindings') == ANCHORS, 'ARTIFACT_SCOPE_MISMATCH')
    for field in ('production_authorization', 'autonomy_eligible', 'hidden_l2_acceptance',
                  'weights_recomputed', 'quantized_metrics_inherited'):
        require(document.get(field) is False, 'UNWITNESSED_PROMOTION:' + field)
    results = {}
    for name, binding in ANCHORS.items():
        root = portfolio.ROOT / binding['local_receipt_dir']
        # Exact committed byte identities also preserve CRLF receipt wrappers.
        for filename, expected in binding['receipt_sha256'].items():
            require(portfolio.sha256_source(root / filename) == expected,
                    'RECEIPT_IDENTITY_MISMATCH:' + name + ':' + filename)
        evidence = portfolio.verify_signed_receipts(root)
        require(evidence['status'] == 'DECLARED_KEY_SIGNATURES_VALID'
                and evidence['key_id'] == KEY_ID, 'DECLARED_KEY_MISMATCH')
        require(evidence['weights_artifact_sha256'] == binding['weights_artifact_sha256'],
                'SIGNED_WEIGHT_LINEAGE_MISMATCH')
        expected_counts = {'abstainCorrect': binding['abstain_correct'], 'abstainTotal': 6,
                           'groundingCorrect': 4, 'groundingTotal': 5, 'planValid': 11,
                           'planTotal': 11, 'hallucinatedCitationCount': 0}
        for key, value in expected_counts.items():
            require(type(evidence.get(key)) is int and evidence[key] == value,
                    'SIGNED_METRIC_MISMATCH:' + name + ':' + key)
        require(evidence['weights_hash_recomputed'] is False, 'WEIGHT_EXECUTION_NOT_PERFORMED')
        results[name] = {'repo_id': binding['repo_id'], 'hub_revision': binding['hub_revision'],
                         'evidence': evidence}
    require(results['original']['evidence']['weights_artifact_sha256'] !=
            results['candidate']['evidence']['weights_artifact_sha256'], 'CANDIDATE_ALIASING')
    current = json.loads(portfolio.DEFAULT_PORTFOLIO.read_text(encoding='utf-8'))
    original = [r for r in current['artifacts'] if r['repo_id'] == ANCHORS['original']['repo_id']]
    require(len(original) == 1 and original[0]['local_receipt_dir'] == 'khipu'
            and original[0]['autonomy_eligible'] is False
            and '2/6' in ' '.join(original[0]['limitations']), 'ORIGINAL_PORTFOLIO_REBOUND')
    require(not any(r.get('local_receipt_dir') == CANDIDATE for r in current['artifacts']),
            'CANDIDATE_PORTFOLIO_ADMISSION_NOT_AUTHORIZED')
    body = {'schema': 'szl.khipu.distinct-candidate-witness/v1',
            'state': 'SOURCE_RECEIPT_INTEGRITY_VALID', 'bindings': results,
            'live_hub_observation': 'PINNED_PRIOR_OBSERVATION_NOT_REFRESHED_BY_THIS_OFFLINE_CHECK',
            'production_authorization': False, 'weights_downloaded': False,
            'weights_recomputed': False, 'model_executed': False,
            'hidden_l2_acceptance': False, 'quantized_metrics_inherited': False,
            'production_disposition': 'HOLD'}
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**body, 'record_sha256': hashlib.sha256(raw).hexdigest()}


def main() -> int:
    try:
        report = verify(json.loads(MANIFEST.read_text(encoding='utf-8')))
    except Exception as exc:
        report = {'state': 'INCOMPLETE', 'error_type': type(exc).__name__,
                  'production_authorization': False, 'production_disposition': 'HOLD'}
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report['state'] == 'SOURCE_RECEIPT_INTEGRITY_VALID' else 1


if __name__ == '__main__':
    raise SystemExit(main())
