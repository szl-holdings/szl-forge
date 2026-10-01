"""Verify qualification records before generating non-operational projections.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
Hash verification is integrity checking, not a signature or independent proof
that a job ran. Match the external job logs and source pin before publishing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location('minicpm_contract', Path(__file__).with_name('minicpm5_qualification.py'))
assert _SPEC is not None and _SPEC.loader is not None
q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q)


class ReportError(ValueError):
    """Inconsistent records cannot become a product or proof projection."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ReportError(reason)


def verify(report: dict[str, Any], source: str, runner_hash: str) -> dict[str, Any]:
    require(q.SHA40.fullmatch(source) is not None, 'expected source must be full SHA')
    require(q.SHA64.fullmatch(runner_hash) is not None, 'expected runner must be SHA-256')
    require(isinstance(report, dict), 'report must be object')
    value = dict(report)
    recorded = value.pop('recordSha256', None)
    require(recorded == q.digest(value), 'record digest mismatch')
    require(value.get('schema') == 'szl.forge.minicpm5-qualification.v1', 'unknown schema')
    require(value.get('sourceRepository') == q.SOURCE and value.get('sourceRevision') == source,
            'source identity mismatch')
    require(value.get('runnerSha256') == runner_hash, 'executed runner mismatch')
    require(value.get('plan') == q.plan(), 'model pin, suite or limits differ')
    require(value.get('productionDisposition') == 'HOLD', 'production promotion prohibited')
    for key in ('sealed', 'publicationEligible', 'runtimeQualified', 'trainingAuthorized',
                'toolExecuted', 'imageDigestVerified'):
        require(value.get(key) is False, 'unsupported authority or qualification claim')
    for key in ('joules', 'costUsd', 'ttftMs'):
        require(key in value and value[key] is None, 'unmeasured field must remain null')
    rows = value.get('cases')
    require(isinstance(rows, list) and len(rows) <= len(q.suite()), 'case count out of bounds')
    expected = q.suite()[:len(rows)]
    allowed_reasons = {'invalid_json', 'truncated_output', 'output_schema', 'decision_or_grounding_mismatch'}
    for row, case in zip(rows, expected, strict=True):
        require(isinstance(row, dict) and row.get('id') == case['id'] and row.get('category') == case['category'],
                'case order, identity or category mismatch')
        require(type(row.get('passed')) is bool, 'case verdict must be boolean')
        reasons = row.get('reasonCodes')
        require(isinstance(reasons, list) and all(isinstance(r, str) and r in allowed_reasons for r in reasons), 'unknown reason')
        require(row['passed'] == (not reasons), 'case verdict disagrees with reasons')
        require(isinstance(row.get('outputSha256'), str) and q.SHA64.fullmatch(row['outputSha256']) is not None,
                'output hash missing')
        for key in ('generationMs', 'peakCudaAllocatedBytes', 'peakCudaReservedBytes', 'generatedTokens'):
            number = row.get(key)
            require(type(number) in (int, float) and math.isfinite(number) and number >= 0, 'invalid measured number')
        require(type(row['generatedTokens']) is int and row['generatedTokens'] <= q.MAX_NEW_TOKENS, 'token bound violated')
        require(not row['passed'] or row['generatedTokens'] > 0, 'empty generation cannot pass')
    require(not rows or value.get('modelLoaded') is True, 'measurements without loaded model')
    derived = q.finalize(value)
    for key in ('status', 'completedCases', 'passedCases', 'expectedCases', 'p50GenerationMs',
                'p95GenerationMs', 'evidenceClass'):
        require(type(value.get(key)) is type(derived[key]) and value.get(key) == derived[key], 'summary does not match case evidence')
    return report


def projections(report: dict[str, Any], source: str, runner_hash: str) -> dict[str, dict[str, Any]]:
    record = verify(report, source, runner_hash)
    common = {'modelId': q.MODEL, 'modelRevision': q.REVISION,
              'sourceRepository': q.SOURCE, 'sourceRevision': source,
              'evidenceRecordSha256': record['recordSha256'], 'productionDisposition': 'HOLD',
              'modelOperational': False, 'independentlyCertified': False,
              'trust': 'UNSIGNED_RECORD_REQUIRES_EXTERNAL_JOB_VERIFICATION'}
    return {
        'product': dict(common, schema='szl.forge.product-evaluation-projection.v1',
            surface='a-11-oy.com', status='EVALUATION', executionState=record['status']),
        'proof': dict(common, schema='szl.forge.proof-evaluation-projection.v1', surface='a11oy.net',
            suiteClass='PUBLIC_SYNTHETIC_CONTRACT_PROBES', completedCases=record['completedCases'],
            expectedCases=record['expectedCases'], passedCases=record['passedCases'],
            p50GenerationMs=record['p50GenerationMs'], p95GenerationMs=record['p95GenerationMs'],
            limitations=q.plan()['remainingGates']),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-source', required=True)
    parser.add_argument('--expected-runner-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        with args.report.open('rb') as stream:
            raw = stream.read(1024 * 1024 + 1)
        require(len(raw) <= 1024 * 1024, 'report exceeds size bound')
        report = json.loads(raw, object_pairs_hook=q._pairs, parse_constant=q._constant)
        result = projections(report, args.expected_source, args.expected_runner_sha256)
        q.atomic_json(args.output, result)
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        # A previously valid projection must not survive failed current verification.
        q.atomic_json(args.output, {'status': 'INVALID', 'modelOperational': False,
            'productionDisposition': 'HOLD', 'errorType': type(exc).__name__})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
