#!/usr/bin/env python3
"""Build a complete static replay from an immutable, admitted research archive."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

ARCHIVE_SHA256 = '869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03'
MANIFEST_SHA256 = '03a13779b09f2e8ad3dd53d928ac460a395d329742f9f43f7e5892fba672c877'
FAMILIES = ('clean', 'shared_bias', 'independent_outliers', 'reference_outliers', 'bias_drift', 'compositional')
POLICIES = ('no_confirmation', 'independent_bias', 'source_aware', 'cheap_only', 'reference_only', 'random_mixed', 'learned')
SEEDS = (17, 23, 41)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def load_json(data: bytes):
    def reject(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    return json.loads(data, parse_constant=reject)


def read_release(archive: Path, source: Path | None = None) -> tuple[bytes, dict[str, bytes]]:
    raw = archive.read_bytes()
    if sha(raw) != ARCHIVE_SHA256:
        raise ValueError('Frozen archive hash mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member')
        files = {name: z.read(name) for name in names}
    if sha(files['release-manifest.json']) != MANIFEST_SHA256:
        raise ValueError('Frozen manifest hash mismatch')
    manifest = load_json(files['release-manifest.json'])
    expected = set(manifest['files']) | {'release-manifest.json'}
    if expected != set(files):
        raise ValueError('Archive file set mismatch')
    for name, declared in manifest['files'].items():
        data = files[name]
        if len(data) != declared['bytes'] or sha(data) != declared['sha256']:
            raise ValueError(f'Archive member mismatch: {name}')
    if source is not None:
        for name, data in files.items():
            if (source / name).read_bytes() != data:
                raise ValueError(f'Source directory drift: {name}')
    return raw, files


def validate_rows(files: dict[str, bytes]) -> tuple[list[dict], dict, dict]:
    protocol = load_json(files['protocol.json'])
    summary = load_json(files['benchmark-summary.json'])
    protocol_sha = sha(files['protocol.json'])
    if files['protocol.sha256'].decode().strip() != protocol_sha:
        raise ValueError('Protocol companion mismatch')
    if summary['protocol_sha256'] != protocol_sha or summary['evaluation_sha256'] != sha(files['evaluation.jsonl']):
        raise ValueError('Summary source binding mismatch')
    if tuple(protocol['families']) != FAMILIES or tuple(protocol['policies']) != POLICIES or tuple(protocol['model_seeds']) != SEEDS:
        raise ValueError('Unexpected protocol population')
    rows = [load_json(line) for line in files['evaluation.jsonl'].splitlines() if line]
    expected = {(f, i, p, s) for f in FAMILIES for i in range(96) for p in POLICIES for s in (SEEDS if p == 'learned' else (None,))}
    observed = set()
    groups = defaultdict(list)
    paired = {}
    for row in rows:
        content = {k: v for k, v in row.items() if k != 'row_sha256'}
        digest = sha(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        if row['row_sha256'] != digest or row['protocol_sha256'] != protocol_sha:
            raise ValueError('Journal row hash or protocol mismatch')
        key = (row['family'], row['episode_index'], row['policy'], row['model_seed'])
        if key not in expected or key in observed:
            raise ValueError('Journal population or duplicate row mismatch')
        observed.add(key)
        if row['seed'] != protocol['test_seed']:
            raise ValueError('Journal seed mismatch')
        pair_key = key[:2]
        pair_value = (row['truth'], row['public'])
        if pair_key in paired and paired[pair_key] != pair_value:
            raise ValueError('Paired world mismatch')
        paired[pair_key] = pair_value
        if len(row['history']) != 4 + row['probes'] or row['inherited_observations'] != 4:
            raise ValueError('Observation accounting mismatch')
        if not math.isclose(row['net_utility'], row['accuracy'] - row['paid_cost'], abs_tol=1e-12):
            raise ValueError('Utility accounting mismatch')
        groups[(row['family'], row['policy'])].append(row)
    if observed != expected or len(rows) != 5184 or summary['rows'] != len(rows):
        raise ValueError('Incomplete evaluation population')
    for (family, policy), group in groups.items():
        metrics = summary['summaries'][family][policy]
        if metrics['rollouts'] != len(group):
            raise ValueError('Aggregate count mismatch')
        for metric, value in metrics.items():
            if metric == 'rollouts':
                continue
            actual = math.fsum(r[metric] for r in group) / len(group)
            if not math.isclose(actual, value, rel_tol=1e-11, abs_tol=1e-12):
                raise ValueError(f'Aggregate metric mismatch: {family}/{policy}/{metric}')
    if summary['primary']['overall_pass'] is not False or summary['primary']['clean_guard_pass'] is not False:
        raise ValueError('Frozen failed gate must remain failed')
    return rows, summary, protocol


def build(archive: Path, output: Path, source: Path | None = None) -> dict:
    raw, files = read_release(archive, source)
    rows, summary, protocol = validate_rows(files)
    generated = {
        'release.zip': raw,
        'summary.json': files['benchmark-summary.json'],
        'protocol.json': files['protocol.json'],
        'RESULTS.md': files['RESULTS.md'],
        'MATH.md': files['MATH.md'],
        'release-manifest.json': files['release-manifest.json'],
        'research.json': files['research/global.json'],
        'agency-research.json': files['research/agencies.json'],
    }
    episodes = {}
    for family in FAMILIES:
        family_rows = [r for r in rows if r['family'] == family]
        path = f'episodes/{family}.json'
        generated[path] = encode(family_rows)
        episodes[family] = {'path': f'data/{path}', 'sha256': sha(generated[path]), 'row_count': len(family_rows), 'bytes': len(generated[path])}
    catalog = {
        'schema': 'szl.confirmation.showcase/v1', 'showcase_version': '0.5',
        'release_name': 'SZL Foundation Confirmation v0.4',
        'research_sealed_utc': load_json(files['release-manifest.json'])['created_utc'],
        'row_count': len(rows), 'world_count': 576, 'episodes_per_family': 96,
        'families': list(FAMILIES), 'policies': list(POLICIES), 'model_seeds': list(SEEDS),
        'protocol_sha256': sha(files['protocol.json']), 'evaluation_sha256': sha(files['evaluation.jsonl']),
        'archive_sha256': ARCHIVE_SHA256, 'archive_bytes': len(raw), 'manifest_sha256': MANIFEST_SHA256,
        'summary_sha256': sha(generated['summary.json']), 'episodes': episodes,
        'generator_sha256': sha(Path(__file__).read_bytes().replace(b'\r\n', b'\n')),
        'source_repository': 'https://github.com/szl-holdings/szl-forge/tree/main/foundation-confirmation',
        'source_revision': None,
        'source_revision_note': 'Deployment commit is recorded separately by the canonical publisher; the research archive is bound by SHA-256.',
        'overall_registered_gate': 'FAILED', 'inference_performed_in_browser': False,
        'price_sensitivity': 'Post-hoc scores only; original observations and actions stay fixed.',
        'files': {f'data/{name}': {'sha256': sha(data), 'bytes': len(data)} for name, data in sorted(generated.items())},
    }
    generated['catalog.json'] = encode(catalog)
    data_dir = output / 'data'
    # A separate output directory is required; never write over input artifacts.
    if source is not None and (output.resolve() == source.resolve() or source.resolve() in output.resolve().parents):
        raise ValueError('Output must not be inside sealed source')
    desired = set(generated)
    if data_dir.exists():
        existing = {p.relative_to(data_dir).as_posix() for p in data_dir.rglob('*') if p.is_file()}
        if existing - desired:
            raise ValueError('Unexpected output files; use a fresh output directory')
    for name, data in generated.items():
        target = data_dir / name
        if target.resolve() == archive.resolve():
            if target.read_bytes() != data:
                raise ValueError('Refusing to overwrite input archive')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    catalog = build(args.archive, args.output, args.source)
    print(json.dumps({'status': 'VERIFIED_AND_BUILT', 'rows': catalog['row_count'], 'files': len(catalog['files']) + 1, 'archive_sha256': ARCHIVE_SHA256}))


if __name__ == '__main__':
    main()
