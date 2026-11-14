# SPDX-License-Identifier: Apache-2.0
"""Offline, exact-membership evidence panels; no network or publication authority."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

START = '<!-- SZL-EVIDENCE-PASSPORT:v1:START -->'
END = '<!-- SZL-EVIDENCE-PASSPORT:v1:END -->'
ANCHOR = b'<details>\n<summary>Technical details and evidence</summary>'
SHA40 = re.compile(r'[0-9a-f]{40}')
SHA64 = re.compile(r'[0-9a-f]{64}')
MODEL_ID = re.compile(r'SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
PATH_PART = re.compile(r'[A-Za-z0-9_.-]+')
SOURCE_ROUTES = {
    'SZLHOLDINGS/SZL-Khipu-1.5B': ('khipu/card/README.md', 'publish-khipu-card.yml'),
    'SZLHOLDINGS/KHIPU-R2': ('khipu-r2/card/README.md', 'publish-khipu-card.yml'),
    'SZLHOLDINGS/chaski': ('chaski/card/README.md', 'publish-chaski-card.yml'),
    'SZLHOLDINGS/chaski-5050': ('chaski-5050/card/README.md', 'publish-chaski-card.yml'),
    'SZLHOLDINGS/chaski-r2': ('chaski-r2/card/README.md', 'publish-chaski-card.yml'),
    'SZLHOLDINGS/chaski-r4': ('chaski_r4/card/README.md', 'publish-chaski-card.yml'),
}

BOUNDARY = ('Documentation and retrieved-record integrity only. No new training, '
            'model evaluation, weight-byte verification, independent signer identity, '
            'runtime qualification or promotion. Missing evidence is not a zero score.')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f'invalid {label}')
    return value


def safe_path(value: object) -> str:
    if not isinstance(value, str) or len(value) > 300:
        raise ValueError('invalid path')
    parts = value.split('/')
    if any(p in ('', '.', '..') or not PATH_PART.fullmatch(p) for p in parts):
        raise ValueError('invalid relative path')
    return value


def timestamp(value: object) -> str:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError('timestamp required')
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('invalid observation time') from exc
    if parsed.tzinfo is None:
        raise ValueError('observation timezone required')
    return parsed.isoformat()


def strict_json(raw: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON member')
            result[key] = value
        return result

    def constant(_: str) -> None:
        raise ValueError('non-finite JSON number')

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('non-finite JSON number')
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant,
                      parse_float=finite_float)


def validate(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError('record must be an object')
    ident = _text(record.get('id'), MODEL_ID, 'model identity')
    revision = _text(record.get('revision'), SHA40, 'immutable Hub revision')
    timestamp(record.get('observed_at'))
    status = record.get('card_http')
    if type(status) is not int or not 100 <= status <= 599:
        raise ValueError('explicit card HTTP observation required')
    if status == 200:
        _text(record.get('card_sha256'), SHA64, 'card digest')
    elif record.get('card_sha256') is not None:
        raise ValueError('unread card must not have an observed digest')
    count = record.get('recognized_artifact_count')
    if count is not None and (type(count) is not int or not 0 <= count <= 100000):
        raise ValueError('invalid observed file count')
    evidence = record.get('evidence')
    if not isinstance(evidence, list) or len(evidence) > 20:
        raise ValueError('bounded evidence list required')
    seen = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError('invalid evidence record')
        path = safe_path(item.get('path'))
        if path in seen:
            raise ValueError('duplicate evidence path')
        seen.add(path)
        expected = f'https://huggingface.co/{ident}/raw/{revision}/{quote(path, safe="/")}'
        if item.get('url') != expected or type(item.get('http')) is not int or item['http'] != 200:
            raise ValueError('evidence identity or retrieval mismatch')
        _text(item.get('sha256'), SHA64, 'evidence digest')
    route = record.get('source_route')
    if route is not None:
        if not isinstance(route, dict):
            raise ValueError('invalid source route')
        _text(route.get('revision'), SHA40, 'source revision')
        safe_path(route.get('path'))
        if route.get('repository') != 'szl-holdings/szl-forge':
            raise ValueError('source route outside reviewed repository')
        if SOURCE_ROUTES.get(ident) != (route.get('path'), route.get('publisher')):
            raise ValueError('model/source/publisher identity mismatch')


def validate_manifest(data: dict) -> None:
    if not isinstance(data, dict) or data.get('schema') != 'szl.card-evidence-coverage/v1':
        raise ValueError('coverage schema required')
    if data.get('scope') != 'anonymous_public' or data.get('complete_pagination') is not True:
        raise ValueError('complete, explicitly scoped pagination required')
    timestamp(data.get('catalog_observed_at'))
    identities = data.get('catalog_ids')
    models = data.get('models')
    if not isinstance(identities, list) or not identities or len(identities) > 5000:
        raise ValueError('bounded non-empty catalog required')
    for ident in identities:
        _text(ident, MODEL_ID, 'catalog identity')
    if len(identities) != len(set(identities)) or not isinstance(models, list):
        raise ValueError('duplicate catalog or invalid records')
    for record in models:
        validate(record)
    filenames = [ident.split('/')[1].casefold() for ident in identities]
    reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)),
                *(f'lpt{i}' for i in range(1, 10))}
    if len(filenames) != len(set(filenames)) or any(
        name.split('.')[0] in reserved or name.endswith('.') or name == 'coverage'
        for name in filenames
    ):
        raise ValueError('unsafe or colliding output filename')
    actual = [r['id'] for r in models]
    if len(actual) != len(set(actual)) or set(actual) != set(identities):
        raise ValueError('catalog/record identity coverage mismatch')
    if data.get('private_inventory') != 'UNQUALIFIED' or data.get('release_authorized') is not False:
        raise ValueError('observation cannot grant completeness or release authority')


def render(record: dict) -> str:
    validate(record)
    ident, rev = record['id'], record['revision']
    observed = timestamp(record['observed_at'])
    count = record.get('recognized_artifact_count')
    files = 'UNOBSERVED' if count is None else f'{count} filename hints; not tensor verification'
    lines = [START, '## Evidence passport', '',
             f'**Dated input observation: {observed}. Existing research limits remain in force.**', '',
             f'[Inspect the immutable repository snapshot](https://huggingface.co/{ident}/tree/{rev}).', '',
             '| Observation | Scope |', '|---|---|',
             f'| Repository revision | `{rev}` |',
             f'| Historical README retrieval | HTTP {record["card_http"]}; not a readback of this proposed panel |',
             f'| Recognized artifact filenames | {files} |',
             f'| Structured records retrieved | {len(record["evidence"])}; bounded selection, not an exhaustive evidence census |', '']
    if record.get('card_sha256'):
        lines += [f'Historical input README SHA-256: `{record["card_sha256"]}`.', '']
    route = record.get('source_route')
    if route:
        url = f'https://github.com/{route["repository"]}/blob/{route["revision"]}/{route["path"]}'
        lines += [f'[Reviewed authoring baseline]({url}). Existing publisher: `{route["publisher"]}`. '
                  'Source changes still require normal review and exact deployed-byte readback.', '']
    else:
        lines += ['**Publisher mapping is not established by this observation.** '
                  'Do not write this panel directly to the Hub.', '']
    if record['evidence']:
        lines += ['### Inspect the records', '', '| Record | Retrieved-byte SHA-256 |', '|---|---|']
        for item in record['evidence']:
            label = html.escape(item['path']).replace('_', '\\_')
            lines.append(f'| [{label}]({item["url"]}) | `{item["sha256"]}` |')
        lines.append('')
    else:
        lines += ['No repository-local structured record was retrieved in this bounded selection. '
                  'Source-linked records may exist elsewhere; no score is inferred.', '']
    lines += ['### Interpretation', '', BOUNDARY,
              'Training loss is not held-out performance. A copied receipt does not qualify a '
              'quantization or merged checkpoint. Existing negative results, license and '
              'technical disclosures remain authoritative.', '', END, '']
    return '\n'.join(lines)


def strip_panel(card: bytes) -> bytes:
    start, end = START.encode(), END.encode()
    if start not in card and end not in card:
        return card
    if card.count(start) != 1 or card.count(end) != 1:
        raise ValueError('partial or duplicate managed region')
    left, right = card.index(start), card.index(end)
    if right <= left or card[right + len(end):right + len(end) + 2] != b'\n\n':
        raise ValueError('malformed managed region')
    return card[:left] + card[right + len(end) + 2:]


def insert_panel(card: bytes, record: dict, expected_unmanaged_sha256: str) -> bytes:
    _text(expected_unmanaged_sha256, SHA64, 'source baseline hash')
    original = strip_panel(card)
    if digest(original) != expected_unmanaged_sha256:
        raise ValueError('unmanaged source changed; reconcile instead of overwriting')
    if not original.startswith(b'---\n') or b'\n---\n' not in original[4:]:
        raise ValueError('existing LF YAML frontmatter required')
    if original.count(ANCHOR) != 1:
        raise ValueError('single technical disclosure required')
    result = original.replace(ANCHOR, render(record).encode('utf-8') + b'\n' + ANCHOR, 1)
    if strip_panel(result) != original:
        raise ValueError('unmanaged bytes changed')
    return result


def coverage_markdown(data: dict) -> str:
    validate_manifest(data)
    lines = ['# Model-card evidence coverage', '',
             f'Dated public catalog: {data["catalog_observed_at"]}. '
             f'{len(data["models"])} model-type repositories, not a trained-model count.', '',
             'Private/gated content, training truth, weight equivalence and production readiness '
             'are not inferred. Unmapped publishers remain explicit open work.', '',
             '| Model repository | Card retrieval | Evidence JSON reads | Publisher mapping | Runtime |',
             '|---|---|---:|---|---|']
    for r in sorted(data['models'], key=lambda x: x['id']):
        label = r['id'].split('/')[1]
        route = r.get('source_route')
        owner = route['publisher'] if route else 'UNMAPPED_IN_THIS_PASS'
        lines.append(f'| [{label}](https://huggingface.co/{r["id"]}/tree/{r["revision"]}) | '
                     f'HTTP {r["card_http"]} | {len(r["evidence"])} | {owner} | NOT_TESTED |')
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    data = strict_json(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(data)
    if args.output_dir:
        products = {r['id'].split('/')[1] + '.md': render(r) for r in data['models']}
        products['COVERAGE.md'] = coverage_markdown(data)
        if len(products) != len(data['models']) + 1:
            raise ValueError('output filename collision')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, text in products.items():
            (args.output_dir / name).write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({'records': len(data['models']), 'identity_coverage': 'EXACT',
                      'private_inventory': 'UNQUALIFIED', 'publication_performed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
