"""Read-only documentation lineage over the existing Forge coverage contract.

An external SHA-256 checks the supplied snapshot, not its issuer or model claims.
No weights, record contents, prompt text, private catalog, provider or job is read.
Dependencies are documentation inputs, NOT a discovered runtime dependency graph.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, urlencode

from .safeio import read_regular, strict_json

MAX_BYTES = 4_000_000
MAX_MODELS = 1000
_SCHEMA = 'szl.card-evidence-coverage/v1'
_MODEL = re.compile(r'SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
_PATH = re.compile(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*')
_TOP = {'schema', 'scope', 'catalog_observed_at', 'complete_pagination',
        'private_inventory', 'release_authorized', 'catalog_ids', 'models'}
_REQUIRED = {'id', 'revision', 'observed_at', 'card_http', 'card_sha256',
             'recognized_artifact_count', 'source_route', 'evidence'}
_OPTIONAL = {'artifact_hint', 'license_metadata', 'reported_github_links',
             'runtime_state', 'publication_state'}


class SelectionError(ValueError):
    """Unknown selection, not an error disclosing the configured filesystem."""


def _need(condition: bool) -> None:
    if not condition:
        raise ValueError('invalid_evidence_snapshot')


def _hash(value: object, size: int = 64) -> str:
    _need(type(value) is str and re.fullmatch(r'[0-9a-f]{%d}' % size, value) is not None)
    return value


def _path(value: object) -> str:
    _need(type(value) is str and 0 < len(value) <= 300 and _PATH.fullmatch(value) is not None)
    _need(not any(part in ('.', '..') for part in value.split('/')))
    return value


def _time(value: object) -> str:
    _need(type(value) is str and len(value) <= 40)
    _need(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})', value) is not None)
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _need(stamp.utcoffset() is not None and stamp <= datetime.now(timezone.utc))
    return value


def _read_contract(raw: bytes, expected_sha256: str) -> dict:
    _hash(expected_sha256)
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES)
    _need(hashlib.sha256(raw).hexdigest() == expected_sha256)
    data = strict_json(raw)
    _need(type(data) is dict and set(data) == _TOP and data['schema'] == _SCHEMA)
    _need(data['scope'] == 'anonymous_public' and data['complete_pagination'] is True)
    _need(data['private_inventory'] == 'UNQUALIFIED' and data['release_authorized'] is False)
    _time(data['catalog_observed_at'])
    ids, rows = data['catalog_ids'], data['models']
    _need(type(ids) is list and 1 <= len(ids) <= MAX_MODELS and type(rows) is list and len(rows) == len(ids))
    _need(all(type(i) is str and _MODEL.fullmatch(i) for i in ids))
    _need(len(set(ids)) == len(ids) and len({i.casefold() for i in ids}) == len(ids))
    seen = set()
    for row in rows:
        _need(type(row) is dict and _REQUIRED <= set(row) <= _REQUIRED | _OPTIONAL)
        ident = row['id']
        _need(type(ident) is str and ident in ids and ident not in seen)
        seen.add(ident)
        rev = _hash(row['revision'], 40)
        _time(row['observed_at'])
        status = row['card_http']
        _need(type(status) is int and 100 <= status <= 599)
        if status == 200:
            _hash(row['card_sha256'])
        else:
            _need(row['card_sha256'] is None)
        count = row['recognized_artifact_count']
        _need(count is None or (type(count) is int and 0 <= count <= 100000))
        # Optional producer annotations are bounded and never rendered as authority.
        _need(row.get('runtime_state', 'NOT_TESTED') == 'NOT_TESTED')
        _need(row.get('publication_state', 'NOT_PERFORMED') == 'NOT_PERFORMED')
        for name in ('artifact_hint', 'license_metadata'):
            value = row.get(name)
            _need(value is None or (type(value) is str and len(value) <= 256))
        links = row.get('reported_github_links', [])
        _need(type(links) is list and len(links) <= 100)
        _need(all(type(link) is str and len(link) <= 4096 for link in links))
        records = row['evidence']
        _need(type(records) is list and len(records) <= 20)
        paths = set()
        for item in records:
            _need(type(item) is dict and set(item) == {'path', 'url', 'http', 'sha256'})
            path = _path(item['path'])
            _need(path not in paths)
            paths.add(path)
            _hash(item['sha256'])
            _need(type(item['http']) is int and item['http'] == 200)
            _need(item['url'] == f'https://huggingface.co/{ident}/raw/{rev}/{quote(path, safe="/")}')
        source = row['source_route']
        if source is not None:
            _need(type(source) is dict and set(source) == {'repository', 'revision', 'path', 'publisher'})
            _need(source['repository'] == 'szl-holdings/szl-forge')
            _hash(source['revision'], 40)
            _path(source['path'])
            _path(source['publisher'])
            _need('/' not in source['publisher'] and source['publisher'].endswith('.yml'))
    _need(seen == set(ids))
    return data


def unavailable_lineage(state: str = 'UNCONFIGURED') -> dict:
    _need(state in {'UNCONFIGURED', 'INVALID_OR_UNAVAILABLE_SNAPSHOT'})
    return {'schema': 'szl.model-lab.evidence-lineage-view/v1', 'state': state,
            'counts': None, 'catalog': [], 'nodes': [], 'edges': [], 'selected_model': None,
            'snapshot_sha256': None, 'snapshot_hash_matched': False, 'observed_at': None,
            'hypothetical': False, 'invalidated_node': None, 'affected_models': None,
            'unresolved': [], 'execution_authority': False, 'publication_authority': False,
            'production_ready': False, 'signature_verified': False,
            'artifact_identity_verified': False, 'evaluation_verified': False,
            'live_freshness': 'NOT_RECHECKED', 'private_inventory': 'UNQUALIFIED',
            'scope': 'documentation input dependencies only; not runtime impact or model quality'}


def _identity(kind: str, *parts: str) -> str:
    return hashlib.sha256(json.dumps([kind, *parts], separators=(',', ':')).encode()).hexdigest()


def project_lineage(raw: bytes, expected_sha256: str, *, model: str | None = None,
                    invalidate: str | None = None) -> dict:
    data = _read_contract(raw, expected_sha256)
    rows = sorted(data['models'], key=lambda r: r['id'])
    chosen = model if model is not None else rows[0]['id']
    if chosen not in data['catalog_ids']:
        raise SelectionError('unknown_evidence_selection')
    nodes, edges, groups, panels = {}, set(), {}, {}

    def add(kind, parts, label, state, url=None, digest=None):
        key = _identity(kind, *parts)
        node = {'node_id': key, 'kind': kind, 'label': label, 'state': state,
                'url': url, 'digest': digest}
        if key in nodes:
            _need(nodes[key] == node)
        nodes[key] = node
        return key

    for row in rows:
        ident, rev = row['id'], row['revision']
        root = f'https://huggingface.co/{ident}'
        snapshot = add('model', [ident, rev], 'Hub repository revision', 'METADATA_REPORTED',
                       f'{root}/tree/{rev}', rev)
        card = add('card', [ident, rev], 'Historical README',
                   'READ_REPORTED' if row['card_http'] == 200 else 'UNAVAILABLE',
                   f'{root}/raw/{rev}/README.md', row['card_sha256'])
        projection = add('projection', [ident, rev], 'Documentation projection', 'INPUTS_ONLY')
        panels[projection] = ident
        local = {snapshot, card, projection}
        edges.update({(snapshot, card), (snapshot, projection), (card, projection)})
        for record in row['evidence']:
            key = add('record', [ident, rev, record['path']], record['path'],
                      'RETRIEVAL_REPORTED', record['url'], record['sha256'])
            local.add(key)
            edges.update({(snapshot, key), (key, projection)})
        source = row['source_route']
        if source:
            key = add('source', [source['repository'], source['revision'], source['path']],
                      'Declared authoring baseline', 'SOURCE_DECLARED',
                      f'https://github.com/{source["repository"]}/blob/{source["revision"]}/{source["path"]}',
                      source['revision'])
            local.add(key)
            edges.add((key, projection))
        groups[ident] = local
    selected_ids = groups[chosen]
    if invalidate is not None and (type(invalidate) is not str or invalidate not in selected_ids):
        raise SelectionError('unknown_evidence_selection')
    impacted = set()
    if invalidate is not None:
        adjacency = {}
        for left, right in edges:
            adjacency.setdefault(left, []).append(right)
        queue = deque([invalidate])
        while queue:
            key = queue.popleft()
            if key not in impacted:
                impacted.add(key)
                queue.extend(adjacency.get(key, []))
    selected = next(r for r in rows if r['id'] == chosen)
    unresolved = []
    if selected['card_http'] != 200: unresolved.append('README_NOT_RETRIEVED')
    if not selected['evidence']: unresolved.append('RECORDS_NOT_RETRIEVED')
    if selected['source_route'] is None: unresolved.append('SOURCE_ROUTE_UNMAPPED')
    result = unavailable_lineage()
    selected_nodes = []
    for key in sorted(selected_ids, key=lambda k: (nodes[k]['kind'], nodes[k]['label'], k)):
        node = dict(nodes[key], impact='RECHECK_REQUIRED_SIMULATION' if key in impacted else 'BASELINE_UNCHANGED')
        node['what_if_url'] = '/evidence?' + urlencode({'model': chosen, 'invalidate': key})
        selected_nodes.append(node)
    result.update(state='SNAPSHOT_HASH_CHECKED_NOT_QUALIFIED', snapshot_sha256=expected_sha256,
                  snapshot_hash_matched=True, observed_at=data['catalog_observed_at'],
                  selected_model=chosen, selected_observed_at=selected['observed_at'],
                  counts={'model_repositories': len(rows),
                          'cards_read_reported': sum(r['card_http'] == 200 for r in rows),
                          'structured_records_reported': sum(len(r['evidence']) for r in rows),
                          'source_routes_declared': sum(r['source_route'] is not None for r in rows)},
                  catalog=[{'model_id': r['id'], 'url': '/evidence?' + urlencode({'model': r['id']})} for r in rows],
                  nodes=selected_nodes,
                  edges=[{'from': a, 'to': b} for a, b in sorted(edges) if a in selected_ids and b in selected_ids],
                  hypothetical=invalidate is not None, invalidated_node=invalidate,
                  affected_models=sorted(panels[key] for key in impacted if key in panels) if invalidate else None,
                  unresolved=unresolved)
    return result


def load_lineage(path: Path, expected_sha256: str, **kwargs) -> dict:
    return project_lineage(read_regular(path, MAX_BYTES), expected_sha256, **kwargs)
