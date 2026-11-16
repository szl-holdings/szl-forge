"""Evidence lineage is one authenticated view, never execution or model quality."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from szl_model_lab.app import Settings, create_app

TOKEN = 'test-only-lineage-' + 'x' * 32
AUTH = ('operator', TOKEN)


def sample():
    rows = []
    for name in ('alpha', 'beta'):
        ident = 'SZLHOLDINGS/' + name
        revision = 'a' * 40
        rows.append({'id': ident, 'revision': revision,
                     'observed_at': '2026-11-13T22:18:00+00:00',
                     'card_http': 200, 'card_sha256': 'b' * 64,
                     'recognized_artifact_count': 1,
                     'source_route': {'repository': 'szl-holdings/szl-forge',
                                      'revision': 'c' * 40, 'path': 'shared/card/README.md',
                                      'publisher': 'publish-example.yml'},
                     'evidence': [{'path': 'eval.json', 'http': 200, 'sha256': 'd' * 64,
                                   'url': f'https://huggingface.co/{ident}/raw/{revision}/eval.json'}]})
    return {'schema': 'szl.card-evidence-coverage/v1', 'scope': 'anonymous_public',
            'complete_pagination': True, 'catalog_observed_at': '2026-11-13T23:00:00Z',
            'private_inventory': 'UNQUALIFIED', 'release_authorized': False,
            'catalog_ids': [r['id'] for r in rows], 'models': rows}


def wire(data):
    raw = json.dumps(data, allow_nan=False).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def project(data=None, **kwargs):
    from szl_model_lab.evidence_view import project_lineage
    raw, digest = wire(data or sample())
    return project_lineage(raw, digest, **kwargs)


def configured(tmp_path, data=None):
    raw, digest = wire(data or sample())
    path = tmp_path / 'coverage.json'
    path.write_bytes(raw)
    settings = Settings(TOKEN, {}, {}, evidence_snapshot=path, evidence_snapshot_sha256=digest)
    return TestClient(create_app(settings), base_url='http://localhost'), path


@pytest.mark.parametrize('route', ['/evidence', '/api/evidence-lineage'])
def test_routes_require_auth(route):
    with TestClient(create_app(Settings(TOKEN, {}, {})), base_url='http://localhost') as client:
        assert client.get(route).status_code == 401


@pytest.mark.parametrize('route', ['/evidence', '/api/evidence-lineage'])
def test_unconfigured_is_unavailable_not_zero(route):
    with TestClient(create_app(Settings(TOKEN, {}, {})), base_url='http://localhost') as client:
        response = client.get(route, auth=AUTH)
        assert response.status_code == 503
        if route.startswith('/api'):
            assert response.json()['counts'] is None
        else:
            assert 'UNCONFIGURED' in response.text


def test_source_api_and_ui_agree(tmp_path):
    client, _ = configured(tmp_path)
    with client:
        data = client.get('/api/evidence-lineage', auth=AUTH).json()
        html = client.get('/evidence', auth=AUTH)
        assert data['counts']['model_repositories'] == 2
        assert data['selected_model'] == 'SZLHOLDINGS/alpha'
        assert data['snapshot_sha256'] in html.text
        for node in data['nodes']:
            assert node['node_id'] in html.text
        assert '/evidence' in client.get('/', auth=AUTH).text
        assert html.headers['cache-control'] == 'no-store'
        assert "frame-ancestors 'none'" in html.headers['content-security-policy']
        assert TOKEN not in html.text


def test_hypothetical_shared_source_propagates_without_writes(tmp_path):
    client, path = configured(tmp_path)
    before = path.read_bytes()
    with client:
        baseline = client.get('/api/evidence-lineage', auth=AUTH).json()
        source = next(n for n in baseline['nodes'] if n['kind'] == 'source')
        args = {'model': baseline['selected_model'], 'invalidate': source['node_id']}
        changed = client.get('/api/evidence-lineage', params=args, auth=AUTH).json()
        html = client.get('/evidence', params=args, auth=AUTH)
        assert changed['hypothetical'] is True
        assert changed['affected_models'] == ['SZLHOLDINGS/alpha', 'SZLHOLDINGS/beta']
        assert 'WHAT-IF' in html.text and 'RECHECK_REQUIRED_SIMULATION' in html.text
        assert client.get('/api/evidence-lineage', auth=AUTH).json() == baseline
    assert path.read_bytes() == before


def test_identical_receipt_bytes_do_not_merge_model_contexts():
    baseline = project()
    node = next(n for n in baseline['nodes'] if n['kind'] == 'record')
    changed = project(invalidate=node['node_id'])
    assert changed['affected_models'] == ['SZLHOLDINGS/alpha']
    assert not changed['artifact_identity_verified']


def test_unread_card_and_missing_records_preserve_unknown():
    data = sample()
    data['models'][0].update(card_http=401, card_sha256=None, evidence=[], source_route=None,
                             recognized_artifact_count=None)
    result = project(data)
    assert next(n for n in result['nodes'] if n['kind'] == 'card')['state'] == 'UNAVAILABLE'
    assert result['counts']['cards_read_reported'] == 1
    assert result['unresolved'] == ['README_NOT_RETRIEVED', 'RECORDS_NOT_RETRIEVED', 'SOURCE_ROUTE_UNMAPPED']


@pytest.mark.parametrize('fault', ['digest', 'private', 'authority', 'partial', 'missing_model',
    'extra_model', 'duplicate_model', 'duplicate_catalog', 'unknown_field', 'boolean_http',
    'unread_digest', 'future_time', 'naive_time', 'wrong_host', 'mutable_revision',
    'unsafe_path', 'duplicate_record', 'boolean_count', 'record_authority', 'foreign_source'])
def test_bad_input_fails_closed(fault):
    from szl_model_lab.evidence_view import project_lineage
    data = sample()
    row = data['models'][0]
    if fault == 'private': data['scope'] = 'authenticated_private'
    elif fault == 'authority': data['release_authorized'] = True
    elif fault == 'partial': data['complete_pagination'] = False
    elif fault == 'missing_model': data['models'].pop()
    elif fault == 'extra_model': data['catalog_ids'].pop()
    elif fault == 'duplicate_model': data['models'].append(copy.deepcopy(row))
    elif fault == 'duplicate_catalog': data['catalog_ids'].append(data['catalog_ids'][0])
    elif fault == 'unknown_field': row['prompt'] = 'must-not-leak'
    elif fault == 'boolean_http': row['card_http'] = True
    elif fault == 'unread_digest': row['card_http'] = 401
    elif fault == 'future_time': row['observed_at'] = '9999-01-01T00:00:00Z'
    elif fault == 'naive_time': row['observed_at'] = '2026-11-13T00:00:00'
    elif fault == 'wrong_host': row['evidence'][0]['url'] = 'https://example.invalid/eval.json'
    elif fault == 'mutable_revision': row['revision'] = 'main'
    elif fault == 'unsafe_path': row['evidence'][0]['path'] = '../secret'
    elif fault == 'duplicate_record': row['evidence'].append(copy.deepcopy(row['evidence'][0]))
    elif fault == 'boolean_count': row['recognized_artifact_count'] = False
    elif fault == 'record_authority': row['evidence'][0]['approved'] = True
    elif fault == 'foreign_source': row['source_route']['repository'] = 'foreign/repository'
    raw, digest = wire(data)
    with pytest.raises(ValueError):
        project_lineage(raw, '0' * 64 if fault == 'digest' else digest)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'\xff', b' ' * 4000001])
def test_bad_wire_is_rejected(raw):
    from szl_model_lab.evidence_view import project_lineage
    with pytest.raises(ValueError):
        project_lineage(raw, hashlib.sha256(raw).hexdigest())


def test_tampering_clears_response_and_hides_local_path(tmp_path):
    client, path = configured(tmp_path)
    path.write_bytes(path.read_bytes() + b' ')
    with client:
        for route in ('/evidence', '/api/evidence-lineage'):
            response = client.get(route, auth=AUTH)
            assert response.status_code == 503
            assert str(path) not in response.text and TOKEN not in response.text
            assert 'INVALID_OR_UNAVAILABLE_SNAPSHOT' in response.text


def test_no_side_effect_endpoints_or_arbitrary_selection(tmp_path):
    client, _ = configured(tmp_path)
    with client:
        assert client.post('/api/evidence-lineage', auth=AUTH, json={}).status_code == 405
        assert client.get('/api/evidence-lineage?model=../../secrets', auth=AUTH).status_code == 404
        assert client.get('/api/evidence-lineage?invalidate=' + '0'*64, auth=AUTH).status_code == 404
        assert client.get('/evidence', auth=AUTH, headers={'Host': 'evil.invalid'}).status_code == 400


def test_projection_never_attests_or_executes():
    with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
        result = project()
    for key in ('execution_authority', 'publication_authority', 'production_ready',
                'signature_verified', 'artifact_identity_verified', 'evaluation_verified'):
        assert result[key] is False
    assert result['hypothetical'] is False and result['affected_models'] is None


def test_real_canonical_coverage_contract(tmp_path):
    root = Path(__file__).resolve().parents[2]
    raw = (root / 'docs/model-card-proof-20261113/closure-input.json').read_bytes()
    from szl_model_lab.evidence_view import project_lineage
    result = project_lineage(raw, hashlib.sha256(raw).hexdigest())
    assert result['counts']['model_repositories'] == len(json.loads(raw)['models'])
    assert result['snapshot_hash_matched'] is True


@pytest.mark.parametrize('configuration', [
    {'evidence_snapshot': Path('unused')},
    {'evidence_snapshot_sha256': 'a' * 64},
    {'evidence_snapshot': 'not-a-path', 'evidence_snapshot_sha256': 'a' * 64},
    {'evidence_snapshot': Path('unused'), 'evidence_snapshot_sha256': 'main'},
])
def test_partial_or_malformed_settings_rejected(configuration):
    with pytest.raises(ValueError):
        Settings(TOKEN, {}, {}, **configuration)


def test_environment_configuration_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv('SZL_LAB_ACCESS_TOKEN', TOKEN)
    monkeypatch.setenv('SZL_LAB_EVIDENCE_SNAPSHOT', str(tmp_path / 'provided.json'))
    monkeypatch.setenv('SZL_LAB_EVIDENCE_SNAPSHOT_SHA256', 'a' * 64)
    value = Settings.from_env()
    assert value.evidence_snapshot == tmp_path / 'provided.json'
    assert value.evidence_snapshot_sha256 == 'a' * 64


def test_unauthenticated_calls_never_read_evidence(tmp_path):
    client, _ = configured(tmp_path)
    with client, patch('szl_model_lab.evidence_web.load_lineage', side_effect=AssertionError('not admitted')):
        for path in ('/evidence', '/api/evidence-lineage'):
            assert client.get(path).status_code == 401
            assert client.get(path, auth=('operator', 'wrong')).status_code == 401


def test_permuting_input_order_does_not_change_identity_or_projection():
    data = sample()
    baseline = project(data)
    data['catalog_ids'].reverse()
    data['models'].reverse()
    changed = project(data)
    assert changed['nodes'] == baseline['nodes'] and changed['edges'] == baseline['edges']
    assert changed['selected_model'] == baseline['selected_model']


def test_reported_free_text_is_not_rendered_or_executed(tmp_path):
    data = sample()
    row = data['models'][0]
    row['reported_github_links'] = ['https://example.invalid/PRIVATE-SENTINEL<script>']
    row['license_metadata'] = 'PRIVATE-SENTINEL<script>'
    client, _ = configured(tmp_path, data)
    with client:
        for route in ('/evidence', '/api/evidence-lineage'):
            response = client.get(route, auth=AUTH)
            assert response.status_code == 200
            assert 'PRIVATE-SENTINEL' not in response.text and '<script>' not in response.text
