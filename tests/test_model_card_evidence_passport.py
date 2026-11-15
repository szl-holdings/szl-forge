# SPDX-License-Identifier: Apache-2.0
"""Offline coverage, preservation and malformed-evidence regression tests."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from tools import model_card_evidence_passport as panel

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs' / 'model-card-proof-20261113'


def record():
    return {'id': 'SZLHOLDINGS/test-model', 'revision': 'a' * 40,
            'observed_at': '2026-11-13T22:18:00+00:00', 'card_http': 200,
            'card_sha256': 'b' * 64, 'recognized_artifact_count': 1,
            'source_route': None, 'evidence': []}


def evidence():
    return {'path': 'training_receipt.json', 'http': 200, 'sha256': 'c' * 64,
            'url': 'https://huggingface.co/SZLHOLDINGS/test-model/raw/'
                   + 'a' * 40 + '/training_receipt.json'}


def manifest():
    return {'schema': 'szl.card-evidence-coverage/v1', 'scope': 'anonymous_public',
            'complete_pagination': True, 'catalog_observed_at': '2026-11-13T23:00:00Z',
            'private_inventory': 'UNQUALIFIED', 'release_authorized': False,
            'catalog_ids': ['SZLHOLDINGS/test-model'], 'models': [record()]}


def source():
    return (b'---\nlicense: apache-2.0\nszl:\n  publication_eligible: false\n---\n'
            b'\n# Research\n\nRelease blocked; keep failure visible.\n\n'
            + panel.ANCHOR + b'\nEvidence bytes remain intact.\n</details>\n')


def test_render_does_not_grant_authority_or_fill_unknown_with_zero():
    r = record()
    r['recognized_artifact_count'] = None
    result = panel.render(r)
    assert 'UNOBSERVED' in result
    assert 'not a zero score' in result
    assert 'No new training' in result
    assert 'Publisher mapping is not established' in result
    assert '/raw/main/' not in result


def test_reject_boolean_file_count():
    r = record()
    r['recognized_artifact_count'] = False
    with pytest.raises(ValueError):
        panel.validate(r)


def test_unread_card_must_not_have_observed_digest():
    r = record()
    r['card_http'] = 401
    with pytest.raises(ValueError):
        panel.validate(r)


def test_duplicate_evidence_is_rejected():
    r = record()
    r['evidence'] = [evidence(), evidence()]
    with pytest.raises(ValueError):
        panel.validate(r)


def test_control_character_evidence_is_rejected():
    r = record()
    e = evidence()
    e['path'] = 'receipt\ninjected.json'
    e['url'] = e['url'].replace('training_receipt.json', 'receipt%0Ainjected.json')
    r['evidence'] = [e]
    with pytest.raises(ValueError):
        panel.validate(r)


@pytest.mark.parametrize('value', ['', 'main', 'A' * 40, 'a' * 39, None, 12])
def test_revision_must_be_immutable(value):
    r = record()
    r['revision'] = value
    with pytest.raises(ValueError):
        panel.validate(r)


@pytest.mark.parametrize('path', ['../x', '/x', 'a//b', 'a/../b', 'a\\b',
                                 'a%2fb', 'x|y', 'x[0]', 'x`y', '<script>', '.'])
def test_invalid_paths_fail_closed(path):
    with pytest.raises(ValueError):
        panel.safe_path(path)


@pytest.mark.parametrize('replacement', ['https://evil.invalid/r',
                                         'https://huggingface.co/other/test/raw/' + 'a'*40 + '/training_receipt.json',
                                         'https://huggingface.co/SZLHOLDINGS/test-model/raw/main/training_receipt.json'])
def test_evidence_cannot_change_host_repo_or_revision(replacement):
    r = record()
    e = evidence()
    e['url'] = replacement
    r['evidence'] = [e]
    with pytest.raises(ValueError):
        panel.validate(r)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e309}'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):
        panel.strict_json(raw)


@pytest.mark.parametrize('kind', ['missing_record', 'duplicate_record', 'duplicate_catalog',
                                 'extra_record', 'incomplete_page', 'authority', 'private_scope'])
def test_coverage_has_no_silent_omissions(kind):
    d = manifest()
    if kind == 'missing_record':
        d['models'] = []
    elif kind == 'duplicate_record':
        d['models'].append(copy.deepcopy(d['models'][0]))
    elif kind == 'duplicate_catalog':
        d['catalog_ids'].append(d['catalog_ids'][0])
    elif kind == 'extra_record':
        extra = record()
        extra['id'] = 'SZLHOLDINGS/extra'
        d['models'].append(extra)
    elif kind == 'incomplete_page':
        d['complete_pagination'] = False
    elif kind == 'authority':
        d['release_authorized'] = True
    else:
        d['private_inventory'] = 'COMPLETE'
    with pytest.raises(ValueError):
        panel.validate_manifest(d)


def test_insert_preserves_every_unmanaged_byte_and_is_idempotent():
    old = source()
    new = panel.insert_panel(old, record(), panel.digest(old))
    assert panel.strip_panel(new) == old
    assert panel.insert_panel(new, record(), panel.digest(old)) == new
    assert new.split(b'\n---\n', 1)[0] == old.split(b'\n---\n', 1)[0]
    assert new.index(b'Release blocked') < new.index(panel.START.encode())


def test_changed_source_is_never_overwritten():
    old = source()
    with pytest.raises(ValueError):
        panel.insert_panel(old + b'other writer\n', record(), panel.digest(old))


@pytest.mark.parametrize('marker', [panel.START, panel.END, panel.START + panel.START])
def test_partial_or_duplicate_markers_fail(marker):
    with pytest.raises(ValueError):
        panel.strip_panel(source() + marker.encode())


def test_no_yaml_no_anchor_or_duplicate_anchor_rejected():
    for old in [b'# Missing YAML\n', source().replace(panel.ANCHOR, b''), source()+panel.ANCHOR]:
        with pytest.raises(ValueError):
            panel.insert_panel(old, record(), panel.digest(old))


def test_renderer_has_no_network_or_process_dependency():
    tree = ast.parse((ROOT/'tools/model_card_evidence_passport.py').read_text())
    modules = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    modules |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not (modules & {'requests', 'httpx', 'huggingface_hub', 'socket', 'subprocess'})


def test_committed_catalog_coverage_and_generated_index_agree():
    data = panel.strict_json((DOCS/'closure-input.json').read_text(encoding='utf-8'))
    panel.validate_manifest(data)
    assert (DOCS/'COVERAGE.md').read_text(encoding='utf-8') == panel.coverage_markdown(data)
    catalog = json.loads((DOCS/'catalog-identities.json').read_text(encoding='utf-8'))
    assert data['catalog_ids'] == catalog['ids']
    assert data['catalog_observed_at'] == catalog['observed_at']


def test_six_source_cards_preserve_baselines_and_exact_generated_panels():
    data = json.loads((DOCS/'closure-input.json').read_text(encoding='utf-8'))
    baselines = json.loads((DOCS/'source-preservation.json').read_text(encoding='utf-8'))
    records = {r['id']: r for r in data['models']}
    assert len(baselines['cards']) == 6
    for row in baselines['cards']:
        raw = (ROOT/row['path']).read_bytes()
        unmanaged = panel.strip_panel(raw)
        assert panel.digest(unmanaged) == row['unmanaged_sha256']
        assert raw == panel.insert_panel(raw, records[row['id']], row['unmanaged_sha256'])
        assert panel.digest(unmanaged.split(b'\n---\n', 1)[0]) == row['yaml_prefix_sha256']
        assert b'publication_eligible: false' in raw


@pytest.mark.parametrize('slug', ['CON', 'nul.txt', 'LPT1', 'coverage', 'tail.'])
def test_cross_platform_output_names_are_safe(slug):
    d = manifest()
    d['catalog_ids'] = ['SZLHOLDINGS/' + slug]
    d['models'][0]['id'] = d['catalog_ids'][0]
    with pytest.raises(ValueError):
        panel.validate_manifest(d)


def test_case_insensitive_filename_collision_is_rejected():
    d = manifest()
    second = copy.deepcopy(d['models'][0])
    second['id'] = 'SZLHOLDINGS/TEST-MODEL'
    d['catalog_ids'].append(second['id'])
    d['models'].append(second)
    with pytest.raises(ValueError):
        panel.validate_manifest(d)


@pytest.mark.parametrize('fault', ['model', 'source_path', 'publisher'])
def test_publisher_route_must_match_exact_model(fault):
    r = record()
    r['id'] = 'SZLHOLDINGS/KHIPU-R2'
    r['source_route'] = {'repository': 'szl-holdings/szl-forge', 'revision': 'a'*40,
                         'path': 'khipu-r2/card/README.md', 'publisher': 'publish-khipu-card.yml'}
    if fault == 'model':
        r['id'] = 'SZLHOLDINGS/chaski'
    elif fault == 'source_path':
        r['source_route']['path'] = 'khipu/card/README.md'
    else:
        r['source_route']['publisher'] = 'publish-chaski-card.yml'
    with pytest.raises(ValueError):
        panel.validate(r)


def test_route_review_covers_catalog_without_reassigning_kernel_owners():
    review = json.loads((DOCS/'publication-route-review.json').read_text())
    data = json.loads((DOCS/'closure-input.json').read_text())
    assert sorted(r['id'] for r in review['rows']) == sorted(data['catalog_ids'])
    policy = json.loads((ROOT/'publishing/kernel-hub-ownership.json').read_text())
    owners = {r['repo_id']: r['owner'] for r in policy['targets']}
    for row in review['rows']:
        if row['id'] in owners:
            assert row['state'] == 'EXTERNAL_KERNEL_OWNER_DECLARED'
            assert row['owner'] == owners[row['id']]
