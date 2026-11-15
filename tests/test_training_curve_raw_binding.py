# SPDX-License-Identifier: Apache-2.0
"""Every CLI trajectory must be re-derived from its digest-bound raw trainer state."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import render_model_training_curves as curves

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/model-card-proof-20261113/training-curves.json'
RAW = DATA.parent / 'training-raw'


def _fixture(tmp_path):
    state = {'global_step': 20, 'log_history': [
        {'step': 10, 'loss': 2.0, 'epoch': 1.0},
        {'step': 20, 'loss': 1.0, 'epoch': 2.0},
        {'step': 20, 'train_loss': 1.5}]}
    raw = json.dumps(state).encode()
    digest = hashlib.sha256(raw).hexdigest()
    model = 'SZLHOLDINGS/example'
    row = {'model': model, 'revision': 'a' * 40, 'sha256': digest,
           'checkpoint_global_step': 20, 'receipt': 'checkpoint-20/trainer_state.json',
           'url': f'https://huggingface.co/{model}/raw/{"a" * 40}/checkpoint-20/trainer_state.json',
           'scope': curves.SCOPE, 'points': [{'step': 10, 'loss': 2.0}, {'step': 20, 'loss': 1.0}]}
    source = tmp_path / 'training-raw'
    source.mkdir()
    (source / f'{digest}.json').write_bytes(raw)
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([row]), encoding='utf-8')
    return row, path, source


def test_cli_refuses_plausible_but_changed_loss_before_any_output(tmp_path):
    row, path, _ = _fixture(tmp_path)
    row['points'][0]['loss'] = 1.99
    path.write_text(json.dumps([row]), encoding='utf-8')
    output = tmp_path / 'graphs'
    with pytest.raises(ValueError, match='projection'):
        curves.main(['--input', str(path), '--output-dir', str(output)])
    assert not output.exists()


def test_cli_requires_raw_evidence_even_in_validation_only_mode(tmp_path):
    _, path, rawdir = _fixture(tmp_path)
    for child in rawdir.iterdir():
        child.unlink()
    with pytest.raises(ValueError, match='raw'):
        curves.main(['--input', str(path)])


def test_cli_rejects_tampered_raw_bytes(tmp_path):
    _, path, rawdir = _fixture(tmp_path)
    target = next(rawdir.iterdir())
    target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises(ValueError, match='digest'):
        curves.main(['--input', str(path)])


def test_cli_attests_projection_not_just_manifest_syntax(tmp_path, capsys):
    _, path, _ = _fixture(tmp_path)
    assert curves.main(['--input', str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['raw_evidence_verified'] is True
    assert report['records'][0]['raw_projection_equal'] is True
    assert report['new_model_evaluation'] is False


@pytest.mark.parametrize('fault', ['checkpoint', 'duplicate_step', 'boolean_step',
                                  'boolean_loss', 'nan_loss', 'no_loss', 'not_object'])
def test_raw_history_semantics_are_checked_even_with_matching_digest(tmp_path, fault):
    row, path, rawdir = _fixture(tmp_path)
    original = next(rawdir.iterdir())
    state = json.loads(original.read_bytes())
    if fault == 'checkpoint':
        state['global_step'] = 21
    elif fault == 'duplicate_step':
        state['log_history'][1]['step'] = 10
    elif fault == 'boolean_step':
        state['log_history'][0]['step'] = True
    elif fault == 'boolean_loss':
        state['log_history'][0]['loss'] = True
    elif fault == 'nan_loss':
        state['log_history'][0]['loss'] = float('nan')
    elif fault == 'no_loss':
        state['log_history'] = [{'step': 20, 'train_loss': 1.0}]
    else:
        state = []
    raw = json.dumps(state).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (rawdir / f'{digest}.json').write_bytes(raw)
    row['sha256'] = digest
    path.write_text(json.dumps([row]), encoding='utf-8')
    with pytest.raises(ValueError):
        curves.main(['--input', str(path)])


def test_all_four_archived_histories_bind_exactly(capsys):
    assert curves.main(['--input', str(DATA)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['raw_evidence_verified'] is True
    assert sum(r['points'] for r in report['records']) == 139
    assert len(report['records']) == 4
    assert all(r['raw_projection_equal'] for r in report['records'])


def test_binding_refuses_symlink_evidence(tmp_path):
    row, _, rawdir = _fixture(tmp_path)
    target = next(rawdir.iterdir())
    original = tmp_path / 'outside.json'
    target.rename(original)
    try:
        target.symlink_to(original)
    except OSError:
        pytest.skip('platform does not grant symlink creation')
    with pytest.raises(ValueError, match='symlink'):
        curves.verify_raw_evidence([row], rawdir)


def test_projection_edit_does_not_rewrite_original_receipt(tmp_path):
    row, _, rawdir = _fixture(tmp_path)
    path = next(rawdir.iterdir())
    before = path.read_bytes()
    changed = copy.deepcopy(row)
    changed['points'][1]['loss'] = 1.01
    with pytest.raises(ValueError, match='projection'):
        curves.verify_raw_evidence([changed], rawdir)
    assert path.read_bytes() == before


@pytest.mark.parametrize('raw', [b'{"global_step":20,"global_step":20,"log_history":[]}',
                                 b'{"global_step":20,"log_history":[],"x":1e999}',
                                 b'\xff', b' ' * 2_000_001],
                         ids=['duplicate-key', 'overflow-number', 'invalid-utf8', 'oversized'])
def test_duplicate_nonfinite_invalid_utf8_and_oversized_raw_refused(tmp_path, raw):
    row, path, rawdir = _fixture(tmp_path)
    digest = hashlib.sha256(raw).hexdigest()
    (rawdir / f'{digest}.json').write_bytes(raw)
    row['sha256'] = digest
    path.write_text(json.dumps([row]), encoding='utf-8')
    with pytest.raises(ValueError):
        curves.main(['--input', str(path)])


def test_missing_raw_directory_refused(tmp_path):
    row, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match='raw evidence directory'):
        curves.verify_raw_evidence([row], tmp_path / 'missing')


def test_explicit_evidence_directory_is_supported(tmp_path, capsys):
    _, path, rawdir = _fixture(tmp_path)
    moved = tmp_path / 'retained'
    rawdir.rename(moved)
    assert curves.main(['--input', str(path), '--evidence-dir', str(moved)]) == 0
    assert json.loads(capsys.readouterr().out)['raw_evidence_verified'] is True
