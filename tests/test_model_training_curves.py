# SPDX-License-Identifier: Apache-2.0
"""Recorded training trajectories cannot become synthetic or unbound score plots."""
import copy
import json
from pathlib import Path

import pytest

from tools import render_model_training_curves as curves

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/model-card-proof-20261113/training-curves.json'
EXPECTED = {
    'SZLHOLDINGS/szl-triage-qwen3.5-0.8b-lora-study5': 'f107248cc86555c38f142ff41f59864c946f2c87286052953c2f4e2b0171d118',
    'SZLHOLDINGS/abstention-v2-candidate': 'b76f3ede0c4785ce4c29e33f4539acf0a7db39f414960e6eb98bbc432fc9142f',
    'SZLHOLDINGS/physics-master-v1-candidate': '93af4a1718b34e553bd312a870f015ce97e5877648bd718f6e8c33058b695f46',
    'SZLHOLDINGS/quant-master-v1-candidate': 'c38d5629cb0c64fe52fb9541a2eec4ac21344c1eeb90d182967f231dd33a66ba',
}


def test_exact_recorded_points_and_digests():
    rows = curves.validate_curves(json.loads(DATA.read_text(encoding='utf-8')))
    assert {r['model']: curves.projection_digest(r) for r in rows} == EXPECTED
    assert sum(len(r['points']) for r in rows) == 139


@pytest.mark.parametrize('fault', ['mutable_revision', 'duplicate_model', 'different_host',
                                 'duplicate_step', 'reversed_steps', 'beyond_checkpoint',
                                 'boolean_step', 'boolean_loss', 'nan_loss', 'zero_loss',
                                 'negative_loss', 'missing_boundary', 'missing_point'])
def test_invalid_curve_input_is_rejected(fault):
    rows = json.loads(DATA.read_text(encoding='utf-8'))
    if fault == 'mutable_revision':
        rows[0]['revision'] = 'main'
    elif fault == 'duplicate_model':
        rows.append(copy.deepcopy(rows[0]))
    elif fault == 'different_host':
        rows[0]['url'] = rows[0]['url'].replace('huggingface.co', 'example.invalid')
    elif fault == 'duplicate_step':
        rows[0]['points'][1]['step'] = rows[0]['points'][0]['step']
    elif fault == 'reversed_steps':
        rows[0]['points'].reverse()
    elif fault == 'beyond_checkpoint':
        rows[0]['points'][-1]['step'] = 999999
    elif fault == 'boolean_step':
        rows[0]['points'][0]['step'] = True
    elif fault == 'boolean_loss':
        rows[0]['points'][0]['loss'] = True
    elif fault == 'nan_loss':
        rows[0]['points'][0]['loss'] = float('nan')
    elif fault == 'zero_loss':
        rows[0]['points'][0]['loss'] = 0
    elif fault == 'negative_loss':
        rows[0]['points'][0]['loss'] = -1
    elif fault == 'missing_boundary':
        rows[0]['scope'] = 'held-out benchmark'
    else:
        rows[0]['points'] = rows[0]['points'][:1]
    with pytest.raises(ValueError):
        curves.validate_curves(rows)


def test_changed_recorded_number_is_detectable():
    rows = json.loads(DATA.read_text(encoding='utf-8'))
    rows[0]['points'][0]['loss'] += 0.01
    assert curves.projection_digest(rows[0]) != EXPECTED[rows[0]['model']]


def test_validation_does_not_require_plotting_dependencies(capsys):
    assert curves.main(['--input', str(DATA)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['new_model_evaluation'] is False
    assert len(report['records']) == 4
