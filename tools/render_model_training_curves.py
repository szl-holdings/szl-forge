# SPDX-License-Identifier: Apache-2.0
"""Render recorded training histories offline; never substitute evaluation claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from tools.model_card_evidence_passport import MODEL_ID, SHA40, SHA64, strict_json

SCOPE = 'reported checkpoint training loss, not evaluation; current released-weight binding not verified'


def validate_curves(curves: object) -> list[dict]:
    if not isinstance(curves, list) or not 1 <= len(curves) <= 100:
        raise ValueError('bounded nonempty curve list required')
    seen = set()
    for curve in curves:
        if not isinstance(curve, dict):
            raise ValueError('invalid curve record')
        model = curve.get('model')
        if not isinstance(model, str) or not MODEL_ID.fullmatch(model) or model.casefold() in seen:
            raise ValueError('invalid or duplicate model identity')
        seen.add(model.casefold())
        rev = curve.get('revision')
        digest = curve.get('sha256')
        if not isinstance(rev, str) or not SHA40.fullmatch(rev):
            raise ValueError('immutable receipt revision required')
        if not isinstance(digest, str) or not SHA64.fullmatch(digest):
            raise ValueError('retrieved receipt digest required')
        checkpoint = curve.get('checkpoint_global_step')
        if type(checkpoint) is not int or not 0 < checkpoint <= 100000000:
            raise ValueError('positive bounded checkpoint step required')
        receipt = f'checkpoint-{checkpoint}/trainer_state.json'
        if curve.get('receipt') != receipt:
            raise ValueError('checkpoint identity mismatch')
        if curve.get('url') != f'https://huggingface.co/{model}/raw/{rev}/{receipt}':
            raise ValueError('receipt URL identity mismatch')
        if curve.get('scope') != SCOPE:
            raise ValueError('training/evaluation boundary must be explicit')
        points = curve.get('points')
        if not isinstance(points, list) or not 2 <= len(points) <= 100000:
            raise ValueError('at least two recorded points required')
        previous = -1
        for point in points:
            if not isinstance(point, dict) or set(point) != {'step', 'loss'}:
                raise ValueError('exact step/loss fields required')
            step, loss = point['step'], point['loss']
            if type(step) is not int or not previous < step <= checkpoint:
                raise ValueError('unique increasing steps within checkpoint required')
            if type(loss) not in (int, float) or not math.isfinite(loss) or loss <= 0:
                raise ValueError('positive finite recorded loss required for log plot')
            previous = step
    return curves


def projection_digest(curve: dict) -> str:
    pairs = [[point['step'], point['loss']] for point in curve['points']]
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    curves = validate_curves(strict_json(args.input.read_text(encoding='utf-8')))
    report = {'training_only': True, 'new_model_evaluation': False,
              'records': [{'model': c['model'], 'points': len(c['points']),
                           'projection_sha256': projection_digest(c)} for c in curves]}
    if args.output_dir:
        # Plotting is optional; validation requires only the standard library.
        import matplotlib.pyplot as plt
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for curve in curves:
            points = curve['points']
            fig, ax = plt.subplots(figsize=(10, 5.8))
            ax.plot([p['step'] for p in points], [p['loss'] for p in points],
                    marker='.', linewidth=1.5)
            ax.set_yscale('log')
            ax.set_xlabel('Recorded optimizer step')
            ax.set_ylabel('Reported training loss - logarithmic scale')
            ax.set_title(f"{curve['model'].split('/')[1]}\nArchived checkpoint "
                         f"{curve['checkpoint_global_step']} - {len(points)} recorded points",
                         loc='left', fontsize=14, pad=16)
            ax.spines[['top', 'right']].set_visible(False)
            fig.text(0.08, 0.055,
                     'No smoothing or invented observations. Training loss is not held-out performance.\n'
                     'Current released-weight equivalence and production readiness are not established.', fontsize=9)
            fig.subplots_adjust(left=0.10, right=0.96, top=0.82, bottom=0.23)
            name = curve['model'].split('/')[1]
            # The restricted model-ID alphabet also prevents destination traversal.
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                raise ValueError('unsafe output name')
            for extension in ('png', 'svg'):
                fig.savefig(args.output_dir / f'{name}-training-history.{extension}', dpi=160)
            plt.close(fig)
        (args.output_dir/'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
