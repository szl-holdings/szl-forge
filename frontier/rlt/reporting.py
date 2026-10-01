"""Render one digest-verified synthetic evaluation; never convert it to promotion."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .runtime import canonical, digest
from .training import VARIANTS


def render(blob: bytes) -> str:
    if len(blob) > 2*1024*1024:
        raise ValueError('evaluation exceeds bound')
    report = json.loads(blob)
    if type(report) is not dict or report.get('schema') != 'szl.rlt.synthetic-training-evaluation.v1':
        raise ValueError('unsupported evaluation')
    stamp = report.pop('receipt_sha256',None)
    if stamp != digest(canonical(report)) or report.get('production_qualified') is not False or report.get('publication_eligible') is not False:
        raise ValueError('invalid digest or authority claim')
    rows=[]
    for run in report['runs']:
        if run['variant'] not in VARIANTS:
            raise ValueError('unknown variant')
        cells=[html.escape(run['variant']),str(int(run['seed'])),str(int(run['parameters']))]
        for length in ('16','32','64'):
            m=run['test'].get(length)
            if m is None:
                cells.append('Not measured')
            else:
                n,correct=m['programs'],m['final_correct']
                if type(n) is not int or type(correct) is not int or not 1<=n<=2048 or not 0<=correct<=n:
                    raise ValueError('invalid numerator/denominator')
                if m['final_accuracy'] != correct/n:
                    raise ValueError('accuracy differs from its counts')
                cells.append(f'{correct}/{n} · {100*correct/n:.1f}%')
        rows.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    # All strings derived from metadata are escaped. No JS, endpoints, CDN or telemetry.
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>SZL recurrent continuity — measured research</title>
<style>body{margin:0;background:#08101e;color:#e7f0fb;font:16px/1.6 system-ui}
main{max-width:1100px;margin:auto;padding:clamp(16px,4vw,48px)}
h1{font-size:clamp(28px,5vw,54px);line-height:1.15}.tag{color:#73e8d0;letter-spacing:.12em}
.card{border:1px solid #32465c;border-radius:14px;padding:20px;margin:24px 0;background:#101d2c}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:690px}
th,td{text-align:left;padding:12px;border-bottom:1px solid #32465c}th{color:#73e8d0}
code,pre{overflow-wrap:anywhere;white-space:pre-wrap}summary{cursor:pointer;padding:12px 0}
:focus-visible{outline:3px solid #73e8d0;outline-offset:4px}
@media(prefers-contrast:more){.card{border-color:white}}
</style><main><p class="tag">SZL / OUROBOROS CONTINUITY</p>
<h1>Trained. Measured.<br>Still research.</h1>
<p>A synthetic revision register learns SET0, SET1, FLIP, KEEP and REVOKE.
This is not a language model, autonomous agent, or production qualification.</p>
<section class="card"><h2>Every seed. Every comparison.</h2><p>Final-state accuracy on held-out programs.
Same examples and optimizer steps; parameters, time and FLOPs are not matched.</p>
<div class="scroll" role="region" aria-label="Evaluation results" tabindex="0"><table>
<thead><tr><th>Variant</th><th>Seed</th><th>Parameters</th><th>16 operations</th><th>32 operations</th><th>64 operations</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div></section>
<section class="card"><h2>Evidence boundary</h2><p>Public evidence handles bind cache validity.
Nemo checks metadata and authority. The independent register oracle checks this synthetic output.
None grants tool authority. Checkpoints and training evidence are different artifacts.</p>
<p>Report SHA-256: <code>'''+stamp+'''</code></p>
<details><summary>Inspect the complete measured receipt</summary><pre>'''+html.escape(json.dumps({**report,'receipt_sha256':stamp},indent=2))+'''</pre></details>
</section></main></html>'''


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.evaluation.stat().st_size>2*1024*1024:
        raise ValueError('evaluation exceeds bound')
    page=render(a.evaluation.read_bytes())
    with a.output.open('x',encoding='utf-8') as f:f.write(page)
    return 0


if __name__=='__main__':raise SystemExit(main())
