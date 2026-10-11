#!/usr/bin/env python3
"""Local presentation evidence only; all provider/effect requests are blocked."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import mimetypes
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
CHROMIUM = os.environ.get("CHROME_PATH")
WIDTHS = (320, 390, 768, 1280, 1920)
UNAVAILABLE = {'status': 'UNAVAILABLE', 'fixture': 'LOCAL_PRESENTATION_ONLY', 'ready': False,
               'failure_code': 'PRESENTATION_FIXTURE_NO_RUNTIME', 'gitSha': None, 'receipts': []}


def assets(root: Path, prefix: str = '') -> dict[str, tuple[bytes, str]]:
    return {prefix + '/' + p.relative_to(root).as_posix():
            (p.read_bytes(), mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            for p in root.rglob('*') if p.is_file() and
            not any(s.startswith('.') or s == '__pycache__' for s in p.relative_to(root).parts) and
            p.suffix in ('.css', '.js', '.svg', '.png', '.json', '.html')}


def cases():
    forge = REPO / 'spaces/szl-forge-lab'
    inference = REPO / 'spaces/szl-model-inference-lab'
    node = next(n for n in ast.parse((inference / 'app.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'index')
    inference_html = ast.literal_eval(node.body[0].value).encode()
    # Calling these two pure source renderers performs no provider or model work.
    sys.path.insert(0, str(REPO))
    candidate = importlib.import_module('model_candidates.workbench')
    model_root = REPO / 'model-lab/src/szl_model_lab'
    env = Environment(loader=FileSystemLoader(model_root / 'templates'),
                      autoescape=select_autoescape(['html']))
    model_html = env.get_template('index.html').render(
        tracks=[{'display_name': 'Browser fixture · advisory research',
                 'state': 'UNAVAILABLE', 'boundary': 'Local presentation fixture; no model loaded.',
                 'proposed_hf_id': None}],
        nodes=None, configured_nodes=0, pool={'state': 'UNAVAILABLE', 'nodes': None},
        corpus={'state': 'UNAVAILABLE', 'counts': None}).encode()
    rows = [
        ('forge-lab', (forge/'index.html').read_bytes(), assets(forge),
         'Exact static source and packaged public snapshot; Hub revision lookup unavailable fixture.'),
        ('inference-lab', inference_html, assets(inference/'szl', '/szl'),
         'Exact index() literal and assets; health, version and evidence unavailable fixtures; no lifespan/model load.'),
        ('clinical-gateway', (REPO/'clinical-gateway/frontend/index.html').read_bytes(),
         assets(REPO/'clinical-gateway/frontend'), 'Exact packaged UI; no operational buttons invoked.'),
        ('model-lab', model_html, assets(model_root/'static/szl', '/szl'),
         'Exact Jinja template and assets; explicitly unavailable presentation fixture; no app/auth/runtime instantiated.'),
        ('model-candidates', candidate.render_page('router').encode(),
         {**assets(REPO/'model_candidates/assets', '/assets'),
          '/assets/workbench.css': (candidate.CSS.encode(), 'text/css')},
         'Actual pure source renderer and blueprint declarations; no training or model imports.'),
        ('retrieval-preview', (REPO/'operational/evidence_lab/operational/retrieval_web/index.html').read_bytes(),
         assets(REPO/'operational/evidence_lab/operational/retrieval_web'),
         'Exact packaged UI; unavailable local-runtime fixture; no retrieval, generation or question submission.'),
        ('sample-console', (REPO/'frontier/sample-console/index.html').read_bytes(),
         assets(REPO/'frontier/sample-console'), 'Exact static sample console; built-in illustrative records unchanged.'),
    ]
    for name, html, files, scope in rows:
        files['/'] = (html, 'text/html; charset=utf-8')
        yield name, files, scope


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report = {'observed_at': datetime.now(timezone.utc).isoformat(), 'scope': 'LOCAL_PRESENTATION_ONLY',
              'deployed_runtime_verified': False, 'model_inference_performed': False,
              'operational_writes_performed': False, 'cases': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=CHROMIUM,
            args=['--disable-dev-shm-usage', '--use-gl=angle', '--use-angle=swiftshader',
                  '--enable-unsafe-swiftshader'])
        for name, files, scope in cases():
            requests, mutations, outside, errors = [], [], [], []
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    path = urlsplit(self.path).path
                    requests.append(path)
                    if path in files:
                        data, mime = files[path]; status = 200
                    else:
                        data, mime, status = json.dumps(UNAVAILABLE).encode(), 'application/json', 503
                    self.send_response(status)
                    self.send_header('Content-Type', mime)
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers(); self.wfile.write(data)
                def do_POST(self):
                    mutations.append(self.path); self.send_error(405)
                do_PUT = do_POST
                do_PATCH = do_POST
                do_DELETE = do_POST
                def log_message(self, *_): pass
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            origin = f'http://127.0.0.1:{server.server_port}'
            context = browser.new_context(reduced_motion='reduce')
            def route(r):
                if r.request.method != 'GET':
                    mutations.append(r.request.url); r.abort(); return
                if r.request.url.startswith(origin + '/'):
                    r.continue_(); return
                outside.append(r.request.url)
                r.fulfill(status=503, content_type='application/json', body=json.dumps(UNAVAILABLE))
            context.route('**/*', route)
            page = context.new_page(); page.on('pageerror', lambda e: errors.append(str(e)))
            page.set_default_timeout(20000)
            results = []
            for width in WIDTHS:
                page.set_viewport_size({'width': width, 'height': 1000})
                page.goto(origin+'/', wait_until='networkidle')
                page.evaluate('document.fonts.ready')
                page.keyboard.press('Tab')
                skip = page.evaluate("document.activeElement.matches('a.skip-link')")
                page.keyboard.press('Enter')
                focus = page.evaluate("""() => {
                  const skip = document.querySelector('a.skip-link');
                  const target = skip && document.querySelector(skip.getAttribute('href'));
                  return {target: target && target.id, focused: document.activeElement === target,
                          active: document.activeElement.tagName, top: target && target.getBoundingClientRect().top};
                }""")
                layout = page.evaluate("""() => {
                  const scrollWidth = document.documentElement.scrollWidth;
                  const visible = [...document.querySelectorAll('button:not(:disabled),input:not([type=checkbox]),select,summary,.nav__link,.btn')]
                    .filter(e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden');
                  const small = visible.filter(e => e.getBoundingClientRect().height < 43.5)
                    .map(e => ({text:(e.textContent||e.getAttribute('aria-label')||e.id).trim().slice(0,50),height:e.getBoundingClientRect().height}));
                  return {scrollWidth, width:innerWidth, small,
                    surface:getComputedStyle(document.body).backgroundColor,
                    h1:document.querySelector('h1')?.textContent.trim(),
                    styles:[...document.styleSheets].map(s => s.href).filter(Boolean)};
                }""")
                results.append({'width': width, 'skip_first': skip, 'skip_focus': focus,
                                'layout': layout, 'contained': layout['scrollWidth'] <= width+1})
                if width in (320, 1280):
                    page.evaluate('window.scrollTo(0,0)')
                    page.screenshot(path=str(output/f'{name}-{width}.png'), full_page=False, timeout=60000)
            page.emulate_media(forced_colors='active', reduced_motion='reduce')
            page.locator('a.skip-link').focus()
            forced = page.locator('a.skip-link').evaluate("e=>({outline:getComputedStyle(e).outlineStyle,width:getComputedStyle(e).outlineWidth,color:getComputedStyle(e).outlineColor})")
            record = {'name': name, 'scope': scope, 'widths': results, 'forced_focus': forced,
                      'page_errors': errors, 'blocked_external_gets': sorted(set(outside)),
                      'mutations': mutations, 'served_requests': sorted(set(requests)),
                      'source_html_sha256': hashlib.sha256(files['/'][0]).hexdigest(),
                      'asset_hashes': {k:hashlib.sha256(v[0]).hexdigest() for k,v in files.items() if k.endswith('.css')}}
            record['passed'] = (all(r['skip_first'] and r['skip_focus']['focused'] and r['contained']
                                    and not r['layout']['small'] for r in results)
                                and forced['outline'] != 'none' and not errors and not mutations)
            report['cases'].append(record)
            print(json.dumps({'name':name,'passed':record['passed'],'errors':errors,
                              'failures':[r for r in results if not r['skip_focus']['focused'] or not r['contained'] or r['layout']['small']]}),flush=True)
            context.close(); server.shutdown(); server.server_close(); thread.join()
        browser.close()
    report['passed'] = all(c['passed'] for c in report['cases'])
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
