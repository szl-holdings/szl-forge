#!/usr/bin/env python3
"""Invoke only A11oy's existing estate workflow; verify its native receipt.

Stdlib Python 3.11+. Default read-only. No token export, alternate publisher,
workflow edit, automatic retry, model execution, or production authorization.
Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = 'szl-holdings/a11oy'
REPO_ID = 1225834126
WORKFLOW_PATH = '.github/workflows/estate-release-train.yml'
PREFIX = f'repos/{REPO}'
DISPATCH = f'{PREFIX}/actions/workflows/estate-release-train.yml/dispatches'
API_VERSION = '2026-03-10'
ACTIVE_STATES = ('queued', 'in_progress', 'waiting', 'requested', 'pending')
WRITERS = frozenset((WORKFLOW_PATH, '.github/workflows/hf-sync.yml', '.github/workflows/repair-cloudflare-product-edge.yml'))
PINS = {
    WORKFLOW_PATH: '57ca410488dda97af6650e162ed26334317f90dc',
    '.github/workflows/hf-sync.yml': 'ff2f4a948f6251545f96ebb4db14a665e45e4d72',
    'scripts/estate_repair_dispatch.py': '81f0e83729390ee66b633a8d45e5f1139374ba41',
    'config/estate-release-train.v1.json': '84e998eca4f2f4b6cabc992f12a1c466875be848',
}
REQUIRED = ('a11oy', 'killinchu', 'lyte', 'vertical-services', 'terra', 'counsel', 'finance')
VECTOR_REPOS = {'a11oy': 'a11oy', 'killinchu': 'killinchu', 'lyte': 'lyte-services', 'vertical-services': 'vertical-services', 'terra': 'a11oy', 'counsel': 'a11oy', 'finance': 'a11oy', 'proof': 'a11oy-net', 'profile': '.github'}
SCOPE = '9060fa8d7edcd5c246b86bcfcf1916df44b18038253325336f2f46208f8001ae'
MAX_BYTES = 8 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024


class OperatorError(ValueError):
    """Unavailable, inconsistent, or unapproved input cannot qualify a repair."""


def need(condition, message):
    if not condition:
        raise OperatorError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def object_json(raw):
    need(len(raw) <= MAX_BYTES, 'JSON exceeds bound')
    def pairs(items):
        obj = {}
        for key, value in items:
            need(key not in obj, 'duplicate JSON key')
            obj[key] = value
        return obj
    def number(text):
        value = float(text)
        need(math.isfinite(value), 'non-finite number')
        return value
    def constant(text):
        raise OperatorError('non-JSON constant')
    try:
        obj = json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise OperatorError('invalid JSON object') from exc
    need(type(obj) is dict, 'expected JSON object')
    return obj


class GHClient:
    """Bounded, non-shell use of the owner's installed GitHub CLI."""
    def __init__(self, executable=None):
        self.executable = executable or shutil.which('gh')
        need(bool(self.executable), 'Authenticated GitHub CLI (gh) is required')
        need(Path(self.executable).suffix.lower() not in ('.cmd', '.bat'), 'native gh executable required')

    def request(self, endpoint, payload=None, binary=False):
        allowed = endpoint.startswith(PREFIX + '/') or endpoint == PREFIX
        allowed = allowed or endpoint in {f'repos/szl-holdings/{repo}/branches/main' for repo in VECTOR_REPOS.values()}
        need(allowed and '\n' not in endpoint and '\r' not in endpoint, 'unapproved GitHub endpoint')
        need(payload is None or endpoint == DISPATCH, 'only native estate dispatch may mutate')
        argv = [self.executable, 'api', '--hostname', 'github.com', '--method', 'POST' if payload is not None else 'GET',
                '-H', 'Accept: application/vnd.github+json', '-H', f'X-GitHub-Api-Version: {API_VERSION}', endpoint]
        encoded = b''
        if payload is not None:
            encoded = json.dumps(payload, allow_nan=False).encode('utf-8')
            argv += ['--input', '-']
        env = dict(os.environ, GH_PROMPT_DISABLED='1', GH_PAGER='cat', NO_COLOR='1', GIT_TERMINAL_PROMPT='0')
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as incoming:
            incoming.write(encoded); incoming.seek(0)
            with subprocess.Popen(argv, stdin=incoming, stdout=output, stderr=errors, env=env, shell=False) as child:
                deadline = time.monotonic() + 120
                while child.poll() is None:
                    if time.monotonic() > deadline or os.fstat(output.fileno()).st_size > MAX_BYTES or os.fstat(errors.fileno()).st_size > 65536:
                        child.kill(); child.wait()
                        raise OperatorError('GitHub request exceeded time/output bound')
                    time.sleep(0.05)
                need(child.returncode == 0, f'GitHub request failed (exit {child.returncode}); inspect gh authentication/Actions without exporting tokens')
            output.seek(0); raw = output.read(MAX_BYTES + 1)
        need(len(raw) <= MAX_BYTES, 'GitHub response exceeds bound')
        return raw if binary else (object_json(raw) if raw.strip() else None)


def check_source(gh, source):
    need(type(source) is str and re.fullmatch(r'[0-9a-f]{40}', source), 'exact lowercase source SHA required')
    branch = gh.request(f'{PREFIX}/branches/main')
    need(type(branch) is dict, 'branch response unavailable')
    need(branch.get('name') == 'main' and branch.get('protected') is True, 'protected main not established')
    need(branch.get('commit', {}).get('sha') == source, 'approved source is no longer current main')


def check_idle(gh):
    for state in ACTIVE_STATES:
        page = gh.request(f'{PREFIX}/actions/runs?branch=main&status={state}&per_page=100')
        need(type(page) is dict, 'active-run response unavailable')
        total = page.get('total_count')
        runs = page.get('workflow_runs')
        need(type(total) is int and type(runs) is list and total == len(runs) and total <= 100, 'incomplete active-run enumeration')
        for run in runs:
            need(type(run) is dict and isinstance(run.get('path'), str), 'invalid active run')
            need(run['path'].split('@')[0] not in WRITERS, 'canonical writer/controller is active; do not start a competing request')


def preflight(gh, source):
    check_source(gh, source)
    for path, pin in PINS.items():
        item = gh.request(f'{PREFIX}/contents/{path}?ref={source}')
        need(type(item) is dict, 'control response unavailable')
        need(item.get('type') == 'file' and item.get('path') == path and item.get('sha') == pin, 'native control identity changed: ' + path)
        need(item.get('encoding') == 'base64', 'native control bytes unavailable')
        try:
            raw = base64.b64decode(''.join(item['content'].split()), validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise OperatorError('native control encoding invalid') from exc
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw, usedforsecurity=False).hexdigest()
        need(len(raw) <= 128 * 1024 and blob == pin, 'native control bytes differ from Git identity')
    workflow = gh.request(f'{PREFIX}/actions/workflows/estate-release-train.yml')
    need(type(workflow) is dict, 'workflow response unavailable')
    need(workflow.get('state') == 'active' and workflow.get('path') == WORKFLOW_PATH, 'canonical workflow unavailable')
    check_idle(gh)
    check_source(gh, source)
    return {'state': 'PREFLIGHT_ONLY', 'source': source, 'production_authorization': False}


def dispatch(gh, source, journal):
    """One mutation attempt. Durable source-scoped journal forbids blind retries."""
    payload = {'ref': 'main', 'inputs': {'repair': True, 'publish_vertical_flagships': True,
               'vertical_plan_json': json.dumps({'complete': True, 'approved': True, 'expected_source_revision': source}, sort_keys=True)}}
    result = {'state': 'STOPPED_BEFORE_DISPATCH', 'source': source, 'production_authorization': False}
    with journal.open('x', encoding='utf-8') as handle:
        def record(event):
            handle.write(json.dumps(dict(event, observed_at=datetime.now(timezone.utc).isoformat()), allow_nan=False) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        attempted = False
        try:
            check_source(gh, source)
            record(dict(result, state='DISPATCH_ATTEMPTED', request=payload))
            attempted = True
            response = gh.request(DISPATCH, payload)
            need(type(response) is dict, 'dispatch response has no exact run ID')
            run_id = response.get('workflow_run_id')
            need(type(run_id) is int and run_id > 0, 'invalid dispatch run ID')
            need(response.get('run_url') == f'https://api.github.com/{PREFIX}/actions/runs/{run_id}', 'dispatch API URL mismatch')
            need(response.get('html_url') == f'https://github.com/{REPO}/actions/runs/{run_id}', 'dispatch UI URL mismatch')
            result.update(state='DISPATCH_ACCEPTED_NOT_VERIFIED', run_id=run_id, url=response['html_url'])
        except Exception as exc:
            result.update(state='UNKNOWN_AFTER_ATTEMPT' if attempted else 'STOPPED_BEFORE_DISPATCH', error_type=type(exc).__name__)
        record(result)
    return result


def validate_run(run, run_id, source):
    need(type(run) is dict, 'run response unavailable')
    need(type(run.get('id')) is int and run['id'] == run_id, 'run ID mismatch')
    need(run.get('head_sha') == source and run.get('head_branch') == 'main', 'run source mismatch')
    need(run.get('event') == 'workflow_dispatch' and run.get('path') == WORKFLOW_PATH, 'run event/workflow mismatch')
    need(run.get('repository', {}).get('id') == REPO_ID, 'run repository mismatch')
    need(type(run.get('run_attempt')) is int and run['run_attempt'] > 0, 'invalid run attempt')


def inspect_archive(raw, source):
    """Validate native receipt integrity, not independent probe execution."""
    need(len(raw) <= MAX_BYTES, 'archive exceeds bound')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        need(0 < len(members) <= 32 and len({m.filename for m in members}) == len(members), 'invalid archive inventory')
        need(sum(m.file_size for m in members) <= MAX_EXPANDED, 'expanded archive exceeds bound')
        for m in members:
            name = m.filename
            need(m.orig_filename == name and re.fullmatch(r'[A-Za-z0-9_./-]+', name) and all(p not in ('', '.', '..') for p in name.split('/')), 'unsafe archive path')
            need(not m.is_dir() and not m.flag_bits & 1 and stat.S_IFMT(m.external_attr >> 16) in (0, stat.S_IFREG), 'nonregular/encrypted archive member')
            need(m.file_size <= MAX_BYTES and m.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'unadmitted member encoding/size')
        def one(basename):
            found = [m for m in members if m.filename.split('/')[-1] == basename]
            need(len(found) == 1, 'missing/ambiguous native receipt: ' + basename)
            return archive.read(found[0])
        estate_raw = one('estate-release-train.json')
        estate = object_json(estate_raw)
        inventory = object_json(one('hf-public-inventory-preflight.json'))
    need(estate.get('schema') == 'szl.estate-release-train.receipt/v1' and inventory.get('schema') == 'szl.public-inventory-preflight/v1', 'native schema mismatch')
    need(estate.get('source_vector', {}).get('a11oy') == source, 'estate source mismatch')
    need(inventory.get('checkout_source_revision') == source and inventory.get('estate_source_revision') == source, 'inventory source mismatch')
    need(inventory.get('estate_receipt_sha256') == digest(estate_raw), 'inventory/estate byte link mismatch')
    scope = inventory.get('scope')
    need(type(scope) is dict and digest(json.dumps(scope, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) == SCOPE, 'inventory scope content changed')
    record = {k: v for k, v in inventory.items() if k != 'record_sha256'}
    need(inventory.get('record_sha256') == digest(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()), 'inventory record digest mismatch')
    need(inventory.get('scope_sha256') == SCOPE and inventory.get('scope', {}).get('id') == 'hf-public-author-membership/v1', 'inventory scope mismatch')
    need(estate.get('authority', {}).get('production_authorization') is False and inventory.get('production_authorization') is False, 'native receipt improperly grants production authority')
    blockers = estate.get('blockers')
    need(type(blockers) is list and all(type(x) is str for x in blockers), 'invalid native blockers')
    complete = estate.get('state') == 'ALIGNED' and inventory.get('state') == 'ALIGNED'
    if complete:
        need(not blockers and inventory.get('comparison', {}).get('aligned') is True and inventory['comparison'].get('blockers') == [], 'contradictory aligned receipt')
        components = estate.get('components')
        need(type(components) is list, 'missing component vector')
        keys = [x.get('key') for x in components if type(x) is dict]
        need(len(keys) == len(components) and len(set(keys)) == len(keys), 'duplicate/invalid component')
        for key in REQUIRED:
            matches = [x for x in components if x.get('key') == key]
            need(len(matches) == 1, 'missing required component: ' + key)
            component = matches[0]
            need(component.get('required') is True and component.get('aligned') is True and component.get('blockers') == [], 'required component is not aligned: ' + key)
            revision = estate['source_vector'].get(key)
            need(isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision), 'missing component source')
            need(component.get('source', {}).get('observed') is True and component['source'].get('sha') == revision, 'component source mismatch')
            need(component.get('runtime', {}).get('observed') is True and component['runtime'].get('revision') == revision, 'component runtime mismatch')
        for key in ('product', 'proof', 'profile_inventory'):
            need(estate.get(key, {}).get('aligned') is True and estate[key].get('blockers') == [], 'native product/proof/profile gap')
        need(estate['product'].get('semantic_parity') is True and estate['product'].get('domain_source', {}).get('revision') == source, 'product source/semantic gap')
        for kind in ('models', 'datasets', 'spaces'):
            need(inventory['comparison'].get('delta', {}).get(kind) == {'state': 'MATCH', 'added': [], 'removed': []}, 'inventory item delta')
    return {'state': 'NATIVE_ALIGNMENT_RECEIPT_VERIFIED' if complete else 'NATIVE_RECEIPT_VERIFIED_HOLD',
            'source': source, 'source_vector': estate['source_vector'], 'observed_at': estate.get('observed_at'),
            'blockers': blockers, 'inventory_state': inventory.get('state'), 'artifact_sha256': digest(raw),
            'independent_live_probe_replay': False, 'production_authorization': False}


def observe(gh, source, run_id, wait=False):
    deadline = time.monotonic() + 3900
    while True:
        run = gh.request(f'{PREFIX}/actions/runs/{run_id}')
        validate_run(run, run_id, source)
        if run.get('status') == 'completed':
            break
        if not wait or time.monotonic() >= deadline:
            return {'state': 'RUN_PENDING_HOLD', 'run_id': run_id, 'production_authorization': False}
        print(f'Native estate run {run_id}: {run.get("status")}', flush=True)
        time.sleep(20)
    page = gh.request(f'{PREFIX}/actions/runs/{run_id}/artifacts?per_page=100')
    need(type(page) is dict, 'artifact response unavailable')
    items = page.get('artifacts')
    need(type(items) is list and type(page.get('total_count')) is int and page['total_count'] == len(items), 'incomplete artifact enumeration')
    name = f'estate-release-train-{run_id}-{run["run_attempt"]}'
    found = [x for x in items if type(x) is dict and x.get('name') == name]
    need(len(found) == 1, 'native artifact missing or ambiguous')
    item = found[0]; artifact_id = item.get('id'); size = item.get('size_in_bytes')
    need(type(artifact_id) is int and artifact_id > 0 and type(size) is int and 0 < size <= MAX_BYTES and item.get('expired') is False, 'invalid/expired artifact')
    need(item.get('workflow_run', {}).get('id') == run_id and item['workflow_run'].get('head_sha') == source and item['workflow_run'].get('repository_id') == REPO_ID, 'artifact source/run mismatch')
    expected = item.get('digest', '')
    need(isinstance(expected, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', expected), 'missing native artifact digest')
    raw = gh.request(f'{PREFIX}/actions/artifacts/{artifact_id}/zip', binary=True)
    need(len(raw) == size and 'sha256:' + digest(raw) == expected, 'downloaded artifact differs from GitHub digest/size')
    result = inspect_archive(raw, source)
    result.update(run_id=run_id, run_attempt=run['run_attempt'], run_conclusion=run.get('conclusion'))
    # Bind observation time to this native run, not a copied earlier receipt.
    def clock(value):
        need(type(value) is str, 'missing native observation time')
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        need(parsed.tzinfo is not None, 'naive native clock')
        return parsed
    need(clock(run['created_at']) <= clock(result['observed_at']) <= clock(run['updated_at']), 'receipt observation is outside its native run')
    if result['state'] == 'NATIVE_ALIGNMENT_RECEIPT_VERIFIED':
        need(run.get('conclusion') == 'success', 'receipt cannot override failed native gate')
        # Refresh the required GitHub authority vector; a concurrent source advance
        # cannot be silently assigned the previous observation's result.
        for key, repo in VECTOR_REPOS.items():
            branch = gh.request(f'repos/szl-holdings/{repo}/branches/main')
            need(branch.get('commit', {}).get('sha') == result['source_vector'].get(key), 'authority vector moved: ' + key)
    check_source(gh, source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-source', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='approve this exact native repair, not blanket production')
    mode.add_argument('--inspect-run', type=int, help='read-only recovery; never dispatch or replay')
    parser.add_argument('--wait', action='store_true', help='wait synchronously up to 65 minutes')
    args = parser.parse_args()
    result = {'state': 'HOLD', 'production_authorization': False}
    try:
        gh = GHClient()
        if args.inspect_run is not None:
            need(args.inspect_run > 0, 'positive run ID required')
            result = observe(gh, args.expected_source, args.inspect_run, args.wait)
        else:
            result = preflight(gh, args.expected_source)
            if args.execute:
                directory = Path.home() / '.szl-estate-operator'
                need(not directory.is_symlink(), 'journal directory must not be a symlink')
                directory.mkdir(mode=0o700, exist_ok=True)
                journal = directory / (args.expected_source + '.attempt.jsonl')
                result = dispatch(gh, args.expected_source, journal)
                print(json.dumps(dict(result, journal=str(journal)), indent=2), flush=True)
                if 'run_id' in result and args.wait:
                    result = observe(gh, args.expected_source, result['run_id'], True)
    except (OperatorError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        result = dict(result, state='HOLD', error_type=type(exc).__name__, message=(str(exc) if isinstance(exc, OperatorError) else 'Inspect native run/journal before retry; no automatic retry was issued.'))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['state'] in ('PREFLIGHT_ONLY', 'NATIVE_ALIGNMENT_RECEIPT_VERIFIED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
