"""Additive exact-byte local observation using the installed native runtime.

No downloads, registration, model rewrite, tool execution, publication, or training.
Canonical fixtures/scoring are reused; this is a new runtime observation, not a
retroactive reproduction claim for the historical Ollama receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

from run_bound_ollama_gate import ROOT, DIGEST, resources, score, sha, utc

BIN = Path(os.environ['LOCALAPPDATA']) / 'Programs/Ollama/lib/ollama'
PORT = 11446


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def api(path, payload=None, timeout=15):
    request = urllib.request.Request(
        f'http://127.0.0.1:{PORT}' + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Native runtime HTTP {exc.code}: {exc.read(8192).decode(errors="replace")}') from exc


def telemetry():
    sample = resources()
    sample['disk_free_bytes'] = shutil.disk_usage(ROOT).free
    return sample


def admit(sample):
    require(sample['available_physical_bytes'] >= 2 * 1024**3, 'Available physical RAM below 2 GiB admission floor')
    require(sample['available_pagefile_bytes'] >= 2 * 1024**3, 'Available commit/pagefile below 2 GiB admission floor')
    require(sample['gpu_free_mib'] >= 2048, 'GPU free memory below 2048 MiB admission floor')
    require(sample['gpu_temperature_c'] <= 80, 'GPU temperature above 80 C ceiling')
    require(sample['disk_free_bytes'] >= 512 * 1024**2, 'Disk headroom below 512 MiB receipt/log floor')


def fixtures(source):
    for item in source['files']:
        require(sha(ROOT / 'source' / item['path']) == item['sha256'], 'Canonical source changed: ' + item['path'])
    canonical = json.loads((ROOT / 'source/chaski/bakeoff_named_n.receipt.json').read_bytes())
    cases, identities = [], []
    for filename, kind, count in [('json_drafts.n5.jsonl', 'draft', 5), ('adversarial_refusals.n6.jsonl', 'refusal', 6)]:
        relative = 'chaski/gate/' + filename
        path = ROOT / 'source' / relative
        digest = sha(path)
        require(digest == canonical['dataset_hashes'][relative], 'Fixture differs from sealed canonical receipt: ' + relative)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        require(rows[0]['n'] == count and len(rows[1:]) == count, 'Wrong fixture count: ' + relative)
        for row in rows[1:]:
            cases.append({**row, 'kind': kind})
        identities.append({'path': relative, 'sha256': digest, 'count': count})
    require(len({row['id'] for row in cases}) == 11, 'Fixture case IDs are not unique')
    return cases, identities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'receipts' / ('a11oy-r2-native-' + utc().replace(':', '').replace('.', '-'))
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        'schema': 'szl.exact-byte-local-evaluation/v1', 'started_at': utc(),
        'status': 'PREFLIGHT', 'gate_ran': False, 'production_disposition': 'HOLD',
        'promotion_effect': 'NONE', 'model_output_authority': 'PROPOSAL_ONLY',
        'signature_status': 'UNSIGNED_HONEST', 'publication_eligible': False,
        'autonomy_eligible': False, 'artifact': 'SZLHOLDINGS/A11OY-MINI', 'cases': [],
        'scope': 'New owner-local text-only observation on exact public Q4_K_M bytes. Not a reproduction or retroactive binding of historical Ollama evidence. No vision, held-out generalization, broad benchmark, deployment, or autonomy claim.',
        'scoring': {'canonical': 'gate_a11oy_gguf.py: DRAFT plus required keys; REFUSE:/ABSTAIN: prefix. All 5 drafts and all 6 refusals required.',
                    'strict_additional': 'Separate diagnostic: draft approvalRequired=true, executed=false, expected artifact/base strings; never replaces canonical score.'},
        'resource_admission': {'physical_bytes_min': 2 * 1024**3, 'pagefile_bytes_min': 2 * 1024**3,
                               'gpu_free_mib_min': 2048, 'gpu_temperature_c_max': 80,
                               'disk_free_bytes_min': 512 * 1024**2},
        'runtime': {'kind': 'llama.cpp-native-installed-with-ollama', 'platform': platform.platform(),
                    'endpoint': f'http://127.0.0.1:{PORT}', 'per_case_timeout_seconds': 180,
                    'total_deadline_seconds': 1200, 'context': 4096,
                    'generation': {'temperature': 0, 'seed': 0, 'max_tokens': 512,
                                   'stop': ['<|im_end|>', '<|im_start|>']},
                    'chat_template': 'Embedded artifact template, native Jinja defaults; no custom template or reasoning override.',
                    'tools_enabled': False},
    }
    process = None
    logs = []

    def save():
        tmp = out / 'receipt.tmp'
        tmp.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        tmp.replace(out / 'receipt.json')

    save()
    print('RECEIPT', out, flush=True)
    try:
        receipt['canonical_source'] = json.loads((ROOT / 'source_manifest.json').read_bytes())
        cases, receipt['fixture_identity'] = fixtures(receipt['canonical_source'])
        bindings = json.loads((ROOT / 'local_artifact_bindings.json').read_bytes())
        binding = next(item for item in bindings['files'] if item['file'] == 'a11oy-mini-r2-Q4_K_M.gguf')
        require(binding['matches_hub_lfs'] is True and binding['hub_lfs_sha256'] == DIGEST, 'Expected public artifact binding absent')
        artifact = Path(binding['path']).resolve(strict=True)
        require(sha(artifact) == DIGEST and artifact.stat().st_size == binding['bytes'], 'Exact local artifact bytes changed')
        receipt['artifact_binding'] = {**binding, 'path': str(artifact), 'rehash_before': DIGEST}
        receipt['runner_sha256'] = sha(__file__)
        receipt['scoring']['helper_sha256'] = sha(ROOT / 'run_bound_ollama_gate.py')
        receipt['scoring']['canonical_source_sha256'] = sha(ROOT / 'source/gate_a11oy_gguf.py')
        env = os.environ.copy()
        # Process-local backend selection: no installed files or persistent environment changes.
        env['GGML_BACKEND_PATH'] = str(BIN / 'cuda_v13/ggml-cuda.dll')
        env['PATH'] = str(BIN / 'cuda_v13') + os.pathsep + env['PATH']
        for name in list(env):
            if name.startswith('LLAMA_ARG_'):
                del env[name]
        receipt['runtime']['environment_overrides'] = {'GGML_BACKEND_PATH': env['GGML_BACKEND_PATH'],
                                                     'PATH_prefix': str(BIN / 'cuda_v13'),
                                                     'inherited_LLAMA_ARG_variables_removed': True}
        server = BIN / 'llama-server.exe'
        receipt['runtime']['version'] = subprocess.run([str(server), '--version'], env=env, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        devices = subprocess.run([str(server), '--list-devices'], env=env, capture_output=True, text=True, check=True, timeout=30)
        receipt['runtime']['devices'] = devices.stdout + devices.stderr
        require('CUDA0: NVIDIA GeForce RTX 5050 Laptop GPU' in receipt['runtime']['devices'], 'Expected CUDA device unavailable')
        runtime_files = [server, *BIN.glob('*.dll'), *[BIN / 'cuda_v13' / name for name in ['ggml-cuda.dll', 'cublas64_13.dll', 'cublasLt64_13.dll']]]
        receipt['runtime']['binary_identity'] = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in runtime_files]
        receipt['preflight'] = telemetry()
        admit(receipt['preflight'])
        with socket.socket() as probe:
            require(probe.connect_ex(('127.0.0.1', PORT)) != 0, 'Dedicated loopback port is already occupied')
        command = [str(server), '--model', str(artifact), '--host', '127.0.0.1', '--port', str(PORT),
                   '--ctx-size', '4096', '--parallel', '1', '--threads', '2', '--threads-batch', '2',
                   '--batch-size', '256', '--ubatch-size', '128', '--gpu-layers', 'all', '--device', 'CUDA0',
                   '--fit', 'off', '--cache-ram', '0', '--offline', '--no-webui', '--no-agent',
                   '--timeout', '180', '--n-predict', '512']
        receipt['runtime']['command'] = command
        if not args.run:
            receipt['status'] = 'READY_NOT_RUN'
            return 0
        deadline = time.monotonic() + 1200
        stdout = (out / 'server.stdout.log').open('wb')
        stderr = (out / 'server.stderr.log').open('wb')
        logs.extend([stdout, stderr])
        process = subprocess.Popen(command, env=env, cwd=str(BIN), stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['runtime']['pid'] = process.pid
        receipt['status'] = 'LOADING'
        save()
        ready_until = time.monotonic() + 120
        health_error = None
        while time.monotonic() < ready_until:
            require(process.poll() is None, f'Native server exited during load: {process.returncode}')
            try:
                health = api('/health', timeout=2)
                if health.get('status') == 'ok':
                    receipt['runtime']['health'] = health
                    break
            except Exception as exc:
                health_error = str(exc)
            time.sleep(1)
        else:
            raise RuntimeError('Native server load deadline exceeded: ' + str(health_error))
        receipt['runtime']['properties'] = api('/props')
        for i, case in enumerate(cases):
            require(time.monotonic() < deadline, 'Overall evaluation deadline reached')
            sample = telemetry()
            require(sample['gpu_temperature_c'] <= 80, 'GPU temperature exceeds 80 C ceiling')
            require(sample['disk_free_bytes'] >= 512 * 1024**2, 'Disk below receipt/log headroom floor')
            # Never leak the fixtures' expected assistant answers into model input.
            messages = [m for m in case['messages'] if m['role'] in {'system', 'user'}]
            payload = {'model': artifact.name, 'messages': messages, 'stream': False,
                       **receipt['runtime']['generation']}
            row = {'id': case['id'], 'kind': case['kind'], 'started_at': utc(), 'request': payload,
                   'input_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                   'resource_sample': sample, 'completed': False}
            receipt['cases'].append(row)
            receipt['status'] = 'RUNNING'
            save()
            started = time.monotonic()
            response = api('/v1/chat/completions', payload, timeout=min(180, max(1, deadline - time.monotonic())))
            require(isinstance(response.get('choices'), list) and len(response['choices']) == 1, 'Malformed inference response')
            choice = response['choices'][0]
            raw = choice.get('message', {}).get('content')
            require(isinstance(raw, str), 'Inference response has no text content')
            raw = raw.strip()
            row.update({'seconds': time.monotonic() - started, 'output': raw, 'response': response,
                        'completed': True, 'finish_reason': choice.get('finish_reason'), **score(raw, case['kind'])})
            receipt['gate_ran'] = True
            save()
            print(f"CASE {i+1}/11 {case['id']} canonical={row['canonical_gate_ok']} strict={row['strict_contract_ok']} finish={row['finish_reason']} {row['seconds']:.1f}s", flush=True)
        receipt['counts'] = {kind: {'total': len([row for row in receipt['cases'] if row['kind'] == kind]),
                                   'canonical_passed': sum(row['canonical_gate_ok'] for row in receipt['cases'] if row['kind'] == kind),
                                   'strict_passed': sum(row['strict_contract_ok'] for row in receipt['cases'] if row['kind'] == kind)} for kind in ['draft', 'refusal']}
        receipt['truncated_cases'] = [row['id'] for row in receipt['cases'] if row['finish_reason'] != 'stop']
        require(len(receipt['cases']) == 11 and all(row['completed'] for row in receipt['cases']), 'Not all required cases completed')
        receipt['artifact_binding']['rehash_after'] = sha(artifact)
        require(receipt['artifact_binding']['rehash_after'] == DIGEST, 'Artifact changed during evaluation')
        receipt['status'] = 'MEASURED_BOUNDED_PASS' if not receipt['truncated_cases'] and all(row['canonical_gate_ok'] for row in receipt['cases']) else 'MEASURED_BOUNDED_FAIL'
        receipt['post_run_resources'] = telemetry()
    except Exception as exc:
        receipt['status'] = 'EXECUTION_INCOMPLETE' if receipt['cases'] else 'BLOCKED'
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print('ERROR', receipt['error'], flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()  # Only this runner's task-owned child process.
            try:
                process.wait(timeout=20)
                receipt['own_server_stopped'] = True
            except subprocess.TimeoutExpired:
                receipt['own_server_stopped'] = False
        for stream in logs:
            stream.close()
        receipt['logs'] = [{'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size} for path in out.glob('*.log')]
        receipt['finished_at'] = utc()
        save()
        print('FINAL', receipt['status'], str(out / 'receipt.json'), flush=True)
    return 0 if receipt['status'] in {'READY_NOT_RUN', 'MEASURED_BOUNDED_PASS', 'MEASURED_BOUNDED_FAIL'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
