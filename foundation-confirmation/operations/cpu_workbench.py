"""Run the immutable local workbench with an admitted CPU environment.

All three learned selectors execute before the listener is constructed. Startup
probes do not mint user receipts and do not qualify the FAILED scientific gate.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

HELPER_SHA256 = 'eef44b5bca9ab1dfe4b84a74aac855d1a70196cccf01fb4b0987013368751628'
FROZEN = {
    'release-manifest.json': '03a13779b09f2e8ad3dd53d928ac460a395d329742f9f43f7e5892fba672c877',
    'verify_release.py': 'b7b5d7775fadc6933144c16e17db2ee64e63c18b43d412a593eba927f53672f9',
    'server.py': 'ffa50276f1956f78dc9738a36f1855e95a71c1621c52e6b1c8a6a54ddcbc55fb',
    'start.ps1': 'cb49ca92d5a207a1a25ac64a9a5a7bf313b62fd0f293a8be603ddee73b07ff78',
    'core.py': '573740e2f3e138b82508b49e29ad55c11e2a6e29c07fd44fe49f973628632c69',
    'policy.py': '02705491ed34ca23ef2e900768ad33dc96903c16729a9981ad72475bf4379919',
    'experiment.py': 'a763286c6e29dd041ff8e09a3c49f6376e3c08383d5be38788d4b81f65854a11',
}
ARCHIVE_SHA256 = '869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_admission():
    path = Path(__file__).with_name('cpu_environment.py')
    return load_exact_module('szl_cpu_environment', path, HELPER_SHA256)


def load_exact_module(name, path, expected_sha256):
    require(name not in sys.modules, 'A conflicting sealed module is already loaded: ' + name)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Exact sealed module bytes differ: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Execute the very bytes admitted above. SourceFileLoader may otherwise
        # read an excluded timestamp-valid .pyc even with -B.
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def require_separate_state(state, lab):
    original = lab / 'state/trials'
    require(state != lab and not state.is_relative_to(original) and not original.is_relative_to(state), 'CPU service state must be separate from frozen trial receipts')


def startup_timeout(seconds):
    try:
        sys.stderr.write(json.dumps({'schema': 'szl.foundation-confirmation.cpu-startup-failure/v1', 'state': 'FAILED', 'reason': 'STARTUP_DEADLINE_EXCEEDED', 'observed_utc': now(), 'timeout_seconds': seconds}) + '\n')
        sys.stderr.flush()
    finally:
        os._exit(124)  # Only this newly launched child; logging cannot prevent it.


def verify_frozen(helper, lab, archive):
    lab = helper.profile_path(lab)
    archive = helper.profile_path(archive, directory=False)
    helper.no_links(lab, recursive=True)
    require(helper.digest_file(archive) == ARCHIVE_SHA256, 'Original sealed archive differs')
    for name, expected in FROZEN.items():
        require(helper.digest_file(lab / name) == expected, 'Frozen execution boundary differs: ' + name)
    result = subprocess.run([str(helper.native_image()), '-I', '-S', '-B', str(lab / 'verify_release.py'), '--root', str(lab)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    require(result.returncode == 0, 'Frozen stdlib verifier failed')
    verification = json.loads(result.stdout)
    require(verification['status'] == 'VERIFIED' and verification['files'] == 70 and verification['checkpoints'] == 3 and verification['evaluation_rows'] == 5184, 'Frozen release admission is incomplete')
    for name, expected in FROZEN.items():
        require(helper.digest_file(lab / name) == expected, 'Frozen source changed during admission: ' + name)
    return lab, verification


def preload(runtime):
    before = runtime.admitted()
    probes = []
    for seed in (17, 23, 41):
        request = {'seed': 20261002, 'index': 10007, 'family': 'shared_bias', 'policy': 'learned', 'model_seed': seed}
        started = now()
        result, binding = runtime.run(request)
        checkpoint = before['checkpoints'][seed]
        expected = {key: before[key] for key in ('protocol_sha256', 'training_summary_sha256', 'data_receipt_sha256', 'source_sha256')}
        expected.update(checkpoint_sha256=checkpoint['sha256'], model_fingerprint=checkpoint['trained_fingerprint'])
        require(binding == expected and result['policy'] == 'learned' and isinstance(result['history'], list), 'Actual startup selector binding/trace differs')
        probes.append({'model_seed': seed, 'request': request, 'started_utc': started, 'completed_utc': now(), 'binding': binding, 'result_sha256': digest(result), 'receipt_minted': False})
    require(runtime.admitted() == before, 'Source/checkpoint binding changed during startup probes')
    status = runtime.status()
    require(status['ready'] is True and status['state'] == 'READY' and status['checkpoints_verified'] == 3 and status['models_loaded'] == [17, 23, 41], 'Actual all-three model startup readiness failed')
    return probes


def process_created():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    times = [ctypes.c_ulonglong() for _ in range(4)]
    kernel.GetProcessTimes.argtypes = (ctypes.c_void_p, *(ctypes.POINTER(ctypes.c_ulonglong) for _ in range(4)))
    require(kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(value) for value in times)) != 0, 'Native creation time is unavailable')
    # Exact equality at CIM's declared microsecond precision, no tolerance.
    epoch = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc)
    return (epoch + dt.timedelta(microseconds=times[0].value // 10)).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def prior_process_absent(pid):
    require(type(pid) is int and pid > 0, 'Prior service PID is invalid')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        kernel.CloseHandle(handle)
        return False
    require(ctypes.get_last_error() == 87, 'Prior service absence could not be established')
    return True


def publish_service(path, receipt):
    if path.exists():
        previous = json.loads(path.read_bytes().decode('utf-8-sig'))
        contract = ('schema', 'environment_binding_sha256', 'server_script', 'lab_root', 'port', 'executable', 'python_image_sha256', 'state_directory', 'environment_root', 'wheelhouse', 'argv', 'startup_timeout_seconds')
        require(all(previous.get(key) == receipt[key] for key in contract) and prior_process_absent(previous.get('pid')), 'Existing CPU service receipt is not an absent admitted predecessor')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(canonical(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lab-root', 'archive', 'environment-root', 'wheelhouse', 'expected-python-sha256', 'environment-binding-sha256', 'state-directory'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--port', type=int, default=18767)
    parser.add_argument('--startup-timeout-seconds', type=int, default=180)
    args = parser.parse_args()
    require(1 <= args.port <= 65535 and 1 <= args.startup_timeout_seconds <= 180, 'Port/startup deadline is outside the bounded contract')
    require(re.fullmatch(r'[0-9a-f]{64}', args.environment_binding_sha256) is not None, 'Incomplete environment binding')
    started = time.monotonic()
    timer = threading.Timer(args.startup_timeout_seconds, startup_timeout, args=(args.startup_timeout_seconds,))
    timer.daemon = True
    timer.start()
    helper = load_admission()
    admission = helper.admit(args.environment_root, args.wheelhouse, args.expected_python_sha256)
    require(admission['binding_sha256'] == args.environment_binding_sha256, 'Installed CPU environment binding changed')
    lab, source_admission = verify_frozen(helper, args.lab_root, args.archive)
    state = helper.profile_path(args.state_directory)
    require_separate_state(state, lab)
    helper.no_links(state, recursive=True)
    helper.activate(admission)
    os.chdir(lab)
    require(not any(Path(value).resolve() == lab for value in sys.path if value), 'The frozen lab must never be a general import directory')
    import torch
    require(torch.__version__ == '2.10.0+cpu' and torch.version.cuda is None and Path(torch.__file__).resolve().is_relative_to(Path(admission['binding']['site_packages'])), 'Actual CPU Torch identity/import path differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    modules = {name: load_exact_module(name, lab / (name + '.py'), FROZEN[name + '.py']) for name in ('core', 'policy', 'experiment')}
    server = load_exact_module('szl_sealed_confirmation_server', lab / 'server.py', FROZEN['server.py'])
    runtime = server.ConfirmationRuntime(lab)
    def load_preverified_modules(admitted):
        require(all(sys.modules.get(name) is module for name, module in modules.items()), 'A sealed module object changed')
        runtime.policy = modules['policy']
        runtime.experiment = modules['experiment']
        runtime.loaded_sources = admitted['source_sha256']
    runtime._load_modules = load_preverified_modules
    probes = preload(runtime)
    original_status = runtime.status
    def status():
        value = original_status()
        value.update(startup_execution_probes=probes, startup_probes_are_user_receipts=False, environment_binding_sha256=admission['binding_sha256'], runtime_kind='ADMITTED_CPU_ONLY', receipt_minted=False)
        if value.get('models_loaded') != [17, 23, 41] or value.get('checkpoints_verified') != 3:
            value.update(ready=False, state='UNAVAILABLE', error='Actual startup model readiness was not retained')
        return value
    runtime.status = status
    # No listener exists until admitted imports and every actual probe pass.
    with server.ConfirmationServer(('127.0.0.1', args.port), root=lab, runtime=runtime, state_directory=state / 'trials') as web:
        receipt = {'schema': 'szl.foundation-confirmation.cpu-service/v1', 'pid': os.getpid(), 'executable': admission['binding']['native_executable'], 'process_created': process_created(), 'launched_utc': now(), 'server_script': str(Path(__file__).resolve()), 'lab_root': str(lab), 'port': args.port, 'state_directory': str(state), 'environment_root': admission['binding']['environment_root'], 'wheelhouse': admission['binding']['wheelhouse'], 'python_image_sha256': admission['binding']['python_image_sha256'], 'environment_binding_sha256': admission['binding_sha256'], 'source_admission': source_admission, 'startup_execution_probes': probes, 'models_loaded': [17, 23, 41], 'checkpoints_verified': 3, 'startup_elapsed_seconds': round(time.monotonic() - started, 6), 'startup_timeout_seconds': args.startup_timeout_seconds, 'argv': sys.argv, 'receipt_minted': False, 'registered_scientific_gate': 'FAILED'}
        publish_service(state / 'service.json', receipt)
        timer.cancel()
        print(json.dumps({'schema': receipt['schema'], 'state': 'READY', 'pid': receipt['pid'], 'models_loaded': receipt['models_loaded'], 'environment_binding_sha256': admission['binding_sha256'], 'startup_elapsed_seconds': receipt['startup_elapsed_seconds']}), flush=True)
        web.serve_forever()


if __name__ == '__main__':
    main()
