"""Network-free stdlib-only CPU admission controls; no model import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
OPS = Path(__file__).resolve().parents[1] / 'foundation-confirmation/operations'
def module(name):
    path = OPS / (name + '.py')
    spec = importlib.util.spec_from_file_location('test_' + name, path)
    value = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), value.__dict__)
    return value
ENV, RUNNER = module('cpu_environment'), module('cpu_workbench')
def payload(root, extra=None):
    environment, wheels = root / 'env', root / 'wheels'
    site = environment / 'Lib/site-packages'
    site.mkdir(parents=True)
    wheels.mkdir()
    original = {'example/__init__.py': b'VALUE = 1\n', 'example-1.0.dist-info/METADATA': b'Name: example\nVersion: 1.0\n', 'example-1.0.dist-info/RECORD': b''}
    files = {**original, **(extra or {})}
    wheel = wheels / 'example-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, raw in files.items():
            info = zipfile.ZipInfo('safe')
            info.filename = name  # Keep invalid raw names; Windows ZIP helpers normalize them.
            archive.writestr(info, raw)
    for name, raw in original.items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = {'name': 'example', 'version': '1.0', 'filename': wheel.name, 'sha256': ENV.digest_file(wheel), 'download_bytes': wheel.stat().st_size, 'expanded_bytes': sum(map(len, files.values()))}
    return environment, wheels, site, row
class Runtime:
    def __init__(self):
        self.loaded, self.bad_binding = [], False
        self.checkpoints = {s: {'sha256': str(s).zfill(64), 'trained_fingerprint': str(s+1).zfill(64)} for s in (17,23,41)}
    def admitted(self):
        return {'protocol_sha256': '1'*64, 'training_summary_sha256': '2'*64, 'data_receipt_sha256': '3'*64, 'source_sha256': {'core.py': '4'*64}, 'checkpoints': self.checkpoints}
    def run(self, request):
        seed = request['model_seed']
        self.loaded.append(seed)
        binding = {k:v for k,v in self.admitted().items() if k != 'checkpoints'}
        binding.update(checkpoint_sha256=self.checkpoints[seed]['sha256'], model_fingerprint=self.checkpoints[seed]['trained_fingerprint'])
        if self.bad_binding: binding['model_fingerprint'] = '0'*64
        return {'policy':'learned', 'history':[{'actual-call':True}]}, binding
    def status(self): return {'ready':True, 'state':'READY', 'checkpoints_verified':3, 'models_loaded':sorted(self.loaded)}
class CpuControls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='szl-cpu-control-')
        self.root = Path(temp.name)
        self.addCleanup(temp.cleanup)
    def test_official_payload_tampering(self):
        env, wh, site, row = payload(self.root)
        self.assertEqual(ENV.verify_payload(env, wh, [row])['files_verified'], 2)
        (site/'example/__init__.py').write_bytes(b'VALUE = 2\n')
        with self.assertRaisesRegex(ValueError,'payload differs'): ENV.verify_payload(env, wh, [row])
    def test_wheel_tampering(self):
        env, wh, _, row = payload(self.root)
        with (wh/row['filename']).open('ab') as f: f.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'Official wheel bytes differ'): ENV.verify_payload(env, wh, [row])
    def test_unsafe_paths(self):
        for i,name in enumerate(['../escape.py','/escape.py','C:/escape.py','example\\escape.py','CON.py','example/name.']):
            with self.subTest(name=name):
                env,wh,_,row = payload(self.root/str(i),{name:b'x'})
                with self.assertRaisesRegex(ValueError,'Unsafe'): ENV.verify_payload(env,wh,[row])
    def test_case_collision(self):
        env,wh,_,row = payload(self.root,{'EXAMPLE/__init__.py':b'bad'})
        with self.assertRaisesRegex(ValueError,'case-folded'): ENV.verify_payload(env,wh,[row])
    def test_unknown_source_hooks_extensions_and_cache(self):
        for i,name in enumerate(['injected.py','injected.pyd','sitecustomize.py','injected.pth','__pycache__/injected.cpython-311.pyc']):
            with self.subTest(name=name):
                env,wh,site,row = payload(self.root/str(i))
                target=site/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError,'Unexpected importable'): ENV.verify_payload(env,wh,[row])
    def test_missing_payload(self):
        env,wh,site,row = payload(self.root)
        (site/'example/__init__.py').unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'): ENV.verify_payload(env,wh,[row])
    def test_nonempty_installer_marker(self):
        env,wh,site,row = payload(self.root)
        (site/'example-1.0.dist-info/REQUESTED').write_bytes(b'code')
        with self.assertRaisesRegex(ValueError,'REQUESTED marker differs'): ENV.verify_payload(env,wh,[row])
    def test_existing_global_site(self):
        with mock.patch.object(sys,'path',['stdlib','global/site-packages']):
            with self.assertRaisesRegex(ValueError,'already active'): ENV.activate({'state':'VERIFIED','binding':{'site_packages':'owned/site-packages'}})
    def test_lock_and_helper_binding(self):
        lock=ENV.read_lock()
        self.assertEqual(len(lock['wheels']),11)
        self.assertEqual(lock['torch_version'],'2.10.0+cpu')
        self.assertEqual(ENV.digest_file(OPS/'cpu_environment.py'),RUNNER.HELPER_SHA256)
    def test_all_three_actual_calls_no_receipts(self):
        runtime=Runtime()
        probes=RUNNER.preload(runtime)
        self.assertEqual(runtime.loaded,[17,23,41])
        self.assertTrue(all(p['receipt_minted'] is False for p in probes))
    def test_wrong_actual_binding(self):
        runtime=Runtime(); runtime.bad_binding=True
        with self.assertRaisesRegex(ValueError,'binding/trace differs'): RUNNER.preload(runtime)
    def test_checkpoint_only_readiness(self):
        runtime=Runtime()
        with mock.patch.object(runtime,'status',return_value={'ready':True,'state':'READY','checkpoints_verified':3,'models_loaded':[]}):
            with self.assertRaisesRegex(ValueError,'all-three model startup readiness'): RUNNER.preload(runtime)
    def test_state_ancestor_alias(self):
        lab=self.root/'lab'
        for state in [lab,lab/'state',lab/'state/trials',lab/'state/trials/child',self.root]:
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError,'separate from frozen'): RUNNER.require_separate_state(state,lab)
        RUNNER.require_separate_state(lab/'state/cpu-runtime',lab)
    def test_source_beats_valid_stale_cache(self):
        path=self.root/'sealed.py'; cached=b"VALUE='CACHED_CODE'\n"; sealed=b"VALUE='SEALED_TEXT'\n"
        self.assertEqual(len(cached),len(sealed)); path.write_bytes(cached); stamp=path.stat()
        py_compile.compile(str(path),doraise=True)
        path.write_bytes(sealed); os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        name='szl_cpu_cache_control'; self.addCleanup(sys.modules.pop,name,None)
        value=RUNNER.load_exact_module(name,path,hashlib.sha256(sealed).hexdigest())
        self.assertEqual(value.VALUE,'SEALED_TEXT')
    def test_conflicting_and_wrong_source(self):
        path=self.root/'sealed.py'; raw=b'VALUE=1\n'; path.write_bytes(raw)
        with mock.patch.dict(sys.modules,{'szl_cpu_shadow_control':object()}):
            with self.assertRaisesRegex(ValueError,'conflicting sealed module'): RUNNER.load_exact_module('szl_cpu_shadow_control',path,hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError,'sealed module bytes differ'): RUNNER.load_exact_module('szl_cpu_wrong_source',path,'0'*64)
    def test_failed_import_removed(self):
        path=self.root/'sealed.py'; raw=b"raise ValueError('control')\n"; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'control'): RUNNER.load_exact_module('szl_cpu_failed_control',path,hashlib.sha256(raw).hexdigest())
        self.assertNotIn('szl_cpu_failed_control',sys.modules)
    def test_full_predecessor_contract_preserved(self):
        receipt={k:'owned' for k in ['schema','environment_binding_sha256','server_script','lab_root','executable','python_image_sha256','state_directory','environment_root','wheelhouse']}
        receipt.update(pid=123,port=12345,argv=['runner','--owned'],startup_timeout_seconds=180)
        path=self.root/'service.json'
        for key in ['state_directory','argv','executable','environment_root','wheelhouse','python_image_sha256']:
            with self.subTest(key=key):
                raw=RUNNER.canonical({**receipt,key:['wrong'] if key=='argv' else 'wrong'}); path.write_bytes(raw)
                with mock.patch.object(RUNNER,'prior_process_absent',return_value=True) as absent:
                    with self.assertRaisesRegex(ValueError,'absent admitted predecessor'): RUNNER.publish_service(path,receipt)
                    absent.assert_not_called()
                self.assertEqual(path.read_bytes(),raw)
        path.write_bytes(RUNNER.canonical(receipt))
        with mock.patch.object(RUNNER,'prior_process_absent',return_value=False):
            with self.assertRaisesRegex(ValueError,'absent admitted predecessor'): RUNNER.publish_service(path,receipt)
        with mock.patch.object(RUNNER,'prior_process_absent',return_value=True): RUNNER.publish_service(path,{**receipt,'pid':456})
        self.assertEqual(json.loads(path.read_bytes())['pid'],456)
    def test_deadline_exit_survives_disk_error(self):
        with mock.patch.object(sys,'stderr') as stderr,mock.patch.object(RUNNER.os,'_exit',side_effect=SystemExit(124)) as terminate:
            stderr.write.side_effect=OSError('disk full')
            with self.assertRaises(SystemExit): RUNNER.startup_timeout(180)
            terminate.assert_called_once_with(124)
    def test_isolated_verifier_ignores_site_hook(self):
        marker=self.root/'hook-ran'
        (self.root/'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("hook")\n')
        result=subprocess.run([sys.executable,'-I','-S','-B','-c','import sys; assert sys.flags.no_site == 1; assert "sitecustomize" not in sys.modules'],env={**os.environ,'PYTHONPATH':str(self.root)},capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr); self.assertFalse(marker.exists())
        self.assertIn('$python -I -S -B',(OPS/'supervise-workbench.ps1').read_text())
if __name__ == '__main__': unittest.main()
