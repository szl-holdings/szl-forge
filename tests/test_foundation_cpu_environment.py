"""Network-free stdlib-only CPU admission controls; no model import."""
import hashlib
import ast
import datetime as dt
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import threading
import time
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
CREATED = '2026-10-03T04:20:46.417523Z'

def fake_kernel(open_handle=123, creation_delta=0, times_ok=1, waited=0, closed=1):
    kernel = mock.Mock()
    kernel.OpenProcess.return_value = open_handle
    native = dt.datetime.fromisoformat(CREATED[:-1] + '+00:00') - dt.datetime(1601,1,1,tzinfo=dt.timezone.utc)
    ticks = (native.days*86400 + native.seconds)*10000000 + native.microseconds*10 + 9 + creation_delta
    def times(handle, *values):
        values[0]._obj.value = ticks
        return times_ok
    kernel.GetProcessTimes.side_effect = times
    kernel.WaitForSingleObject.return_value = waited
    kernel.CloseHandle.return_value = closed
    return kernel
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
    def test_bounded_parallel_hashes_and_tamper_failure(self):
        extra = {'example/file'+str(i)+'.py': b'VALUE=1\n' for i in range(20)}
        env,wh,site,row = payload(self.root,extra)
        for name,raw in extra.items(): (site/name).write_bytes(raw)
        guard = threading.Lock(); counts = {'active':0,'maximum':0,'calls':0}
        original = ENV.digest_file
        def measured(path):
            if not Path(path).is_relative_to(site): return original(path)
            with guard:
                counts['active']+=1; counts['calls']+=1
                counts['maximum']=max(counts['maximum'],counts['active'])
            try:
                time.sleep(0.02)
                return original(path)
            finally:
                with guard: counts['active']-=1
        with mock.patch.object(ENV,'digest_file',side_effect=measured):
            result=ENV.verify_payload(env,wh,[row])
        self.assertEqual(result['files_verified'],22)
        self.assertEqual(counts['calls'],22)
        self.assertGreater(counts['maximum'],1)
        self.assertLessEqual(counts['maximum'],4)
        self.assertEqual(counts['active'],0)
        (site/'example/file19.py').write_bytes(b'TAMPER\n')
        with self.assertRaisesRegex(ValueError,'payload differs'):
            ENV.verify_payload(env,wh,[row])
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
        receipt.update(pid=123,process_created=CREATED,port=12345,argv=['runner','--owned'],startup_timeout_seconds=180)
        path=self.root/'service.json'
        for key in ['schema','environment_binding_sha256','server_script','lab_root','port','executable','python_image_sha256','state_directory','environment_root','wheelhouse','argv','startup_timeout_seconds']:
            with self.subTest(key=key):
                raw=RUNNER.canonical({**receipt,key:['wrong'] if key=='argv' else 'wrong'}); path.write_bytes(raw)
                with mock.patch.object(RUNNER,'prior_process_exited',return_value=True) as exited:
                    with self.assertRaisesRegex(ValueError,'exited admitted predecessor'): RUNNER.publish_service(path,receipt)
                    exited.assert_not_called()
                self.assertEqual(path.read_bytes(),raw)
        path.write_bytes(RUNNER.canonical(receipt))
        with mock.patch.object(RUNNER,'prior_process_exited',return_value=False):
            with self.assertRaisesRegex(ValueError,'exited admitted predecessor'): RUNNER.publish_service(path,receipt)
        with mock.patch.object(RUNNER,'prior_process_exited',return_value=True) as exited:
            RUNNER.publish_service(path,{**receipt,'pid':456})
            exited.assert_called_once_with(123,CREATED)
        self.assertEqual(json.loads(path.read_bytes())['pid'],456)
    def test_prior_process_requires_strict_identity_before_native_query(self):
        with mock.patch.object(RUNNER.ctypes,'WinDLL',create=True) as native:
            for pid in [None,True,False,0,-1,0x100000000,'123',123.0]:
                with self.subTest(pid=pid):
                    with self.assertRaises(ValueError): RUNNER.prior_process_exited(pid,CREATED)
            for created in [None,False,123,[],CREATED.replace('Z','+00:00'),CREATED.replace('.417523','.41752'),CREATED.replace('10-03','02-30'),'1600-01-01T00:00:00.000000Z']:
                with self.subTest(created=created):
                    with self.assertRaises(ValueError): RUNNER.prior_process_exited(123,created)
            native.assert_not_called()
    def test_prior_process_absence_requires_exact_native_error(self):
        for error in [87,5,0]:
            kernel = fake_kernel(open_handle=0)
            with self.subTest(error=error),mock.patch.object(RUNNER.ctypes,'WinDLL',return_value=kernel,create=True),mock.patch.object(RUNNER.ctypes,'get_last_error',return_value=error,create=True):
                if error == 87: self.assertTrue(RUNNER.prior_process_exited(123,CREATED))
                else:
                    with self.assertRaisesRegex(ValueError,'absence could not be established'): RUNNER.prior_process_exited(123,CREATED)
            kernel.OpenProcess.assert_called_once_with(0x101000,False,123)
            kernel.GetProcessTimes.assert_not_called(); kernel.WaitForSingleObject.assert_not_called(); kernel.CloseHandle.assert_not_called()
    def test_retained_exit_requires_exact_native_creation(self):
        kernel = fake_kernel()
        with mock.patch.object(RUNNER.ctypes,'WinDLL',return_value=kernel,create=True):
            self.assertTrue(RUNNER.prior_process_exited(123,CREATED))
        kernel.GetProcessTimes.assert_called_once(); kernel.WaitForSingleObject.assert_called_once_with(123,0)
        kernel.CloseHandle.assert_called_once_with(123)
        for options,message in [({'creation_delta':10},'creation identity differs'),({'times_ok':0},'creation time is unavailable')]:
            kernel = fake_kernel(**options)
            with self.subTest(options=options),mock.patch.object(RUNNER.ctypes,'WinDLL',return_value=kernel,create=True):
                with self.assertRaisesRegex(ValueError,message): RUNNER.prior_process_exited(123,CREATED)
            kernel.WaitForSingleObject.assert_not_called(); kernel.CloseHandle.assert_called_once_with(123)
    def test_native_wait_and_close_failures_deny(self):
        for waited in [0x102,0xffffffff,1,0x80]:
            kernel = fake_kernel(waited=waited)
            with self.subTest(waited=waited),mock.patch.object(RUNNER.ctypes,'WinDLL',return_value=kernel,create=True):
                if waited == 0x102: self.assertFalse(RUNNER.prior_process_exited(123,CREATED))
                else:
                    with self.assertRaisesRegex(ValueError,'exit could not be established'): RUNNER.prior_process_exited(123,CREATED)
            kernel.CloseHandle.assert_called_once_with(123)
        kernel = fake_kernel(closed=0)
        with mock.patch.object(RUNNER.ctypes,'WinDLL',return_value=kernel,create=True):
            with self.assertRaisesRegex(ValueError,'query handle could not be closed'): RUNNER.prior_process_exited(123,CREATED)
    def test_native_rejection_preserves_predecessor_receipt_bytes(self):
        receipt={key:'owned' for key in ['schema','environment_binding_sha256','server_script','lab_root','executable','python_image_sha256','state_directory','environment_root','wheelhouse']}
        receipt.update(pid=123,process_created=CREATED,port=12345,argv=['runner','--owned'],startup_timeout_seconds=180)
        path=self.root/'service.json'; original=RUNNER.canonical(receipt)
        for options,error in [({'waited':0x102},87),({'waited':0xffffffff},87),({'creation_delta':10},87),({'times_ok':0},87),({'open_handle':0},5),({'closed':0},87)]:
            path.write_bytes(original); kernel=fake_kernel(**options)
            with self.subTest(options=options),mock.patch.object(RUNNER.ctypes,'WinDLL',return_value=kernel,create=True),mock.patch.object(RUNNER.ctypes,'get_last_error',return_value=error,create=True):
                with self.assertRaises(ValueError): RUNNER.publish_service(path,{**receipt,'pid':456})
            self.assertEqual(path.read_bytes(),original)
        for key,value in [('pid',True),('process_created','2026-02-30T00:00:00.000000Z')]:
            raw=RUNNER.canonical({**receipt,key:value}); path.write_bytes(raw)
            with mock.patch.object(RUNNER.ctypes,'WinDLL',create=True) as native:
                with self.assertRaises(ValueError): RUNNER.publish_service(path,{**receipt,'pid':456})
                native.assert_not_called()
            self.assertEqual(path.read_bytes(),raw)
    @unittest.skipUnless(os.name == 'nt','Actual retained native handle requires Windows')
    def test_actual_retained_native_handle_publication(self):
        native=module('test-cpu-predecessor').run()
        self.assertEqual(native['status'],'VERIFIED'); self.assertEqual(native['count'],5)
        self.assertFalse(native['production_services_changed']); self.assertFalse(native['model_imported'])
    def test_deadline_exit_survives_disk_error(self):
        with mock.patch.object(sys,'stderr') as stderr,mock.patch.object(RUNNER.os,'_exit',side_effect=SystemExit(124)) as terminate:
            stderr.write.side_effect=OSError('disk full')
            with self.assertRaises(SystemExit): RUNNER.startup_timeout(180)
            terminate.assert_called_once_with(124)
    def test_phase_events_are_stderr_only_and_monotonic(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys,'stdout',stdout),mock.patch.object(sys,'stderr',stderr),mock.patch.object(RUNNER,'process_created',return_value=CREATED):
            phases = RUNNER.StartupPhaseLog()
            self.assertIsNone(phases.event('environment_admission','ENTERED'))
            self.assertIsNone(phases.event('environment_admission','RETURNED'))
        rows = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertEqual(stdout.getvalue(),'')
        self.assertEqual([row['event'] for row in rows],['ENTERED','RETURNED'])
        self.assertTrue(all(row['pid'] == os.getpid() and row['process_created'] == CREATED and row['class'] == 'MEASURED' and row['diagnostic_only'] is True and row['signed'] is False for row in rows))
        self.assertEqual([row['sequence'] for row in rows],[1,2])
        self.assertGreaterEqual(rows[1]['monotonic_elapsed_seconds'],rows[0]['monotonic_elapsed_seconds'])
        self.assertGreaterEqual(rows[1]['phase_elapsed_seconds'],0)
    def test_phase_failures_preserve_original_exception(self):
        sentinel = ValueError('original admission failure')
        stderr = io.StringIO()
        with mock.patch.object(sys,'stderr',stderr),mock.patch.object(RUNNER,'process_created',return_value=CREATED):
            phases = RUNNER.StartupPhaseLog()
            phases.event('frozen_verification','ENTERED')
            try:
                try: raise sentinel
                except BaseException as error:
                    phases.failed(error)
                    raise
            except ValueError as error: self.assertIs(error,sentinel)
        failure = json.loads(stderr.getvalue().splitlines()[-1])
        self.assertEqual((failure['phase'],failure['event'],failure['error_type']),('frozen_verification','FAILED','ValueError'))
        self.assertNotIn('original admission failure',stderr.getvalue())
    def test_phase_logging_failure_cannot_replace_operation_or_deadline(self):
        sentinel = ValueError('admission must still fail')
        fallback = io.StringIO()
        with mock.patch.object(sys,'stderr') as stderr,mock.patch.object(sys,'__stderr__',fallback),mock.patch.object(RUNNER,'process_created',return_value=CREATED):
            stderr.write.side_effect=OSError('disk full')
            phases = RUNNER.StartupPhaseLog()
            phases.event('environment_admission','ENTERED')
            try:
                try: raise sentinel
                except BaseException as error:
                    phases.failed(error)
                    raise
            except ValueError as error: self.assertIs(error,sentinel)
            with mock.patch.object(RUNNER.os,'_exit',side_effect=SystemExit(124)) as terminate:
                with self.assertRaises(SystemExit): RUNNER.startup_timeout(180)
                terminate.assert_called_once_with(124)
        self.assertEqual(json.loads(fallback.getvalue().splitlines()[0])['class'],'UNAVAILABLE')
    def test_phase_sessions_are_distinct_and_timer_origin_is_unchanged(self):
        with mock.patch.object(sys,'stderr',io.StringIO()),mock.patch.object(RUNNER,'process_created',return_value=CREATED):
            first,second = RUNNER.StartupPhaseLog(),RUNNER.StartupPhaseLog()
        self.assertNotEqual(first.session_id,second.session_id)
        source = (OPS/'cpu_workbench.py').read_text()
        self.assertLess(source.index('timer.start()'),source.index('_startup_phases = StartupPhaseLog(started)'))
        self.assertIn("threading.Timer(args.startup_timeout_seconds, startup_timeout, args=(args.startup_timeout_seconds,))",source)
        self.assertIn('timer.cancel()',source)
    def test_instrumented_main_keeps_admission_failure_and_started_timer(self):
        sentinel = ValueError('environment admission sentinel')
        helper, timer = mock.Mock(), mock.Mock()
        helper.admit.side_effect = sentinel
        argv = ['cpu_workbench.py']
        for name in ('lab-root','archive','environment-root','wheelhouse','expected-python-sha256','state-directory'):
            argv.extend(['--'+name,'fixture'])
        argv.extend(['--environment-binding-sha256','b'*64])
        stderr = io.StringIO()
        with mock.patch.object(sys,'argv',argv),mock.patch.object(sys,'stderr',stderr),mock.patch.object(RUNNER,'process_created',return_value=CREATED),mock.patch.object(RUNNER,'load_admission',return_value=helper),mock.patch.object(RUNNER.threading,'Timer',return_value=timer) as deadline,mock.patch.object(RUNNER,'publish_service') as publish:
            try: RUNNER.main()
            except ValueError as error: self.assertIs(error,sentinel)
            else: self.fail('Original admission failure was swallowed')
            deadline.assert_called_once_with(180,RUNNER.startup_timeout,args=(180,))
            timer.start.assert_called_once_with(); timer.cancel.assert_not_called()
            helper.admit.assert_called_once_with('fixture','fixture','fixture')
            publish.assert_not_called()
        rows = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertEqual([(row['phase'],row['event']) for row in rows],[('admission_helper_load','ENTERED'),('admission_helper_load','RETURNED'),('environment_admission','ENTERED')])
    def test_phase_initialization_failure_cannot_replace_admission(self):
        sentinel = ValueError('unchanged original admission')
        helper, timer, fallback = mock.Mock(), mock.Mock(), io.StringIO()
        helper.admit.side_effect = sentinel
        argv = ['cpu_workbench.py']
        for name in ('lab-root','archive','environment-root','wheelhouse','expected-python-sha256','state-directory'):
            argv.extend(['--'+name,'fixture'])
        argv.extend(['--environment-binding-sha256','b'*64])
        with mock.patch.object(sys,'argv',argv),mock.patch.object(sys,'__stderr__',fallback),mock.patch.object(RUNNER.uuid,'uuid4',side_effect=OSError('random source unavailable')),mock.patch.object(RUNNER,'load_admission',return_value=helper),mock.patch.object(RUNNER.threading,'Timer',return_value=timer):
            try: RUNNER.main()
            except ValueError as error:
                self.assertIs(error,sentinel)
                RUNNER._startup_phases.failed(error) # Disabled telemetry must also remain safe.
            else: self.fail('Telemetry initialization replaced admission')
            helper.admit.assert_called_once_with('fixture','fixture','fixture')
            timer.start.assert_called_once_with(); timer.cancel.assert_not_called()
        self.assertEqual(json.loads(fallback.getvalue())['class'],'UNAVAILABLE')
    def test_diagnostic_io_is_before_final_deadline_admission(self):
        source = ast.parse((OPS/'cpu_workbench.py').read_text())
        main = next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name == 'main')
        publication = next(node for node in ast.walk(main) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr == 'event' and [value.value for value in node.value.args if isinstance(value,ast.Constant)] == ['owned_service_publication','RETURNED'])
        cancel = next(node for node in ast.walk(main) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id == 'timer' and node.value.func.attr == 'cancel')
        self.assertLess(publication.lineno,cancel.lineno)
        launcher = (OPS/'launch-cpu-workbench.ps1').read_text()
        self.assertLess(launcher.index("Write-FoundationPhaseEvent 'child_readiness_admission' 'RETURNED'"),launcher.index('$guard.Complete()'))
    def test_actual_entrypoint_rethrows_and_retains_failed_phase(self):
        sentinel = ValueError('original entrypoint failure')
        stderr = io.StringIO()
        with mock.patch.object(sys,'stderr',stderr),mock.patch.object(RUNNER,'process_created',return_value=CREATED):
            phases = RUNNER.StartupPhaseLog(); phases.event('all_three_preload','ENTERED')
            entrypoint = ast.parse((OPS/'cpu_workbench.py').read_text()).body[-1]
            namespace = {'__name__':'__main__','main':mock.Mock(side_effect=sentinel),'_startup_phases':phases}
            try: exec(compile(ast.Module(body=[entrypoint],type_ignores=[]),'<actual entrypoint>','exec'),namespace)
            except ValueError as error: self.assertIs(error,sentinel)
            else: self.fail('Actual entrypoint swallowed the failure')
        row = json.loads(stderr.getvalue().splitlines()[-1])
        self.assertEqual((row['phase'],row['event']),('all_three_preload','FAILED'))
    def test_isolated_verifier_ignores_site_hook(self):
        marker=self.root/'hook-ran'
        (self.root/'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("hook")\n')
        result=subprocess.run([sys.executable,'-I','-S','-B','-c','import sys; assert sys.flags.no_site == 1; assert "sitecustomize" not in sys.modules'],env={**os.environ,'PYTHONPATH':str(self.root)},capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr); self.assertFalse(marker.exists())
        self.assertIn('$python -I -S -B',(OPS/'supervise-workbench.ps1').read_text())
if __name__ == '__main__': unittest.main()
