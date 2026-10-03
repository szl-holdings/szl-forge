"""Actual Windows retained-handle controls; only one owned fixture child."""
import ctypes
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run():
    if os.name != 'nt':
        raise RuntimeError('The native predecessor control requires Windows')
    path = Path(__file__).with_name('cpu_workbench.py')
    spec = importlib.util.spec_from_file_location('fixture_cpu_workbench', path)
    runner = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), runner.__dict__)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = (ctypes.c_void_p, *(ctypes.POINTER(ctypes.c_ulonglong) for _ in range(4)))
    kernel.GetProcessTimes.restype = ctypes.c_int
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    checks = []
    # A child-side deadline bounds even a failed parent-side fixture cleanup.
    code = 'import os,sys,threading; threading.Timer(15,lambda:os._exit(0)).start(); sys.stdin.buffer.read(1); os._exit(0)'
    child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', code],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        runner.require(kernel.GetProcessTimes(int(child._handle), *(ctypes.byref(value) for value in times)) != 0,
                       'Fixture creation time is unavailable')
        created = (dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc) +
                   dt.timedelta(microseconds=times[0].value // 10)).isoformat(timespec='microseconds').replace('+00:00', 'Z')
        runner.require(not runner.prior_process_exited(child.pid, created), 'Live fixture was admitted')
        checks.append('ACTUAL_LIVE_NATIVE_CHILD_REFUSED')
        receipt = {key: 'fixture' for key in ('schema', 'environment_binding_sha256', 'server_script', 'lab_root', 'executable',
                                              'python_image_sha256', 'state_directory', 'environment_root', 'wheelhouse')}
        receipt.update(pid=child.pid, process_created=created, port=12345, argv=['fixture'], startup_timeout_seconds=180)
        successor = {**receipt, 'pid': os.getpid(), 'process_created': runner.process_created()}
        with tempfile.TemporaryDirectory(prefix='szl-native-predecessor-') as directory:
            service = Path(directory) / 'service.json'
            original = runner.canonical(receipt)
            service.write_bytes(original)
            try:
                runner.publish_service(service, successor)
            except ValueError as exc:
                runner.require('not an exited admitted predecessor' in str(exc), 'Unexpected live fixture refusal')
            else:
                raise AssertionError('A live native predecessor receipt was replaced')
            runner.require(service.read_bytes() == original, 'Live predecessor receipt bytes changed')
            checks.append('ACTUAL_LIVE_PREDECESSOR_RECEIPT_BYTES_PRESERVED')
            child.stdin.write(b'x')
            child.stdin.flush()
            runner.require(child.wait(timeout=20) == 0, 'Fixture child did not exit successfully')
            # Popen deliberately retains its native process handle across this wait.
            query = kernel.OpenProcess(0x1000, False, child.pid)
            runner.require(bool(query), 'Exited fixture object was not retained for the regression control')
            runner.require(kernel.CloseHandle(query) != 0, 'Fixture regression query handle could not be closed')
            runner.require(runner.prior_process_exited(child.pid, created), 'Retained exited fixture was refused')
            checks.append('ACTUAL_RETAINED_NATIVE_CHILD_EXIT_ADMITTED')
            changed = (dt.datetime.fromisoformat(created[:-1] + '+00:00') + dt.timedelta(microseconds=1)).isoformat(timespec='microseconds').replace('+00:00', 'Z')
            try:
                runner.prior_process_exited(child.pid, changed)
            except ValueError as exc:
                runner.require('creation identity differs' in str(exc), 'Unexpected creation mismatch refusal')
            else:
                raise AssertionError('An altered native creation identity was admitted')
            checks.append('ACTUAL_NEXT_MICROSECOND_NATIVE_IDENTITY_REFUSED')
            runner.publish_service(service, successor)
            runner.require(service.read_bytes() == runner.canonical(successor), 'Exited fixture receipt replacement differs')
            checks.append('ACTUAL_RETAINED_EXIT_ALLOWS_EXACT_CONTRACT_SUCCESSOR_RECEIPT')
        return {'schema': 'szl.foundation-confirmation.native-predecessor-controls/v1',
                'class': 'MEASURED', 'status': 'VERIFIED', 'checks': checks, 'count': len(checks),
                'only_owned_fixture_child': True, 'production_services_changed': False, 'model_imported': False}
    finally:
        try:
            if child.poll() is None:
                try:
                    child.stdin.write(b'x')
                    child.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass  # The owned child's deadline may race this cleanup signal.
                child.wait(timeout=20)
        finally:
            try:
                try:
                    child.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            finally:
                child._handle.Close()


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
