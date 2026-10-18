# SPDX-License-Identifier: Apache-2.0
"""Synthetic subprocess startup checks; no native admission or model execution."""
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).absolute().parent
HELPERS = ("native_identity", "native_wheel_identity")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="szl-bootstrap-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).absolute()
        self.source = self.root / "source"
        self.source.mkdir()
        self.bootstrap = self.source / "native_bootstrap.py"
        for filename in ("native_bootstrap.py", *(name + ".py" for name in HELPERS)):
            shutil.copyfile(HERE / filename, self.source / filename)
        self.wheels = self.root / "wheels"
        self.wheels.mkdir()
        self.installed = self.root / "installed"
        self.installed.mkdir()
        self.sentinel = self.root / "NEVER_CREATED_PRIVATE_SENTINEL"
        self.members = {"package/__init__.py": b"raise RuntimeError('never import')\n",
                        "example-1.0.dist-info/RECORD": b"opaque RECORD\n",
                        "startup.pth": ("import pathlib; pathlib.Path(" + repr(str(self.sentinel))
                                        + ").touch()\n").encode()}
        for name, raw in self.members.items():
            path = self.installed / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.wheel = self.wheels / "example-1.0-py3-none-any.whl"
        with zipfile.ZipFile(self.wheel, "w", allowZip64=False) as archive:
            for name, raw in self.members.items():
                archive.writestr(name, raw)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_bytes(canonical({"schema": "szl.receiptagent.native-file-manifest/v1",
            "files": [{"path": name, "size_bytes": len(raw), "sha256": digest(raw)}
                      for name, raw in sorted(self.members.items())]}))
        self.inventory = self.root / "inventory.json"
        self.inventory.write_bytes(canonical({"schema": "szl.receiptagent.native-wheel-inventory/v1",
            "native_manifest_sha256": digest(self.manifest.read_bytes()),
            "wheels": [{"path": self.wheel.name, "size_bytes": self.wheel.stat().st_size,
                        "sha256": digest(self.wheel.read_bytes())}]}))
        self.arguments = ["--wheel-root", str(self.wheels), "--inventory", str(self.inventory),
                          "--installed-root", str(self.installed), "--native-manifest", str(self.manifest),
                          "--inventory-sha256", digest(self.inventory.read_bytes()),
                          "--native-manifest-sha256", digest(self.manifest.read_bytes()),
                          "--native-identity-sha256", digest((self.source / "native_identity.py").read_bytes()),
                          "--native-wheel-identity-sha256", digest((self.source / "native_wheel_identity.py").read_bytes())]

    def invoke(self, *, flags=("-I", "-S", "-B"), arguments=None, wrapper=None, env=None):
        prefix = [sys.executable, *flags]
        if sys.flags.optimize:
            prefix.append("-O")
        if wrapper is None:
            command = [*prefix, str(self.bootstrap)]
        else:
            command = [*prefix, "-c", wrapper, str(self.bootstrap)]
        result = subprocess.run([*command, *(self.arguments if arguments is None else arguments)],
                                cwd=self.root, env=env, capture_output=True, timeout=20)
        self.assertEqual(result.stderr, b"")
        self.assertLessEqual(len(result.stdout), 4096)
        return result, json.loads(result.stdout)

    def denied(self, result, value, code=None):
        self.assertEqual(result.returncode, 70)
        self.assertEqual(value["state"], "BLOCKED")
        for flag in ("training_eligible", "publication_eligible", "execution_authority", "operational"):
            self.assertIs(value[flag], False)
        if code:
            self.assertEqual(value["code"], code)
        self.assertFalse(self.sentinel.exists())
        self.assertNotIn(str(self.root), result.stdout.decode())

    def pin_helper(self, name, raw):
        (self.source / (name + ".py")).write_bytes(raw)
        flag = "--" + name.replace("_", "-") + "-sha256"
        self.arguments[self.arguments.index(flag) + 1] = digest(raw)

    def sentinel_source(self):
        return ("from pathlib import Path\nPath(" + repr(str(self.sentinel)) + ").touch()\n").encode()

    def test_exact_bytes_observed_without_authority_or_package_import(self):
        wrapper = ("import sys,types; p=sys.argv[1]; m=types.ModuleType('fixture'); m.__file__=p; "
                   "exec(compile(open(p,'rb').read(),p,'exec'),m.__dict__); "
                   "before=tuple(sys.path); result=m.main(sys.argv[2:])\n"
                   "if tuple(sys.path)!=before: raise RuntimeError('search path changed')\n"
                   "if any(n in sys.modules for n in ('site','sitecustomize','usercustomize',"
                   "'jsonschema','cryptography','torch','transformers','peft','bitsandbytes',"
                   "'safetensors','tokenizers','accelerate','native_identity','native_wheel_identity')): "
                   "raise RuntimeError('unexpected import')\n"
                   "sys.exit(result)")
        result, value = self.invoke(wrapper=wrapper)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(value["schema"], "szl.receiptagent.native-bootstrap-observation/v1")
        self.assertEqual(value["state"], "NATIVE_BYTES_OBSERVED_NOT_ADMITTED")
        self.assertEqual(value["evidence_class"], "MEASURED")
        self.assertEqual(value["file_count"], len(self.members))
        self.assertEqual(value["wheel_count"], 1)
        self.assertEqual(value["total_bytes"], sum(map(len, self.members.values())))
        self.assertEqual(value["startup"], {"isolated": True, "no_site": True, "dont_write_bytecode": True})
        self.assertEqual(value["interpreter_identity"], "EXTERNALLY_TRUSTED_NOT_AUTHENTICATED")
        self.assertEqual(value["stdlib_identity"], "EXTERNALLY_TRUSTED_NOT_AUTHENTICATED")
        for flag in ("training_eligible", "publication_eligible", "execution_authority", "operational"):
            self.assertIs(value[flag], False)
        self.assertFalse(self.sentinel.exists())

    def test_missing_startup_flags_denied_before_helper_or_input_reads(self):
        self.pin_helper("native_identity", self.sentinel_source())
        self.inventory.unlink()
        for flags in (("-S", "-B"), ("-I", "-B"), ("-I", "-S")):
            with self.subTest(flags=flags):
                self.denied(*self.invoke(flags=flags), "BOOTSTRAP_STARTUP_INVALID")

    def test_preloaded_site_helpers_and_origin_aliases_denied(self):
        prefixes = ["sys.modules['site']=types.ModuleType('site'); ",
                    "sys.modules['native_identity']=types.ModuleType('native_identity'); ",
                    "a=types.ModuleType('alias'); a.__file__=p.replace('native_bootstrap.py','native_identity.py'); sys.modules['alias']=a; "]
        for prefix in prefixes:
            wrapper = ("import sys,types; p=sys.argv[1]; " + prefix
                       + "m=types.ModuleType('fixture'); m.__file__=p; exec(compile(open(p,'rb').read(),p,'exec'),m.__dict__)")
            with self.subTest(prefix=prefix):
                self.denied(*self.invoke(wrapper=wrapper))

    def test_cwd_pythonpath_and_startup_sentinels_do_not_execute(self):
        for filename in ("sitecustomize.py", "usercustomize.py", "native_identity.py", "startup.pth"):
            (self.root / filename).write_bytes(self.sentinel_source())
        environment = dict(os.environ, PYTHONPATH=str(self.root), PYTHONSTARTUP=str(self.root / "sitecustomize.py"))
        result, _ = self.invoke(env=environment)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(self.sentinel.exists())

    def test_module_getattr_does_not_execute_during_startup_observation(self):
        wrapper = ("import sys,types; p=sys.argv[1]; a=types.ModuleType('ambient'); "
                   "a.__dict__['__getattr__']=lambda name: (_ for _ in ()).throw(RuntimeError('PRIVATE_CALLBACK')); "
                   "sys.modules['ambient']=a; m=types.ModuleType('fixture'); m.__file__=p; "
                   "exec(compile(open(p,'rb').read(),p,'exec'),m.__dict__); sys.exit(m.main(sys.argv[2:]))")
        result, _ = self.invoke(wrapper=wrapper)
        self.assertEqual(result.returncode, 0)

    def test_dynamic_spec_origin_is_rejected_without_property_execution(self):
        wrapper = ("import sys,types; p=sys.argv[1]\n"
                   "class DynamicSpec:\n"
                   " @property\n"
                   " def origin(self): raise RuntimeError('PRIVATE_CALLBACK')\n"
                   "a=types.ModuleType('ambient'); a.__spec__=DynamicSpec(); sys.modules['ambient']=a; "
                   "m=types.ModuleType('fixture'); m.__file__=p; "
                   "exec(compile(open(p,'rb').read(),p,'exec'),m.__dict__)")
        self.denied(*self.invoke(wrapper=wrapper), "BOOTSTRAP_STARTUP_INVALID")

    def test_both_helper_pins_precede_any_helper_execution(self):
        self.pin_helper("native_identity", self.sentinel_source())
        self.arguments[-1] = "0" * 64
        self.denied(*self.invoke(), "BOOTSTRAP_HELPER_PIN_MISMATCH")

    def test_both_sources_compile_before_either_executes(self):
        self.pin_helper("native_identity", self.sentinel_source())
        self.pin_helper("native_wheel_identity", b"def invalid(:\n")
        self.denied(*self.invoke(), "BOOTSTRAP_VERIFICATION_FAILED")

    def test_poisoned_matching_timestamp_bytecode_is_not_consumed(self):
        cache = self.source / "__pycache__"
        cache.mkdir()
        for name in HELPERS:
            source = self.source / (name + ".py")
            header = (importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, int(source.stat().st_mtime),
                                                                  source.stat().st_size))
            poison = header + marshal.dumps(compile(self.sentinel_source(), str(source), "exec"))
            for suffix in ("", ".opt-1", ".opt-2"):
                (cache / (name + "." + sys.implementation.cache_tag + suffix + ".pyc")).write_bytes(poison)
            (self.source / (name + ".pyc")).write_bytes(poison)
        result, _ = self.invoke()
        self.assertEqual(result.returncode, 0)
        self.assertFalse(self.sentinel.exists())

    def test_unknown_duplicate_missing_and_private_arguments_are_redacted(self):
        for arguments in ([], ["--help"], ["--unknown", str(self.sentinel)],
                          [*self.arguments, "--wheel-root", str(self.sentinel)],
                          [*self.arguments[:-1], "PRIVATE_BAD_DIGEST_VALUE"]):
            with self.subTest(count=len(arguments)):
                result, value = self.invoke(arguments=arguments)
                self.denied(result, value, "BOOTSTRAP_ARGUMENTS_INVALID")
                self.assertNotIn("PRIVATE_BAD_DIGEST_VALUE", result.stdout.decode())

    def test_relative_input_path_is_denied(self):
        arguments = list(self.arguments)
        arguments[1] = "relative/wheels"
        self.denied(*self.invoke(arguments=arguments), "BOOTSTRAP_ARGUMENTS_INVALID")

    def test_oversized_helper_is_denied_before_source_execution(self):
        self.pin_helper("native_identity", self.sentinel_source() + b"#" * (256 * 1024))
        self.denied(*self.invoke(), "BOOTSTRAP_HELPER_BYTE_CAP")

    def test_wrong_wheel_tree_pin_and_missing_path_are_redacted(self):
        for flag, wrong in (("--inventory-sha256", "0" * 64),
                            ("--native-manifest-sha256", "0" * 64),
                            ("--inventory", str(self.sentinel))):
            arguments = list(self.arguments)
            arguments[arguments.index(flag) + 1] = wrong
            with self.subTest(flag=flag):
                self.denied(*self.invoke(arguments=arguments), "BOOTSTRAP_VERIFICATION_FAILED")

    def test_helper_hardlink_is_rejected(self):
        helper = self.source / "native_identity.py"
        alias = self.root / "helper_alias.py"
        try:
            os.link(helper, alias)
        except OSError:
            self.skipTest("Hardlinks unavailable")
        self.denied(*self.invoke(), "BOOTSTRAP_HELPER_FILE_INVALID")

    def test_helper_symlink_is_rejected(self):
        helper = self.source / "native_identity.py"
        raw = helper.read_bytes()
        alias = self.root / "helper_alias.py"
        helper.unlink()
        alias.write_bytes(raw)
        try:
            helper.symlink_to(alias)
        except OSError:
            helper.write_bytes(raw)
            self.skipTest("Symlink creation unavailable")
        self.denied(*self.invoke(), "BOOTSTRAP_HELPER_FILE_INVALID")

    def test_helper_change_during_streaming_denied(self):
        wrapper = ("import sys,types; p=sys.argv[1]; m=types.ModuleType('fixture'); m.__file__=p; "
                   "exec(compile(open(p,'rb').read(),p,'exec'),m.__dict__); original=m.os.read; changed=False\n"
                   "def mutate(fd,size):\n"
                   " global changed\n"
                   " data=original(fd,size)\n"
                   " if data and not changed:\n"
                   "  changed=True\n"
                   "  with open(p.replace('native_bootstrap.py','native_identity.py'),'ab') as out: out.write(b'changed')\n"
                   " return data\n"
                   "m.os.read=mutate\nsys.exit(m.main(sys.argv[2:]))")
        self.denied(*self.invoke(wrapper=wrapper))

    def test_helper_failure_cleans_inserted_modules_and_redacts_exceptions(self):
        self.pin_helper("native_identity", b"raise RuntimeError('PRIVATE_EXCEPTION_SENTINEL')\n")
        wrapper = ("import sys,types; p=sys.argv[1]; m=types.ModuleType('fixture'); m.__file__=p; "
                   "exec(compile(open(p,'rb').read(),p,'exec'),m.__dict__); result=m.main(sys.argv[2:])\n"
                   "if 'native_identity' in sys.modules or 'native_wheel_identity' in sys.modules: "
                   "raise RuntimeError('helper cleanup failed')\n"
                   "sys.exit(result)")
        result, value = self.invoke(wrapper=wrapper)
        self.denied(result, value, "BOOTSTRAP_VERIFICATION_FAILED")
        self.assertNotIn("PRIVATE_EXCEPTION_SENTINEL", result.stdout.decode())


if __name__ == "__main__":
    unittest.main()
