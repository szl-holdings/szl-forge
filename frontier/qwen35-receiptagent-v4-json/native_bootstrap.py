# SPDX-License-Identifier: Apache-2.0
"""Verify-only pre-site entrypoint; no admission, package import or worker launch.

Invoke with an independently trusted interpreter and stdlib using -I -S -B.
The flags and current sys.modules are observations, not proof that previously
executed-and-unloaded code never ran. In particular rejection without -S cannot
undo startup hooks that ran before this script. This script does not authenticate
itself, the executing interpreter, its stdlib/bytecode, loaded native libraries,
owner approval, dependency completeness or permission to consume verified code.

The only repository code loaded is the fixed adjacent native_identity.py and
native_wheel_identity.py. Both externally pinned source buffers are bounded-read
and verified before either is compiled/executed. Compilation uses retained bytes,
never ordinary helper imports or cached .pyc. No installed directory is added to
sys.path; installed startup hooks, packages and shared libraries stay opaque.
The existing helper's exact wheel/tree comparison remains a byte observation.

Trusted single-writer inputs and a trusted bootstrap are external assumptions,
not a hostile same-UID sandbox or an immutable snapshot. Independently freeze
inputs before later consumption. There is no configurable module, target, code,
callback, admission, network, installation, cleanup or dispatch mode. Normal CLI
errors are value-free; this is not secrecy against Python introspection.
"""
import sys

_HELPERS = ("native_identity", "native_wheel_identity")


def _startup_code():
    # No non-sys imports or repository/input-path reads before this check.
    if not (sys.flags.isolated == 1 and sys.flags.no_site == 1
            and sys.flags.dont_write_bytecode == 1 and sys.dont_write_bytecode is True):
        return "BOOTSTRAP_STARTUP_INVALID"
    if any(name in sys.modules for name in ("site", "sitecustomize", "usercustomize")):
        return "BOOTSTRAP_STARTUP_INVALID"
    for name, module in tuple(sys.modules.items()):
        if type(name) is not str:
            return "BOOTSTRAP_STARTUP_INVALID"
        if name.rsplit(".", 1)[-1] in _HELPERS:
            return "BOOTSTRAP_PRELOADED_HELPER"
        if module is None:
            continue
        # PEP 562 module __getattr__ and spec properties must never execute as
        # part of this observation. Unsupported ambient objects fail closed.
        if type(module) is not type(sys):
            return "BOOTSTRAP_STARTUP_INVALID"
        namespace = object.__getattribute__(module, "__dict__")
        spec = namespace.get("__spec__")
        if spec is not None and type(spec) is not type(sys.__spec__):
            return "BOOTSTRAP_STARTUP_INVALID"
        spec_origin = None if spec is None else object.__getattribute__(spec, "__dict__").get("origin")
        origins = (namespace.get("__file__"), spec_origin)
        for origin in origins:
            if type(origin) is str:
                basename = origin.replace("\\", "/").rsplit("/", 1)[-1].casefold()
                if any(basename == helper + ".py" or basename == helper + ".pyc"
                       or (basename.startswith(helper + ".") and basename.endswith(".pyc"))
                       for helper in _HELPERS):
                    return "BOOTSTRAP_PRELOADED_HELPER"
    return None


def _early_stop(code):
    # Only two fixed internal codes reach this pre-stdlib JSON emitter.
    if code not in ("BOOTSTRAP_STARTUP_INVALID", "BOOTSTRAP_PRELOADED_HELPER"):
        code = "BOOTSTRAP_STARTUP_INVALID"
    sys.stdout.write('{"code":"' + code + '","execution_authority":false,'
                     '"operational":false,"publication_eligible":false,'
                     '"state":"BLOCKED","training_eligible":false}\n')
    raise SystemExit(70)


_initial_denial = _startup_code()
if _initial_denial is not None:
    _early_stop(_initial_denial)

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import ModuleType

SCHEMA = "szl.receiptagent.native-bootstrap-observation/v1"
MAX_HELPER_BYTES = 256 * 1024
HASH_CHUNK_BYTES = 64 * 1024
_PIN = re.compile(r"[0-9a-f]{64}")
_PATH_ARGUMENTS = ("--wheel-root", "--inventory", "--installed-root", "--native-manifest")
_PIN_ARGUMENTS = ("--inventory-sha256", "--native-manifest-sha256",
                  "--native-identity-sha256", "--native-wheel-identity-sha256")


class BootstrapError(ValueError):
    """Fixed internal normal-display codes, never input paths or source text."""


def _require(condition, code):
    if condition is not True:
        raise BootstrapError(code)


def _arguments(argv):
    # A closed pair grammar avoids argparse's usage/value/path echo entirely.
    names = set(_PATH_ARGUMENTS + _PIN_ARGUMENTS)
    _require(type(argv) is list and len(argv) == 2 * len(names), "BOOTSTRAP_ARGUMENTS_INVALID")
    result = {}
    for index in range(0, len(argv), 2):
        name, value = argv[index:index + 2]
        _require(type(name) is str and name in names and name not in result
                 and type(value) is str and 0 < len(value) <= 4096 and "\x00" not in value,
                 "BOOTSTRAP_ARGUMENTS_INVALID")
        result[name] = value
    _require(set(result) == names, "BOOTSTRAP_ARGUMENTS_INVALID")
    _require(all(_PIN.fullmatch(result[name]) is not None for name in _PIN_ARGUMENTS),
             "BOOTSTRAP_ARGUMENTS_INVALID")
    _require(all(Path(result[name]).is_absolute() and ".." not in Path(result[name]).parts
                 for name in _PATH_ARGUMENTS), "BOOTSTRAP_ARGUMENTS_INVALID")
    return result


def _plain(meta):
    return not stat.S_ISLNK(meta.st_mode) and not getattr(meta, "st_file_attributes", 0) & 0x400


def _checked_path(path):
    _require(path.is_absolute() and ".." not in path.parts, "BOOTSTRAP_HELPER_FILE_INVALID")
    for ancestor in reversed((path, *path.parents)):
        _require(_plain(ancestor.lstat()), "BOOTSTRAP_HELPER_FILE_INVALID")


def _stable(meta):
    return (meta.st_dev, meta.st_ino, meta.st_mode, meta.st_size, meta.st_mtime_ns, meta.st_nlink)


def _stamp(meta):
    return (*_stable(meta), meta.st_ctime_ns)


def _capture_source(path):
    _checked_path(path)
    before_path = path.lstat()
    _require(_plain(before_path) and stat.S_ISREG(before_path.st_mode)
             and before_path.st_nlink == 1, "BOOTSTRAP_HELPER_FILE_INVALID")
    _require(0 < before_path.st_size <= MAX_HELPER_BYTES, "BOOTSTRAP_HELPER_BYTE_CAP")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    try:
        before_fd = os.fstat(descriptor)
        _require(_stable(before_path) == _stable(before_fd), "BOOTSTRAP_HELPER_CHANGED")
        chunks, size, hasher = [], 0, hashlib.sha256()
        while True:
            block = os.read(descriptor, min(HASH_CHUNK_BYTES, MAX_HELPER_BYTES - size + 1))
            if not block:
                break
            size += len(block)
            _require(size <= MAX_HELPER_BYTES, "BOOTSTRAP_HELPER_BYTE_CAP")
            chunks.append(block)
            hasher.update(block)
        after_fd, after_path = os.fstat(descriptor), path.lstat()
        _require(_stamp(before_fd) == _stamp(after_fd) and _stamp(before_path) == _stamp(after_path)
                 and _stable(after_fd) == _stable(after_path) and size == before_fd.st_size,
                 "BOOTSTRAP_HELPER_CHANGED")
        _checked_path(path)
        return b"".join(chunks), hasher.hexdigest(), _stamp(after_path)
    finally:
        os.close(descriptor)


def _sources_unchanged(sources):
    for path, _raw, _digest, stamp in sources.values():
        _checked_path(path)
        _require(_stamp(path.lstat()) == stamp, "BOOTSTRAP_HELPER_CHANGED")


def _observe(arguments):
    bootstrap_path = Path(__file__).absolute()
    _checked_path(bootstrap_path)
    sources = {}
    for name in _HELPERS:
        path = bootstrap_path.parent / (name + ".py")
        raw, digest, stamp = _capture_source(path)
        pin = arguments["--" + name.replace("_", "-") + "-sha256"]
        _require(digest == pin, "BOOTSTRAP_HELPER_PIN_MISMATCH")
        sources[name] = (path, raw, digest, stamp)
    _sources_unchanged(sources)
    # Compile BOTH retained buffers before either executes. The only acceptable
    # source code is these separately reviewed/pinned fixed helper filenames.
    compiled = {name: compile(raw, str(path), "exec", dont_inherit=True)
                for name, (path, raw, _digest, _stamp_value) in sources.items()}
    search_path = tuple(sys.path)
    inserted = []
    try:
        modules = {}
        for name in _HELPERS:
            _require(name not in sys.modules, "BOOTSTRAP_PRELOADED_HELPER")
            module = ModuleType(name)
            module.__file__ = str(sources[name][0])
            module.__package__ = ""
            sys.modules[name] = module
            inserted.append(name)
            exec(compiled[name], module.__dict__)
            modules[name] = module
        _require(tuple(sys.path) == search_path, "BOOTSTRAP_SEARCH_PATH_CHANGED")
        result = modules["native_wheel_identity"].verify_native_wheel_inventory(
            Path(arguments["--wheel-root"]), Path(arguments["--inventory"]),
            Path(arguments["--installed-root"]), Path(arguments["--native-manifest"]),
            expected_inventory_sha256=arguments["--inventory-sha256"],
            expected_native_manifest_sha256=arguments["--native-manifest-sha256"])
        _require(type(result.wheel_count) is int and 1 <= result.wheel_count <= 128
                 and type(result.file_count) is int and 1 <= result.file_count <= 200000
                 and type(result.total_bytes) is int and 0 <= result.total_bytes <= 64 * 1024**3
                 and result.inventory_sha256 == arguments["--inventory-sha256"]
                 and result.native_manifest_sha256 == arguments["--native-manifest-sha256"],
                 "BOOTSTRAP_RESULT_INVALID")
        _sources_unchanged(sources)
        _require(tuple(sys.path) == search_path, "BOOTSTRAP_SEARCH_PATH_CHANGED")
        _require(not any(name in sys.modules for name in ("site", "sitecustomize", "usercustomize")),
                 "BOOTSTRAP_STARTUP_INVALID")
        return {"schema": SCHEMA, "state": "NATIVE_BYTES_OBSERVED_NOT_ADMITTED",
                "evidence_class": "MEASURED", "wheel_count": result.wheel_count,
                "file_count": result.file_count, "total_bytes": result.total_bytes,
                "inventory_sha256": result.inventory_sha256,
                "native_manifest_sha256": result.native_manifest_sha256,
                "native_identity_sha256": sources["native_identity"][2],
                "native_wheel_identity_sha256": sources["native_wheel_identity"][2],
                "startup": {"isolated": True, "no_site": True, "dont_write_bytecode": True},
                "interpreter_identity": "EXTERNALLY_TRUSTED_NOT_AUTHENTICATED",
                "stdlib_identity": "EXTERNALLY_TRUSTED_NOT_AUTHENTICATED",
                "training_eligible": False, "publication_eligible": False,
                "execution_authority": False, "operational": False}
    finally:
        for name in inserted:
            sys.modules.pop(name, None)


def main(argv=None):
    """One closed CLI operation; no model, admission, callable or target options."""
    code = _startup_code()
    if code is not None:
        _early_stop(code)
    try:
        result = _observe(_arguments(list(sys.argv[1:]) if argv is None else argv))
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        _require(len(raw) <= 4095, "BOOTSTRAP_RESULT_BYTE_CAP")
    except BaseException as error:
        # Includes syntax, filesystem and helper exceptions. Normal display must
        # never echo private paths, source snippets, argv or nested tracebacks.
        code = str(error) if type(error) is BootstrapError else "BOOTSTRAP_VERIFICATION_FAILED"
        raw = json.dumps({"state": "BLOCKED", "code": code, "training_eligible": False,
                          "publication_eligible": False, "execution_authority": False,
                          "operational": False}, sort_keys=True, separators=(",", ":"))
        sys.stdout.write(raw + "\n")
        return 70
    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
