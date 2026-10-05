# SPDX-License-Identifier: Apache-2.0
"""Bind admission to captured NeMo source, not an ambient installed package.

This is a source-integrity boundary in a trusted Python process, not a sandbox,
signed provenance, independent verification, or training/execution authority.
The caller supplies its bounded, alias-rejecting reader and separately verifies
the Forge gate wrapper before passing that wrapper to ``exec_gate``.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Iterator
import uuid


NEMO_REPOSITORY = "https://github.com/szl-holdings/szl-nemo"
NEMO_REVISION = "f7b33e1b1fe8f3b9b729a27567bcba375e2e0d6e"
SOURCE_HASHES = {
    "szl_nemo/__init__.py": "dbc411bfe684613ed321e3ce9b9d198bf43656ccc956a9093cc004a38d8d102c",
    "szl_nemo/engine.py": "9023b4c0663bf78a30aa3f94f2b87471528c36b24e0dc58a73e9306a7602dd21",
    "szl_nemo/envelope.py": "90206e0a4a426b85ca28a2380f3797b79ad07e6ebeb8ce9e284a49667af4f4f5",
    "szl_nemo/receipt.py": "9c478c25385aac860a0dad8903deb8b56593da9444a42bc1ff2e38056a820924",
    "szl_nemo/rules.py": "10330e75c0b44a0d7b415e75fadf0db1b7a3487b6b588da7ed4e839166583cdd",
    "szl_nemo/schema.py": "735aff6e000166508361c991104a7e800e9d6d72b94880402bdfe62c80b109a0",
}
MAX_SOURCE_BYTES = 256 * 1024
_MODULE_PATHS = {
    "szl_nemo": "szl_nemo/__init__.py",
    **{f"szl_nemo.{name}": f"szl_nemo/{name}.py"
       for name in ("engine", "envelope", "receipt", "rules", "schema")},
}
# Direct imports in the six pinned modules and the pinned Forge gate wrapper.
# Their standard-library dependencies remain part of the trusted interpreter.
_STDLIB_IMPORTS = frozenset({
    "__future__", "argparse", "collections.abc", "dataclasses", "hashlib",
    "json", "os", "re", "sys", "typing",
})


class BindingError(ValueError):
    """Required source or import provenance could not be established."""


class MissingSourceError(BindingError):
    """The caller did not supply the required explicit NeMo source tree."""


class BoundKernel:
    """Context-scoped access to the same captured kernel used by the gate."""

    def __init__(self, loader: "_CapturedLoader", hashes: dict[str, str]) -> None:
        self._loader = loader
        self.provenance = {
            "schema": "szl.receiptagent.v4-nemo-source-binding/v1",
            "repo": NEMO_REPOSITORY,
            "revision": NEMO_REVISION,
            "source_sha256": dict(hashes),
            "evidence_class": "MEASURED",
            "canonicalization": "local",
            "canonicalization_profile": "szl.lambda/v1",
            "key_trust": "REPO_DECLARED",
            "authenticated": False,
            "execution_authority": False,
        }

    def input_hash(self, *args: Any, **kwargs: Any) -> str:
        self._loader.require_active()
        return self._loader.load("szl_nemo.engine").input_hash(*args, **kwargs)

    def verify_chain(self, *args: Any, **kwargs: Any) -> bool:
        self._loader.require_active()
        return self._loader.load("szl_nemo.receipt").verify_chain(*args, **kwargs)

    def exec_gate(self, source: bytes, filename: str) -> ModuleType:
        """Execute caller-verified wrapper bytes using this kernel's imports."""
        return self._loader.exec_gate(source, filename)


class _CapturedLoader:
    def __init__(self, root: Path, sources: dict[str, bytes]) -> None:
        self.root = root
        self.sources = sources
        self.prefix = "_szl_v4_nemo_" + uuid.uuid4().hex
        while any(name == self.prefix or name.startswith(self.prefix + ".")
                  for name in sys.modules):
            self.prefix = "_szl_v4_nemo_" + uuid.uuid4().hex
        self.modules: dict[str, ModuleType] = {}
        self.registered: dict[str, ModuleType] = {}
        self.active = True
        self.gate_count = 0
        self.standard_import = builtins.__import__

    def require_active(self) -> None:
        if not self.active:
            raise BindingError("captured NeMo context is closed")

    def private_name(self, name: str) -> str:
        return self.prefix + name[len("szl_nemo"):]

    def module(self, name: str, filename: str, *, package: bool = False) -> ModuleType:
        module = ModuleType(name)
        module.__file__ = filename
        module.__package__ = self.prefix
        if package:
            # No filesystem search path: every NeMo import uses captured bytes.
            module.__path__ = []
        local_builtins = dict(vars(builtins))
        local_builtins["__import__"] = self.bound_import
        module.__dict__["__builtins__"] = local_builtins
        # dataclasses consults sys.modules for its defining module. Only unique
        # private names are registered; ambient szl_nemo entries are untouched.
        sys.modules[name] = module
        self.registered[name] = module
        return module

    def load(self, name: str) -> ModuleType:
        self.require_active()
        if name not in _MODULE_PATHS:
            raise ImportError("NeMo import is outside the captured source set")
        if name in self.modules:
            return self.modules[name]
        path = _MODULE_PATHS[name]
        module = self.module(self.private_name(name), str(self.root / path),
                             package=name == "szl_nemo")
        self.modules[name] = module
        exec(compile(self.sources[path], module.__file__, "exec", dont_inherit=True),
             module.__dict__)
        if name != "szl_nemo" and "szl_nemo" in self.modules:
            setattr(self.modules["szl_nemo"], name.rsplit(".", 1)[1], module)
        return module

    def bound_import(self, name: str, globals: Any = None, locals: Any = None,
                     fromlist: Any = (), level: int = 0) -> Any:
        self.require_active()
        if level:
            if level != 1 or not isinstance(globals, dict) or \
                    globals.get("__package__") != self.prefix:
                raise ImportError("relative import is outside the captured package")
            name = "szl_nemo" + ("." + name if name else "")
        if name == "szl_nemo" or name.startswith("szl_nemo."):
            module = self.load(name)
            if fromlist:
                if name == "szl_nemo":
                    for attribute in fromlist:
                        child = name + "." + attribute
                        if child in _MODULE_PATHS:
                            setattr(module, attribute, self.load(child))
                return module
            return self.load("szl_nemo")
        # Force the byte-pinned upstream receipt implementation's local
        # canonicalizer; never consult installed or preloaded evidence-core.
        if name == "szl_evidence_core" or name.startswith("szl_evidence_core."):
            raise ImportError("captured NeMo requires its local canonicalizer")
        if level or name not in _STDLIB_IMPORTS:
            raise ImportError("import is outside the audited dependency set")
        return self.standard_import(name, globals, locals, fromlist, 0)

    def exec_gate(self, source: bytes, filename: str) -> ModuleType:
        self.require_active()
        if type(source) is not bytes or not 0 < len(source) <= MAX_SOURCE_BYTES:
            raise BindingError("gate must be bounded caller-verified source bytes")
        if not isinstance(filename, str) or not filename:
            raise BindingError("gate filename must be an explicit nonempty string")
        name = self.prefix + ".gate_" + str(self.gate_count)
        self.gate_count += 1
        module = self.module(name, filename)
        try:
            exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
        except BaseException as exc:
            sys.modules.pop(name, None)
            self.registered.pop(name, None)
            if isinstance(exc, Exception) and not isinstance(exc, BindingError):
                raise BindingError("captured NeMo gate could not be executed") from exc
            raise
        return module

    def close(self) -> None:
        self.active = False
        for name in self.registered:
            sys.modules.pop(name, None)
        self.registered.clear()
        self.modules.clear()


@contextmanager
def load_kernel(source_root: Path,
                read_bytes: Callable[[Path, int], bytes]) -> Iterator[BoundKernel]:
    """Capture, hash-check, and load the declared six-file NeMo source closure.

    ``source_root`` must explicitly contain ``szl_nemo``. There is no package
    discovery, installation, download, bytecode loading, or second source read.
    Hash commitments identify repository-declared bytes, not a trusted signer.
    """
    if source_root is None:
        raise MissingSourceError("explicit NeMo source root is required")
    try:
        root = Path(source_root)
    except (TypeError, ValueError) as exc:
        raise MissingSourceError("explicit NeMo source root is required") from exc
    hashes = dict(SOURCE_HASHES)
    if set(hashes) != set(_MODULE_PATHS.values()):
        raise BindingError("NeMo commitment does not cover the six-file source closure")
    sources: dict[str, bytes] = {}
    for path, expected in hashes.items():
        try:
            source = read_bytes(root / path, MAX_SOURCE_BYTES)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise MissingSourceError("required explicit NeMo source is missing") from exc
        if type(source) is not bytes or not 0 < len(source) <= MAX_SOURCE_BYTES:
            raise BindingError("NeMo source must be bounded nonempty bytes")
        if hashlib.sha256(source).hexdigest() != expected:
            raise BindingError("NeMo source differs from its pinned byte commitment")
        sources[path] = source
    # Nothing above executes source. Every file is captured and verified before
    # creating any module, including __init__, schema, and optional imports.
    loader = _CapturedLoader(root, sources)
    try:
        try:
            loader.load("szl_nemo")
            receipt = loader.load("szl_nemo.receipt")
            if getattr(receipt, "_CANON_SOURCE", None) != "local" or \
                    getattr(receipt, "_CANON_PROFILE", None) != "szl.lambda/v1" or \
                    getattr(receipt, "_shared_canonical_bytes", object()) is not None:
                raise BindingError("captured NeMo did not select its local canonicalizer")
            for name, attribute in (("szl_nemo.engine", "input_hash"),
                                    ("szl_nemo.receipt", "verify_chain")):
                if not callable(getattr(loader.load(name), attribute, None)):
                    raise BindingError("captured NeMo is missing a required verifier")
        except Exception as exc:
            if isinstance(exc, BindingError):
                raise
            raise BindingError("captured NeMo source could not be executed") from exc
        yield BoundKernel(loader, hashes)
    finally:
        loader.close()
