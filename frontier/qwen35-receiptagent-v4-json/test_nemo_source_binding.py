# SPDX-License-Identifier: Apache-2.0
"""SIMULATED source-loader tests, not kernel, model, or training evidence.

All six-module source fixtures below are deliberately synthetic. Patching their
hash commitments exercises loader mechanics only; the genuine upstream kernel
must be byte-checked and exercised separately. No curriculum or model is read.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemo_source_binding as binding


def simulated_sources():
    """A tiny package with the same import shape, never the doctrine kernel."""
    return {
        "szl_nemo/__init__.py": b"from .engine import input_hash\n",
        "szl_nemo/engine.py": (
            b"from .rules import MARKER\n"
            b"def input_hash(value):\n    return MARKER + ':' + value\n"
        ),
        "szl_nemo/envelope.py": b"ENVELOPE = 'SIMULATED'\n",
        "szl_nemo/receipt.py": (
            b"try:\n"
            b"    from szl_evidence_core.canonical import canonical_bytes as _shared_canonical_bytes\n"
            b"    _CANON_SOURCE = 'shared'\n"
            b"except ImportError:\n"
            b"    _shared_canonical_bytes = None\n"
            b"    _CANON_SOURCE = 'local'\n"
            b"_CANON_PROFILE = 'szl.lambda/v1'\n"
            b"def verify_chain(value):\n    return value == ['SIMULATED']\n"
        ),
        "szl_nemo/rules.py": b"MARKER = 'SIMULATED-CAPTURED'\n",
        "szl_nemo/schema.py": b"SCHEMA = 'SIMULATED'\n",
    }


def private_modules():
    return {name for name in sys.modules if name.startswith("_szl_v4_nemo_")}


class PoisonModule(ModuleType):
    """Records any attribute lookup that would consume an ambient package."""
    def __getattr__(self, name):
        raise AssertionError("ambient poison module was consulted: " + name)


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        # No directory or source files are created. Only the injected reader can
        # supply fixture bytes; every path it receives is checked against root.
        self.root = Path(__file__).resolve().parent / "SIMULATED-no-source-on-disk"
        self.sources = simulated_sources()
        self.reads = []
        self.before = private_modules()

    def tearDown(self):
        self.assertEqual(private_modules(), self.before, "private modules leaked")

    def read_bytes(self, path, limit):
        self.assertEqual(limit, binding.MAX_SOURCE_BYTES)
        self.assertEqual(path.parent, self.root / "szl_nemo")
        relative = path.relative_to(self.root).as_posix()
        self.assertIn(relative, self.sources)
        self.assertNotIn(relative, self.reads, "source was read twice")
        self.reads.append(relative)
        return self.sources[relative]

    @contextmanager
    def fixture(self, *, reader=None, hashes=None):
        commitments = hashes if hashes is not None else {
            path: hashlib.sha256(raw).hexdigest() for path, raw in self.sources.items()
        }
        with patch.object(binding, "SOURCE_HASHES", commitments):
            with binding.load_kernel(self.root, reader or self.read_bytes) as kernel:
                yield kernel

    def test_all_six_files_are_captured_before_any_loader_is_created(self):
        actual_loader = binding._CapturedLoader

        def inspected_loader(root, captured):
            self.assertEqual(set(self.reads), set(self.sources))
            self.assertEqual(captured, self.sources)
            return actual_loader(root, captured)

        with patch.object(binding, "_CapturedLoader", side_effect=inspected_loader):
            with self.fixture() as kernel:
                self.assertEqual(kernel.input_hash("value"), "SIMULATED-CAPTURED:value")
        self.assertEqual(len(self.reads), 6)

    def test_late_hash_drift_prevents_even_init_execution(self):
        self.sources["szl_nemo/__init__.py"] = b"raise RuntimeError('must not execute')\n"
        hashes = {path: hashlib.sha256(raw).hexdigest() for path, raw in self.sources.items()}
        hashes["szl_nemo/schema.py"] = "0" * 64
        with patch.object(binding, "_CapturedLoader") as loader:
            with self.assertRaisesRegex(binding.BindingError, "pinned byte commitment"):
                with self.fixture(hashes=hashes):
                    self.fail("drifted source was accepted")
            loader.assert_not_called()

    def test_missing_source_is_distinct_and_executes_nothing(self):
        def missing(path, limit):
            if path.name == "schema.py":
                raise FileNotFoundError(path)
            return self.read_bytes(path, limit)

        with patch.object(binding, "_CapturedLoader") as loader:
            with self.assertRaises(binding.MissingSourceError):
                with self.fixture(reader=missing):
                    self.fail("missing source was accepted")
            loader.assert_not_called()

    def test_missing_explicit_root_does_not_invoke_reader(self):
        with patch.object(binding, "_CapturedLoader") as loader:
            with self.assertRaises(binding.MissingSourceError):
                with binding.load_kernel(None, self.read_bytes):
                    self.fail("implicit source discovery was permitted")
            loader.assert_not_called()
        self.assertEqual(self.reads, [])

    def test_capture_is_used_without_rereading_mutated_source(self):
        original = dict(self.sources)

        def mutate_after_last_read(path, limit):
            raw = self.read_bytes(path, limit)
            if len(self.reads) == 6:
                self.sources["szl_nemo/rules.py"] = b"MARKER = 'MUTATED-AFTER-CAPTURE'\n"
            return raw

        with self.fixture(reader=mutate_after_last_read) as kernel:
            self.assertEqual(kernel.input_hash("one"), "SIMULATED-CAPTURED:one")
            self.assertEqual(kernel.input_hash("two"), "SIMULATED-CAPTURED:two")
            self.assertEqual(kernel.provenance["source_sha256"]["szl_nemo/rules.py"],
                             hashlib.sha256(original["szl_nemo/rules.py"]).hexdigest())
        self.assertEqual(len(self.reads), 6)

    def test_reader_receives_only_six_scoped_paths_and_its_rejection_propagates(self):
        with self.fixture():
            pass
        self.assertEqual(set(self.reads), set(self.sources))
        self.reads.clear()

        def reject_alias(path, limit):
            self.assertEqual(path.parent, self.root / "szl_nemo")
            raise binding.BindingError("SIMULATED reader rejects linked source")

        with self.assertRaisesRegex(binding.BindingError, "reader rejects linked source"):
            with self.fixture(reader=reject_alias):
                self.fail("reader scope rejection was ignored")

    def test_commitment_scope_changes_fail_before_reader(self):
        for change in ("missing", "extra", "traversal"):
            with self.subTest(change=change):
                hashes = {path: hashlib.sha256(raw).hexdigest()
                          for path, raw in self.sources.items()}
                if change != "extra":
                    hashes.pop("szl_nemo/schema.py")
                if change != "missing":
                    hashes["../schema.py" if change == "traversal" else "szl_nemo/extra.py"] = "0" * 64
                with self.assertRaisesRegex(binding.BindingError, "six-file source closure"):
                    with self.fixture(hashes=hashes):
                        self.fail("invalid commitment closure was accepted")
        self.assertEqual(self.reads, [])

    def test_source_requires_exact_bounded_nonempty_bytes(self):
        for invalid in (b"", "text", bytearray(b"x"), memoryview(b"x"),
                        b"x" * (binding.MAX_SOURCE_BYTES + 1)):
            with self.subTest(kind=type(invalid).__name__, size=len(invalid)):
                with self.assertRaisesRegex(binding.BindingError, "bounded nonempty bytes"):
                    with self.fixture(reader=lambda path, limit: invalid):
                        self.fail("invalid source bytes were accepted")

    def test_ambient_nemo_and_evidence_core_are_unused_and_unchanged(self):
        names = ["szl_nemo", "szl_nemo.engine", "szl_nemo.receipt", "szl_nemo.rules",
                 "szl_evidence_core", "szl_evidence_core.canonical"]
        poison = {name: PoisonModule(name) for name in names}
        with patch.dict(sys.modules, poison):
            with self.fixture() as kernel:
                self.assertEqual(kernel.input_hash("x"), "SIMULATED-CAPTURED:x")
                self.assertTrue(kernel.verify_chain(["SIMULATED"]))
                gate = kernel.exec_gate(
                    b"import szl_nemo.engine\n"
                    b"from szl_nemo.receipt import verify_chain\n"
                    b"VALUE = szl_nemo.engine.input_hash('gate')\n"
                    b"VALID = verify_chain(['SIMULATED'])\n", "SIMULATED-gate.py")
                self.assertEqual(gate.VALUE, "SIMULATED-CAPTURED:gate")
                self.assertTrue(gate.VALID)
                self.assertEqual(kernel.provenance["canonicalization"], "local")
                for name, module in poison.items():
                    self.assertIs(sys.modules[name], module)
            for name, module in poison.items():
                self.assertIs(sys.modules[name], module)

    def test_local_profile_and_non_authority_provenance_are_explicit(self):
        with self.fixture() as kernel:
            provenance = kernel.provenance
            self.assertEqual(provenance["schema"], "szl.receiptagent.v4-nemo-source-binding/v1")
            self.assertEqual(provenance["repo"], binding.NEMO_REPOSITORY)
            self.assertEqual(provenance["revision"], binding.NEMO_REVISION)
            self.assertEqual(provenance["canonicalization_profile"], "szl.lambda/v1")
            self.assertEqual(provenance["key_trust"], "REPO_DECLARED")
            self.assertIs(provenance["authenticated"], False)
            self.assertIs(provenance["execution_authority"], False)
            self.assertEqual(provenance["source_sha256"], binding.SOURCE_HASHES)
            self.assertIsNot(provenance["source_sha256"], binding.SOURCE_HASHES)

    def test_nonlocal_or_wrong_profile_receipts_are_rejected(self):
        original = self.sources["szl_nemo/receipt.py"]
        for assignment in (b"_CANON_SOURCE = 'shared'\n",
                           b"_shared_canonical_bytes = lambda value: b'wrong'\n",
                           b"_CANON_PROFILE = 'wrong/profile'\n"):
            with self.subTest(assignment=assignment):
                self.sources["szl_nemo/receipt.py"] = original + assignment
                self.reads.clear()
                with self.assertRaises(binding.BindingError):
                    with self.fixture():
                        self.fail("nonlocal or wrong canonicalization was accepted")
                self.assertEqual(private_modules(), self.before)

    def test_required_verifiers_must_be_callable(self):
        for path, assignment in (("szl_nemo/engine.py", b"input_hash = None\n"),
                                 ("szl_nemo/receipt.py", b"verify_chain = None\n")):
            with self.subTest(path=path):
                self.sources = simulated_sources()
                self.sources[path] += assignment
                self.reads.clear()
                with self.assertRaisesRegex(binding.BindingError, "required verifier"):
                    with self.fixture():
                        self.fail("non-callable verifier was accepted")

    def test_cleanup_on_success_and_no_changes_to_existing_private_like_module(self):
        sentinel_name = "_szl_v4_nemo_SIMULATED_existing"
        sentinel = ModuleType(sentinel_name)
        with patch.dict(sys.modules, {sentinel_name: sentinel}):
            before = private_modules()
            with self.fixture() as kernel:
                gate = kernel.exec_gate(b"VALUE = 'SIMULATED'\n", "SIMULATED-gate.py")
                self.assertIn(gate.__name__, sys.modules)
                self.assertGreater(len(private_modules()), len(before))
            self.assertEqual(private_modules(), before)
            self.assertIs(sys.modules[sentinel_name], sentinel)

    def test_cleanup_when_initialization_raises(self):
        self.sources["szl_nemo/__init__.py"] = b"raise RuntimeError('SIMULATED init failure')\n"
        with self.assertRaisesRegex(binding.BindingError, "source could not be executed"):
            with self.fixture():
                self.fail("failing initialization yielded a kernel")

    def test_cleanup_when_context_body_raises(self):
        with self.assertRaisesRegex(RuntimeError, "SIMULATED body"):
            with self.fixture() as kernel:
                self.assertTrue(kernel.verify_chain(["SIMULATED"]))
                raise RuntimeError("SIMULATED body failure")

    def test_failed_gate_is_removed_while_context_remains_usable(self):
        with self.fixture() as kernel:
            before = private_modules()
            with self.assertRaisesRegex(binding.BindingError, "gate could not be executed"):
                kernel.exec_gate(b"raise RuntimeError('SIMULATED gate failure')\n", "SIMULATED-gate.py")
            self.assertEqual(private_modules(), before)
            self.assertTrue(kernel.verify_chain(["SIMULATED"]))

    def test_context_is_closed_after_exit(self):
        with self.fixture() as kernel:
            pass
        for call in (lambda: kernel.input_hash("x"),
                     lambda: kernel.verify_chain([]),
                     lambda: kernel.exec_gate(b"VALUE = 1\n", "SIMULATED-gate.py")):
            with self.subTest(call=call):
                with self.assertRaisesRegex(binding.BindingError, "context is closed"):
                    call()

    def test_gate_rejects_unbounded_or_nonbyte_source_and_missing_filename(self):
        with self.fixture() as kernel:
            for source in (b"", "VALUE = 1", bytearray(b"VALUE = 1"),
                           b"x" * (binding.MAX_SOURCE_BYTES + 1)):
                with self.subTest(source_type=type(source).__name__):
                    with self.assertRaises(binding.BindingError):
                        kernel.exec_gate(source, "SIMULATED-gate.py")
            for filename in ("", None, Path("SIMULATED-gate.py")):
                with self.subTest(filename=filename):
                    with self.assertRaises(binding.BindingError):
                        kernel.exec_gate(b"VALUE = 1\n", filename)

    def test_allowed_stdlib_and_relative_imports_work_with_dataclasses(self):
        self.sources["szl_nemo/schema.py"] = (
            b"from __future__ import annotations\n"
            b"from dataclasses import dataclass\n"
            b"@dataclass\nclass Sample:\n    value: str\n"
        )
        with self.fixture() as kernel:
            gate = kernel.exec_gate(
                b"import hashlib\nimport json\nfrom typing import Any\n"
                b"from .schema import Sample\n"
                b"VALUE = Sample(json.loads('{\"value\": \"SIMULATED\"}')[\"value\"])\n",
                "SIMULATED-gate.py")
            self.assertEqual(gate.VALUE.value, "SIMULATED")

    def test_direct_imports_outside_audited_set_fail_closed(self):
        imports = (b"import socket\n", b"import pathlib\n", b"import os.path\n",
                   b"import importlib\n", b"from szl_nemo.unknown import value\n",
                   b"from .. import something\n", b"import szl_evidence_core\n")
        with self.fixture() as kernel:
            for source in imports:
                with self.subTest(source=source):
                    before = private_modules()
                    with self.assertRaises(binding.BindingError):
                        kernel.exec_gate(source, "SIMULATED-forbidden-import.py")
                    self.assertEqual(private_modules(), before)

    def test_forbidden_import_in_kernel_fails_before_yield(self):
        self.sources["szl_nemo/__init__.py"] = b"import socket\n"
        with self.assertRaises(binding.BindingError):
            with self.fixture():
                self.fail("kernel imported an unaudited direct dependency")


if __name__ == "__main__":
    unittest.main()
