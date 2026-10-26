"""SIMULATED revision-boundary tests: no frameworks, models, data or network."""
from __future__ import annotations

import builtins
from contextlib import contextmanager, nullcontext
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = (ROOT / "tools/estate_payload/kit",
             ROOT / "frontier/brain-navigator-r2-candidate-20261001")
PIN = "0123456789abcdef" * 2 + "01234567"
BAD = (None, True, False, 40, 1.5, b"a" * 40, [], {}, "", "main", "v1",
       "a" * 39, "a" * 41, "g" * 40, "A" * 40, " " + "a" * 40,
       "a" * 40 + "\n", "a" * 19 + "\n" + "a" * 20, "ａ" * 40)
FRAMEWORKS = {"torch", "transformers", "peft", "unsloth", "numpy", "huggingface_hub"}


def load_source(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


@contextmanager
def consumer(directory, entry=None):
    saved_path = sys.path[:]
    with patch.dict(sys.modules):
        try:
            lib = load_source(directory / "candidate_lib.py", "_synthetic_revision_lib")
            sys.modules["candidate_lib"] = lib
            module = load_source(directory / (entry + ".py"), "_synthetic_entry") if entry else None
            yield lib, module
        finally:
            sys.path[:] = saved_path


@contextmanager
def no_framework_imports():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in FRAMEWORKS:
            raise AssertionError("framework import before pin rejection: " + name)
        return original(name, *args, **kwargs)
    with patch.object(builtins, "__import__", side_effect=guarded):
        yield


def candidate(revision):
    return {"actual_training_base": {"repo_id": "SAMPLE/base", "revision": revision},
            "training_recipe": {"max_length": 8}}


class CandidateRevisionBoundaryTests(unittest.TestCase):
    def test_exact_pin_is_returned_unchanged_without_imports(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory), consumer(directory) as (lib, _), no_framework_imports():
                self.assertEqual(lib.require_pinned_revision(PIN), PIN)

    def test_mutable_malformed_and_wrong_type_pins_rejected(self):
        for directory in CONSUMERS:
            with consumer(directory) as (lib, _), no_framework_imports():
                for revision in BAD:
                    with self.subTest(consumer=directory, revision=revision):
                        with self.assertRaisesRegex(ValueError, "BASE_REVISION_UNPINNED"):
                            lib.require_pinned_revision(revision)

    def test_direct_loaders_reject_before_frameworks_even_with_cpu_permission(self):
        for directory in CONSUMERS:
            with consumer(directory) as (lib, _), no_framework_imports():
                for revision in BAD:
                    with self.subTest(consumer=directory, revision=revision):
                        with self.assertRaisesRegex(ValueError, "BASE_REVISION_UNPINNED"):
                            lib.load_tokenizer("SAMPLE/base", revision)
                        for training in (False, True):
                            for cpu in (False, True):
                                with self.assertRaisesRegex(ValueError, "BASE_REVISION_UNPINNED"):
                                    lib.load_base_model(candidate(revision), training, allow_cpu=cpu)

    def test_runtime_entrypoints_reject_before_reports_data_gpu_network_or_outputs(self):
        for directory in CONSUMERS:
            for entry, argv in (("train_candidate", ["--smoke"]), ("evaluate_candidate", []),
                                ("export_gguf", []), ("qualify_runtime", []),
                                ("qualify_runtime", ["--offline", "--allow-cpu"])):
                with consumer(directory, entry) as (lib, module):
                    for revision in (None, "main", "g" * 40, "a" * 40 + "\n", True):
                        value = candidate(revision)
                        initial = copy.deepcopy(value)
                        with self.subTest(consumer=directory, entry=entry, revision=revision), \
                                patch.object(lib, "load_candidate", return_value=value), \
                                patch.object(sys, "argv", ["SIMULATED", *argv]), \
                                patch.object(lib, "load_gates", side_effect=AssertionError("gate read")) as gates, \
                                patch.object(lib, "load_splits", side_effect=AssertionError("data read")) as splits, \
                                patch.object(lib, "require_report", side_effect=AssertionError("report read")) as reports, \
                                patch.object(lib, "installed_versions", side_effect=AssertionError("package probe")) as versions, \
                                patch.object(lib, "gpu_snapshot", side_effect=AssertionError("GPU probe")) as gpu, \
                                patch.object(lib, "save_candidate", side_effect=AssertionError("lock write")) as save, \
                                patch.object(lib, "write_json", side_effect=AssertionError("report write")) as write, \
                                patch.object(Path, "mkdir", side_effect=AssertionError("output creation")) as mkdir, \
                                no_framework_imports():
                            with self.assertRaisesRegex(ValueError, "BASE_REVISION_UNPINNED"):
                                module.main()
                            for effect in (gates, splits, reports, versions, gpu, save, write, mkdir):
                                effect.assert_not_called()
                            self.assertEqual(value, initial)

    def test_metadata_only_read_still_allows_unbound_candidate(self):
        for directory in CONSUMERS:
            value = candidate(None)
            with self.subTest(consumer=directory), consumer(directory) as (lib, _), \
                    patch.object(lib, "read_json", return_value=value), no_framework_imports():
                self.assertIs(lib.load_candidate(), value)
                self.assertIsNone(value["actual_training_base"]["revision"])

    def test_plain_mocked_model_and_tokenizer_receive_exact_revision(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory), consumer(directory) as (lib, _):
                model = object()
                tok = types.SimpleNamespace(pad_token=None, eos_token="SAMPLE_EOS")
                loader, tokenizer = Mock(return_value=model), Mock(return_value=tok)
                torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                              float32="SAMPLE_FP32")
                transformers = types.SimpleNamespace(AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=loader),
                                                     AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer))
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "unsloth": None}), \
                        patch("builtins.print"):
                    result = lib.load_base_model(candidate(PIN), for_training=False, allow_cpu=True)
                self.assertIs(result[0], model)
                self.assertEqual(loader.call_args.kwargs["revision"], PIN)
                self.assertEqual(tokenizer.call_args.kwargs["revision"], PIN)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(tok.pad_token, tok.eos_token)

    def test_mocked_unsloth_path_receives_exact_revision(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory), consumer(directory) as (lib, _):
                model = object()
                tok = types.SimpleNamespace(pad_token=None, eos_token="SAMPLE_EOS")
                loader = Mock(return_value=(model, tok))
                torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
                unsloth = types.SimpleNamespace(FastLanguageModel=types.SimpleNamespace(from_pretrained=loader))
                with patch.dict(sys.modules, {"torch": torch, "unsloth": unsloth}):
                    result = lib.load_base_model(candidate(PIN), for_training=False)
                backend = "unsloth-4bit" if directory == CONSUMERS[0] else "unsloth"
                self.assertEqual(result, (model, tok, backend))
                self.assertEqual(loader.call_args.kwargs["revision"], PIN)


class QualifierResolvedRevisionTests(unittest.TestCase):
    """All observed environments, Hub replies and write sinks are SIMULATED."""

    def qualify(self, directory, metadata, *, offline=False, frozen=False,
                relock=False, drift=False, environment_ok=True, error=None):
        with consumer(directory, "qualify_runtime") as (lib, module):
            value = candidate(PIN)
            value["candidate_id"] = "SIMULATED_ONLY"
            versions = {name: "SIMULATED" for name in lib.LOCK_PACKAGES}
            versions["python"] = "SIMULATED_PYTHON"
            if frozen:
                value["runtime_lock"] = dict(versions)
                if drift:
                    value["runtime_lock"]["python"] = "SIMULATED_DRIFT"
            initial = copy.deepcopy(value)
            observed = dict(versions)
            if not environment_ok:
                observed["torch"] = None
            query = Mock(return_value=metadata, side_effect=error)
            factory = Mock(return_value=types.SimpleNamespace(model_info=query))
            hub = types.SimpleNamespace(HfApi=factory)
            argv = ["SIMULATED", *(["--offline"] if offline else []),
                    *(["--relock"] if relock else [])]
            with patch.object(lib, "load_candidate", return_value=value), \
                    patch.object(lib, "installed_versions", return_value=observed), \
                    patch.object(lib, "gpu_snapshot", return_value={
                        "cuda_available": True, "free_gib": "8", "device": "SIMULATED"}), \
                    patch.object(module.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20 * 2**30)), \
                    patch.object(lib, "host_info", return_value={"kind": "SIMULATED"}), \
                    patch.object(lib, "save_candidate") as save, \
                    patch.object(lib, "write_json") as write, \
                    patch.object(sys, "argv", argv), \
                    patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    patch("builtins.print"), \
                    (no_framework_imports() if offline else nullcontext()):
                code = module.main()
            write.assert_called_once()
            report = write.call_args.args[1]
            return {"code": code, "report": report, "saved": save.call_count,
                    "candidate": value, "initial": initial, "query": query, "factory": factory}

    def assert_blocked(self, result):
        self.assertNotEqual(result["code"], 0)
        self.assertEqual(result["report"]["verdict"], "FAIL")
        self.assertEqual(result["saved"], 0)
        self.assertEqual(result["candidate"], result["initial"])

    def test_wrong_missing_malformed_resolution_never_locks(self):
        metadata = [types.SimpleNamespace()]
        metadata += [types.SimpleNamespace(sha=value) for value in (*BAD, "b" * 40)]
        for directory in CONSUMERS:
            for item in metadata:
                for frozen, relock in ((False, False), (True, False), (True, True), (False, True)):
                    with self.subTest(consumer=directory, metadata=item, frozen=frozen, relock=relock):
                        result = self.qualify(directory, item, frozen=frozen, relock=relock)
                        self.assert_blocked(result)
                        rows = {row["check"]: row for row in result["report"]["checks"]}
                        self.assertEqual(rows["base_revision_resolved"]["status"], "FAIL")
                        result["query"].assert_called_once_with("SAMPLE/base", revision=PIN)

    def test_offline_unknown_never_qualifies_or_locks(self):
        for directory in CONSUMERS:
            for frozen, relock in ((False, False), (True, False), (True, True), (False, True)):
                with self.subTest(consumer=directory, frozen=frozen, relock=relock):
                    result = self.qualify(directory, types.SimpleNamespace(sha=PIN),
                                          offline=True, frozen=frozen, relock=relock)
                    self.assert_blocked(result)
                    rows = {row["check"]: row for row in result["report"]["checks"]}
                    self.assertEqual(rows["base_revision_resolved"]["status"], "UNKNOWN")
                    result["factory"].assert_not_called()
                    result["query"].assert_not_called()

    def test_unavailable_hub_never_qualifies_or_locks(self):
        for directory in CONSUMERS:
            for frozen, relock in ((False, False), (True, False), (True, True)):
                with self.subTest(consumer=directory, frozen=frozen, relock=relock):
                    result = self.qualify(directory, None, error=RuntimeError("SIMULATED_UNAVAILABLE"),
                                          frozen=frozen, relock=relock)
                    self.assert_blocked(result)
                    rows = {row["check"]: row for row in result["report"]["checks"]}
                    self.assertEqual(rows["base_revision_resolved"]["status"], "FAIL")

    def test_exact_resolution_retains_environment_and_lock_gates(self):
        for directory in CONSUMERS:
            for frozen, relock, drift, environment_ok in (
                    (False, False, False, True), (True, False, False, True),
                    (True, False, True, True), (True, True, True, True),
                    (False, False, False, False), (True, True, False, False)):
                with self.subTest(consumer=directory, frozen=frozen, relock=relock,
                                  drift=drift, environment_ok=environment_ok):
                    result = self.qualify(directory, types.SimpleNamespace(sha=PIN), frozen=frozen,
                                          relock=relock, drift=drift, environment_ok=environment_ok)
                    rows = {row["check"]: row for row in result["report"]["checks"]}
                    self.assertEqual(rows["base_revision_resolved"]["status"], "PASS")
                    result["query"].assert_called_once_with("SAMPLE/base", revision=PIN)
                    if not environment_ok or (drift and not relock):
                        self.assert_blocked(result)
                    else:
                        self.assertEqual(result["code"], 0)
                        self.assertEqual(result["report"]["verdict"], "PASS")
                        self.assertEqual(result["saved"], int(not frozen or relock))


if __name__ == "__main__":
    unittest.main()
