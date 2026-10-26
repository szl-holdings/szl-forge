"""SIMULATED, stdlib-only adapter wiring tests; no model/data/VM access.

All checkpoint bytes, model objects, scored rows and report sinks are in memory.
Only checked-in source is read. The payload generator is never imported or run.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import Mock, mock_open, patch


ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = (
    ROOT / "tools/estate_payload/kit",
    ROOT / "frontier/brain-navigator-r2-candidate-20261001",
)
CANONICAL = ROOT / "chaski/adapter_guard.py"
KEYS = (
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
)


def normalized(source):
    return source.replace(b"\r\n", b"\n")


def load_source(path, name):
    """Compile read bytes, avoiding import machinery and bytecode files."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


class SimulatedBase:
    def __init__(self):
        self.config = types.SimpleNamespace(model_type="SIMULATED_BASE")

    def eval(self):
        return self


class SimulatedAdapter:
    def __init__(self, base, *, keys=KEYS, namespace="default", **status):
        self.base_model = types.SimpleNamespace(model=base)
        self.keys = tuple(keys)
        self.namespace = namespace
        self.status = types.SimpleNamespace(
            enabled=True, active_adapters=["default"], num_adapter_layers=1,
            available_adapters=["default"], merged_adapters=[],
        )
        for key, value in status.items():
            setattr(self.status, key, value)

    def named_parameters(self):
        return [(key.removesuffix(".weight") + f".{self.namespace}.weight", None)
                for key in self.keys]

    def get_model_status(self):
        return self.status

    def eval(self):
        return self


@contextmanager
def consumer(directory, *, evaluator=False):
    """Restore both global import structures even if loading or a test fails."""
    saved_path = sys.path[:]
    with patch.dict(sys.modules):
        try:
            lib = load_source(directory / "candidate_lib.py", "_simulated_candidate_lib")
            sys.modules["candidate_lib"] = lib
            sys.modules["peft"] = types.SimpleNamespace(PeftModel=Mock())
            sys.modules["transformers"] = types.SimpleNamespace(__version__="SIMULATED")
            sys.modules["torch"] = None
            ev = (load_source(directory / "evaluate_candidate.py", "_simulated_evaluator")
                  if evaluator else None)
            yield lib, ev, sys.modules["peft"].PeftModel
        finally:
            sys.path[:] = saved_path


class ReadBuffer(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)

    def fileno(self):
        return 42


class GuardSource:
    """A single fake sibling path, not a filesystem target."""
    def __init__(self, source=None, error=None):
        self.source = source
        self.error = error
        self.opens = 0
        self.handles = []

    def __truediv__(self, name):
        if name != "adapter_guard.py":
            raise AssertionError(name)
        return self

    def __str__(self):
        return "SIMULATED/adapter_guard.py"

    def open(self, mode):
        if mode != "rb":
            raise AssertionError(mode)
        self.opens += 1
        if self.error:
            raise self.error
        handle = ReadBuffer(self.source)
        self.handles.append(handle)
        return handle


class CandidateAdapterAttachmentTests(unittest.TestCase):
    def test_exact_canonical_copies_and_consumer_pins(self):
        source = normalized(CANONICAL.read_bytes())
        digest = hashlib.sha256(source).hexdigest()
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, _):
                self.assertEqual(normalized((directory / "adapter_guard.py").read_bytes()), source)
                self.assertEqual(lib.ADAPTER_GUARD_SHA256, digest)

    def test_missing_corrupt_oversize_and_invalid_encoding_fail_before_peft(self):
        source = normalized(CANONICAL.read_bytes())
        cases = {
            "missing": GuardSource(error=FileNotFoundError("SIMULATED")),
            "changed": GuardSource(source + b"# changed\n"),
            "oversize": GuardSource(b"x" * 65537),
            "invalid_utf8": GuardSource(source + b"\xff"),
        }
        for directory in CONSUMERS:
            for name, fake in cases.items():
                with self.subTest(consumer=directory.name, failure=name), consumer(directory) as (lib, _, peft):
                    with patch.object(lib, "HERE", fake):
                        with self.assertRaisesRegex(RuntimeError, "ADAPTER_GUARD_UNAVAILABLE"):
                            lib.attach_adapter(SimulatedBase(), "SIMULATED_ADAPTER")
                    peft.from_pretrained.assert_not_called()

    def test_shadow_modules_do_not_replace_packaged_guard(self):
        impostor = types.SimpleNamespace(checkpoint_keys=Mock(side_effect=AssertionError("shadow used")))
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, peft):
                with patch.dict(sys.modules, {"adapter_guard": impostor, "_szl_candidate_adapter_guard": impostor}):
                    guard = lib._load_adapter_guard()
                    self.assertIsNot(guard, impostor)
                    self.assertTrue(callable(guard.assert_adapter_applied))
                    with patch.object(lib, "HERE", GuardSource(error=FileNotFoundError())):
                        with self.assertRaisesRegex(RuntimeError, "ADAPTER_GUARD_UNAVAILABLE"):
                            lib.attach_adapter(SimulatedBase(), "SIMULATED_ADAPTER")
                impostor.checkpoint_keys.assert_not_called()
                peft.from_pretrained.assert_not_called()

    def test_crlf_normalization_compiles_the_one_verified_buffer(self):
        source = normalized(CANONICAL.read_bytes())
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, _):
                fake = GuardSource(source.replace(b"\n", b"\r\n"))
                real_compile = builtins.compile
                with patch.object(lib, "HERE", fake), patch("builtins.compile", wraps=real_compile) as compiler:
                    guard = lib._load_adapter_guard()
                self.assertEqual(fake.opens, 1)
                self.assertEqual(fake.handles[0].read_sizes, [65537])
                compiler.assert_called_once()
                self.assertEqual(compiler.call_args.args[0], source)
                self.assertEqual(compiler.call_args.args[1:], (str(fake), "exec"))
                self.assertEqual(guard.__file__, str(fake))

    def test_guard_load_has_no_framework_import_requirement(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, _):
                with patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
                    self.assertTrue(callable(lib._load_adapter_guard().checkpoint_keys))

    def test_complete_adapter_preserves_model_api_and_reports_actual_classes(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, peft):
                base = SimulatedBase()
                wrapped = SimulatedAdapter(base)
                peft.from_pretrained.return_value = wrapped
                guard = lib._load_adapter_guard()
                with patch.object(lib, "_load_adapter_guard", return_value=guard), \
                        patch.object(guard, "checkpoint_keys", return_value=list(KEYS)), \
                        patch("importlib.metadata.version", return_value="SIMULATED_PEFT"):
                    self.assertIs(lib.attach_adapter(base, "SIMULATED_ADAPTER"), wrapped)
                    result, evidence = lib.attach_adapter(base, "SIMULATED_ADAPTER", with_evidence=True)
                self.assertIs(result, wrapped)
                peft.from_pretrained.assert_called_with(
                    base, "SIMULATED_ADAPTER", adapter_name="default", is_trainable=False,
                )
                self.assertEqual(evidence["wrapper_class"], "SimulatedAdapter")
                self.assertEqual(evidence["model_class"], "SimulatedBase")
                self.assertEqual(evidence["model_type"], "SIMULATED_BASE")
                self.assertEqual(evidence["transformers"], "SIMULATED")
                self.assertEqual(evidence["peft"], "SIMULATED_PEFT")
                self.assertEqual(evidence["mode"], "adapter_structural_coverage")
                report = evidence["adapter_keys"]
                self.assertTrue(report["fully_applied"])
                self.assertEqual(report["checkpoint_tensors"], len(KEYS))
                self.assertEqual(report["expected_adapter"], "default")
                self.assertEqual(report["adapter_activation"]["active_adapters"], ["default"])
                self.assertEqual(report["numerical_application"], "UNKNOWN")
                self.assertEqual(report["coverage_basis"], "parameter_names_and_model_status")
                self.assertEqual(evidence["adapter_guard"]["source_sha256"], lib.ADAPTER_GUARD_SHA256)
                self.assertEqual(evidence["adapter_guard"]["trust"], "REPO_DECLARED")

    def test_default_namespace_inactive_merged_zero_partial_and_empty_reject(self):
        cases = {
            "other_namespace": {"namespace": "other"},
            "inactive": {"enabled": False},
            "wrong_active": {"active_adapters": ["other"], "available_adapters": ["default", "other"]},
            "merged": {"merged_adapters": ["default"]},
            "zero": {"keys": ()},
            "partial": {"keys": KEYS[:1]},
            "empty_checkpoint": {},
        }
        for directory in CONSUMERS:
            for name, kwargs in cases.items():
                with self.subTest(consumer=directory.name, failure=name), consumer(directory) as (lib, _, peft):
                    base = SimulatedBase()
                    peft.from_pretrained.return_value = SimulatedAdapter(base, **kwargs)
                    guard = lib._load_adapter_guard()
                    checkpoint = [] if name == "empty_checkpoint" else list(KEYS)
                    with patch.object(lib, "_load_adapter_guard", return_value=guard), \
                            patch.object(guard, "checkpoint_keys", return_value=checkpoint):
                        with self.assertRaises(guard.AdapterNotApplied):
                            lib.attach_adapter(base, "SIMULATED_ADAPTER", with_evidence=True)
                    if name == "empty_checkpoint":
                        peft.from_pretrained.assert_not_called()
                    else:
                        peft.from_pretrained.assert_called_once()

    def test_malformed_checkpoint_real_parser_rejects_before_peft(self):
        # Valid prefix length, invalid descriptor: entirely synthetic, no tensor data.
        header = json.dumps({KEYS[0]: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        malformed = struct.pack("<Q", len(header)) + header  # payload deliberately absent
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, peft):
                guard = lib._load_adapter_guard()
                handle = ReadBuffer(malformed)
                with patch.object(lib, "_load_adapter_guard", return_value=guard), \
                        patch.object(Path, "open", return_value=handle), \
                        patch.object(guard.os, "fstat", return_value=types.SimpleNamespace(st_size=len(malformed))):
                    with self.assertRaises(guard.InvalidAdapterCheckpoint):
                        lib.attach_adapter(SimulatedBase(), "SIMULATED_ADAPTER")
                peft.from_pretrained.assert_not_called()

    def test_explicit_base_evidence_does_not_claim_adapter_coverage(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), consumer(directory) as (lib, _, peft):
                with patch("importlib.metadata.version", return_value="SIMULATED_PEFT"):
                    evidence = lib.describe_base_loader(SimulatedBase())
                self.assertEqual(evidence["mode"], "bare_base")
                self.assertIsNone(evidence["adapter_keys"])
                self.assertEqual(evidence["wrapper_class"], "SimulatedBase")
                self.assertEqual(evidence["model_class"], "SimulatedBase")
                peft.from_pretrained.assert_not_called()


class MemorySource:
    def __init__(self, name, content):
        self.name, self.content = name, content
        self.suffix = Path(name).suffix

    def __lt__(self, other):
        return self.name < other.name

    def is_file(self):
        return True

    def read_bytes(self):
        return self.content

    def read_text(self, *, encoding):
        return self.content.decode(encoding)


class MemoryFolder:
    def __init__(self):
        self.outputs = {}

    def __truediv__(self, name):
        return types.SimpleNamespace(
            write_bytes=lambda value: self.outputs.__setitem__(name, value),
            write_text=lambda value, encoding: self.outputs.__setitem__(name, value.encode(encoding)),
        )


class AdapterPackagingTests(unittest.TestCase):
    def test_bootstrap_explicit_download_list_contains_guard(self):
        source = (ROOT / "tools/estate_payload/SZL-PAYLOAD-V4.ps1").read_text(encoding="utf-8")
        fetch_list = source.split("$Files =", 1)[1].split("foreach ($f in $Files)", 1)[0]
        self.assertIn("'adapter_guard.py'", fetch_list)
        self.assertIn('"kit/$_"', fetch_list)

    def test_only_ast_extracted_copy_loop_preserves_guard_without_substitution(self):
        source = (ROOT / "tools/estate_payload/szl_payload_v4.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        generator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gen_candidate")
        loops = [node for node in generator.body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "src"]
        self.assertEqual(len(loops), 1)
        isolated = ast.Module(body=loops, type_ignores=[])
        folder = MemoryFolder()
        guard_bytes = b"# __VERSION__\r\nvalue = '__CANDIDATE_ID__'\r\n"
        sources = [MemorySource("adapter_guard.py", guard_bytes),
                   MemorySource("README.md", b"__VERSION__ __CANDIDATE_ID__"),
                   MemorySource("ignored.json", b"{}")]
        namespace = {
            "KIT_DIR": types.SimpleNamespace(iterdir=lambda: iter(sources)), "folder": folder,
            "repl": {"__VERSION__": "SIMULATED_VERSION", "__CANDIDATE_ID__": "SIMULATED_ID"},
        }
        exec(compile(isolated, "SIMULATED_COPY_LOOP", "exec"), namespace)
        self.assertEqual(folder.outputs["adapter_guard.py"], normalized(guard_bytes))
        self.assertEqual(folder.outputs["README.md"], b"SIMULATED_VERSION SIMULATED_ID")
        self.assertNotIn("ignored.json", folder.outputs)


class EvaluatorAdapterEvidenceTests(unittest.TestCase):
    @contextmanager
    def harness(self, directory, *, comparator=False, bare=False, failure=None):
        with consumer(directory, evaluator=True) as (lib, ev, peft):
            guard = lib._load_adapter_guard()
            base = SimulatedBase()
            candidate_model = SimulatedAdapter(base)
            comparator_model = SimulatedAdapter(SimulatedBase())
            if failure == "candidate":
                candidate_model.status.enabled = False
            if failure == "comparator":
                comparator_model.status.enabled = False
            peft.from_pretrained.side_effect = [candidate_model, comparator_model]
            candidate = {
                "candidate_id": "SIMULATED_ONLY", "actual_training_base": {"id": "SIMULATED_BASE"},
                "training_data": {"rights": "UNDECLARED"},
                "evaluation_protocol": {"required_strict_case_improvement_over_predecessor": 1,
                                        "claim_scope": "SIMULATED_TEST_ONLY"},
            }
            gates = {"reproducibility": {"rows": 1}, "schema_validity_min": 0.99,
                     "evidence_handle_validity_min": 0.99, "ece_max": 0.05,
                     "true_abstain_min": 1.0, "false_abstain_max": 0.0,
                     "red_team_refusal_min": 1.0, "quant_parity_min": 0.99}
            original_gates = copy.deepcopy(gates)
            reports = {}
            scored_models = []

            def score(model, *_args):
                scored_models.append(model)
                return [{"output": "SIMULATED"}], {
                    "n": 1, "schema_ok": 1, "exact": 1, "abstain_expected": 0,
                    "abstain_hit": 0, "answerable": 1, "false_abstain": 0,
                    "handles_rows": 0, "handles_valid": 0, "conf": [], "conf_correct": [],
                }

            argv = ["SIMULATED_EVALUATOR", "--adapter", "none" if bare else "SIMULATED_ADAPTER"]
            if comparator:
                argv.extend(["--comparator", "SIMULATED_COMPARATOR"])
            with patch.object(lib, "_load_adapter_guard", return_value=guard), \
                    patch.object(guard, "checkpoint_keys", return_value=list(KEYS)), \
                    patch("importlib.metadata.version", return_value="SIMULATED_PEFT"), \
                    patch.object(lib, "load_candidate", return_value=candidate), \
                    patch.object(lib, "load_gates", return_value=gates), \
                    patch.object(lib, "require_report", return_value={"verdict": "PASS"}), \
                    patch.object(lib, "load_splits", return_value=({"dev": [{"SIMULATED": True}], "test": [], "adversarial": []}, None, None)), \
                    patch.object(lib, "load_base_model", return_value=(base, object(), "SIMULATED")), \
                    patch.object(lib, "sha256_dir", return_value="SIMULATED_HASH"), \
                    patch.object(lib, "installed_versions", return_value={"SIMULATED": True}), \
                    patch.object(lib, "host_info", return_value={"SIMULATED": True}), \
                    patch.object(lib, "write_json", side_effect=lambda path, value: reports.__setitem__(path.name, value)), \
                    patch.object(ev, "score_rows", side_effect=score), \
                    patch.object(Path, "is_dir", return_value=True), \
                    patch.object(Path, "is_file", return_value=False), \
                    patch.object(sys, "argv", argv), \
                    patch("builtins.open", mock_open()), patch("builtins.print"):
                yield types.SimpleNamespace(
                    run=ev.main, guard=guard, reports=reports, scored=scored_models,
                    base=base, candidate=candidate_model, comparator=comparator_model,
                    gates=gates, original_gates=original_gates, peft=peft,
                )

    def test_candidate_rejection_precedes_candidate_scoring_and_final_reports(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), self.harness(directory, failure="candidate") as h:
                with self.assertRaises(h.guard.AdapterNotApplied):
                    h.run()
                self.assertEqual(h.scored, [])
                self.assertEqual(h.reports, {})
                self.assertEqual(h.gates, h.original_gates)

    def test_comparator_rejection_precedes_its_scoring_and_final_reports(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), self.harness(directory, comparator=True, failure="comparator") as h:
                with self.assertRaises(h.guard.AdapterNotApplied):
                    h.run()
                self.assertTrue(h.scored)
                self.assertTrue(all(model is h.candidate for model in h.scored))
                self.assertNotIn(h.comparator, h.scored)
                self.assertEqual(h.reports, {})
                self.assertEqual(h.gates, h.original_gates)

    def test_success_reports_both_structural_loaders_and_preserves_thresholds(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), self.harness(directory, comparator=True) as h:
                self.assertEqual(h.run(), 0)
                report = h.reports["evaluation_report.json"]
                for evidence in (report["loader"], report["comparator"]["loader"]):
                    self.assertEqual(evidence["wrapper_class"], "SimulatedAdapter")
                    self.assertEqual(evidence["model_class"], "SimulatedBase")
                    self.assertEqual(evidence["mode"], "adapter_structural_coverage")
                    self.assertEqual(evidence["adapter_keys"]["numerical_application"], "UNKNOWN")
                self.assertIn(h.comparator, h.scored)
                self.assertEqual(h.gates, h.original_gates)
                by_id = {gate["gate"]: gate for gate in h.reports["gate_results.json"]["gates"]}
                self.assertEqual(by_id["G03_schema_validity"]["threshold"], "0.99")
                self.assertEqual(by_id["G09_regression_vs_parent"]["threshold"], 1)
                self.assertEqual(by_id["G09_regression_vs_parent"]["status"], "FAIL")
                self.assertEqual(report["gate_verdict"], "BLOCKED")
                self.assertEqual(report["promotion"], "NOT_PROMOTABLE")

    def test_explicit_bare_base_mode_never_attaches_an_adapter(self):
        for directory in CONSUMERS:
            with self.subTest(consumer=directory.name), self.harness(directory, bare=True) as h:
                self.assertEqual(h.run(), 0)
                report = h.reports["evaluation_report.json"]
                self.assertEqual(report["loader"]["mode"], "bare_base")
                self.assertIsNone(report["loader"]["adapter_keys"])
                self.assertIsNone(report["adapterSha256"])
                self.assertIsNone(report["comparator"])
                h.peft.from_pretrained.assert_not_called()
                self.assertTrue(all(model is h.base for model in h.scored))
                self.assertEqual(h.gates, h.original_gates)

    def test_module_loading_restores_sys_path_and_preexisting_module_entry(self):
        sentinel = types.ModuleType("candidate_lib")
        before_path = sys.path[:]
        with patch.dict(sys.modules, {"candidate_lib": sentinel}):
            for directory in CONSUMERS:
                with consumer(directory, evaluator=True):
                    self.assertIsNot(sys.modules["candidate_lib"], sentinel)
                self.assertIs(sys.modules["candidate_lib"], sentinel)
                self.assertEqual(sys.path, before_path)


if __name__ == "__main__":
    unittest.main()
