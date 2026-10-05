# SPDX-License-Identifier: Apache-2.0
"""SIMULATED admission software tests; mock reports are never doctrine evidence.

These tests construct their own protocol fixtures. They do not open the real
training/dev/held-out curriculum, read prior evaluation cases, invoke a model,
sign anything or claim that a test double is the actual Nemo doctrine kernel.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import json_contract as contract
from test_json_contract import reference_response, request


def fixture_row(family, kind, index, *, status=None, authority=None):
    """Independent synthetic unit-test row, never a curriculum authoring input."""
    statuses = tuple(contract.RECOVERY_RULES)
    authorities = tuple(sorted(admission.BLOCKED_AUTHORITIES))
    req = request(authority=(authority or authorities[index % 5]) if kind == "REFUSAL"
                  else "PROPOSE_ONLY", status=(status or statuses[index % 5])
                  if kind == "RECOVERY" else "OK")
    req["task"] = f"SIMULATED admission unit fixture for {family}, {kind}, sample-{index}."
    content = {key: value for key, value in req.items() if key != "requestId"}
    req["requestId"] = contract.namespaced_request_id("train", kind, family, content)
    return {"id": req["requestId"], "split": "train", "family": family, "kind": kind,
            "messages": [{"role": "system", "content": contract.SYSTEM_PROMPT},
                         {"role": "user", "content": contract.prompt_envelope(contract.canonical_json(req))},
                         {"role": "assistant", "content": contract.canonical_json(reference_response(req))}]}


def serialize(rows):
    return b"".join((contract.canonical_json(row) + "\n").encode("utf-8") for row in rows)


def fixture_manifest(raw):
    """Commitments only; imaginary dev/test bytes are never created or opened."""
    return {"schema": admission.SCHEMA, "candidate_id": contract.PROFILE,
            "seed": admission.SEED, "contract_source_revision": admission.CONTRACT_REVISION,
            "contract_source_sha256": admission.CONTRACT_SOURCE_SHA256,
            "train_families": sorted(admission.TRAIN_FAMILIES),
            "dev_families": sorted(admission.DEV_FAMILIES),
            "heldout_families": sorted(admission.HELDOUT_FAMILIES),
            "splits": {
                "train": {"path": "train.jsonl", "sha256": hashlib.sha256(raw).hexdigest(),
                          "rows": 360, "kind_counts": {kind: 120 for kind in contract.CLAIMS}},
                "dev": {"path": "dev.jsonl", "sha256": "b" * 64, "rows": 90,
                        "kind_counts": {kind: 30 for kind in contract.CLAIMS}},
                "test": {"path": "heldout/test.jsonl", "sha256": "c" * 64, "rows": 180,
                         "kind_counts": {kind: 60 for kind in contract.CLAIMS}}},
            "training_eligible": False, "publication_eligible": False,
            "execution_authority": False, "evidence_class": "SIMULATED",
            "signature": "UNAVAILABLE", "runtime_binding": "UNAVAILABLE"}


def simulated_gate_modules(raw, generic_overrides=None, nemo_overrides=None, mutate=None):
    """Test doubles ONLY. Parent runs real pinned doctrine separately."""
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    generic_report = {"status": "VALID", "records": 360, "min_examples_required": 360,
                      "dataset_sha256": digest, "errors": []}
    doctrine_report = {"status": "VALID", "persona": "finetuned", "records": 360,
                       "checked": 360, "skipped_no_chat_pair": 0, "dataset_sha256": digest,
                       "violation_counts": {}, "violations": [],
                       "receipt_chain": [],
                       "receipt_chain_tip": "SIMULATED-test-double-not-a-receipt"}
    generic_report.update(generic_overrides or {})
    doctrine_report.update(nemo_overrides or {})

    def generic(path, min_examples):
        if min_examples != 360:
            raise AssertionError("wrong generic coverage argument")
        if mutate:
            mutate(Path(path))
        return generic_report

    def doctrine(path, persona):
        if persona != "finetuned":
            raise AssertionError("wrong doctrine persona")
        return doctrine_report

    return {"validate_sft_dataset.py": SimpleNamespace(validate_dataset=generic),
            "nemo_doctrine_gate.py": SimpleNamespace(_IMPORT_ERROR=None, gate_dataset=doctrine)}


def simulated_kernel_context(*args, **kwargs):
    """Only for report/snapshot unit tests, never source-provenance evidence."""
    return contextlib.nullcontext(SimpleNamespace(provenance={"evidence_class": "SIMULATED"}))


class ReadOnceMetadataTests(unittest.TestCase):
    """SIMULATED metadata races; no curriculum or kernel is loaded."""

    def read_with_metadata(self, *, opened_changes=None, after_changes=None,
                           current_changes=None):
        path = Path(os.path.abspath("synthetic-reader.jsonl"))
        fields = dict(st_mode=0o100600, st_file_attributes=0, st_dev=1,
                      st_ino=2, st_size=7, st_mtime_ns=500, st_ctime_ns=100,
                      st_nlink=1)
        before = SimpleNamespace(**fields)
        # Windows can expose creation time through lstat and change time through
        # fstat. Stable values from these different clocks must not be equated.
        descriptor_fields = {**fields, "st_ctime_ns": 200}
        opened = SimpleNamespace(**{**descriptor_fields, **(opened_changes or {})})
        after = SimpleNamespace(**{**descriptor_fields, **(after_changes or {})})
        current = SimpleNamespace(**{**fields, **(current_changes or {})})
        target_reads = 0

        def metadata(candidate):
            nonlocal target_reads
            if candidate == path:
                target_reads += 1
                return current if target_reads == 3 else before
            return SimpleNamespace(st_mode=0o040700, st_file_attributes=0)

        with patch.object(Path, "lstat", metadata), \
             patch.object(admission.os, "open", return_value=42), \
             patch.object(admission.os, "fstat", side_effect=[opened, after]), \
             patch.object(admission.os, "fdopen", return_value=io.BytesIO(b"bounded")), \
             patch.object(admission.os, "close") as close:
            try:
                return admission._read_once(path, 7)
            finally:
                close.assert_called_once_with(42)

    def test_stable_distinct_path_and_descriptor_ctime_are_accepted(self):
        self.assertEqual(self.read_with_metadata(), b"bounded")

    def test_descriptor_ctime_change_is_still_rejected(self):
        with self.assertRaisesRegex(admission.AdmissionError, "changed during read"):
            self.read_with_metadata(after_changes={"st_ctime_ns": 201})

    def test_path_ctime_change_is_still_rejected(self):
        # Match the descriptor clock deliberately: cross-API equality must not
        # conceal a path timestamp mutation since the original lstat call.
        with self.assertRaisesRegex(admission.AdmissionError, "changed after read"):
            self.read_with_metadata(current_changes={"st_ctime_ns": 200})

    def test_path_replacement_or_metadata_change_is_still_rejected(self):
        for field, value in (("st_dev", 9), ("st_ino", 9), ("st_size", 8),
                             ("st_mtime_ns", 501), ("st_nlink", 2)):
            with self.subTest(field=field), self.assertRaisesRegex(
                    admission.AdmissionError, "changed after read"):
                self.read_with_metadata(current_changes={field: value})

    def test_descriptor_size_or_mtime_change_is_still_rejected(self):
        for field, value in (("st_size", 8), ("st_mtime_ns", 501)):
            with self.subTest(field=field), self.assertRaisesRegex(
                    admission.AdmissionError, "changed during read"):
                self.read_with_metadata(after_changes={field: value})

    def test_replacement_before_open_is_still_rejected(self):
        for field in ("st_dev", "st_ino"):
            with self.subTest(field=field), self.assertRaisesRegex(
                    admission.AdmissionError, "changed before read"):
                self.read_with_metadata(opened_changes={field: 9})


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [fixture_row(family, kind, index)
                    for family in sorted(admission.TRAIN_FAMILIES)
                    for kind in contract.CLAIMS for index in range(30)]
        cls.raw = serialize(cls.rows)
        cls.manifest = fixture_manifest(cls.raw)

    def reject_rows(self, rows):
        raw = serialize(rows)
        with self.assertRaises((admission.AdmissionError, contract.ContractError)):
            admission.validate_train_bytes(raw, fixture_manifest(raw))

    def test_fixture_exact_balanced_protocol_conformance(self):
        admission.validate_manifest(self.manifest)
        report = admission.validate_train_bytes(self.raw, self.manifest)
        self.assertEqual(report["rows"], 360)
        self.assertEqual(report["family_counts"], {family: 90 for family in admission.TRAIN_FAMILIES})
        self.assertEqual(report["kind_counts"], {kind: 120 for kind in contract.CLAIMS})
        self.assertIs(report["heldout_content_opened"], False)

    def test_manifest_extra_missing_or_wrong_authority_fields_rejected(self):
        for key, replacement in (("unexpected", True), ("training_eligible", 0),
                                  ("publication_eligible", True), ("execution_authority", 0),
                                  ("seed", True), ("signature", "SIGNED"),
                                  ("runtime_binding", "VALID"), ("evidence_class", "MEASURED")):
            value = copy.deepcopy(self.manifest)
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(admission.AdmissionError):
                admission.validate_manifest(value)
        value = copy.deepcopy(self.manifest)
        del value["seed"]
        with self.assertRaises(admission.AdmissionError):
            admission.validate_manifest(value)

    def test_manifest_rejects_source_drift_even_with_matching_claimed_hash(self):
        with patch.object(admission, "_read_once", return_value=b"modified contract"):
            value = copy.deepcopy(self.manifest)
            value["contract_source_sha256"] = hashlib.sha256(b"modified contract").hexdigest()
            with self.assertRaises(admission.AdmissionError):
                admission.validate_manifest(value)

    def test_manifest_revision_and_digest_are_exact(self):
        for key, replacement in (("contract_source_revision", "0" * 40),
                                  ("contract_source_sha256", "A" * 64),
                                  ("contract_source_sha256", admission.CONTRACT_SOURCE_SHA256 + "\n")):
            value = copy.deepcopy(self.manifest)
            value[key] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(admission.AdmissionError):
                admission.validate_manifest(value)

    def test_manifest_exact_split_types_paths_and_hashes(self):
        for field, replacement in (("path", "../train.jsonl"), ("rows", True),
                                    ("rows", 359), ("sha256", "bad"),
                                    ("kind_counts", {"DRAFT": 120, "RECOVERY": 120, "REFUSAL": 119})):
            value = copy.deepcopy(self.manifest)
            value["splits"]["train"][field] = replacement
            with self.subTest(field=field), self.assertRaises(admission.AdmissionError):
                admission.validate_manifest(value)
        value = copy.deepcopy(self.manifest)
        value["splits"]["dev"]["sha256"] = value["splits"]["test"]["sha256"]
        with self.assertRaises(admission.AdmissionError):
            admission.validate_manifest(value)

    def test_manifest_family_commitments_exact_and_nonoverlapping(self):
        for key, replacement in (("train_families", ["wrong-family"] * 4),
                                  ("dev_families", ["different-dev-family", "another-dev-family"]),
                                  ("heldout_families", ["wrong-family"] * 3)):
            value = copy.deepcopy(self.manifest)
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(admission.AdmissionError):
                admission.validate_manifest(value)

    def test_training_hash_length_newline_and_crlf_rejected(self):
        for raw in (b"", self.raw[:-1], self.raw.replace(b"\n", b"\r\n"),
                    self.raw + b"\n", self.raw + b" " * admission.MAX_FILE_BYTES):
            with self.subTest(length=len(raw)), self.assertRaises(admission.AdmissionError):
                admission.validate_train_bytes(raw, fixture_manifest(raw))
        with self.assertRaises(admission.AdmissionError):
            admission.validate_train_bytes(self.raw, fixture_manifest(self.raw + b"x"))

    def test_duplicate_identity_and_incomplete_row_count_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1] = copy.deepcopy(rows[0])
        self.reject_rows(rows)
        self.reject_rows(self.rows[:-1])

    def test_row_metadata_roles_system_envelope_and_target_strict(self):
        for mutate in (lambda row: row.update(extra="unknown"),
                       lambda row: row.update(split="dev"),
                       lambda row: row.update(family="package-license-ledger"),
                       lambda row: row.update(id="train-json-v4-wrong-id-fixture"),
                       lambda row: row["messages"][0].update(content="other prompt"),
                       lambda row: row["messages"][1].update(role="tool"),
                       lambda row: row["messages"][1].update(extra=False),
                       lambda row: row["messages"][1].update(content='{"request":{},"requestSha256":"wrong"}'),
                       lambda row: row["messages"][2].update(content="REFUSE: plaintext"),
                       lambda row: row["messages"][2].update(content=row["messages"][2]["content"] + " ")):
            rows = copy.deepcopy(self.rows)
            mutate(rows[0])
            self.reject_rows(rows)

    def test_training_duplicate_json_nonfinite_invalid_utf8_fail_closed(self):
        first, rest = self.raw.split(b"\n", 1)
        for modified in (first.replace(b'{', b'{"id":"duplicate",', 1),
                         b'{"x":NaN}', b'{"x":"\xff"}', b'{"x":"' + b"a" * contract.MAX_BYTES + b'"}'):
            raw = modified + b"\n" + rest
            with self.assertRaises(admission.AdmissionError):
                admission.validate_train_bytes(raw, fixture_manifest(raw))

    def test_training_measured_evidence_label_rejected(self):
        rows = copy.deepcopy(self.rows)
        row = rows[0]
        req = contract.strict_object(row["messages"][1]["content"])["request"]
        req["evidence"][0]["label"] = "MEASURED"
        content = {key: value for key, value in req.items() if key != "requestId"}
        req["requestId"] = contract.namespaced_request_id("train", row["kind"], row["family"], content)
        row["id"] = req["requestId"]
        row["messages"][1]["content"] = contract.prompt_envelope(contract.canonical_json(req))
        row["messages"][2]["content"] = contract.canonical_json(reference_response(req))
        self.reject_rows(rows)

    def test_exact_global_counts_do_not_hide_family_imbalance(self):
        rows = copy.deepcopy(self.rows)
        source, destination = sorted(admission.TRAIN_FAMILIES)[:2]
        index = next(i for i, row in enumerate(rows) if row["family"] == source and row["kind"] == "DRAFT")
        rows[index] = fixture_row(destination, "DRAFT", 700)
        self.reject_rows(rows)

    def test_family_totals_do_not_hide_per_class_imbalance(self):
        rows = copy.deepcopy(self.rows)
        first, second = sorted(admission.TRAIN_FAMILIES)[:2]
        i = next(i for i, row in enumerate(rows) if row["family"] == first and row["kind"] == "DRAFT")
        j = next(i for i, row in enumerate(rows) if row["family"] == second and row["kind"] == "REFUSAL")
        rows[i] = fixture_row(first, "REFUSAL", 700)
        rows[j] = fixture_row(second, "DRAFT", 700)
        self.reject_rows(rows)

    def test_per_class_totals_do_not_hide_recovery_status_imbalance(self):
        rows = copy.deepcopy(self.rows)
        index = next(i for i, row in enumerate(rows) if row["kind"] == "RECOVERY")
        row = rows[index]
        rows[index] = fixture_row(row["family"], "RECOVERY", 700, status="STALE")
        self.reject_rows(rows)

    def test_per_class_totals_do_not_hide_refused_authority_imbalance(self):
        rows = copy.deepcopy(self.rows)
        index = next(i for i, row in enumerate(rows) if row["kind"] == "REFUSAL")
        row = rows[index]
        rows[index] = fixture_row(row["family"], "REFUSAL", 700, authority="RESEND_QUARANTINED")
        self.reject_rows(rows)

    def test_successful_mock_conformance_keeps_authority_false_and_reads_train_only(self):
        opened = []
        original = admission._read_once

        def record(path, maximum):
            opened.append(Path(path))
            self.assertNotIn("heldout", Path(path).parts)
            self.assertNotEqual(Path(path).name, "dev.jsonl")
            return original(path, maximum)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train = directory / "train.jsonl"
            manifest = directory / "curriculum-manifest.json"
            train.write_bytes(self.raw)
            manifest.write_bytes(contract.canonical_json(self.manifest).encode("utf-8"))
            modules = simulated_gate_modules(self.raw)
            with patch.object(admission.nemo_binding, "load_kernel", side_effect=simulated_kernel_context), \
                 patch.object(admission, "_load_gate", side_effect=lambda name, kernel=None: modules[name]), \
                 patch.object(admission, "_verify_doctrine_receipts"), \
                 patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(admission, "_read_once", side_effect=record):
                report, code = admission.check_conformance(train, manifest)
        self.assertEqual(code, 0)
        self.assertEqual(report["state"], "BLOCKED")
        for key in ("training_eligible", "publication_eligible", "execution_authority", "signature_valid",
                    "heldout_content_opened", "gate_reports_authenticated"):
            self.assertIs(report[key], False)
        self.assertTrue(report["contract_conforms"])
        self.assertTrue(report["nemo_valid"])
        self.assertEqual(report["nemo_source_binding"], {"evidence_class": "SIMULATED"})
        self.assertTrue(opened)

    def test_missing_kernel_is_not_ready_without_green_conformance_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train, manifest = directory / "train.jsonl", directory / "curriculum-manifest.json"
            train.write_bytes(self.raw)
            manifest.write_bytes(contract.canonical_json(self.manifest).encode("utf-8"))
            with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                 patch.object(admission, "_load_gate", side_effect=admission.GateNotReady("missing")):
                report, code = admission.check_conformance(train, manifest)
        self.assertEqual(code, 2)
        self.assertIs(report["contract_conforms"], True)
        self.assertIs(report["nemo_valid"], False)
        self.assertIs(report["training_eligible"], False)

    def test_invalid_input_fails_before_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train, manifest = directory / "train.jsonl", directory / "curriculum-manifest.json"
            train.write_bytes(self.raw[:-1])
            # Read only the public aggregate manifest, not curriculum rows.
            # Keep the real source commitment so the malformed train, rather
            # than an unrelated synthetic manifest hash, is what fails here.
            manifest.write_bytes((admission.HERE / "curriculum-manifest.json").read_bytes())
            with patch.object(admission, "run_gates") as gates:
                report, code = admission.check_conformance(train, manifest)
            gates.assert_not_called()
        self.assertEqual(code, 1)
        self.assertIs(report["contract_conforms"], False)

    def test_frozen_manifest_hash_substitution_fails_before_parse_train_or_gates(self):
        # Only this synthetic unit fixture patches the source hash constant.
        # Runtime CLI has no override, and CI requires the real frozen bytes.
        committed = contract.canonical_json(self.manifest).encode("utf-8")
        substituted = copy.deepcopy(self.manifest)
        substituted["splits"]["test"]["sha256"] = "d" * 64
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train, manifest = directory / "train.jsonl", directory / "curriculum-manifest.json"
            manifest.write_bytes(contract.canonical_json(substituted).encode("utf-8"))
            with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(committed).hexdigest()), \
                 patch.object(admission, "validate_manifest") as parse_manifest, \
                 patch.object(admission, "run_gates") as gates:
                report, code = admission.check_conformance(train, manifest)
            parse_manifest.assert_not_called()
            gates.assert_not_called()
        self.assertEqual(code, 1)
        self.assertIs(report["contract_conforms"], False)
        self.assertIs(report["training_eligible"], False)

    def test_unregistered_synthetic_manifest_cannot_pass_runtime_fixed_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train, manifest = directory / "train.jsonl", directory / "curriculum-manifest.json"
            train.write_bytes(self.raw)
            manifest.write_bytes(contract.canonical_json(self.manifest).encode("utf-8"))
            with patch.object(admission, "run_gates") as gates:
                report, code = admission.check_conformance(train, manifest)
            gates.assert_not_called()
        self.assertEqual(code, 1)
        self.assertIs(report["training_eligible"], False)

    def test_wrong_input_names_or_parent_directory_fail_before_reads(self):
        with patch.object(admission, "_read_once") as reads:
            for train, manifest in ((Path("dev.jsonl"), Path("curriculum-manifest.json")),
                                     (Path("one/train.jsonl"), Path("two/curriculum-manifest.json"))):
                self.assertEqual(admission.check_conformance(train, manifest)[1], 1)
            reads.assert_not_called()

    def test_gate_report_mismatch_violation_skips_wrong_persona_or_bool_count_rejected(self):
        generic_changes = ({"status": "INVALID"}, {"records": True},
                           {"records": 359}, {"min_examples_required": True},
                           {"dataset_sha256": "sha256:" + "0" * 64}, {"errors": ["bad"]})
        doctrine_changes = ({"status": "VIOLATIONS"}, {"checked": 359},
                            {"skipped_no_chat_pair": False}, {"skipped_no_chat_pair": 1},
                            {"persona": "wrapper"}, {"violation_counts": {"R1": 1}},
                            {"violations": [{}]}, {"dataset_sha256": "sha256:" + "0" * 64})
        for changes in generic_changes:
            modules = simulated_gate_modules(self.raw, generic_overrides=changes)
            with self.subTest(generic=changes), \
                 patch.object(admission.nemo_binding, "load_kernel", side_effect=simulated_kernel_context), \
                 patch.object(admission, "_load_gate", side_effect=lambda name, kernel=None: modules[name]):
                with self.assertRaises(admission.AdmissionError):
                    admission.run_gates(self.raw)
        for changes in doctrine_changes:
            modules = simulated_gate_modules(self.raw, nemo_overrides=changes)
            with self.subTest(doctrine=changes), \
                 patch.object(admission.nemo_binding, "load_kernel", side_effect=simulated_kernel_context), \
                 patch.object(admission, "_load_gate", side_effect=lambda name, kernel=None: modules[name]):
                with self.assertRaises(admission.AdmissionError):
                    admission.run_gates(self.raw)

    def test_snapshot_mutation_stops_before_doctrine(self):
        modules = simulated_gate_modules(self.raw, mutate=lambda path: path.write_bytes(b"changed"))
        doctrine = modules["nemo_doctrine_gate.py"].gate_dataset
        with patch.object(modules["nemo_doctrine_gate.py"], "gate_dataset", wraps=doctrine) as gate, \
             patch.object(admission.nemo_binding, "load_kernel", side_effect=simulated_kernel_context), \
             patch.object(admission, "_load_gate", side_effect=lambda name, kernel=None: modules[name]):
            with self.assertRaises(admission.AdmissionError):
                admission.run_gates(self.raw)
        gate.assert_not_called()

    def test_missing_nemo_dependency_not_ready(self):
        modules = simulated_gate_modules(self.raw)
        modules["nemo_doctrine_gate.py"]._IMPORT_ERROR = ImportError("SIMULATED missing kernel")
        with patch.object(admission.nemo_binding, "load_kernel", side_effect=simulated_kernel_context), \
             patch.object(admission, "_load_gate", side_effect=lambda name, kernel=None: modules[name]):
            with self.assertRaises(admission.GateNotReady):
                admission.run_gates(self.raw)

    def test_real_generic_gate_source_hash_is_the_pinned_baseline(self):
        generic = admission._load_gate("validate_sft_dataset.py")
        self.assertTrue(callable(generic.validate_dataset))

    def test_receipt_helper_checks_chain_tip_and_exact_pair_binding_with_test_doubles(self):
        # These modules and receipts are SIMULATED software inputs only. The
        # real maintained verifier and pinned kernel are exercised by the root.
        engine, receipt_module = ModuleType("szl_nemo.engine"), ModuleType("szl_nemo.receipt")
        engine.input_hash = lambda prompt, answer: hashlib.sha256((prompt + answer).encode()).hexdigest()
        receipt_module.verify_chain = lambda receipts: True
        receipts = []
        for index, row in enumerate(self.rows):
            messages = row["messages"]
            receipts.append({"schema": "szl.nemo.receipt.v1", "sequence": index,
                             "prev_receipt_sha256": "SIMULATED", "receipt_sha256": "SIMULATED-tip",
                             "receipt_status": "UNSIGNED_HONEST", "decision": {
                                 "schema_version": "szl.nemo.decision.v1", "decision": "ALLOW",
                                 "violated_rules": [], "reasons": [],
                                 "rule_version": "doctrine-v11/R1-R5 persona/finetuned",
                                 "input_hash": engine.input_hash(messages[1]["content"], messages[2]["content"]),
                                 "receipt_status": "UNSIGNED_HONEST"}})
        report = {"receipt_schema": "szl.nemo.receipt.v1", "receipt_chain": receipts,
                  "receipt_chain_tip": "SIMULATED-tip",
                  "kernel": {"repo": "https://github.com/szl-holdings/szl-nemo",
                             "rule_version": "doctrine-v11/R1-R5"}}
        kernel = SimpleNamespace(input_hash=engine.input_hash,
                                 verify_chain=lambda receipts: receipt_module.verify_chain(receipts))
        with patch.dict(sys.modules, {"szl_nemo.engine": None, "szl_nemo.receipt": None}):
            admission._verify_doctrine_receipts(report, self.raw, kernel)
            for mutate in (lambda value: value.update(receipt_chain_tip="wrong"),
                           lambda value: value.update(receipt_chain=value["receipt_chain"][:-1]),
                           lambda value: value["receipt_chain"][0]["decision"].update(input_hash="wrong"),
                           lambda value: value["receipt_chain"][0]["decision"].update(decision="BLOCK"),
                           lambda value: value["receipt_chain"][0].update(receipt_status="SIGNED")):
                value = copy.deepcopy(report)
                mutate(value)
                with self.assertRaises(admission.AdmissionError):
                    admission._verify_doctrine_receipts(value, self.raw, kernel)
            receipt_module.verify_chain = lambda receipts: False
            with self.assertRaises(admission.AdmissionError):
                admission._verify_doctrine_receipts(report, self.raw, kernel)

    def test_source_root_is_required_before_loading_gates(self):
        with patch.object(admission, "_load_gate") as load:
            with self.assertRaises(admission.GateNotReady):
                admission.run_gates(self.raw)
        load.assert_not_called()

    def test_source_drift_is_not_ready_or_valid_conformance(self):
        with patch.object(admission.nemo_binding, "load_kernel",
                          side_effect=admission.nemo_binding.BindingError("SIMULATED drift")), \
             patch.object(admission, "_load_gate") as load:
            with self.assertRaises(admission.AdmissionError) as caught:
                admission.run_gates(self.raw, Path("SIMULATED-source"))
        self.assertNotIsInstance(caught.exception, admission.GateNotReady)
        load.assert_not_called()

    def test_nemo_wrapper_cannot_use_ambient_kernel_without_binding(self):
        with self.assertRaises(admission.GateNotReady):
            admission._load_gate("nemo_doctrine_gate.py")

    def test_read_once_reparse_point_is_rejected_before_file_open(self):
        metadata = SimpleNamespace(st_mode=0o100600, st_file_attributes=0x400)
        with patch.object(Path, "lstat", return_value=metadata), patch.object(admission.os, "open") as opened:
            with self.assertRaises(admission.AdmissionError):
                admission._read_once(Path("synthetic-reparse.jsonl"), 100)
            opened.assert_not_called()

    def test_fixed_gate_unknown_missing_or_hash_drift_rejected_before_compile(self):
        with self.assertRaises(admission.AdmissionError):
            admission._load_gate("../arbitrary.py")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(admission, "TOOLS", directory):
                with self.assertRaises(admission.GateNotReady):
                    admission._load_gate("validate_sft_dataset.py")
                (directory / "validate_sft_dataset.py").write_bytes(b"raise RuntimeError('must not execute')\n")
                with self.assertRaises(admission.AdmissionError):
                    admission._load_gate("validate_sft_dataset.py")

    def test_read_once_limits_hardlinks_symlinks_and_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / "original.jsonl"
            original.write_bytes(b"bounded")
            self.assertEqual(admission._read_once(original, 7), b"bounded")
            with self.assertRaises(admission.AdmissionError):
                admission._read_once(original, 6)
            with self.assertRaises(admission.AdmissionError):
                admission._read_once(directory, 100)
            alias = directory / "hardlink.jsonl"
            try:
                os.link(original, alias)
            except OSError as exc:
                self.skipTest(f"hardlinks unavailable: {exc}")
            with self.assertRaises(admission.AdmissionError):
                admission._read_once(original, 100)
            alias.unlink()
            link = directory / "symlink.jsonl"
            try:
                link.symlink_to(original)
            except OSError:
                # Windows without symlink privilege still runs the hardlink check.
                return
            with self.assertRaises(admission.AdmissionError):
                admission._read_once(link, 100)

    def test_read_once_detects_change_during_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.jsonl"
            path.write_bytes(b"input")
            original = os.fstat
            calls = []

            def changed(descriptor):
                result = original(descriptor)
                calls.append(descriptor)
                if len(calls) == 2:
                    return SimpleNamespace(st_size=result.st_size + 1,
                                           st_mtime_ns=result.st_mtime_ns,
                                           st_ctime_ns=result.st_ctime_ns)
                return result

            with patch.object(admission.os, "fstat", side_effect=changed):
                with self.assertRaises(admission.AdmissionError):
                    admission._read_once(path, 100)

    def test_strict_json_manifest_utf8_duplicate_members_and_depth_rejected(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"x":NaN}', b'[]', b'{"x":"\xff"}',
                    b'{"x":' + b'[' * 40 + b'0' + b']' * 40 + b'}'):
            with self.subTest(raw=raw[:32]), self.assertRaises(admission.AdmissionError):
                admission._strict(raw)

    def test_cli_invalid_explicit_input_is_fail_closed_without_model_imports(self):
        proc = subprocess.run([sys.executable, "-I", "-B", str(admission.HERE / "curriculum_admission.py"),
                               "--check-conformance", "--train", "dev.jsonl",
                               "--manifest", "curriculum-manifest.json"],
                              capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        report = contract.strict_object(proc.stdout)
        self.assertEqual(report["state"], "BLOCKED")
        self.assertIs(report["execution_authority"], False)

    def test_main_prints_machine_report_and_preserves_exit_code(self):
        with patch.object(admission, "check_conformance", return_value=({"state": "BLOCKED"}, 2)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = admission.main(["--check-conformance", "--train", "train.jsonl",
                                       "--manifest", "curriculum-manifest.json"])
        self.assertEqual(code, 2)
        self.assertEqual(contract.strict_object(output.getvalue()), {"state": "BLOCKED"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
