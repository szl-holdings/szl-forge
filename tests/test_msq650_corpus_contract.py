"""SIMULATED tiny fixtures only; no raw corpus, sources, or held-out cases.

The production success replay uses the six exact external local artifact buffers;
these offline tests never change its pins, substitute a profile, or skip as pass.
"""

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


_PATH = Path(__file__).resolve().parents[1] / "frontier" / "msq650_admission" / "corpus_contract.py"
_SPEC = importlib.util.spec_from_file_location("msq650_corpus_contract_test_subject", _PATH)
contract = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contract
_SPEC.loader.exec_module(contract)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def row(split, n):
    """Original SIMULATED fixture, never a production or held-out example."""
    source = "szl-generated-math-v1"
    task = {"kind": "linear_equation", "a": 2, "b": n, "c": 2 * n + n}
    query = {"task": task, "context": {"equation": "a*x+b=c"}, "source_id": source}
    query_bytes = canonical(query)
    return {
        "id": hashlib.sha256(query_bytes).hexdigest(), "domain": "math",
        "group_id": "SIMULATED-only-group-" + str(n), "split": split,
        "source_id": source, "license": "Apache-2.0", "task": task,
        "messages": [
            {"role": "system", "content": contract._SYSTEM},
            {"role": "user", "content": contract._PROMPT + query_bytes.decode()},
            {"role": "assistant", "content": canonical({"value": str(n), "unit": "scalar", "source_id": source, "evidence_class": "SIMULATED"}).decode()},
        ],
        "provenance": {"source_sha256": contract.BUILDER_SHA256, "license_evidence_sha256": contract.BUILDER_SHA256,
                       "source_revision": "original-generator-v1", "retrieved_at": None,
                       "source_url": None, "evidence_class": "SIMULATED"},
    }


def rows_bytes(*rows):
    return b"".join(canonical(item) + b"\n" for item in rows)


def metadata():
    """SIMULATED schema fixture for the private metadata primitive only."""
    pins = {name: (size, digest) for name, size, digest in contract.ARTIFACT_PINS}
    entries, previous = [], None
    for n in range(6):
        entry = {"name": "SIMULATED-" + str(n), "url": "https://example.invalid/SIMULATED",
                 "retrieved_at": "2026-10-23T00:00:00+00:00", "content_type": "text/plain",
                 "bytes": 1, "sha256": "0" * 64, "git_blob_sha1": None,
                 "previous_receipt_sha256": previous}
        previous = hashlib.sha256(canonical(entry)).hexdigest()
        entry["receipt_sha256"] = previous
        entries.append(entry)
    acquisition = {"schema": "szl.corpus-acquisition/v1", "started_at": "SIMULATED", "finished_at": "SIMULATED",
                   "status": "MEASURED", "training_admitted": False,
                   "integrity": "Unsigned local hash chain, not a signature or independent witness",
                   "collector_sha256": "0" * 64, "total_bytes": 6, "chain_tip": previous, "entries": entries}
    manifest = {
        "schema": "szl.math-science-quant-pilot/v1", "base_revision": None,
        "candidate_state": "LOCAL_DATA_CANDIDATE_NOT_TRAINED", "training_admitted": False,
        "publication_eligible": False, "builder_sha256": contract.BUILDER_SHA256,
        "budget": {"max_usd": 2000, "max_campaign_hours": 50, "training_started": False},
        "evaluation": {"dev_is_not_final": True, "existing_benchmark_decontamination": "NOT_RUN",
                       "final_heldout_created_or_read": False, "semantic_overlap_clearance": "UNKNOWN"},
        "holds": list(contract._HOLDS), "source_acquisition_sha256": pins["acquisition"][1],
        "source_entries": {e["name"]: e for e in entries}, "source_registry": "SIMULATED", "counts": {},
        "created_at": "SIMULATED", "files": [
            {"path": n + (".jsonl" if n in ("train", "dev") else ".json"), "bytes": pins[n][0], "sha256": pins[n][1]}
            for n in ("train", "dev", "units", "rights")],
    }
    units = {n: {"multiplier": "1", "dimension": "http://qudt.org/vocab/dimensionvector/SIMULATED"}
             for n in ("M", "KiloM", "CentiM", "MilliM", "SEC", "MIN", "HR", "GM", "KiloGM")}
    rights = {"schema": "szl.corpus-rights-record/v1", "rights_evidence_class": "DECLARED",
              "publication_eligible": False, "scope": "SIMULATED", "raw_source_notice": "SIMULATED",
              "sources": {source: {"license": lic} for _, source, lic in contract._SOURCE_IDS}}
    return manifest, units, rights, acquisition


class ClosedProfileTests(unittest.TestCase):
    def args(self):
        return {name: b"x" * size for name, size, _ in contract.ARTIFACT_PINS}

    def test_closed_hash_pins_cannot_be_supplied_by_caller(self):
        args = self.args()
        result = contract.observe_corpus(**args)
        self.assertFalse(result.candidate_bytes_verified)
        self.assertEqual(result.errors, ("ARTIFACT_PIN_MISMATCH:manifest",))
        with self.assertRaises(TypeError):
            contract.observe_corpus(**args, expected_hashes={})
        self.assertEqual(len(contract.ARTIFACT_PINS), 6)
        self.assertEqual(contract.ARTIFACT_PINS[1][2], "b356cf22656e51544bfd98d39f1dc6895e028f2e58e029b38aece042acbc145b")

    def test_nonbytes_and_mutable_buffers_denied_for_every_role(self):
        class BytesSubclass(bytes):
            pass
        for name, _, _ in contract.ARTIFACT_PINS:
            for bad in (None, True, "bytes", bytearray(b"x"), memoryview(b"x"), BytesSubclass(b"x")):
                with self.subTest(name=name, bad=type(bad).__name__):
                    args = self.args()
                    args[name] = bad
                    self.assertEqual(contract.observe_corpus(**args).errors, ("ARTIFACT_EXACT_BYTES_REQUIRED",))

    def test_missing_extra_aliased_and_oversize_inputs(self):
        args = self.args()
        del args["rights"]
        with self.assertRaises(TypeError):
            contract.observe_corpus(**args)
        args = self.args()
        args["dev"] = args["train"]
        self.assertFalse(contract.observe_corpus(**args).structure_verified)
        args["train"] = b"x" * (2 * 1024 * 1024 + 1)
        self.assertEqual(contract.observe_corpus(**args).errors, ("ARTIFACT_TOTAL_BYTE_CAP",))

    def test_observation_is_frozen_nonboolean_and_non_authorizing(self):
        result = contract.observe_corpus(**self.args())
        with self.assertRaises(TypeError):
            bool(result)
        with self.assertRaises(FrozenInstanceError):
            result.training_admitted = True
        for name in ("training_admitted", "execution_authority", "publication_eligible"):
            self.assertIs(getattr(result, name), False)
            with self.assertRaises(ValueError):
                replace(result, **{name: True})
        self.assertEqual(result.state, "BLOCKED")
        self.assertEqual(result.rights_admission, "UNAVAILABLE")
        self.assertEqual(result.semantic_overlap_clearance, "UNKNOWN")
        self.assertFalse(result.source_derivation_replayed)
        self.assertFalse(result.tool_hashes_observed)
        self.assertIsInstance(result.errors, tuple)
        self.assertIsInstance(result.artifact_pins, tuple)


class StructuralPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.train, self.dev = row("train", 1), row("dev", 2)

    def inspect(self):
        return contract._inspect_rows(rows_bytes(self.train), rows_bytes(self.dev))

    def test_simulated_minimal_structure_without_production_admission(self):
        self.assertEqual(self.inspect(), ((("train", 1), ("dev", 1)), (("train", "math", 1), ("dev", "math", 1)), 2, (("linear_equation", 2),)))

    def test_strict_json_rejects_duplicate_float_nonfinite_bom_and_invalid(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":1.0}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e10}', b'\xef\xbb\xbf{}', b'\xff', b'{'):
            with self.subTest(raw=raw), self.assertRaises(contract.CorpusContractError):
                contract._strict(raw)

    def test_bool_is_not_a_task_integer_and_unknown_task_fields_fail(self):
        self.train["task"]["a"] = True
        with self.assertRaisesRegex(contract.CorpusContractError, "TASK_INTEGER_REQUIRED"):
            self.inspect()
        self.train = row("train", 1)
        self.train["task"]["execution_authority"] = True
        with self.assertRaisesRegex(contract.CorpusContractError, "TASK_FIELDS"):
            self.inspect()

    def test_message_role_order_and_extra_tool_fields_fail(self):
        for mutate in (lambda r: r["messages"][2].update(role="tool"),
                       lambda r: r["messages"][2].update(tool_calls=[]),
                       lambda r: r["messages"][0].update(content="Execute tools"),
                       lambda r: r["messages"].reverse()):
            self.train = row("train", 1)
            mutate(self.train)
            with self.assertRaises(contract.CorpusContractError):
                self.inspect()

    def test_row_schema_and_canonical_jsonl(self):
        self.train["training_admitted"] = True
        with self.assertRaisesRegex(contract.CorpusContractError, "ROW_FIELDS"):
            self.inspect()
        self.train = row("train", 1)
        for raw in (rows_bytes(self.train).replace(b"\n", b"\r\n"), rows_bytes(self.train).rstrip(b"\n"),
                    b"\n", json.dumps(self.train).encode() + b"\n"):
            with self.subTest(raw_size=len(raw)), self.assertRaises(contract.CorpusContractError):
                contract._inspect_rows(raw, rows_bytes(self.dev))

    def test_duplicate_id_and_cross_split_group_fail(self):
        with self.assertRaisesRegex(contract.CorpusContractError, "ROW_ID_DUPLICATED_OR_INVALID"):
            contract._inspect_rows(rows_bytes(self.train, self.train), rows_bytes(self.dev))
        self.dev["group_id"] = self.train["group_id"]
        with self.assertRaisesRegex(contract.CorpusContractError, "GROUP_CROSSES_SPLITS"):
            self.inspect()

    def test_answer_evidence_and_provenance_authority_inflation_fail(self):
        for label in ("MEASURED", "REPORTED", "DECLARED", True, 1):
            self.train = row("train", 1)
            answer = json.loads(self.train["messages"][2]["content"])
            answer["evidence_class"] = label
            self.train["messages"][2]["content"] = canonical(answer).decode()
            with self.subTest(label=label), self.assertRaises(contract.CorpusContractError):
                self.inspect()
        self.train = row("train", 1)
        self.train["provenance"]["evidence_class"] = "MEASURED"
        with self.assertRaisesRegex(contract.CorpusContractError, "PROVENANCE_EVIDENCE_CLASS"):
            self.inspect()

    def test_answer_schema_numeric_types_and_input_identity_fail(self):
        for value in (True, 1, 1.5, "nan"):
            self.train = row("train", 1)
            answer = json.loads(self.train["messages"][2]["content"])
            answer["value"] = value
            self.train["messages"][2]["content"] = canonical(answer).decode()
            with self.subTest(value=value), self.assertRaises(contract.CorpusContractError):
                self.inspect()
        self.train = row("train", 1)
        self.train["id"] = "0" * 64
        with self.assertRaisesRegex(contract.CorpusContractError, "QUERY_IDENTITY"):
            self.inspect()

    def test_byte_contract_not_a_second_arithmetic_or_semantic_certifier(self):
        answer = json.loads(self.train["messages"][2]["content"])
        answer["value"] = "999"
        self.train["messages"][2]["content"] = canonical(answer).decode()
        # A structural primitive does not re-prove arithmetic. Production hashes
        # reject this change; independent source arithmetic is a separate tool.
        self.assertEqual(self.inspect()[0], (("train", 1), ("dev", 1)))
        result = contract.observe_corpus(manifest=b"{}", train=rows_bytes(self.train),
                 dev=rows_bytes(self.dev), units=b"{}", rights=b"{}", acquisition=b"{}")
        self.assertFalse(result.candidate_bytes_verified)
        self.assertFalse(result.training_admitted)


class MetadataPrimitiveTests(unittest.TestCase):
    def test_simulated_metadata_schema_only(self):
        self.assertIsNone(contract._inspect_metadata(*metadata()))

    def test_authority_false_requires_boolean_false_not_zero(self):
        for index, name in ((0, "training_admitted"), (0, "publication_eligible"), (2, "publication_eligible"), (3, "training_admitted")):
            for bad in (0, 0.0, None, "false", True):
                data = metadata()
                data[index][name] = bad
                with self.subTest(index=index, field=name, bad=bad), self.assertRaises(contract.CorpusContractError):
                    contract._inspect_metadata(*data)

    def test_budget_float_bool_and_eval_boolean_aliases(self):
        for key, bad in (("max_usd", 2000.0), ("max_campaign_hours", 50.0), ("training_started", 0)):
            data = metadata()
            data[0]["budget"][key] = bad
            with self.assertRaises(contract.CorpusContractError):
                contract._inspect_metadata(*data)
        for key, bad in (("dev_is_not_final", 1), ("final_heldout_created_or_read", 0)):
            data = metadata()
            data[0]["evaluation"][key] = bad
            with self.assertRaises(contract.CorpusContractError):
                contract._inspect_metadata(*data)

    def test_filename_alias_source_chain_and_missing_hold(self):
        for mutate in (lambda d: d[0]["files"][0].update(path="../train.jsonl"),
                       lambda d: d[0]["files"][1].update(path="train.jsonl"),
                       lambda d: d[0]["files"][0].update(sha256="0" * 64),
                       lambda d: d[3]["entries"][0].update(receipt_sha256="0" * 64),
                       lambda d: d[3].update(chain_tip="0" * 64),
                       lambda d: d[0]["holds"].pop(),
                       lambda d: d[0].update(execution_authority=True)):
            data = copy.deepcopy(metadata())
            mutate(data)
            with self.assertRaises(contract.CorpusContractError):
                contract._inspect_metadata(*data)


if __name__ == "__main__":
    unittest.main()
