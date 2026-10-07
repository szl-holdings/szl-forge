"""SIMULATED shape fixtures for a non-authorizing observational verifier.

No key generation, signing, real public-key pin, corpus or model access occurs.
The literal zero key/signature bytes below are NOT a signature or a trust pin.
These CPU tests are not training, evaluation, authorization or publication proof.
The plan's 44 negative cases are not claimed as implemented full coverage.
"""

import ast
from copy import deepcopy
from dataclasses import fields, FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "signed_admission.py"
SPEC = importlib.util.spec_from_file_location("synthetic_signed_admission", MODULE_PATH)
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def simulated_shape(rule):
    """Only schema literals/shape; no fixture has real qualification evidence."""
    if "const" in rule:
        return rule["const"]
    if "enum" in rule:
        return rule["enum"][0]
    kind = rule["type"]
    if kind == "object":
        return {key: simulated_shape(child) for key, child in rule["properties"].items()}
    if kind == "array":
        result = []
        for index in range(rule["minItems"]):
            item = simulated_shape(rule["items"])
            item["id"] = "SIMULATED-" + str(index)
            result.append(item)
        return result
    if kind == "integer":
        return rule["minimum"]
    if kind == "boolean":
        return False
    if "[0-9a-f]{64}" in rule.get("pattern", ""):
        return "0" * 64
    if "[0-9a-f]{40}" in rule.get("pattern", ""):
        return "0" * 40
    return "SIMULATED"


def synthetic_payload():
    result = simulated_shape(admission.payload_schema())
    result["identity"].update(issued_at_epoch=100, not_before_epoch=110, expires_at_epoch=200)
    result["runner"].update(qualified_at_epoch=100, max_qualification_age_seconds=30)
    result["bounds"].update(microbatch_rows=1, accumulation_batches=360,
                           sequence_tokens=1, total_tokens=360)
    return result


def synthetic_envelope(payload=None):
    payload = synthetic_payload() if payload is None else deepcopy(payload)
    return {"profile": admission.PROFILE, "payload": payload,
            "signature": {"algorithm": "Ed25519",
                          "key_id": payload["identity"]["signer_key_id"],
                          "value": "0" * 128}}


def leaf_rules(rule, prefix=()):
    if rule["type"] == "object":
        for key, child in rule["properties"].items():
            yield from leaf_rules(child, prefix + (key,))
    else:
        yield prefix, rule


def get_path(value, path):
    for key in path:
        value = value[key]
    return value


def set_path(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def mutated_value(value, rule):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is list:
        replacement = deepcopy(value)
        replacement[0]["sha256"] = "1" * 64
        return replacement
    if "enum" in rule:
        return next(item for item in rule["enum"] if item != value)
    if "[0-9a-f]{" in rule.get("pattern", ""):
        return ("1" if value[0] != "1" else "2") * len(value)
    return value + "-SIMULATED-ALT"


class SyntheticCase(unittest.TestCase):
    def assert_rejection(self, raw, code, envelope=False):
        parser = admission.parse_canonical_envelope if envelope else admission.parse_canonical_bytes
        with self.assertRaises(admission.AdmissionError) as caught:
            parser(raw)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)

    def observe(self, payload=None, **changes):
        payload = synthetic_payload() if payload is None else payload
        arguments = dict(expected_payload=canonical(payload),
                         public_keys={payload["identity"]["signer_key_id"]: b"\x00" * 32},
                         key_trust="INDEPENDENTLY_PINNED", now_epoch=110,
                         revoked_key_ids=frozenset(), max_age_seconds=100)
        arguments.update(changes)
        return admission.observe_admission(canonical(synthetic_envelope(payload)), **arguments)

    def assert_not_authorization(self, result):
        self.assertIsNone(result.signature_matches)
        self.assertEqual(result.cryptographic_readiness, "NOT_READY")
        self.assertEqual(result.cryptographic_verification, "NOT_RUN")
        self.assertIn("CRYPTO_BACKEND_NOT_QUALIFIED", result.reasons)
        for name in ("allow", "allowed", "permit", "capability", "training_eligible",
                     "publication_eligible", "execution_authority", "launch"):
            self.assertFalse(hasattr(result, name), name)
        with self.assertRaisesRegex(TypeError, "ADMISSION_OBSERVATION_IS_NOT_AUTHORIZATION"):
            bool(result)


class ByteProfileTests(SyntheticCase):
    def test_exact_bytes_only(self):
        for value in (None, "{}", bytearray(b"{}"), memoryview(b"{}"), True, 1):
            with self.subTest(type=type(value).__name__):
                self.assert_rejection(value, "BYTES_REQUIRED")

    def test_canonical_scalars_and_escaping(self):
        for value in (0, admission.MAX_INTEGER, True, False, None, '"\\/', " ~", [], {}):
            with self.subTest(value=repr(value)):
                self.assertEqual(admission.parse_canonical_bytes(canonical(value)), value)

    def test_noncanonical_encoding(self):
        examples = ((b"\xef\xbb\xbf{}", "BOM_FORBIDDEN"),
                    (b'"\xff"', "INVALID_UTF8"),
                    (b'{"b":1,"a":2}', "NONCANONICAL_BYTES"),
                    (b'"\\/"', "NONCANONICAL_BYTES"),
                    (b'"\\u0041"', "NONCANONICAL_BYTES"),
                    (b'"\\u0020"', "NONCANONICAL_BYTES"),
                    (b'"\\n"', "STRING_PROFILE"),
                    ('"é"'.encode("utf-8"), "STRING_PROFILE"),
                    (b'"\\u00e9"', "STRING_PROFILE"),
                    (b'"\\ud800"', "STRING_PROFILE"),
                    (b'"\x00"', "STRING_PROFILE"),
                    (b"{} ", "TRAILING_OR_NONCANONICAL_DATA"),
                    (b"{}{}", "TRAILING_OR_NONCANONICAL_DATA"))
        for raw, code in examples:
            with self.subTest(raw=raw):
                self.assert_rejection(raw, code)

    def test_syntax_and_duplicate_keys(self):
        for raw in (b"", b" ", b"[", b"{", b"{\"a\"}", b"[0,]", b'"\\uGGGG"'):
            with self.subTest(raw=raw):
                with self.assertRaises(admission.AdmissionError):
                    admission.parse_canonical_bytes(raw)
        for raw in (b'{"a":0,"a":1}', b'{"a":{"x":0,"x":1}}'):
            with self.subTest(raw=raw):
                self.assert_rejection(raw, "DUPLICATE_KEY")

    def test_integer_profile_and_overflow(self):
        for raw, code in ((b"01", "INTEGER_SPELLING"),
                          (b"9223372036854775808", "INTEGER_LIMIT"),
                          (b"10000000000000000000", "INTEGER_LIMIT"),
                          (b"1.0", "INTEGER_PROFILE"), (b"1e0", "INTEGER_PROFILE"),
                          (b"1E0", "INTEGER_PROFILE"), (b"1+0", "INTEGER_PROFILE"),
                          (b"1-0", "INTEGER_PROFILE"),
                          (b"-1", "JSON_SYNTAX_OR_NUMBER_PROFILE"),
                          (b"+1", "JSON_SYNTAX_OR_NUMBER_PROFILE"),
                          (b"NaN", "JSON_SYNTAX_OR_NUMBER_PROFILE")):
            with self.subTest(raw=raw):
                self.assert_rejection(raw, code)
        self.assertEqual(admission.parse_canonical_bytes(b"9223372036854775807"), admission.MAX_INTEGER)

    def test_string_limit_inclusive(self):
        self.assertEqual(admission.parse_canonical_bytes(canonical("x" * 8192)), "x" * 8192)
        self.assert_rejection(canonical("x" * 8193), "STRING_LIMIT")

    def test_envelope_byte_limit_inclusive(self):
        values = ["x" * 8192] * 7 + ["x" * 8167]
        raw = canonical(values)
        self.assertEqual(len(raw), 65536)
        self.assertEqual(admission.parse_canonical_bytes(raw), values)
        self.assert_rejection(raw + b" ", "ENVELOPE_TOO_LARGE")

    def test_container_limits_inclusive(self):
        mapping = {"k" + str(index).zfill(3): 0 for index in range(128)}
        self.assertEqual(admission.parse_canonical_bytes(canonical(mapping)), mapping)
        mapping["k128"] = 0
        self.assert_rejection(canonical(mapping), "OBJECT_FIELD_LIMIT")
        self.assertEqual(len(admission.parse_canonical_bytes(canonical([0] * 1024))), 1024)
        self.assert_rejection(canonical([0] * 1025), "ARRAY_ITEM_LIMIT")

    def test_container_depth_root_is_one(self):
        raw = b"[" * 12 + b"0" + b"]" * 12
        admission.parse_canonical_bytes(raw)
        self.assert_rejection(b"[" + raw + b"]", "DEPTH_LIMIT")
        raw = b'{"a":' * 12 + b"0" + b"}" * 12
        admission.parse_canonical_bytes(raw)
        self.assert_rejection(b'{"a":' + raw + b"}", "DEPTH_LIMIT")


class ClosedSchemaTests(SyntheticCase):
    def test_simulated_envelope_shape(self):
        envelope = synthetic_envelope()
        self.assertEqual(admission.parse_canonical_envelope(canonical(envelope)), envelope)
        self.assertEqual(envelope["payload"]["curriculum"]["evidence_class"], "SIMULATED")

    def test_all_required_fields_and_groups_closed(self):
        rules = [((), admission.profile_schema())]
        def visit(rule, prefix):
            if rule["type"] == "object":
                for key, child in rule["properties"].items():
                    if child["type"] == "object":
                        rules.append((prefix + (key,), child))
                        visit(child, prefix + (key,))
        visit(admission.profile_schema(), ())
        for path, rule in rules:
            for key in rule["properties"]:
                with self.subTest(path=path, missing=key):
                    value = synthetic_envelope()
                    del get_path(value, path)[key]
                    self.assert_rejection(canonical(value), "SCHEMA_FIELDS", envelope=True)
            with self.subTest(path=path, unexpected=True):
                value = synthetic_envelope()
                get_path(value, path)["SIMULATED_UNEXPECTED"] = 0
                self.assert_rejection(canonical(value), "SCHEMA_FIELDS", envelope=True)

    def test_every_leaf_rejects_wrong_exact_type(self):
        for path, rule in leaf_rules(admission.profile_schema()):
            with self.subTest(path=path):
                value = synthetic_envelope()
                set_path(value, path, None)
                self.assert_rejection(canonical(value), "SCHEMA_TYPE", envelope=True)
        for path, rule in leaf_rules(admission.payload_schema()):
            if rule["type"] == "integer":
                with self.subTest(bool_as_integer=path):
                    value = synthetic_envelope()
                    set_path(value["payload"], path, True)
                    self.assert_rejection(canonical(value), "SCHEMA_TYPE", envelope=True)

    def test_all_closed_constants(self):
        for path, rule in leaf_rules(admission.profile_schema()):
            if "const" in rule:
                with self.subTest(path=path):
                    value = synthetic_envelope()
                    set_path(value, path, mutated_value(get_path(value, path), rule))
                    self.assert_rejection(canonical(value), "SCHEMA_CONSTANT", envelope=True)

    def test_signature_shapes_and_key_id_binding(self):
        for replacement, code in (("0" * 127, "SCHEMA_STRING_PATTERN"),
                                  ("0" * 129, "SCHEMA_STRING_LIMIT"),
                                  ("A" * 128, "SCHEMA_STRING_PATTERN"),
                                  ("g" * 128, "SCHEMA_STRING_PATTERN"),
                                  ("", "SCHEMA_STRING_LIMIT")):
            with self.subTest(replacement=replacement[:8], length=len(replacement)):
                value = synthetic_envelope()
                value["signature"]["value"] = replacement
                self.assert_rejection(canonical(value), code, envelope=True)
        value = synthetic_envelope()
        value["signature"]["key_id"] = "SIMULATED-OTHER"
        self.assert_rejection(canonical(value), "SIGNER_KEY_ID_MISMATCH", envelope=True)

    def test_closure_cardinality_duplicate_ids_and_closed_members(self):
        for path, rule in leaf_rules(admission.payload_schema()):
            if rule["type"] != "array":
                continue
            for replacement, code in (([], "SCHEMA_ARRAY_LIMIT"),
                                      ([{"id": "SIMULATED", "sha256": "0" * 64}] *
                                       (rule["maxItems"] + 1), "SCHEMA_ARRAY_LIMIT")):
                with self.subTest(path=path, length=len(replacement)):
                    value = synthetic_envelope()
                    set_path(value["payload"], path, replacement)
                    self.assert_rejection(canonical(value), code, envelope=True)
            with self.subTest(path=path, duplicate=True):
                value = synthetic_envelope()
                replacement = get_path(value["payload"], path)
                if len(replacement) == 1:
                    replacement.append(deepcopy(replacement[0]))
                else:
                    replacement[1]["id"] = replacement[0]["id"]
                self.assert_rejection(canonical(value), "DUPLICATE_CLOSURE_ID", envelope=True)
            with self.subTest(path=path, member_closed=True):
                value = synthetic_envelope()
                get_path(value["payload"], path)[0]["extra"] = 0
                self.assert_rejection(canonical(value), "SCHEMA_FIELDS", envelope=True)

    def test_relational_shape_requirements(self):
        for group, key, replacement, code in (
                ("identity", "issued_at_epoch", 111, "TIME_ORDER"),
                ("identity", "not_before_epoch", 200, "TIME_ORDER"),
                ("identity", "expires_at_epoch", 110, "TIME_ORDER"),
                ("bounds", "accumulation_batches", 359, "GRADIENT_COVERAGE_SHAPE"),
                ("bounds", "sequence_tokens", 361, "TOKEN_BOUND_ORDER")):
            with self.subTest(group=group, key=key):
                value = synthetic_envelope()
                value["payload"][group][key] = replacement
                self.assert_rejection(canonical(value), code, envelope=True)

    def test_closure_maximum_cardinality_inclusive(self):
        for path, rule in leaf_rules(admission.payload_schema()):
            if rule["type"] == "array":
                with self.subTest(path=path):
                    value = synthetic_envelope()
                    closure = [{"id": "SIMULATED-" + str(i), "sha256": "0" * 64}
                               for i in range(rule["maxItems"])]
                    set_path(value["payload"], path, closure)
                    self.assertEqual(admission.parse_canonical_envelope(canonical(value)), value)

    def test_identifier_maximum_inclusive_and_above_rejected(self):
        for path, rule in leaf_rules(admission.payload_schema()):
            if rule["type"] == "string" and "[A-Za-z0-9]" in rule.get("pattern", ""):
                with self.subTest(path=path, at_maximum=True):
                    value = synthetic_envelope()
                    set_path(value["payload"], path, "S" + "x" * 127)
                    if path == ("identity", "signer_key_id"):
                        value["signature"]["key_id"] = value["payload"]["identity"]["signer_key_id"]
                    self.assertEqual(admission.parse_canonical_envelope(canonical(value)), value)
                with self.subTest(path=path, above_maximum=True):
                    value = synthetic_envelope()
                    set_path(value["payload"], path, "S" + "x" * 128)
                    self.assert_rejection(canonical(value), "SCHEMA_STRING_LIMIT", envelope=True)

    def test_integer_schema_boundaries(self):
        for path, rule in leaf_rules(admission.payload_schema()):
            if rule["type"] != "integer" or "minimum" not in rule:
                continue
            with self.subTest(path=path, below=True):
                if rule["minimum"] == 0:
                    continue  # Negative lexical spelling has its separate parser test.
                value = synthetic_envelope()
                set_path(value["payload"], path, rule["minimum"] - 1)
                self.assert_rejection(canonical(value), "SCHEMA_INTEGER_BOUND", envelope=True)
            if rule["maximum"] < admission.MAX_INTEGER:
                with self.subTest(path=path, above=True):
                    value = synthetic_envelope()
                    set_path(value["payload"], path, rule["maximum"] + 1)
                    self.assert_rejection(canonical(value), "SCHEMA_INTEGER_BOUND", envelope=True)

    def test_schema_artifact_exact_parity_and_integer_precision(self):
        # This is the only artifact read other than the tested module itself.
        saved = json.loads((HERE / "schemas" / "training-admission.schema.json").read_bytes())
        self.assertEqual(saved, admission.profile_schema())
        self.assertEqual(saved["properties"]["payload"]["properties"]["identity"]
                         ["properties"]["issued_at_epoch"]["maximum"], 9223372036854775807)
        self.assertNotEqual(9223372036854775807, 9223372036854776000)

    def test_mutable_schema_helpers_do_not_weaken_later_checker(self):
        first = admission.profile_schema()
        first["properties"]["payload"]["properties"]["curriculum"]["properties"]["train_rows"]["const"] = 1
        first["properties"]["signature"]["properties"]["value"]["pattern"] = ".*"
        second = admission.payload_schema()
        second["properties"]["gates"]["properties"]["nemo_module_closure"]["items"]["properties"]["sha256"]["pattern"] = ".*"
        value = synthetic_envelope()
        value["payload"]["curriculum"]["train_rows"] = 1
        self.assert_rejection(canonical(value), "SCHEMA_CONSTANT", envelope=True)
        value = synthetic_envelope()
        value["payload"]["gates"]["nemo_module_closure"][0]["sha256"] = "SIMULATED"
        self.assert_rejection(canonical(value), "SCHEMA_STRING_PATTERN", envelope=True)


class ContextObservationTests(SyntheticCase):
    def test_matching_simulated_shape_is_never_authorization(self):
        result = self.observe()
        self.assertTrue(result.document_matches_profile)
        self.assertTrue(result.context_matches)
        self.assertTrue(result.time_window_matches)
        self.assertTrue(result.public_trust_observed)  # Caller string observed, not attested.
        self.assertTrue(result.revocation_observed)
        self.assertEqual(result.mismatch_fields, ())
        self.assert_not_authorization(result)
        with self.assertRaises(FrozenInstanceError):
            result.signature_matches = True

    def test_every_expected_payload_leaf_mutation_is_detected_or_invalid(self):
        baseline = synthetic_payload()
        for path, rule in leaf_rules(admission.payload_schema()):
            with self.subTest(path=path):
                expected = deepcopy(baseline)
                set_path(expected, path, mutated_value(get_path(expected, path), rule))
                result = self.observe(baseline, expected_payload=canonical(expected))
                self.assertTrue(result.document_matches_profile)
                self.assertIsNot(result.context_matches, True)
                if result.context_matches is False:
                    self.assertIn("payload." + ".".join(path), result.mismatch_fields)
                    self.assertIn("EXPECTED_CONTEXT_MISMATCH", result.reasons)
                else:
                    self.assertIn("EXPECTED_CONTEXT_INVALID", result.reasons)
                self.assert_not_authorization(result)

    def test_expected_closure_list_order_matters(self):
        baseline = synthetic_payload()
        expected = deepcopy(baseline)
        expected["gates"]["nemo_module_closure"].reverse()
        result = self.observe(baseline, expected_payload=canonical(expected))
        self.assertFalse(result.context_matches)
        self.assertIn("payload.gates.nemo_module_closure", result.mismatch_fields)

    def test_invalid_expected_payload_is_not_compared(self):
        for raw in (None, {}, True, b"{}", b"[]", b"null", b"{bad}", b" {}",
                    b'{"x":0,"x":1}'):
            with self.subTest(type=type(raw).__name__, value=repr(raw)):
                result = self.observe(expected_payload=raw)
                self.assertIsNone(result.context_matches)
                self.assertEqual(result.mismatch_fields, ())
                self.assertIn("EXPECTED_CONTEXT_INVALID", result.reasons)
                self.assert_not_authorization(result)

    def test_malformed_document_stays_non_authorizing(self):
        result = admission.observe_admission(b"{}", expected_payload=None,
                                            public_keys=None, key_trust=None,
                                            now_epoch=None, revoked_key_ids=None,
                                            max_age_seconds=None)
        self.assertFalse(result.document_matches_profile)
        self.assertIsNone(result.context_matches)
        self.assert_not_authorization(result)

    def test_missing_or_malformed_public_keys_short_circuit(self):
        malformed = (None, [], True, False, 0, "SIMULATED", {},
                     {"SIMULATED": b"0" * 31}, {"SIMULATED": b"0" * 33},
                     {"SIMULATED": "0" * 32}, {"SIMULATED": bytearray(32)},
                     {"SIMULATED": memoryview(b"0" * 32)},
                     {True: b"0" * 32}, {"!": b"0" * 32},
                     {"SIMULATED-OTHER": b"0" * 32},
                     {"K" + str(i): b"0" * 32 for i in range(129)})
        for public_keys in malformed:
            with self.subTest(type=type(public_keys).__name__, size=len(public_keys) if hasattr(public_keys, "__len__") else None):
                result = self.observe(public_keys=public_keys)
                self.assertFalse(result.public_trust_observed)
                self.assertIn("PUBLIC_KEY_PIN_MISSING_OR_INVALID", result.reasons)
                self.assert_not_authorization(result)

    def test_repository_declared_trust_is_not_independent(self):
        for key_trust in (None, True, 1, "", "REPO_DECLARED", "DECLARED", "independently_pinned"):
            with self.subTest(trust=key_trust):
                result = self.observe(key_trust=key_trust)
                self.assertFalse(result.public_trust_observed)
                self.assertIn("PUBLIC_TRUST_NOT_INDEPENDENTLY_PINNED", result.reasons)

    def test_caller_key_and_revocation_cardinality_maximum_inclusive(self):
        keys = {"SIMULATED": b"\x00" * 32}
        keys.update({"SIMULATED-" + str(i): b"\x00" * 32 for i in range(127)})
        result = self.observe(public_keys=keys)
        self.assertEqual(len(keys), 128)
        self.assertTrue(result.public_trust_observed)
        revoked = frozenset("SIMULATED-OTHER-" + str(i) for i in range(128))
        result = self.observe(revoked_key_ids=revoked)
        self.assertTrue(result.revocation_observed)
        self.assert_not_authorization(result)

    def test_revocation_requires_fresh_caller_shape(self):
        for state in (None, [], set(), True, frozenset({True}), frozenset({"!"}),
                      frozenset("K" + str(i) for i in range(129))):
            with self.subTest(type=type(state).__name__):
                result = self.observe(revoked_key_ids=state)
                self.assertFalse(result.revocation_observed)
                self.assertIn("REVOCATION_STATE_UNAVAILABLE", result.reasons)
        result = self.observe(revoked_key_ids=frozenset({"SIMULATED"}))
        self.assertFalse(result.revocation_observed)
        self.assertIn("KEY_REVOKED", result.reasons)
        result = self.observe(revoked_key_ids=frozenset({"SIMULATED-OTHER"}))
        self.assertTrue(result.revocation_observed)

    def test_clock_exact_integer_and_max_age_required(self):
        for key, values in (("now_epoch", (None, True, 1.0, -1, admission.MAX_INTEGER + 1)),
                            ("max_age_seconds", (None, True, 1.0, 0, -1, admission.MAX_INTEGER + 1))):
            for value in values:
                with self.subTest(key=key, value=value):
                    result = self.observe(**{key: value})
                    self.assertIsNone(result.time_window_matches)
                    self.assertIn("CLOCK_OR_MAX_AGE_UNAVAILABLE", result.reasons)

    def test_not_before_inclusive_and_expiry_exclusive(self):
        payload = synthetic_payload()
        payload["runner"]["max_qualification_age_seconds"] = 200
        for now, matches, reason in ((109, False, "NOT_YET_VALID"),
                                     (110, True, None), (199, True, None),
                                     (200, False, "EXPIRED"), (201, False, "EXPIRED")):
            with self.subTest(now=now):
                result = self.observe(payload, now_epoch=now, max_age_seconds=200)
                self.assertEqual(result.time_window_matches, matches)
                if reason:
                    self.assertIn(reason, result.reasons)

    def test_admission_maximum_age_inclusive(self):
        for now, matches in ((120, True), (121, False)):
            with self.subTest(now=now):
                result = self.observe(now_epoch=now, max_age_seconds=20)
                self.assertEqual(result.time_window_matches, matches)
                self.assertEqual("ADMISSION_TOO_OLD" in result.reasons, not matches)

    def test_qualification_maximum_age_inclusive_and_future_denied(self):
        for now, matches in ((130, True), (131, False)):
            with self.subTest(now=now):
                result = self.observe(now_epoch=now)
                self.assertEqual(result.time_window_matches, matches)
                self.assertEqual("RUNNER_QUALIFICATION_STALE_OR_FUTURE" in result.reasons, not matches)
        payload = synthetic_payload()
        payload["runner"]["qualified_at_epoch"] = 111
        result = self.observe(payload, now_epoch=110)
        self.assertFalse(result.time_window_matches)
        self.assertIn("RUNNER_QUALIFICATION_STALE_OR_FUTURE", result.reasons)

    @unittest.skip("NOT_RUN V14 real Ed25519 mismatch: no admitted backend or verified canonical-domain fixture")
    def test_V14_real_signature_verification_mismatch_not_run(self):
        """Coverage gap, not a crypto test or an expected success."""

    @unittest.skip("NOT_RUN V42 positive canonical signature plus false eligibility: backend/fixture absent; no candidate reads")
    def test_V42_positive_signature_and_false_policy_full_coverage_not_run(self):
        """The current false-policy/full positive-crypto path is not covered."""


class ImportBoundaryTests(unittest.TestCase):
    def test_module_ast_has_only_observational_standard_library_imports(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        allowed_imports = {"dataclasses", "copy", "json", "re"}
        forbidden_names = {"__import__", "eval", "exec", "open", "compile", "input",
                           "subprocess", "torch", "transformers", "tokenizers",
                           "huggingface_hub", "requests", "httpx", "cryptography", "nacl",
                           "os", "sys", "pathlib", "socket", "ctypes", "importlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name, allowed_imports)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertIn(node.module, allowed_imports)
            elif isinstance(node, ast.Name):
                self.assertNotIn(node.id, forbidden_names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"sign", "generate", "spawn", "Popen", "run",
                                                 "system", "load_model", "from_pretrained",
                                                 "read_bytes", "read_text", "write_bytes", "write_text"})
        self.assertEqual({item.name for item in fields(admission.AdmissionObservation)},
                         {"document_matches_profile", "context_matches", "public_trust_observed",
                          "time_window_matches", "revocation_observed", "signature_matches",
                          "mismatch_fields", "reasons", "cryptographic_readiness",
                          "cryptographic_verification"})


if __name__ == "__main__":
    unittest.main()
