# SPDX-License-Identifier: Apache-2.0
"""SIMULATED native signature/policy fixtures; never real admission or trust.

Ephemeral Ed25519 private keys exist only in test memory, are never serialized
or persisted, and sign synthetic shape-only payloads. No curriculum/model file,
VM, network, subprocess, nonce, installation or production signer is used.
These tests require the pinned cryptography backend rather than skipping real
signature coverage when it is absent. All matching observations remain inert.
"""
import ast
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load_source(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_source("signed_admission")
native = load_source("native_signed_admission")
LEGACY_PROFILE = "SZL_RECEIPTAGENT_V4_ASCII_JSON_V1"
LEGACY_DOMAIN = b"SZL-RECEIPTAGENT-V4-SUPERVISED-GPU-SMOKE-V1\x00"
NATIVE_PROFILE = "SZL_RECEIPTAGENT_V4_NATIVE_ASCII_JSON_V1"
NATIVE_DOMAIN = b"SZL-RECEIPTAGENT-V4-NATIVE-SUPERVISED-GPU-SMOKE-V1\x00"
PUBLIC_RFC_KEY_HEX = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
NATIVE_HASH_FIELDS = {
    "native_descriptor_sha256", "runtime_manifest_sha256", "wheel_inventory_sha256",
    "installed_manifest_sha256", "bootstrap_sha256", "import_policy_sha256",
    "dependency_graph_sha256", "readonly_consumption_policy_sha256",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def shape(rule):
    """Construct only declared schema shape, never real runtime evidence."""
    if "const" in rule:
        return deepcopy(rule["const"])
    if "enum" in rule:
        return deepcopy(rule["enum"][0])
    kind = rule["type"]
    if kind == "object":
        return {key: shape(child) for key, child in rule["properties"].items()}
    if kind == "array":
        values = []
        for index in range(rule["minItems"]):
            item = shape(rule["items"])
            item["id"] = "SIMULATED-" + str(index)
            values.append(item)
        return values
    if kind == "integer":
        return rule["minimum"]
    if kind == "boolean":
        return False
    if "[0-9a-f]{64}" in rule.get("pattern", ""):
        return "0" * 64
    if "[0-9a-f]{40}" in rule.get("pattern", ""):
        return "0" * 40
    return "SIMULATED"


def leaf_rules(rule, path=()):
    if rule["type"] == "object":
        for key, child in rule["properties"].items():
            yield from leaf_rules(child, path + (key,))
    else:
        yield path, rule


def replace_path(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def get_path(value, path):
    for key in path:
        value = value[key]
    return value


class NativeCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cryptography
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        if cryptography.__version__ != "50.0.0":
            raise AssertionError("PINNED_TEST_CRYPTOGRAPHY_50_REQUIRED")
        cls.simulated_private_key = Ed25519PrivateKey.generate()
        cls.simulated_public_key = cls.simulated_private_key.public_key().public_bytes_raw()

    def setUp(self):
        self.policy = {
            "schema": "szl.receiptagent.v4-native-signing-policy/v1",
            "profile": NATIVE_PROFILE, "purpose": native.PURPOSE,
            "candidate_id": native.CANDIDATE,
            "issued_at_epoch": 80, "not_before_epoch": 90, "expires_at_epoch": 300,
            "max_admission_age_seconds": 100,
            "revocation_observed_at_epoch": 100, "max_revocation_age_seconds": 30,
            "public_keys": [{
                "id": "SIMULATED", "algorithm": "Ed25519",
                "public_key_hex": self.simulated_public_key.hex(),
                "public_key_sha256": sha(self.simulated_public_key),
                "not_before_epoch": 90, "expires_at_epoch": 250,
                "profile": NATIVE_PROFILE, "purpose": native.PURPOSE,
                "candidate_id": native.CANDIDATE,
            }],
            "revoked_key_ids": [],
        }
        self.payload = shape(native.payload_schema())
        self.payload["identity"].update(
            issued_at_epoch=100, not_before_epoch=110, expires_at_epoch=200,
            signer_key_id="SIMULATED", signing_policy_sha256=sha(canonical(self.policy)))
        self.payload["runner"].update(qualified_at_epoch=100, max_qualification_age_seconds=30)
        self.payload["bounds"].update(microbatch_rows=1, accumulation_batches=360,
                                      sequence_tokens=1, total_tokens=360)

    def envelope(self, payload=None, *, domain=NATIVE_DOMAIN, key=None):
        payload = deepcopy(self.payload if payload is None else payload)
        signing_key = self.simulated_private_key if key is None else key
        return {"profile": NATIVE_PROFILE, "payload": payload,
                "signature": {"algorithm": "Ed25519",
                              "key_id": payload["identity"].get("signer_key_id", "SIMULATED"),
                              "value": signing_key.sign(domain + canonical(payload)).hex()}}

    def observe(self, *, policy=None, payload=None, raw=None, **changes):
        selected_policy = deepcopy(self.policy if policy is None else policy)
        selected_payload = deepcopy(self.payload if payload is None else payload)
        policy_raw = canonical(selected_policy)
        # Bind an intentionally mutated policy freshly so negative tests reach
        # policy semantics, rather than all stopping at an unrelated old pin.
        if policy is not None and payload is None:
            selected_payload["identity"]["signing_policy_sha256"] = sha(policy_raw)
        arguments = dict(expected_payload=canonical(selected_payload), policy_raw=policy_raw,
                         expected_policy_sha256=sha(policy_raw),
                         policy_trust="INDEPENDENTLY_PINNED", now_epoch=110)
        arguments.update(changes)
        return native.observe_native_admission(
            canonical(self.envelope(selected_payload)) if raw is None else raw, **arguments)

    def assert_inert(self, observed):
        for name in ("allow", "allowed", "permit", "capability", "training_eligible",
                     "publication_eligible", "execution_authority", "operational", "launch"):
            self.assertFalse(hasattr(observed, name), name)
        with self.assertRaises(TypeError):
            bool(observed)
        if observed.admission is not None:
            with self.assertRaises(TypeError):
                bool(observed.admission)

    def assert_match(self, observed):
        self.assert_inert(observed)
        self.assertIs(observed.policy_pin_matched, True)
        self.assertIs(observed.key_policy_matched, True)
        self.assertEqual(observed.reasons, ())
        self.assertIsNotNone(observed.admission)
        for name in ("document_matches_profile", "context_matches", "public_trust_observed",
                     "time_window_matches", "revocation_observed", "signature_matches"):
            self.assertIs(getattr(observed.admission, name), True, name)
        self.assertEqual(observed.admission.reasons, ())
        self.assertEqual(observed.admission.cryptographic_verification, "RUN")

    def assert_denied(self, observed):
        self.assert_inert(observed)
        admission = observed.admission
        nested_failure = admission is None or any(
            getattr(admission, name) is not True for name in (
                "document_matches_profile", "context_matches", "public_trust_observed",
                "time_window_matches", "revocation_observed", "signature_matches"))
        self.assertTrue(observed.policy_pin_matched is not True
                        or observed.key_policy_matched is not True
                        or nested_failure or observed.reasons)


class NativeProfileTests(NativeCase):
    def test_exact_profiles_and_domains_remain_distinct(self):
        self.assertEqual(native.PROFILE, NATIVE_PROFILE)
        self.assertEqual(native.DOMAIN, NATIVE_DOMAIN)
        self.assertEqual(legacy.PROFILE, LEGACY_PROFILE)
        self.assertEqual(legacy.DOMAIN, LEGACY_DOMAIN)
        self.assertEqual(native.PURPOSE, legacy.PURPOSE)
        self.assertEqual(native.CANDIDATE, legacy.CANDIDATE)

    def test_valid_native_shape_and_signature_are_not_authority(self):
        parsed = native.parse_canonical_envelope(canonical(self.envelope()))
        self.assertEqual(parsed["payload"], self.payload)
        self.assert_match(self.observe())

    def test_observation_is_frozen(self):
        observed = self.observe()
        with self.assertRaises(FrozenInstanceError):
            observed.policy_pin_matched = False

    def test_native_schema_changes_only_explicit_profile_fields(self):
        original = legacy.payload_schema()
        modified = native.payload_schema()
        for group in original["properties"]:
            if group not in {"identity", "runtime", "runner"}:
                self.assertEqual(modified["properties"][group], original["properties"][group], group)
        old_runtime = original["properties"]["runtime"]["properties"]
        new_runtime = modified["properties"]["runtime"]["properties"]
        self.assertEqual(set(new_runtime), (set(old_runtime) - {"container_sha256"})
                         | NATIVE_HASH_FIELDS | {"kind"})
        self.assertEqual(new_runtime["kind"]["const"], "NATIVE_TREE_V1")
        for field in set(old_runtime) - {"container_sha256"}:
            self.assertEqual(old_runtime[field], new_runtime[field])
        self.assertEqual(modified["properties"]["runner"]["properties"]["backend"]["const"],
                         "LOCAL_ISOLATED")

    def test_legacy_and_native_envelopes_are_mutually_rejected(self):
        with self.assertRaises(ValueError):
            legacy.parse_canonical_envelope(canonical(self.envelope()))
        old_payload = shape(legacy.payload_schema())
        old_payload["identity"].update(issued_at_epoch=100, not_before_epoch=110, expires_at_epoch=200)
        old_payload["bounds"].update(microbatch_rows=1, accumulation_batches=360,
                                      sequence_tokens=1, total_tokens=360)
        old_envelope = {"profile": LEGACY_PROFILE, "payload": old_payload,
                        "signature": {"algorithm": "Ed25519", "key_id": "SIMULATED",
                                      "value": "0" * 128}}
        legacy.parse_canonical_envelope(canonical(old_envelope))
        with self.assertRaises(ValueError):
            native.parse_canonical_envelope(canonical(old_envelope))

    def test_container_identity_field_and_remote_backend_are_forbidden(self):
        payload = deepcopy(self.payload)
        payload["runtime"]["container_sha256"] = "0" * 64
        self.assert_denied(self.observe(payload=payload))
        payload = deepcopy(self.payload)
        payload["runner"]["backend"] = "HF_JOBS"
        self.assert_denied(self.observe(payload=payload))

    def test_closed_payload_groups_and_required_fields(self):
        for group, members in self.payload.items():
            for name in members:
                with self.subTest(group=group, missing=name):
                    payload = deepcopy(self.payload)
                    del payload[group][name]
                    self.assert_denied(self.observe(payload=payload))
            with self.subTest(group=group, unknown=True):
                payload = deepcopy(self.payload)
                payload[group]["unexpected"] = True
                self.assert_denied(self.observe(payload=payload))

    def test_every_payload_leaf_rejects_wrong_exact_type(self):
        for path, rule in leaf_rules(native.payload_schema()):
            with self.subTest(path=path):
                payload = deepcopy(self.payload)
                wrong = True if rule["type"] == "integer" else None
                replace_path(payload, path, wrong)
                self.assert_denied(self.observe(payload=payload))

    def test_every_expected_payload_leaf_change_is_detected(self):
        for path, rule in leaf_rules(native.payload_schema()):
            with self.subTest(path=path):
                expected = deepcopy(self.payload)
                value = get_path(expected, path)
                if type(value) is list:
                    changed = deepcopy(value)
                    changed[0]["sha256"] = "1" * 64
                elif type(value) is bool:
                    changed = not value
                elif type(value) is int:
                    changed = value + 1
                elif "[0-9a-f]{" in rule.get("pattern", ""):
                    changed = ("1" if value[0] != "1" else "2") * len(value)
                else:
                    changed = value + "-ALT"
                replace_path(expected, path, changed)
                self.assert_denied(self.observe(expected_payload=canonical(expected)))

    def test_schema_return_mutation_cannot_weaken_future_validation(self):
        schema = native.payload_schema()
        schema["properties"]["runtime"]["properties"].clear()
        schema["properties"]["identity"]["properties"]["purpose"]["const"] = "ALTERED"
        self.assert_match(self.observe())
        self.assertEqual(legacy.PROFILE, LEGACY_PROFILE)


class NativePolicyTests(NativeCase):
    def test_policy_pin_and_payload_policy_binding_are_separate(self):
        self.assert_denied(self.observe(expected_policy_sha256="1" * 64))
        payload = deepcopy(self.payload)
        payload["identity"]["signing_policy_sha256"] = "1" * 64
        self.assert_denied(self.observe(payload=payload))

    def test_policy_pin_exact_shape(self):
        for pin in (None, True, 1, b"0" * 64, "", "0" * 63, "A" * 64, "g" * 64):
            with self.subTest(pin=pin):
                self.assert_denied(self.observe(expected_policy_sha256=pin))

    def test_self_pinning_or_payload_hash_policy_fields_are_rejected(self):
        for field in ("policy_sha256", "expected_payload_sha256", "signature", "key_trust"):
            with self.subTest(field=field):
                policy = deepcopy(self.policy)
                policy[field] = "0" * 64
                self.assert_denied(self.observe(policy=policy))

    def test_policy_and_key_records_are_closed_and_complete(self):
        for key in self.policy:
            with self.subTest(missing=key):
                policy = deepcopy(self.policy)
                del policy[key]
                self.assert_denied(self.observe(policy=policy))
        for key in self.policy["public_keys"][0]:
            with self.subTest(missing_key_field=key):
                policy = deepcopy(self.policy)
                del policy["public_keys"][0][key]
                self.assert_denied(self.observe(policy=policy))
        policy = deepcopy(self.policy)
        policy["public_keys"][0]["unknown"] = True
        self.assert_denied(self.observe(policy=policy))

    def test_policy_and_key_exact_scope(self):
        for scope in ("profile", "purpose", "candidate_id"):
            for in_key in (False, True):
                with self.subTest(scope=scope, in_key=in_key):
                    policy = deepcopy(self.policy)
                    target = policy["public_keys"][0] if in_key else policy
                    target[scope] = "SIMULATED-OTHER"
                    self.assert_denied(self.observe(policy=policy))

    def test_key_algorithm_and_fingerprint_are_bound(self):
        for field, value in (("algorithm", "RSA"), ("public_key_sha256", "1" * 64),
                             ("public_key_hex", "00" * 32), ("public_key_hex", "0" * 63),
                             ("public_key_hex", "AA" * 32), ("public_key_hex", "ssh-ed25519 AAAA")):
            with self.subTest(field=field, value=value):
                policy = deepcopy(self.policy)
                policy["public_keys"][0][field] = value
                self.assert_denied(self.observe(policy=policy))

    def test_rfc_public_fixture_key_is_never_an_owner_key(self):
        policy = deepcopy(self.policy)
        key = policy["public_keys"][0]
        key["public_key_hex"] = PUBLIC_RFC_KEY_HEX
        key["public_key_sha256"] = sha(bytes.fromhex(PUBLIC_RFC_KEY_HEX))
        observed = self.observe(policy=policy)
        self.assertIsNot(observed.key_policy_matched, True)
        self.assert_denied(observed)

    def test_missing_duplicate_and_unsorted_key_ids(self):
        for keys in ([], [dict(self.policy["public_keys"][0], id="OTHER")],
                     self.policy["public_keys"] * 2,
                     [dict(self.policy["public_keys"][0], id="Z"),
                      dict(self.policy["public_keys"][0], id="A")]):
            with self.subTest(keys=keys):
                policy = deepcopy(self.policy)
                policy["public_keys"] = keys
                self.assert_denied(self.observe(policy=policy))

    def test_selected_signer_revoked_but_historical_revocation_allowed(self):
        policy = deepcopy(self.policy)
        policy["revoked_key_ids"] = ["SIMULATED"]
        self.assert_denied(self.observe(policy=policy))
        policy["revoked_key_ids"] = ["HISTORICAL"]
        self.assert_match(self.observe(policy=policy))

    def test_revocation_collection_exact_shape_sorted_and_unique(self):
        for revocations in (None, {}, [True], ["Z", "A"], ["A", "A"], ["bad id"]):
            with self.subTest(revocations=revocations):
                policy = deepcopy(self.policy)
                policy["revoked_key_ids"] = revocations
                self.assert_denied(self.observe(policy=policy))

    def test_independent_trust_not_inferred_from_valid_signature(self):
        for trust in (None, True, "", "REPO_DECLARED", "DECLARED", "independently_pinned"):
            with self.subTest(trust=trust):
                self.assert_denied(self.observe(policy_trust=trust))

    def test_policy_interval_future_expired_and_invalid_order(self):
        for field, value in (("issued_at_epoch", 111), ("not_before_epoch", 111),
                             ("expires_at_epoch", 110), ("not_before_epoch", 300),
                             ("issued_at_epoch", 91)):
            with self.subTest(field=field, value=value):
                policy = deepcopy(self.policy)
                policy[field] = value
                self.assert_denied(self.observe(policy=policy))

    def test_admission_interval_must_fit_policy_and_selected_key(self):
        for in_key, field, value in ((False, "expires_at_epoch", 199),
                                     (False, "not_before_epoch", 111),
                                     (True, "expires_at_epoch", 199),
                                     (True, "not_before_epoch", 111)):
            with self.subTest(in_key=in_key, field=field):
                policy = deepcopy(self.policy)
                target = policy["public_keys"][0] if in_key else policy
                target[field] = value
                self.assert_denied(self.observe(policy=policy))

    def test_exact_policy_and_key_expiry_boundary_contains_admission(self):
        policy = deepcopy(self.policy)
        policy["expires_at_epoch"] = 200
        policy["public_keys"][0]["expires_at_epoch"] = 200
        self.assert_match(self.observe(policy=policy))

    def test_revocation_future_and_stale_denied_age_boundary_inclusive(self):
        for observed in (79, 111):
            with self.subTest(observed=observed):
                policy = deepcopy(self.policy)
                policy["revocation_observed_at_epoch"] = observed
                self.assert_denied(self.observe(policy=policy))
        policy = deepcopy(self.policy)
        policy["revocation_observed_at_epoch"] = 80
        self.assert_match(self.observe(policy=policy))

    def test_policy_integer_fields_and_clock_reject_booleans(self):
        for field, value in self.policy.items():
            if type(value) is int:
                with self.subTest(field=field):
                    policy = deepcopy(self.policy)
                    policy[field] = True
                    self.assert_denied(self.observe(policy=policy))
        for field in ("not_before_epoch", "expires_at_epoch"):
            policy = deepcopy(self.policy)
            policy["public_keys"][0][field] = True
            self.assert_denied(self.observe(policy=policy))
        for clock in (None, True, -1, 110.0, "110"):
            with self.subTest(clock=clock):
                self.assert_denied(self.observe(now_epoch=clock))

    def test_policy_age_caps_reject_zero_and_above_hour(self):
        for field in ("max_admission_age_seconds", "max_revocation_age_seconds"):
            for value in (0, -1, 3601):
                with self.subTest(field=field, value=value):
                    policy = deepcopy(self.policy)
                    policy[field] = value
                    self.assert_denied(self.observe(policy=policy))

    def test_admission_age_boundary_is_inclusive(self):
        policy = deepcopy(self.policy)
        policy["max_admission_age_seconds"] = 10
        self.assert_match(self.observe(policy=policy))
        policy["max_admission_age_seconds"] = 9
        self.assert_denied(self.observe(policy=policy))

    def test_payload_activation_inclusive_expiry_exclusive(self):
        self.assert_denied(self.observe(now_epoch=109))
        self.assert_match(self.observe(now_epoch=110))
        self.assert_denied(self.observe(now_epoch=200))


class NativeSignatureTests(NativeCase):
    def test_real_cross_domain_nul_and_prehash_signatures_fail(self):
        for domain in (LEGACY_DOMAIN, NATIVE_DOMAIN[:-1], NATIVE_DOMAIN + b"\x00",
                       b"SZL-OTHER-PURPOSE\x00"):
            with self.subTest(domain=domain):
                observed = self.observe(raw=canonical(self.envelope(domain=domain)))
                self.assert_denied(observed)
                self.assertIs(observed.admission.signature_matches, False)
        envelope = self.envelope()
        envelope["signature"]["value"] = self.simulated_private_key.sign(
            hashlib.sha256(NATIVE_DOMAIN + canonical(self.payload)).digest()).hex()
        self.assertIs(self.observe(raw=canonical(envelope)).admission.signature_matches, False)

    def test_signature_key_id_payload_and_signature_byte_tampering(self):
        for mutation in ("key_id", "payload", "signature"):
            with self.subTest(mutation=mutation):
                envelope = self.envelope()
                if mutation == "key_id":
                    envelope["signature"]["key_id"] = "OTHER"
                elif mutation == "payload":
                    envelope["payload"]["runtime"]["bootstrap_sha256"] = "1" * 64
                else:
                    signature = envelope["signature"]["value"]
                    envelope["signature"]["value"] = ("0" if signature[0] != "0" else "1") + signature[1:]
                self.assert_denied(self.observe(raw=canonical(envelope)))

    def test_different_ephemeral_public_key_cannot_verify_signature(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        other = Ed25519PrivateKey.generate()
        self.assertIs(self.observe(raw=canonical(self.envelope(key=other))).admission.signature_matches, False)

    def test_backend_version_mismatch_cannot_report_signature_match(self):
        import cryptography
        with patch.object(cryptography, "__version__", "0.0.0"):
            observed = self.observe()
        self.assert_denied(observed)
        self.assertIsNot(observed.admission.signature_matches, True)
        self.assertEqual(observed.admission.cryptographic_verification, "NOT_RUN")

    def test_invalid_inputs_are_observations_not_unhandled_errors(self):
        for raw in (None, True, 1, "{}", bytearray(b"{}"), b"{}", b"{", b"[]",
                    b'{"a":1,"a":2}', b'{"a":NaN}', b" " * 65537):
            with self.subTest(raw_type=type(raw).__name__):
                self.assert_denied(self.observe(raw=raw) if raw is not None else
                                   native.observe_native_admission(
                                       raw, expected_payload=canonical(self.payload),
                                       policy_raw=canonical(self.policy),
                                       expected_policy_sha256=sha(canonical(self.policy)),
                                       policy_trust="INDEPENDENTLY_PINNED", now_epoch=110))

    def test_malformed_policy_and_expected_context_are_observations(self):
        for field in ("policy_raw", "expected_payload"):
            for raw in (None, True, {}, "{}", bytearray(b"{}"), b"{}", b"[]", b"{",
                        b'{"a":1,"a":2}', b" " * 65537):
                with self.subTest(field=field, kind=type(raw).__name__):
                    self.assert_denied(self.observe(**{field: raw}))

    def test_noncanonical_policy_and_envelope_bytes_rejected(self):
        raw = json.dumps(self.policy, indent=2).encode("ascii")
        self.assert_denied(self.observe(policy_raw=raw, expected_policy_sha256=sha(raw)))
        self.assert_denied(self.observe(raw=json.dumps(self.envelope(), indent=2).encode("ascii")))


class NativeImportBoundaryTests(unittest.TestCase):
    def test_production_observer_has_no_launch_filesystem_or_private_key_api(self):
        tree = ast.parse((HERE / "native_signed_admission.py").read_text(encoding="utf-8"))
        allowed = {"__future__", "dataclasses", "copy", "hashlib", "json", "re", "signed_admission",
                   "cryptography", "cryptography.exceptions",
                   "cryptography.hazmat.primitives.asymmetric.ed25519"}
        forbidden_names = {"__import__", "eval", "exec", "open", "compile", "input", "subprocess",
                           "torch", "transformers", "tokenizers", "huggingface_hub", "requests",
                           "httpx", "nacl", "Ed25519PrivateKey", "os", "sys", "pathlib", "socket",
                           "ctypes", "importlib", "BoundRun"}
        forbidden_calls = {"sign", "generate", "spawn", "Popen", "run", "system", "load_model",
                           "from_pretrained", "read_bytes", "read_text", "write_bytes", "write_text",
                           "load_bound_run", "consume_nonce"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name, allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Name):
                self.assertNotIn(node.id, forbidden_names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden_calls)


if __name__ == "__main__":
    unittest.main()
