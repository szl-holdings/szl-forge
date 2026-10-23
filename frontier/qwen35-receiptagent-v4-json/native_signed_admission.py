# SPDX-License-Identifier: Apache-2.0
"""Closed native-context signature observations; never a training permit.

The existing container profile/domain is unchanged. Native commitments bind
different bytes and use a distinct signature domain. This module does not read
files, authenticate the executing interpreter, establish dependency completeness,
enroll a key, sign, consume a nonce or dispatch a worker. The caller must obtain
the policy pin, clock, revocation freshness and expected context independently.
A policy describing its own provenance is not accepted as evidence of trust.

The dependency direction is acyclic: external pin -> policy -> public keys;
the signed payload commits the policy hash, never the reverse. Model/runtime
identity observations and signature integrity remain separate from authority.
"""
from dataclasses import dataclass
import hashlib

import signed_admission as container

PROFILE = "SZL_RECEIPTAGENT_V4_NATIVE_ASCII_JSON_V1"
PURPOSE = container.PURPOSE
CANDIDATE = container.CANDIDATE
DOMAIN = b"SZL-RECEIPTAGENT-V4-NATIVE-SUPERVISED-GPU-SMOKE-V1\x00"
POLICY_SCHEMA = "szl.receiptagent.v4-native-signing-policy/v1"
CRYPTOGRAPHY_VERSION = container.CRYPTOGRAPHY_VERSION
AdmissionError = container.AdmissionError
parse_canonical_bytes = container.parse_canonical_bytes
PUBLIC_TEST_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
NATIVE_HASH_FIELDS = (
    "native_descriptor_sha256", "runtime_manifest_sha256", "wheel_inventory_sha256",
    "installed_manifest_sha256", "bootstrap_sha256", "import_policy_sha256",
    "dependency_graph_sha256", "readonly_consumption_policy_sha256",
)


def _require(condition, code):
    if condition is not True:
        raise AdmissionError(code)


def payload_schema():
    """Separate fixed profile; the v1 schema is neither mutated nor accepted."""
    result = container.payload_schema()
    identity = result["properties"]["identity"]
    identity["properties"]["schema_version"] = container._constant(
        "szl.receiptagent.v4-native-training-admission/v1")
    identity["properties"]["profile"] = container._constant(PROFILE)
    identity["properties"]["signing_policy_sha256"] = dict(container._HASH)
    identity["required"].append("signing_policy_sha256")
    runtime = result["properties"]["runtime"]
    del runtime["properties"]["container_sha256"]
    runtime["required"].remove("container_sha256")
    runtime["properties"]["kind"] = container._constant("NATIVE_TREE_V1")
    runtime["properties"].update(container._hash_fields(*NATIVE_HASH_FIELDS))
    runtime["required"].extend(("kind", *NATIVE_HASH_FIELDS))
    result["properties"]["runner"]["properties"]["backend"] = container._constant("LOCAL_ISOLATED")
    return result


def profile_schema():
    result = container.profile_schema()
    result["title"] = "ReceiptAgent v4 native non-authorizing sidecar profile"
    result["properties"]["profile"] = container._constant(PROFILE)
    result["properties"]["payload"] = payload_schema()
    return result


def _validate_payload(payload):
    container._validate(payload, payload_schema())
    identity, bounds = payload["identity"], payload["bounds"]
    _require(identity["issued_at_epoch"] <= identity["not_before_epoch"]
             < identity["expires_at_epoch"], "TIME_ORDER")
    _require(bounds["microbatch_rows"] * bounds["accumulation_batches"] == 360,
             "GRADIENT_COVERAGE_SHAPE")
    _require(bounds["total_tokens"] >= bounds["sequence_tokens"], "TOKEN_BOUND_ORDER")


def parse_canonical_envelope(raw):
    envelope = parse_canonical_bytes(raw)
    container._validate(envelope, profile_schema())
    _validate_payload(envelope["payload"])
    _require(envelope["signature"]["key_id"]
             == envelope["payload"]["identity"]["signer_key_id"], "SIGNER_KEY_ID_MISMATCH")
    return envelope


def _policy_schema():
    scope = {"profile": container._constant(PROFILE), "purpose": container._constant(PURPOSE),
             "candidate_id": container._constant(CANDIDATE)}
    times = {name: container._integer(0) for name in ("not_before_epoch", "expires_at_epoch")}
    return container._object({
        "schema": container._constant(POLICY_SCHEMA), **scope, **times,
        "issued_at_epoch": container._integer(0),
        "max_admission_age_seconds": container._integer(1, 3600),
        "revocation_observed_at_epoch": container._integer(0),
        "max_revocation_age_seconds": container._integer(1, 3600),
        "public_keys": {"type": "array", "minItems": 1, "maxItems": 16,
                        "items": container._object({
                            "id": dict(container._ID), "algorithm": container._constant("Ed25519"),
                            "public_key_hex": dict(container._HASH),
                            "public_key_sha256": dict(container._HASH), **times, **scope})},
        # Validated separately because v1's closure-array rule requires object IDs.
        "revoked_key_ids": {"type": "array", "minItems": 0, "maxItems": 128,
                            "items": dict(container._ID)},
    })


def _validate_policy(policy, payload, now):
    _require(type(policy) is dict and set(policy) == set(_policy_schema()["properties"]),
             "SIGNING_POLICY_FIELDS_INVALID")
    schema = _policy_schema()
    revoked = policy["revoked_key_ids"]
    _require(type(revoked) is list and len(revoked) <= 128, "REVOCATION_STATE_INVALID")
    for name in revoked:
        container._validate(name, container._ID)
    _require(revoked == sorted(set(revoked)), "REVOCATION_STATE_INVALID")
    # No coercion of Boolean clocks or integer lookalikes.
    _require(type(now) is int and 0 <= now <= container.MAX_INTEGER, "CLOCK_UNAVAILABLE")
    for name, rule in schema["properties"].items():
        if name != "revoked_key_ids":
            container._validate(policy[name], rule)
    _require(policy["issued_at_epoch"] <= policy["not_before_epoch"]
             <= now < policy["expires_at_epoch"], "SIGNING_POLICY_NOT_CURRENT")
    _require(policy["issued_at_epoch"] <= policy["revocation_observed_at_epoch"] <= now
             and now - policy["revocation_observed_at_epoch"] <= policy["max_revocation_age_seconds"],
             "REVOCATION_STALE_OR_FUTURE")
    keys = policy["public_keys"]
    _require([key["id"] for key in keys] == sorted(key["id"] for key in keys),
             "PUBLIC_KEYS_NOT_SORTED")
    fingerprints = set()
    for key in keys:
        raw = bytes.fromhex(key["public_key_hex"])
        fingerprint = hashlib.sha256(raw).hexdigest()
        _require(key["public_key_hex"] != PUBLIC_TEST_KEY, "PUBLIC_TEST_KEY_FORBIDDEN")
        _require(key["public_key_sha256"] == fingerprint
                 and fingerprint not in fingerprints, "PUBLIC_KEY_FINGERPRINT_INVALID_OR_DUPLICATE")
        fingerprints.add(fingerprint)
        _require(key["not_before_epoch"] < key["expires_at_epoch"], "PUBLIC_KEY_TIME_ORDER")
    identity = payload["identity"]
    selected = next((key for key in keys if key["id"] == identity["signer_key_id"]), None)
    _require(selected is not None, "PUBLIC_KEY_PIN_MISSING_OR_INVALID")
    _require(selected["id"] not in revoked, "KEY_REVOKED")
    _require(policy["not_before_epoch"] <= identity["issued_at_epoch"]
             and identity["expires_at_epoch"] <= policy["expires_at_epoch"]
             and selected["not_before_epoch"] <= identity["issued_at_epoch"]
             and identity["expires_at_epoch"] <= selected["expires_at_epoch"],
             "ADMISSION_OUTSIDE_POLICY_OR_KEY_WINDOW")
    _require(identity["not_before_epoch"] <= now < identity["expires_at_epoch"]
             and now - identity["issued_at_epoch"] <= policy["max_admission_age_seconds"],
             "ADMISSION_NOT_CURRENT")
    runner = payload["runner"]
    _require(1 <= runner["max_qualification_age_seconds"] <= 3600
             and 0 <= now - runner["qualified_at_epoch"] <= runner["max_qualification_age_seconds"],
             "RUNNER_QUALIFICATION_STALE_OR_FUTURE")
    return bytes.fromhex(selected["public_key_hex"])


def _signature_observation(payload, signature, public_key):
    """Maintained Ed25519 verifier, fixed native domain; no signer or fallback."""
    try:
        from cryptography import __version__
        from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return None, "NOT_READY", "NOT_RUN", "CRYPTO_BACKEND_UNAVAILABLE"
    if __version__ != CRYPTOGRAPHY_VERSION:
        return None, "NOT_READY", "NOT_RUN", "CRYPTO_BACKEND_VERSION_MISMATCH"
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(signature), DOMAIN + container._canonical(payload))
    except UnsupportedAlgorithm:
        return None, "NOT_READY", "NOT_RUN", "CRYPTO_BACKEND_UNSUPPORTED"
    except InvalidSignature:
        return False, "AVAILABLE", "RUN", "SIGNATURE_MISMATCH"
    except (TypeError, ValueError):
        return False, "AVAILABLE", "RUN", "SIGNATURE_OR_KEY_INVALID"
    return True, "AVAILABLE", "RUN", None


@dataclass(frozen=True)
class NativeAdmissionObservation:
    policy_pin_matched: bool
    key_policy_matched: bool
    admission: container.AdmissionObservation
    reasons: tuple[str, ...]

    def __bool__(self):
        raise TypeError("NATIVE_ADMISSION_OBSERVATION_IS_NOT_AUTHORIZATION")


def observe_native_admission(raw, *, expected_payload, policy_raw,
                             expected_policy_sha256, policy_trust, now_epoch):
    """Observe independently supplied context and pinned policy, never admit.

    `policy_trust` is a caller observation, not evidence generated here. A hash
    match cannot establish the out-of-band procedure behind that observation.
    Malformed or absent policy, context, time or signature data fails closed.
    Nothing from this result may override source eligibility or runtime holds.
    """
    pin, key_match, document, context, trust = False, False, False, None, False
    time_match, revocation, signature = None, False, None
    readiness, verification = "NOT_READY", "NOT_RUN"
    reasons, mismatches = [], []
    try:
        envelope = parse_canonical_envelope(raw)
        document = True
        payload = envelope["payload"]
        expected = parse_canonical_bytes(expected_payload)
        _validate_payload(expected)
        mismatches = container._differences(payload, expected)
        context = not mismatches
        _require(context, "EXPECTED_CONTEXT_MISMATCH")
        container._validate(expected_policy_sha256, container._HASH)
        _require(type(policy_raw) is bytes and len(policy_raw) <= container.MAX_BYTES,
                 "SIGNING_POLICY_BYTES_INVALID")
        _require(hashlib.sha256(policy_raw).hexdigest() == expected_policy_sha256,
                 "SIGNING_POLICY_PIN_MISMATCH")
        pin = True
        _require(payload["identity"]["signing_policy_sha256"] == expected_policy_sha256,
                 "SIGNED_POLICY_BINDING_MISMATCH")
        _require(type(policy_trust) is str and policy_trust == "INDEPENDENTLY_PINNED",
                 "POLICY_PROVENANCE_UNAVAILABLE")
        trust = True
        policy = parse_canonical_bytes(policy_raw)
        public_key = _validate_policy(policy, payload, now_epoch)
        key_match, time_match, revocation = True, True, True
        signature, readiness, verification, reason = _signature_observation(
            payload, envelope["signature"]["value"], public_key)
        if reason is not None:
            reasons.append(reason)
    except AdmissionError as exc:
        reasons.append(exc.code)
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError, OverflowError):
        reasons.append("NATIVE_ADMISSION_INPUT_INVALID")
    observation = container.AdmissionObservation(
        document, context, trust, time_match, revocation, signature,
        tuple(mismatches), tuple(reasons), readiness, verification)
    return NativeAdmissionObservation(pin, key_match, observation, tuple(reasons))
