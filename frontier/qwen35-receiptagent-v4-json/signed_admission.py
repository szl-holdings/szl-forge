"""Bounded, non-authorizing v4 sidecar observations; no cryptographic backend.

This preparation implements byte/profile/context checks only. Ed25519 verification
is NOT_READY and NOT_RUN until a separately admitted dependency profile and
verified positive canonical-domain fixtures exist. Nothing here permits work.
No I/O, model imports, signing, nonce mutation, callbacks or launch interfaces.
"""

from dataclasses import dataclass
from copy import deepcopy
import json
import re


PROFILE = "SZL_RECEIPTAGENT_V4_ASCII_JSON_V1"
PURPOSE = "SUPERVISED_GPU_SMOKE"
CANDIDATE = "SZL-ReceiptAgent-Qwen3.5-0.8B-v4-JSON"
DOMAIN = b"SZL-RECEIPTAGENT-V4-SUPERVISED-GPU-SMOKE-V1\x00"
MAX_BYTES = 65536
MAX_DEPTH = 12
MAX_STRING = 8192
MAX_FIELDS = 128
MAX_ITEMS = 1024
MAX_INTEGER = 9223372036854775807
_MAX_INTEGER_TEXT = str(MAX_INTEGER)


class AdmissionError(ValueError):
    """A stable, value-free profile rejection reason."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class _Parser:
    """Bound before allocation/conversion; this is not general-purpose JSON."""

    def __init__(self, raw):
        if type(raw) is not bytes:
            raise AdmissionError("BYTES_REQUIRED")
        if len(raw) > MAX_BYTES:
            raise AdmissionError("ENVELOPE_TOO_LARGE")
        if raw.startswith(b"\xef\xbb\xbf"):
            raise AdmissionError("BOM_FORBIDDEN")
        try:
            self.text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise AdmissionError("INVALID_UTF8") from exc
        self.i = 0

    def parse(self):
        value = self.value(0)
        if self.i != len(self.text):
            raise AdmissionError("TRAILING_OR_NONCANONICAL_DATA")
        return value

    def take(self, expected):
        if self.i >= len(self.text) or self.text[self.i] != expected:
            raise AdmissionError("JSON_SYNTAX")
        self.i += 1

    def value(self, depth):
        if self.i >= len(self.text):
            raise AdmissionError("JSON_SYNTAX")
        char = self.text[self.i]
        if char in "{[":
            if depth >= MAX_DEPTH:
                raise AdmissionError("DEPTH_LIMIT")
            return self.object(depth + 1) if char == "{" else self.array(depth + 1)
        if char == '"':
            return self.string()
        if "0" <= char <= "9":
            return self.integer()
        for spelling, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(spelling, self.i):
                self.i += len(spelling)
                return value
        raise AdmissionError("JSON_SYNTAX_OR_NUMBER_PROFILE")

    def object(self, depth):
        self.take("{")
        result = {}
        if self.i < len(self.text) and self.text[self.i] == "}":
            self.i += 1
            return result
        while True:
            if len(result) >= MAX_FIELDS:
                raise AdmissionError("OBJECT_FIELD_LIMIT")
            key = self.string()
            if key in result:
                raise AdmissionError("DUPLICATE_KEY")
            self.take(":")
            result[key] = self.value(depth)
            if self.i < len(self.text) and self.text[self.i] == "}":
                self.i += 1
                return result
            self.take(",")

    def array(self, depth):
        self.take("[")
        result = []
        if self.i < len(self.text) and self.text[self.i] == "]":
            self.i += 1
            return result
        while True:
            if len(result) >= MAX_ITEMS:
                raise AdmissionError("ARRAY_ITEM_LIMIT")
            result.append(self.value(depth))
            if self.i < len(self.text) and self.text[self.i] == "]":
                self.i += 1
                return result
            self.take(",")

    def string(self):
        self.take('"')
        result = []
        while self.i < len(self.text):
            char = self.text[self.i]
            self.i += 1
            if char == '"':
                return "".join(result)
            if char == "\\":
                if self.i >= len(self.text):
                    raise AdmissionError("JSON_SYNTAX")
                escape = self.text[self.i]
                self.i += 1
                if escape in ('"', "\\", "/"):
                    char = escape
                elif escape == "u":
                    digits = self.text[self.i:self.i + 4]
                    if len(digits) != 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                        raise AdmissionError("JSON_SYNTAX")
                    char = chr(int(digits, 16))
                    self.i += 4
                else:
                    raise AdmissionError("STRING_PROFILE")
            if not 32 <= ord(char) <= 126:
                raise AdmissionError("STRING_PROFILE")
            if len(result) >= MAX_STRING:
                raise AdmissionError("STRING_LIMIT")
            result.append(char)
        raise AdmissionError("JSON_SYNTAX")

    def integer(self):
        start = self.i
        while self.i < len(self.text) and "0" <= self.text[self.i] <= "9":
            if self.i - start >= 19:
                raise AdmissionError("INTEGER_LIMIT")
            self.i += 1
        token = self.text[start:self.i]
        if len(token) > 1 and token[0] == "0":
            raise AdmissionError("INTEGER_SPELLING")
        if len(token) == 19 and token > _MAX_INTEGER_TEXT:
            raise AdmissionError("INTEGER_LIMIT")
        if self.i < len(self.text) and self.text[self.i] in ".eE+-":
            raise AdmissionError("INTEGER_PROFILE")
        return int(token)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def parse_canonical_bytes(raw):
    """Bounded JSON byte observation; not closed sidecar validation or authority."""
    value = _Parser(raw).parse()
    if _canonical(value) != raw:
        raise AdmissionError("NONCANONICAL_BYTES")
    return value


def _string(pattern=r"[ -~]+", max_length=MAX_STRING):
    return {"type": "string", "minLength": 1, "maxLength": max_length,
            "pattern": "^(?:" + pattern + ")$"}


def _constant(value):
    kind = "boolean" if type(value) is bool else "integer" if type(value) is int else "string"
    return {"type": kind, "const": value}


def _integer(minimum=1, maximum=MAX_INTEGER):
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


_HASH = _string(r"[0-9a-f]{64}", 64)
_GIT = _string(r"[0-9a-f]{40}", 40)
_ID = _string(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", 128)
_CLOSURE = {"type": "array", "minItems": 1, "maxItems": 128,
            "items": _object({"id": _ID, "sha256": _HASH})}


def _hash_fields(*names):
    return {name: dict(_HASH) for name in names}


def payload_schema():
    """Fresh closed profile description; no schema resolver or external I/O."""
    return deepcopy(_object({
        "identity": _object({
            "schema_version": _constant("szl.receiptagent.v4-training-admission/v1"),
            "profile": _constant(PROFILE), "purpose": _constant(PURPOSE),
            "candidate_id": _constant(CANDIDATE), "admission_id": dict(_ID),
            "attempt_nonce": dict(_HASH), "signer_key_id": dict(_ID),
            "issued_at_epoch": _integer(0), "not_before_epoch": _integer(0),
            "expires_at_epoch": _integer(0), "one_use": _constant(True),
        }),
        "source": _object({
            "repository": _constant("szl-holdings/szl-forge"),
            "revision": dict(_GIT), "tree": dict(_GIT), "code_closure": _CLOSURE,
            **_hash_fields("candidate_sha256", "publishing_binding_sha256", "supervisor_policy_sha256"),
        }),
        "curriculum": _object({
            "train_sha256": _constant("26c3b11c64d0d6201e29716656c1898893fb449249f6a67611a68f477a64699f"),
            "manifest_sha256": _constant("7ec31cfddf31228164a7a67e7edd8152c611c2f04a4860d2ad85c168339337ae"),
            "dev_sha256": _constant("24914bdf060efd4900d9fbbc7058982a8b46ef71977f6d9cd030775e4b6d4608"),
            "held_out_sha256": _constant("973e5f5a272ad7c1304acec0db36fae77a6d0c2e0ec0c10c4401361b7889c575"),
            "train_rows": _constant(360), "seed": _constant(2601003),
            "split_access": _constant("PUBLIC_FROZEN_NOT_BLIND"),
            "disjointness": _constant("DECLARED_TEXTUAL_NOT_SEMANTIC_PROOF"),
            "evidence_class": _constant("SIMULATED"),
        }),
        "gates": _object({
            **_hash_fields("contract_sha256", "preregistration_sha256", "gate_sha256", "harness_sha256",
                           "generic_sft_validator_sha256", "nemo_wrapper_sha256"),
            "schema_closure": _CLOSURE, "nemo_repository": _constant("szl-holdings/szl-nemo"),
            "nemo_revision": _constant("f7b33e1b1fe8f3b9b729a27567bcba375e2e0d6e"),
            "nemo_module_closure": {**_CLOSURE, "minItems": 6, "maxItems": 6},
        }),
        "base": _object({
            "repository": _constant("unsloth/Qwen3.5-0.8B"),
            "revision": _constant("23c69c53358a07516b5827588b3fdb12ae78fd65"),
            "assets_closure": _CLOSURE, "loader_class": dict(_ID), "architecture": dict(_ID),
        }),
        "runtime": _object(_hash_fields("quantization_recipe_sha256", "peft_recipe_sha256",
                                         "dependency_lock_sha256", "wheel_lock_sha256",
                                         "container_sha256", "worker_closure_sha256")),
        "tokens": _object({
            "assets_closure": _CLOSURE,
            **_hash_fields("template_sha256", "special_tokens_sha256", "mask_policy_sha256",
                           "eos_policy_sha256", "thinking_policy_sha256", "cpu_qualification_sha256",
                           "tensor_ledger_sha256"), "complete_rows": _constant(360),
            "truncation": _constant(False), "packing": _constant(False), "drop_last": _constant(False),
        }),
        "runner": _object({
            "backend": {"type": "string", "enum": ["LOCAL_ISOLATED", "HF_JOBS"]},
            "runner_id": dict(_ID), "device_id": dict(_ID),
            **_hash_fields("device_qualification_sha256", "telemetry_qualification_sha256",
                           "containment_policy_sha256", "containment_qualification_sha256"),
            "qualified_at_epoch": _integer(0), "max_qualification_age_seconds": _integer(),
        }),
        "bounds": _object({
            "optimizer_updates": _constant(1), "gradient_coverage": _constant("FULL_360_ONCE"),
            "row_coverage": _constant(360), "row_order_sha256": dict(_HASH),
            "microbatch_rows": _integer(1, 360), "accumulation_batches": _integer(1, 360),
            **{name: _integer() for name in (
                "sequence_tokens", "total_tokens", "wall_seconds", "kill_seconds", "drain_seconds",
                "gpu_memory_bytes", "host_memory_bytes", "temperature_limit_millicelsius",
                "telemetry_max_gap_seconds", "disk_reserve_bytes", "evidence_reserve_bytes",
                "log_cap_bytes", "output_cap_bytes", "artifact_cap_bytes")},
            "currency": _constant("USD"), "cost_cap_minor_units": _integer(0),
        }),
        "closure": _object(_hash_fields("evidence_format_sha256", "chain_policy_sha256",
                                         "durable_output_policy_sha256", "process_drain_policy_sha256",
                                         "terminal_contract_sha256", "anti_replay_namespace_sha256")),
    }))


def profile_schema():
    """Fresh schema artifact description. Byte/relational checks are additional."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ReceiptAgent v4 non-authorizing sidecar profile",
        "$comment": "Signature verification NOT_READY/NOT_RUN; no launch or eligibility override. "
                    "Bounded canonical-byte and relational checks are required in addition to this schema.",
        **_object({"profile": _constant(PROFILE), "payload": payload_schema(),
                   "signature": _object({"algorithm": _constant("Ed25519"),
                                          "key_id": dict(_ID), "value": _string(r"[0-9a-f]{128}", 128)})}),
    }


def _validate(value, rule):
    kind = rule["type"]
    exact_types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}
    if type(value) is not exact_types[kind]:
        raise AdmissionError("SCHEMA_TYPE")
    if "const" in rule and value != rule["const"]:
        raise AdmissionError("SCHEMA_CONSTANT")
    if "enum" in rule and value not in rule["enum"]:
        raise AdmissionError("SCHEMA_ENUM")
    if kind == "object":
        if set(value) != set(rule["properties"]):
            raise AdmissionError("SCHEMA_FIELDS")
        for key, child_rule in rule["properties"].items():
            _validate(value[key], child_rule)
    elif kind == "array":
        if not rule["minItems"] <= len(value) <= rule["maxItems"]:
            raise AdmissionError("SCHEMA_ARRAY_LIMIT")
        ids = set()
        for item in value:
            _validate(item, rule["items"])
            if item["id"] in ids:
                raise AdmissionError("DUPLICATE_CLOSURE_ID")
            ids.add(item["id"])
    elif kind == "string":
        if "minLength" in rule and not rule["minLength"] <= len(value) <= rule["maxLength"]:
            raise AdmissionError("SCHEMA_STRING_LIMIT")
        if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
            raise AdmissionError("SCHEMA_STRING_PATTERN")
    elif kind == "integer" and "minimum" in rule:
        if not rule["minimum"] <= value <= rule["maximum"]:
            raise AdmissionError("SCHEMA_INTEGER_BOUND")


def _validate_payload(payload):
    _validate(payload, payload_schema())
    identity = payload["identity"]
    if not identity["issued_at_epoch"] <= identity["not_before_epoch"] < identity["expires_at_epoch"]:
        raise AdmissionError("TIME_ORDER")
    bounds = payload["bounds"]
    if bounds["microbatch_rows"] * bounds["accumulation_batches"] != 360:
        raise AdmissionError("GRADIENT_COVERAGE_SHAPE")
    if bounds["total_tokens"] < bounds["sequence_tokens"]:
        raise AdmissionError("TOKEN_BOUND_ORDER")


def parse_canonical_envelope(raw):
    """Return fresh structural data only; not authenticated or authorized data."""
    envelope = parse_canonical_bytes(raw)
    _validate(envelope, profile_schema())
    _validate_payload(envelope["payload"])
    if envelope["signature"]["key_id"] != envelope["payload"]["identity"]["signer_key_id"]:
        raise AdmissionError("SIGNER_KEY_ID_MISMATCH")
    return envelope


@dataclass(frozen=True)
class AdmissionObservation:
    document_matches_profile: bool
    context_matches: bool | None
    public_trust_observed: bool
    time_window_matches: bool | None
    revocation_observed: bool
    signature_matches: None
    mismatch_fields: tuple[str, ...]
    reasons: tuple[str, ...]
    cryptographic_readiness: str = "NOT_READY"
    cryptographic_verification: str = "NOT_RUN"

    def __bool__(self):
        raise TypeError("ADMISSION_OBSERVATION_IS_NOT_AUTHORIZATION")


def _differences(left, right, prefix="payload"):
    result = []
    for key in sorted(left):
        path = prefix + "." + key
        if type(left[key]) is dict:
            result.extend(_differences(left[key], right[key], path))
        elif left[key] != right[key]:
            result.append(path)
    return result


def observe_admission(raw, *, expected_payload, public_keys, key_trust,
                      now_epoch, revoked_key_ids, max_age_seconds):
    """Observe caller-owned context/trust/time; always NOT_READY for cryptography.

    expected_payload is complete canonical payload bytes, not a partial mapping.
    Public key mapping values must be exact bytes(32). No sidecar key is trusted.
    Explicit clock/revocation observations are not authenticated by this module.
    The caller is responsible for their independently pinned provenance.
    """
    reasons = []
    mismatches = []
    context = None
    trust = False
    time_match = None
    revocation = False
    try:
        envelope = parse_canonical_envelope(raw)
    except AdmissionError as exc:
        return AdmissionObservation(False, None, False, None, False, None, (),
                                    (exc.code, "CRYPTO_BACKEND_NOT_QUALIFIED"))
    payload = envelope["payload"]
    try:
        expected = parse_canonical_bytes(expected_payload)
        _validate_payload(expected)
        mismatches = _differences(payload, expected)
        context = not mismatches
        if not context:
            reasons.append("EXPECTED_CONTEXT_MISMATCH")
    except AdmissionError:
        reasons.append("EXPECTED_CONTEXT_INVALID")
    key_id = payload["identity"]["signer_key_id"]
    keys_valid = type(public_keys) is dict and len(public_keys) <= MAX_FIELDS
    if keys_valid:
        keys_valid = all(type(name) is str and re.fullmatch(_ID["pattern"], name)
                         and type(key) is bytes and len(key) == 32
                         for name, key in public_keys.items())
    if not keys_valid or key_id not in public_keys:
        reasons.append("PUBLIC_KEY_PIN_MISSING_OR_INVALID")
    elif type(key_trust) is not str or key_trust != "INDEPENDENTLY_PINNED":
        reasons.append("PUBLIC_TRUST_NOT_INDEPENDENTLY_PINNED")
    else:
        trust = True
    if (type(revoked_key_ids) is not frozenset or len(revoked_key_ids) > MAX_FIELDS
            or not all(type(name) is str and re.fullmatch(_ID["pattern"], name)
                       for name in revoked_key_ids)):
        reasons.append("REVOCATION_STATE_UNAVAILABLE")
    elif key_id in revoked_key_ids:
        reasons.append("KEY_REVOKED")
    else:
        revocation = True
    if (type(now_epoch) is not int or not 0 <= now_epoch <= MAX_INTEGER
            or type(max_age_seconds) is not int or not 1 <= max_age_seconds <= MAX_INTEGER):
        reasons.append("CLOCK_OR_MAX_AGE_UNAVAILABLE")
    else:
        identity = payload["identity"]
        time_match = True
        if now_epoch < identity["not_before_epoch"]:
            time_match = False
            reasons.append("NOT_YET_VALID")
        if now_epoch >= identity["expires_at_epoch"]:
            time_match = False
            reasons.append("EXPIRED")
        if now_epoch - identity["issued_at_epoch"] > max_age_seconds:
            time_match = False
            reasons.append("ADMISSION_TOO_OLD")
        runner = payload["runner"]
        if (runner["qualified_at_epoch"] > now_epoch
                or now_epoch - runner["qualified_at_epoch"] > runner["max_qualification_age_seconds"]):
            time_match = False
            reasons.append("RUNNER_QUALIFICATION_STALE_OR_FUTURE")
    reasons.extend(("CRYPTO_BACKEND_NOT_QUALIFIED", "POSITIVE_CANONICAL_SIGNATURE_FIXTURE_NOT_VERIFIED"))
    return AdmissionObservation(True, context, trust, time_match, revocation, None,
                                tuple(mismatches), tuple(reasons))
