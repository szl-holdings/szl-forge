#!/usr/bin/env python3
"""Stdlib governed-receipt adapter for OAC System Health v2 (workstream E, ours).

Emits receipts shaped to szl-holdings/governed-receipt-spec @ 2c82320a
(schema/governed-receipt.schema.json + verify.py).  Standard library only, so
it can ship next to the dependency-free v2 kernel.

WHAT A RECEIPT ATTESTS: integrity and origin only -- "this payload, with this
SHA-256, was recorded by this tool at this position of this hash chain".  It is
NOT a quality, correctness, calibration or safety claim, NOT a signature (see
below), NOT zero-knowledge and NOT a proof of computation.

Spec facts mirrored (REPORTED from governed-receipt-spec @ 2c82320a):
  * Required decision fields: action, ns, seq, prev, digest, payload_digest, ts.
  * seq >= 0; genesis seq 0 has prev = 64 zero hex chars; prev == previous
    receipt's digest; seq increments by exactly 1 (verify.py::check_chain).
  * prev / digest / payload_digest: 64 lowercase hex (SHA-256).
  * ts: number (epoch seconds) or ISO-8601 date-time string.
  * decision (optional) in {allow, deny, block, review, abstain}.
  * Optional DSSE envelope under `dsse` (or `envelope`): payloadType, base64
    payload, signatures[], signed flag, `_pae_sha256` = sha256(PAE) where
    PAE = b"DSSEv1 " + len(type) + b" " + type + b" " + len(body) + b" " + body,
    lengths in ASCII decimal over the DECODED payload bytes.
  * Clear fields beside a nested envelope must be byte-identical (canonical
    JSON) to the sealed payload fields (verify.py::check_clear_claim_binding),
    and the envelope may only carry the spec's known extension keys.
  * The spec verifier does NOT re-derive `digest`; that serialization is
    internal to the emitter.

Our choices (documented, not spec-mandated):
  * canonical JSON = json.dumps(obj, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False, allow_nan=False).encode("utf-8").  NaN/Inf are
    rejected (fail closed) because they are not JSON.
  * payload_digest = sha256(payload) for bytes; sha256(utf-8) for str;
    sha256(canonical JSON) for any other JSON value.  The payload itself is
    never embedded in the receipt.
  * digest = sha256(canonical JSON of the decision body with the keys
    digest / dsse / envelope / signature removed).
  * Envelope is UNSIGNED: signed=false, signatures=[].  The stdlib has no
    ECDSA and the spec forbids hand-rolled crypto, so we sign nothing and
    say so.  An unsigned receipt is tamper-EVIDENT only against accidental or
    naive edits (anyone can recompute the hashes); origin is asserted, not
    cryptographically proven.
  * payloadType = application/vnd.oac-frontier.receipt+json (ours).

Usage:
    python -B math/receipt_adapter.py --self-test
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "ZERO_HASH", "PAYLOAD_TYPE", "SCHEMA_ID", "ADAPTER_VERSION", "HONESTY",
    "canonical_json", "sha256_hex", "payload_digest", "dsse_pae",
    "make_receipt", "receipt_digest", "load_spec_schema", "validate_schema",
    "verify_receipt", "verify_chain", "write_receipt", "read_receipt",
]

ADAPTER_VERSION = "1.0.0"
ZERO_HASH = "0" * 64
PAYLOAD_TYPE = "application/vnd.oac-frontier.receipt+json"
SCHEMA_ID = "oac-frontier.governed-receipt/v1"
SPEC_REF = "szl-holdings/governed-receipt-spec@2c82320a"
HONESTY = ("Integrity and origin only: binds a payload SHA-256 to a position in a "
           "hash chain emitted by oac-frontier/math/receipt_adapter.py. UNSIGNED "
           "(no signature, origin asserted not proven). Not a claim about quality, "
           "correctness, calibration or safety.")
DECISIONS = ("allow", "deny", "block", "review", "abstain")
REQUIRED = ("action", "ns", "seq", "prev", "digest", "payload_digest", "ts")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
# keys excluded from the digest pre-image
_DIGEST_EXCLUDED = ("digest", "dsse", "envelope", "signature")
# keys make_receipt owns; `extra` may not override them
_RESERVED = set(REQUIRED) | {"dsse", "envelope", "signature", "decision", "organ",
                             "schema", "origin", "honesty", "payload"}
# envelope keys the spec verifier accepts (verify.py::check_clear_claim_binding)
_ENVELOPE_KEYS = {"_dsse", "_pae_sha256", "_signed_at", "honesty", "payload",
                  "payloadSha256", "payloadType", "signatures", "signed", "signing",
                  "verify_key_url"}

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC_SCHEMA = os.path.normpath(os.path.join(
    _HERE, "..", "src", "governed-receipt-spec", "schema", "governed-receipt.schema.json"))

# Minimal embedded fallback (subset of the spec schema, REPORTED @ 2c82320a), used
# only when the spec checkout is absent (e.g. v2 shipped standalone).
_EMBEDDED_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": list(REQUIRED),
    "additionalProperties": True,
    "properties": {
        "action": {"type": "string"},
        "ns": {"type": "string"},
        "organ": {"type": "string"},
        "seq": {"type": "integer", "minimum": 0},
        "prev": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "payload_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "ts": {"oneOf": [{"type": "number"}, {"type": "string", "format": "date-time"}]},
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "signature": {"type": "string"},
        "schema": {"type": "string"},
        "dsse": {"$ref": "#/$defs/dsseEnvelope"},
        "envelope": {"$ref": "#/$defs/dsseEnvelope"},
    },
    "$defs": {"dsseEnvelope": {
        "type": "object", "required": ["payloadType", "payload", "signatures"],
        "additionalProperties": True,
        "properties": {
            "payloadType": {"type": "string"}, "payload": {"type": "string"},
            "signatures": {"type": "array", "items": {"type": "object"}},
            "signed": {"type": "boolean"},
            "_pae_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        }}},
}


# --------------------------------------------------------------------------- #
# hashing / canonicalization                                                  #
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> bytes:
    """Canonical JSON bytes (sorted keys, compact, UTF-8, NaN/Inf rejected)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_digest(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return sha256_hex(bytes(payload))
    if isinstance(payload, str):
        return sha256_hex(payload.encode("utf-8"))
    return sha256_hex(canonical_json(payload))


def dsse_pae(payload_type: str, body: bytes) -> bytes:
    """DSSE v1 Pre-Authentication Encoding over the decoded payload bytes."""
    if not isinstance(payload_type, str) or not payload_type:
        raise ValueError("payload_type must be a non-empty string")
    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    t = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(t)).encode("ascii") + b" " + t + b" " + \
        str(len(body)).encode("ascii") + b" " + body


def receipt_digest(body: Dict[str, Any]) -> str:
    pre = {k: v for k, v in body.items() if k not in _DIGEST_EXCLUDED}
    return sha256_hex(canonical_json(pre))


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _is_iso_datetime(s: str) -> bool:
    if not isinstance(s, str) or "T" not in s:
        return False
    try:
        _dt.datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# construction                                                                #
# --------------------------------------------------------------------------- #
def make_receipt(payload: Any, *, action: str = "inference", ns: str = "oac-frontier",
                 prev_receipt: Optional[Dict[str, Any]] = None,
                 seq: Optional[int] = None, prev: Optional[str] = None,
                 ts: Any = None, decision: Optional[str] = None,
                 organ: Optional[str] = None, origin: Optional[Dict[str, Any]] = None,
                 extra: Optional[Dict[str, Any]] = None, envelope: bool = True,
                 payload_type: str = PAYLOAD_TYPE) -> Dict[str, Any]:
    """Build one governed receipt (a dict) for ``payload``.

    Chain position: pass ``prev_receipt`` (seq/prev derived from it), or
    ``seq``/``prev`` explicitly, or neither (genesis: seq 0, prev = 64 zeros).
    ``ts`` defaults to the current UTC time (ISO-8601 'Z'); pass a fixed value
    for byte-identical output.  ``extra`` adds non-reserved extension fields.
    Raises ValueError/TypeError on anything that would not conform (fail closed).
    """
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string")
    if not isinstance(ns, str) or not ns:
        raise ValueError("ns must be a non-empty string")
    if prev_receipt is not None:
        if seq is not None or prev is not None:
            raise ValueError("pass prev_receipt OR seq/prev, not both")
        p_seq, p_dig = prev_receipt.get("seq"), prev_receipt.get("digest")
        if not isinstance(p_seq, int) or isinstance(p_seq, bool) or not (
                isinstance(p_dig, str) and _HEX64.match(p_dig)):
            raise ValueError("prev_receipt lacks a valid seq/digest")
        seq, prev = p_seq + 1, p_dig
    if seq is None:
        seq = 0
    if prev is None:
        prev = ZERO_HASH
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ValueError("seq must be an int >= 0")
    if not (isinstance(prev, str) and _HEX64.match(prev)):
        raise ValueError("prev must be 64 lowercase hex chars")
    if seq == 0 and prev != ZERO_HASH:
        raise ValueError("genesis receipt (seq 0) must have prev = 64 zeros")
    if ts is None:
        ts = _now_iso()
    if isinstance(ts, bool) or not (isinstance(ts, (int, float)) or _is_iso_datetime(ts)):
        raise ValueError("ts must be epoch seconds or an ISO-8601 date-time string")
    if isinstance(ts, float) and (ts != ts or ts in (float("inf"), float("-inf"))):
        raise ValueError("ts must be finite")
    if decision is not None and decision not in DECISIONS:
        raise ValueError("decision must be one of %s" % (DECISIONS,))

    body: Dict[str, Any] = {
        "action": action, "ns": ns, "seq": seq, "prev": prev,
        "payload_digest": payload_digest(payload), "ts": ts,
        "schema": SCHEMA_ID, "honesty": HONESTY,
        "origin": {"tool": "oac-frontier/math/receipt_adapter.py",
                   "adapter_version": ADAPTER_VERSION, "spec": SPEC_REF},
    }
    if origin:
        if not isinstance(origin, dict):
            raise TypeError("origin must be a dict")
        body["origin"].update(copy.deepcopy(origin))
    if decision is not None:
        body["decision"] = decision
    if organ is not None:
        body["organ"] = organ
    if extra:
        bad = sorted(set(extra) & _RESERVED)
        if bad:
            raise ValueError("extra may not set reserved keys: %s" % bad)
        lam = extra.get("lambda")
        if lam is not None:
            label = str(lam.get("label", "")) if isinstance(lam, dict) else ""
            if "Conjecture" not in label or re.search(r"\bproven\b", label, re.I):
                raise ValueError("lambda.label must carry the honest status "
                                 "('Λ = Conjecture 1 — never green'), never 'proven'")
        body.update(copy.deepcopy(extra))
    canonical_json(body)  # fail closed on NaN/Inf/non-JSON before hashing
    body["digest"] = receipt_digest(body)

    if envelope:
        sealed = canonical_json(body)
        body = dict(body)
        body["dsse"] = {
            "_dsse": "DSSEv1",
            "payloadType": payload_type,
            "payload": base64.b64encode(sealed).decode("ascii"),
            "signatures": [],
            "signed": False,
            "_pae_sha256": sha256_hex(dsse_pae(payload_type, sealed)),
            "honesty": "UNSIGNED - content-bound by DSSE PAE sha256 only; no "
                       "signature, so origin is asserted, not cryptographically proven.",
        }
    return body


# --------------------------------------------------------------------------- #
# stdlib JSON-Schema subset validator (independent implementation)             #
# keywords: type, required, properties, additionalProperties, enum, const,    #
# pattern, minimum, items, oneOf, anyOf, $ref(#/...), format(date-time).       #
# --------------------------------------------------------------------------- #
def _type_ok(v: Any, t: str) -> bool:
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def _resolve(ref: str, root: Dict[str, Any]) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError("only local $ref supported: %s" % ref)
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def validate_schema(value: Any, schema: Dict[str, Any], root: Optional[Dict[str, Any]] = None,
                    path: str = "$") -> List[str]:
    root = schema if root is None else root
    errs: List[str] = []
    if "$ref" in schema:
        errs += validate_schema(value, _resolve(schema["$ref"], root), root, path)
    t = schema.get("type")
    if t is not None:
        ts_ = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in ts_):
            return errs + ["%s: expected type %s, got %s" % (path, t, type(value).__name__)]
    if "enum" in schema and value not in schema["enum"]:
        errs.append("%s: %r not in enum %r" % (path, value, schema["enum"]))
    if "const" in schema and value != schema["const"]:
        errs.append("%s: %r != const %r" % (path, value, schema["const"]))
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append("%s: %r does not match %s" % (path, value, schema["pattern"]))
        if schema.get("format") == "date-time" and not _is_iso_datetime(value):
            errs.append("%s: %r is not an ISO-8601 date-time" % (path, value))
    if _type_ok(value, "number") and "minimum" in schema and value < schema["minimum"]:
        errs.append("%s: %r < minimum %r" % (path, value, schema["minimum"]))
    for kw in ("oneOf", "anyOf"):
        if kw in schema:
            n = sum(1 for sub in schema[kw] if not validate_schema(value, sub, root, path))
            if (kw == "oneOf" and n != 1) or (kw == "anyOf" and n < 1):
                errs.append("%s: %d of %d %s branches matched" % (path, n, len(schema[kw]), kw))
    if isinstance(value, dict):
        for r in schema.get("required", []):
            if r not in value:
                errs.append("%s: missing required field %r" % (path, r))
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs += validate_schema(v, props[k], root, "%s.%s" % (path, k))
            elif addl is False:
                errs.append("%s: additional property %r not allowed" % (path, k))
            elif isinstance(addl, dict):
                errs += validate_schema(v, addl, root, "%s.%s" % (path, k))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errs += validate_schema(item, schema["items"], root, "%s[%d]" % (path, i))
    return errs


def load_spec_schema(path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Return (schema, source). Prefers the spec checkout; falls back to the embedded subset."""
    cand = path or os.environ.get("OAC_RECEIPT_SCHEMA") or DEFAULT_SPEC_SCHEMA
    if os.path.isfile(cand):
        with open(cand, "r", encoding="utf-8") as fh:
            return json.load(fh), "spec-file:" + cand
    return copy.deepcopy(_EMBEDDED_SCHEMA), "embedded-subset (spec file not found: %s)" % cand


# --------------------------------------------------------------------------- #
# verification (stdlib; mirrors the spec verifier's checks + our digest)       #
# --------------------------------------------------------------------------- #
def _envelope_of(receipt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in ("dsse", "envelope"):
        env = receipt.get(k)
        if isinstance(env, dict):
            return env
    return None


def verify_receipt(receipt: Dict[str, Any], payload: Any = None, *,
                   schema: Optional[Dict[str, Any]] = None,
                   check_payload: bool = False) -> List[str]:
    """Return a list of problems (empty list = conforms). Never raises on bad input."""
    errs: List[str] = []
    if not isinstance(receipt, dict):
        return ["receipt is not an object"]
    if schema is None:
        schema, _ = load_spec_schema()
    body = {k: v for k, v in receipt.items() if k not in ("dsse", "envelope")}
    errs += ["schema " + e for e in validate_schema(receipt, schema)]
    try:
        if receipt.get("digest") != receipt_digest(body):
            errs.append("digest mismatch (recomputed %s)" % receipt_digest(body)[:16])
    except (TypeError, ValueError) as exc:
        errs.append("digest not computable: %s" % exc)
    if receipt.get("seq") == 0 and receipt.get("prev") != ZERO_HASH:
        errs.append("genesis prev must be 64 zeros")
    if check_payload:
        try:
            if payload_digest(payload) != receipt.get("payload_digest"):
                errs.append("payload_digest does not match supplied payload")
        except (TypeError, ValueError) as exc:
            errs.append("payload not hashable: %s" % exc)
    env = _envelope_of(receipt)
    if env is not None:
        unknown = sorted(set(env) - _ENVELOPE_KEYS)
        if unknown:
            errs.append("envelope has unbound extension keys %s" % unknown)
        sigs = env.get("signatures")
        if not isinstance(sigs, list):
            errs.append("envelope.signatures not a list")
        elif env.get("signed") is True and not sigs:
            errs.append("signed=true with no signatures")
        elif env.get("signed") is False and sigs:
            errs.append("signed=false with signatures present")
        try:
            sealed = base64.b64decode(env.get("payload", ""), validate=True)
        except Exception as exc:  # noqa: BLE001
            errs.append("envelope payload not base64: %s" % exc)
            sealed = None
        if sealed is not None:
            if "_pae_sha256" in env and sha256_hex(
                    dsse_pae(str(env.get("payloadType") or "?"), sealed)) != env["_pae_sha256"]:
                errs.append("DSSE PAE sha256 mismatch")
            try:
                sealed_obj = json.loads(sealed.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                errs.append("sealed payload not JSON: %s" % exc)
                sealed_obj = None
            if isinstance(sealed_obj, dict):
                for k, v in body.items():
                    if k not in sealed_obj or canonical_json(sealed_obj[k]) != canonical_json(v):
                        errs.append("clear field %r not bound to sealed payload" % k)
                for k in sealed_obj:
                    if k not in body:
                        errs.append("sealed field %r missing from clear body" % k)
    return errs


def verify_chain(receipts: Iterable[Dict[str, Any]]) -> List[str]:
    rs = sorted(list(receipts), key=lambda r: r.get("seq", -1))
    errs: List[str] = []
    if not rs:
        return ["empty chain"]
    if rs[0].get("seq") == 0 and rs[0].get("prev") != ZERO_HASH:
        errs.append("genesis prev must be 64 zeros")
    for a, b in zip(rs, rs[1:]):
        if b.get("prev") != a.get("digest"):
            errs.append("seq %s prev != seq %s digest" % (b.get("seq"), a.get("seq")))
        if b.get("seq") != a.get("seq", -2) + 1:
            errs.append("seq not contiguous: %s after %s" % (b.get("seq"), a.get("seq")))
    return errs


def write_receipt(receipt: Dict[str, Any], directory: str, name: Optional[str] = None) -> str:
    """Write one receipt as UTF-8 JSON (sorted keys, indent 2). Returns the path."""
    os.makedirs(directory, exist_ok=True)
    name = name or "%06d-%s.json" % (receipt["seq"], receipt["digest"][:12])
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False,
                            allow_nan=False))
        fh.write("\n")
    return path


def read_receipt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# self-test                                                                   #
# --------------------------------------------------------------------------- #
def _self_test(out_dir: Optional[str] = None) -> int:
    import tempfile
    schema, source = load_spec_schema()
    print("receipt_adapter %s self-test | python %s" % (ADAPTER_VERSION, sys.version.split()[0]))
    print("schema source: %s" % source)
    print("validation mode: stdlib structural check against the spec JSON schema "
          "(spec verify.py NOT used here: its pinned dep in-toto-attestation is not installed)")
    print("payloads: SYNTHETIC dummy dicts; results describe format conformance only")
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print("  %-4s %s %s" % ("PASS" if ok else "FAIL", name, detail))

    fixed_ts = "2026-09-26T00:00:00Z"
    payloads = [{"trial": i, "params": {"lr": 0.1 * (i + 1), "depth": i + 2}} for i in range(5)]
    chain: List[Dict[str, Any]] = []
    for p in payloads:
        chain.append(make_receipt(p, action="hyperparameter-trial", ns="oac-frontier",
                                  prev_receipt=chain[-1] if chain else None, ts=fixed_ts))
    for r, p in zip(chain, payloads):
        e = verify_receipt(r, p, schema=schema, check_payload=True)
        check("valid receipt seq=%d" % r["seq"], not e, "; ".join(e))
    check("chain intact (5)", not verify_chain(chain), "; ".join(verify_chain(chain)))
    check("genesis prev zeros", chain[0]["prev"] == ZERO_HASH)
    # determinism
    again = []
    for p in payloads:
        again.append(make_receipt(p, action="hyperparameter-trial", ns="oac-frontier",
                                  prev_receipt=again[-1] if again else None, ts=fixed_ts))
    check("deterministic (fixed ts -> byte-identical)",
          [canonical_json(a) for a in again] == [canonical_json(c) for c in chain])
    # key-order independence of payload_digest
    check("payload_digest key-order independent",
          payload_digest({"a": 1, "b": 2}) == payload_digest({"b": 2, "a": 1}))
    # DSSE PAE worked vector (hand-computed structure, not from memory):
    check("PAE layout", dsse_pae("t/x", b"hello") == b"DSSEv1 3 t/x 5 hello")
    # epoch-float ts + decision + no envelope
    r2 = make_receipt(b"raw-bytes", ts=1790000000.5, decision="abstain", envelope=False)
    check("epoch ts, decision=abstain, no envelope", not verify_receipt(r2, schema=schema))
    # tamper detection (each must produce >= 1 problem)
    base = chain[2]

    def tampered(fn) -> Dict[str, Any]:
        t = copy.deepcopy(base)
        fn(t)
        return t

    cases = {
        "tamper payload (wrong payload supplied)":
            (base, {"trial": 99}, True),
        "tamper clear field ns (digest+binding)":
            (tampered(lambda t: t.__setitem__("ns", "other")), None, False),
        "tamper sealed payload (PAE)":
            (tampered(lambda t: t["dsse"].__setitem__(
                "payload", base64.b64encode(b'{"x":1}').decode())), None, False),
        "missing required field ts":
            (tampered(lambda t: t.pop("ts")), None, False),
        "bad prev pattern (uppercase)":
            (tampered(lambda t: t.__setitem__("prev", t["prev"].upper())), None, False),
        "decision outside enum":
            (tampered(lambda t: t.__setitem__("decision", "approve")), None, False),
        "unknown envelope key":
            (tampered(lambda t: t["dsse"].__setitem__("quality", "high")), None, False),
        "signed=true with no signatures":
            (tampered(lambda t: t["dsse"].__setitem__("signed", True)), None, False),
    }
    for name, (rec, pl, chk) in cases.items():
        e = verify_receipt(rec, pl, schema=schema, check_payload=chk)
        check("detects: " + name, bool(e), "(%d problem(s): %s)" % (len(e), e[0] if e else "-"))
    broken = copy.deepcopy(chain)
    broken[3]["prev"] = ZERO_HASH
    check("detects: broken chain link", bool(verify_chain(broken)))
    gap = [chain[0], chain[1], chain[3]]
    check("detects: seq gap", bool(verify_chain(gap)))
    # fail-closed construction
    for name, kw in {
        "rejects NaN in payload": dict(payload={"x": float("nan")}),
        "rejects reserved extra key": dict(payload={}, extra={"digest": "0" * 64}),
        "rejects 'proven' lambda label": dict(payload={}, extra={"lambda": {
            "score": 0.9, "label": "Λ proven"}}),
        "rejects genesis with non-zero prev": dict(payload={}, seq=0, prev="a" * 64),
        "rejects bad ts": dict(payload={}, ts="yesterday"),
    }.items():
        try:
            make_receipt(**kw)
            check(name, False, "no exception")
        except (ValueError, TypeError) as exc:
            check(name, True, "(%s)" % type(exc).__name__)
    ok_lam = make_receipt({}, extra={"lambda": {"score": None,
                                                "label": "Λ = Conjecture 1 — never green"}},
                          ts=fixed_ts)
    check("accepts honest lambda label", not verify_receipt(ok_lam, schema=schema))
    # round-trip through disk
    d = out_dir or tempfile.mkdtemp(prefix="receipt_selftest_")
    paths = [write_receipt(r, d) for r in chain]
    back = [read_receipt(p) for p in paths]
    check("disk round-trip identical", back == chain, "(%s)" % d)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("SUMMARY: %d checks, %d failed -> %s" % (len(results), n_fail,
                                                   "PASS" if n_fail == 0 else "FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        od = None
        if "--out" in sys.argv:
            od = sys.argv[sys.argv.index("--out") + 1]
        raise SystemExit(_self_test(od))
    print(__doc__)
