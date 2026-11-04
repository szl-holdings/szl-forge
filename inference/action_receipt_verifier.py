"""Verifier for A11oy-signed action admissions (DSSE, ECDSA P-256).

``make_dsse_action_receipt_verifier`` builds a callable for the ``receipt_verifier=``
hook of ``production_infer`` and ``verify_external_execution``. It returns True only
when every check passes:

1. ``receipt["dsse"]`` is a DSSE envelope whose payloadType is ACTION_PAYLOAD_TYPE.
2. ``receipt["sha256"]`` equals the SHA-256 of the decoded payload bytes.
3. A signature whose keyid equals the configured key's keyid verifies with
   ECDSA-P256-SHA256 over the DSSEv1 PAE (the A11oy ``szl_dsse`` format; keyid is the
   SHA-256 of the normalized SPKI PEM).
4. The payload is strict JSON (no duplicate keys, no NaN/Infinity) with schema
   ACTION_SCHEMA, authority A11OY, human_approval APPROVED, ``admission_sha256`` equal
   to the canonical SHA-256 of the admission being checked, and ``not_after`` (UTC,
   ``YYYY-MM-DDTHH:MM:SSZ``) not yet passed; an optional ``not_before`` must have passed.

Anything else returns False, including a missing ``cryptography`` package. The
public key is supplied by the host; this module pins no key. Replay of a still-valid
receipt for an identical admission is not detected here.
"""
from __future__ import annotations

import base64
import binascii
import datetime as _dt
import hashlib
import json
import re
from typing import Any, Callable, Mapping

ACTION_PAYLOAD_TYPE = "application/vnd.szl.a11oy.action-admission+json"
ACTION_SCHEMA = "szl.a11oy.action-admission/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


def canonical_bytes(value: Any) -> bytes:
    """Sorted-key compact UTF-8 JSON, identical to Forge and A11oy canonical JSON."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def pae(payload_type: str, body: bytes) -> bytes:
    """DSSEv1 pre-authentication encoding."""
    kind = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(kind), kind, len(body), body)


def _normalized_public_pem(public_pem: str) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = serialization.load_pem_public_key(public_pem.encode("utf-8"))
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp256r1":
        raise ValueError("action receipt key must be an ECDSA P-256 public key")
    return key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")


def keyid_for_public_pem(public_pem: str) -> str:
    """A11oy keyid: SHA-256 of the normalized SPKI PEM."""
    normalized = _normalized_public_pem(public_pem)
    return hashlib.sha256(normalized.strip().encode("ascii")).hexdigest()


def _reject_constant(token: str) -> Any:
    raise ValueError("non-finite number " + token)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key " + key)
        out[key] = value
    return out


def _strict_json(body: bytes) -> Any:
    return json.loads(
        body.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_no_duplicate_keys,
    )


def _parse_utc(value: Any) -> _dt.datetime:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError("timestamp must be YYYY-MM-DDTHH:MM:SSZ")
    parsed = _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return parsed.replace(tzinfo=_dt.timezone.utc)


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def make_dsse_action_receipt_verifier(
    public_pem: str,
    *,
    expected_keyid: str | None = None,
    now: Callable[[], _dt.datetime] = _utcnow,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool]:
    """Return a fail-closed receipt verifier bound to one A11oy public key.

    ``expected_keyid`` pins the key out of band. A mismatch raises ValueError here,
    so a wrong key is a configuration error rather than a silent pass.
    """
    normalized = _normalized_public_pem(public_pem)
    keyid = hashlib.sha256(normalized.strip().encode("ascii")).hexdigest()
    if expected_keyid is not None and str(expected_keyid).strip().lower() != keyid:
        raise ValueError("public key does not match the pinned keyid")

    def verify(admission: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
        try:
            return _verify(admission, receipt, keyid, normalized, now)
        except Exception:
            return False

    verify.keyid = keyid  # type: ignore[attr-defined]
    return verify


def _verify(
    admission: Mapping[str, Any],
    receipt: Mapping[str, Any],
    keyid: str,
    public_pem: str,
    now: Callable[[], _dt.datetime],
) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    if not isinstance(admission, Mapping) or not isinstance(receipt, Mapping):
        return False
    envelope = receipt.get("dsse")
    if not isinstance(envelope, Mapping):
        return False
    if envelope.get("payloadType") != ACTION_PAYLOAD_TYPE:
        return False
    payload_b64 = envelope.get("payload")
    if not isinstance(payload_b64, str):
        return False
    body = base64.b64decode(payload_b64, validate=True)
    claimed = str(receipt.get("sha256") or "").lower()
    if not _HEX64.fullmatch(claimed) or hashlib.sha256(body).hexdigest() != claimed:
        return False
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list):
        return False
    key = serialization.load_pem_public_key(public_pem.encode("ascii"))
    message = pae(ACTION_PAYLOAD_TYPE, body)
    signed = False
    for entry in signatures:
        if not isinstance(entry, Mapping) or entry.get("keyid") != keyid:
            continue
        sig = entry.get("sig")
        if not isinstance(sig, str):
            continue
        try:
            key.verify(
                base64.b64decode(sig, validate=True), message, ec.ECDSA(hashes.SHA256())
            )
        except (InvalidSignature, binascii.Error, ValueError):
            continue
        signed = True
        break
    if not signed:
        return False
    payload = _strict_json(body)
    if not isinstance(payload, dict):
        return False
    if payload.get("schema") != ACTION_SCHEMA:
        return False
    if payload.get("authority") != "A11OY":
        return False
    if payload.get("human_approval") != "APPROVED":
        return False
    admission_digest = hashlib.sha256(canonical_bytes(dict(admission))).hexdigest()
    if payload.get("admission_sha256") != admission_digest:
        return False
    current = now()
    if not isinstance(current, _dt.datetime) or current.tzinfo is None:
        return False
    if current > _parse_utc(payload.get("not_after")):
        return False
    if "not_before" in payload and current < _parse_utc(payload["not_before"]):
        return False
    return True
