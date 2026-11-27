"""Governed-receipt reward: JSON Schema plus a DSSE signature.

Schemas are the files vendored from ``szl-holdings/governed-receipt-spec`` at
the commit in ``PIN.json``. Signatures are ECDSA P-256 SHA-256 over the DSSE
PAE of the decoded payload bytes, the same check as that repository's
verifier. A schema-valid envelope with no signature scores 0. Callers pass a
public key; this module never embeds one.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from training.rlvr.receipt import RewardBatch, batch_reward, completion_text, failed, passed

VERIFIER = "szl.rlvr.receipt"
_VENDOR = Path(__file__).resolve().parent / "vendor" / "governed-receipt-spec"
_IN_TOTO = "https://in-toto.io/Statement/v1"
_MAX_BYTES = 1_000_000


def dsse_pae(payload_type: str, body: bytes) -> bytes:
    """DSSE v1 pre-authentication encoding of the decoded payload."""
    if not isinstance(payload_type, str) or not payload_type:
        raise ValueError("payload_type must be a non-empty string")
    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    return (
        b"DSSEv1 "
        + str(len(payload_type.encode("utf-8"))).encode("ascii")
        + b" "
        + payload_type.encode("utf-8")
        + b" "
        + str(len(body)).encode("ascii")
        + b" "
        + body
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_schemas() -> dict[str, Any]:
    pin = _load_json(_VENDOR / "PIN.json")
    receipt = _load_json(_VENDOR / "schema" / "governed-receipt.schema.json")
    action = _load_json(_VENDOR / "schema" / "governed-action.v1.schema.json")
    envelope = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **receipt["$defs"]["dsseEnvelope"],
    }
    return {"pin": pin, "receipt": receipt, "action": action, "envelope": envelope}


def _validator(schema: dict[str, Any]):
    from jsonschema import Draft202012Validator, FormatChecker

    return Draft202012Validator(schema, format_checker=FormatChecker())


def _schema_errors(instance: Any, schema: dict[str, Any]) -> list[str]:
    errors = sorted(_validator(schema).iter_errors(instance), key=lambda item: list(item.path))
    return [error.message for error in errors]


def _decode_payload(envelope: dict[str, Any]) -> tuple[bytes | None, Any, str | None]:
    raw = envelope.get("payload")
    if not isinstance(raw, str):
        return None, None, "payload missing"
    try:
        body = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError):
        return None, None, "payload is not strict base64"
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return body, None, "payload is not JSON"
    return body, decoded, None


def _select_schema(payload: Any, schemas: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, Any]:
    if not isinstance(payload, dict):
        return None, None, payload
    if payload.get("_type") == _IN_TOTO:
        return "governed-action", schemas["action"], payload.get("predicate")
    action = payload.get("action")
    if isinstance(action, str):
        return "governed-receipt", schemas["receipt"], payload
    if isinstance(action, dict):
        return "governed-action", schemas["action"], payload
    return None, None, payload


def _verify_signature(envelope: dict[str, Any], body: bytes, public_key_pem: str | None) -> str | None:
    """Return None when the signature verifies, else a reason."""
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        return "unsigned receipt"
    if not public_key_pem:
        return "unsigned receipt: no verify key"
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError:
        return "cryptography unavailable"
    try:
        preimage = dsse_pae(envelope.get("payloadType", ""), body)
        key = serialization.load_pem_public_key(
            public_key_pem.encode("utf-8") if isinstance(public_key_pem, str) else public_key_pem
        )
    except Exception:  # noqa: BLE001
        return "verify key or PAE rejected"
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1):
        return "verify key is not an ECDSA P-256 public key"
    for index, entry in enumerate(signatures):
        if not isinstance(entry, dict) or not isinstance(entry.get("sig"), str):
            return f"signature[{index}] malformed"
        try:
            signature = base64.b64decode(entry["sig"], validate=True)
            key.verify(signature, preimage, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            return f"signature[{index}] mismatch"
        except Exception:  # noqa: BLE001
            return f"signature[{index}] invalid"
    return None


def _parse_completion(completion: Any) -> Any:
    if isinstance(completion, (dict, list)):
        return completion
    text = completion_text(completion)
    if len(text.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("completion too large")
    return json.loads(text)


def score_receipt(
    completion: Any,
    *,
    verify_key_pem: str | None = None,
    **_ignored: Any,
):
    pin_commit = ""
    try:
        schemas = load_schemas()
        pin_commit = schemas["pin"]["commit"]
    except Exception as exc:  # noqa: BLE001
        return failed(VERIFIER, completion, f"vendored schema unreadable: {type(exc).__name__}")
    extra = {"schema_commit": pin_commit, "schema_valid": False}
    try:
        document = _parse_completion(completion)
    except Exception:  # noqa: BLE001
        return failed(VERIFIER, completion, "malformed receipt JSON", **extra)
    if not isinstance(document, dict):
        return failed(VERIFIER, completion, "receipt is not a JSON object", **extra)
    envelope = document.get("envelope") or document.get("dsse") or document
    if not isinstance(envelope, dict):
        return failed(VERIFIER, completion, "DSSE envelope missing", **extra)
    envelope_errors = _schema_errors(envelope, schemas["envelope"])
    if envelope_errors:
        return failed(VERIFIER, completion, "DSSE schema: " + envelope_errors[0], **extra)
    body, payload, payload_error = _decode_payload(envelope)
    if payload_error or body is None:
        return failed(VERIFIER, completion, payload_error or "payload undecodable", **extra)
    name, schema, instance = _select_schema(payload, schemas)
    if schema is None:
        return failed(
            VERIFIER,
            completion,
            "payload matches neither governed receipt nor governed action",
            **extra,
        )
    schema_errors = _schema_errors(instance, schema)
    if schema_errors:
        return failed(VERIFIER, completion, f"{name} schema: " + schema_errors[0], **extra)
    extra["schema_valid"] = True
    extra["schema"] = name
    signature_error = _verify_signature(envelope, body, verify_key_pem)
    if signature_error:
        return failed(VERIFIER, completion, signature_error, **extra)
    return passed(
        VERIFIER,
        completion,
        f"{name} schema valid and DSSE signature verified",
        **extra,
    )


def receipt_reward(completions: list[Any], **kwargs: Any) -> RewardBatch:
    """TRL GRPO reward. Returns floats; receipts are on ``.receipts``."""
    score_receipt.verifier_name = VERIFIER  # type: ignore[attr-defined]
    return batch_reward(completions, score_receipt, **kwargs)
