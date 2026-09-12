"""Strict, bounded evidence primitives. Hashes are not signatures or permission.

This module makes no network calls, loads no optional model library, and mutates
only a caller-selected local output file. Production authority belongs to A11oy.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

MAX_JSON = 4 * 1024 * 1024
AUTHORITY_ORDER = ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"]
DENIED = {
    "productionPromotion": False,
    "providerWrites": False,
    "trainingLaunch": False,
    "billableJobCreation": False,
    "weightPublication": False,
    "toolExecution": False,
}


class EvidenceError(ValueError):
    """Invalid, missing or ambiguous evidence; never interpret this as PASS."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise EvidenceError(reason)


def finite(value: Any, name: str = "number") -> float:
    require(type(value) in (int, float), name + "_not_numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvidenceError(name + "_not_finite") from exc
    require(math.isfinite(result), name + "_not_finite")
    return result


def integer(value: Any, low: int, high: int, name: str = "integer") -> int:
    require(type(value) is int and low <= value <= high, name + "_out_of_bounds")
    return value


def text(value: Any, limit: int = 1024) -> str:
    require(isinstance(value, str) and 0 < len(value) <= limit, "invalid_text")
    require(not any(ord(c) < 32 for c in value), "text_has_control_characters")
    return value


def digest(value: Any, size: int = 64) -> str:
    require(isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{%d}" % size, value)), "invalid_digest")
    return value


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as exc:
        raise EvidenceError("noncanonical_json") from exc


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in out, "duplicate_json_key")
        out[key] = value
    return out


def _nonfinite(_: str) -> None:
    raise EvidenceError("nonfinite_json")


def parse_json(raw: bytes) -> dict[str, Any]:
    require(type(raw) is bytes and 0 < len(raw) <= MAX_JSON, "json_size_bound")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
        require(type(data) is dict, "json_object_required")
        canonical(data)  # Reject exponent overflow and invalid Unicode surrogates.
        return data
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise EvidenceError("invalid_json") from exc


def load_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return parse_json(stream.read(MAX_JSON + 1))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(text(value, 64).replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "timezone_required")
        return parsed.astimezone(dt.timezone.utc)
    except (ValueError, TypeError, OverflowError) as exc:
        raise EvidenceError("invalid_timestamp") from exc


def age_seconds(value: str, *, now: str | None = None) -> float:
    return (timestamp(now or utc_now()) - timestamp(value)).total_seconds()


def receipt(payload: dict[str, Any]) -> dict[str, Any]:
    require("receiptSha256" not in payload, "receipt_already_sealed")
    # Copy through canonical JSON: avoid caller mutation through shared references.
    copied = parse_json(canonical(payload))
    return {**copied, "receiptSha256": sha256(copied)}


def verify_receipt(value: dict[str, Any], expected: str | None = None) -> None:
    require(type(value) is dict, "receipt_object_required")
    actual = digest(value.get("receiptSha256"))
    require(sha256({k: v for k, v in value.items() if k != "receiptSha256"}) == actual, "receipt_hash_mismatch")
    if expected is not None:
        require(actual == digest(expected), "external_receipt_pin_mismatch")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomic local replacement. Not authenticated or permanent immutable storage."""
    raw = canonical(value) + b"\n"
    require(len(raw) <= MAX_JSON, "output_exceeds_reader_bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".frontier-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_sha256(path: Path, max_bytes: int) -> str:
    """Hash local bytes without loading the whole file or trusting its size label."""
    integer(max_bytes, 1, 10**15, "max_bytes")
    require(not path.is_symlink() and path.is_file(), "regular_file_required")
    total, hasher = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while body := stream.read(min(1024 * 1024, max_bytes - total + 1)):
            total += len(body)
            require(total <= max_bytes, "file_exceeds_bound")
            hasher.update(body)
    return hasher.hexdigest()


def observation_shell(schema: str, source_revision: str) -> dict[str, Any]:
    return {"schema": text(schema, 120), "sourceRevision": digest(source_revision, 40),
            "observedAt": utc_now(), "authorityOrder": list(AUTHORITY_ORDER),
            "authority": dict(DENIED), "productionDisposition": "HOLD",
            "signatureState": "UNSIGNED"}
