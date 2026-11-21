"""Bounded, CPU-only document-boundary conservation witness.

This is an offline diagnostic for token arrays accompanied by independent
document-length metadata. It does not launch training, build attention masks,
verify a dataset publisher's identity, or grant release authority.

The upstream lesson is OLMo-core v3.0.0 (Apache-2.0): EOS inference can
silently merge documents when truncation removed their EOS tokens. This
implementation is original, dependency-free SZL code, not an OLMo port.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MAX_BYTES = 2 * 1024 * 1024
MAX_TOKENS = 131_072
MAX_DOCUMENTS = 4_096
MAX_SEGMENTS = 8_192
MAX_BLOCK_SIZE = 32_768


class BoundaryWitnessError(ValueError):
    """A required source boundary or boundedness invariant failed."""


def _strict_int(value: Any) -> bool:
    return type(value) is int


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise BoundaryWitnessError("DUPLICATE_JSON_KEY")
        result[name] = value
    return result


def _deny_constant(_value: str) -> None:
    raise BoundaryWitnessError("NONFINITE_JSON_VALUE")


def read_manifest(path: Path) -> dict[str, Any]:
    """Load one small JSON manifest; no network or model/tokenizer imports."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise BoundaryWitnessError("INPUT_TOO_LARGE")
        value = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_deny_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BoundaryWitnessError("INVALID_OR_UNREADABLE_JSON") from exc
    if not isinstance(value, dict):
        raise BoundaryWitnessError("MANIFEST_NOT_OBJECT")
    return value


def audit_document_boundaries(manifest: dict[str, Any]) -> dict[str, Any]:
    """Preserve every input token by chunking at *explicit* document offsets.

    Returned segments are index ranges into the provided token list, NOT
    evidence that a downstream trainer actually consumed those ranges.
    A source-supplied length vector may itself be untrustworthy; upstream
    custody and immutable dataset revision must be verified separately.
    """
    if not isinstance(manifest, dict):
        raise BoundaryWitnessError("MANIFEST_NOT_OBJECT")
    if set(manifest) - {"tokens", "document_lengths", "block_size", "eos_token_id", "source_revision"}:
        raise BoundaryWitnessError("UNKNOWN_MANIFEST_FIELDS")
    tokens = manifest.get("tokens")
    lengths = manifest.get("document_lengths")
    block = manifest.get("block_size")
    eos = manifest.get("eos_token_id")
    source = manifest.get("source_revision", "UNKNOWN")
    if not isinstance(tokens, list) or not tokens or len(tokens) > MAX_TOKENS:
        raise BoundaryWitnessError("INVALID_TOKEN_ARRAY")
    if not all(_strict_int(t) and 0 <= t <= 2**31 - 1 for t in tokens):
        raise BoundaryWitnessError("INVALID_TOKEN_VALUE")
    if not isinstance(lengths, list) or not 1 <= len(lengths) <= MAX_DOCUMENTS:
        raise BoundaryWitnessError("MISSING_OR_INVALID_DOCUMENT_LENGTHS")
    if not all(_strict_int(n) and n > 0 for n in lengths) or sum(lengths) != len(tokens):
        raise BoundaryWitnessError("DOCUMENT_LENGTH_MISMATCH")
    if not _strict_int(block) or not 1 <= block <= MAX_BLOCK_SIZE:
        raise BoundaryWitnessError("INVALID_BLOCK_SIZE")
    if eos is not None and (not _strict_int(eos) or not 0 <= eos <= 2**31 - 1):
        raise BoundaryWitnessError("INVALID_EOS_TOKEN_ID")
    if not isinstance(source, str) or not source or len(source) > 160 or any(ord(c) < 32 for c in source):
        raise BoundaryWitnessError("INVALID_SOURCE_REVISION")

    count = sum((n + block - 1) // block for n in lengths)
    if count > MAX_SEGMENTS:
        raise BoundaryWitnessError("TOO_MANY_SEGMENTS")

    segments: list[dict[str, int]] = []
    eos_unsafe: list[int] = []
    cursor = 0
    for doc_id, n in enumerate(lengths):
        stop = cursor + n
        if eos is not None and tokens[stop - 1] != eos:
            eos_unsafe.append(doc_id)
        for part_id, start in enumerate(range(cursor, stop, block)):
            end = min(start + block, stop)
            segments.append({
                "document": doc_id,
                "chunk": part_id,
                "start": start,
                "stop": end,
                "length": end - start,
            })
        cursor = stop

    # Independent range-based replay guards against a future change
    # accidentally skipping or duplicating a token.
    reconstructed: list[int] = []
    for seg in segments:
        reconstructed.extend(tokens[seg["start"]:seg["stop"]])
    if reconstructed != tokens or sum(seg["length"] for seg in segments) != len(tokens):
        raise BoundaryWitnessError("TOKEN_CONSERVATION_FAILED")

    if eos is None:
        eos_status = "UNKNOWN_NO_EOS_ID"
    elif eos_unsafe:
        eos_status = "UNSAFE_FOR_EOS_ONLY_INFERENCE"
    else:
        eos_status = "TERMINATORS_OBSERVED_NOT_SOURCE_PROOF"
    receipt: dict[str, Any] = {
        "schema": "szl.document-boundary-witness/v1",
        "status": "METADATA_CONSERVATION_PASSED",
        "source_revision": source,
        "source_identity_verified": False,
        "training_consumption_verified": False,
        "training_authority": "NONE",
        "token_count": len(tokens),
        "document_count": len(lengths),
        "segment_count": len(segments),
        "nonpadding_token_utilization_numerator": len(tokens),
        "nonpadding_token_utilization_denominator": len(tokens),
        "token_array_sha256": _sha(tokens),
        "document_lengths_sha256": _sha(lengths),
        "eos_only_inference": eos_status,
        "eos_unsafe_document_indices": eos_unsafe,
        "segments": segments,
    }
    receipt["receipt_sha256"] = _sha(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Bounded JSON fixture with tokens and explicit document_lengths")
    args = parser.parse_args()
    try:
        result = audit_document_boundaries(read_manifest(args.input))
    except BoundaryWitnessError as exc:
        print(json.dumps({"schema": "szl.document-boundary-witness/v1", "status": "DENY", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
