"""Read-only projection of the existing a11oy compute-pool contract.

This consumer checks serialization, structure, counts, age and an independently
supplied snapshot digest. It does NOT re-verify the producer's DSSE receipts,
repeat its probes, import its executable code, or authorize scheduling.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .safeio import read_regular, strict_json

SCHEMA = "szl.compute-pool/v1"
SCHEMA_SOURCE = {
    "repository": "szl-holdings/a11oy",
    "revision": "4845411b81f6aa32c9802dae0b851928fc4f36c6",
    "path": "szl_compute_pool_contract.py",
    "git_blob": "d73d753dcacb7574c2fc7c83607fb4f703dd14d6",
    "meaning": "INSPECTED_SCHEMA_REFERENCE_NOT_PRODUCER_ATTESTATION",
}
_STATES = ("DECLARED", "CONFIGURED", "REACHABLE", "DISCOVERED", "QUALIFIED", "SERVING")
_CLASSES = {"MEASURED", "REPORTED", "UNKNOWN", "UNAVAILABLE"}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_NODE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_MAX_BYTES = 512 * 1024


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("pool_schema_mismatch")
    return value


def _time(value: Any) -> datetime:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("timestamp_required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone_required")
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError("timestamp_outside_supported_range") from exc


def _nullable_text(value: Any, limit: int) -> None:
    if value is not None and (not isinstance(value, str) or len(value) > limit):
        raise ValueError("bounded_text_required")


def _evidence(value: Any) -> None:
    row = _object(value, {"value", "evidence_class", "observed_at"})
    if row["value"] is not None and type(row["value"]) is not bool:
        raise ValueError("boolean_evidence_required")
    if not isinstance(row["evidence_class"], str) or row["evidence_class"] not in _CLASSES:
        raise ValueError("unknown_evidence_class")
    if row["observed_at"] is not None:
        _time(row["observed_at"])


def project_pool(raw: bytes, *, expected_sha256: str, now: datetime | None = None,
                 max_age_seconds: int = 300) -> dict[str, Any]:
    """Return reported observations; even upstream ready=true grants no authority."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= _MAX_BYTES:
        raise ValueError("pool_snapshot_size_rejected")
    if not isinstance(expected_sha256, str) or not _HEX.fullmatch(expected_sha256):
        raise ValueError("externally_supplied_snapshot_digest_required")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("pool_snapshot_digest_mismatch")
    if type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 3600:
        raise ValueError("invalid_local_freshness_limit")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("aware_clock_required")
    doc = _object(strict_json(raw), {"schema_version", "generated_at", "source_surface",
                                   "control_plane_relation", "receipt_freshness_seconds", "counts", "nodes"})
    if (doc["schema_version"] != SCHEMA
            or doc["source_surface"] != "szl_backend_hardening.probe_fabric_pool"
            or doc["control_plane_relation"] != "PYTHON_HF_PROJECTION_NOT_REPLIT_CONTROL_PLANE"):
        raise ValueError("wrong_pool_contract")
    freshness = doc["receipt_freshness_seconds"]
    if type(freshness) is not int or not 30 <= freshness <= 3600:
        raise ValueError("invalid_producer_freshness")
    generated = _time(doc["generated_at"])
    age = (current.astimezone(timezone.utc) - generated).total_seconds()
    fresh = -30 <= age <= min(max_age_seconds, freshness)
    if not isinstance(doc["nodes"], list) or len(doc["nodes"]) > 128:
        raise ValueError("bounded_nodes_required")
    counts = _object(doc["counts"], {s.lower() for s in _STATES} | {"ready"})
    if any(type(v) is not int or not 0 <= v <= 128 for v in counts.values()):
        raise ValueError("invalid_pool_counts")
    seen: set[str] = set()
    rows = []
    ranks = []
    for item in doc["nodes"]:
        node = _object(item, {"node_id", "kind", "sovereign", "endpoint_label", "state", "ready",
                              "evidence_class", "configuration", "reachability", "inference_receipt"})
        name = node["node_id"]
        if not isinstance(name, str) or not _NODE.fullmatch(name) or name in seen:
            raise ValueError("invalid_or_duplicate_node")
        seen.add(name)
        if (not isinstance(node["state"], str) or node["state"] not in _STATES
                or not isinstance(node["evidence_class"], str) or node["evidence_class"] not in _CLASSES):
            raise ValueError("invalid_node_state")
        if type(node["ready"]) is not bool or type(node["sovereign"]) is not bool:
            raise ValueError("strict_boolean_required")
        if node["ready"] != (node["state"] in {"QUALIFIED", "SERVING"}):
            raise ValueError("inconsistent_upstream_readiness")
        _nullable_text(node["kind"], 128)
        _nullable_text(node["endpoint_label"], 2048)
        _evidence(node["configuration"])
        _evidence(node["reachability"])
        receipt = _object(node["inference_receipt"], {"evidence_class", "verified", "fresh", "bounded",
                          "observed_at", "model_id", "model_digest_sha256", "receipt_sha256", "reason"})
        if (not isinstance(receipt["evidence_class"], str) or receipt["evidence_class"] not in _CLASSES
                or any(type(receipt[k]) is not bool for k in ("verified", "fresh", "bounded"))):
            raise ValueError("invalid_receipt_summary")
        if receipt["observed_at"] is not None:
            _time(receipt["observed_at"])
        _nullable_text(receipt["model_id"], 256)
        _nullable_text(receipt["reason"], 2048)
        for key in ("model_digest_sha256", "receipt_sha256"):
            if receipt[key] is not None and (not isinstance(receipt[key], str) or not _HEX.fullmatch(receipt[key])):
                raise ValueError("invalid_receipt_digest")
        # Only display a narrow subset. Never echo endpoint addresses, keys or raw bodies.
        rows.append({"node_id": name, "reported_state": node["state"],
                     "upstream_ready_reported": node["ready"],
                     "model_id_reported": receipt["model_id"],
                     "receipt_observed_at_reported": receipt["observed_at"],
                     "receipt_fresh_reported": receipt["fresh"],
                     "model_digest_sha256_reported": receipt["model_digest_sha256"],
                     "ready": False, "qualification_verified_here": False})
        ranks.append(_STATES.index(node["state"]))
    expected_counts = {state.lower(): sum(rank >= index for rank in ranks)
                       for index, state in enumerate(_STATES)}
    expected_counts["ready"] = sum(row["upstream_ready_reported"] for row in rows)
    if counts != expected_counts:
        raise ValueError("pool_counts_do_not_match_items")
    return {"state": "REPORTED_SNAPSHOT" if fresh else "STALE_REPORTED_SNAPSHOT",
            "schema": SCHEMA, "schema_checked": True, "schema_source_inspected": dict(SCHEMA_SOURCE),
            "snapshot_sha256": expected_sha256, "integrity": "EXTERNAL_DIGEST_MATCH_ONLY",
            "generated_at": doc["generated_at"], "snapshot_fresh": fresh,
            "producer_revision_verified": False, "pool_qualification_verified": False,
            "signature_verified": False, "ready": False,
            "counts_reported": dict(counts), "nodes": rows}


def load_pool_snapshot(path: Path, expected_sha256: str, *, now: datetime | None = None) -> dict[str, Any]:
    return project_pool(read_regular(path, _MAX_BYTES), expected_sha256=expected_sha256, now=now)


def unavailable_pool(state: str = "UNCONFIGURED") -> dict[str, Any]:
    return {"state": state, "schema": SCHEMA, "schema_checked": False,
            "nodes": None, "counts_reported": None, "snapshot_fresh": False,
            "pool_qualification_verified": False, "signature_verified": False, "ready": False}
