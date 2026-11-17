"""Bounded, read-only release identity comparison; never an admission authority.

All "observed" values come from an owner-supplied, externally hash-pinned
snapshot. They are NOT independently fetched, authenticated or signed here.
The actual admission gate must separately verify each source and its authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .safeio import read_regular, strict_json

SCHEMA = "szl.release-decision-input/v1"
VIEW_SCHEMA = "szl.model-lab.release-decision-view/v1"
MAX_BYTES = 32_768
MAX_SNAPSHOT_AGE_SECONDS = 900
RAILS = ("source", "space", "gateway", "receipt")
FIELDS = {
    "source": frozenset({"repository", "revision"}),
    "space": frozenset({"space_id", "revision", "boot_id", "run_nonce"}),
    "gateway": frozenset({"repository", "revision", "route"}),
    "receipt": frozenset({"repository", "repo_type", "revision", "path", "sha256"}),
}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ATOM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
RECEIPT_REPO = re.compile(r"SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


class UnknownRail(ValueError):
    """Invalid user selection, rather than a filesystem or provider error."""


def _require(ok: bool) -> None:
    if not ok:
        raise ValueError("invalid_release_snapshot")


def _digest(value: object, size: int) -> str:
    pattern = HEX40 if size == 40 else HEX64
    _require(type(value) is str and pattern.fullmatch(value) is not None)
    return value


def _atom(value: object) -> str:
    _require(type(value) is str and ATOM.fullmatch(value) is not None)
    return value


def _path(value: object) -> str:
    _require(
        type(value) is str and len(value) <= 256 and PATH.fullmatch(value) is not None
    )
    _require(not any(p in (".", "..") for p in value.split("/")))
    return value


def _timestamp(value: object, now: datetime) -> datetime:
    _require(type(value) is str and len(value) <= 40)
    _require(
        re.fullmatch(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|[+-]\d\d:\d\d)", value
        )
        is not None
    )
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(moment.tzinfo is not None and moment.utcoffset() is not None)
    _require(moment <= now)
    return moment


def _identity(name: str, value: object) -> dict:
    _require(type(value) is dict and set(value) == FIELDS[name])
    if name == "source":
        _require(value["repository"] == "szl-holdings/a11oy")
    elif name == "space":
        _require(value["space_id"] == "SZLHOLDINGS/a11oy")
        _atom(value["boot_id"])
        _atom(value["run_nonce"])
    elif name == "gateway":
        _require(value["repository"] == "szl-holdings/szl-router")
        _atom(value["route"])
        _require(value["route"] == "a11oy")
    else:
        _require(
            type(value["repository"]) is str
            and RECEIPT_REPO.fullmatch(value["repository"]) is not None
        )
        _require(value["repo_type"] == "dataset")
        _path(value["path"])
        _digest(value["sha256"], 64)
    _digest(value["revision"], 40)
    return value


def unavailable_decision(state: str = "UNCONFIGURED") -> dict:
    _require(state in ("UNCONFIGURED", "INVALID_OR_UNAVAILABLE_SNAPSHOT"))
    return {
        "schema": VIEW_SCHEMA,
        "state": state,
        "release_id": None,
        "snapshot_sha256": None,
        "snapshot_hash_matched": False,
        "snapshot_observed_at": None,
        "snapshot_age_seconds": None,
        "rails": [],
        "counts": None,
        "blocking_reasons": ["MISSING_TRUSTED_RELEASE_EVIDENCE"],
        "hypothetical": False,
        "invalidated_rail": None,
        "execution_authority": False,
        "publication_authority": False,
        "signature_verified": False,
        "independent_witness_verified": False,
        "source_byte_parity_verified": False,
        "live_readback_performed": False,
        "production_ready": False,
        "action_performed": False,
        "scope": "owner-supplied identity comparison; no live provider verification",
    }


def project_decision(
    raw: bytes,
    expected_sha256: str,
    *,
    invalidate: str | None = None,
    now: datetime | None = None,
) -> dict:
    if invalidate is not None and invalidate not in RAILS:
        raise UnknownRail("unknown_release_rail")
    _digest(expected_sha256, 64)
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256)
    current = now or datetime.now(timezone.utc)
    _require(current.tzinfo is not None and current.utcoffset() is not None)
    body = strict_json(raw)
    _require(
        type(body) is dict
        and set(body) == {"schema", "scope", "release_id", "observed_at", "rails"}
    )
    _require(body["schema"] == SCHEMA and body["scope"] == "owner_local_unattested")
    release_id = _atom(body["release_id"])
    when = _timestamp(body["observed_at"], current)
    age = max(0, int((current - when).total_seconds()))
    entries = body["rails"]
    _require(type(entries) is dict and set(entries) == set(RAILS))
    rails = []
    reasons = [
        "SIGNER_AND_AUTHORITY_NOT_INDEPENDENTLY_VERIFIED",
        "PROVIDER_BYTE_PARITY_AND_LIVE_READBACK_NOT_PERFORMED",
    ]
    for name in RAILS:
        entry = entries[name]
        _require(type(entry) is dict and set(entry) == {"expected", "observed"})
        expected = _identity(name, entry["expected"])
        observed = entry["observed"]
        if observed is None:
            status = "MISSING_OBSERVATION"
            reasons.append(name.upper() + "_OBSERVATION_MISSING")
        else:
            _identity(name, observed)
            status = (
                "MATCH_IN_SUPPLIED_SNAPSHOT"
                if observed == expected
                else "CONFLICT_IN_SUPPLIED_SNAPSHOT"
            )
            if status == "CONFLICT_IN_SUPPLIED_SNAPSHOT":
                reasons.append(name.upper() + "_IDENTITY_CONFLICT")
        if invalidate == name:
            shown = "RECHECK_REQUIRED_SIMULATION"
        else:
            shown = status
        rails.append(
            {
                "name": name,
                "state": shown,
                "baseline_state": status,
                "expected": expected,
                "observed": observed,
                "identity_subject_sha256": hashlib.sha256(
                    json.dumps(
                        {"rail": name, "expected": expected, "observed": observed},
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            }
        )
    if age > MAX_SNAPSHOT_AGE_SECONDS:
        reasons.append("SUPPLIED_SNAPSHOT_OLDER_THAN_15_MINUTES")
    if invalidate is not None:
        reasons.append("HYPOTHETICAL_INPUT_RECHECK_ONLY")
    result = unavailable_decision()
    result.update(
        {
            "state": "HOLD_CONFLICT_REPORTED"
            if any(
                r["baseline_state"] == "CONFLICT_IN_SUPPLIED_SNAPSHOT" for r in rails
            )
            else "HOLD_UNVERIFIED",
            "release_id": release_id,
            "snapshot_sha256": expected_sha256,
            "snapshot_hash_matched": True,
            "snapshot_observed_at": body["observed_at"],
            "snapshot_age_seconds": age,
            "rails": rails,
            "counts": {
                "rails_total": 4,
                "reported_matching": sum(
                    r["baseline_state"] == "MATCH_IN_SUPPLIED_SNAPSHOT" for r in rails
                ),
                "reported_conflicting": sum(
                    r["baseline_state"] == "CONFLICT_IN_SUPPLIED_SNAPSHOT"
                    for r in rails
                ),
                "missing": sum(
                    r["baseline_state"] == "MISSING_OBSERVATION" for r in rails
                ),
            },
            "blocking_reasons": reasons,
            "hypothetical": invalidate is not None,
            "invalidated_rail": invalidate,
        }
    )
    return result


def load_decision(path: Path, expected_sha256: str, **kwargs) -> dict:
    return project_decision(read_regular(path, MAX_BYTES), expected_sha256, **kwargs)
