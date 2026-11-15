# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, offline readiness projection. Never grants execution authority."""
from __future__ import annotations

import datetime as dt
import re

SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
ALLOWED = {"PASS", "FAIL", "UNAVAILABLE", "STALE"}
MAX_AGE_SECONDS = 86400


def _time(value: object) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone required")
    return parsed.astimezone(dt.timezone.utc)


def project(record: object, *, now: dt.datetime) -> dict:
    """Project evidence state, not production admission; reject unknown schema."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not isinstance(record, dict) or set(record) != {
        "schema", "source_sha", "artifact_sha256", "dataset_sha256",
        "grader_sha256", "observed_at", "checks", "execution_admitted"
    } or record["schema"] != "szl.evidence-readiness/v1":
        raise ValueError("unknown or incomplete evidence schema")
    for name, length in (("source_sha", SHA40), ("artifact_sha256", SHA64),
                         ("dataset_sha256", SHA64), ("grader_sha256", SHA64)):
        value = record[name]
        if not isinstance(value, str) or not length.fullmatch(value):
            raise ValueError(f"invalid {name}")
    if record["execution_admitted"] is not False:
        raise ValueError("offline evidence cannot authorize execution")
    observed = _time(record["observed_at"])
    age = (now.astimezone(dt.timezone.utc) - observed).total_seconds()
    if age < 0:
        raise ValueError("future evidence observation")
    checks = record["checks"]
    if not isinstance(checks, list) or not 1 <= len(checks) <= 100:
        raise ValueError("bounded nonempty checks required")
    names = set()
    results = []
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"name", "passed", "total", "state"}:
            raise ValueError("invalid check fields")
        name = check["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or name in names:
            raise ValueError("invalid or duplicate check")
        names.add(name)
        state = check["state"]
        passed, total = check["passed"], check["total"]
        if state not in ALLOWED or type(passed) is not int or type(total) is not int or not (0 <= passed <= total <= 1000000):
            raise ValueError("invalid check observation")
        if total == 0 and state != "UNAVAILABLE":
            raise ValueError("empty denominator must be unavailable")
        if total > 0 and state == "PASS" and passed != total:
            raise ValueError("passing check must have complete numerator")
        if total > 0 and state == "UNAVAILABLE":
            raise ValueError("unavailable check must have empty denominator")
        results.append({"name": name, "state": state, "passed": passed, "total": total})
    states = {x["state"] for x in results}
    # FAIL dominates STALE, which dominates UNAVAILABLE, which dominates PASS.
    state = next((s for s in ("FAIL", "STALE", "UNAVAILABLE", "PASS") if s in states), "UNAVAILABLE")
    if age > MAX_AGE_SECONDS and state != "FAIL":
        state = "STALE"
    return {
        "schema": "szl.evidence-readiness-projection/v1",
        "state": state,
        "checks": results,
        "age_seconds": int(age),
        "execution_authorized": False,
        "production_ready": False,
        "scope": "offline evidence projection only; no signer identity or runtime attestation",
    }
