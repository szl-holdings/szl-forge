# SPDX-License-Identifier: Apache-2.0
"""Public sealed-run counter: dates and counts only.

After merge #193, provider-unavailable evaluation runs seal a negative
receipt locally and must not publish the payload bundle to the public
receipts dataset. Public silence is then ambiguous: the runner may not
have fired, or it may have sealed a negative.

This module is the honest resolution. The runner may publish
``runs/SEALED_COUNT.json`` — UTC dates and integer counts, nothing else.
No provider names, no scores, no receipts, no run identities, no payloads.

The counter starts at zero when it first exists. Zero does not mean
"no seals occurred during the pre-counter silence." That interval is
UNAVAILABLE, not a filled zero. See ``known_bounds``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .core import canonical_bytes, canonical_sha256

SCHEMA = "szl.frontier.sealed-run-counter.v1"
EPOCH = "post-193"
COUNTER_PATH = "runs/SEALED_COUNT.json"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ALLOWED_KEYS = frozenset(
    {
        "schema",
        "epoch",
        "updated_at",
        "total_sealed",
        "days",
        "parent_sha256",
        "counter_sha256",
        "known_bounds",
    }
)
DAY_KEYS = frozenset({"date", "sealed"})
KNOWN_BOUNDS = (
    "counts_and_utc_dates_only",
    "no_payloads",
    "no_provider_details",
    "pre_counter_silence_is_unavailable_not_zero",
)
FORBIDDEN_KEY_FRAGMENTS = (
    "provider",
    "payload",
    "receipt",
    "model",
    "score",
    "run_id",
    "candidate",
    "secret",
    "token",
    "error_summary",
    "bundle",
    "latency",
    "http",
    "prompt",
    "answer",
    "evidence",
)


class SealedCountError(ValueError):
    """Fail-closed counter error. Never coerce a bad document into a count."""


def utc_date(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise SealedCountError("counter time must be timezone-aware")
    return stamp.astimezone(timezone.utc).date().isoformat()


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def _reject_forbidden_keys(value: Mapping[str, Any]) -> None:
    for key in _walk_keys(value):
        lowered = key.casefold()
        for fragment in FORBIDDEN_KEY_FRAGMENTS:
            if fragment in lowered:
                raise SealedCountError(
                    f"sealed counter must not carry key {key!r}"
                )


def _hashable(counter: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in counter.items()
        if key != "counter_sha256"
    }


def _with_hash(counter: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(counter)
    sealed["counter_sha256"] = canonical_sha256(_hashable(sealed))
    return sealed


def genesis() -> dict[str, Any]:
    """Empty public counter. total_sealed=0 is 'no counted seals yet'."""

    return _with_hash(
        {
            "schema": SCHEMA,
            "epoch": EPOCH,
            "updated_at": None,
            "total_sealed": 0,
            "days": [],
            "parent_sha256": None,
            "known_bounds": list(KNOWN_BOUNDS),
        }
    )


def encode(counter: Mapping[str, Any]) -> bytes:
    validate(counter)
    return canonical_bytes(dict(counter))


def validate(counter: Mapping[str, Any]) -> None:
    if not isinstance(counter, Mapping):
        raise SealedCountError("counter must be an object")
    _reject_forbidden_keys(counter)
    extra = set(counter) - ALLOWED_KEYS
    missing = ALLOWED_KEYS - set(counter)
    if extra:
        raise SealedCountError(f"unexpected counter keys: {sorted(extra)}")
    if missing:
        raise SealedCountError(f"missing counter keys: {sorted(missing)}")
    if counter["schema"] != SCHEMA:
        raise SealedCountError("unknown sealed-count schema")
    if counter["epoch"] != EPOCH:
        raise SealedCountError("unknown sealed-count epoch")
    if counter["known_bounds"] != list(KNOWN_BOUNDS):
        raise SealedCountError("known_bounds must match the declared tuple")
    total = counter["total_sealed"]
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise SealedCountError("total_sealed must be a non-negative int")
    days = counter["days"]
    if not isinstance(days, list):
        raise SealedCountError("days must be a list")
    seen: list[str] = []
    summed = 0
    for row in days:
        if not isinstance(row, Mapping) or set(row) != DAY_KEYS:
            raise SealedCountError("each day must have only date and sealed")
        date = row["date"]
        sealed = row["sealed"]
        if not isinstance(date, str) or not DATE_RE.fullmatch(date):
            raise SealedCountError("day date must be YYYY-MM-DD")
        if not isinstance(sealed, int) or isinstance(sealed, bool) or sealed < 1:
            raise SealedCountError("day sealed must be an int >= 1")
        if seen and date <= seen[-1]:
            raise SealedCountError("days must be strictly increasing by date")
        seen.append(date)
        summed += sealed
    if summed != total:
        raise SealedCountError("total_sealed does not equal sum of days")
    updated = counter["updated_at"]
    if total == 0:
        if days:
            raise SealedCountError("zero total cannot carry day rows")
        if updated is not None:
            raise SealedCountError("genesis updated_at stays null")
        if counter["parent_sha256"] is not None:
            raise SealedCountError("genesis parent_sha256 stays null")
    else:
        if not isinstance(updated, str) or not updated:
            raise SealedCountError("updated_at required after the first seal")
        parent = counter["parent_sha256"]
        if not isinstance(parent, str) or not HEX64.fullmatch(parent):
            raise SealedCountError("parent_sha256 must be a 64-hex digest")
    digest = counter["counter_sha256"]
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise SealedCountError("counter_sha256 must be a 64-hex digest")
    expected = canonical_sha256(_hashable(counter))
    if digest != expected:
        raise SealedCountError("counter_sha256 does not match payload bytes")


def load_counter(raw: bytes | str | None) -> dict[str, Any]:
    """Load a counter document. None yields genesis. Unreadable input fails."""

    if raw is None:
        return genesis()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, (bytes, bytearray)) or not raw.strip():
        raise SealedCountError("unreadable sealed-count document")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedCountError("unreadable sealed-count document") from error
    if not isinstance(parsed, dict):
        raise SealedCountError("sealed-count document is not an object")
    validate(parsed)
    return parsed


def increment(
    previous: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the next counter. Counts only; previous evidence is not copied."""

    current = genesis() if previous is None else dict(previous)
    validate(current)
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise SealedCountError("counter time must be timezone-aware")
    stamp = stamp.astimezone(timezone.utc)
    day = stamp.date().isoformat()
    parent = str(current["counter_sha256"])
    days = [dict(row) for row in current["days"]]
    if days and days[-1]["date"] == day:
        days[-1]["sealed"] = int(days[-1]["sealed"]) + 1
    else:
        if days and day < days[-1]["date"]:
            raise SealedCountError("refusing to seal on a date before the head")
        days.append({"date": day, "sealed": 1})
    nxt = _with_hash(
        {
            "schema": SCHEMA,
            "epoch": EPOCH,
            "updated_at": stamp.isoformat(),
            "total_sealed": int(current["total_sealed"]) + 1,
            "days": days,
            "parent_sha256": parent,
            "known_bounds": list(KNOWN_BOUNDS),
        }
    )
    validate(nxt)
    if nxt["total_sealed"] <= current["total_sealed"]:
        raise SealedCountError("sealed count must strictly increase")
    return nxt


def apply_seal(previous_bytes: bytes | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Load previous bytes (or genesis) and increment. Used by the publisher."""

    return increment(load_counter(previous_bytes), now=now)


def public_view(counter: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the document to the public surface. Still counts and dates only."""

    validate(counter)
    return {
        "schema": counter["schema"],
        "epoch": counter["epoch"],
        "updated_at": counter["updated_at"],
        "total_sealed": counter["total_sealed"],
        "days": [dict(row) for row in counter["days"]],
        "parent_sha256": counter["parent_sha256"],
        "counter_sha256": counter["counter_sha256"],
        "known_bounds": list(counter["known_bounds"]),
        "path_in_repo": COUNTER_PATH,
    }


def publish_sealed_count(
    *,
    token: str,
    dataset_id: str,
    now: datetime | None = None,
    fetch_bytes=None,
    commit_bytes=None,
) -> dict[str, Any]:
    """Publish only ``runs/SEALED_COUNT.json``. Never uploads a receipt payload.

    ``fetch_bytes`` and ``commit_bytes`` are injectable for offline tests.
    Production uses Hugging Face Hub with the same write token as receipt
    publication, but this function never reads evaluation output directories.
    """

    if not token:
        raise SealedCountError("HF_TOKEN is required to publish the sealed counter")
    previous = fetch_bytes(dataset_id) if fetch_bytes else _hub_fetch(token, dataset_id)
    nxt = apply_seal(previous, now=now)
    payload = encode(nxt)
    if token.encode("utf-8") in payload:
        raise SealedCountError("publication credential found in sealed counter")
    commit = (
        commit_bytes(dataset_id, payload)
        if commit_bytes
        else _hub_commit(token, dataset_id, payload, nxt)
    )
    view = public_view(nxt)
    view["commit_oid"] = commit["commit_oid"]
    view["commit_url"] = commit.get("commit_url") or ""
    view["status"] = "SEALED_COUNT_ONLY"
    view["production_disposition"] = "HOLD"
    view["promotion_effect"] = "NONE"
    _reject_forbidden_keys(view)
    return view


def _hub_fetch(token: str, dataset_id: str) -> bytes | None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi(token=token)
    try:
        path = api.hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=COUNTER_PATH,
            revision="main",
        )
    except EntryNotFoundError:
        return None
    data = Path(path).read_bytes()
    if token.encode("utf-8") in data:
        raise SealedCountError("publication credential found in existing counter")
    return data


def _hub_commit(
    token: str, dataset_id: str, payload: bytes, counter: Mapping[str, Any]
) -> dict[str, str]:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset", revision="main")
    parent = str(info.sha or "")
    if not re.fullmatch(r"[0-9a-f]{40}", parent):
        raise SealedCountError("dataset head could not be identified")
    commit = api.create_commit(
        repo_id=dataset_id,
        repo_type="dataset",
        revision="main",
        parent_commit=parent,
        operations=[
            CommitOperationAdd(path_in_repo=COUNTER_PATH, path_or_fileobj=payload)
        ],
        commit_message=(
            f"eval: sealed-count {counter['total_sealed']} "
            f"{str(counter['counter_sha256'])[:12]}"
        ),
    )
    oid = str(getattr(commit, "oid", "") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", oid):
        raise SealedCountError("sealed-count publication returned no commit identity")
    return {
        "commit_oid": oid,
        "commit_url": str(getattr(commit, "commit_url", "") or ""),
    }
