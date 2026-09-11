# SPDX-License-Identifier: Apache-2.0
"""Observe the public sealed-run counter. Do not invent counts.

Extends the existing HF frontier watcher. A missing public
``runs/SEALED_COUNT.json`` is UNAVAILABLE, not total_sealed=0.
Zero is only honest after the genesis document exists on the Hub.
No credentials, payloads, provider names, or promotion.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from frontier.evaluation.sealed_count import (
    COUNTER_PATH,
    SCHEMA,
    SealedCountError,
    load_counter,
    public_view,
    validate,
)

DATASET_ID = "SZLHOLDINGS/szl-frontier-evaluation-receipts"
PUBLIC_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{COUNTER_PATH}"
)
OBSERVATION_SCHEMA = "szl.forge.sealed-count-observation.v1"
MAX_BYTES = 64 * 1024
TIMEOUT_SECONDS = 20


class SealedCountWatchError(ValueError):
    """Invalid public counter evidence must not become a count."""


def _timestamp(value: str | None = None) -> str:
    if value is not None:
        return value
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def observe(
    raw: bytes | None,
    *,
    http_status: int | None,
    observed_at: str | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Build an observation from fetched bytes. Never fills UNAVAILABLE as 0."""

    stamp = _timestamp(observed_at)
    base: dict[str, Any] = {
        "schema": OBSERVATION_SCHEMA,
        "observedAt": stamp,
        "sourceRepository": "szl-holdings/szl-forge",
        "sourceRevision": source_revision,
        "datasetId": DATASET_ID,
        "counterPath": COUNTER_PATH,
        "counterSchema": SCHEMA,
        "productionDisposition": "HOLD",
        "promotionEffect": "NONE",
        "known_bounds": [
            "counts_and_utc_dates_only",
            "missing_public_counter_is_unavailable_not_zero",
            "pre_counter_silence_is_unavailable_not_zero",
            "no_payloads",
            "no_provider_details",
        ],
    }
    if http_status == 404:
        base["observationStatus"] = "UNAVAILABLE"
        base["reasonCode"] = "public_counter_absent"
        return base
    if raw is None or (http_status is not None and http_status != 200):
        base["observationStatus"] = "UNAVAILABLE"
        base["reasonCode"] = "public_counter_fetch_failed"
        return base
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_BYTES:
        raise SealedCountWatchError("public counter exceeded bound or was not bytes")
    try:
        counter = load_counter(bytes(raw))
        validate(counter)
        view = public_view(counter)
    except (SealedCountError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SealedCountWatchError("public counter failed closed") from exc
    if "total_sealed" not in view:
        raise SealedCountWatchError("public view omitted total_sealed")
    base["observationStatus"] = "WATCH"
    base["reasonCode"] = "public_counter_present"
    base["total_sealed"] = view["total_sealed"]
    base["updated_at"] = view.get("updated_at")
    base["day_count"] = len(view.get("days") or [])
    base["counter_sha256"] = counter.get("counter_sha256")
    return base


def fetch_public(url: str = PUBLIC_URL) -> tuple[bytes | None, int]:
    """Unauthenticated GET. 404 is a first-class UNAVAILABLE, not an exception."""

    request = Request(url, method="GET", headers={"User-Agent": "szl-forge-sealed-count-watch"})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_BYTES + 1)
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        if exc.code == 404:
            return None, 404
        return None, int(exc.code)
    except (URLError, TimeoutError, OSError):
        return None, 0
    if len(body) > MAX_BYTES:
        raise SealedCountWatchError("public counter exceeded bound")
    return body, status


def refresh(*, observed_at: str | None = None, source_revision: str | None = None) -> dict[str, Any]:
    raw, status = fetch_public()
    return observe(
        raw,
        http_status=status,
        observed_at=observed_at,
        source_revision=source_revision,
    )
