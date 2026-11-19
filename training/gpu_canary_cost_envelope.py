#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, read-only cost envelope for ONE proposed GPU training canary.

This is NOT training authorization: prices, rights, images, provenance, and
budgets supplied by a caller are untrusted assertions until the existing
Forge/operator gates independently verify them. No credentials, network, model
libraries, provider mutation, or external actions live in this module.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import json
import re
from typing import Any

SCHEMA = "szl.forge.gpu-canary-cost-envelope/v1"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
MONEY = re.compile(r"(?:0|[1-9][0-9]{0,3})(?:\.[0-9]{1,4})?\Z")
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
MAX_SECONDS = 7200
MAX_PLAN_BYTES = 64 * 1024
QUOTE_TTL = timedelta(minutes=30)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous members at every nesting depth, including unknown keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_PLAN_JSON_MEMBER")
        result[key] = value
    return result


def _refuse_number(_value: str) -> None:
    # All numeric fields in the approved schema must be integers; money is
    # encoded as an exact decimal string. Do not admit NaN, infinity or floats.
    raise ValueError("UNSUPPORTED_PLAN_JSON_NUMBER")


def parse_plan_bytes(raw: bytes) -> dict[str, Any]:
    """Fail closed before assessing a bounded, unambiguous JSON document."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_PLAN_BYTES:
        raise ValueError("INVALID_PLAN_BYTE_LENGTH")
    parsed = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_float=_refuse_number,
        parse_constant=_refuse_number,
    )
    if not isinstance(parsed, dict):
        raise ValueError("PLAN_OBJECT_REQUIRED")
    return parsed


def _money(value: object) -> Decimal | None:
    # JSON floats introduce binary rounding and are never treated as quotes.
    if not isinstance(value, str) or not MONEY.fullmatch(value):
        return None
    amount = Decimal(value)
    return amount if amount > 0 else None


def _utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.utcoffset() == timedelta(0) else None


def assess(plan: object, *, now: datetime) -> dict[str, Any]:
    """Return a modeled cost bound or a typed HOLD, never launch permission.

    All source and price fields must be independently corroborated by the
    one authorized launch owner. An estimate can never establish data rights,
    actual available GPU supply, billing, checkpoint durability or signatures.
    """
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("an aware UTC observation clock is required")
    now = now.astimezone(timezone.utc)
    codes: list[str] = []
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        return {
            "schema": SCHEMA, "verdict": "HOLD",
            "codes": ["INVALID_PLAN_SCHEMA"], "modeled_gpu_cost_ceiling_usd": None,
            "launch_authorized": False, "provider_mutations": 0,
        }

    if plan.get("provider") != "salad" or plan.get("organization") != "alloyscape" or plan.get("project") != "default":
        codes.append("PROVIDER_SCOPE_NOT_CANONICAL")
    group = plan.get("container_group")
    if not isinstance(group, str) or NAME.fullmatch(group) is None:
        codes.append("CONTAINER_GROUP_INVALID")

    for key in ("student_revision", "dataset_revision"):
        value = plan.get(key)
        if not isinstance(value, str) or SHA40.fullmatch(value) is None:
            codes.append(f"{key.upper()}_UNPINNED")
    for key in ("train_sha256", "holdout_sha256"):
        value = plan.get(key)
        if not isinstance(value, str) or SHA64.fullmatch(value) is None:
            codes.append(f"{key.upper()}_MISSING")
    if plan.get("train_sha256") == plan.get("holdout_sha256"):
        codes.append("TRAIN_HOLDOUT_IDENTICAL")
    image = plan.get("worker_image")
    if not isinstance(image, str) or IMAGE.fullmatch(image) is None:
        codes.append("WORKER_IMAGE_DIGEST_UNPINNED")

    # Proposals cannot smuggle a fleet scale-up or retry campaign into a canary.
    if type(plan.get("gpu_count")) is not int or plan["gpu_count"] != 1:
        codes.append("SINGLE_GPU_ONLY")
    seconds = plan.get("max_running_seconds")
    if type(seconds) is not int or not 1 <= seconds <= MAX_SECONDS:
        codes.append("RUNTIME_CAP_INVALID")
    if type(plan.get("max_attempts")) is not int or plan["max_attempts"] != 1:
        codes.append("RETRY_CAP_INVALID")

    price = _money(plan.get("observed_gpu_rate_usd_hour"))
    cap = _money(plan.get("operator_cost_cap_usd"))
    if price is None or cap is None:
        codes.append("PROVIDER_QUOTE_OR_CAP_INVALID")
    quote_at = _utc(plan.get("price_observed_utc"))
    if quote_at is None or quote_at > now or now - quote_at > QUOTE_TTL:
        codes.append("PROVIDER_PRICE_NOT_FRESH")

    modeled: str | None = None
    if price is not None and cap is not None and type(seconds) is int and 1 <= seconds <= MAX_SECONDS:
        amount = price * Decimal(seconds) / Decimal(3600)
        ceil_cents = amount.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        modeled = str(ceil_cents)
        if ceil_cents > cap:
            codes.append("MODELED_COST_EXCEEDS_DECLARED_CAP")

    return {
        "schema": SCHEMA,
        "verdict": "HOLD" if codes else "COST_ENVELOPE_MODELED_NOT_ADMITTED",
        "codes": list(dict.fromkeys(codes)),
        "modeled_gpu_cost_ceiling_usd": modeled,
        "modeled_scope": "ONE_GPU_RUNNING_SECONDS_ONLY; other fees and billable state unverified",
        "quote_provenance": "CALLER_ASSERTED_NOT_PROVIDER_AUTHENTICATED",
        "rights_verified": False, "owner_signature_verified": False,
        "dataset_disjointness_verified": False,
        "trainer_checkpoint_verified": False,
        "launch_authorized": False,
        "training_started": False,
        "provider_mutations": 0,
    }


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    try:
        with args.plan.open("rb") as stream:
            proposal = parse_plan_bytes(stream.read(MAX_PLAN_BYTES + 1))
        result = assess(proposal, now=datetime.now(timezone.utc))
    except (OSError, ValueError, UnicodeError, RecursionError):
        result = assess(None, now=datetime.now(timezone.utc))
    print(json.dumps(result, sort_keys=True))
    # Deliberately non-zero even for a cost-valid proposal; an untrusted
    # budget estimate must never become a launch-success exit code.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
