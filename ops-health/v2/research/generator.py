"""ops-health-generator/2.0.0 -- synthetic operational transport telemetry (SYNTHETIC only).

Implements v2/PREREGISTRATION.md section 3 exactly.  Standard library only.

Adapted from the v1 generator (szl-holdings/szl-forge commit 9a6fdd2c,
tools/train_operational_health_model.py, Apache-2.0, SZL Holdings): the nominal feature
draws, their order, the rounding, the label rule and the canonical JSONL writer are the v1
ones, so that the v1-parity mode reproduces the v1 public rows byte for byte.

The rows are operational counters only.  Nothing here is, or resembles, a message, a
person, a sample, or a result of any kind.

Per-row draw order (one ``random.Random`` per split; rows are drawn sequentially)
--------------------------------------------------------------------------------
nominal (identical to v1):
    1. listener_running           int(rng.random() < 0.94)
    2. tls_enabled                int(rng.random() < 0.90)
    3. peer_allowlist_configured  int(rng.random() < 0.93)
    4. queue_utilization          round(min(1.0, rng.betavariate(1.25, 4.5)), 6)
    5. consecutive_failures       v1 weighted draw: u = rng.random();
                                  u<0.72 -> 0; u<0.86 -> 1; u<0.93 -> 2;
                                  u<0.97 -> rng.randint(3, 5); else rng.randint(6, 20)
    6. seconds_since_last_success round(min(86400.0, rng.expovariate(1/900)), 6)
    7. ledger_integrity_ok        int(rng.random() < 0.985)
    8. configuration_valid        int(rng.random() < 0.96)
    9. label                      rng.random() < p(latent features)
queue_saturation: step 4 becomes round(min(1.0, rng.betavariate(4.0, 1.5)), 6); rest nominal.
failure_bursts:   step 5 becomes: b = rng.random(); if b < 0.35: rng.randint(3, 20)
                  else: the nominal v1 weighted draw (which then consumes its own draws).
missing_tls:      step 2 becomes int(rng.random() < 0.30); rest nominal.
clock_skew:       steps 1-8 nominal (these are the latent, true features), then
                  8b. skew = rng.uniform(0.0, 7200.0)
                  then step 9 (label from the latent features).  The observed
                  seconds_since_last_success written to the row is
                  round(min(86400.0, latent_seconds + skew), 6).  The latent value and
                  the skew are never written to the row.

Label rule (generator truth, PREREGISTRATION section 2): p(x) = 1/(1+exp(-z)) with z computed
from the v1-normalized latent features in exactly v1's floating-point operation order.

Seeds
-----
v2 splits: ``random.Random(derive_seed(split))`` with
``derive_seed(s) = int.from_bytes(sha256(f"oac-ops-health-v2:{M}:{s}".encode()).digest()[:8], "big")``
and M = 20260926.  DEV data for smoke/unit tests uses split name "dev".
v1-parity mode: one ``random.Random(2500)`` shared sequentially by train (768) ->
validation (192) -> test (240), v1 row schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

GENERATOR_VERSION = "ops-health-generator/2.0.0"
MASTER_SEED = 20260926
SEED_NAMESPACE = "oac-ops-health-v2"
ROW_SCHEMA = "szl-oac/ops-health-observation/v2"
V1_ROW_SCHEMA = "szl-oac/transport-health-observation/v1"
V1_SEED = 2500
V1_SPLIT_ROWS = (("train", 768), ("validation", 192), ("test", 240))
REGIMES = ("nominal", "queue_saturation", "failure_bursts", "clock_skew", "missing_tls")

# (name, minimum, maximum, kind) -- identical to the v1 kernel's FEATURE_SPECS.
FEATURE_SPECS = (
    ("listener_running", 0.0, 1.0, "binary"),
    ("tls_enabled", 0.0, 1.0, "binary"),
    ("peer_allowlist_configured", 0.0, 1.0, "binary"),
    ("queue_utilization", 0.0, 1.0, "continuous"),
    ("consecutive_failures", 0.0, 20.0, "continuous"),
    ("seconds_since_last_success", 0.0, 86400.0, "continuous"),
    ("ledger_integrity_ok", 0.0, 1.0, "binary"),
    ("configuration_valid", 0.0, 1.0, "binary"),
)
FEATURE_NAMES = tuple(spec[0] for spec in FEATURE_SPECS)


def derive_seed(split: str, master_seed: int = MASTER_SEED) -> int:
    """PREREGISTRATION section 3 seed derivation."""
    material = f"{SEED_NAMESPACE}:{master_seed}:{split}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def normalize(features: Mapping[str, Any]) -> list[float]:
    """v1 arithmetic: binary -> float(value); continuous -> (x - min) / (max - min)."""
    out = []
    for name, minimum, maximum, kind in FEATURE_SPECS:
        value = features[name]
        if kind == "binary":
            out.append(1.0 if value else 0.0)
        else:
            out.append((float(value) - minimum) / (maximum - minimum))
    return out


def truth_linear(features: Mapping[str, Any]) -> float:
    """Generator-truth logit z(x), same expression and evaluation order as v1."""
    values = dict(zip(FEATURE_NAMES, normalize(features), strict=True))
    return (
        -3.6
        + 3.1 * (1.0 - values["listener_running"])
        + 1.15 * (1.0 - values["tls_enabled"])
        + 1.35 * (1.0 - values["peer_allowlist_configured"])
        + 4.2 * values["queue_utilization"]
        + 2.7 * values["consecutive_failures"]
        + 2.2 * values["seconds_since_last_success"]
        + 4.4 * (1.0 - values["ledger_integrity_ok"])
        + 3.4 * (1.0 - values["configuration_valid"])
    )


def truth_probability(features: Mapping[str, Any]) -> float:
    return 1.0 / (1.0 + math.exp(-truth_linear(features)))


def _weighted_failure_count(rng: random.Random) -> int:
    draw = rng.random()
    if draw < 0.72:
        return 0
    if draw < 0.86:
        return 1
    if draw < 0.93:
        return 2
    if draw < 0.97:
        return rng.randint(3, 5)
    return rng.randint(6, 20)


def draw_row(rng: random.Random, regime: str) -> tuple[dict, dict, bool]:
    """Draw one row. Returns (observed_features, latent_features, label)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    latent: dict[str, int | float] = {}
    latent["listener_running"] = int(rng.random() < 0.94)
    latent["tls_enabled"] = int(rng.random() < (0.30 if regime == "missing_tls" else 0.90))
    latent["peer_allowlist_configured"] = int(rng.random() < 0.93)
    if regime == "queue_saturation":
        latent["queue_utilization"] = round(min(1.0, rng.betavariate(4.0, 1.5)), 6)
    else:
        latent["queue_utilization"] = round(min(1.0, rng.betavariate(1.25, 4.5)), 6)
    if regime == "failure_bursts":
        if rng.random() < 0.35:
            latent["consecutive_failures"] = rng.randint(3, 20)
        else:
            latent["consecutive_failures"] = _weighted_failure_count(rng)
    else:
        latent["consecutive_failures"] = _weighted_failure_count(rng)
    latent["seconds_since_last_success"] = round(min(86400.0, rng.expovariate(1.0 / 900.0)), 6)
    latent["ledger_integrity_ok"] = int(rng.random() < 0.985)
    latent["configuration_valid"] = int(rng.random() < 0.96)
    observed = dict(latent)
    if regime == "clock_skew":
        skew = rng.uniform(0.0, 7200.0)
        observed["seconds_since_last_success"] = round(
            min(86400.0, latent["seconds_since_last_success"] + skew), 6
        )
    label = rng.random() < truth_probability(latent)
    return observed, latent, label


def make_row(split: str, index: int, regime: str, features: dict, label: bool) -> dict:
    return {
        "schema": ROW_SCHEMA,
        "synthetic": True,
        "sample_id": f"{split}-{index:06d}",
        "regime": regime,
        "features": features,
        "label": {"operator_attention_required": bool(label)},
    }


def generate_rows(
    split: str, n_rows: int, regime: str = "nominal", *, master_seed: int = MASTER_SEED
) -> list[dict]:
    """Rows 0..n_rows-1 of split ``split`` (own derived seed)."""
    rng = random.Random(derive_seed(split, master_seed))
    rows = []
    for index in range(n_rows):
        observed, _latent, label = draw_row(rng, regime)
        rows.append(make_row(split, index, regime, observed, label))
    return rows


def generate_with_latent(
    split: str, n_rows: int, regime: str, *, master_seed: int = MASTER_SEED
) -> list[tuple[dict, dict, bool]]:
    """Research/test helper exposing latent features (never written to any data file)."""
    rng = random.Random(derive_seed(split, master_seed))
    return [draw_row(rng, regime) for _ in range(n_rows)]


def dev_rows(n_rows: int, regime: str = "nominal") -> list[dict]:
    """DEV data for smoke and unit tests: split name "dev" (never a registered split)."""
    return generate_rows("dev", n_rows, regime)


def generate_v1_parity() -> dict[str, list[dict]]:
    """v1-parity mode: seed 2500, one sequential Random, v1 row schema, no regime field."""
    rng = random.Random(V1_SEED)
    splits: dict[str, list[dict]] = {}
    for split, row_count in V1_SPLIT_ROWS:
        rows = []
        for index in range(row_count):
            observed, _latent, label = draw_row(rng, "nominal")
            rows.append(
                {
                    "schema": V1_ROW_SCHEMA,
                    "synthetic": True,
                    "sample_id": f"{split}-{index:06d}",
                    "features": observed,
                    "label": {"operator_attention_required": label},
                }
            )
        splits[split] = rows
    return splits


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Canonical JSONL, identical to v1's writer: sort_keys, (",", ":"), "\\n" per row."""
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write canonical JSONL; returns the sha256 of the bytes written."""
    data = jsonl_bytes(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
