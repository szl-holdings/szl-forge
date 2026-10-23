"""Registered constants of OAC Ops Health v3 (SYNTHETIC only).

Every value here is copied from the registered protocol ``v3/PREREGISTRATION.md`` (sha256 below)
as amended by the seal re-registration ``v3/REREGISTRATION_r1.md``. Section numbers (s.) refer to
the protocol. Nothing in this file is a result.

Paths are relative to the research root: the directory that holds ``v2/`` and ``v3/`` (in the
public home, ``ops-health/`` in szl-holdings/szl-forge).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "v2"
V3 = ROOT / "v3"

# -- registered documents (s.15.2: pinned by sha256, never edited) -------------------------
PREREG_PATH = V3 / "PREREGISTRATION.md"
PREREG_SHA256 = "24c239e7e57049180b609b70e1b283b1d3f61253122e9e189232c69a1a3e1d0e"
REREG_R1_PATH = V3 / "REREGISTRATION_r1.md"
REREG_R1_SHA256 = "6c5f83ae92e89f91041e2f69dd0e43e3bf518d541006dc3e67dd04fa52b03ac3"
AMENDMENTS_PATH = V3 / "AMENDMENTS.md"

# -- seeds (s.3.2 items 1 and 2) -----------------------------------------------------------
SEED_NAMESPACE = "oac-ops-health-v3"
MASTER_SEED = 20261013
# v2's namespace and master seed: used only by the v2 parity test (s.3.3), digests only.
V2_SEED_NAMESPACE = "oac-ops-health-v2"
V2_MASTER_SEED = 20260926
V2_SEALED_SPLITS = (
    "test",
    "shift_queue_saturation",
    "shift_failure_bursts",
    "shift_clock_skew",
    "shift_missing_tls",
)
V2_PARITY_SPLITS = ("train", "calibration", "conformal", "validation")
V2_DATA_DIR = V2 / "data"
V2_SEALED_DIR = V2_DATA_DIR / "sealed"
V2_MANIFEST_PATH = V2_DATA_DIR / "MANIFEST.json"
V1_LEGACY_DIR = V2_DATA_DIR / "legacy_v1"

# -- beacon seal (s.3.4 as amended by r1: R means R' = 6659606) ----------------------------
CHAIN_HASH = "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce"
CHAIN_SCHEME = "pedersen-bls-chained"
CHAIN_GROUP_HASH = "176f93498eac9ca337150b46d21dd58673ea4e3581185f869672e59fa4cb390a"
CHAIN_PERIOD = 30
CHAIN_GENESIS_TIME = 1595431050
CHAIN_PUBLIC_KEY = (
    "868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31"
)
BEACON_ROUND = 6659606  # R' (r1)
BEACON_ROUND_TIME = CHAIN_GENESIS_TIME + (BEACON_ROUND - 1) * CHAIN_PERIOD  # 2026-11-21T00:00:00Z
FIRST_REGISTERED_ROUND = 6578966  # abandoned by r1; its randomness is public from its time
FIRST_REGISTERED_ROUND_TIME = 1792800000  # 2026-10-24T00:00:00Z: frozen-state-fix window closed
RELAYS = ("https://drand.cloudflare.com", "https://api.drand.sh", "https://api2.drand.sh")

# -- splits (s.4) ---------------------------------------------------------------------------
# (name, regime, mode, rows, sealed, stream, use)
#   regime: the v2 regime of the per-row draws (s.3.1);
#   mode:   plain | D | noD | twin_observed | twin_latent (s.3.2 items 3 and 4);
#   stream: the split whose seed drives the draws. A paired file (train_noD, train_kg_k_noD,
#           h3_clock_skew_latent) is written from its partner's stream (s.3.2: "two paired files
#           from one stream"; "the same row with its latent seconds").
TRAIN_ROWS = 32768
BAND_RETRAINS = 100
KNOWN_GOOD_RETRAINS = 20
DECLARED_FITS = 227
PI_D = 0.25


def _split_table() -> tuple:
    rows = [
        ("train", "nominal", "D", TRAIN_ROWS, False, "train", "fit v3"),
        ("train_noD", "nominal", "noD", TRAIN_ROWS, False, "train", "fit the no-D ablation arms only"),
        ("calibration", "nominal", "plain", 4096, False, "calibration", "ABL-calibration's Platt fit only"),
        ("conformal", "nominal", "plain", 8192, False, "conformal", "Mondrian quantiles only"),
        ("validation", "nominal", "plain", 8192, False, "validation",
         "threshold selection; ablation and conformal keep/drop decisions"),
        ("gates", "nominal", "plain", 8192, False, "gates", "behavioral gates of s.11 only"),
    ]
    for k in range(1, BAND_RETRAINS + 1):
        name = f"train_band_{k:03d}"
        rows.append((name, "nominal", "D", TRAIN_ROWS, False, name, "band retrain (s.11.3); digest only"))
    for k in range(1, KNOWN_GOOD_RETRAINS + 1):
        name = f"train_kg_{k:02d}"
        rows.append((name, "nominal", "D", TRAIN_ROWS, False, name, "known-good retrain; digest only"))
        rows.append((f"{name}_noD", "nominal", "noD", TRAIN_ROWS, False, name,
                     "known-good retrain, no-D arm; digest only"))
    rows += [
        ("test", "nominal", "plain", 100000, True, "test", "SEALED; opened once"),
        ("shift_queue_saturation", "queue_saturation", "plain", 25000, True, "shift_queue_saturation",
         "SEALED; pooled shift suite"),
        ("shift_failure_bursts", "failure_bursts", "plain", 25000, True, "shift_failure_bursts",
         "SEALED; pooled shift suite"),
        ("shift_clock_skew", "clock_skew", "plain", 25000, True, "shift_clock_skew",
         "SEALED; pooled shift suite"),
        ("shift_missing_tls", "missing_tls", "plain", 25000, True, "shift_missing_tls",
         "SEALED; pooled shift suite"),
        ("h3_clock_skew", "clock_skew", "twin_observed", 100000, True, "h3_clock_skew", "SEALED; H3 only"),
        ("h3_clock_skew_latent", "clock_skew", "twin_latent", 100000, True, "h3_clock_skew",
         "SEALED; H3 only"),
    ]
    return tuple(rows)


SPLITS = _split_table()
SPLIT_TABLE = {row[0]: row[1:] for row in SPLITS}
SEALED_SPLITS = tuple(row[0] for row in SPLITS if row[4])
NON_SEALED_SPLITS = tuple(row[0] for row in SPLITS if not row[4])
STORED_SPLITS = ("train", "train_noD", "calibration", "conformal", "validation", "gates")
RETRAIN_SPLITS = tuple(n for n in NON_SEALED_SPLITS if n not in STORED_SPLITS)
POOLED_SHIFT_SUITE = (
    "shift_queue_saturation",
    "shift_failure_bursts",
    "shift_clock_skew",
    "shift_missing_tls",
)
LEGACY_V1_TEST = "legacy_v1_test"

# -- DEV rule (s.15.5) -------------------------------------------------------------------------
DEV_SPLIT = "dev"
DEV_MAX_ROWS_PER_REGIME = 2048
DEV_MAX_FIT_ROWS = 2048

# -- recipe constants (s.6) --------------------------------------------------------------------
L2 = 1e-3
ALPHA = 0.10
THRESHOLD_GRID_MIN = 1
THRESHOLD_GRID_MAX = 99
SECONDS_SCALE = 900.0  # representation R (s.6.2)

# -- runtime pins (s.13 step 9) ------------------------------------------------------------------
PINNED_PYTHON = (3, 12, 10)
PINNED_INVOCATION = "python -I"
PINNED_HASHSEED = "0"

# -- kernel and code paths -------------------------------------------------------------------------
KERNEL_PATH = V3 / "ops-health" / "ops_health.py"
GENERATOR_PATH = V3 / "research" / "generator.py"
V2_KERNEL_PATH = V2 / "ops-health" / "ops_health.py"
DEV_LOG_DIR = ROOT / "logs" / "v3" / "dev"


def relpath(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()
