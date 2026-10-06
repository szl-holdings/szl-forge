"""Fixed paths and the registered split table (PREREGISTRATION sections 3-4)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "v2"
DATA_DIR = V2 / "data"
SEALED_DIRNAME = "sealed"
SEALED_DIR = DATA_DIR / SEALED_DIRNAME
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"
LEGACY_DIR = DATA_DIR / "legacy_v1"
RUNS_DIR = ROOT / "runs"
TEST_OPENED_NAME = "TEST_OPENED.json"
PREREG_PATH = V2 / "PREREGISTRATION.md"
PREREG_SHA256 = "145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b"
GENERATOR_PATH = V2 / "research" / "generator.py"
KERNEL_PATH = V2 / "ops-health" / "ops_health.py"
V1_HUB_DIR = ROOT / "hub" / "oac-v1"
V1_HUB_REVISION = "dd7d109813abcd90c5250106dc2eabd2804e7ac3"
V1_BASE_SOURCE = f"SZLHOLDINGS/oac-system-health-v1@{V1_HUB_REVISION}:oac_operational_health.py"

# (split, regime, rows, sealed, registered use)
SPLITS = (
    ("train", "nominal", 16384, False, "fit weights"),
    ("calibration", "nominal", 4096, False, "fit the Platt/isotonic calibrator only"),
    ("conformal", "nominal", 4096, False, "split-conformal quantiles only"),
    ("validation", "nominal", 4096, False, "hyperparameter, calibrator, and threshold selection only"),
    ("test", "nominal", 50000, True, "SEALED; opened once"),
    ("shift_queue_saturation", "queue_saturation", 12500, True, "SEALED"),
    ("shift_failure_bursts", "failure_bursts", 12500, True, "SEALED"),
    ("shift_clock_skew", "clock_skew", 12500, True, "SEALED"),
    ("shift_missing_tls", "missing_tls", 12500, True, "SEALED"),
)
SPLIT_TABLE = {name: (regime, rows, sealed, use) for name, regime, rows, sealed, use in SPLITS}
SEALED_SPLITS = tuple(name for name, _r, _n, sealed, _u in SPLITS if sealed)
OPEN_SPLITS = tuple(name for name, _r, _n, sealed, _u in SPLITS if not sealed)


def split_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    regime, _rows, sealed, _use = SPLIT_TABLE[name]
    return (data_dir / SEALED_DIRNAME / f"{name}.jsonl") if sealed else (data_dir / f"{name}.jsonl")


def relpath(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()
