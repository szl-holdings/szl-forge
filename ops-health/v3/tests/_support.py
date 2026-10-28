"""Shared DEV fixtures for the v3 step-2 tests (SYNTHETIC; DEV data only, prereg s.15.5).

DEV rows come only from the split name ``dev`` (v3 namespace, M3 = 20261013), within the DEV cap
of 2,048 rows per regime per DEV run (one process = one DEV run; ``guards.DEV_LEDGER`` counts).
Fixed DEV layout of this suite:

    nominal plain, 1,024 rows      [0:342] threshold, [342:684] conformal, [684:1024] gates
    design D, 512 draws            D arm (the DEV fit's 512 training rows) and no-D arm
                                   (2 x 512 = 1,024 nominal rows)          -> nominal: 2,048
    clock_skew                     twin 256 draws (512 rows), plain 256 rows, v2-namespace twin
                                   256 draws (512 rows), v2 module's own 256 draws with latent
                                   (512 rows)                                -> 1,792
    queue_saturation, failure_bursts, missing_tls: 1,024 rows each

No registered split is read, nothing under v2/data/sealed/ is opened, no fit uses more than 2,048
DEV rows, and no P or H statistic, tolerance, detection ratio, power or WIN-chance quantity is
computed. Receipts frozen here carry FAKE_COMMIT, which no origin check can accept.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from v3.research import freeze, generator, guards, kernel_bridge, pipeline, registry

ROOT = registry.ROOT
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
DEV_NOMINAL = 1024
DEV_D = 512
DEV_TWIN = 256
DEV_SHIFT = 1024
THRESHOLD = slice(0, 342)
CONFORMAL = slice(342, 684)
GATES = slice(684, 1024)
DEV_TAG = "oac-v3-fit"
DEV_FIT_NAME = "dev-fit (DEV data; not a registered fit)"


@lru_cache(maxsize=1)
def nominal_rows() -> tuple:
    return tuple(generator.dev_rows(DEV_NOMINAL, "nominal"))


@lru_cache(maxsize=1)
def d_rows() -> tuple:
    d_arm, nod_arm = generator.dev_rows_d(DEV_D)
    return tuple(d_arm), tuple(nod_arm)


@lru_cache(maxsize=4)
def shift_rows(regime: str) -> tuple:
    if regime not in ("queue_saturation", "failure_bursts", "missing_tls", "clock_skew"):
        raise ValueError(regime)
    n = DEV_TWIN if regime == "clock_skew" else DEV_SHIFT
    return tuple(generator.dev_rows(n, regime))


@lru_cache(maxsize=1)
def twin_rows() -> tuple:
    observed, latent = generator.dev_rows_twin(DEV_TWIN)
    return tuple(observed), tuple(latent)


def threshold_rows() -> list:
    return list(nominal_rows()[THRESHOLD])


def conformal_rows() -> list:
    return list(nominal_rows()[CONFORMAL])


def gate_rows() -> list:
    return list(nominal_rows()[GATES])


def train_rows() -> list:
    return list(d_rows()[0])


def kernel():
    return kernel_bridge.kernel()


@lru_cache(maxsize=1)
def dev_manifest_bytes() -> bytes:
    """A DEV data manifest (the receipt binds its sha256; there is no registered manifest yet)."""
    streams = {
        "dev_nominal": pipeline.rows_sha256(nominal_rows()),
        "dev_D_arm": pipeline.rows_sha256(d_rows()[0]),
        "dev_noD_arm": pipeline.rows_sha256(d_rows()[1]),
    }
    manifest = {
        "schema": "szl-oac/ops-health-dev-manifest/v3",
        "dev": True,
        "synthetic": True,
        "note": "DEV data only (prereg s.15.5); not v3/data/MANIFEST.json",
        "generator_sha256": generator.source_sha256(),
        "generator_version": generator.GENERATOR_VERSION,
        "streams_sha256": streams,
    }
    return (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build(rows_train=None):
    """(artifact, diagnostics) of the DEV candidate, rebuilt from scratch."""
    return pipeline.build_candidate(
        train_rows=list(rows_train) if rows_train is not None else train_rows(),
        threshold_rows=threshold_rows(),
        conformal_rows=conformal_rows(),
        split_names={"train": "dev", "threshold": "dev", "conformal": "dev"},
    )


@lru_cache(maxsize=1)
def _cached() -> str:
    artifact, diagnostics = build()
    return json.dumps({"artifact": artifact, "diagnostics": diagnostics})


def candidate() -> dict:
    return json.loads(_cached())["artifact"]


def diagnostics() -> dict:
    return json.loads(_cached())["diagnostics"]


def dev_trace_bytes() -> bytes:
    """The DEV fit's trace: the canonical bytes of its diagnostics (12-decimal rounding, as for
    every canonical artifact), which a DEV run writes as ``dev_fit/diagnostics.json``.

    The receipt's ``fit.trace_sha256`` is the sha256 of these bytes, so it can be checked from the
    committed DEV output. Until 2026-10-28 the digest covered ``json.dumps`` of the in-memory
    diagnostics at full float precision: bytes never written to any file (unverifiable), and they
    differed between H-local (Windows) and the VM (Linux) in two IRLS round-off residuals near 1e-16
    (MEASURED, DEV run comparison at cd62f144). DEV only; the registered fit trace is not built here.
    """
    return freeze.canonical_json_bytes(diagnostics())


def dev_fit_block() -> dict:
    return {"segment": 1, "declared_fit": DEV_FIT_NAME,
            "trace_sha256": hashlib.sha256(dev_trace_bytes()).hexdigest(), "receipts_dir_sha256": None}


class TempDir:
    """Context manager for a temporary directory (removed on exit)."""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="oac_v3_test_"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def freeze_dev(out_dir: Path, artifact: dict | None = None, *, commit: str = FAKE_COMMIT,
               tree_clean: bool = True) -> dict:
    """Freeze a DEV candidate with the TEST-ONLY source-state override."""
    return freeze.freeze(
        artifact if artifact is not None else candidate(),
        out_dir,
        tag=DEV_TAG,
        fit=dev_fit_block(),
        data_manifest_bytes=dev_manifest_bytes(),
        source_state_override={"commit": commit, "tree_clean": tree_clean},
    )


def write_consistent_pair(directory: Path, model_bytes: bytes, receipt: dict) -> tuple[Path, Path]:
    return freeze.write_pair(directory, model_bytes, freeze.consistent_receipt(receipt, model_bytes))


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"mutation target occurs {count} times: {old!r}")
    return text.replace(old, new)


_COUNTER = [0]


def load_kernel_copy(directory: Path, source: str | None = None, *, mutate=None):
    """Write the kernel source (optionally mutated) to a fresh sub-directory and load it."""
    text = source if source is not None else registry.KERNEL_PATH.read_text(encoding="utf-8")
    if mutate is not None:
        text = mutate(text)
    _COUNTER[0] += 1
    sub = Path(directory) / f"kernel_copy_{_COUNTER[0]}"
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / "ops_health.py"
    path.write_bytes(text.encode("utf-8"))
    return kernel_bridge.load(path), path


def rebind_receipt(receipt_path: Path, kernel_path: Path, out_path: Path) -> Path:
    """A copy of the receipt bound to another kernel file's sha256 (TEST ONLY)."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    receipt["kernel_sha256"] = sha256_file(kernel_path)
    out_path.write_bytes(freeze.canonical_json_bytes(receipt))
    return out_path


def fail_open_mutation(text: str) -> str:
    """A deliberately broken, fail-open v3 kernel: no screening, no schema, finiteness or range
    checks (missing fields default to 0, extras are ignored). Fixture of s.11.1."""
    text = replace_once(text, "            match = prohibited_match(key)\n", "            match = None\n")
    text = replace_once(
        text,
        "    elif isinstance(value, (str, bytes, bytearray)):\n"
        "        raise ModelInputError(\n"
        "            f\"text values are not accepted at {_shown_path(path)}; input is numeric only\"\n"
        "        )\n",
        "",
    )
    text = replace_once(text, "    if depth > MAX_INPUT_DEPTH:\n", "    if False:\n")
    text = replace_once(text, "    if keys != expected:\n", "    if False:\n")
    text = replace_once(
        text,
        "        _normalize_value(name, minimum, maximum, kind, features[name], *table[name])\n",
        "        _normalize_value(name, minimum, maximum, kind, features.get(name, 0), *table[name])\n",
    )
    text = replace_once(
        text,
        "                     offset: float, scale: float) -> float:\n",
        "                     offset: float, scale: float) -> float:\n"
        "    try:\n"
        "        number = float(value) if not isinstance(value, (list, dict, type(None))) else 0.0\n"
        "    except (TypeError, ValueError, OverflowError):\n"
        "        number = 0.0\n"
        "    if number != number or number in (float('inf'), float('-inf')):\n"
        "        number = 0.0\n"
        "    return (min(max(number, minimum), maximum) - offset) / scale\n",
    )
    text = replace_once(text, "    if set(payload) != {\"features\"}:\n", "    if False:\n")
    text = replace_once(
        text,
        "    if \"features\" not in payload:\n"
        "        raise ModelInputError('input must be wrapped as {\"features\": {...}}')\n",
        "    if \"features\" not in payload:\n"
        "        payload = {\"features\": payload}\n",
    )
    return text


def v2_kernel():
    """The v2 kernel module (for the by-reference AUTHORITY_MAP check)."""
    from v2.research import kernel_bridge as v2_bridge  # noqa: PLC0415

    return v2_bridge.kernel()


def ledger_snapshot() -> dict:
    return guards.DEV_LEDGER.snapshot()
