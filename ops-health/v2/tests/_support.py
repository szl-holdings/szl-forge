"""Shared DEV fixtures for the v2 stage-1 tests (SYNTHETIC; DEV data only).

DEV rows come from ``generator.dev_rows`` (split name "dev", seed derived from
"oac-ops-health-v2:20260926:dev"); no registered split is ever read here, and nothing under
v2/data/sealed/ is parsed.  Fixed DEV layout:
    [0:4096]        train
    [4096:6144]     calibration
    [6144:10240]    conformal (4,096 rows, like the registered conformal split)
    [10240:12288]   threshold
    [12288:32288]   evaluation (20,000 rows)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from v2.research import freeze, generator, kernel_bridge, pipeline, registry

ROOT = registry.ROOT
DEV_ROWS = 32288
TRAIN = slice(0, 4096)
CALIBRATION = slice(4096, 6144)
CONFORMAL = slice(6144, 10240)
THRESHOLD = slice(10240, 12288)
EVAL = slice(12288, 32288)
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SELECTION = {
    "procedure": "stage-1 unit test (DEV data; not the registered search)",
    "selection_split": "dev",
    "trials": 0,
    "selected_trial": None,
    "stop_reason": None,
    "search_trace_sha256": None,
    "receipts_dir_sha256": None,
    "amendments_sha256": None,
    "conformal_keep": None,
}


ORIGIN_SKIP_REASON = ("receipt source commit 92872f88 is in the unpublished research repository; "
                      "origin layer UNAVAILABLE here")


@lru_cache(maxsize=1)
def origin_commit_present() -> bool:
    """True when the frozen v2 receipt's source commit and its parent resolve in ROOT's git.

    The origin layer (``freeze.verify_origin``, the parent-commit receipt mutant) needs that
    commit.  It exists only in the unpublished research repository; in any other checkout (for
    example szl-forge's ops-health/) the tests that need it are skipped with ORIGIN_SKIP_REASON
    instead of failing.  Read-only git; TEST ONLY.
    """
    receipt = json.loads((registry.V2 / "ops-health" / freeze.RECEIPT_NAME).read_text(encoding="utf-8"))
    commit = receipt["source"]["commit"]
    for ref in (f"{commit}^{{commit}}", f"{commit}^^{{commit}}"):
        try:
            done = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", ref],
                                  capture_output=True, check=False)
        except OSError:
            return False
        if done.returncode != 0:
            return False
    return True


@lru_cache(maxsize=1)
def dev_rows() -> tuple:
    return tuple(generator.dev_rows(DEV_ROWS))


def kernel():
    return kernel_bridge.kernel()


def build(l2: float = 1e-3, calibrator: str = "platt", rows: tuple | None = None):
    """(artifact, diagnostics) of one DEV candidate, rebuilt from scratch on every call."""
    dev = list(rows) if rows is not None else list(dev_rows())
    return pipeline.build_candidate(
        train_rows=dev[TRAIN],
        calibration_rows=dev[CALIBRATION],
        conformal_rows=dev[CONFORMAL],
        threshold_rows=dev[THRESHOLD],
        l2=l2,
        calibrator=calibrator,
        split_names={role: f"dev-{role}" for role in pipeline.SPLIT_ROLES},
    )


@lru_cache(maxsize=8)
def _cached(l2: float, calibrator: str) -> str:
    return json.dumps(build(l2, calibrator)[0])


def candidate(l2: float = 1e-3, calibrator: str = "platt") -> dict:
    """A fresh copy of the cached DEV candidate artifact."""
    return json.loads(_cached(l2, calibrator))


class TempDir:
    """Context manager for a temporary directory (removed on exit)."""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="oac_v2_test_"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def freeze_dev(out_dir: Path, artifact: dict | None = None, *, commit: str = FAKE_COMMIT,
               tree_clean: bool = True) -> dict:
    """Freeze a DEV candidate with the TEST-ONLY source-state override."""
    return freeze.freeze(
        artifact if artifact is not None else candidate(),
        out_dir,
        selection=SELECTION,
        source_state_override={"commit": commit, "tree_clean": tree_clean},
    )


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    """Source mutation helper: the target must occur exactly once."""
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"mutation target occurs {count} times: {old!r}")
    return text.replace(old, new)


_MUTANT_COUNTER = [0]


def load_kernel_copy(directory: Path, source: str | None = None, *, mutate=None):
    """Write the kernel source (optionally mutated) to directory/ops_health.py and load it."""
    text = source if source is not None else registry.KERNEL_PATH.read_text(encoding="utf-8")
    if mutate is not None:
        text = mutate(text)
    _MUTANT_COUNTER[0] += 1
    sub = Path(directory) / f"kernel_copy_{_MUTANT_COUNTER[0]}"
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / "ops_health.py"
    path.write_bytes(text.encode("utf-8"))
    return kernel_bridge.load(path), path


def fail_open_mutation(text: str) -> str:
    """A deliberately broken, fail-open kernel: no screening, no schema, finiteness or range
    checks (missing fields default to 0, extras are ignored)."""
    text = replace_once(
        text,
        "            match = prohibited_match(key)\n",
        "            match = None\n",
    )
    text = replace_once(
        text,
        "    elif isinstance(value, (str, bytes, bytearray)):\n"
        "        raise ModelInputError(\n"
        "            f\"text values are not accepted at {_shown_path(path)}; input is numeric only\"\n"
        "        )\n",
        "",
    )
    text = replace_once(
        text,
        "    if depth > MAX_INPUT_DEPTH:\n",
        "    if False:\n",
    )
    text = replace_once(text, "    if keys != expected:\n", "    if False:\n")
    text = replace_once(
        text,
        "        _normalize_value(name, minimum, maximum, kind, features[name])\n",
        "        _normalize_value(name, minimum, maximum, kind, features.get(name, 0))\n",
    )
    text = replace_once(
        text,
        "def _normalize_value(name: str, minimum: float, maximum: float, kind: str, value: Any) -> float:\n",
        "def _normalize_value(name: str, minimum: float, maximum: float, kind: str, value: Any) -> float:\n"
        "    try:\n"
        "        number = float(value) if not isinstance(value, (list, dict, type(None))) else 0.0\n"
        "    except (TypeError, ValueError, OverflowError):\n"
        "        number = 0.0\n"
        "    if number != number or number in (float('inf'), float('-inf')):\n"
        "        number = 0.0\n"
        "    return min(max((number - minimum) / (maximum - minimum), 0.0), 1.0)\n",
    )
    text = replace_once(
        text,
        "    if set(payload) != {\"features\"}:\n",
        "    if False:\n",
    )
    text = replace_once(
        text,
        "    if \"features\" not in payload:\n"
        "        raise ModelInputError('input must be wrapped as {\"features\": {...}}')\n",
        "    if \"features\" not in payload:\n"
        "        payload = {\"features\": payload}\n",
    )
    return text
