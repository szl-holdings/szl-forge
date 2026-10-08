"""The single final opening of the sealed splits, and per-row scores of every frozen contender.

PREREGISTRATION sections 4, 5, 7 and 8; AMENDMENTS AM-5 and AM-6.  SYNTHETIC.  Standard library.

This module is the only research module besides the guard that names the final purpose (see
v2/tests/test_sealed_guard.py).  It writes scores only; every metric is computed afterwards by
v2/research/final_analyze.py from the saved score files, which never re-opens anything.

Real run (``--final --confirm FINAL_TEST_OPENING``), one process, in this order:
  1. Pre-opening checks.  Any failure refuses before anything is opened or written:
     runs/TEST_OPENED.json absent; v2/results/final absent or empty; ``git status --porcelain
     --untracked-files=all -- v2 runs`` empty; free disk >= 2 GiB; available commit memory
     >= 1 GiB (Windows); Python 3.12; sha256 of PREREGISTRATION.md, AMENDMENTS.md, the data
     MANIFEST, legacy_v1/test.jsonl, contenders/INDEX.json and every frozen contender file;
     mutation_readiness.json present; the v2 and ABL-data pairs load in the v2 kernel and pass
     ``freeze.verify_origin``; the v1 Hub kernel loads its pair (receipt check) with threshold
     0.16; v1 on legacy_v1_test reproduces its published confusion (tp 40, fp 57, tn 132,
     fn 11; REPORTED from the v1 receipt); v2 on example_input.json gives NO_ALERT, score
     0.069311; the section 10 validity inputs (receipt check, fail-closed probes) of every
     contender.
  2. ONE call ``sealed_guard.open_split("sealed", "FINAL_TEST_OPENING", contenders=...)``.  The
     guard verifies each sealed file's sha256, creates runs/TEST_OPENED.json exclusively (utc,
     HEAD, contender file hashes, sealed digests) and returns test plus the four shift splits.
  3. For each sealed split in registry order: score every row with every contender and write
     v2/results/final/scores_<split>.jsonl at once (temporary name, fsync, rename, re-read and
     hash).  Then scores_legacy_v1_test.jsonl (rows of v2/data/legacy_v1/test.jsonl, scored in
     step 1).  Then MANIFEST.json: sha256, bytes, rows and positives of every score file, the
     marker's sha256 and content, the checks, the contender files and the validity inputs.
     A disk-full error during any of these writes removes the partial file, waits 30 s and
     retries (up to 20 times; the bytes stay in memory).  Any other exception after the
     opening writes INCIDENT.json next to the score files and re-raises.

Contenders (all frozen before the opening; ABL-calibration is NOT_APPLICABLE and not scored):
  v2             v2/ops-health pair in its kernel: unrounded calibrated score (kernel ``evaluate``
                 after the kernel's own input screen), threshold decision (score >= 0.20),
                 Mondrian conformal advisory, abstain reason and prediction set.
  v1             Hub kernel hub/oac-v1 @ dd7d1098 (``OperationalHealthKernel.score``):
                 operator_attention_score exactly as emitted (12 decimals, AM-5) and
                 operator_attention_required (unrounded score >= 0.16).
  majority       always NO_ALERT: decision 0, no score.
  rule           ALERT iff consecutive_failures > 0: no score.
  ABL-data       the ABL-data pair in the v2 kernel: calibrated score and threshold decision.
  ABL-optimizer  params.json (v1 GD on the full train split): sigmoid(intercept + w.x) in kernel
                 order (``stage2.v1_kernel_order_scores``), decision score >= 0.17.

Score record (one line per row; json.dumps(sort_keys=True, separators=(",", ":"))):
  {"split", "sample_id", "regime", "label" (0/1), "features" (the eight observed values),
   "v2": {"score", "decision", "advisory", "abstain_reason", "prediction_set"},
   "v1": {"score", "decision"}, "majority": {"decision"}, "rule": {"decision"},
   "ABL-data": {"score", "decision"}, "ABL-optimizer": {"score", "decision"}}

Other modes (neither opens a sealed file nor writes under v2/ or runs/):
  --preflight              every step-1 check with the real paths, plus a parity check: v2,
                           ABL-data and ABL-optimizer re-scored on the registered validation
                           split (purpose ABLATION) must reproduce their recorded confusions.
  --dry-run --out DIR      DEV stand-in: generates DEV rows for split names "dev-standin-<split>"
                           (never a registered split) in DIR/standin/data/sealed/, a stand-in
                           guard (the real SealedGuard class bound to DIR/standin/data and
                           DIR/standin/runs) opens them, and the scores go to DIR/final.  Tree,
                           disk and memory checks are recorded but not fatal.

Usage (repository root, Git Bash):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.final_score --preflight
    PYTHONUTF8=1 py -3.12 -B -m v2.research.final_score --dry-run --out <dir outside v2 and runs>
    PYTHONUTF8=1 py -3.12 -B -m v2.research.final_score --final --confirm FINAL_TEST_OPENING
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from . import freeze, gates, generator, kernel_bridge, metrics, registry, sealed_guard, stage2
from .dataio import load_rows

FINAL_PURPOSE = sealed_guard.FINAL_TEST_OPENING
SEALED_SET = sealed_guard.SEALED_SET_NAME
SPLITS = registry.SEALED_SPLITS  # test + the four shift splits, registry order
POOLED_SHIFT = tuple(name for name in SPLITS if name != "test")
LEGACY_SPLIT = "legacy_v1_test"
LEGACY_REGIME = "legacy_v1"
LEGACY_REL = "v2/data/legacy_v1/test.jsonl"

OUT_DIR = registry.V2 / "results" / "final"
MANIFEST_NAME = "MANIFEST.json"
INCIDENT_NAME = "INCIDENT.json"
RECORD_SCHEMA = "szl-oac/ops-health-final-score-row/v2"
MANIFEST_SCHEMA = "szl-oac/ops-health-final-scores-manifest/v2"
SCRIPT_REL = "v2/research/final_score.py"

CONTENDERS = ("v2", "v1", "majority", "rule", "ABL-data", "ABL-optimizer")
SCORED_CONTENDERS = ("v2", "v1", "ABL-data", "ABL-optimizer")

# ---- pinned digests (REPORTED from the stage-2 and mutation reports; re-hashed at run time) ----
AMENDMENTS_REL = "v2/AMENDMENTS.md"
AMENDMENTS_SHA256 = "5152cd7f816c41e810861bbcd6489dee1d2598da89a36f6b418ef4a9867f181d"
PREREG_REL = "v2/PREREGISTRATION.md"
DATA_MANIFEST_REL = "v2/data/MANIFEST.json"
DATA_MANIFEST_SHA256 = "c0d2fa0a924162e42c4d50c8eaf7d155f3e991f8f47ca518b1b87c3927ad7252"
LEGACY_SHA256 = "7fc6ea635fcc501d56eed6a8b9e4f5f3a9057d0aa04505502759197150d79679"
INDEX_REL = "v2/results/contenders/INDEX.json"
INDEX_SHA256 = "086dbca90c69b9a2026bdedeb9fa9657f51af036884d79a839707c64a54b5718"
MUTATION_READINESS_REL = "v2/results/mutation_readiness.json"
SEARCH_TRACE_REL = "v2/results/search_trace.json"
V2_MODEL_REL = "v2/ops-health/model.json"
V2_RECEIPT_REL = "v2/ops-health/artifact_receipt.json"
V2_KERNEL_REL = "v2/ops-health/ops_health.py"
V2_EXAMPLE_REL = "v2/ops-health/example_input.json"
V2_FILES = {
    V2_MODEL_REL: "b830a5edca271d667ab09b378dd5d3ab505d71a7e3451de8d9ca2bacfeed977c",
    V2_RECEIPT_REL: "442486a3b451f0ad765aac253830cdc5455bad3a34186b4cf73388cf207056f2",
    V2_KERNEL_REL: "b0a64ff3f26ea284b0588de351ed7795a6089203b35fded0e7d82cbd4871aed9",
}
V1_MODEL_REL = "hub/oac-v1/model.json"
V1_RECEIPT_REL = "hub/oac-v1/artifact_receipt.json"
V1_KERNEL_REL = "hub/oac-v1/oac_operational_health.py"
V1_FILES = {  # MEASURED byte-identical to the Hub revision in the v1 audit (audit/v1_bytecompare.json)
    V1_MODEL_REL: "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557",
    V1_RECEIPT_REL: "23239bb39b0a5b6db1e41284e6372f352cee61d7938da1e84472246dcdaf95a1",
    V1_KERNEL_REL: "2fd8ffd3fd3ca8d0f960d4e0d471593b86e6cccb03607116390cbe9300992669",
}
ABL_DATA_MODEL_REL = "v2/results/contenders/ABL-data/model.json"
ABL_DATA_RECEIPT_REL = "v2/results/contenders/ABL-data/artifact_receipt.json"
ABL_DATA_FILES = {
    ABL_DATA_MODEL_REL: "0cbe0e4a4d76184c6a53945a7ddfe8ab1ec16318e959296caafd5a0415d5d598",
    ABL_DATA_RECEIPT_REL: "876bde74b7289e0d7b0edcd4f720c4d1766777c857d14356a88bd3d43c67fa23",
}
ABL_OPT_PARAMS_REL = "v2/results/contenders/ABL-optimizer/params.json"
ABL_OPT_FILES = {
    ABL_OPT_PARAMS_REL: "3a89941b066db919b10b608b3537ce4044872b2314761a732b7e828a975e4a1f",
}
CONTENDER_FILES = {
    "v2": V2_FILES,
    "v1": V1_FILES,
    "majority": {SCRIPT_REL: None},  # defined in this file (hashed at run time)
    "rule": {SCRIPT_REL: None},
    "ABL-data": ABL_DATA_FILES,
    "ABL-optimizer": ABL_OPT_FILES,
}
V1_THRESHOLD = 0.16
# REPORTED: the v1 receipt's metrics.test confusion (v2/tests/test_metrics.py V1_TEST_COUNTS).
V1_LEGACY_CONFUSION = {"tp": 40, "fp": 57, "tn": 132, "fn": 11}
V2_EXAMPLE_ADVISORY = "NO_ALERT"
V2_EXAMPLE_SCORE_6DP = 0.069311
MIN_FREE_DISK_BYTES = 2 * 1024 ** 3
MIN_AVAILABLE_COMMIT_BYTES = 1 * 1024 ** 3
STRICT_ONLY = ("tree_clean_v2_runs", "disk_free", "memory_available")
V1_MODULE_NAME = "oac_v1_hub_kernel_final"


class FinalScoreRefused(RuntimeError):
    """A pre-opening check failed; nothing was opened and nothing was written."""


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _root_path(rel: str) -> Path:
    return registry.ROOT / rel


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(registry.ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def available_commit_bytes() -> int | None:
    """Available commit charge (physical + page file) on Windows; None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes  # noqa: PLC0415

    class _Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = _Status()
    status.dwLength = ctypes.sizeof(_Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPageFile)


def record_line(record: Mapping) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def load_v1_module(path: Path) -> ModuleType:
    """Load the Hub kernel file by path under a private module name (never imported by name)."""
    if V1_MODULE_NAME in sys.modules and getattr(sys.modules[V1_MODULE_NAME], "__file__", None) \
            == str(path):
        return sys.modules[V1_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(V1_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the v1 kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[V1_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# the frozen contenders
# --------------------------------------------------------------------------------------
class FrozenContenders:
    """Loads every frozen contender with its own integrity layer and scores rows."""

    def __init__(self) -> None:
        K = kernel_bridge.kernel()
        self.K = K
        self.v2 = K.OpsHealthKernel(_root_path(V2_MODEL_REL), _root_path(V2_RECEIPT_REL))
        self.abl_data = K.OpsHealthKernel(_root_path(ABL_DATA_MODEL_REL),
                                          _root_path(ABL_DATA_RECEIPT_REL))
        params = json.loads(_root_path(ABL_OPT_PARAMS_REL).read_text(encoding="utf-8"))
        if tuple(params["feature_order"]) != tuple(K.FEATURE_NAMES):
            raise ValueError("ABL-optimizer feature order differs from the kernel's")
        if params.get("calibrator") != "none" or params.get("conformal") != "none":
            raise ValueError("ABL-optimizer params declare a calibrator or conformal layer")
        self.opt_intercept = float(params["intercept"])
        self.opt_weights = [float(params["weights"][name]) for name in K.FEATURE_NAMES]
        self.opt_threshold = float(params["decision_threshold"])
        self.v1_module = load_v1_module(_root_path(V1_KERNEL_REL))
        self.v1 = self.v1_module.OperationalHealthKernel(_root_path(V1_MODEL_REL),
                                                         _root_path(V1_RECEIPT_REL))
        self.v1_threshold = float(self.v1._artifact["decision_threshold"])  # noqa: SLF001
        self.v2_threshold = float(self.v2.artifact["decision_threshold"])
        self.abl_data_threshold = float(self.abl_data.artifact["decision_threshold"])

    # -- per-contender entry points --------------------------------------------------------
    def normalize(self, features: Mapping) -> list[float]:
        """The v2 kernel's input path: whole-payload screen, wrapper check, exact schema."""
        return self.K.normalize_features(self.K.unwrap_payload({"features": features}))

    def opt_score_vectors(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        return stage2.v1_kernel_order_scores(self.opt_intercept, self.opt_weights, vectors)

    def v1_cli_advise(self, payload: Any) -> dict:
        """v1's CLI input path (JSON text through ``_read_cli_input``) then ``score``."""
        text = json.dumps(payload)
        saved = sys.stdin
        sys.stdin = io.StringIO(text)
        try:
            parsed = self.v1_module._read_cli_input("-")  # noqa: SLF001
        finally:
            sys.stdin = saved
        return self.v1.score(parsed)

    def opt_advise(self, payload: Any) -> float:
        vector = self.K.normalize_features(self.K.unwrap_payload(payload))
        return self.opt_score_vectors([vector])[0]

    @staticmethod
    def majority_advise(payload: Any) -> dict:  # noqa: ARG004
        return {"advisory": "NO_ALERT"}

    @staticmethod
    def rule_advise(payload: Any) -> dict:
        """The one-line rule applied to the wrapped payload, with no input screen of its own."""
        return {"decision": 1 if payload["features"]["consecutive_failures"] > 0 else 0}

    # -- scoring ---------------------------------------------------------------------------
    def score_rows(self, split: str, rows: Sequence[Mapping], *,
                   default_regime: str | None = None) -> list[dict]:
        names = self.K.FEATURE_NAMES
        vectors = [self.normalize(row["features"]) for row in rows]
        opt_scores = self.opt_score_vectors(vectors)
        out = []
        for row, vector, s_opt in zip(rows, vectors, opt_scores, strict=True):
            features = row["features"]
            label = row["label"]["operator_attention_required"]
            if not isinstance(label, bool):
                raise ValueError(f"{split}: non-boolean label at {row.get('sample_id')!r}")
            e2 = self.v2.evaluate(vector)
            ea = self.abl_data.evaluate(vector)
            a1 = self.v1.score(features)
            out.append({
                "split": split,
                "sample_id": row["sample_id"],
                "regime": row.get("regime", default_regime),
                "label": 1 if label else 0,
                "features": {name: features[name] for name in names},
                "v2": {
                    "score": e2["calibrated_score"],
                    "decision": 1 if e2["threshold_decision"] else 0,
                    "advisory": e2["advisory"],
                    "abstain_reason": e2["abstain_reason"],
                    "prediction_set": list(e2["prediction_set"]),
                },
                "v1": {
                    "score": a1["operator_attention_score"],
                    "decision": 1 if a1["operator_attention_required"] else 0,
                },
                "majority": {"decision": 0},
                "rule": {"decision": 1 if features["consecutive_failures"] > 0 else 0},
                "ABL-data": {
                    "score": ea["calibrated_score"],
                    "decision": 1 if ea["threshold_decision"] else 0,
                },
                "ABL-optimizer": {
                    "score": s_opt,
                    "decision": 1 if s_opt >= self.opt_threshold else 0,
                },
            })
        return out

    # -- section 10 validity inputs ----------------------------------------------------------
    def validity_inputs(self, receipt_checks: Mapping[str, Mapping]) -> dict:
        def probes(advise: Callable[[Any], Any], method: str) -> dict:
            report = gates.run_failclosed_probes(advise)
            return {"refused": report["refused"], "total": report["total"],
                    "escaped": list(report["escaped"]), "method": method}

        return {
            "v2": {**receipt_checks["v2"], "probes": probes(
                self.v2.advise, "v2 kernel OpsHealthKernel.advise on each gates.FAILCLOSED_PROBES "
                "payload")},
            "v1": {**receipt_checks["v1"], "probes": probes(
                self.v1_cli_advise, "v1 Hub kernel CLI input path: json.dumps(payload) read by "
                "_read_cli_input('-'), then OperationalHealthKernel.score")},
            "majority": {"receipt_verified": False,
                         "receipt_check": "no artifact, no receipt, no kernel",
                         "probes": probes(self.majority_advise,
                                          "constant NO_ALERT callable (as in mutation.py)")},
            "rule": {"receipt_verified": False,
                     "receipt_check": "no artifact, no receipt, no kernel",
                     "probes": probes(self.rule_advise,
                                      "the one-line rule on the wrapped payload, no input screen")},
            "ABL-data": {**receipt_checks["ABL-data"], "probes": probes(
                self.abl_data.advise, "v2 kernel OpsHealthKernel.advise with the ABL-data pair")},
            "ABL-optimizer": {"receipt_verified": False,
                              "receipt_check": "params file bound by sha256 in contenders/INDEX.json; "
                                               "it has no artifact receipt and no kernel",
                              "probes": probes(self.opt_advise,
                                               "v2 kernel input path (unwrap_payload, "
                                               "normalize_features) then the params arithmetic")},
        }


# --------------------------------------------------------------------------------------
# pre-opening checks
# --------------------------------------------------------------------------------------
def confusion_of(records: Sequence[Mapping], contender: str) -> dict[str, int]:
    return metrics.confusion([r["label"] for r in records],
                             [r[contender]["decision"] for r in records])


def pre_opening_checks(*, marker: Path, out_dir: Path, strict: bool) -> dict:
    """Every check that can run without a sealed split.  Returns
    {"checks", "failed", "contenders", "legacy_records", "validity_inputs"}."""
    checks: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    def guarded(name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 -- recorded as a failed check
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:300]}"
        record(name, ok, detail)

    record("cwd_is_repository_root", Path.cwd().resolve() == registry.ROOT.resolve(),
           "the marker records contender paths relative to the repository root")
    record("marker_absent", not marker.exists(), marker.name)
    record("output_dir_empty", not out_dir.exists() or not any(out_dir.iterdir()), out_dir.name)

    def tree() -> tuple[bool, Any]:
        status = _git("status", "--porcelain", "--untracked-files=all", "--", "v2", "runs")
        lines = [line for line in status.splitlines() if line.strip()]
        return not lines, {"porcelain_lines": len(lines), "first": lines[:5]}

    guarded("tree_clean_v2_runs", tree)
    guarded("git_head", lambda: (True, _git("rev-parse", "HEAD").strip()))
    free = shutil.disk_usage(registry.ROOT).free
    record("disk_free", free >= MIN_FREE_DISK_BYTES,
           {"free_bytes": free, "minimum_bytes": MIN_FREE_DISK_BYTES})
    commit = available_commit_bytes()
    record("memory_available", commit is None or commit >= MIN_AVAILABLE_COMMIT_BYTES,
           {"available_commit_bytes": commit, "minimum_bytes": MIN_AVAILABLE_COMMIT_BYTES})
    record("python_3_12", sys.version_info[:2] == (3, 12), platform.python_version())

    def digest(rel: str, expected: str) -> Callable[[], tuple[bool, Any]]:
        return lambda: ((got := sha256_file(_root_path(rel))) == expected, {"path": rel, "sha256": got})

    guarded("preregistration_sha256", digest(PREREG_REL, registry.PREREG_SHA256))
    guarded("amendments_sha256", digest(AMENDMENTS_REL, AMENDMENTS_SHA256))
    guarded("data_manifest_sha256", digest(DATA_MANIFEST_REL, DATA_MANIFEST_SHA256))
    guarded("legacy_v1_test_sha256", digest(LEGACY_REL, LEGACY_SHA256))
    guarded("contenders_index_sha256", digest(INDEX_REL, INDEX_SHA256))
    record("mutation_readiness_present", _root_path(MUTATION_READINESS_REL).is_file(),
           MUTATION_READINESS_REL)
    for contender in ("v2", "v1", "ABL-data", "ABL-optimizer"):
        for rel, expected in CONTENDER_FILES[contender].items():
            guarded(f"file_sha256:{rel}", digest(rel, expected))

    def index_consistent() -> tuple[bool, Any]:
        index = json.loads(_root_path(INDEX_REL).read_text(encoding="utf-8"))["contenders"]
        detail = {
            "ABL-data_files": index["ABL-data"]["files"] == ABL_DATA_FILES,
            "ABL-optimizer_files": index["ABL-optimizer"]["files"] == ABL_OPT_FILES,
            "ABL-calibration_not_applicable": index["ABL-calibration"]["status"] == "NOT_APPLICABLE",
            "ABL-conformal_keep": index["ABL-conformal"]["keep"] is True,
        }
        return all(detail.values()), detail

    guarded("contenders_index_consistent", index_consistent)

    contenders = None
    try:
        contenders = FrozenContenders()
        record("contenders_load", True, "v2 and ABL-data pairs accepted by the v2 kernel; v1 pair "
                                        "accepted by the Hub kernel")
    except Exception as exc:  # noqa: BLE001
        record("contenders_load", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    receipt_checks: dict[str, dict] = {}
    legacy_records: list[dict] = []
    validity: dict = {}
    if contenders is not None:
        c = contenders
        for name, kernel_obj in (("v2", c.v2), ("ABL-data", c.abl_data)):
            try:
                freeze.verify_origin(kernel_obj.receipt)
                receipt_checks[name] = {"receipt_verified": True, "receipt_check":
                                        "v2 kernel OpsHealthKernel (receipt, kernel self-hash, model "
                                        "hash) and freeze.verify_origin (committed blobs)"}
            except Exception as exc:  # noqa: BLE001
                receipt_checks[name] = {"receipt_verified": False,
                                        "receipt_check": f"{type(exc).__name__}: {str(exc)[:300]}"}
            record(f"receipt_verified:{name}", receipt_checks[name]["receipt_verified"],
                   receipt_checks[name]["receipt_check"])
        receipt_checks["v1"] = {"receipt_verified": True, "receipt_check":
                                "Hub kernel OperationalHealthKernel accepted its receipt (schema, "
                                "purpose, model sha256); files pinned by sha256"}
        record("v1_threshold", c.v1_threshold == V1_THRESHOLD, c.v1_threshold)
        record("thresholds", True, {"v2": c.v2_threshold, "v1": c.v1_threshold,
                                    "ABL-data": c.abl_data_threshold,
                                    "ABL-optimizer": c.opt_threshold})

        def legacy() -> tuple[bool, Any]:
            rows = load_rows(_root_path(LEGACY_REL))
            legacy_records.extend(c.score_rows(LEGACY_SPLIT, rows, default_regime=LEGACY_REGIME))
            got = confusion_of(legacy_records, "v1")
            return got == V1_LEGACY_CONFUSION, {"rows": len(rows), "v1_confusion": got,
                                                "published": V1_LEGACY_CONFUSION}

        guarded("v1_reproduces_published_legacy_test_confusion", legacy)

        def example() -> tuple[bool, Any]:
            payload = json.loads(_root_path(V2_EXAMPLE_REL).read_text(encoding="utf-8"))
            advice = c.v2.advise(payload)
            got = (advice["advisory"], round(advice["operator_attention_score"], 6))
            return got == (V2_EXAMPLE_ADVISORY, V2_EXAMPLE_SCORE_6DP), list(got)

        guarded("v2_example_input", example)
        try:
            validity = c.validity_inputs(receipt_checks)
            record("validity_inputs", True, {k: {"receipt_verified": v["receipt_verified"],
                                                 "probes_refused": v["probes"]["refused"],
                                                 "probes_total": v["probes"]["total"]}
                                             for k, v in validity.items()})
        except Exception as exc:  # noqa: BLE001
            record("validity_inputs", False, f"{type(exc).__name__}: {str(exc)[:300]}")

    failed = [name for name, chk in checks.items()
              if not chk["ok"] and (strict or name not in STRICT_ONLY)]
    return {"checks": checks, "failed": failed, "contenders": contenders,
            "legacy_records": legacy_records, "validity_inputs": validity}


# --------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------
DISK_FULL_RETRIES = 20
DISK_FULL_WAIT_SECONDS = 30
_sleep = time.sleep


def _is_disk_full(exc: OSError) -> bool:
    return exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112  # ERROR_DISK_FULL


def write_bytes_atomic(final: Path, data: bytes, *, exclusive: bool = False,
                       log: Callable[[str], None] = print) -> str:
    """Write via <name>.partial, fsync, rename, then re-read and compare.  A disk-full error
    (the host disk has been intermittently full) removes the partial file, waits
    DISK_FULL_WAIT_SECONDS and retries, up to DISK_FULL_RETRIES times; the bytes stay in memory,
    so a transient full disk delays the write instead of losing it.  Returns the sha256."""
    if exclusive and final.exists():
        raise FileExistsError(f"{final.name} already exists; refusing to overwrite")
    partial = final.with_name(final.name + ".partial")
    for attempt in range(DISK_FULL_RETRIES + 1):
        try:
            with open(partial, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, final)
            break
        except OSError as exc:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            if not _is_disk_full(exc) or attempt == DISK_FULL_RETRIES:
                raise
            log(f"disk full while writing {final.name}; retry {attempt + 1}/{DISK_FULL_RETRIES} "
                f"in {DISK_FULL_WAIT_SECONDS} s")
            _sleep(DISK_FULL_WAIT_SECONDS)
    reread = final.read_bytes()
    digest = sha256_bytes(data)
    if len(reread) != len(data) or sha256_bytes(reread) != digest:
        raise OSError(f"{final.name}: re-read bytes differ from the bytes written")
    return digest


def write_score_file(out_dir: Path, split: str, records: Sequence[Mapping], *,
                     log: Callable[[str], None] = print) -> dict:
    """Write scores_<split>.jsonl (write_bytes_atomic, exclusive) and describe it."""
    data = b"".join(record_line(r) for r in records)
    final = out_dir / f"scores_{split}.jsonl"
    digest = write_bytes_atomic(final, data, exclusive=True, log=log)
    return {
        "file": final.name,
        "sha256": digest,
        "bytes": len(data),
        "rows": len(records),
        "positives": sum(r["label"] for r in records),
        "regimes": sorted({r["regime"] for r in records}),
    }


def _write_json_exclusive(path: Path, value: Any, *, log: Callable[[str], None] = print) -> str:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return write_bytes_atomic(path, data, exclusive=True, log=log)


def marker_contenders() -> dict[str, list[Path]]:
    """Contender files hashed into the marker (paths relative to the repository root)."""
    return {name: [Path(rel) for rel in files] for name, files in CONTENDER_FILES.items()}


def contender_block(contenders: FrozenContenders) -> dict:
    files = {name: {rel: (sha if sha is not None else sha256_file(_root_path(rel)))
                    for rel, sha in rels.items()} for name, rels in CONTENDER_FILES.items()}
    index = json.loads(_root_path(INDEX_REL).read_text(encoding="utf-8"))["contenders"]
    return {
        "v2": {"kind": "v2_artifact", "files": files["v2"],
               "decision_threshold": contenders.v2_threshold,
               "score": "calibrated_score from the v2 kernel's evaluate (unrounded); decision = "
                        "score >= decision_threshold; conformal advisory, abstain reason and set",
               "selected_trial": "T07_l2_1e-3_none"},
        "v1": {"kind": "hub_kernel", "hub_revision": registry.V1_HUB_REVISION, "files": files["v1"],
               "decision_threshold": contenders.v1_threshold,
               "score": "operator_attention_score exactly as the Hub kernel emits it (12 decimals; "
                        "AM-5, no recalibration); decision = operator_attention_required"},
        "majority": {"kind": "baseline", "files": files["majority"],
                     "definition": "always NO_ALERT (decision 0); no score"},
        "rule": {"kind": "baseline", "files": files["rule"],
                 "definition": "ALERT iff consecutive_failures > 0; no score"},
        "ABL-data": {"kind": "v2_artifact", "files": files["ABL-data"],
                     "decision_threshold": contenders.abl_data_threshold,
                     "score": "calibrated_score from the v2 kernel's evaluate with the ABL-data pair"},
        "ABL-optimizer": {"kind": "params_file", "files": files["ABL-optimizer"],
                          "decision_threshold": contenders.opt_threshold,
                          "score": "stage2.v1_kernel_order_scores with the params file's intercept "
                                   "and weights; decision = score >= decision_threshold"},
        "ABL-calibration": {"kind": "not_applicable", "status": index["ABL-calibration"]["status"],
                            "reason": index["ABL-calibration"]["reason"]},
        "ABL-conformal": {"kind": "decision", "status": index["ABL-conformal"]["status"],
                          "keep": index["ABL-conformal"]["keep"],
                          "note": "evaluated from v2's own advisory fields (with and without "
                                  "abstention)"},
    }


RECORD_FORMAT = {
    "encoding": "one JSON object per line: json.dumps(record, sort_keys=True, "
                "separators=(',', ':'), allow_nan=False) + '\\n', utf-8",
    "fields": {
        "split": "split name",
        "sample_id": "row id from the data file",
        "regime": "data regime (legacy rows: legacy_v1)",
        "label": "1 iff operator_attention_required",
        "features": "the eight observed feature values as stored in the data file",
        "v2": "score, decision, advisory, abstain_reason, prediction_set",
        "v1": "score (as emitted, 12 decimals), decision",
        "majority": "decision",
        "rule": "decision",
        "ABL-data": "score, decision",
        "ABL-optimizer": "score, decision",
    },
    "schema": RECORD_SCHEMA,
}


# --------------------------------------------------------------------------------------
# the opening
# --------------------------------------------------------------------------------------
def run_opening(*, opener: Callable[..., Mapping[str, list]], marker: Path, out_dir: Path,
                strict: bool, mode: str, expected_rows: Mapping[str, int],
                log: Callable[[str], None] = print) -> dict:
    """Pre-opening checks, the single opening, per-split score files, then MANIFEST.json."""
    started = _utc()
    pre = pre_opening_checks(marker=marker, out_dir=out_dir, strict=strict)
    for name, chk in pre["checks"].items():
        log(f"check {name}: {'ok' if chk['ok'] else 'FAILED'} {json.dumps(chk['detail'], sort_keys=True)}")
    if pre["failed"]:
        raise FinalScoreRefused(f"pre-opening checks failed: {pre['failed']}; nothing was opened")
    contenders: FrozenContenders = pre["contenders"]
    script_sha = sha256_file(_root_path(SCRIPT_REL))
    head = pre["checks"]["git_head"]["detail"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"opening: {SEALED_SET} purpose {FINAL_PURPOSE} ({mode}) at {_utc()}")
    sealed = opener(SEALED_SET, FINAL_PURPOSE, contenders=marker_contenders())
    marker_bytes = marker.read_bytes()
    marker_record = json.loads(marker_bytes.decode("utf-8"))
    log(f"opened: marker {marker.name} sha256 {sha256_bytes(marker_bytes)}")

    written: dict[str, dict] = {}
    try:
        for split in SPLITS:
            rows = sealed[split]
            regime = registry.SPLIT_TABLE[split][0]
            records = contenders.score_rows(split, rows)
            entry = write_score_file(out_dir, split, records, log=log)
            entry["expected_rows"] = expected_rows[split]
            entry["expected_regime"] = regime
            entry["source_sha256"] = marker_record["sealed_splits_sha256"][split]
            written[split] = entry
            log(f"wrote {entry['file']}: rows {entry['rows']} positives {entry['positives']} "
                f"bytes {entry['bytes']} sha256 {entry['sha256']}")
            del records
            sealed[split] = None  # release the parsed rows
        entry = write_score_file(out_dir, LEGACY_SPLIT, pre["legacy_records"], log=log)
        entry["expected_rows"] = 240
        entry["expected_regime"] = LEGACY_REGIME
        entry["source_sha256"] = LEGACY_SHA256
        written[LEGACY_SPLIT] = entry
        log(f"wrote {entry['file']}: rows {entry['rows']} positives {entry['positives']} "
            f"bytes {entry['bytes']} sha256 {entry['sha256']}")
        consistency = {
            split: {"rows_as_registered": e["rows"] == e["expected_rows"],
                    "regime_as_registered": e["regimes"] == [e["expected_regime"]]}
            for split, e in written.items()
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "synthetic": True,
            "mode": mode,
            "truth_label": "MEASURED: written by v2/research/final_score.py in the process that "
                           "performed the opening; SYNTHETIC data; scores only (no metric here)",
            "started_utc": started,
            "completed_utc": _utc(),
            "opening": {
                "marker_file": marker.name,
                "marker_sha256": sha256_bytes(marker_bytes),
                "marker": marker_record,
            },
            "script": {"path": SCRIPT_REL, "sha256": script_sha, "git_head_at_run": head,
                       "python": platform.python_version(), "argv": sys.argv[1:]},
            "pre_opening_checks": pre["checks"],
            "contenders": contender_block(contenders),
            "validity_inputs": pre["validity_inputs"],
            "splits": list(SPLITS) + [LEGACY_SPLIT],
            "pooled_shift_suite": list(POOLED_SHIFT),
            "files": written,
            "consistency": consistency,
            "record_format": RECORD_FORMAT,
        }
        manifest_sha = _write_json_exclusive(out_dir / MANIFEST_NAME, manifest, log=log)
        log(f"wrote {MANIFEST_NAME}: sha256 {manifest_sha}")
        return {"manifest": manifest, "manifest_sha256": manifest_sha}
    except BaseException as exc:
        incident = {
            "what": "exception after the sealed opening; score files may be missing or incomplete",
            "mode": mode,
            "utc": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "traceback_tail": traceback.format_exc()[-4000:],
            "files_written": written,
        }
        try:
            _write_json_exclusive(out_dir / INCIDENT_NAME, incident, log=log)
        except Exception:  # noqa: BLE001 -- the original exception is what matters
            pass
        raise


# --------------------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------------------
def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def build_standin(base: Path, sizes: Mapping[str, int]) -> tuple[Path, Path]:
    """DEV stand-in data: split names 'dev-standin-<split>' (never registered), same regimes."""
    data_dir = base / "standin" / "data"
    runs_dir = base / "standin" / "runs"
    manifest = {"schema": "szl-oac/ops-health-dev-standin-manifest/v2", "synthetic": True,
                "dev_only": True, "splits": {}}
    for split in SPLITS:
        regime = registry.SPLIT_TABLE[split][0]
        rows = generator.generate_rows(f"dev-standin-{split}", sizes[split], regime)
        digest = generator.write_jsonl(registry.split_path(split, data_dir), rows)
        manifest["splits"][split] = {"sha256": digest, "rows": len(rows), "regime": regime,
                                     "generated_as": f"dev-standin-{split}"}
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                            encoding="utf-8")
    return data_dir, runs_dir


def dry_run(base: Path, *, test_rows: int | None = None, shift_rows: int | None = None,
            log: Callable[[str], None] = print) -> dict:
    base = Path(base)
    for forbidden in (registry.V2, registry.RUNS_DIR, registry.DATA_DIR):
        if _inside(base, forbidden):
            raise FinalScoreRefused(f"dry-run output must be outside {registry.relpath(forbidden)}")
    sizes = {split: registry.SPLIT_TABLE[split][1] for split in SPLITS}
    if test_rows is not None:
        sizes["test"] = test_rows
    if shift_rows is not None:
        for split in POOLED_SHIFT:
            sizes[split] = shift_rows
    data_dir, runs_dir = build_standin(base, sizes)
    guard = sealed_guard.SealedGuard(data_dir=data_dir, runs_dir=runs_dir, repo_root=registry.ROOT)
    return run_opening(opener=guard.open_split, marker=runs_dir / registry.TEST_OPENED_NAME,
                       out_dir=base / "final", strict=False, mode="DRY_RUN_DEV_STANDIN",
                       expected_rows=sizes, log=log)


def final_run(log: Callable[[str], None] = print) -> dict:
    sizes = {split: registry.SPLIT_TABLE[split][1] for split in SPLITS}
    return run_opening(opener=sealed_guard.open_split,
                       marker=registry.RUNS_DIR / registry.TEST_OPENED_NAME,
                       out_dir=OUT_DIR, strict=True, mode="FINAL", expected_rows=sizes, log=log)


def preflight(log: Callable[[str], None] = print) -> dict:
    """All pre-opening checks (strict) plus the validation parity check; opens nothing sealed."""
    pre = pre_opening_checks(marker=registry.RUNS_DIR / registry.TEST_OPENED_NAME,
                             out_dir=OUT_DIR, strict=True)
    checks = pre["checks"]
    contenders = pre["contenders"]
    parity: dict = {}
    if contenders is not None:
        try:
            rows = sealed_guard.open_split("validation", "ABLATION")
            records = contenders.score_rows("validation", rows)
            trace = json.loads(_root_path(SEARCH_TRACE_REL).read_text(encoding="utf-8"))
            selected = trace["selection"]["selected_trial"]
            t07 = next(t["record"] for t in trace["trials"] if t["record"]["trial_id"] == selected)
            index = json.loads(_root_path(INDEX_REL).read_text(encoding="utf-8"))["contenders"]
            expected = {"v2": t07["validation"]["confusion"],
                        "ABL-data": index["ABL-data"]["validation"]["confusion"],
                        "ABL-optimizer": index["ABL-optimizer"]["validation"]["confusion"]}
            for name, exp in expected.items():
                got = confusion_of(records, name)
                parity[name] = {"ok": got == exp, "got": got, "recorded": exp}
            checks["validation_parity"] = {"ok": all(p["ok"] for p in parity.values()),
                                           "detail": {"rows": len(rows), "purpose": "ABLATION",
                                                      **parity}}
        except Exception as exc:  # noqa: BLE001
            checks["validation_parity"] = {"ok": False,
                                           "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}
    for name, chk in checks.items():
        log(f"check {name}: {'ok' if chk['ok'] else 'FAILED'} {json.dumps(chk['detail'], sort_keys=True)}")
    failed = [name for name, chk in checks.items() if not chk["ok"]]
    log(f"preflight: {'PASS' if not failed else 'FAIL'} failed={failed}")
    return {"checks": checks, "failed": failed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, help="dry-run output directory (outside v2 and runs)")
    parser.add_argument("--test-rows", type=int, default=None, help="dry-run only")
    parser.add_argument("--shift-rows", type=int, default=None, help="dry-run only")
    parser.add_argument("--confirm", default="", help=f"--final requires {FINAL_PURPOSE}")
    args = parser.parse_args(argv)
    os.chdir(registry.ROOT)

    def log(line: str) -> None:
        print(line, flush=True)

    try:
        if args.preflight:
            return 0 if not preflight(log)["failed"] else 1
        if args.dry_run:
            if args.out is None:
                parser.error("--dry-run needs --out")
            result = dry_run(args.out, test_rows=args.test_rows, shift_rows=args.shift_rows, log=log)
            log(f"dry-run: done; manifest sha256 {result['manifest_sha256']}")
            return 0
        if args.confirm != FINAL_PURPOSE:
            parser.error(f"--final needs --confirm {FINAL_PURPOSE}")
        if args.test_rows is not None or args.shift_rows is not None:
            parser.error("--test-rows/--shift-rows are dry-run only")
        result = final_run(log)
        log(f"final: done; manifest sha256 {result['manifest_sha256']}")
        return 0
    except FinalScoreRefused as exc:
        log(f"REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
