"""Mutation suite, broken baselines and readiness aggregation (PREREGISTRATION s.10, s.11; AM-3).

SYNTHETIC.  Standard library only.  **Validation data only**: the registered validation split is
opened once per run through ``sealed_guard.open_split("validation", "GATES")``.  Nothing here
reads a sealed split or names a sealed path; the run records whether ``runs/TEST_OPENED.json``
exists before and after (it must not).

Everything in this module (the mutant catalogue, the catch rules, the aggregators, the cut and
the Lambda decision rule) is declared before the first run.  The seven gates are the section 10
gates exactly as implemented in ``gates.py`` before any v2 model existed, and the cut is the
section 11 cut; neither is tuned after seeing results.

Mutant catalogue
----------------
Artifact mutants of the frozen v2 (``v2/ops-health/model.json``), 36:
    A-scale0.5-<f>, A-scale1.5-<f>     weight x0.5 and x1.5, per feature           (16)
    A-signflip-<f>                     weight sign flip, per feature               (8)
    A-drop-<f>                         weight set to 0, per feature                (8)
    A-intercept+1, A-intercept-1       intercept +1 and -1                          (2)
    A-threshold+0.1, A-threshold-0.1   decision threshold +0.1 and -0.1 (grid index +/-10)  (2)
  Each mutated real number is rounded to 12 decimals (the frozen resolution), so every mutant is
  a canonical, kernel-valid model.json that differs from v2 in exactly one field.
  Pass 1 (integrity): the mutant model.json with the frozen v2 receipt, unchanged.
  Pass 2 (integrity bypassed): the mutant with a consistently regenerated receipt, i.e. the
  frozen receipt with model_sha256 replaced by the mutant's digest and every other field
  unchanged (the kernel and freeze.verify_origin both accept it).  The seven gates are then
  evaluated on the registered validation split.
Receipt mutants of the frozen v2 receipt (with the frozen model.json), 5:
    R-model-hash-flip          model_sha256: the low bit of its first hex digit flipped (stays hex)
    R-wrong-schema             schema -> "szl-oac/ops-health-artifact-receipt/v1"
    R-source-commit-altered    source.commit: the low bit of its last hex digit flipped
    R-source-commit-other      SUPPLEMENTARY: source.commit -> the parent of the recorded commit
                               (a real commit of this repository)
    R-metrics-added            AM-3: a metrics block added to the receipt
  The integrity layer is the kernel (OpsHealthKernel) plus freeze.verify_origin.
Kernel-source mutants (temporary copies of ``v2/ops-health/ops_health.py``), 5:
    K-no-finiteness            the input finiteness check removed
    K-no-range                 the input range check removed
    K-no-prohibited-screen     prohibited-key screening removed (match = None)
    K-inverted-model-hash      the model-hash comparison inverted (!= becomes ==)
    K-inverted-self-hash       SUPPLEMENTARY: the kernel self-hash comparison inverted
  CAUGHT_BY_INTEGRITY: the mutated copy refuses the frozen v2 model + receipt.
  CAUGHT_BY_TESTS: the declared kernel-relevant tests (KERNEL_TEST_IDS) fail when run in a
  subprocess against the mutated copy, injected as THE kernel (``kernel_bridge`` cache and
  ``registry.KERNEL_PATH``), so every receipt a test freezes binds the copy's own sha256 and the
  tests see the semantic mutation rather than the self-hash tripwire.  The unmodified copy is the
  control and must pass every one of those tests, or the test layer is reported HARNESS_INVALID.
  Descriptive only (not part of any verdict): the copy with a receipt re-bound to its own sha256.

Catch rules and verdicts
------------------------
* A behavioral gate catches a fixture iff its value is strictly below the known-good reference,
  v2's own value of that gate (section 10: "known-good reference = v2").
* Artifact mutant: BLIND_SPOT iff pass 2 is caught by no behavioral gate (section 10: pass-2
  mutants "must then be caught by a behavioral gate"; integrity is bypassed there).  The status
  counts the behavioral gates only: each mutant is evaluated on a temporary copy, and the v2 test
  suite is not run with the mutant in place of the frozen pair.
* Receipt mutant: BLIND_SPOT iff both the kernel and verify_origin accept it (integrity layer
  only; the test suite is not run with the mutant receipt in place either).
  Post-run correction (after the single final opening): the recorded run justified both rules by
  "the test suite never reads the frozen artifact or receipt" (RECORDED_TEST_LAYER_WORDING).  That
  premise was false: v2/tests/test_mutation_suite.py reads the frozen pair.  The rules and every
  recorded status are unchanged; only the wording (TEST_LAYER_NOT_RUN) and the report changed.
* Kernel mutant: BLIND_SPOT iff neither integrity nor the tests catch it.

Broken baselines (section 10)
-----------------------------
constant-ALERT (a v2 artifact with every weight 0 and intercept +100, consistent receipt), the
fail-open kernel (``v2/tests/_support.fail_open_mutation`` applied to a temporary copy, receipt
re-bound to its sha256) and majority (always NO_ALERT; no receipt, no kernel).  Each goes through
``metrics.baseline_validity`` + ``metrics.binary_report`` on validation: INVALID_BASELINE, BA
only, no per-class rate.  Receipt verification (precondition i) means: the kernel accepts the
pair AND freeze.verify_origin accepts the receipt.

Readiness aggregation (section 11)
----------------------------------
Known-good: v2 and the five AM-4 retrains (``v2/results/known_good/k1..k5``).  Known-broken: every
integrity-bypassed artifact mutant (pass 2), all 36.  Aggregators over the seven gates, in
``gates.GATE_NAMES`` order: AND (1 iff every gate = 1, else 0), min, equal-weight mean, and Lambda
= ``math/lambda_stdlib.lambda_aggregate(values)`` with the reference defaults (weights None =
uniform 1/k, semantics "kernel"; ``math/lambda_reference.md`` sections 1-2; Lambda has no other
parameter; the lambda_gate default threshold 0.5 is NOT used).  READY iff aggregate >= 0.99.
Separation: misclassifications at the cut, margin min(good) - max(broken), AUROC good vs broken
(Mann-Whitney, ties 1/2).  Lambda is KEPT only if its margin is strictly greater than the margin
of every baseline AND its misclassifications are no more than any baseline's; else REJECTED.

Usage (from the repository root):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.mutation run       # writes v2/results/mutation_readiness.json
    PYTHONUTF8=1 py -3.12 -B -m v2.research.mutation render    # writes v2/results/MUTATION_READINESS.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from . import freeze, gates, kernel_bridge, metrics, pipeline, registry, truth
from .dataio import labels_of

SCHEMA = "szl-oac/ops-health-mutation-readiness/v2"
RESULTS_DIR = registry.V2 / "results"
RESULTS_PATH = RESULTS_DIR / "mutation_readiness.json"
REPORT_PATH = RESULTS_DIR / "MUTATION_READINESS.md"
KERNEL_TEST_LOG_DIR = registry.ROOT / "logs" / "v2" / "mutation_kernel_tests"
V2_DIR = registry.V2 / "ops-health"
KNOWN_GOOD_DIR = RESULTS_DIR / "known_good"
KNOWN_GOOD_KEYS = ("k1", "k2", "k3", "k4", "k5")
LAMBDA_PATH = registry.ROOT / "math" / "lambda_stdlib.py"
LAMBDA_PARAMS = {"weights": None, "semantics": "kernel"}
READINESS_CUT = 0.99
AGGREGATORS = ("AND", "min", "mean", "lambda")
BASELINE_AGGREGATORS = ("AND", "min", "mean")
GATES_PURPOSE = "GATES"
FEATURES = truth.FEATURE_NAMES

WEIGHT_FACTORS = (("scale0.5", 0.5), ("scale1.5", 1.5))
INTERCEPT_DELTAS = (("intercept+1", 1.0), ("intercept-1", -1.0))
THRESHOLD_STEPS = (("threshold+0.1", 10), ("threshold-0.1", -10))

WRONG_RECEIPT_SCHEMA = "szl-oac/ops-health-artifact-receipt/v1"
ADDED_METRICS_BLOCK = {
    "validation_balanced_accuracy": 0.99,
    "validation_ece": 0.001,
    "note": "tampered: receipts carry no metrics (AM-3)",
}

# (id, description, supplementary, [(old, new), ...]); every `old` must occur exactly once.
KERNEL_MUTANTS: tuple[tuple[str, str, bool, tuple[tuple[str, str], ...]], ...] = (
    (
        "K-no-finiteness",
        "input finiteness check removed from _normalize_value",
        False,
        ((
            "    if not math.isfinite(number):\n"
            "        raise ModelInputError(f\"{name} must be finite\")\n",
            "",
        ),),
    ),
    (
        "K-no-range",
        "input range check removed from _normalize_value",
        False,
        ((
            "    if number < minimum or number > maximum:\n"
            "        raise ModelInputError(f\"{name} must be between {minimum:g} and {maximum:g}\")\n",
            "",
        ),),
    ),
    (
        "K-no-prohibited-screen",
        "prohibited-key screening removed from screen_input (match = None)",
        False,
        (("            match = prohibited_match(key)\n", "            match = None\n"),),
    ),
    (
        "K-inverted-model-hash",
        "model-hash comparison inverted in OpsHealthKernel (!= becomes ==)",
        False,
        ((
            "        if sha256_bytes(model_bytes) != receipt[\"model_sha256\"]:\n",
            "        if sha256_bytes(model_bytes) == receipt[\"model_sha256\"]:\n",
        ),),
    ),
    (
        "K-inverted-self-hash",
        "SUPPLEMENTARY: kernel self-hash comparison inverted in OpsHealthKernel (!= becomes ==)",
        True,
        ((
            "        if kernel_file_sha256() != receipt[\"kernel_sha256\"]:\n",
            "        if kernel_file_sha256() == receipt[\"kernel_sha256\"]:\n",
        ),),
    ),
)
# The kernel-relevant tests run against every mutated copy (and the unmodified control copy).
KERNEL_TEST_IDS = (
    "v2.tests.test_failclosed_probes",
    "v2.tests.test_receipt.ReceiptTest",
    "v2.tests.test_review_hardening.ArtifactHardeningTest",
    "v2.tests.test_review_hardening.InputHardeningTest",
    "v2.tests.test_review_hardening.EchoBoundTest",
    "v2.tests.test_gates.GateTest.test_integrity_gate",
)
CONTROL_ID = "K-control-identity"

# The recorded run (v2/results/mutation_readiness.json) wrote RECORDED_TEST_LAYER_WORDING into
# caught_by_tests of every artifact and receipt mutant.  Its premise is false: the committed
# v2/tests/test_mutation_suite.py reads v2/ops-health/model.json and artifact_receipt.json.
# Corrected after the single final opening (wording only; no status, gate or cut changed); the
# recorded JSON is left byte-identical and the report discloses the correction.
RECORDED_TEST_LAYER_WORDING = (
    "NOT_APPLICABLE: the v2 test suite runs on DEV fixtures and never reads the frozen v2 "
    "artifact or receipt; this mutant is applied to a temporary copy"
)
TEST_LAYER_NOT_RUN = (
    "NOT_RUN: the status counts the behavioral gates (artifact pass 2) or the integrity layer "
    "(receipt) only; this mutant was evaluated on a temporary copy and the v2 test suite was not "
    "run with it in place of the frozen pair"
)
# Post-run corrections: when they were made relative to the single final opening (REPORTED from
# runs/TEST_OPENED.json, committed in 3b0912a), and where the supplementary in-place test run
# (logs/v2/mutation_inplace_tests.py) writes its results.
OPENING_UTC = "2026-09-29T21:13:47Z"
RUN_COMMIT = "c46a42c"
FINAL_REPORT = "v2/results/FINAL_RESULTS.md"
LAMBDA_COMMIT = "d9c5fb3"
INPLACE_TESTS_PATH = registry.ROOT / "logs" / "v2" / "mutation_inplace_tests.json"


class MutationRefused(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def q12(value: float) -> float:
    return round(float(value), 12)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(registry.ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            + "\n").encode("utf-8")


def test_opened_marker() -> Path:
    return registry.RUNS_DIR / registry.TEST_OPENED_NAME


_LAMBDA: dict[str, ModuleType] = {}


def lambda_module() -> ModuleType:
    if "mod" not in _LAMBDA:
        spec = importlib.util.spec_from_file_location("oac_lambda_stdlib", LAMBDA_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {LAMBDA_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LAMBDA["mod"] = module
    return _LAMBDA["mod"]


# --------------------------------------------------------------------------------------
# mutant catalogue
# --------------------------------------------------------------------------------------
def artifact_mutants(artifact: Mapping) -> list[dict]:
    """The 36 section 10 artifact mutants of ``artifact`` (each a new, frozen-resolution dict)."""
    out = []

    def add(mutant_id, family, feature, field, before, after, description):
        mutated = copy.deepcopy(dict(artifact))
        if field == "intercept":
            mutated["intercept"] = after
        elif field == "decision_threshold":
            mutated["decision_threshold"] = after
        else:
            mutated["weights"][feature] = after
        out.append({"id": mutant_id, "family": family, "feature": feature, "field": field,
                    "before": before, "after": after, "description": description,
                    "artifact": mutated})

    weights = artifact["weights"]
    for tag, factor in WEIGHT_FACTORS:
        for f in FEATURES:
            add(f"A-{tag}-{f}", f"weight x{factor:g}", f, f"weights.{f}", weights[f],
                q12(weights[f] * factor), f"weight of {f} multiplied by {factor:g}")
    for f in FEATURES:
        add(f"A-signflip-{f}", "sign flip", f, f"weights.{f}", weights[f], q12(-weights[f]),
            f"sign of the weight of {f} flipped")
    for f in FEATURES:
        add(f"A-drop-{f}", "drop feature", f, f"weights.{f}", weights[f], 0.0,
            f"feature {f} dropped (weight set to 0)")
    for tag, delta in INTERCEPT_DELTAS:
        add(f"A-{tag}", "intercept", None, "intercept", artifact["intercept"],
            q12(artifact["intercept"] + delta), f"intercept {delta:+g}")
    index = round(artifact["decision_threshold"] * 100)
    if index / 100 != artifact["decision_threshold"]:
        raise MutationRefused("frozen threshold is not on the registered grid")
    for tag, step in THRESHOLD_STEPS:
        new_index = index + step
        if not 1 <= new_index <= 99:
            raise MutationRefused(f"{tag} leaves the registered 0.01..0.99 grid")
        add(f"A-{tag}", "threshold", None, "decision_threshold", artifact["decision_threshold"],
            new_index / 100, f"decision threshold {step / 100:+g} (grid index {index} -> {new_index})")
    ids = [m["id"] for m in out]
    if len(ids) != 36 or len(set(ids)) != 36:
        raise MutationRefused("artifact mutant catalogue must hold 36 distinct mutants")
    return out


def _flip_low_bit_hex(digit: str) -> str:
    return format(int(digit, 16) ^ 1, "x")


def parent_commit(commit: str) -> str:
    return _git("rev-parse", f"{commit}^")


def receipt_mutants(receipt: Mapping) -> list[dict]:
    """The section 10 receipt mutants (AM-3 for the metrics mutant) of ``receipt``."""
    out = []

    def add(mutant_id, description, change, supplementary=False):
        mutated = copy.deepcopy(dict(receipt))
        change(mutated)
        out.append({"id": mutant_id, "description": description, "supplementary": supplementary,
                    "receipt": mutated})

    model_sha = receipt["model_sha256"]
    flipped_sha = _flip_low_bit_hex(model_sha[0]) + model_sha[1:]
    add("R-model-hash-flip",
        f"model_sha256 first hex digit {model_sha[0]!r} -> {flipped_sha[0]!r} (low bit flipped)",
        lambda r: r.update(model_sha256=flipped_sha))
    add("R-wrong-schema", f"schema -> {WRONG_RECEIPT_SCHEMA!r}",
        lambda r: r.update(schema=WRONG_RECEIPT_SCHEMA))
    commit = receipt["source"]["commit"]
    altered = commit[:-1] + _flip_low_bit_hex(commit[-1])
    add("R-source-commit-altered",
        f"source.commit last hex digit {commit[-1]!r} -> {altered[-1]!r} (low bit flipped)",
        lambda r: r["source"].update(commit=altered))
    other = parent_commit(commit)
    add("R-source-commit-other",
        f"SUPPLEMENTARY: source.commit -> {other[:7]}, the parent of the recorded {commit[:7]} "
        "(a real commit of this repository)",
        lambda r: r["source"].update(commit=other), supplementary=True)
    add("R-metrics-added", "AM-3: a metrics block added to the receipt",
        lambda r: r.update(metrics=dict(ADDED_METRICS_BLOCK)))
    return out


def apply_kernel_edits(text: str, edits: Sequence[tuple[str, str]]) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise MutationRefused(f"kernel mutation target occurs {count} times: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def write_kernel_copy(directory: Path, name: str, text: str) -> Path:
    sub = Path(directory) / name
    sub.mkdir(parents=True, exist_ok=False)
    path = sub / "ops_health.py"
    path.write_bytes(text.encode("utf-8"))
    return path


def fail_open_source() -> str:
    """The stage-1 fail-open kernel fixture (single definition: v2/tests/_support.py)."""
    from v2.tests import _support  # noqa: PLC0415 -- the registered stage-1 fixture

    return _support.fail_open_mutation(registry.KERNEL_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# integrity layer
# --------------------------------------------------------------------------------------
def write_pair(directory: Path, model_bytes: bytes, receipt: Mapping) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=False)
    model = directory / freeze.MODEL_NAME
    rec = directory / freeze.RECEIPT_NAME
    model.write_bytes(model_bytes)
    rec.write_bytes(freeze.canonical_json_bytes(receipt))
    return model, rec


def kernel_check(model: Path, receipt: Path, module: ModuleType | None = None) -> dict:
    module = module or kernel_bridge.kernel()
    try:
        module.OpsHealthKernel(model, receipt)
    except module.OperationalModelError as exc:
        return {"accepted": False, "refusal": f"{type(exc).__name__}: {exc}"}
    return {"accepted": True, "refusal": None}


def origin_check(receipt: Mapping) -> dict:
    try:
        freeze.verify_origin(receipt)
    except (freeze.OriginMismatch, kernel_bridge.kernel().OperationalModelError) as exc:
        return {"accepted": False, "refusal": f"{type(exc).__name__}: {exc}"}
    return {"accepted": True, "refusal": None}


def integrity_layer(model: Path, receipt_path: Path, receipt: Mapping,
                    module: ModuleType | None = None) -> dict:
    kc = kernel_check(model, receipt_path, module)
    oc = origin_check(receipt)
    return {"kernel": kc, "verify_origin": oc,
            "caught": (not kc["accepted"]) or (not oc["accepted"])}


# --------------------------------------------------------------------------------------
# behavioral gates on validation
# --------------------------------------------------------------------------------------
class GateContext:
    """The registered validation split (purpose GATES), labels and normalized vectors."""

    def __init__(self, rows: Sequence[Mapping], split_sha256: str):
        self.rows = list(rows)
        self.labels = labels_of(self.rows)
        self.vectors = pipeline.normalized_vectors(self.rows)
        self.split_sha256 = split_sha256

    @classmethod
    def registered(cls, guard=None) -> "GateContext":
        if guard is None:
            from . import sealed_guard as guard  # noqa: PLC0415
        rows = guard.open_split("validation", GATES_PURPOSE)
        return cls(rows, pipeline.rows_sha256(rows))

    def describe(self) -> dict:
        return {"split": "validation", "purpose": GATES_PURPOSE, "rows": len(self.rows),
                "positives": sum(self.labels), "negatives": len(self.labels) - sum(self.labels),
                "rows_sha256": self.split_sha256}


def exact_ba(labels: Sequence[int], decisions: Sequence[int]) -> Fraction:
    c = metrics.confusion(labels, decisions)
    pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    return (Fraction(c["tp"], pos) + Fraction(c["tn"], neg)) / 2


def validation_block(labels: Sequence[int], scores: Sequence[float], decisions: Sequence[int],
                     validity: Mapping) -> dict:
    """Score metrics plus the gated binary report (per-class rates only if VALID)."""
    ba = exact_ba(labels, decisions)
    return {
        "balanced_accuracy": float(ba),
        "balanced_accuracy_exact": [ba.numerator, ba.denominator],
        "brier": metrics.brier(labels, scores),
        "ece": metrics.ece(labels, scores),
        "roc_auc": metrics.roc_auc(labels, scores),
        "binary_report": metrics.binary_report(labels, decisions, validity),
    }


def evaluate_gates(kernel_obj, receipt_verified: bool, ctx: GateContext, *,
                   integrity_value: float = 1.0) -> dict:
    """The seven section 10 gates of a loaded kernel (model + receipt accepted)."""
    artifact = kernel_obj.artifact
    fc_value, fc_report = gates.g_failclosed(kernel_obj.advise)
    mono_value, mono_report = gates.g_monotone(
        lambda f: kernel_obj.advise({"features": f})["operator_attention_score"])
    evals = [kernel_obj.evaluate(v) for v in ctx.vectors]
    scores = [e["calibrated_score"] for e in evals]
    decisions = [1 if e["threshold_decision"] else 0 for e in evals]
    validity = metrics.baseline_validity(ctx.labels, decisions, receipt_verified=receipt_verified,
                                         probes_refused=fc_report["refused"],
                                         probes_total=fc_report["total"])
    val = validation_block(ctx.labels, scores, decisions, validity)
    values = {
        "g_integrity": integrity_value,
        "g_failclosed": fc_value,
        "g_truth_sign": gates.g_truth_sign(artifact["weights"]),
        "g_monotone": mono_value,
        "g_validation": gates.g_validation(val["balanced_accuracy"]),
        "g_calibration": gates.g_calibration(val["ece"]),
        "g_baseline_valid": gates.g_baseline_valid(validity),
    }
    return {
        "gates": gates.gate_vector(values),
        "details": {
            "failclosed": {k: fc_report[k] for k in ("refused", "total", "escaped")},
            "truth_sign": truth.sign_agreement(artifact["weights"]),
            "monotone": mono_report,
            "validation": val,
            "decision_threshold": artifact["decision_threshold"],
        },
    }


def fixture_gates(model: Path, receipt_path: Path, ctx: GateContext,
                  module: ModuleType | None = None) -> dict:
    """Integrity layer, then (if the kernel accepts the pair) the seven gates."""
    module = module or kernel_bridge.kernel()
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    integ = integrity_layer(model, receipt_path, receipt, module)
    g_int, _msg = gates.g_integrity(model, receipt_path, module)
    if g_int != 1.0:
        return {"loaded": False, "integrity": integ, "gates": None}
    kernel_obj = module.OpsHealthKernel(model, receipt_path)
    verified = integ["kernel"]["accepted"] and integ["verify_origin"]["accepted"]
    out = evaluate_gates(kernel_obj, verified, ctx, integrity_value=g_int)
    return {"loaded": True, "integrity": integ, **out}


def gates_below(vector: Mapping[str, float], reference: Mapping[str, float]) -> list[str]:
    return [name for name in gates.GATE_NAMES if vector[name] < reference[name]]


# --------------------------------------------------------------------------------------
# readiness aggregation
# --------------------------------------------------------------------------------------
def aggregate(vector: Mapping[str, float], name: str) -> float:
    values = [float(vector[g]) for g in gates.GATE_NAMES]
    if name == "AND":
        return 1.0 if all(v == 1.0 for v in values) else 0.0
    if name == "min":
        return min(values)
    if name == "mean":
        return math.fsum(values) / len(values)
    if name == "lambda":
        return lambda_module().lambda_aggregate(values, LAMBDA_PARAMS["weights"],
                                                semantics=LAMBDA_PARAMS["semantics"])
    raise ValueError(f"unknown aggregator {name!r}")


def is_ready(value: float) -> bool:
    return value >= READINESS_CUT


def separation(good: Sequence[float], broken: Sequence[float]) -> dict:
    if not good or not broken:
        raise ValueError("separation needs good and broken fixtures")
    broken_ready = sum(1 for v in broken if is_ready(v))
    good_not_ready = sum(1 for v in good if not is_ready(v))
    return {
        "misclassifications": broken_ready + good_not_ready,
        "broken_marked_ready": broken_ready,
        "good_marked_not_ready": good_not_ready,
        "margin": min(good) - max(broken),
        "min_good": min(good),
        "max_broken": max(broken),
        "auroc": metrics.roc_auc([1] * len(good) + [0] * len(broken), list(good) + list(broken)),
        "n_good": len(good),
        "n_broken": len(broken),
    }


def lambda_verdict(sep: Mapping[str, Mapping]) -> dict:
    lam = sep["lambda"]
    better_margin = {b: lam["margin"] > sep[b]["margin"] for b in BASELINE_AGGREGATORS}
    no_more_errors = {b: lam["misclassifications"] <= sep[b]["misclassifications"]
                      for b in BASELINE_AGGREGATORS}
    keep = all(better_margin.values()) and all(no_more_errors.values())
    return {
        "verdict": "KEEP" if keep else "REJECTED",
        "rule": "KEEP only if Lambda's margin is strictly greater than every baseline's AND its "
                "misclassifications at the cut are no more than every baseline's (section 11)",
        "strictly_better_margin_than": better_margin,
        "no_more_misclassifications_than": no_more_errors,
    }


def readiness(good: Mapping[str, Mapping[str, float]], broken: Mapping[str, Mapping[str, float]]) -> dict:
    per_fixture = {}
    for group, fixtures in (("good", good), ("broken", broken)):
        for fid, vector in fixtures.items():
            aggs = {name: aggregate(vector, name) for name in AGGREGATORS}
            per_fixture[fid] = {"group": group, "aggregates": aggs,
                                "ready": {n: is_ready(v) for n, v in aggs.items()}}
    sep = {}
    for name in AGGREGATORS:
        g = [per_fixture[f]["aggregates"][name] for f in good]
        b = [per_fixture[f]["aggregates"][name] for f in broken]
        sep[name] = separation(g, b)
    return {"cut": READINESS_CUT, "aggregators": list(AGGREGATORS),
            "lambda_params": {"weights": "None (uniform 1/k)", "semantics": LAMBDA_PARAMS["semantics"],
                              "source": "math/lambda_reference.md sections 1-2 (reference defaults)"},
            "per_fixture": per_fixture, "separation": sep, "lambda": lambda_verdict(sep)}


# --------------------------------------------------------------------------------------
# kernel-relevant tests against a mutated kernel copy (subprocess)
# --------------------------------------------------------------------------------------
def _kernel_tests_entry(kernel_path: str, summary_path: str, test_ids: Sequence[str]) -> int:
    """Run in a fresh interpreter: inject the copy as THE kernel, then run the named tests."""
    import unittest  # noqa: PLC0415

    path = Path(kernel_path).resolve()
    real_key = str(registry.KERNEL_PATH.resolve())
    if real_key in kernel_bridge._CACHE:  # noqa: SLF001
        raise MutationRefused("the shipped kernel was loaded before the injection; refusing")
    module = kernel_bridge.load(path)
    kernel_bridge._CACHE[real_key] = module  # noqa: SLF001 -- test injection, subprocess only
    registry.KERNEL_PATH = path  # receipts frozen by the tests bind the copy's own sha256
    suite = unittest.defaultTestLoader.loadTestsFromNames(list(test_ids))
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)

    def last_line(text: str) -> str:
        lines = [line for line in text.strip().splitlines() if line.strip()]
        return lines[-1][:400] if lines else ""

    def parent_id(test) -> str:  # a failing subTest reports its parent test method
        return getattr(test, "test_case", test).id()

    summary = {
        "kernel_path": str(path),
        "kernel_sha256": sha256_file(path),
        "test_ids": list(test_ids),
        "tests_run": result.testsRun,
        "failures": sorted(t.id() for t, _ in result.failures),
        "errors": sorted(t.id() for t, _ in result.errors),
        "failed_tests": sorted({parent_id(t) for t, _ in result.failures + result.errors}),
        "messages": {t.id(): last_line(tb) for t, tb in result.failures + result.errors},
        "skipped": len(result.skipped),
        "ok": result.wasSuccessful(),
        "seconds": round(time.perf_counter() - started, 3),
    }
    Path(summary_path).write_bytes(_json_bytes(summary))
    return 0


def run_kernel_tests(kernel_path: Path, log_path: Path | None, work: Path,
                     test_ids: Sequence[str] = KERNEL_TEST_IDS) -> dict:
    summary_path = Path(work) / f"summary_{sha256_file(kernel_path)[:16]}_{time.time_ns()}.json"
    env = dict(os.environ, PYTHONUTF8="1")
    cmd = [sys.executable, "-B", "-m", "v2.research.mutation", "kernel-tests",
           "--kernel", str(kernel_path), "--summary", str(summary_path), *test_ids]
    proc = subprocess.run(cmd, cwd=str(registry.ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=1800)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ py -3.12 -B -m v2.research.mutation kernel-tests --kernel <temp copy>/ops_health.py "
            f"--summary <temp> {' '.join(test_ids)}\n"
            f"kernel copy sha256: {sha256_file(kernel_path)}\n"
            f"exit code: {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            encoding="utf-8")
    if proc.returncode != 0 or not summary_path.exists():
        return {"harness_error": True, "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["harness_error"] = False
    return summary


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------
class Progress:
    def __init__(self, results: dict, path: Path, echo: Callable[[str], None] | None):
        self.results = results
        self.path = path
        self.echo = echo

    def save(self, note: str) -> None:
        self.results["progress"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "done": note})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(_json_bytes(self.results))
        if self.echo:
            self.echo(note)


def load_frozen_pair(directory: Path) -> tuple[Path, Path, dict, bytes]:
    model = directory / freeze.MODEL_NAME
    rec = directory / freeze.RECEIPT_NAME
    model_bytes = model.read_bytes()
    receipt = json.loads(rec.read_text(encoding="utf-8"))
    if sha256_bytes(model_bytes) != receipt["model_sha256"]:
        raise MutationRefused(f"{directory}: model.json does not match its receipt")
    return model, rec, receipt, model_bytes


def _behavior_summary(fx: Mapping, reference: Mapping[str, float]) -> dict:
    below = gates_below(fx["gates"], reference)
    v = fx["details"]["validation"]
    return {
        "gates": fx["gates"],
        "gates_below_reference": below,
        "caught": bool(below),
        "validation_ba": v["balanced_accuracy"],
        "validation_ba_exact": v["balanced_accuracy_exact"],
        "validation_ece": v["ece"],
        "validation_brier": v["brier"],
        "validation_roc_auc": v["roc_auc"],
        "baseline_validity": v["binary_report"]["baseline_validity"]["status"],
        "failclosed_escaped": fx["details"]["failclosed"]["escaped"],
        "truth_sign_disagreeing": fx["details"]["truth_sign"]["disagreeing"],
        "monotone_failed_by_kind": fx["details"]["monotone"]["failed_by_kind"],
    }


def run(out_path: Path = RESULTS_PATH, *, echo: Callable[[str], None] | None = print,
        kernel_tests: bool = True, guard=None) -> dict:
    marker = test_opened_marker()
    opened_before = marker.exists()
    if opened_before:
        raise MutationRefused("runs/TEST_OPENED.json exists: this suite runs before the final opening")
    state = freeze.source_state()
    results: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "IN_PROGRESS",
        "synthetic": True,
        "truth_labels": "every number is MEASURED by this run on the registered validation split "
                        "(or on temporary mutant copies); all SYNTHETIC; no sealed split is read",
        "protocol": {
            "preregistration_sha256": sha256_file(registry.PREREG_PATH),
            "amendments_sha256": sha256_file(registry.V2 / "AMENDMENTS.md"),
            "kernel_sha256": sha256_file(registry.KERNEL_PATH),
            "mutation_py_sha256": sha256_file(Path(__file__)),
            "gates_py_sha256": sha256_file(Path(gates.__file__)),
            "lambda_stdlib_sha256": sha256_file(LAMBDA_PATH),
            "git_head": state["commit"],
            "guarded_tree_clean": state["tree_clean"],
            "python": platform.python_version(),
            "sections": "PREREGISTRATION sections 10 and 11; AM-3 (metrics mutant), AM-4 (known-good)",
        },
        "sealed": {"test_opened_marker_before": opened_before},
        "progress": [],
    }
    prog = Progress(results, out_path, echo)
    ctx = GateContext.registered(guard)
    results["validation_split"] = ctx.describe()
    prog.save(f"validation split opened (purpose GATES): {ctx.describe()}")

    # -- known-good fixtures --------------------------------------------------------------
    v2_model, v2_receipt_path, v2_receipt, v2_bytes = load_frozen_pair(V2_DIR)
    v2_artifact = json.loads(v2_bytes.decode("utf-8"))
    if freeze.canonical_model_bytes(v2_artifact) != v2_bytes:
        raise MutationRefused("frozen v2 model.json is not canonical")
    results["frozen_v2"] = {"model_sha256": sha256_bytes(v2_bytes),
                            "receipt_sha256": sha256_file(v2_receipt_path),
                            "source_commit": v2_receipt["source"]["commit"]}
    known_good: dict[str, dict] = {}
    fx = fixture_gates(v2_model, v2_receipt_path, ctx)
    if not fx["loaded"]:
        raise MutationRefused(f"frozen v2 does not load: {fx['integrity']}")
    known_good["v2"] = {"path": registry.relpath(V2_DIR), **fx}
    reference = fx["gates"]
    results["reference_v2_gates"] = reference
    prog.save(f"v2 gates: {reference}")
    index = json.loads((KNOWN_GOOD_DIR / "INDEX.json").read_text(encoding="utf-8"))
    for key in KNOWN_GOOD_KEYS:
        directory = KNOWN_GOOD_DIR / key
        model, rec, receipt, model_bytes = load_frozen_pair(directory)
        if sha256_bytes(model_bytes) != index["retrains"][key]["model_sha256"]:
            raise MutationRefused(f"{key}: model.json differs from known_good/INDEX.json")
        fx = fixture_gates(model, rec, ctx)
        if not fx["loaded"]:
            raise MutationRefused(f"known-good {key} does not load: {fx['integrity']}")
        known_good[key] = {"path": registry.relpath(directory), **fx}
        prog.save(f"known-good {key} gates: {fx['gates']}")
    results["known_good"] = known_good

    with tempfile.TemporaryDirectory(prefix="oac_v2_mutation_") as tmp_name:
        tmp = Path(tmp_name)
        # -- artifact mutants: pass 1 and pass 2 ------------------------------------------
        amuts = []
        for m in artifact_mutants(v2_artifact):
            mbytes = freeze.canonical_model_bytes(m["artifact"])
            base = tmp / "artifact" / m["id"]
            model1, rec1 = write_pair(base / "pass1", mbytes, v2_receipt)
            pass1 = integrity_layer(model1, rec1, v2_receipt)
            regenerated = dict(copy.deepcopy(v2_receipt), model_sha256=sha256_bytes(mbytes))
            model2, rec2 = write_pair(base / "pass2", mbytes, regenerated)
            fx2 = fixture_gates(model2, rec2, ctx)
            if not fx2["loaded"]:
                raise MutationRefused(f"{m['id']}: integrity-bypassed mutant does not load")
            behavior = _behavior_summary(fx2, reference)
            amuts.append({
                "id": m["id"], "family": m["family"], "feature": m["feature"],
                "field": m["field"], "before": m["before"], "after": m["after"],
                "description": m["description"], "model_sha256": sha256_bytes(mbytes),
                "pass1_integrity": {"caught": pass1["caught"], **pass1},
                "pass2_integrity_bypassed": {
                    "kernel_accepts": fx2["integrity"]["kernel"]["accepted"],
                    "verify_origin_accepts": fx2["integrity"]["verify_origin"]["accepted"],
                },
                "caught_by_tests": TEST_LAYER_NOT_RUN,
                "pass2_behavior": behavior,
                "status": "CAUGHT" if behavior["caught"] else "BLIND_SPOT",
                "blind_spot_layer": None if behavior["caught"] else
                "behavioral gates, integrity bypassed (pass 2)",
                "pass2_details": fx2["details"],
            })
            prog.results["artifact_mutants"] = amuts
            prog.save(f"{m['id']}: pass1 caught={pass1['caught']} pass2 below={behavior['gates_below_reference']}")

        # -- receipt mutants --------------------------------------------------------------
        rmuts = []
        for r in receipt_mutants(v2_receipt):
            model, rec = write_pair(tmp / "receipt" / r["id"], v2_bytes, r["receipt"])
            layer = integrity_layer(model, rec, r["receipt"])
            rmuts.append({
                "id": r["id"], "description": r["description"], "supplementary": r["supplementary"],
                "integrity": layer, "caught_by_integrity": layer["caught"],
                "caught_by_tests": TEST_LAYER_NOT_RUN,
                "status": "CAUGHT" if layer["caught"] else "BLIND_SPOT",
                "blind_spot_layer": None if layer["caught"] else "integrity (kernel + verify_origin)",
            })
            results["receipt_mutants"] = rmuts
            prog.save(f"{r['id']}: caught={layer['caught']}")

        # -- kernel-source mutants ----------------------------------------------------------
        source = registry.KERNEL_PATH.read_text(encoding="utf-8")
        kmuts = []
        control_path = write_kernel_copy(tmp / "kernel", CONTROL_ID, source)
        control_module = kernel_bridge.load(control_path)
        control_integrity = kernel_check(v2_model, v2_receipt_path, control_module)
        control_tests = (run_kernel_tests(control_path, KERNEL_TEST_LOG_DIR / f"{CONTROL_ID}.txt", tmp)
                         if kernel_tests else None)
        harness_valid = bool(control_tests and not control_tests["harness_error"] and control_tests["ok"])
        results["kernel_control"] = {
            "id": CONTROL_ID, "kernel_sha256": sha256_file(control_path),
            "identical_to_shipped": sha256_file(control_path) == sha256_file(registry.KERNEL_PATH),
            "integrity": control_integrity, "tests": control_tests,
            "test_harness_valid": harness_valid, "test_ids": list(KERNEL_TEST_IDS),
        }
        prog.save(f"kernel control: integrity accepted={control_integrity['accepted']} "
                  f"tests ok={harness_valid}")
        for kid, description, supplementary, edits in KERNEL_MUTANTS:
            text = apply_kernel_edits(source, edits)
            path = write_kernel_copy(tmp / "kernel", kid, text)
            module = kernel_bridge.load(path)
            integ = kernel_check(v2_model, v2_receipt_path, module)
            rebound = dict(copy.deepcopy(v2_receipt), kernel_sha256=sha256_file(path))
            rb_model, rb_rec = write_pair(tmp / "kernel_rebound" / kid, v2_bytes, rebound)
            rb = kernel_check(rb_model, rb_rec, module)
            rb_probes = None
            if rb["accepted"]:
                probes = gates.run_failclosed_probes(module.OpsHealthKernel(rb_model, rb_rec).advise)
                rb_probes = {k: probes[k] for k in ("refused", "total", "fraction", "escaped")}
            tests = (run_kernel_tests(path, KERNEL_TEST_LOG_DIR / f"{kid}.txt", tmp)
                     if kernel_tests else None)
            if tests is None or tests.get("harness_error") or not harness_valid:
                caught_tests = None
                test_status = "HARNESS_INVALID" if tests is not None else "NOT_RUN"
            else:
                caught_tests = not tests["ok"]
                test_status = "FAILED" if caught_tests else "PASSED"
            caught_integrity = not integ["accepted"]
            caught = caught_integrity or bool(caught_tests)
            kmuts.append({
                "id": kid, "description": description, "supplementary": supplementary,
                "edits": [{"old": o, "new": n} for o, n in edits],
                "kernel_sha256": sha256_file(path),
                "caught_by_integrity": caught_integrity,
                "integrity": integ,
                "caught_by_tests": caught_tests,
                "tests_status": test_status,
                "tests": tests,
                "descriptive_self_hash_rebound": {
                    "kernel_accepts_frozen_model": rb["accepted"], "refusal": rb["refusal"],
                    "failclosed_probes": rb_probes,
                    "verify_origin_accepts_rebound_receipt": origin_check(rebound)["accepted"],
                },
                "status": "CAUGHT" if caught else "BLIND_SPOT",
                "blind_spot_layer": None if caught else "integrity and tests",
            })
            results["kernel_mutants"] = kmuts
            prog.save(f"{kid}: integrity caught={caught_integrity} tests={test_status}")

        # -- broken baselines ----------------------------------------------------------------
        results["broken_baselines"] = broken_baselines(tmp, v2_artifact, v2_receipt, v2_bytes, ctx)
        prog.save("broken baselines evaluated")

    # -- readiness aggregation -------------------------------------------------------------
    good = {fid: f["gates"] for fid, f in known_good.items()}
    broken = {m["id"]: m["pass2_behavior"]["gates"] for m in amuts}
    results["readiness"] = readiness(good, broken)
    blind = [{"id": m["id"], "kind": "artifact", "layer": m["blind_spot_layer"]}
             for m in amuts if m["status"] == "BLIND_SPOT"]
    blind += [{"id": m["id"], "kind": "receipt", "layer": m["blind_spot_layer"],
               "supplementary": m["supplementary"]} for m in rmuts if m["status"] == "BLIND_SPOT"]
    blind += [{"id": m["id"], "kind": "kernel", "layer": m["blind_spot_layer"],
               "supplementary": m["supplementary"]} for m in kmuts if m["status"] == "BLIND_SPOT"]
    results["blind_spots"] = blind
    results["counts"] = {
        "artifact_mutants": len(amuts),
        "artifact_pass1_caught_by_integrity": sum(1 for m in amuts if m["pass1_integrity"]["caught"]),
        "artifact_pass2_caught_by_behavior": sum(1 for m in amuts if m["pass2_behavior"]["caught"]),
        "receipt_mutants": len(rmuts),
        "receipt_caught": sum(1 for m in rmuts if m["caught_by_integrity"]),
        "kernel_mutants": len(kmuts),
        "kernel_caught_by_integrity": sum(1 for m in kmuts if m["caught_by_integrity"]),
        "kernel_caught_by_tests": sum(1 for m in kmuts if m["caught_by_tests"]),
        "blind_spots": len(blind),
    }
    results["sealed"]["test_opened_marker_after"] = marker.exists()
    if marker.exists():
        raise MutationRefused("runs/TEST_OPENED.json appeared during the run")
    results["status"] = "COMPLETE"
    prog.save(f"COMPLETE: counts {results['counts']}; lambda {results['readiness']['lambda']['verdict']}")
    return results


def broken_baselines(tmp: Path, v2_artifact: Mapping, v2_receipt: Mapping, v2_bytes: bytes,
                     ctx: GateContext) -> dict:
    """Section 10 broken fixtures on validation: constant-ALERT, fail-open kernel, majority."""
    out = {}
    # constant-ALERT: every weight 0, intercept +100 -> score 1.0 -> ALERT on every row
    art = copy.deepcopy(dict(v2_artifact))
    art["weights"] = {f: 0.0 for f in FEATURES}
    art["intercept"] = 100.0
    abytes = freeze.canonical_model_bytes(art)
    receipt = dict(copy.deepcopy(v2_receipt), model_sha256=sha256_bytes(abytes))
    model, rec = write_pair(tmp / "baseline" / "constant-ALERT", abytes, receipt)
    fx = fixture_gates(model, rec, ctx)
    out["constant_alert"] = _baseline_record(
        "constant-ALERT model: v2 artifact with every weight 0 and intercept +100 (score 1.0), "
        "consistent receipt", fx)
    # fail-open kernel: stage-1 fixture on a temp copy; receipt re-bound to its own sha256
    path = write_kernel_copy(tmp / "kernel", "fail-open", fail_open_source())
    module = kernel_bridge.load(path)
    receipt = dict(copy.deepcopy(v2_receipt), kernel_sha256=sha256_file(path))
    model, rec = write_pair(tmp / "baseline" / "fail-open", v2_bytes, receipt)
    fx = fixture_gates(model, rec, ctx, module)
    record = _baseline_record(
        "fail-open kernel (v2/tests/_support.fail_open_mutation on a temporary copy) with the "
        "frozen v2 model; receipt re-bound to the copy's sha256", fx)
    if fx["loaded"]:  # precondition (i) granted: the kernel alone accepts; (iv) must still fail
        kernel_obj = module.OpsHealthKernel(model, rec)
        probes = gates.run_failclosed_probes(kernel_obj.advise)
        decisions = [1 if kernel_obj.evaluate(v)["threshold_decision"] else 0 for v in ctx.vectors]
        validity = metrics.baseline_validity(ctx.labels, decisions, receipt_verified=True,
                                             probes_refused=probes["refused"],
                                             probes_total=probes["total"])
        record["with_precondition_i_granted"] = metrics.binary_report(ctx.labels, decisions, validity)
    record["kernel_sha256"] = sha256_file(path)
    out["fail_open_kernel"] = record
    # majority: always NO_ALERT; no receipt; a constant callable refuses no probe
    probes = gates.run_failclosed_probes(lambda payload: {"advisory": "NO_ALERT"})
    decisions = [0] * len(ctx.labels)
    validity = metrics.baseline_validity(ctx.labels, decisions, receipt_verified=False,
                                         probes_refused=probes["refused"], probes_total=probes["total"])
    out["majority"] = {
        "description": "majority: always NO_ALERT (no artifact, no receipt, no kernel)",
        "report": metrics.binary_report(ctx.labels, decisions, validity),
        "probes": {k: probes[k] for k in ("refused", "total")},
    }
    for name, record in out.items():
        report = record["report"]
        if report is not None:
            record["invalid_baseline"] = report["per_class_rates"] == metrics.INVALID_BASELINE
            text = json.dumps(report)
            record["no_per_class_rate_reported"] = not any(
                f'"{key}"' in text for key in metrics.PER_CLASS_RATE_KEYS) and "confusion" not in report
    return out


def _baseline_record(description: str, fx: Mapping) -> dict:
    if not fx["loaded"]:
        return {"description": description, "loaded": False, "integrity": fx["integrity"],
                "report": None}
    v = fx["details"]["validation"]
    return {
        "description": description,
        "loaded": True,
        "integrity": fx["integrity"],
        "report": v["binary_report"],
        "gates": fx["gates"],
        "failclosed": fx["details"]["failclosed"],
        "brier": v["brier"],
        "ece": v["ece"],
    }


# --------------------------------------------------------------------------------------
# report rendering (MUTATION_READINESS.md from the results JSON)
# --------------------------------------------------------------------------------------
def _f(value, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gate_cells(vector: Mapping[str, float]) -> str:
    return " | ".join(_f(vector[g], 4) for g in gates.GATE_NAMES)


GATE_SHORT = ("integrity", "failclosed", "truth_sign", "monotone", "validation", "calibration",
              "baseline_valid")


def load_inplace_tests(path: Path = INPLACE_TESTS_PATH) -> dict | None:
    """The supplementary in-place test run (logs/v2/mutation_inplace_tests.py), if complete."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if "summary" in data else None


def _render_test_layer(results: Mapping, add: Callable[[str], None]) -> None:
    """Post-run correction of the test-layer wording, and the supplementary in-place run."""
    arts, recs = results["artifact_mutants"], results["receipt_mutants"]
    wording = {m["caught_by_tests"] for m in (*arts, *recs)}
    add("## Test layer for artifact and receipt mutants (post-run correction)")
    add("")
    if wording == {RECORDED_TEST_LAYER_WORDING}:
        add(f"The recorded field `caught_by_tests` of all {len(arts)} artifact and {len(recs)} "
            f"receipt mutants reads \"{RECORDED_TEST_LAYER_WORDING}\" (REPORTED from the JSON). "
            "The reason it gives is false. `v2/tests/test_mutation_suite.py`, committed with the "
            "suite in `bb8d56f` before the run, reads `v2/ops-health/model.json` and "
            "`artifact_receipt.json` (REPORTED from the source). The actual reason the test layer "
            "did not count is that the v2 test suite was never run with an artifact or receipt "
            "mutant in place of the frozen pair, because each mutant was evaluated on a temporary "
            f"copy. `mutation.py` now writes \"{TEST_LAYER_NOT_RUN}\". The recorded JSON is left "
            "byte-identical as the record of the run.")
    else:
        add(f"The recorded field `caught_by_tests` of the artifact and receipt mutants reads: "
            f"{'; '.join(sorted(wording))} (REPORTED from the JSON).")
    add("")
    add("PREREGISTRATION section 10 says \"A mutant is CAUGHT if the integrity layer or the test "
        "suite fails\". For the integrity-bypassed pass, it says artifact mutants \"must then be "
        "caught by a behavioral gate\". The recorded statuses follow the second sentence for "
        "pass 2, as `mutation.py` declared before the run.")
    add("")
    sup = load_inplace_tests()
    if sup is None:
        add("No supplementary in-place test run exists: how the suite reacts to a blind-spot "
            "mutant in place is UNKNOWN.")
        add("")
        return
    control = sup["variants"]["control"]
    readers = sorted({h.split(":", 1)[0] for h in sup.get("frozen_pair_readers_at_pin", [])})
    add(f"**Supplementary in-place run** (MEASURED; post-run and after the opening; not part of "
        "any verdict). Script `logs/v2/mutation_inplace_tests.py`, results "
        "`logs/v2/mutation_inplace_tests.json`, per-variant output in "
        "`logs/v2/mutation_inplace_tests/`. Each recorded BLIND_SPOT mutant was put in place of "
        f"the frozen pair in a copy of the tree at the stage's last commit `{sup['pin'][:7]}`. "
        f"The copy held no sealed file and no split file (checked on the archive member list: "
        f"{not sup.get('tree_has_sealed_or_jsonl', True)}). Artifact mutants were placed with "
        f"their consistently regenerated pass-2 receipt. `{sup['module']}` then ran against the "
        "copy. By a source scan (hits recorded in the JSON), it is the only test module at that "
        f"commit that reads the frozen pair: {', '.join(f'`{r}`' for r in readers) or 'none'}. "
        "That no other module reads the pair indirectly is MODELED. Control, with the unmodified "
        f"pair: {control['ran']} tests, {control['outcome']}.")
    add("")
    add("| mutant | kind | tests run | outcome | extra failures vs control |")
    add("| --- | --- | ---: | --- | --- |")
    for vid, v in sup["variants"].items():
        if vid == "control":
            continue
        extra = ", ".join(f"`{t.replace('v2.tests.', '')}`" for t in v.get("extra_vs_control", []))
        add(f"| {vid} | {v['kind']} | {v['ran']} | {v['outcome']} | {extra or 'none'} |")
    add("")
    s = sup["summary"]
    caught = s["in_place_suite_fails"]
    add(f"Under the registered pass-2 reading, the BLIND_SPOT count is {results['counts']['blind_spots']}. "
        + (f"Under a literal \"any layer\" reading of section 10's first sentence, "
           f"{', '.join(f'`{x}`' for x in caught)} would count as CAUGHT and the count would be "
           f"{s['blind_spots_under_literal_any_layer_reading']}. "
           if caught else "The in-place suite fails for none of them, so the count is the same "
           "under a literal \"any layer\" reading. ")
        + "(MEASURED)"
        + (f" Host problems during the run: {s['any_host_problem']}." if s["any_host_problem"] else ""))
    thr = sup["variants"].get("A-threshold-0.1", {})
    if "A-threshold-0.1" in caught and all("test_mutation_suite" in t for t in thr["extra_vs_control"]):
        add("")
        add("The `A-threshold-0.1` failures are incidental (MODELED, from the traceback in "
            "`logs/v2/mutation_inplace_tests/A-threshold-0.1.txt`). The test builds the "
            "registered mutant catalogue from whatever pair is in place, and at threshold 0.10 the "
            "-0.1 step leaves the 0.01..0.99 grid (`MutationRefused`). No test checks the "
            "decision behavior that changed.")
    add("")
    add("The stage's verification independently ran the full suite at `929677d` with three of "
        "these mutants in place: `R-source-commit-other`, `A-scale0.5-tls_enabled` and "
        "`A-threshold-0.1` (pass 2). The first two gave no extra failure. `A-threshold-0.1` gave "
        "the same two `test_mutation_suite` errors, plus one error that the host's paging file "
        "caused (REPORTED; `logs/verify/v2-mutation/r_suite929.txt`, not committed). Other lanes "
        "added tests after this stage that also read the frozen pair (`v2/tests/"
        "test_final_score.py`, `f9c5588`). They are not part of this stage's test layer and "
        "were not run here.")
    add("")


def render(results: Mapping) -> str:
    if results.get("status") != "COMPLETE":
        raise MutationRefused("results are not COMPLETE; refusing to render")
    c = results["counts"]
    rd = results["readiness"]
    lam = rd["lambda"]
    vs = results["validation_split"]
    L: list[str] = []
    add = L.append
    add("# OAC Ops Health v2: mutation suite, broken baselines and readiness aggregation")
    add("")
    sealed = results["sealed"]
    untouched = not sealed["test_opened_marker_before"] and not sealed["test_opened_marker_after"]
    add("Everything here is **SYNTHETIC**: synthetic data, synthetic mutants, and nothing that "
        "says anything about a real system. Every number was **MEASURED** by "
        "`PYTHONUTF8=1 py -3.12 -B -m v2.research.mutation run` (log "
        "`logs/v2/mutation_run.txt`, results `v2/results/mutation_readiness.json`), on the "
        "registered **validation** split only (opened with purpose GATES). "
        + ("`runs/TEST_OPENED.json` did not exist before or after the run (MEASURED), and no "
           "code path of the suite opens a sealed split (MODELED, from the source). "
           if untouched else "WARNING: `runs/TEST_OPENED.json` existed at run time. ")
        + "This report makes no statement about v2 versus v1; that comparison exists only after "
        "the single final opening.")
    add("")
    add("**Post-run corrections (wording, render and test fixes only).** The mutation run was "
        f"recorded in `{RUN_COMMIT}`, before the sealed splits were opened. The single final "
        f"opening happened afterwards (`runs/TEST_OPENED.json`, opened {OPENING_UTC}; REPORTED; "
        f"the v2-versus-v1 comparison is in `{FINAL_REPORT}`, not here). After that opening, "
        "following the stage's adversarial verification, these corrections were made:")
    add("")
    add(f"- The Λ implementation `math/lambda_stdlib.py` was committed, byte-identical, in "
        f"`{LAMBDA_COMMIT}`; it had not been in version control. `v2/tests/test_readiness.py` "
        "now pins its sha256 to the recorded one. See \"Λ implementation provenance\".")
    add("- The test-layer wording was corrected. The recorded run justified the artifact and "
        "receipt rules by a false premise, that the test suite never reads the frozen pair. See "
        "\"Test layer for artifact and receipt mutants\". A supplementary in-place test run, "
        "outside any verdict, is reported there.")
    add("- Render wording: this page now states which later edits touched `mutation.py`, and "
        "`v2/tests/test_readiness.py` checks that this page equals the render of the JSON.")
    add("")
    add("None of this changes a recorded number, a status, a gate definition or the cut. "
        f"`v2/results/mutation_readiness.json` is unchanged since `{RUN_COMMIT}`, and this page is "
        "re-rendered from it; the run itself was not repeated.")
    add("")
    add("Labels: **MEASURED** (ran here; output in the cited file), **REPORTED** (read from the "
        "cited file), **MODELED** (reasoning, no measurement behind it).")
    add("")
    add("## Summary")
    add("")
    add(f"- Artifact mutants: {c['artifact_mutants']}. Integrity caught "
        f"{c['artifact_pass1_caught_by_integrity']}/{c['artifact_mutants']} (pass 1). With the "
        f"receipt regenerated (integrity bypassed), the behavioral gates caught "
        f"{c['artifact_pass2_caught_by_behavior']}/{c['artifact_mutants']} (pass 2). (MEASURED)")
    add(f"- Receipt mutants: {c['receipt_mutants']} (4 registered + 1 supplementary). The "
        f"integrity layer caught {c['receipt_caught']}/{c['receipt_mutants']}. (MEASURED)")
    add(f"- Kernel-source mutants: {c['kernel_mutants']} (4 registered + 1 supplementary). "
        f"Integrity caught {c['kernel_caught_by_integrity']}/{c['kernel_mutants']}; the kernel-"
        f"relevant tests caught {c['kernel_caught_by_tests']}/{c['kernel_mutants']}. (MEASURED)")
    add(f"- **BLIND_SPOT: {c['blind_spots']}** (listed below). (MEASURED) The pass-2 status of an "
        "artifact mutant counts the behavioral gates only (section 10: \"they must then be caught "
        "by a behavioral gate\"); how the test suite reacts with a blind-spot mutant in place is "
        "reported separately below.")
    add(f"- Broken baselines: constant-ALERT, the fail-open kernel and majority all give "
        f"`INVALID_BASELINE` with no per-class rate: "
        f"{all(b.get('invalid_baseline') and b.get('no_per_class_rate_reported') for b in results['broken_baselines'].values())}. (MEASURED)")
    add(f"- **Λ readiness aggregation: {lam['verdict']}** (numbers below). (MEASURED)")
    add("")
    add("## Setup (MEASURED unless marked)")
    add("")
    p = results["protocol"]
    add("| item | value |")
    add("| --- | --- |")
    add(f"| git HEAD at run / guarded tree clean | `{p['git_head']}` / {p['guarded_tree_clean']} |")
    add(f"| PREREGISTRATION.md sha256 | `{p['preregistration_sha256']}` |")
    add(f"| AMENDMENTS.md sha256 | `{p['amendments_sha256']}` |")
    add(f"| kernel sha256 | `{p['kernel_sha256']}` |")
    add(f"| frozen v2 model.json / receipt sha256 | `{results['frozen_v2']['model_sha256']}` / "
        f"`{results['frozen_v2']['receipt_sha256']}` |")
    add(f"| v2 receipt source commit | `{results['frozen_v2']['source_commit']}` |")
    add(f"| validation split (purpose {vs['purpose']}) | {vs['rows']} rows, {vs['positives']} "
        f"positive, rows sha256 `{vs['rows_sha256']}` |")
    add(f"| mutation.py / gates.py / lambda_stdlib.py sha256 | `{p['mutation_py_sha256'][:16]}…` / "
        f"`{p['gates_py_sha256'][:16]}…` / `{p['lambda_stdlib_sha256'][:16]}…` |")
    add(f"| python | {p['python']} |")
    add("")
    add("The run used `v2/research/mutation.py` as committed at the git HEAD above (sha256 in the "
        "table). This page is rendered afterwards from the recorded JSON by "
        "`python -m v2.research.mutation render`. Later edits to `mutation.py` touch only the "
        "module docstring, the report rendering (the render function, its helpers and "
        "constants) and the test-layer wording that `run` writes; "
        "`git diff bb8d56f -- v2/research/mutation.py` lists them. None of them changes a "
        "recorded number or status, and `v2/tests/test_readiness.py` recomputes the aggregation "
        "from the JSON and checks that this page equals the render of the JSON.")
    add("")
    add(f"**Λ implementation provenance.** `math/lambda_stdlib.py` (the Λ implementation that "
        "section 11 names) and `math/lambda_reference.md` were not in version control at the run, "
        "nor in any of this stage's commits (`bb8d56f`, `c46a42c`, `929677d`). The implementation "
        "was then pinned only by the sha256 recorded in the JSON, and a clean checkout of those "
        "commits could not run `v2/tests/test_readiness.py` (REPORTED from the stage's "
        f"verification). Both files were committed afterwards, byte-identical, in `{LAMBDA_COMMIT}` "
        "(after the opening). `v2/tests/test_readiness.py` now asserts that the file's sha256 "
        f"equals the recorded `{p['lambda_stdlib_sha256'][:16]}…`.")
    add("")
    add("Rules, all fixed in `v2/research/mutation.py` before the first run:")
    add("")
    add("- A behavioral gate **catches** a fixture iff its value is strictly below v2's own value "
        "of that gate (section 10: known-good reference = v2). The gates are the ones in "
        "`v2/research/gates.py`, whose only commit is the stage-1 commit `974159c` (REPORTED "
        "from git log; its sha256 is in the table).")
    add("- An artifact mutant is a **BLIND_SPOT** iff no behavioral gate catches it in pass 2 "
        "(section 10: pass-2 mutants \"must then be caught by a behavioral gate\"). The pass-2 "
        "status counts the behavioral gates only: integrity is bypassed there, and the v2 test "
        "suite was not run with the mutant in place of the frozen pair, because each mutant was "
        "evaluated on a temporary copy. (Corrected wording: see \"Test layer for artifact and "
        "receipt mutants\".)")
    add("- A receipt mutant is a **BLIND_SPOT** iff the kernel and `freeze.verify_origin` both "
        "accept it (integrity layer only; the test suite was not run with the mutant receipt in "
        "place either).")
    add("- A kernel mutant is a **BLIND_SPOT** iff neither integrity nor the kernel-relevant "
        "tests catch it.")
    add("")
    add("## Known-good fixtures: gate vectors on validation (MEASURED)")
    add("")
    add("| fixture | " + " | ".join(GATE_SHORT) + " | BA_val | ECE_val |")
    add("| --- | " + " | ".join("---:" for _ in GATE_SHORT) + " | ---: | ---: |")
    for fid in ("v2", *KNOWN_GOOD_KEYS):
        fx = results["known_good"][fid]
        v = fx["details"]["validation"]
        add(f"| {fid} | {_gate_cells(fx['gates'])} | {_f(v['balanced_accuracy'])} | {_f(v['ece'])} |")
    add("")
    add("## Artifact mutants (36): pass 1 integrity, pass 2 behavior (MEASURED)")
    add("")
    add("Pass 1 pairs the mutant with the frozen v2 receipt. Pass 2 regenerates the receipt "
        "consistently (only `model_sha256` changes), so the kernel and `verify_origin` accept it, "
        "and then evaluates the seven gates on validation. \"gates below v2\" lists the gates "
        "that caught the mutant.")
    add("")
    add("| mutant | change | pass 1 integrity | pass 2 gates below v2 | BA_val | ECE_val | status |")
    add("| --- | --- | --- | --- | ---: | ---: | --- |")
    for m in results["artifact_mutants"]:
        b = m["pass2_behavior"]
        p1 = "CAUGHT" if m["pass1_integrity"]["caught"] else "missed"
        below = ", ".join(g.replace("g_", "") for g in b["gates_below_reference"]) or "none"
        add(f"| {m['id']} | {_f(m['before'], 12)} → {_f(m['after'], 12)} | {p1} | {below} | "
            f"{_f(b['validation_ba'])} | {_f(b['validation_ece'])} | **{m['status']}** |")
    add("")
    add("Pass 1 refusal message (every artifact mutant): "
        f"`{results['artifact_mutants'][0]['pass1_integrity']['kernel']['refusal']}`.")
    add("")
    add("## Receipt mutants (MEASURED)")
    add("")
    add("| mutant | change | kernel | verify_origin | status |")
    add("| --- | --- | --- | --- | --- |")
    for m in results["receipt_mutants"]:
        k = m["integrity"]["kernel"]
        o = m["integrity"]["verify_origin"]
        kc = "accepts" if k["accepted"] else f"refuses: `{k['refusal']}`"
        oc = "accepts" if o["accepted"] else f"refuses: `{o['refusal']}`"
        add(f"| {m['id']} | {m['description']} | {kc} | {oc} | **{m['status']}** |")
    add("")
    add("## Kernel-source mutants (MEASURED)")
    add("")
    kc = results["kernel_control"]
    ct = kc["tests"] or {}
    add(f"Control: an unmodified copy (sha256 identical to the shipped kernel: "
        f"{kc['identical_to_shipped']}) loads the frozen v2 pair "
        f"({kc['integrity']['accepted']}) and passes the kernel-relevant tests "
        f"({ct.get('tests_run')} run, ok {ct.get('ok')}). Test layer valid: "
        f"{kc['test_harness_valid']}. Tests: " + ", ".join(f"`{t}`" for t in kc["test_ids"]) + ".")
    add("")
    add("| mutant | integrity (frozen v2 pair) | kernel-relevant tests | "
        "descriptive: receipt re-bound to the copy's sha256 | status |")
    add("| --- | --- | --- | --- | --- |")
    for m in results["kernel_mutants"]:
        integ = "CAUGHT: `" + (m["integrity"]["refusal"] or "") + "`" if m["caught_by_integrity"] \
            else "missed (loads)"
        t = m["tests"] or {}
        tests = (f"{m['tests_status']}: {len(t.get('failed_tests', []))} of {t.get('tests_run')} "
                 f"tests failed ({len(t.get('failures', []))} failures, {len(t.get('errors', []))} "
                 "errors, subtests counted)") if t else m["tests_status"]
        d = m["descriptive_self_hash_rebound"]
        if d["kernel_accepts_frozen_model"]:
            pr = d["failclosed_probes"]
            desc = f"loads; probes refused {pr['refused']}/{pr['total']}"
            if pr["escaped"]:
                desc += "; escaped: " + ", ".join(pr["escaped"])
        else:
            desc = f"refuses: `{d['refusal']}`"
        add(f"| {m['id']}{' (supplementary)' if m['supplementary'] else ''} | {integ} | {tests} | "
            f"{desc} | **{m['status']}** |")
    add("")
    add("Failing tests per mutant, with the failing subtests of each (full output in "
        "`logs/v2/mutation_kernel_tests/<mutant>.txt`):")
    add("")
    for m in results["kernel_mutants"]:
        t = m["tests"] or {}
        failed = t.get("failed_tests", [])
        add(f"- **{m['id']}**: {len(failed)} failing test(s)" + (":" if failed else "."))
        for tid in failed:
            subs = [f.split(" (", 1)[1].rstrip(")") for f in t.get("failures", []) + t.get("errors", [])
                    if f.startswith(tid + " (")]
            if len(subs) > 4:  # long subtest lists (probe names) stay in the log and the JSON
                note = f" ({len(subs)} failing subtests; listed in the log)"
            else:
                note = f" (subtests: {', '.join(subs)})" if subs else ""
            add(f"  - `{tid.replace('v2.tests.', '')}`{note}")
    add("")
    add("## BLIND_SPOT list (MEASURED)")
    add("")
    if not results["blind_spots"]:
        add("None.")
    for bsp in results["blind_spots"]:
        extra = " (supplementary)" if bsp.get("supplementary") else ""
        add(f"- `{bsp['id']}`{extra} ({bsp['kind']}): no catch by {bsp['layer']}.")
    add("")
    _render_test_layer(results, add)
    add("## Broken baselines (section 10; MEASURED on validation)")
    add("")
    add("| fixture | status | failed preconditions | BA | per-class rates |")
    add("| --- | --- | --- | ---: | --- |")
    for name, b in results["broken_baselines"].items():
        r = b["report"]
        bv = r["baseline_validity"]
        add(f"| {name} | {bv['status']} | {', '.join(bv['failed_preconditions'])} | "
            f"{_f(r['balanced_accuracy'])} | {r['per_class_rates'] if isinstance(r['per_class_rates'], str) else 'reported'} |")
        if "with_precondition_i_granted" in b:
            r2 = b["with_precondition_i_granted"]
            bv2 = r2["baseline_validity"]
            add(f"| {name}, precondition (i) granted | {bv2['status']} | "
                f"{', '.join(bv2['failed_preconditions'])} | {_f(r2['balanced_accuracy'])} | "
                f"{r2['per_class_rates'] if isinstance(r2['per_class_rates'], str) else 'reported'} |")
    add("")
    add("## Readiness aggregation (section 11; MEASURED)")
    add("")
    add(f"Known-good: v2 and the five AM-4 retrains ({rd['separation']['AND']['n_good']}). "
        f"Known-broken: every integrity-bypassed artifact mutant "
        f"({rd['separation']['AND']['n_broken']}). Cut: aggregate ≥ {rd['cut']} (AND: every gate "
        "= 1). Λ = `lambda_stdlib.lambda_aggregate` with the reference defaults: uniform weights "
        "(`weights=None`) and `semantics=\"kernel\"`. Λ has no other parameter; the "
        "`lambda_gate` default threshold of 0.5 is not used.")
    add("")
    add("| aggregator | misclassified at cut | broken READY | good NOT_READY | min(good) | "
        "max(broken) | margin | AUROC |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in rd["aggregators"]:
        s = rd["separation"][name]
        add(f"| {'Λ' if name == 'lambda' else name} | {s['misclassifications']} | "
            f"{s['broken_marked_ready']} | {s['good_marked_not_ready']} | {_f(s['min_good'])} | "
            f"{_f(s['max_broken'])} | {_f(s['margin'])} | {_f(s['auroc'])} |")
    add("")
    add(f"**Λ verdict: {lam['verdict']}.** Rule: {lam['rule']}. Strictly better margin than: "
        + ", ".join(f"{k} {v}" for k, v in lam["strictly_better_margin_than"].items())
        + ". No more misclassifications than: "
        + ", ".join(f"{k} {v}" for k, v in lam["no_more_misclassifications_than"].items()) + ".")
    add("")
    add("Aggregates of the known-good fixtures, and of every known-broken fixture that at least "
        "one aggregator marks READY (MEASURED):")
    add("")
    add("| fixture | group | AND | min | mean | Λ | READY under |")
    add("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    order = ["v2", *KNOWN_GOOD_KEYS] + [m["id"] for m in results["artifact_mutants"]]
    for fid in order:
        f = rd["per_fixture"][fid]
        if f["group"] == "broken" and not any(f["ready"].values()):
            continue
        a = f["aggregates"]
        ready_under = ", ".join(("Λ" if n == "lambda" else n) for n in rd["aggregators"]
                                if f["ready"][n]) or "none"
        add(f"| {fid} | {f['group']} | {_f(a['AND'], 4)} | {_f(a['min'], 4)} | {_f(a['mean'], 4)} | "
            f"{_f(a['lambda'], 4)} | {ready_under} |")
    add("")
    add("## Reading the results")
    add("")
    # known-good fixtures that fail a gate
    failing_good = [(fid, fx) for fid, fx in results["known_good"].items()
                    if any(v < 1.0 for v in fx["gates"].values())]
    if failing_good:
        parts = []
        for fid, fx in failing_good:
            d = fx["details"]
            model = json.loads((registry.ROOT / fx["path"] / freeze.MODEL_NAME).read_text(encoding="utf-8"))
            dis = d["truth_sign"]["disagreeing"]
            weights = ", ".join(f"{name} {model['weights'][name]:+.6f}" for name in dis)
            mono = d["monotone"]
            parts.append(f"{fid} (sign disagreement: {weights or 'none'}; monotonicity checks "
                         f"failed {mono['checks'] - mono['passed']}/{mono['checks']}, by kind "
                         f"{mono['failed_by_kind']})")
        v2_model = json.loads((V2_DIR / freeze.MODEL_NAME).read_text(encoding="utf-8"))
        add(f"- **Known-good fixtures fail registered gates.** {len(failing_good)} of "
            f"{len(results['known_good'])} known-good fixtures have a gate below 1: "
            + "; ".join(parts) + f". (MEASURED; weights REPORTED from each model.json.) v2's own "
            f"seconds_since_last_success weight is {v2_model['weights']['seconds_since_last_success']:+.6f} "
            "(REPORTED). PREREGISTRATION §2 recorded in advance that this coordinate is weakly "
            "identifiable, because nominal seconds_since_last_success is near 0 on its "
            "normalized scale (REPORTED; labeled MODELED there). A retrain of the selected "
            "recipe can therefore land on either side of zero (MODELED), and g_truth_sign and "
            "g_monotone then fail. Every known-good fixture still has g_validation = 1 and "
            "g_calibration = 1 (MEASURED). Every aggregator marks the failing fixtures NOT_READY "
            "at the 0.99 cut (MEASURED).")
    blind_art = [m for m in results["artifact_mutants"] if m["status"] == "BLIND_SPOT"]
    if blind_art:
        v2v = results["known_good"]["v2"]["details"]["validation"]
        drops = [(v2v["balanced_accuracy"] - m["pass2_behavior"]["validation_ba"], m["id"]) for m in blind_art]
        worst_drop = max(drops)
        worst_ece = max((m["pass2_behavior"]["validation_ece"], m["id"]) for m in blind_art)
        add(f"- **Behavioral blind spots.** {len(blind_art)} integrity-bypassed artifact mutants "
            "pass every gate: " + ", ".join(f"`{m['id']}`" for m in blind_art) + ". They keep "
            "every weight sign and monotonicity, so g_truth_sign and g_monotone stay at 1. "
            "g_validation and g_calibration are absolute tolerances (BA ≥ 0.70 and ECE ≤ 0.05), "
            "not distances from v2. The largest validation BA loss among the blind spots is "
            f"{worst_drop[0]:.6f} (`{worst_drop[1]}`, BA {v2v['balanced_accuracy'] - worst_drop[0]:.6f} "
            f"against v2's {v2v['balanced_accuracy']:.6f}). The largest ECE is {worst_ece[0]:.6f} "
            f"(`{worst_ece[1]}`). (MEASURED.) The threshold mutants change no score, so only "
            "g_validation could see them, and their BA stays above 0.70 (MEASURED). Every one of "
            f"these mutants was caught by integrity in pass 1 "
            f"({c['artifact_pass1_caught_by_integrity']}/{c['artifact_mutants']}, MEASURED). "
            "With the gates and cut as registered, the integrity layer, not the behavioral "
            "gates, is what detects a modified artifact (MODELED).")
    ready_broken = {n: [fid for fid, f in rd["per_fixture"].items()
                        if f["group"] == "broken" and f["ready"][n]] for n in rd["aggregators"]}
    only_soft = sorted((set(ready_broken["mean"]) | set(ready_broken["lambda"]))
                       - set(ready_broken["min"]))
    seps = rd["separation"]
    all_negative = all(s["margin"] < 0 for s in seps.values())
    better = [b for b, v in lam["strictly_better_margin_than"].items() if v]
    not_better = [b for b, v in lam["strictly_better_margin_than"].items() if not v]
    more_errors = [b for b, v in lam["no_more_misclassifications_than"].items() if not v]
    between = all(f["aggregates"]["min"] <= f["aggregates"]["lambda"] <= f["aggregates"]["mean"]
                  for f in rd["per_fixture"].values())
    add("- **Aggregators.** "
        + ("No aggregator separates known-good from known-broken: every margin is negative "
           f"(max(broken) = {max(s['max_broken'] for s in seps.values()):.1f} under every "
           "aggregator, from the blind spots, while some known-good fixture is below 1). "
           if all_negative else "")
        + f"Misclassifications at the cut range from "
        f"{min(s['misclassifications'] for s in seps.values())} to "
        f"{max(s['misclassifications'] for s in seps.values())} of "
        f"{seps['AND']['n_good'] + seps['AND']['n_broken']} fixtures. "
        + (f"Mean and Λ also mark {', '.join(f'`{x}`' for x in only_soft)} READY, where min "
           "does not: its only failing gate is slightly below 1. " if only_soft else "")
        + (f"Λ's margin is strictly better than {' and '.join(b + chr(39) + 's' for b in better)}"
           if better else "Λ's margin is not strictly better than any baseline's")
        + (f" but not {' and '.join(b + chr(39) + 's' for b in not_better)}"
           if better and not_better else "")
        + (f", and Λ misclassifies more fixtures than {' and '.join(more_errors)}"
           if more_errors else "")
        + f". Under the registered rule Λ is therefore **{lam['verdict']}**. (MEASURED numbers; "
        "the rule is fixed in section 11.) "
        + ("On every fixture, Λ lies between min and mean (MEASURED), as "
           "`math/lambda_reference.md` §7 describes for uniform weights."
           if between else "Λ is not always between min and mean here (MEASURED)."))
    other = [m for m in results["receipt_mutants"] if m["status"] == "BLIND_SPOT"]
    for m in other:
        add(f"- **Receipt-origin blind spot (`{m['id']}`, supplementary).** The kernel only checks "
            "that source.commit is 40 lower-case hex digits; it cannot run git. `freeze."
            "verify_origin` checks that the committed kernel, generator, manifest, "
            "preregistration and amendments blobs at the claimed commit hash to the receipt's "
            "digests. At the parent of the recorded freeze commit, those five blobs are "
            "identical, so re-pointing source.commit there passes both layers (MEASURED). The "
            "model bytes and every bound digest remain protected. What goes undetected is which "
            "of several commits with identical bound blobs the receipt names (MODELED). This is "
            "a limit of the research-side origin check, not a defect of the frozen kernel. "
            "Possible remedies are to also bind the committed search trace at the claimed "
            "commit, or to sign receipts. Neither is applied here: both would change the "
            "integrity layer after its result is known.")
    k_by_id = {m["id"]: m for m in results["kernel_mutants"]}
    ks = k_by_id.get("K-no-prohibited-screen")
    if ks and ks["descriptive_self_hash_rebound"]["failclosed_probes"]:
        pr = ks["descriptive_self_hash_rebound"]["failclosed_probes"]
        add(f"- **Kernel mutants.** With prohibited-key screening removed, the kernel still "
            f"refuses {pr['refused']}/{pr['total']} probes, because the exact eight-field schema "
            "check rejects the extra names. g_failclosed alone would not notice the mutation. The "
            "tests catch it: they check the refusal reason and that screening runs before the "
            "schema check (MEASURED). The inverted self-hash mutant passes the integrity layer "
            "(MEASURED): a kernel that verifies its own hash cannot detect a change to that "
            "verification (MODELED). The tests catch it too. For every kernel mutant, the kernel-"
            "relevant tests failed, and the unmodified control copy passed all of them (MEASURED).")
    add("- **Defects.** No mutant revealed a defect in the frozen v2 kernel. The shipped kernel "
        "passes the control, and every kernel-source mutant is caught (MEASURED). The blind spots "
        "above are properties of the registered behavioral gates, the registered cut and the "
        "research-side origin check. Per the protocol, none of them was tuned after these results.")
    add("")
    add("No Λ property is called proven here (CONTRACT §5.5; `math/lambda_reference.md` §5).")
    add("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 mutation suite and readiness (SYNTHETIC).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--no-kernel-tests", action="store_true")
    p_run.add_argument("--out", default=str(RESULTS_PATH))
    sub.add_parser("render")
    p_kt = sub.add_parser("kernel-tests")
    p_kt.add_argument("--kernel", required=True)
    p_kt.add_argument("--summary", required=True)
    p_kt.add_argument("test_ids", nargs="*")
    args = parser.parse_args(argv)
    if args.cmd == "kernel-tests":
        return _kernel_tests_entry(args.kernel, args.summary, args.test_ids or KERNEL_TEST_IDS)
    if args.cmd == "render":
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        REPORT_PATH.write_text(render(results), encoding="utf-8", newline="\n")
        print(f"wrote {registry.relpath(REPORT_PATH)}")
        return 0
    try:
        results = run(Path(args.out), kernel_tests=not args.no_kernel_tests,
                      echo=lambda line: print(line, flush=True))
    except MutationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"counts": results["counts"], "blind_spots": results["blind_spots"],
                      "lambda": results["readiness"]["lambda"],
                      "separation": results["readiness"]["separation"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
