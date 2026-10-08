"""Registered hyperparameter search for OAC Ops Health v2 (PREREGISTRATION section 6; AM-2).

SYNTHETIC.  Standard library only.  Nothing here reads a sealed split: the four registered open
splits are opened through ``sealed_guard.open_split`` with their registered purposes
(train TRAIN, calibration CALIBRATION, conformal CONFORMAL, validation REGISTERED_SEARCH).

Search space (fixed, 12 trials, visited in this order):
    l2 in (0, 1e-4, 1e-3, 1e-2)  x  calibrator in (none, platt, isotonic)
Loop: ``math/ouroboros_adapter.run_bounded_search`` with the hard cap 12.  The only accepted
stop reasons are SEARCH_SPACE_EXHAUSTED and CAP_REACHED; anything else raises.  The adapter
writes one governed receipt per trial (math/receipt_adapter.py, governed-receipt-spec shape)
plus one loop-summary receipt into ``runs/receipts/<run_id>/``.  Receipts attest the integrity
and origin of each trial record only, never the quality of a candidate.

One trial = ``pipeline.build_candidate``:
    IRLS fit on train (lr.fit), calibrator fit on calibration, threshold chosen on validation by
    the section 6 rule (max BA, ties to higher F1, then t closer to 0.5; exact fractions),
    Mondrian conformal at alpha 0.10 on conformal.
The trial record carries the validation BA (also as an exact fraction), Brier, ECE, F1, log
loss and ROC AUC of the frozen candidate at its own threshold, the IRLS and Platt convergence
diagnostics, and the sha256 of the candidate's canonical model.json.

Eligibility (AM-2): a trial whose IRLS fit, or whose Platt fit, stopped with CAP_REACHED or
LINE_SEARCH_FAILED is run, receipted and reported, but cannot be selected.  If no trial is
eligible, v2 does not exist.

Selection (section 6): maximize validation BA at the trial's own threshold (exact fractions);
ties to lower validation Brier, then lower validation ECE, then larger l2, then calibrator order
none < platt < isotonic.

ABL-conformal (section 9): for the selected candidate, coverage of the Mondrian prediction sets
on VALIDATION; the conformal layer is kept only if marginal coverage >= 0.88 and each class's
coverage >= 0.85.  Selective metrics with and without abstention are recorded alongside.

Receipts-directory digest: sha256 over the UTF-8 text of sha256sum-style lines
"<sha256 of file bytes>  <file name>\\n" for every file in the directory, sorted by name.

Usage (from the repository root; runs the registered search ONCE, refuses to overwrite):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.search --run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from . import calibration, freeze, generator, kernel_bridge, metrics, pipeline, registry
from .dataio import labels_of

L2_GRID = (0.0, 1e-4, 1e-3, 1e-2)
L2_TAGS = {0.0: "0", 1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2"}
CALIBRATORS = calibration.KINDS  # ("none", "platt", "isotonic"): also the tie-break order
HARD_CAP = 12
ALLOWED_STOP_REASONS = ("SEARCH_SPACE_EXHAUSTED", "CAP_REACHED")
CONFORMAL_MARGINAL_MIN = 0.88
CONFORMAL_CLASS_MIN = 0.85

AMENDMENTS_PATH = registry.V2 / "AMENDMENTS.md"
# AM-1..AM-6, committed in ed3fc88 before any search; the registered search refuses to run if
# the file on disk differs (an amendment appended after that point needs a new decision).
AMENDMENTS_SHA256 = "5152cd7f816c41e810861bbcd6489dee1d2598da89a36f6b418ef4a9867f181d"
RESULTS_DIR = registry.V2 / "results"
TRACE_PATH = RESULTS_DIR / "search_trace.json"
CANDIDATES_DIR = RESULTS_DIR / "search_candidates"
RECEIPTS_ROOT = registry.RUNS_DIR / "receipts"
DEFAULT_RUN_ID = "v2-registered-search"
MATH_DIR = registry.ROOT / "math"
OUROBOROS_PATH = MATH_DIR / "ouroboros_adapter.py"
RECEIPT_ADAPTER_PATH = MATH_DIR / "receipt_adapter.py"
TRACE_SCHEMA = "szl-oac/ops-health-search-trace/v2"
LOOP_LABEL = "oac-ops-health-v2.registered-search"
RECEIPT_NS = "oac-frontier"
RECEIPT_ACTION = "hyperparameter-trial"

# role (pipeline) -> (registered split, registered purpose)
REGISTERED_SPLITS = {
    "train": ("train", "TRAIN"),
    "calibration": ("calibration", "CALIBRATION"),
    "conformal": ("conformal", "CONFORMAL"),
    "threshold": ("validation", "REGISTERED_SEARCH"),
}
SELECTION_RULE = (
    "eligible trials only (AM-2: IRLS and Platt fits converged); maximize validation balanced "
    "accuracy at the trial's own threshold (exact fractions); ties to lower validation Brier, "
    "then lower validation ECE, then larger l2, then calibrator order none < platt < isotonic"
)
SELECTION_KEYS_IN_ORDER = (
    "validation_balanced_accuracy", "validation_brier", "validation_ece", "l2", "calibrator_order",
)
CONFORMAL_RULE = (
    "keep the conformal layer only if validation coverage is >= 0.88 marginal and >= 0.85 for "
    "each class (PREREGISTRATION section 9, ABL-conformal)"
)


class SearchRefused(RuntimeError):
    """A precondition of the registered search does not hold."""


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def search_space() -> list[dict]:
    """The registered 12-trial space, l2-major, calibrator-minor, in registered order."""
    space = []
    for l2 in L2_GRID:
        for cal in CALIBRATORS:
            index = len(space) + 1
            space.append(
                {"trial_id": f"T{index:02d}_l2_{L2_TAGS[l2]}_{cal}", "l2": l2, "calibrator": cal}
            )
    return space


def trial_id_for(l2: float, calibrator: str) -> str:
    for cand in search_space():
        if cand["l2"] == l2 and cand["calibrator"] == calibrator:
            return cand["trial_id"]
    raise KeyError(f"(l2={l2!r}, calibrator={calibrator!r}) is not in the registered space")


_ADAPTER_CACHE: dict[str, ModuleType] = {}


def ouroboros() -> ModuleType:
    """math/ouroboros_adapter.py, loaded by path (the directory name shadows stdlib math)."""
    if "mod" not in _ADAPTER_CACHE:
        spec = importlib.util.spec_from_file_location("oac_ouroboros_adapter", OUROBOROS_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {OUROBOROS_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["oac_ouroboros_adapter"] = module
        spec.loader.exec_module(module)
        _ADAPTER_CACHE["mod"] = module
    return _ADAPTER_CACHE["mod"]


def directory_digest(directory: Path | str) -> dict:
    """sha256 over sorted sha256sum-style lines of every file in ``directory`` (flat only)."""
    directory = Path(directory)
    files = {}
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_dir():
            raise ValueError(f"unexpected subdirectory in receipts dir: {path.name}")
        files[path.name] = sha256_file(path)
    text = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    return {
        "sha256": sha256_bytes(text.encode("utf-8")),
        "files": files,
        "method": "sha256 of UTF-8 lines '<sha256>  <name>\\n' over every file, sorted by name",
    }


# --------------------------------------------------------------------------------------
# evaluation on the threshold (validation) rows
# --------------------------------------------------------------------------------------
def _evaluations(artifact: Mapping, rows: Sequence[Mapping]) -> list[dict]:
    scorer = kernel_bridge.kernel().ArtifactScorer(artifact)
    return pipeline.score_rows(scorer, rows)


def exact_balanced_accuracy(conf: Mapping[str, int]) -> Fraction:
    pos = conf["tp"] + conf["fn"]
    neg = conf["tn"] + conf["fp"]
    if pos == 0 or neg == 0:
        raise ValueError("balanced accuracy needs both classes")
    return (Fraction(conf["tp"], pos) + Fraction(conf["tn"], neg)) / 2


def decision_report(labels: Sequence[int], scores: Sequence[float], decisions: Sequence[int],
                    threshold: float) -> dict:
    """Threshold-decision and score metrics (the candidate's own metrics; not a baseline)."""
    conf = metrics.confusion(labels, decisions)
    rates = metrics.rates_from_counts(conf["tp"], conf["fp"], conf["tn"], conf["fn"])
    ba = exact_balanced_accuracy(conf)
    return {
        "rows": len(labels),
        "positives": conf["tp"] + conf["fn"],
        "negatives": conf["tn"] + conf["fp"],
        "threshold": threshold,
        "confusion": conf,
        "balanced_accuracy": float(ba),
        "balanced_accuracy_exact": [ba.numerator, ba.denominator],
        "f1": rates["f1"],
        "precision": rates["precision"],
        "recall": rates["recall"],
        "specificity": rates["specificity"],
        "brier": metrics.brier(labels, scores),
        "ece": metrics.ece(labels, scores),
        "log_loss": metrics.log_loss(labels, scores),
        "roc_auc": metrics.roc_auc(labels, scores),
    }


def validation_report(artifact: Mapping, rows: Sequence[Mapping]) -> dict:
    evals = _evaluations(artifact, rows)
    labels = labels_of(rows)
    scores = [e["calibrated_score"] for e in evals]
    decisions = [1 if e["threshold_decision"] else 0 for e in evals]
    return decision_report(labels, scores, decisions, artifact["decision_threshold"])


def eligibility(calibrator: str, lr_diag: Mapping, cal_diag: Mapping) -> list[str]:
    """AM-2 reasons for ineligibility (empty list = eligible)."""
    reasons = []
    if lr_diag.get("converged") is not True or lr_diag.get("stop_reason") != "CONVERGED":
        reasons.append(f"IRLS not converged: {lr_diag.get('stop_reason')}")
    if calibrator == "platt" and (
        cal_diag.get("converged") is not True or cal_diag.get("stop_reason") != "CONVERGED"
    ):
        reasons.append(f"Platt not converged: {cal_diag.get('stop_reason')}")
    return reasons


def run_trial(candidate: Mapping, rows: Mapping[str, Sequence[Mapping]],
              split_names: Mapping[str, str]) -> tuple[dict, bytes]:
    """Build one candidate; returns (JSON-able trial record, canonical model.json bytes)."""
    artifact, diag = pipeline.build_candidate(
        train_rows=rows["train"],
        calibration_rows=rows["calibration"],
        conformal_rows=rows["conformal"],
        threshold_rows=rows["threshold"],
        l2=candidate["l2"],
        calibrator=candidate["calibrator"],
        split_names=split_names,
    )
    model_bytes = freeze.canonical_model_bytes(artifact)
    val = validation_report(artifact, rows["threshold"])
    if val["confusion"] != diag["threshold"]["confusion"]:
        raise RuntimeError("validation confusion differs from the threshold selection's; refusing")
    reasons = eligibility(candidate["calibrator"], diag["lr"], diag["calibrator"])
    cal_diag = dict(diag["calibrator"])
    record = {
        "trial_id": candidate["trial_id"],
        "l2": candidate["l2"],
        "calibrator": candidate["calibrator"],
        "model_sha256": sha256_bytes(model_bytes),
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "validation": val,
        "fit": {"irls": dict(diag["lr"]), "calibrator": cal_diag},
        "threshold_selection": {
            "threshold": diag["threshold"]["threshold"],
            "grid_index": diag["threshold"]["grid_index"],
            "balanced_accuracy": diag["threshold"]["balanced_accuracy"],
            "f1": diag["threshold"]["f1"],
        },
        "conformal_fit": {
            "q_hat": dict(artifact["conformal"]["q_hat"]),
            "n_calibration": dict(artifact["conformal"]["n_calibration"]),
            "rank": dict(diag["conformal"]["rank"]),
            "q_hat_exact": dict(diag["conformal"]["q_hat_exact"]),
        },
        "intercept": artifact["intercept"],
        "weights": dict(artifact["weights"]),
        "labels": "MEASURED on the threshold (validation) rows; SYNTHETIC",
    }
    return record, model_bytes


# --------------------------------------------------------------------------------------
# selection and the ABL-conformal decision
# --------------------------------------------------------------------------------------
def selection_rank(record: Mapping) -> tuple:
    """Smaller is better (sort key)."""
    num, den = record["validation"]["balanced_accuracy_exact"]
    return (
        -Fraction(num, den),
        record["validation"]["brier"],
        record["validation"]["ece"],
        -float(record["l2"]),
        calibration.KIND_ORDER[record["calibrator"]],
    )


def select(records: Sequence[Mapping]) -> dict:
    eligible = [r for r in records if r["eligible"]]
    ineligible = [
        {"trial_id": r["trial_id"], "reasons": list(r["ineligible_reasons"])}
        for r in records if not r["eligible"]
    ]
    out = {
        "rule": SELECTION_RULE,
        "eligible_trials": [r["trial_id"] for r in eligible],
        "ineligible_trials": ineligible,
    }
    if not eligible:
        out.update({"selected_trial": None, "status": "NO_ELIGIBLE_TRIAL", "ranking": [],
                    "decided_by": None})
        return out
    ranked = sorted(eligible, key=selection_rank)
    decided_by = "only_eligible_trial"
    if len(ranked) > 1:
        first, second = selection_rank(ranked[0]), selection_rank(ranked[1])
        decided_by = next(
            (name for name, a, b in zip(SELECTION_KEYS_IN_ORDER, first, second) if a != b), None
        )
        if decided_by is None:
            raise RuntimeError("two trials share every selection key; the space has a duplicate")
    out.update({
        "selected_trial": ranked[0]["trial_id"],
        "status": "SELECTED",
        "ranking": [r["trial_id"] for r in ranked],
        "decided_by": decided_by,
    })
    return out


def conformal_keep(coverage: Mapping) -> bool:
    marginal = coverage.get("marginal")
    per_class = coverage.get("per_class", {})
    if marginal is None or marginal < CONFORMAL_MARGINAL_MIN:
        return False
    for cls in ("0", "1"):
        value = per_class.get(cls)
        if value is None or value < CONFORMAL_CLASS_MIN:
            return False
    return True


def conformal_decision(artifact: Mapping, rows: Sequence[Mapping]) -> dict:
    """ABL-conformal on the threshold (validation) rows for one candidate."""
    evals = _evaluations(artifact, rows)
    labels = labels_of(rows)
    cov = metrics.coverage(labels, [e["prediction_set"] for e in evals])
    with_abstention = metrics.selective(
        labels, [e["advisory"] for e in evals], [e["abstain_reason"] for e in evals]
    )
    decisions = [1 if e["threshold_decision"] else 0 for e in evals]
    keep = conformal_keep(cov)
    return {
        "alpha": artifact["conformal"]["alpha"],
        "q_hat": dict(artifact["conformal"]["q_hat"]),
        "n_calibration": dict(artifact["conformal"]["n_calibration"]),
        "coverage": cov,
        "with_abstention": with_abstention,
        "without_abstention": {
            "rows": len(labels),
            "abstention_rate": 0.0,
            "balanced_accuracy": metrics.balanced_accuracy(labels, decisions),
        },
        "rule": CONFORMAL_RULE,
        "thresholds": {"marginal": CONFORMAL_MARGINAL_MIN, "per_class": CONFORMAL_CLASS_MIN},
        "keep": keep,
        "labels": "MEASURED on the threshold (validation) rows; SYNTHETIC",
    }


# --------------------------------------------------------------------------------------
# the bounded search
# --------------------------------------------------------------------------------------
def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def run_search(
    rows: Mapping[str, Sequence[Mapping]],
    split_names: Mapping[str, str],
    *,
    receipts_dir: Path | str,
    run_id: str,
    candidates_dir: Path | str | None = None,
    ts_fn: Callable[[], Any] | None = None,
    origin: Mapping | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, bytes]]:
    """Run the 12-trial bounded search.  Returns (search record, {trial_id: model bytes}).

    Every trial's receipt is written as the trial completes; with ``candidates_dir`` each
    trial's canonical model.json is also written (exclusively) as it completes.
    """
    if set(rows) != set(pipeline.SPLIT_ROLES) or set(split_names) != set(pipeline.SPLIT_ROLES):
        raise ValueError(f"rows and split_names must name {pipeline.SPLIT_ROLES}")
    ob = ouroboros()
    space = search_space()
    models: dict[str, bytes] = {}

    def trial_fn(candidate, iteration):
        started = time.perf_counter()
        record, model_bytes = run_trial(candidate, rows, split_names)
        models[candidate["trial_id"]] = model_bytes
        if candidates_dir is not None:
            _write_exclusive(Path(candidates_dir) / f"{candidate['trial_id']}.model.json",
                             model_bytes)
        if progress is not None:
            v = record["validation"]
            progress(
                f"trial {iteration + 1:2d}/{len(space)} {candidate['trial_id']:<22} "
                f"eligible={record['eligible']} t={v['threshold']:.2f} "
                f"BA={v['balanced_accuracy']:.6f} Brier={v['brier']:.6f} ECE={v['ece']:.6f} "
                f"F1={v['f1']:.6f} irls={record['fit']['irls']['stop_reason']}"
                f"/{record['fit']['irls']['iterations']} "
                f"({time.perf_counter() - started:.1f}s)"
            )
        return record

    loop = ob.run_bounded_search(
        trial_fn, space, max_iterations=HARD_CAP, receipts_dir=str(receipts_dir), run_id=run_id,
        label=LOOP_LABEL, ns=RECEIPT_NS, action=RECEIPT_ACTION, ts_fn=ts_fn,
        origin=dict(origin) if origin else None,
    )
    if loop["stop_reason"] not in ALLOWED_STOP_REASONS:
        raise RuntimeError(f"registered loop stopped with {loop['stop_reason']}; refusing")
    records = [step["result"] for step in loop["steps"]]
    for record in records:
        if sha256_bytes(models[record["trial_id"]]) != record["model_sha256"]:
            raise RuntimeError("candidate bytes differ from the receipted model_sha256")
    verify_errors = ob.verify_run_dir(str(receipts_dir))
    selection = select(records)
    conformal_block = None
    if selection["selected_trial"] is not None:
        selected = json.loads(models[selection["selected_trial"]].decode("utf-8"))
        conformal_block = conformal_decision(selected, rows["threshold"])
        conformal_block["trial_id"] = selection["selected_trial"]
    record = {
        "loop": {
            "run_id": loop["run_id"],
            "label": loop["label"],
            "stop_reason": loop["stop_reason"],
            "iterations_run": loop["iterations_run"],
            "max_iterations": loop["max_iterations"],
            "search_space_size": loop["search_space_size"],
            "search_space_digest": loop["search_space_digest"],
            "order": loop["order"],
            "summary_receipt_file": loop["summary_receipt_file"],
            "summary_receipt_digest": loop["summary_receipt_digest"],
            "total_duration_ms": loop["total_duration_ms"],
            "semantics": loop["semantics"],
        },
        "search_space": space,
        "trials": [
            {
                "iteration": step["iteration"],
                "receipt_file": step["receipt_file"],
                "receipt_digest": step["receipt_digest"],
                "duration_ms": step["duration_ms"],
                "record": step["result"],
            }
            for step in loop["steps"]
        ],
        "receipts": {
            "dir": Path(receipts_dir).as_posix(),
            "verify_run_dir_errors": verify_errors,
            "verified": not verify_errors,
            "dir_digest": directory_digest(receipts_dir),
        },
        "selection": selection,
        "conformal": conformal_block,
    }
    return record, models


# --------------------------------------------------------------------------------------
# the registered run (CLI)
# --------------------------------------------------------------------------------------
def load_registered_rows(guard=None) -> tuple[dict, dict]:
    """Open the four registered open splits through the sealed guard (never a sealed split)."""
    if guard is None:
        from . import sealed_guard as guard  # noqa: PLC0415
    rows = {role: guard.open_split(name, purpose)
            for role, (name, purpose) in REGISTERED_SPLITS.items()}
    names = {role: name for role, (name, _purpose) in REGISTERED_SPLITS.items()}
    return rows, names


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(registry.ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def protocol_state() -> dict:
    status = _git("status", "--porcelain", "--", "v2/research", "v2/ops-health/ops_health.py",
                  "v2/tests", "v2/PREREGISTRATION.md", "v2/AMENDMENTS.md", "v2/data")
    dirty = [line for line in status.splitlines() if line.strip()]
    manifest = registry.MANIFEST_PATH.read_bytes()
    return {
        "git_head": _git("rev-parse", "HEAD").strip(),
        "tree_clean": not dirty,
        "dirty": dirty,
        "preregistration_sha256": sha256_file(registry.PREREG_PATH),
        "amendments_sha256": sha256_file(AMENDMENTS_PATH),
        "data_manifest_sha256": sha256_bytes(manifest),
        "generator_sha256": generator.source_sha256(),
        "kernel_sha256": sha256_file(registry.KERNEL_PATH),
        "search_module_sha256": sha256_file(Path(__file__)),
        "pipeline_module_sha256": sha256_file(Path(pipeline.__file__)),
        "adapters": {
            "math/ouroboros_adapter.py": sha256_file(OUROBOROS_PATH),
            "math/receipt_adapter.py": sha256_file(RECEIPT_ADAPTER_PATH),
            "git_tracked": False,
            "note": "math/ is not tracked by git in this repository; the sha256 values above "
                    "identify the adapter bytes that ran",
        },
        "test_opened_marker_present": (registry.RUNS_DIR / registry.TEST_OPENED_NAME).exists(),
        "python": platform.python_version(),
    }


def registered_preconditions(state: Mapping, run_id: str) -> None:
    problems = []
    if state["preregistration_sha256"] != registry.PREREG_SHA256:
        problems.append("PREREGISTRATION.md differs from the registered digest")
    if state["amendments_sha256"] != AMENDMENTS_SHA256:
        problems.append("AMENDMENTS.md differs from the AM-1..AM-6 digest recorded before search")
    if not state["tree_clean"]:
        problems.append(f"working tree not clean: {state['dirty']}")
    if state["test_opened_marker_present"]:
        problems.append("runs/TEST_OPENED.json exists: the sealed set was opened; no search")
    if TRACE_PATH.exists():
        problems.append(f"{registry.relpath(TRACE_PATH)} exists: the registered search already ran")
    receipts_dir = RECEIPTS_ROOT / run_id
    if receipts_dir.exists() and any(receipts_dir.iterdir()):
        problems.append(f"{receipts_dir} is not empty")
    candidates = CANDIDATES_DIR
    if candidates.exists() and any(candidates.iterdir()):
        problems.append(f"{candidates} is not empty")
    if problems:
        raise SearchRefused("; ".join(problems))


def run_registered(run_id: str = DEFAULT_RUN_ID) -> dict:
    state = protocol_state()
    registered_preconditions(state, run_id)
    print(f"registered search | run_id {run_id} | HEAD {state['git_head']} | python "
          f"{state['python']} | SYNTHETIC", flush=True)
    rows, names = load_registered_rows()
    for role in pipeline.SPLIT_ROLES:
        print(f"  {role:<11} <- {names[role]:<11} rows={len(rows[role])} "
              f"sha256={pipeline.rows_sha256(rows[role])}", flush=True)
    receipts_dir = RECEIPTS_ROOT / run_id
    origin = {
        "procedure": "PREREGISTRATION section 6 registered search (AM-2 eligibility)",
        "git_head": state["git_head"],
        "tree_clean": state["tree_clean"],
        "preregistration_sha256": state["preregistration_sha256"],
        "amendments_sha256": state["amendments_sha256"],
        "data_manifest_sha256": state["data_manifest_sha256"],
        "kernel_sha256": state["kernel_sha256"],
        "search_module_sha256": state["search_module_sha256"],
    }
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record, _models = run_search(
        rows, names, receipts_dir=receipts_dir, run_id=run_id, candidates_dir=CANDIDATES_DIR,
        origin=origin, progress=lambda line: print("  " + line, flush=True),
    )
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace = {
        "schema": TRACE_SCHEMA,
        "synthetic": True,
        "started_utc": started,
        "finished_utc": finished,
        "protocol": state,
        "splits": {
            role: {"name": names[role], "rows": len(rows[role]),
                   "sha256": pipeline.rows_sha256(rows[role]),
                   "purpose": REGISTERED_SPLITS[role][1]}
            for role in pipeline.SPLIT_ROLES
        },
        "receipts_dir_relpath": registry.relpath(receipts_dir),
        "candidates_dir_relpath": registry.relpath(CANDIDATES_DIR),
        "truth_labels": {
            "metrics": "MEASURED on the registered validation split (threshold rows); SYNTHETIC",
            "receipts": "attest integrity and origin of each trial record only; not quality",
        },
        **record,
    }
    data = (json.dumps(trace, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            + "\n").encode("utf-8")
    _write_exclusive(TRACE_PATH, data)
    sel = trace["selection"]
    print(f"stop_reason {trace['loop']['stop_reason']} | iterations {trace['loop']['iterations_run']}"
          f" | receipts verified {trace['receipts']['verified']} "
          f"({len(trace['receipts']['verify_run_dir_errors'])} errors)", flush=True)
    print(f"receipts dir sha256 {trace['receipts']['dir_digest']['sha256']}", flush=True)
    print(f"ineligible {sel['ineligible_trials']}", flush=True)
    print(f"selected {sel['selected_trial']} (decided by {sel['decided_by']})", flush=True)
    if trace["conformal"] is not None:
        c = trace["conformal"]
        print(f"conformal coverage marginal {c['coverage']['marginal']:.6f} per_class "
              f"{c['coverage']['per_class']} keep {c['keep']}", flush=True)
    print(f"search_trace.json sha256 {sha256_bytes(data)}", flush=True)
    return trace


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registered v2 search (SYNTHETIC).")
    parser.add_argument("--run", action="store_true", help="run the registered search once")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    if not args.run:
        print(__doc__)
        return 0
    try:
        run_registered(args.run_id)
    except SearchRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
