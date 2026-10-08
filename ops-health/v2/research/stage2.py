"""Stage-2 freezing after the registered search: v2, ablation contenders, known-good retrains.

SYNTHETIC.  Standard library only.  Nothing here reads a sealed split.  Every step reads the
registered search trace ``v2/results/search_trace.json`` and first re-verifies it:
  * the receipts directory re-verifies with ``ouroboros_adapter.verify_run_dir`` and its digest
    equals the one recorded in the trace;
  * the candidate model.json of every trial it uses hashes to the trial's receipted
    model_sha256.
Freezing goes through ``freeze.freeze`` (clean-tree rule, HEAD as source.commit, exclusive
writes) and each receipt is checked with ``freeze.verify_origin``.  Every receipt's selection
block carries the search provenance: the selected trial, the sha256 of search_trace.json, the
receipts-directory digest, the sha256 of v2/AMENDMENTS.md recorded by the search, and the
ABL-conformal keep decision.

Steps (CLI subcommands; each refuses to overwrite what it already froze):
  freeze-v2   the selected trial -> v2/ops-health/{model.json, artifact_receipt.json,
              example_input.json}; the candidate is rebuilt from the registered splits first and
              must be byte-identical to the searched one.
  contenders  section 9 ablations -> v2/results/contenders/<name>/ and INDEX.json:
              ABL-data        the selected recipe on the first 768 train rows;
              ABL-optimizer   v1's exact GD (2,400 epochs, lr 0.42, l2 0.01) on the full train
                              split, threshold by v1's own rule on validation (params file);
              ABL-calibration the selected model without its calibrator, threshold re-selected
                              on validation (NOT_APPLICABLE when the selected calibrator is none);
              ABL-conformal   decision only (from the search trace).
  known-good  AM-4 retrains k = 1..5 -> v2/results/known_good/k<k>/ and INDEX.json.

Usage (from the repository root):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.stage2 freeze-v2
    PYTHONUTF8=1 py -3.12 -B -m v2.research.stage2 contenders
    PYTHONUTF8=1 py -3.12 -B -m v2.research.stage2 known-good
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from operator import mul
from pathlib import Path
from typing import Mapping, Sequence

from . import freeze, generator, kernel_bridge, pipeline, registry, search, v1_gd
from .dataio import labels_of

V2_OPS_DIR = registry.V2 / "ops-health"
EXAMPLE_INPUT_NAME = "example_input.json"
CONTENDERS_DIR = search.RESULTS_DIR / "contenders"
KNOWN_GOOD_DIR = search.RESULTS_DIR / "known_good"
INDEX_NAME = "INDEX.json"
ABL_DATA_ROWS = 768
KNOWN_GOOD_KS = (1, 2, 3, 4, 5)
KNOWN_GOOD_TRAIN_ROWS = 16384
RETRAIN_SHARED_ROLES = ("calibration", "conformal", "threshold")
V1_THRESHOLD_STEPS = range(5, 96)  # v1 choose_threshold: t = step / 100, step 5..95
# The v1 example input values (gates.VALID_FEATURES), wrapped as the v2 kernel requires.
EXAMPLE_INPUT = {
    "features": {
        "listener_running": 1,
        "tls_enabled": 1,
        "peer_allowlist_configured": 1,
        "queue_utilization": 0.12,
        "consecutive_failures": 0,
        "seconds_since_last_success": 14.0,
        "ledger_integrity_ok": 1,
        "configuration_valid": 1,
    }
}


class Stage2Refused(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            + "\n").encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _rel(path: Path | str) -> str:
    return registry.relpath(Path(path))


# --------------------------------------------------------------------------------------
# the search trace
# --------------------------------------------------------------------------------------
def load_trace(path: Path = search.TRACE_PATH) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8")), sha256_bytes(data)


def trial_record(trace: Mapping, trial_id: str) -> dict:
    for trial in trace["trials"]:
        if trial["record"]["trial_id"] == trial_id:
            return trial["record"]
    raise Stage2Refused(f"trial {trial_id!r} is not in the search trace")


def check_trace(trace: Mapping, root: Path = registry.ROOT) -> dict:
    """Re-verify the receipts directory and its digest; refuse if v2 does not exist."""
    receipts_dir = root / trace["receipts_dir_relpath"]
    errors = search.ouroboros().verify_run_dir(str(receipts_dir))
    if errors:
        raise Stage2Refused(f"receipts dir does not verify: {errors[:3]}")
    digest = search.directory_digest(receipts_dir)["sha256"]
    if digest != trace["receipts"]["dir_digest"]["sha256"]:
        raise Stage2Refused("receipts dir digest differs from the one in the search trace")
    if trace["loop"]["stop_reason"] not in search.ALLOWED_STOP_REASONS:
        raise Stage2Refused("search trace stop reason is not registered")
    if trace["selection"]["selected_trial"] is None:
        raise Stage2Refused("no eligible trial: v2 does not exist (AM-2)")
    return {"receipts_dir": trace["receipts_dir_relpath"], "verify_errors": errors,
            "dir_sha256": digest}


def candidate_bytes(trace: Mapping, trial_id: str, root: Path = registry.ROOT) -> bytes:
    path = root / trace["candidates_dir_relpath"] / f"{trial_id}.model.json"
    data = path.read_bytes()
    if sha256_bytes(data) != trial_record(trace, trial_id)["model_sha256"]:
        raise Stage2Refused(f"{path.name} does not hash to the receipted model_sha256")
    return data


def selection_block(trace: Mapping, trace_sha256: str, *, procedure: str, trials: int,
                    stop_reason: str | None) -> dict:
    conformal = trace.get("conformal") or {}
    return {
        "procedure": procedure,
        "selection_split": "validation",
        "trials": trials,
        "selected_trial": trace["selection"]["selected_trial"],
        "stop_reason": stop_reason,
        "search_trace_sha256": trace_sha256,
        "receipts_dir_sha256": trace["receipts"]["dir_digest"]["sha256"],
        "amendments_sha256": trace["protocol"]["amendments_sha256"],
        "conformal_keep": conformal.get("keep"),
    }


def _fit_summary(diag: Mapping, calibrator: str) -> dict:
    lr_diag = diag["lr"]
    out = {
        "irls": {key: lr_diag[key] for key in ("iterations", "converged", "stop_reason",
                                                "gradient_max_abs")},
        "calibrator_kind": calibrator,
        "calibrator": dict(diag["calibrator"]),
        "am2_converged": not search.eligibility(calibrator, lr_diag, diag["calibrator"]),
    }
    out["calibrator"].pop("block_counts", None)
    return out


def _receipt_of(info: Mapping) -> dict:
    return json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))


def _kernel_loads(model_path: Path, receipt_path: Path) -> None:
    kernel_bridge.kernel().OpsHealthKernel(model_path, receipt_path)


# --------------------------------------------------------------------------------------
# step 3: freeze v2
# --------------------------------------------------------------------------------------
def freeze_v2(out_dir: Path = V2_OPS_DIR, *, rebuild: bool = True) -> dict:
    trace, trace_sha = load_trace()
    checked = check_trace(trace)
    selected = trace["selection"]["selected_trial"]
    record = trial_record(trace, selected)
    model_bytes = candidate_bytes(trace, selected)
    artifact = json.loads(model_bytes.decode("utf-8"))
    if freeze.canonical_model_bytes(artifact) != model_bytes:
        raise Stage2Refused("searched candidate is not canonical model.json bytes")
    rebuilt = None
    if rebuild:
        rows, names = search.load_registered_rows()
        candidate = next(c for c in search.search_space() if c["trial_id"] == selected)
        _record, rebuilt_bytes = search.run_trial(candidate, rows, names)
        rebuilt = {"model_sha256": sha256_bytes(rebuilt_bytes),
                   "byte_identical": rebuilt_bytes == model_bytes}
        if rebuilt_bytes != model_bytes:
            raise Stage2Refused("rebuilt selected candidate differs from the searched one")
    selection = selection_block(
        trace, trace_sha,
        procedure=("PREREGISTRATION section 6 registered search: 12 trials in a bounded loop "
                   "(hard cap 12), AM-2 eligibility, validation BA with the registered tie rules"),
        trials=trace["loop"]["iterations_run"],
        stop_reason=trace["loop"]["stop_reason"],
    )
    info = freeze.freeze(artifact, out_dir, selection=selection)
    if info["model_sha256"] != record["model_sha256"]:
        raise Stage2Refused("frozen model.json differs from the selected trial's model_sha256")
    receipt = _receipt_of(info)
    origin = freeze.verify_origin(receipt)
    _kernel_loads(Path(info["model_path"]), Path(info["receipt_path"]))
    example_path = Path(out_dir) / EXAMPLE_INPUT_NAME
    _write_exclusive(example_path, _json_bytes(EXAMPLE_INPUT))
    return {
        "selected_trial": selected,
        "search_trace_sha256": trace_sha,
        "trace_check": checked,
        "rebuild": rebuilt,
        "freeze": {key: info[key] for key in ("model_sha256", "receipt_sha256", "commit",
                                              "tree_clean")},
        "files": {
            _rel(info["model_path"]): info["model_sha256"],
            _rel(info["receipt_path"]): info["receipt_sha256"],
            _rel(example_path): sha256_bytes(example_path.read_bytes()),
        },
        "selection": selection,
        "verify_origin": origin,
    }


# --------------------------------------------------------------------------------------
# step 4: ablation contenders
# --------------------------------------------------------------------------------------
def v1_choose_threshold(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, dict]:
    """v1's choose_threshold, transcribed: t = step/100 for step 5..95; rank =
    (round(BA, 12), round(F1, 12), -|t - 0.5|); a strictly greater rank replaces the best, so
    equal ranks keep the lower t.  Rates use v1's safe division (0.0 on a zero denominator)."""
    y = [int(v) for v in labels]
    best = None
    chosen = None
    for step in V1_THRESHOLD_STEPS:
        threshold = step / 100.0
        tp = tn = fp = fn = 0
        for label, score in zip(y, scores, strict=True):
            predicted = int(score >= threshold)
            if label == 1 and predicted == 1:
                tp += 1
            elif label == 0 and predicted == 0:
                tn += 1
            elif label == 0 and predicted == 1:
                fp += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        rank = (round((recall + specificity) / 2.0, 12), round(f1, 12), -abs(threshold - 0.5))
        if best is None or rank > best:
            best = rank
            chosen = threshold
    return chosen, {"balanced_accuracy": best[0], "f1": best[1]}


def v1_trainer_scores(intercept: float, weights: Sequence[float],
                      vectors: Sequence[Sequence[float]]) -> list[float]:
    """v1 trainer's predict_probabilities order: sigmoid(intercept + sum(w * x))."""
    return [v1_gd._sigmoid(intercept + sum(map(mul, weights, vec))) for vec in vectors]


def v1_kernel_order_scores(intercept: float, weights: Sequence[float],
                           vectors: Sequence[Sequence[float]]) -> list[float]:
    """v1/v2 kernel operation order: linear = intercept, then += w_j * x_j in feature order."""
    out = []
    for vec in vectors:
        linear = intercept
        for w, x in zip(weights, vec, strict=True):
            linear += w * x
        out.append(v1_gd._sigmoid(linear))
    return out


def _registered_rows(purposes: Mapping[str, str], guard=None,
                     roles: Sequence[str] = pipeline.SPLIT_ROLES) -> tuple[dict, dict]:
    """Open registered open splits by role through the sealed guard (never a sealed split)."""
    if guard is None:
        from . import sealed_guard as guard  # noqa: PLC0415
    rows = {}
    names = {}
    for role in roles:
        name, _registered = search.REGISTERED_SPLITS[role]
        rows[role] = guard.open_split(name, purposes[role])
        names[role] = name
    return rows, names


def _write_index(path: Path, index: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(index))


def _frozen_entry(info: Mapping) -> dict:
    return {
        "files": {_rel(info["model_path"]): info["model_sha256"],
                  _rel(info["receipt_path"]): info["receipt_sha256"]},
        "model_sha256": info["model_sha256"],
        "receipt_sha256": info["receipt_sha256"],
        "source_commit": info["commit"],
    }


def _existing_frozen(out_dir: Path) -> dict | None:
    model = out_dir / freeze.MODEL_NAME
    receipt = out_dir / freeze.RECEIPT_NAME
    if not model.exists() and not receipt.exists():
        return None
    rec = json.loads(receipt.read_text(encoding="utf-8"))
    if sha256_bytes(model.read_bytes()) != rec["model_sha256"]:
        raise Stage2Refused(f"{out_dir}: existing model.json does not match its receipt")
    return {"model_path": str(model), "receipt_path": str(receipt),
            "model_sha256": rec["model_sha256"],
            "receipt_sha256": sha256_bytes(receipt.read_bytes()),
            "commit": rec["source"]["commit"], "tree_clean": rec["source"]["tree_clean"]}


def freeze_contenders(out_root: Path = CONTENDERS_DIR, *, guard=None) -> dict:
    trace, trace_sha = load_trace()
    check_trace(trace)
    selected = trace["selection"]["selected_trial"]
    sel = trial_record(trace, selected)
    l2, cal = sel["l2"], sel["calibrator"]
    purposes = {role: "ABLATION" for role in pipeline.SPLIT_ROLES}
    rows, names = _registered_rows(purposes, guard)
    val_rows = rows["threshold"]
    val_labels = labels_of(val_rows)
    index_path = Path(out_root) / INDEX_NAME
    index = {
        "schema": "szl-oac/ops-health-contenders-index/v2",
        "synthetic": True,
        "search_trace_sha256": trace_sha,
        "selected_trial": selected,
        "selected_recipe": {"l2": l2, "calibrator": cal},
        "splits": {role: {"name": names[role], "rows": len(rows[role]),
                          "sha256": pipeline.rows_sha256(rows[role]), "purpose": "ABLATION"}
                   for role in pipeline.SPLIT_ROLES},
        "truth_labels": "validation metrics MEASURED on the registered validation split; "
                        "SYNTHETIC; no sealed split was read",
        "contenders": {},
    }

    def block(procedure: str) -> dict:
        return selection_block(trace, trace_sha, procedure=procedure, trials=1, stop_reason=None)

    # ABL-data -------------------------------------------------------------------------
    out = Path(out_root) / "ABL-data"
    train_768 = rows["train"][:ABL_DATA_ROWS]
    art, diag = pipeline.build_candidate(
        train_rows=train_768, calibration_rows=rows["calibration"],
        conformal_rows=rows["conformal"], threshold_rows=val_rows, l2=l2, calibrator=cal,
        split_names={**names, "train": f"train[0:{ABL_DATA_ROWS}]"},
    )
    info = _existing_frozen(out) or freeze.freeze(art, out, selection=block(
        f"ABL-data (PREREGISTRATION section 9): the selected recipe {selected} refit on train "
        f"rows 0..{ABL_DATA_ROWS - 1}; calibrator, threshold and conformal on the registered "
        "calibration, validation and conformal splits"))
    if info["model_sha256"] != sha256_bytes(freeze.canonical_model_bytes(art)):
        raise Stage2Refused("ABL-data: frozen model differs from the rebuilt candidate")
    freeze.verify_origin(_receipt_of(info))
    index["contenders"]["ABL-data"] = {
        "kind": "v2_artifact",
        "status": "FROZEN",
        "description": f"selected recipe (l2={l2!r}, calibrator={cal}) on the first "
                       f"{ABL_DATA_ROWS} rows of the registered train split",
        "train_rows": ABL_DATA_ROWS,
        "train_sha256": pipeline.rows_sha256(train_768),
        **_frozen_entry(info),
        "fit": _fit_summary(diag, cal),
        "validation": search.validation_report(art, val_rows),
    }
    _write_index(index_path, index)

    # ABL-optimizer --------------------------------------------------------------------
    out = Path(out_root) / "ABL-optimizer"
    params_path = out / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
    else:
        x_train = pipeline.normalized_vectors(rows["train"])
        intercept, weights = v1_gd.train_v1_gd(x_train, labels_of(rows["train"]))
        x_val = pipeline.normalized_vectors(val_rows)
        chosen, chosen_rank = v1_choose_threshold(
            val_labels, v1_trainer_scores(intercept, weights, x_val))
        r_intercept = round(intercept, 12)
        r_weights = [round(w, 12) for w in weights]
        frozen_scores = v1_kernel_order_scores(r_intercept, r_weights, x_val)
        rechosen, _ = v1_choose_threshold(val_labels, frozen_scores)
        decisions = [1 if s >= chosen else 0 for s in frozen_scores]
        state = freeze.source_state()
        params = {
            "schema": "szl-oac/ops-health-contender-params/v2",
            "name": "ABL-optimizer",
            "synthetic": True,
            "description": "v1's exact training algorithm (PREREGISTRATION section 9) retrained "
                           "on the full registered v2 train split",
            "algorithm": "batch_gradient_descent_logistic_regression (v2.research.v1_gd."
                         "train_v1_gd, v1 operation order)",
            "epochs": v1_gd.EPOCHS,
            "learning_rate": v1_gd.LEARNING_RATE,
            "l2_penalty": v1_gd.L2_PENALTY,
            "initialization": "zeros",
            "feature_order": list(generator.FEATURE_NAMES),
            "intercept": r_intercept,
            "weights": dict(zip(generator.FEATURE_NAMES, r_weights, strict=True)),
            "parameters_rounding": "12 decimals, as v1 publishes its model.json",
            "decision_threshold": chosen,
            "threshold_rule": "v1 choose_threshold: t = k/100, k = 5..95; maximize "
                              "(round(BA,12), round(F1,12), -|t-0.5|); equal ranks keep the "
                              "lower t; scores from the unrounded parameters in v1 trainer order",
            "threshold_rule_rank": chosen_rank,
            "threshold_same_with_rounded_parameters": rechosen == chosen,
            "calibrator": "none",
            "conformal": "none",
            "scoring": "score = sigmoid(intercept + w . x) with v1 normalization, accumulated "
                       "in kernel order (intercept, then each w_j * x_j in feature_order); "
                       "ALERT iff score >= decision_threshold",
            "splits": {
                "train": index["splits"]["train"],
                "threshold": index["splits"]["threshold"],
            },
            "validation": search.decision_report(val_labels, frozen_scores, decisions, chosen),
            "source": {
                "commit": state["commit"],
                "tree_clean": state["tree_clean"],
                "v1_gd_sha256": sha256_bytes(Path(v1_gd.__file__).read_bytes()),
                "python": platform.python_version(),
            },
            "search_trace_sha256": trace_sha,
            "truth_label": "validation metrics MEASURED; SYNTHETIC",
        }
        if not params["source"]["tree_clean"]:
            raise Stage2Refused("ABL-optimizer: working tree not clean; refusing to freeze")
        _write_exclusive(params_path, _json_bytes(params))
    params_sha = sha256_bytes(params_path.read_bytes())
    index["contenders"]["ABL-optimizer"] = {
        "kind": "params_file",
        "status": "FROZEN",
        "description": "v1 GD (2,400 epochs, lr 0.42, l2 0.01) on the full v2 train split; "
                       "threshold by v1's own rule on validation",
        "files": {_rel(params_path): params_sha},
        "params_sha256": params_sha,
        "decision_threshold": params["decision_threshold"],
        "threshold_same_with_rounded_parameters":
            params["threshold_same_with_rounded_parameters"],
        "validation": params["validation"],
    }
    _write_index(index_path, index)

    # ABL-calibration ------------------------------------------------------------------
    if cal == "none":
        index["contenders"]["ABL-calibration"] = {
            "kind": "not_applicable",
            "status": "NOT_APPLICABLE",
            "reason": "the selected calibrator is none, so the selected model has no calibrator "
                      "to remove; the search itself compared none, platt and isotonic "
                      "(section 9)",
        }
    else:
        out = Path(out_root) / "ABL-calibration"
        art, diag = pipeline.build_candidate(
            train_rows=rows["train"], calibration_rows=rows["calibration"],
            conformal_rows=rows["conformal"], threshold_rows=val_rows, l2=l2,
            calibrator="none", split_names=names,
        )
        same_trial = search.trial_id_for(l2, "none")
        rebuilt_sha = sha256_bytes(freeze.canonical_model_bytes(art))
        info = _existing_frozen(out) or freeze.freeze(art, out, selection=block(
            f"ABL-calibration (PREREGISTRATION section 9): the selected model {selected} without "
            "its calibrator; threshold re-selected on validation, conformal refit on the "
            "uncalibrated scores"))
        if info["model_sha256"] != rebuilt_sha:
            raise Stage2Refused("ABL-calibration: frozen model differs from the rebuilt one")
        freeze.verify_origin(_receipt_of(info))
        index["contenders"]["ABL-calibration"] = {
            "kind": "v2_artifact",
            "status": "FROZEN",
            "description": f"selected model (l2={l2!r}) with calibrator none",
            "identical_to_search_trial": same_trial,
            "identical_to_search_trial_model": rebuilt_sha == trial_record(
                trace, same_trial)["model_sha256"],
            **_frozen_entry(info),
            "fit": _fit_summary(diag, "none"),
            "validation": search.validation_report(art, val_rows),
        }
    _write_index(index_path, index)

    # ABL-conformal (decision only) -----------------------------------------------------
    conf = trace["conformal"]
    index["contenders"]["ABL-conformal"] = {
        "kind": "decision",
        "status": "DECIDED_ON_VALIDATION",
        "keep": conf["keep"],
        "rule": conf["rule"],
        "coverage": conf["coverage"],
        "with_abstention": conf["with_abstention"],
        "without_abstention": conf["without_abstention"],
        "source": "search_trace.json conformal block",
    }
    _write_index(index_path, index)
    return index


# --------------------------------------------------------------------------------------
# step 5: known-good retrains (AM-4)
# --------------------------------------------------------------------------------------
def known_good_seed_material(k: int) -> str:
    return f"{generator.SEED_NAMESPACE}:{generator.MASTER_SEED + k}:train"


def known_good_rows(k: int, n_rows: int = KNOWN_GOOD_TRAIN_ROWS) -> list[dict]:
    return generator.generate_rows("train", n_rows, "nominal",
                                   master_seed=generator.MASTER_SEED + k)


def known_good(ks: Sequence[int] = KNOWN_GOOD_KS, out_root: Path = KNOWN_GOOD_DIR, *,
               guard=None, progress=None) -> dict:
    trace, trace_sha = load_trace()
    check_trace(trace)
    selected = trace["selection"]["selected_trial"]
    sel = trial_record(trace, selected)
    l2, cal = sel["l2"], sel["calibrator"]
    purposes = {"calibration": "CALIBRATION", "conformal": "CONFORMAL", "threshold": "GATES"}
    index_path = Path(out_root) / INDEX_NAME
    index = (json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {
        "schema": "szl-oac/ops-health-known-good-index/v2",
        "synthetic": True,
        "amendment": "AM-4",
        "search_trace_sha256": trace_sha,
        "selected_trial": selected,
        "selected_recipe": {"l2": l2, "calibrator": cal},
        "note": "hashes recorded before any section 10/11 gate is evaluated; the train splits "
                "are nominal, 16,384 rows, not sealed",
        "retrains": {},
    })
    if index["search_trace_sha256"] != trace_sha:
        raise Stage2Refused("known_good INDEX.json belongs to another search trace")
    shared = None
    for k in ks:
        key = f"k{k}"
        out = Path(out_root) / key
        material = known_good_seed_material(k)
        rows = known_good_rows(k)
        data = generator.jsonl_bytes(rows)
        train_path = out / "train.jsonl"
        if train_path.exists():
            if train_path.read_bytes() != data:
                raise Stage2Refused(f"{train_path} differs from the regenerated split")
        else:
            _write_exclusive(train_path, data)
        if shared is None:
            shared, _names = _registered_rows(purposes, guard, roles=RETRAIN_SHARED_ROLES)
        art, diag = pipeline.build_candidate(
            train_rows=rows, calibration_rows=shared["calibration"],
            conformal_rows=shared["conformal"], threshold_rows=shared["threshold"],
            l2=l2, calibrator=cal,
            split_names={"train": f"known_good_{key}_train", "calibration": "calibration",
                         "conformal": "conformal", "threshold": "validation"},
        )
        info = _existing_frozen(out) or freeze.freeze(art, out, selection=selection_block(
            trace, trace_sha,
            procedure=(f"AM-4 known-good retrain {key}: the selected recipe {selected} end "
                       f"to end on a fresh nominal train split (seed material {material}); "
                       "calibrator, threshold and conformal on the registered splits"),
            trials=1, stop_reason=None))
        if info["model_sha256"] != sha256_bytes(freeze.canonical_model_bytes(art)):
            raise Stage2Refused(f"{key}: frozen model differs from the rebuilt candidate")
        fit = _fit_summary(diag, cal)
        threshold = art["decision_threshold"]
        freeze.verify_origin(_receipt_of(info))
        _kernel_loads(Path(info["model_path"]), Path(info["receipt_path"]))
        index["retrains"][key] = {
            "k": k,
            "seed_material": material,
            "seed": generator.derive_seed("train", generator.MASTER_SEED + k),
            "train": {"path": _rel(train_path), "rows": len(rows), "bytes": len(data),
                      "sha256": sha256_bytes(data)},
            **_frozen_entry(info),
            "decision_threshold": threshold,
            "fit": fit,
        }
        index["retrains"] = dict(sorted(index["retrains"].items()))
        _write_index(index_path, index)
        if progress is not None:
            progress(f"{key}: train sha256 {sha256_bytes(data)} model {info['model_sha256']} "
                     f"threshold {threshold} am2_converged {fit and fit['am2_converged']}")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-2 freezing (SYNTHETIC).")
    parser.add_argument("step", choices=("freeze-v2", "contenders", "known-good"))
    args = parser.parse_args(argv)
    try:
        if args.step == "freeze-v2":
            result = freeze_v2()
        elif args.step == "contenders":
            result = freeze_contenders()
        else:
            result = known_good(progress=lambda line: print(line, flush=True))
    except (Stage2Refused, freeze.FreezeRefused, freeze.OriginMismatch) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
