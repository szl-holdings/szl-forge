#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline launch contract shared by preflight and the SFT trainer.

One config is the launch document. This module is the only place that decides
whether that document may proceed. It reads local files and returns a verdict.
It does not import a model stack, call a teacher, start training, or write to
the Hub.

The generic JSONL linter ``tools/validate_sft_dataset.py`` stays a structural
check with its own caller-chosen minimum. A training launch always calls it
with ``MIN_ACCEPTED_EXAMPLES`` and a required held-out file. A count of 32
does not waive rights, an immutable pin, a denied corpus, or holdout overlap.

PR 703 (``ab0f1dd09adaf2196aedb550f8c2822cb42d39f6``) is a separate proposal.
Its ``training/distill/distill_wave.py`` does not accept ``--config``, loads
teachers, and uploads to the Hub before rejecting zero accepts. This module
does not vendor or start that runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "szl.forge.launch-contract/v1"
CONFIG_SCHEMA = "szl.forge.wave-config/v1"
ADMISSION_SCHEMA = "szl.forge.training-admission/v1"
MIN_ACCEPTED_EXAMPLES = 32
IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
TRAINABLE_METHODS = ("qlora", "lora16")

# These corpora stay refused when a receipt says otherwise. The labels are the
# roles already recorded for them. They are not licenses. Keys are casefolded
# because Hub repository identity is not case-sensitive.
_DENIED_DATASET_ROLES = {
    "SZLHOLDINGS/szl-quant-sft-v1": "HELD_COUNSEL",
    "SZLHOLDINGS/killinchu-osint-corpus": "NOT_ADMITTED",
    "SZLHOLDINGS/SZLHOLDINGS": "METADATA_ONLY",
    "SZLHOLDINGS/why-we-lead": "HISTORICAL_METADATA_ONLY",
}
DENIED_DATASETS = {repo_id.casefold(): role for repo_id, role in _DENIED_DATASET_ROLES.items()}

_VALIDATOR = None


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    chosen: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            chosen.append(item)
    return chosen


def _validator():
    global _VALIDATOR
    if _VALIDATOR is None:
        path = Path(__file__).resolve().parents[1] / "tools" / "validate_sft_dataset.py"
        spec = importlib.util.spec_from_file_location("szl_validate_sft_dataset", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _VALIDATOR = module
    return _VALIDATOR


def _empty_report() -> Dict[str, Any]:
    return {
        "schema": "szl.forge-sft-dataset-report/v2",
        "status": "INVALID",
        "records": 0,
        "errors": [],
        "cross_dataset_overlaps": 0,
    }


def _report(
    *,
    codes: List[str],
    errors: List[str],
    accepted: int,
    config: Optional[dict],
    kind: Optional[str],
    dataset_report: Dict[str, Any],
    student_id: Optional[str],
    student_revision: Optional[str],
    dataset_id: Optional[str],
    dataset_revision: Optional[str],
    teacher_id: Optional[str],
    teacher_revision: Optional[str],
    train_jsonl: Optional[str],
    holdout_jsonl: Optional[str],
    corpus_sha256: Optional[str],
) -> Dict[str, Any]:
    unique = _dedupe(codes)
    return {
        "schema": SCHEMA,
        "ok": not unique,
        "verdict": "GATES_SATISFIED_NOT_LAUNCHED" if not unique else "REJECT",
        "codes": unique,
        "errors": errors[:25],
        "accepted_examples": accepted,
        "min_accepted_examples": MIN_ACCEPTED_EXAMPLES,
        "kind": kind,
        "config": config,
        "dataset_report": dataset_report,
        "student_id": student_id,
        "student_revision": student_revision,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "teacher_id": teacher_id,
        "teacher_revision": teacher_revision,
        "train_jsonl": train_jsonl,
        "holdout_jsonl": holdout_jsonl,
        "corpus_sha256": corpus_sha256,
        "launch_executed": False,
        "training_started": False,
        "teacher_invoked": False,
        "model_loaded": False,
        "hub_write": False,
    }


def _as_path(config_dir: Path, value: object) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = config_dir / path
    return path


def _pin(value: object, role: str) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
    codes: List[str] = []
    errors: List[str] = []
    if not isinstance(value, dict):
        return None, None, ["LAUNCH_PIN_NOT_IMMUTABLE"], [f"{role} pin is missing"]
    ident = value.get("id")
    revision = value.get("revision")
    if not isinstance(ident, str) or REPO_ID.fullmatch(ident) is None:
        codes.append("LAUNCH_PIN_NOT_IMMUTABLE")
        errors.append(f"{role} id is not an org/name repository id")
        ident = ident if isinstance(ident, str) else None
    if not isinstance(revision, str) or IMMUTABLE_REVISION.fullmatch(revision) is None:
        codes.append("LAUNCH_PIN_NOT_IMMUTABLE")
        errors.append(
            f"{role} revision is not an immutable 40-character lowercase commit SHA"
        )
        revision = None
    return ident, revision, codes, errors


def _student_pin(config: dict) -> object:
    if "student" in config:
        return config.get("student")
    base_model = config.get("base_model")
    if isinstance(base_model, dict):
        return base_model
    if isinstance(base_model, str):
        return {"id": base_model, "revision": config.get("base_model_revision")}
    return None


def _deny_ids(idents: Sequence[Optional[str]]) -> Tuple[List[str], List[str]]:
    codes: List[str] = []
    errors: List[str] = []
    for ident in idents:
        if not isinstance(ident, str) or not ident:
            continue
        reason = DENIED_DATASETS.get(ident.casefold())
        if reason is not None:
            codes.append("LAUNCH_DATASET_DENIED")
            errors.append(f"{ident} is {reason} and is not a training corpus")
    return codes, errors


def _dataset_ids(config: dict) -> Tuple[List[str], List[str], List[str]]:
    codes: List[str] = []
    errors: List[str] = []
    idents: List[str] = []
    raw = config.get("datasets")
    if "datasets" not in config:
        return idents, codes, errors
    if not isinstance(raw, list):
        return idents, ["LAUNCH_CONFIG_INVALID"], ["datasets must be a list when present"]
    for item in raw:
        if isinstance(item, str):
            idents.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            idents.append(item["id"])
        else:
            codes.append("LAUNCH_CONFIG_INVALID")
            errors.append("datasets entries must be repository ids")
    return idents, codes, errors


def _check_admission(
    path: Optional[Path],
    *,
    dataset_id: Optional[str],
    revision: Optional[str],
    corpus_sha256: Optional[str],
) -> Tuple[List[str], List[str]]:
    if path is None or not path.is_file():
        return ["LAUNCH_ADMISSION_MISSING"], ["admission receipt is missing"]
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ["LAUNCH_ADMISSION_INVALID"], [f"admission receipt is unreadable ({exc})"]
    if not isinstance(receipt, dict):
        return ["LAUNCH_ADMISSION_INVALID"], ["admission receipt is not a JSON object"]

    codes: List[str] = []
    errors: List[str] = []
    if receipt.get("schema") != ADMISSION_SCHEMA:
        codes.append("LAUNCH_ADMISSION_INVALID")
        errors.append(f"admission schema must be {ADMISSION_SCHEMA}")
    license_name = receipt.get("license")
    rights_ok = (
        receipt.get("rights") == "ADMITTED_FOR_TRAINING"
        and receipt.get("commercial_use") == "ADMITTED"
        and receipt.get("basis") == "OPERATOR_ADMISSION_RECEIPT"
        and isinstance(license_name, str)
        and bool(license_name.strip())
    )
    if not rights_ok:
        codes.append("LAUNCH_RIGHTS_NOT_ADMITTED")
        errors.append(
            "admission requires rights ADMITTED_FOR_TRAINING, commercial_use ADMITTED, "
            "basis OPERATOR_ADMISSION_RECEIPT, and a non-empty operator-supplied license; "
            "organization ownership and a public Hub card are not admission"
        )
    if (
        receipt.get("dataset_id") != dataset_id
        or receipt.get("revision") != revision
        or receipt.get("split") != "train"
        or corpus_sha256 is None
        or receipt.get("corpus_sha256") != corpus_sha256
    ):
        codes.append("LAUNCH_ADMISSION_MISMATCH")
        errors.append(
            "admission receipt must match the dataset id, immutable revision, "
            "train split, and train-file sha256"
        )
    return codes, errors


def evaluate(
    config: object,
    *,
    config_dir: Path,
    expected_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one wave config. ``ok`` means the gates passed and nothing ran."""
    codes: List[str] = []
    errors: List[str] = []
    blank = _report(
        codes=["LAUNCH_CONFIG_INVALID"],
        errors=["config is not a JSON object"],
        accepted=0,
        config=None,
        kind=None,
        dataset_report=_empty_report(),
        student_id=None,
        student_revision=None,
        dataset_id=None,
        dataset_revision=None,
        teacher_id=None,
        teacher_revision=None,
        train_jsonl=None,
        holdout_jsonl=None,
        corpus_sha256=None,
    )
    if not isinstance(config, dict):
        return blank

    if config.get("schema") != CONFIG_SCHEMA:
        codes.append("LAUNCH_CONFIG_INVALID")
        errors.append(f"config schema must be {CONFIG_SCHEMA}")

    kind = config.get("kind")
    if kind not in ("sft", "distill"):
        codes.append("LAUNCH_CONFIG_INVALID")
        errors.append("kind must be sft or distill")
        kind = None
    elif expected_kind is not None and kind != expected_kind:
        codes.append("LAUNCH_KIND_MISMATCH")
        errors.append(f"this launcher requires kind {expected_kind}")

    if "max_samples" in config:
        codes.append("LAUNCH_SAMPLE_CAP_FORBIDDEN")
        errors.append("max_samples cannot shrink a corpus that cleared the launch floor")

    student_id, student_revision, pin_codes, pin_errors = _pin(_student_pin(config), "student")
    codes.extend(pin_codes)
    errors.extend(pin_errors)

    teacher_id = None
    teacher_revision = None
    if kind == "distill" or "teacher" in config:
        teacher_id, teacher_revision, pin_codes, pin_errors = _pin(config.get("teacher"), "teacher")
        codes.extend(pin_codes)
        errors.extend(pin_errors)

    dataset_obj = config.get("dataset") if isinstance(config.get("dataset"), dict) else None
    if "dataset" in config and dataset_obj is None:
        codes.append("LAUNCH_PIN_NOT_IMMUTABLE")
        errors.append("dataset pin is missing")
    listed, list_codes, list_errors = _dataset_ids(config)
    codes.extend(list_codes)
    errors.extend(list_errors)

    dataset_id = None
    dataset_revision = None
    if dataset_obj is not None:
        dataset_id, dataset_revision, pin_codes, pin_errors = _pin(dataset_obj, "dataset")
        codes.extend(pin_codes)
        errors.extend(pin_errors)
        if dataset_obj.get("split") != "train":
            codes.append("LAUNCH_SPLIT_NOT_TRAIN")
            errors.append("dataset split must be train")
        if listed and listed != [dataset_id]:
            codes.append("LAUNCH_CONFIG_INVALID")
            errors.append("a launch config pins exactly one dataset")
    elif listed:
        codes.append("LAUNCH_PIN_NOT_IMMUTABLE")
        errors.append("dataset list has no immutable dataset pin")
        dataset_id = listed[0]
    else:
        codes.append("LAUNCH_PIN_NOT_IMMUTABLE")
        errors.append("dataset pin is missing")

    deny_codes, deny_errors = _deny_ids([dataset_id, *listed])
    codes.extend(deny_codes)
    errors.extend(deny_errors)

    if kind == "sft" and config.get("method") not in TRAINABLE_METHODS:
        codes.append("LAUNCH_METHOD_NOT_TRAINABLE")
        errors.append("sft method must be qlora or lora16")

    train_path = _as_path(config_dir, config.get("train_jsonl"))
    holdout_path = _as_path(config_dir, config.get("holdout_jsonl"))
    admission_path = _as_path(config_dir, config.get("admission"))
    if holdout_path is None or not holdout_path.is_file():
        codes.append("LAUNCH_HOLDOUT_MISSING")
        errors.append("holdout jsonl is required and must be disjoint from train")

    dataset_report = _empty_report()
    accepted = 0
    corpus_sha256 = None
    if train_path is None or not train_path.is_file():
        codes.append("LAUNCH_CORPUS_FLOOR")
        errors.append(f"0 accepted examples; launch requires {MIN_ACCEPTED_EXAMPLES}")
    else:
        try:
            corpus_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
        except OSError as exc:
            codes.append("LAUNCH_CORPUS_INVALID")
            errors.append(f"train jsonl is unreadable ({exc})")
        disjoint = [str(holdout_path)] if holdout_path is not None and holdout_path.is_file() else None
        dataset_report = _validator().validate_dataset(
            str(train_path),
            min_examples=MIN_ACCEPTED_EXAMPLES,
            disjoint_from=disjoint,
        )
        records = dataset_report.get("records")
        accepted = records if isinstance(records, int) else 0
        if accepted < MIN_ACCEPTED_EXAMPLES:
            codes.append("LAUNCH_CORPUS_FLOOR")
            errors.append(
                f"{accepted} accepted examples; launch requires {MIN_ACCEPTED_EXAMPLES}"
            )
        if dataset_report.get("cross_dataset_overlaps"):
            codes.append("LAUNCH_HOLDOUT_OVERLAP")
            errors.append("train inputs overlap the holdout file")
        for error in dataset_report.get("errors") or []:
            if "min-examples" in error or "overlaps reference" in error:
                continue
            if error.startswith("reference ") and (
                "unreadable" in error or "not found" in error
            ):
                codes.append("LAUNCH_HOLDOUT_MISSING")
                continue
            codes.append("LAUNCH_CORPUS_INVALID")
            errors.append(error)

    admission_codes, admission_errors = _check_admission(
        admission_path,
        dataset_id=dataset_id,
        revision=dataset_revision,
        corpus_sha256=corpus_sha256,
    )
    codes.extend(admission_codes)
    errors.extend(admission_errors)

    return _report(
        codes=codes,
        errors=errors,
        accepted=accepted,
        config=config,
        kind=kind,
        dataset_report=dataset_report,
        student_id=student_id,
        student_revision=student_revision,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        teacher_id=teacher_id,
        teacher_revision=teacher_revision,
        train_jsonl=str(train_path.resolve()) if train_path is not None and train_path.is_file() else None,
        holdout_jsonl=str(holdout_path.resolve()) if holdout_path is not None and holdout_path.is_file() else None,
        corpus_sha256=corpus_sha256,
    )


def evaluate_path(path: str | Path, *, expected_kind: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _report(
            codes=["LAUNCH_CONFIG_UNREADABLE"],
            errors=[f"config is unreadable ({exc})"],
            accepted=0,
            config=None,
            kind=None,
            dataset_report=_empty_report(),
            student_id=None,
            student_revision=None,
            dataset_id=None,
            dataset_revision=None,
            teacher_id=None,
            teacher_revision=None,
            train_jsonl=None,
            holdout_jsonl=None,
            corpus_sha256=None,
        )
    return evaluate(config, config_dir=config_path.parent, expected_kind=expected_kind)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--kind", choices=("sft", "distill"), default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    decision = evaluate_path(args.config, expected_kind=args.kind)
    text = json.dumps(decision, indent=2, sort_keys=True)
    print(text)
    if decision["ok"]:
        print(
            "launch contract: gates satisfied; no model was loaded and no training was started",
            file=sys.stderr,
        )
        return 0
    print("launch contract: REJECT " + ",".join(decision["codes"]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
