#!/usr/bin/env python3
"""SZL Forge: offline, fail-closed physical-world evaluation admission.

No model loading, downloads, simulation, inference, execution or publication.
Evidence rows are externally supplied measurements, never generated scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
STATES = ("HOLD", "STRUCTURALLY_ADMISSIBLE_UNVERIFIED")
MAX_BYTES = 256_000


def _exact_keys(value, required):
    return type(value) is dict and set(value) == set(required)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def evaluate(record):
    """Evaluate a frozen *paired* scenario set, preserving independent gates.

    Do not compare across different dataset/split hashes or unpaired seeds.
    Lower physical error is better. Higher success is better. Improvement
    without positive sample coverage and failure-mode evidence is a HOLD.
    """
    reasons = []
    fields = (
        "schema",
        "model",
        "source",
        "dataset",
        "rights",
        "baseline",
        "candidate",
        "scenarios",
        "evaluation",
    )
    if (
        not _exact_keys(record, fields)
        or record.get("schema") != "szl.world-eval-input/v1"
    ):
        return {"state": "HOLD", "reasons": ["INVALID_SCHEMA"], "paired": 0}
    for field in (
        "model",
        "source",
        "dataset",
        "baseline",
        "candidate",
        "rights",
        "evaluation",
    ):
        if type(record[field]) is not dict:
            return {
                "state": "HOLD",
                "reasons": ["INVALID_" + field.upper()],
                "paired": 0,
            }
    for field in ("model", "source", "dataset"):
        obj = record[field]
        if not _exact_keys(obj, ("id", "revision", "sha256")):
            return {
                "state": "HOLD",
                "reasons": ["INVALID_" + field.upper()],
                "paired": 0,
            }
        if not all(type(obj[k]) is str and obj[k] for k in obj):
            reasons.append("INVALID_" + field.upper())
        if not SHA40.fullmatch(obj["revision"]) or not SHA64.fullmatch(obj["sha256"]):
            reasons.append("UNPINNED_" + field.upper())
    if not _exact_keys(record["rights"], ("approved", "scope", "evidence_sha256")):
        reasons.append("RIGHTS_UNVERIFIED")
    else:
        r = record["rights"]
        if (
            r["approved"] is not True
            or r["scope"] != "EVALUATION_ONLY"
            or not isinstance(r["evidence_sha256"], str)
            or not SHA64.fullmatch(r["evidence_sha256"])
        ):
            reasons.append("RIGHTS_UNVERIFIED")
    if not _exact_keys(
        record["evaluation"],
        (
            "protocol_sha256",
            "heldout",
            "independent",
            "synthetic_labeled",
            "hardware_observed",
        ),
    ):
        reasons.append("EVALUATION_CONTRACT_UNVERIFIED")
    else:
        e = record["evaluation"]
        if (
            not isinstance(e["protocol_sha256"], str)
            or not SHA64.fullmatch(e["protocol_sha256"])
            or e["heldout"] is not True
            or e["independent"] is not True
            or e["synthetic_labeled"] is not True
            or e["hardware_observed"] is not True
        ):
            reasons.append("EVALUATION_CONTRACT_UNVERIFIED")
    for name in ("baseline", "candidate"):
        obj = record[name]
        if not _exact_keys(obj, ("model_revision", "adapter_sha256")):
            return {
                "state": "HOLD",
                "reasons": ["INVALID_" + name.upper()],
                "paired": 0,
            }
        if (
            not isinstance(obj["model_revision"], str)
            or not SHA40.fullmatch(obj["model_revision"])
            or not isinstance(obj["adapter_sha256"], str)
            or not SHA64.fullmatch(obj["adapter_sha256"])
        ):
            reasons.append("UNPINNED_" + name.upper())
    rows = record["scenarios"]
    if type(rows) is not list or not rows or len(rows) > 5000:
        return {
            "state": "HOLD",
            "reasons": sorted(set(reasons + ["NO_BOUNDED_PAIRED_SCENARIOS"])),
            "paired": 0,
        }
    seen = set()
    error_base = error_candidate = success_base = success_candidate = 0.0
    total = 0
    for row in rows:
        keys = (
            "id",
            "seed",
            "split",
            "baseline_physics_error",
            "candidate_physics_error",
            "baseline_success",
            "candidate_success",
            "safety_violation",
        )
        if not _exact_keys(row, keys):
            reasons.append("INVALID_SCENARIO_ROW")
            continue
        identity = (row["id"], row["seed"])
        if (
            type(row["id"]) is not str
            or not row["id"]
            or type(row["seed"]) is not int
            or row["split"] != "heldout"
            or identity in seen
        ):
            reasons.append("INVALID_OR_DUPLICATE_PAIR")
            continue
        seen.add(identity)
        if not all(
            _number(row[k]) and 0 <= row[k] <= 1
            for k in ("baseline_physics_error", "candidate_physics_error")
        ) or any(
            type(row[k]) is not bool
            for k in ("baseline_success", "candidate_success", "safety_violation")
        ):
            reasons.append("INVALID_MEASUREMENTS")
            continue
        total += 1
        error_base += row["baseline_physics_error"]
        error_candidate += row["candidate_physics_error"]
        success_base += int(row["baseline_success"])
        success_candidate += int(row["candidate_success"])
        if row["safety_violation"]:
            reasons.append("SAFETY_VIOLATION")
    if total != len(rows):
        reasons.append("INCOMPLETE_PAIR_COVERAGE")
    if total < 30:
        reasons.append("INSUFFICIENT_SAMPLE_SIZE")
    if total:
        if error_candidate / total > error_base / total:
            reasons.append("PHYSICAL_CONSISTENCY_REGRESSION")
        if success_candidate / total < success_base / total:
            reasons.append("TASK_SUCCESS_REGRESSION")
    if reasons:
        return {"state": "HOLD", "reasons": sorted(set(reasons)), "paired": total}
    return {
        "state": "STRUCTURALLY_ADMISSIBLE_UNVERIFIED",
        "reasons": [],
        "paired": total,
        "physics_error_baseline_mean": error_base / total,
        "physics_error_candidate_mean": error_candidate / total,
        "success_baseline": success_base / total,
        "success_candidate": success_candidate / total,
        "production_qualified": False,
        "training_performed": False,
        "robot_deployment_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.input.is_symlink() or args.output.is_symlink() or args.output.exists():
            raise ValueError("PATH_NOT_EXCLUSIVE")
        raw = args.input.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("INPUT_TOO_LARGE")
        record = json.loads(raw)
        result = evaluate(record)
        result["schema"] = "szl.world-eval-admission/v1"
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        result["source"] = "OFFLINE_SUPPLIED_MEASUREMENTS_NOT_INDEPENDENTLY_VERIFIED"
        body = (
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        with args.output.open("xb") as f:
            f.write(body)
        print(
            json.dumps(
                {
                    "state": result["state"],
                    "paired": result["paired"],
                    "reasons": result["reasons"],
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
        )
        return 0 if result["state"] == "STRUCTURALLY_ADMISSIBLE_UNVERIFIED" else 2
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"state": "HOLD", "reason": type(exc).__name__}), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
