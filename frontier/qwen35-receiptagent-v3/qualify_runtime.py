#!/usr/bin/env python3
"""Fail-closed ReceiptAgent v3 runtime and GPU preflight."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train_candidate import (
    QualificationError,
    ROOT,
    fresh_exact_source,
    gpu_gate,
    enforce_runtime_lock,
    load_committed_json,
    raw_gpu_preflight,
    runtime_versions,
    sanitized_error,
    sha256_json,
)


def qualify(source_commit: str) -> dict[str, Any]:
    source = fresh_exact_source(source_commit)
    candidate = load_committed_json(source_commit, "candidate.json")
    raw_gpu = raw_gpu_preflight(candidate["training_recipe"])

    import unsloth  # noqa: F401 - patch before runtime interface imports
    import torch
    from peft import PeftModel
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.chat_templates import train_on_responses_only
    from unsloth.trainer import UnslothVisionDataCollator

    versions = enforce_runtime_lock(candidate)
    required_interfaces = {
        "FastVisionModel.from_pretrained": callable(
            getattr(FastVisionModel, "from_pretrained", None)
        ),
        "FastVisionModel.get_peft_model": callable(
            getattr(FastVisionModel, "get_peft_model", None)
        ),
        "PeftModel.from_pretrained": callable(getattr(PeftModel, "from_pretrained", None)),
        "SFTConfig": callable(SFTConfig),
        "SFTTrainer": callable(SFTTrainer),
        "UnslothVisionDataCollator": callable(UnslothVisionDataCollator),
        "train_on_responses_only": callable(train_on_responses_only),
    }
    if not all(required_interfaces.values()):
        raise QualificationError(f"required training interfaces are absent: {required_interfaces}")
    gpu = gpu_gate(torch, candidate["training_recipe"])
    gpu["preRuntimeImport"] = raw_gpu
    report = {
        "schema": "szl.frontier-runtime-preflight/v1",
        "candidateId": candidate["candidate_id"],
        "state": "MEASURED_RUNTIME_PREFLIGHT_PASSED",
        "measuredAt": datetime.now(timezone.utc).isoformat(),
        "hostClass": "LOCAL_GPU_RUNNER_REDACTED",
        "source": source,
        "runtimePackages": versions,
        "interfaces": required_interfaces,
        "gpu": gpu,
        "trainingStarted": False,
        "receiptEligible": False,
        "publicationEligible": False,
        "autonomyEligible": False,
        "claimBoundary": (
            "This is a runtime/GPU preflight only. It is not training, evaluation, "
            "publication, deployment, or a live inference witness."
        ),
    }
    report["reportSha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    resolved = args.report.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        parser.error("--report must be outside the repository")
    try:
        report = qualify(args.source_commit)
        code = 0
    except Exception as exc:  # noqa: BLE001 - fail closed with bounded evidence
        report = {
            "schema": "szl.frontier-runtime-preflight/v1",
            "state": "UNAVAILABLE",
            "measuredAt": datetime.now(timezone.utc).isoformat(),
            "fatal": sanitized_error(exc),
            "trainingStarted": False,
            "receiptEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }
        report["reportSha256"] = sha256_json(report)
        code = 1
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    resolved.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
