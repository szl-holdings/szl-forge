#!/usr/bin/env python3
"""Export an allowlisted, non-promoting summary of local batch evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_report(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("report exceeds bound")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("report is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


def summarize(ollama_path: Path, training_paths: list[Path], native_path: Path | None = None) -> dict:
    ollama, ollama_hash = read_report(ollama_path)
    if ollama.get("schema") != "szl.local-ollama-smoke/v2":
        raise ValueError("v2 protocol required")
    if type(ollama.get("provider_cost_usd")) is not int or ollama["provider_cost_usd"] != 0:
        raise ValueError("zero-provider-cost evidence required")
    result = {"schema": "szl.local-batch-summary/v1", "provider_cost_usd": 0,
              "electricity_cost": "NOT_MEASURED", "publication_eligible": False,
              "autonomy_eligible": False,
              "ollama": {"raw_report_sha256": ollama_hash,
                  **select(ollama, ("started_at", "finished_at", "runtime", "protocol_sha256", "runner_sha256", "generation")),
                  "models": [select(model, ("name", "digest", "state", "passed", "completed", "median_wall_seconds"))
                             for model in ollama["models"]]}, "training": []}
    for path in training_paths:
        training, raw_hash = read_report(path)
        if training.get("schema") != "szl.native-bf16-continuation/v1":
            raise ValueError("unrecognized training protocol")
        if type(training.get("provider_cost_usd")) is not int or training["provider_cost_usd"] != 0:
            raise ValueError("zero-provider-cost evidence required")
        without_hash = dict(training)
        claimed = without_hash.pop("report_sha256", None)
        canonical = json.dumps(without_hash, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != claimed:
            raise ValueError("training report integrity mismatch")
        summary = select(training, ("candidate_id", "state", "recipe", "source_revision", "runner_sha256",
            "versions", "elapsed_seconds", "trainable_parameters", "unique_training_rows",
            "training_rows_oversampled", "heldout_rows", "exact_prompt_overlap", "baseline_heldout_loss",
            "post_heldout_loss", "heldout_loss_delta", "parent_artifacts_unchanged", "error_code",
            "publication_eligible", "autonomy_eligible"))
        summary["raw_report_sha256"] = raw_hash
        summary["curriculum_source_revision"] = summary.pop("source_revision", "UNAVAILABLE")
        summary["optimizer_steps_completed"] = len(training.get("steps", []))
        summary["candidate_weight_files"] = [select(item, ("path", "bytes", "sha256"))
            for item in training.get("candidate_files", []) if item.get("path", "").endswith(".safetensors")]
        summary["promotion_decision"] = "HOLD_NOT_QUALIFIED"
        result["training"].append(summary)
    if native_path is not None:
        native, native_hash = read_report(native_path)
        if native.get("schema") != "szl.native-reloaded-smoke/v1" or native.get("provider_cost_usd") != 0:
            raise ValueError("native zero-provider-cost smoke evidence required")
        result["native_reload"] = {"raw_report_sha256": native_hash,
            **select(native, ("state", "candidate_id", "started_at", "finished_at", "expected_training_report_sha256",
                              "runner_sha256", "grader_sha256", "protocol_sha256", "generation", "versions")),
            "models": [{"adapter": model["adapter"], "passed": model.get("passed"),
                "cases": [select(row, ("id", "passed", "eos_reached", "generated_tokens", "wall_seconds")) for row in model.get("cases", [])]}
                for model in native.get("models", [])]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", type=Path, required=True)
    parser.add_argument("--training", type=Path, action="append", default=[])
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.ollama, args.training, args.native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "models": len(report["ollama"]["models"]),
                      "training_attempts": len(report["training"]), "publication_eligible": False}))
