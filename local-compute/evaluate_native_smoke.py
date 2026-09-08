#!/usr/bin/env python3
"""Reload a locally trained candidate and compare six frozen checks; never promote."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

# Explicit repository-owned modules; -I still excludes user/global PYTHONPATH.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_ollama as benchmark  # noqa: E402
import train_receiptagent as training  # noqa: E402


def admit_candidate(directory, report_path, expected_report_sha256):
    training.require(training.HEX64.fullmatch(expected_report_sha256 or ""), "TRUSTED_REPORT_DIGEST_REQUIRED")
    raw = report_path.read_bytes()
    training.require(training.sha(raw) == expected_report_sha256, "TRAINING_REPORT_NOT_AUTHENTICATED")
    report = training.strict_json(raw)
    unsigned = dict(report)
    claimed = unsigned.pop("report_sha256", None)
    training.require(training.sha(training.canonical(unsigned)) == claimed, "TRAINING_REPORT_INTEGRITY")
    training.require(report.get("state") == "MEASURED_LOCAL_CONTINUATION_COMPLETED", "TRAINING_NOT_COMPLETE")
    for field, repo, revision in (("base", training.BASE_REPO, training.BASE_REVISION),
                                  ("adapter_parent", training.ADAPTER_REPO, training.ADAPTER_REVISION)):
        training.require(report.get(field, {}).get("repo_id") == repo and report[field].get("revision") == revision, "CANDIDATE_LINEAGE")
    training.require(report.get("parent_artifacts_unchanged") is True, "PARENT_INTEGRITY_NOT_PROVED")
    files = report.get("candidate_files")
    training.require(isinstance(files, list) and 1 <= len(files) <= 32, "CANDIDATE_FILES")
    declared = set()
    for item in files:
        name = item.get("path")
        path = training.artifact_path(directory, name)
        training.require(name not in declared and path.is_file(), "CANDIDATE_FILE_MISSING")
        training.require(path.stat().st_size == item.get("bytes") and training.file_sha(path) == item.get("sha256"), "CANDIDATE_BYTES_MISMATCH")
        training.require(not name.endswith((".bin", ".pt", ".pth", ".pkl", ".joblib", ".py")), "UNSAFE_CANDIDATE_SERIALIZATION")
        declared.add(name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    training.require(actual == declared and "adapter_model.safetensors" in declared, "CANDIDATE_FILE_SET")
    config = training.strict_json((directory / "adapter_config.json").read_bytes())
    training.validate_adapter_config(config)
    training.require(config.get("revision") == training.BASE_REVISION, "CANDIDATE_BASE_PIN_REQUIRED")
    return report


def score(case, response, eos_reached):
    return eos_reached is True and benchmark.grade(case, response)


def evaluate(args, report):
    training.configure_local_environment(args.output.parent)
    training.verify_artifact(args.base_dir, args.base_manifest, training.BASE_REPO, training.BASE_REVISION)
    training.verify_artifact(args.parent_dir, args.parent_manifest, training.ADAPTER_REPO, training.ADAPTER_REVISION)
    candidate = admit_candidate(args.candidate_dir, args.training_report, args.expected_report_sha256)
    report["candidate_id"] = candidate["candidate_id"]
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    started = time.monotonic()
    guard_args = argparse.Namespace(**vars(args))
    guard_args.output = args.output.parent
    training.runtime_guard(torch, guard_args, started, before_load=True)
    processor = AutoProcessor.from_pretrained(str(args.base_dir), local_files_only=True, trust_remote_code=False)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.base_dir),
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    training.require(not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "BASE_LOADING_KEYS")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(model, str(args.parent_dir), adapter_name="parent",
            is_trainable=False, local_files_only=True)
        model.load_adapter(str(args.candidate_dir), adapter_name="candidate", is_trainable=False, local_files_only=True)
    training.require(not any(any(cue in str(w.message).lower() for cue in ("missing", "unexpected", "ignored", "mismatch")) for w in caught), "ADAPTER_LOADING_WARNING")
    model.to("cuda").eval()
    report["gpu"] = torch.cuda.get_device_name(0)
    report["versions"] = {name: training.importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    training.require(bool(eos_ids), "EOS_ID_REQUIRED")
    for name in ("parent", "candidate"):
        model.set_adapter(name)
        model.requires_grad_(False)
        result = {"adapter": name, "cases": []}
        report["models"].append(result)
        for case in benchmark.CASES:
            training.runtime_guard(torch, guard_args, started)
            messages = [{"role": "system", "content": benchmark.SYSTEM}, {"role": "user", "content": case["prompt"]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
            prompt_tokens = inputs["input_ids"].shape[1]
            training.require(prompt_tokens + 128 <= 1024, "CONTEXT_BUDGET")
            beginning = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=128, use_cache=True)
            ids = generated[0, prompt_tokens:]
            eos_reached = len(ids) > 0 and int(ids[-1]) in eos_ids
            response = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            row = {"id": case["id"], "response": response, "eos_reached": eos_reached,
                "generated_tokens": len(ids), "wall_seconds": time.perf_counter() - beginning,
                "passed": score(case, response, eos_reached)}
            result["cases"].append(row)
            print(json.dumps({"adapter": name, "case": case["id"], "passed": row["passed"]}), flush=True)
            del inputs, generated, ids
        result["passed"] = sum(row["passed"] for row in result["cases"])
    training.verify_artifact(args.base_dir, args.base_manifest, training.BASE_REPO, training.BASE_REVISION)
    training.verify_artifact(args.parent_dir, args.parent_manifest, training.ADAPTER_REPO, training.ADAPTER_REVISION)
    admit_candidate(args.candidate_dir, args.training_report, args.expected_report_sha256)
    report["state"] = "MEASURED_RELOADED_NATIVE_SMOKE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-dir", "base-manifest", "parent-dir", "parent-manifest", "candidate-dir", "training-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    args = parser.parse_args()
    training.require(not args.output.exists(), "OUTPUT_ALREADY_EXISTS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.max_runtime_seconds, args.max_temperature_c, args.min_free_gib = 1200, 80, 4
    report = {"schema": "szl.native-reloaded-smoke/v1", "state": "UNAVAILABLE", "models": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "provider_cost_usd": 0,
        "publication_eligible": False, "autonomy_eligible": False,
        "claim_boundary": "Six synthetic checks only; not broad capability, safety or clinical qualification.",
        "expected_training_report_sha256": args.expected_report_sha256,
        "runner_sha256": training.file_sha(Path(__file__)),
        "grader_sha256": training.file_sha(HERE / "benchmark_ollama.py"),
        "protocol_sha256": benchmark.digest(benchmark.canonical({"system": benchmark.SYSTEM, "cases": benchmark.CASES})),
        "generation": {"max_new_tokens": 128, "context_budget": 1024, "do_sample": False, "thinking": False}}
    lock, owned = HERE / ".native-training.lock", False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write("native-evaluation")
        owned = True
        evaluate(args, report)
    except BaseException as error:
        report["state"] = "FAILED_CLOSED"
        report["error_type"] = type(error).__name__
        report["error_code"] = str(error) if isinstance(error, training.GateError) else "LOCAL_RELOAD_OR_EVALUATION_FAILED"
    finally:
        if owned:
            lock.unlink()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"state": report["state"], "publication_eligible": False}))
    return 0 if report["state"] == "MEASURED_RELOADED_NATIVE_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
