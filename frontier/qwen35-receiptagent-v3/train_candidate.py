#!/usr/bin/env python3
"""Train a bounded ReceiptAgent v3 adapter from exact, freshly observed main."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE = "frontier/qwen35-receiptagent-v3"
GIB = 1024**3
CANONICAL_ORIGINS = {
    "https://github.com/szl-holdings/szl-forge",
    "https://github.com/szl-holdings/szl-forge.git",
    "git@github.com:szl-holdings/szl-forge.git",
}
ALLOWED_ADAPTER_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "video_preprocessor_config.json",
}
RUNTIME_PACKAGES = (
    "unsloth",
    "torch",
    "transformers",
    "trl",
    "datasets",
    "peft",
    "bitsandbytes",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "unsloth-zoo",
)


class QualificationError(RuntimeError):
    """A mandatory source, runtime, data, hardware, or artifact gate failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, capture_output=True, timeout=60
    )


def fresh_exact_source(source_commit: str) -> dict[str, Any]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise QualificationError("source commit must be an exact lowercase Git SHA")
    origin = git("remote", "get-url", "origin").stdout.decode().strip()
    if origin not in CANONICAL_ORIGINS:
        raise QualificationError("origin does not identify the canonical szl-forge repository")
    remote = git("ls-remote", "--exit-code", "origin", "refs/heads/main")
    parts = remote.stdout.decode().strip().split()
    if len(parts) != 2 or parts[1] != "refs/heads/main":
        raise QualificationError("fresh remote main observation was malformed")
    remote_main = parts[0]
    if remote_main != source_commit:
        raise QualificationError(
            f"fresh remote main {remote_main} does not match source {source_commit}"
        )
    head = git("rev-parse", "HEAD").stdout.decode().strip()
    branch = git("branch", "--show-current").stdout.decode().strip()
    cached_main = git("rev-parse", "refs/remotes/origin/main").stdout.decode().strip()
    dirty = git("status", "--porcelain", "--untracked-files=all").stdout.decode().strip()
    if head != source_commit:
        raise QualificationError(f"HEAD {head} does not match source {source_commit}")
    if cached_main != source_commit:
        raise QualificationError("cached origin/main differs from the freshly observed main")
    if branch != "main":
        raise QualificationError(f"training requires local main, observed {branch!r}")
    if dirty:
        raise QualificationError("training checkout is dirty")
    return {
        "repository": "szl-holdings/szl-forge",
        "revision": source_commit,
        "branch": "main",
        "originIdentityVerified": True,
        "freshRemoteMainObserved": True,
        "cachedRemoteTrackingMatches": True,
        "workingTreeClean": True,
        "commitSignatureVerifiedByThisTool": False,
    }


def supervised_local_source(source_commit: str) -> dict[str, Any]:
    """Recheck local exact-main state without giving the worker network credentials."""
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise QualificationError("source commit must be an exact lowercase Git SHA")
    origin = git("remote", "get-url", "origin").stdout.decode().strip()
    if origin not in CANONICAL_ORIGINS:
        raise QualificationError("origin does not identify the canonical szl-forge repository")
    head = git("rev-parse", "HEAD").stdout.decode().strip()
    branch = git("branch", "--show-current").stdout.decode().strip()
    cached_main = git("rev-parse", "refs/remotes/origin/main").stdout.decode().strip()
    dirty = git("status", "--porcelain", "--untracked-files=all").stdout.decode().strip()
    if head != source_commit:
        raise QualificationError(f"HEAD {head} does not match source {source_commit}")
    if cached_main != source_commit:
        raise QualificationError("cached origin/main differs from the supervised source")
    if branch != "main":
        raise QualificationError(f"training requires local main, observed {branch!r}")
    if dirty:
        raise QualificationError("training checkout is dirty")
    return {
        "repository": "szl-holdings/szl-forge",
        "revision": source_commit,
        "branch": "main",
        "originIdentityVerified": True,
        "freshRemoteMainObserved": False,
        "freshRemoteMainObservationDelegatedToSupervisor": True,
        "cachedRemoteTrackingMatches": True,
        "workingTreeClean": True,
        "commitSignatureVerifiedByThisTool": False,
    }


def committed_bytes(source_commit: str, path: str) -> bytes:
    return git("show", f"{source_commit}:{path}").stdout


def load_committed_json(source_commit: str, filename: str) -> dict[str, Any]:
    return json.loads(committed_bytes(source_commit, f"{RELATIVE}/{filename}"))


def curriculum(source_commit: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_bytes = committed_bytes(source_commit, f"{RELATIVE}/curriculum-manifest.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "szl.receiptagent-v3-curriculum-manifest/v2":
        raise QualificationError("curriculum manifest schema is unsupported")
    declared_files = manifest.get("files", {})
    train_entry = declared_files.get("train.jsonl")
    if not isinstance(train_entry, dict) or train_entry.get("trainingEligible") is not True:
        raise QualificationError("manifest does not admit train.jsonl")
    for held_out in ("dev.jsonl", "test.jsonl"):
        entry = declared_files.get(held_out)
        if not isinstance(entry, dict) or entry.get("trainingEligible") is not False:
            raise QualificationError(f"manifest does not exclude {held_out}")

    # This is the only split content the trainer opens. Held-out bytes are never read.
    train_bytes = committed_bytes(source_commit, f"{RELATIVE}/train.jsonl")
    train_sha = sha256_bytes(train_bytes)
    if train_sha != train_entry.get("sha256"):
        raise QualificationError("train.jsonl differs from its committed manifest digest")
    rows = [json.loads(line) for line in train_bytes.splitlines() if line.strip()]
    if len(rows) != 180 or len(rows) != train_entry.get("rows"):
        raise QualificationError("v3 requires exactly 180 admitted training rows")
    if any(row.get("split") != "TRAIN" for row in rows):
        raise QualificationError("a non-training split entered admitted rows")
    if any(
        row.get("rightsBasis") != "PROJECT_AUTHORED_POLICY_AND_SCHEMA" for row in rows
    ):
        raise QualificationError("a training row lacks the project-authored rights basis")
    if any(row.get("datasetOrigin") != "PROJECT_AUTHORED_SYNTHETIC" for row in rows):
        raise QualificationError("a training row has an unapproved dataset origin")
    if len({row.get("caseId") for row in rows}) != len(rows):
        raise QualificationError("training case IDs are not unique")
    inputs = [canonical_json(row.get("messages", [])[:2]) for row in rows]
    targets = [row.get("messages", [])[-1].get("content") for row in rows]
    if len(set(inputs)) != len(rows):
        raise QualificationError("duplicate training inputs are forbidden")
    if len(set(targets)) != len(rows):
        raise QualificationError("duplicate training targets are forbidden")
    kind_counts = {
        kind: sum(row.get("kind") == kind for row in rows)
        for kind in ("DRAFT", "RECOVERY", "REFUSAL")
    }
    if kind_counts != {"DRAFT": 60, "RECOVERY": 60, "REFUSAL": 60}:
        raise QualificationError(f"training strata drifted: {kind_counts}")
    source_bundle = {
        "manifestSha256": sha256_bytes(manifest_bytes),
        "trainSha256": train_sha,
        "trainBytes": len(train_bytes),
        "uniqueTrainingRows": len(rows),
        "kindCounts": kind_counts,
        "heldOutCommitments": {
            name: {
                "rows": declared_files[name]["rows"],
                "sha256": declared_files[name]["sha256"],
            }
            for name in ("dev.jsonl", "test.jsonl")
        },
        "trainerOpenedSplitContent": ["TRAIN"],
    }
    source_bundle["bundleSha256"] = sha256_json(source_bundle)
    return source_bundle, rows


def runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def enforce_runtime_lock(candidate: dict[str, Any]) -> dict[str, str]:
    observed = runtime_versions()
    expected = candidate["runtime_lock"]
    if observed != expected:
        drift = {
            name: {"expected": expected.get(name), "observed": observed.get(name)}
            for name in sorted(set(expected) | set(observed))
            if observed.get(name) != expected.get(name)
        }
        raise QualificationError(f"runtime package lock drifted: {drift}")
    return observed


def gpu_temperature_c(executable: str = "nvidia-smi") -> int:
    measured = subprocess.run(
        [
            executable,
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values = [line.strip() for line in measured.stdout.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise QualificationError("GPU temperature probe returned an unexpected result")
    return int(values[0])


def raw_gpu_preflight(
    policy: dict[str, Any], executable: str = "nvidia-smi"
) -> dict[str, Any]:
    measured = subprocess.run(
        [
            executable,
            "--query-gpu=uuid,name,temperature.gpu,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    lines = [line.strip() for line in measured.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise QualificationError("raw GPU preflight requires exactly one visible GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5:
        raise QualificationError("raw GPU preflight returned an unexpected result")
    gpu_uuid, name, temperature_text, free_mib_text, total_mib_text = fields
    try:
        temperature = int(temperature_text)
        free_mib = int(free_mib_text)
        total_mib = int(total_mib_text)
    except ValueError as exc:
        raise QualificationError("raw GPU preflight returned non-numeric telemetry") from exc
    max_temp = int(policy["maximum_gpu_temperature_c"])
    min_free_gib = float(policy["minimum_free_gpu_gib"])
    if temperature > max_temp:
        raise QualificationError(
            f"GPU temperature {temperature} C exceeds the fixed {max_temp} C policy"
        )
    if free_mib < int(min_free_gib * 1024):
        raise QualificationError(
            f"GPU free memory {free_mib / 1024:.2f} GiB is below {min_free_gib:.2f} GiB"
        )
    return {
        "uuid": gpu_uuid,
        "name": name,
        "temperatureCBeforeRuntimeImport": temperature,
        "freeMiBBeforeRuntimeImport": free_mib,
        "totalMiB": total_mib,
    }


def gpu_gate(
    torch: Any, policy: dict[str, Any], telemetry_executable: str = "nvidia-smi"
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise QualificationError("CUDA is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    device = torch.cuda.get_device_properties(0)
    temperature = gpu_temperature_c(telemetry_executable)
    min_free_gib = float(policy["minimum_free_gpu_gib"])
    max_temp_c = int(policy["maximum_gpu_temperature_c"])
    if free_bytes < int(min_free_gib * GIB):
        raise QualificationError(
            f"GPU free memory {free_bytes / GIB:.2f} GiB is below {min_free_gib:.2f} GiB"
        )
    if temperature > max_temp_c:
        raise QualificationError(
            f"GPU temperature {temperature} C exceeds the fixed {max_temp_c} C policy"
        )
    return {
        "name": device.name,
        "computeCapability": f"{device.major}.{device.minor}",
        "totalBytes": total_bytes,
        "freeBytesBeforeLoad": free_bytes,
        "temperatureCBeforeLoad": temperature,
        "maximumTemperaturePolicyC": max_temp_c,
        "minimumFreeMemoryPolicyGiB": min_free_gib,
        "torchVersion": torch.__version__,
        "cudaRuntime": torch.version.cuda,
    }


def validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise QualificationError("output directory must be outside the repository")
    if resolved.exists():
        if not resolved.is_dir():
            raise QualificationError("output path exists and is not a directory")
        if any(resolved.iterdir()):
            raise QualificationError("output directory must be new or empty")
    else:
        resolved.mkdir(parents=True)
    return resolved


def vlm_conversation(row: dict[str, Any]) -> dict[str, Any]:
    converted = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise QualificationError("training rows require system, user, and assistant messages")
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise QualificationError("training message has no text content")
        if "<think>" in content.lower() or "</think>" in content.lower():
            raise QualificationError("thinking tags are forbidden in v3 training data")
        converted.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": content}],
            }
        )
    return {"messages": converted}


def hash_adapter(directory: Path) -> tuple[str, list[dict[str, Any]]]:
    from safetensors import safe_open

    files: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    observed_names: set[str] = set()
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise QualificationError(f"adapter artifact contains a symlink: {path.name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise QualificationError(f"adapter artifact is not a regular file: {path.name}")
        relative = path.relative_to(directory).as_posix()
        if "/" in relative or relative not in ALLOWED_ADAPTER_FILES:
            raise QualificationError(f"adapter artifact file is not allowlisted: {relative}")
        observed_names.add(relative)
        data = path.read_bytes()
        if len(data) > 256 * 1024 * 1024:
            raise QualificationError(f"adapter artifact file is unexpectedly large: {relative}")
        parsed: dict[str, Any] = {}
        if path.suffix == ".json":
            parsed_json = json.loads(data.decode("utf-8"))
            if not isinstance(parsed_json, dict):
                raise QualificationError(f"adapter JSON must be an object: {relative}")
            parsed["jsonKeys"] = len(parsed_json)
        elif path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor_keys = list(handle.keys())
            if not tensor_keys:
                raise QualificationError("SafeTensors adapter contains no tensors")
            parsed["tensorCount"] = len(tensor_keys)
        elif path.suffix in {".md", ".jinja", ".model"}:
            if path.suffix != ".model":
                data.decode("utf-8")
        digest = sha256_bytes(data)
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        files.append(
            {"path": relative, "bytes": len(data), "sha256": digest, **parsed}
        )
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(observed_names):
        raise QualificationError("adapter save omitted required config or SafeTensors weights")
    return combined.hexdigest(), files


def sanitized_error(exc: Exception) -> str:
    message = str(exc).replace(str(ROOT), "<REPOSITORY>")
    message = re.sub(r"[A-Za-z]:\\[^\s]+", "<LOCAL_PATH>", message)
    message = re.sub(
        r"(?i)(authorization\s*:\s*bearer|bearer)\s+\S+",
        r"\1 <REDACTED>",
        message,
    )
    message = re.sub(
        r"(?i)\b(hf_token|token|secret|password|private[_-]?key)\s*[:=]\s*\S+",
        r"\1=<REDACTED>",
        message,
    )
    message = re.sub(
        r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@",
        r"\1<REDACTED>@",
        message,
    )
    return f"{type(exc).__name__}: {message[:500]}"


def train(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    source = supervised_local_source(args.source_commit)
    candidate = load_committed_json(args.source_commit, "candidate.json")
    if candidate.get("candidate_id") != "SZL-ReceiptAgent-Qwen3.5-0.8B-v3":
        raise QualificationError("unexpected candidate identity")
    if candidate.get("state") != "SOURCE_READY_NOT_TRAINED":
        raise QualificationError("candidate source state is not ready")
    if candidate.get("publication_eligible") is not False:
        raise QualificationError("source-ready candidate cannot be publication eligible")
    if candidate.get("autonomy_eligible") is not False:
        raise QualificationError("candidate cannot be autonomy eligible")
    recipe = candidate["training_recipe"]
    telemetry_executable = candidate["supervision_policy"]["nvidia_smi_executable"]
    supervision_policy_sha = sha256_json(candidate["supervision_policy"])
    training_recipe_sha = sha256_json(recipe)
    worker_source_sha = sha256_bytes(
        committed_bytes(args.source_commit, f"{RELATIVE}/train_candidate.py")
    )
    expected_steps = (
        recipe["smoke_optimizer_steps"]
        if args.run_kind == "smoke"
        else recipe["full_optimizer_steps"]
    )
    source_bundle, rows = curriculum(args.source_commit)
    raw_gpu = raw_gpu_preflight(recipe, telemetry_executable)

    import unsloth  # noqa: F401 - patch before Transformers/PEFT imports
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.chat_templates import train_on_responses_only
    from unsloth.trainer import UnslothVisionDataCollator

    versions = enforce_runtime_lock(candidate)
    gpu = gpu_gate(torch, recipe, telemetry_executable)
    gpu["preRuntimeImport"] = raw_gpu

    thermal_samples = [gpu["temperatureCBeforeLoad"]]
    max_temp_c = int(recipe["maximum_gpu_temperature_c"])

    class ThermalGuard(TrainerCallback):
        def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            temperature = gpu_temperature_c(telemetry_executable)
            thermal_samples.append(temperature)
            if temperature > max_temp_c:
                raise QualificationError(
                    f"GPU temperature {temperature} C exceeded fixed {max_temp_c} C policy "
                    f"at optimizer step {state.global_step}"
                )
            return control

    implementation = candidate["actual_training_base"]
    model, processor = FastVisionModel.from_pretrained(
        model_name=implementation["repo_id"],
        revision=implementation["revision"],
        load_in_4bit=implementation["load_in_4bit"],
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=recipe["finetune_vision_layers"],
        finetune_language_layers=recipe["finetune_language_layers"],
        finetune_attention_modules=recipe["finetune_attention_modules"],
        finetune_mlp_modules=recipe["finetune_mlp_modules"],
        r=recipe["lora_r"],
        lora_alpha=recipe["lora_alpha"],
        lora_dropout=recipe["lora_dropout"],
        bias="none",
        random_state=recipe["seed"],
        use_rslora=False,
        loftq_config=None,
    )
    checkpoints = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    warmup_steps = recipe["warmup_steps"] if args.run_kind == "full" else 0
    config_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": recipe["per_device_batch_size"],
        "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
        "warmup_steps": warmup_steps,
        "max_steps": expected_steps,
        "learning_rate": recipe["learning_rate"],
        "logging_steps": 1,
        "optim": recipe["optimizer"],
        "weight_decay": recipe["weight_decay"],
        "lr_scheduler_type": recipe["lr_scheduler"],
        "seed": recipe["seed"],
        "output_dir": str(checkpoints),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "eos_token": processor.tokenizer.eos_token,
        "pad_token": processor.tokenizer.pad_token or processor.tokenizer.eos_token,
        "max_length": recipe["max_length"],
        "save_strategy": "no",
    }
    converted = [vlm_conversation(row) for row in rows]
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=Dataset.from_list(converted),
        args=SFTConfig(**config_kwargs),
        callbacks=[ThermalGuard()],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        tokenizer=processor,
    )
    FastVisionModel.for_training(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stats = trainer.train()
    duration = time.perf_counter() - started
    if int(stats.global_step) != expected_steps:
        raise QualificationError(
            f"trainer completed {stats.global_step} steps, expected {expected_steps}"
        )
    terminal_temperature = gpu_temperature_c(telemetry_executable)
    thermal_samples.append(terminal_temperature)
    if terminal_temperature > max_temp_c:
        raise QualificationError("GPU exceeded the fixed thermal policy at run completion")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    adapter_sha, adapter_files = hash_adapter(adapter_dir)
    metrics = {
        key: value
        for key, value in stats.metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    is_full = args.run_kind == "full"
    state = (
        "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED"
        if is_full
        else "MEASURED_SMOKE_COMPLETED_NOT_QUALIFIED"
    )
    scheduled_examples = expected_steps * recipe["gradient_accumulation_steps"]
    report = {
        "schema": "szl.frontier-training-run/v3",
        "candidateId": candidate["candidate_id"],
        "supervisorRunId": args.supervisor_run_id,
        "supervisionPolicySha256": supervision_policy_sha,
        "workerSourceSha256": worker_source_sha,
        "trainingRecipeSha256": training_recipe_sha,
        "state": state,
        "runKind": args.run_kind.upper(),
        "measuredAt": datetime.now(timezone.utc).isoformat(),
        "hostClass": "LOCAL_GPU_RUNNER_REDACTED",
        "source": source,
        "sourceBundle": source_bundle,
        "implementation": implementation,
        "runtimePackages": versions,
        "uniqueTrainingRows": len(converted),
        "scheduledExamples": scheduled_examples,
        "optimizerSteps": expected_steps,
        "configuration": {
            **config_kwargs,
            "output_dir": "<OUTSIDE_REPOSITORY>/checkpoints",
            "finetuneVisionLayers": recipe["finetune_vision_layers"],
            "loraR": recipe["lora_r"],
            "loraAlpha": recipe["lora_alpha"],
            "responseOnlyLoss": recipe["response_only_loss"],
            "enableThinking": recipe["enable_thinking"],
        },
        "gpu": {
            **gpu,
            "temperatureSamplesC": thermal_samples,
            "maximumObservedTemperatureC": max(thermal_samples),
            "temperatureCAfterRun": terminal_temperature,
            "peakReservedBytesTraining": torch.cuda.max_memory_reserved(),
        },
        "training": {"durationSeconds": round(duration, 6), "metrics": metrics},
        "adapter": {
            "relativePath": "adapter",
            "formatPolicy": "PARSED_SAFETENSORS_AND_ALLOWLISTED_METADATA",
            "aggregateSha256": adapter_sha,
            "files": adapter_files,
        },
        "integrityDigestIsAuthentication": False,
        "authenticatedTrainingEnvelopePresent": False,
        "qualificationEligible": is_full,
        "receiptEligible": False,
        "publicationEligible": False,
        "autonomyEligible": False,
        "claimBoundary": (
            "This is measured local training, not evaluation or authenticated receipt "
            "evidence. Train loss is not an evaluation metric."
        ),
    }
    report["reportSha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supervisor-run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.supervisor_run_id):
        parser.error("--supervisor-run-id must be exactly 32 lowercase hex characters")
    output_admitted = False
    report_published = False
    try:
        output_dir = validate_output_dir(args.output_dir)
        output_admitted = True
        report = train(args, output_dir)
        code = 0
    except Exception as exc:  # noqa: BLE001 - fail closed with bounded evidence
        report = {
            "schema": "szl.frontier-training-run/v3",
            "state": "UNAVAILABLE",
            "supervisorRunId": args.supervisor_run_id,
            "runKind": args.run_kind.upper(),
            "measuredAt": datetime.now(timezone.utc).isoformat(),
            "fatal": sanitized_error(exc),
            "qualificationEligible": False,
            "receiptEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }
        report["reportSha256"] = sha256_json(report)
        code = 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    try:
        if output_admitted:
            report_path = output_dir / "training-report.json"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(report_path, flags, 0o600)
            try:
                data = rendered.encode("utf-8")
                while data:
                    written = os.write(descriptor, data)
                    data = data[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            report_published = True
    except OSError:
        code = 1
    if report_published:
        print(rendered, end="")
    else:
        print(
            json.dumps(
                {
                    "schema": "szl.frontier-training-run/v3",
                    "state": "UNAVAILABLE_REPORT_NOT_PUBLISHED",
                    "runKind": args.run_kind.upper(),
                    "supervisorRunId": args.supervisor_run_id,
                    "receiptEligible": False,
                    "publicationEligible": False,
                    "autonomyEligible": False,
                },
                sort_keys=True,
            )
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
