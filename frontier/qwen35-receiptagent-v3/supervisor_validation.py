"""Strict, side-effect-free validation for successful supervised training reports.

This module validates report semantics only.  It does not authenticate a report,
inspect a live process, read an adapter, authorize evaluation, or mint a receipt.
The supervisor must independently obtain the expected source bundle, worker hash,
GPU UUID, and adapter hashes before calling :func:`validate_successful_report`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "szl.frontier-training-run/v3"
CLAIM_BOUNDARY = (
    "This is measured local training, not evaluation or authenticated receipt "
    "evidence. Train loss is not an evaluation metric."
)
SUCCESS_STATE_BY_KIND = {
    "SMOKE": "MEASURED_SMOKE_COMPLETED_NOT_QUALIFIED",
    "FULL": "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
}
OBSERVATION_STATE_BY_KIND = {
    "SMOKE": "SUPERVISOR_OBSERVED_SMOKE_OUTPUT_BOUND_NOT_QUALIFIED",
    "FULL": "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
}
HEX_32 = re.compile(r"^[0-9a-f]{32}$")
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]{16,96}$")

TOP_LEVEL_KEYS = {
    "schema",
    "candidateId",
    "supervisorRunId",
    "supervisionPolicySha256",
    "workerSourceSha256",
    "trainingRecipeSha256",
    "state",
    "runKind",
    "measuredAt",
    "hostClass",
    "source",
    "sourceBundle",
    "implementation",
    "runtimePackages",
    "uniqueTrainingRows",
    "scheduledExamples",
    "optimizerSteps",
    "configuration",
    "gpu",
    "training",
    "adapter",
    "integrityDigestIsAuthentication",
    "authenticatedTrainingEnvelopePresent",
    "qualificationEligible",
    "receiptEligible",
    "publicationEligible",
    "autonomyEligible",
    "claimBoundary",
    "reportSha256",
}

CONFIGURATION_KEYS = {
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "max_steps",
    "learning_rate",
    "logging_steps",
    "optim",
    "weight_decay",
    "lr_scheduler_type",
    "seed",
    "output_dir",
    "report_to",
    "remove_unused_columns",
    "dataset_text_field",
    "dataset_kwargs",
    "eos_token",
    "pad_token",
    "max_length",
    "save_strategy",
    "finetuneVisionLayers",
    "loraR",
    "loraAlpha",
    "responseOnlyLoss",
    "enableThinking",
}

TRAINING_RECIPE_KEYS = {
    "smoke_optimizer_steps",
    "full_optimizer_steps",
    "full_scheduled_examples",
    "full_epochs_over_unique_rows",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "max_length",
    "learning_rate",
    "warmup_steps",
    "optimizer",
    "weight_decay",
    "lr_scheduler",
    "seed",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "response_only_loss",
    "finetune_vision_layers",
    "finetune_language_layers",
    "finetune_attention_modules",
    "finetune_mlp_modules",
    "enable_thinking",
    "minimum_free_gpu_gib",
    "maximum_gpu_temperature_c",
}

SOURCE_BUNDLE_KEYS = {
    "manifestSha256",
    "trainSha256",
    "trainBytes",
    "uniqueTrainingRows",
    "kindCounts",
    "heldOutCommitments",
    "trainerOpenedSplitContent",
    "bundleSha256",
}

GPU_KEYS = {
    "name",
    "computeCapability",
    "totalBytes",
    "freeBytesBeforeLoad",
    "temperatureCBeforeLoad",
    "maximumTemperaturePolicyC",
    "minimumFreeMemoryPolicyGiB",
    "torchVersion",
    "cudaRuntime",
    "preRuntimeImport",
    "temperatureSamplesC",
    "maximumObservedTemperatureC",
    "temperatureCAfterRun",
    "peakReservedBytesTraining",
}

PRE_RUNTIME_GPU_KEYS = {
    "uuid",
    "name",
    "temperatureCBeforeRuntimeImport",
    "freeMiBBeforeRuntimeImport",
    "totalMiB",
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


class SupervisorValidationError(ValueError):
    """A child report failed a mandatory supervisor binding."""


@dataclass(frozen=True)
class ValidatedTrainingReport:
    """Narrow result of successful semantic validation.

    ``local_evaluation_input_binding_satisfied`` is deliberately true only for
    the fixed full run.  It is not an evaluation, receipt, or publication gate.
    """

    observation_state: str
    run_kind: str
    candidate_id: str
    report_sha256: str
    source_revision: str
    source_bundle_sha256: str
    supervisor_run_id: str
    gpu_uuid: str
    worker_source_sha256: str
    training_recipe_sha256: str
    supervision_policy_sha256: str
    adapter_aggregate_sha256: str
    local_evaluation_input_binding_satisfied: bool
    authenticated_training_envelope_present: bool = False
    receipt_eligible: bool = False
    publication_eligible: bool = False
    autonomy_eligible: bool = False


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _fail(message: str) -> None:
    raise SupervisorValidationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{label} keys differ: missing={missing}, extra={extra}")


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        _fail(f"{label} differs")


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        _fail(f"{label} must be {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _finite_number(value: Any, label: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < (0.0 if allow_zero else 0.000000000001):
        _fail(
            f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return number


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        _fail("measuredAt must be an RFC3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SupervisorValidationError(
            "measuredAt must be an RFC3339 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("measuredAt must include a UTC offset")


def _validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_dict = _mapping(candidate, "candidate")
    required = {
        "candidate_id",
        "state",
        "actual_training_base",
        "training_data",
        "training_recipe",
        "supervision_policy",
        "runtime_lock",
        "publication_eligible",
        "autonomy_eligible",
    }
    missing = sorted(required - set(candidate_dict))
    if missing:
        _fail(f"candidate is missing required fields: {missing}")
    _exact(candidate_dict["state"], "SOURCE_READY_NOT_TRAINED", "candidate state")
    _exact(
        candidate_dict["publication_eligible"], False, "candidate publication boundary"
    )
    _exact(candidate_dict["autonomy_eligible"], False, "candidate autonomy boundary")
    recipe = _mapping(candidate_dict["training_recipe"], "candidate training recipe")
    _exact_keys(recipe, TRAINING_RECIPE_KEYS, "candidate training recipe")
    for key in (
        "smoke_optimizer_steps",
        "full_optimizer_steps",
        "full_scheduled_examples",
        "full_epochs_over_unique_rows",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "lora_r",
        "lora_alpha",
        "maximum_gpu_temperature_c",
    ):
        _positive_int(recipe[key], f"candidate training recipe.{key}")
    for key in ("warmup_steps", "seed"):
        _positive_int(recipe[key], f"candidate training recipe.{key}", allow_zero=True)
    for key in ("learning_rate", "minimum_free_gpu_gib"):
        _finite_number(
            recipe[key], f"candidate training recipe.{key}", allow_zero=False
        )
    for key in ("weight_decay", "lora_dropout"):
        _finite_number(recipe[key], f"candidate training recipe.{key}")
    for key in ("optimizer", "lr_scheduler"):
        if not isinstance(recipe[key], str) or not recipe[key]:
            _fail(f"candidate training recipe.{key} must be nonempty string")
    for key in (
        "response_only_loss",
        "finetune_vision_layers",
        "finetune_language_layers",
        "finetune_attention_modules",
        "finetune_mlp_modules",
        "enable_thinking",
    ):
        if type(recipe[key]) is not bool:
            _fail(f"candidate training recipe.{key} must be boolean")
    _exact(
        recipe["full_optimizer_steps"] * recipe["gradient_accumulation_steps"],
        recipe["full_scheduled_examples"],
        "candidate full scheduled examples",
    )
    training_data = _mapping(candidate_dict["training_data"], "candidate training data")
    for key, expected in (("train_rows", 180), ("dev_rows", 36), ("test_rows", 72)):
        _exact(training_data.get(key), expected, f"candidate training data.{key}")
    _exact(
        recipe["full_scheduled_examples"],
        training_data["train_rows"] * recipe["full_epochs_over_unique_rows"],
        "candidate full epochs over unique rows",
    )
    if not _mapping(
        candidate_dict["supervision_policy"], "candidate supervision policy"
    ):
        _fail("candidate supervision policy must not be empty")
    runtime_lock = _mapping(candidate_dict["runtime_lock"], "candidate runtime lock")
    if not runtime_lock or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in runtime_lock.items()
    ):
        _fail("candidate runtime lock must contain nonempty string bindings")
    _sha256(sha256_json(recipe), "computed recipe digest")
    _sha256(sha256_json(candidate_dict["supervision_policy"]), "computed policy digest")
    return candidate_dict


def _validate_source_bundle(
    observed: Any,
    expected: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    expected_dict = _mapping(expected, "expected source bundle")
    observed_dict = _mapping(observed, "sourceBundle")
    _exact_keys(expected_dict, SOURCE_BUNDLE_KEYS, "expected source bundle")
    if observed_dict != expected_dict:
        _fail("sourceBundle differs from the supervisor-computed bundle")
    bundle_sha = _sha256(observed_dict.get("bundleSha256"), "sourceBundle.bundleSha256")
    unsigned = dict(observed_dict)
    unsigned.pop("bundleSha256", None)
    if sha256_json(unsigned) != bundle_sha:
        _fail("sourceBundle self-digest is invalid")
    training_data = _mapping(candidate["training_data"], "candidate training_data")
    _sha256(observed_dict["manifestSha256"], "sourceBundle manifest digest")
    _sha256(observed_dict["trainSha256"], "sourceBundle train digest")
    _positive_int(observed_dict["trainBytes"], "sourceBundle train bytes")
    _exact(
        observed_dict.get("uniqueTrainingRows"),
        training_data.get("train_rows"),
        "sourceBundle unique training rows",
    )
    _exact(
        observed_dict.get("kindCounts"),
        {"DRAFT": 60, "RECOVERY": 60, "REFUSAL": 60},
        "sourceBundle kind counts",
    )
    _exact(
        observed_dict.get("trainerOpenedSplitContent"),
        ["TRAIN"],
        "opened split content",
    )
    held_out = _mapping(observed_dict.get("heldOutCommitments"), "held-out commitments")
    if set(held_out) != {"dev.jsonl", "test.jsonl"}:
        _fail("held-out commitments must bind exactly dev.jsonl and test.jsonl")
    for name, row_key in (("dev.jsonl", "dev_rows"), ("test.jsonl", "test_rows")):
        commitment = _mapping(held_out[name], f"held-out commitment {name}")
        _exact_keys(commitment, {"rows", "sha256"}, f"held-out commitment {name}")
        _exact(commitment["rows"], training_data[row_key], f"held-out rows {name}")
        _sha256(commitment["sha256"], f"held-out digest {name}")
    return observed_dict


def _expected_configuration(recipe: Mapping[str, Any], run_kind: str) -> dict[str, Any]:
    steps = (
        recipe["smoke_optimizer_steps"]
        if run_kind == "SMOKE"
        else recipe["full_optimizer_steps"]
    )
    warmup = 0 if run_kind == "SMOKE" else recipe["warmup_steps"]
    return {
        "per_device_train_batch_size": recipe["per_device_batch_size"],
        "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
        "warmup_steps": warmup,
        "max_steps": steps,
        "learning_rate": recipe["learning_rate"],
        "logging_steps": 1,
        "optim": recipe["optimizer"],
        "weight_decay": recipe["weight_decay"],
        "lr_scheduler_type": recipe["lr_scheduler"],
        "seed": recipe["seed"],
        "output_dir": "<OUTSIDE_REPOSITORY>/checkpoints",
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": recipe["max_length"],
        "save_strategy": "no",
        "finetuneVisionLayers": recipe["finetune_vision_layers"],
        "loraR": recipe["lora_r"],
        "loraAlpha": recipe["lora_alpha"],
        "responseOnlyLoss": recipe["response_only_loss"],
        "enableThinking": recipe["enable_thinking"],
    }


def _validate_configuration(
    observed: Any, recipe: Mapping[str, Any], run_kind: str
) -> None:
    configuration = _mapping(observed, "configuration")
    _exact_keys(configuration, CONFIGURATION_KEYS, "configuration")
    expected = _expected_configuration(recipe, run_kind)
    for key, value in expected.items():
        _exact(configuration.get(key), value, f"configuration.{key}")
    for key in ("eos_token", "pad_token"):
        if not isinstance(configuration.get(key), str) or not configuration[key]:
            _fail(f"configuration.{key} must be a nonempty string")


def _validate_gpu(observed: Any, recipe: Mapping[str, Any], expected_uuid: str) -> None:
    if not isinstance(expected_uuid, str) or GPU_UUID.fullmatch(expected_uuid) is None:
        _fail("expected GPU UUID is malformed")
    gpu = _mapping(observed, "gpu")
    _exact_keys(gpu, GPU_KEYS, "gpu")
    pre_runtime = _mapping(gpu["preRuntimeImport"], "gpu.preRuntimeImport")
    _exact_keys(pre_runtime, PRE_RUNTIME_GPU_KEYS, "gpu.preRuntimeImport")
    _exact(pre_runtime["uuid"], expected_uuid, "GPU UUID")
    _exact(pre_runtime["name"], gpu["name"], "GPU name across preflight/runtime")
    maximum = recipe["maximum_gpu_temperature_c"]
    minimum_free = recipe["minimum_free_gpu_gib"]
    _exact(gpu["maximumTemperaturePolicyC"], maximum, "GPU temperature policy")
    _exact(gpu["minimumFreeMemoryPolicyGiB"], minimum_free, "GPU memory policy")
    temperatures = gpu["temperatureSamplesC"]
    if not isinstance(temperatures, list) or not temperatures:
        _fail("gpu.temperatureSamplesC must be a nonempty array")
    if any(type(value) is not int for value in temperatures):
        _fail("GPU temperature samples must be integers")
    if temperatures[0] != gpu["temperatureCBeforeLoad"]:
        _fail("first GPU temperature sample differs from pre-load temperature")
    if temperatures[-1] != gpu["temperatureCAfterRun"]:
        _fail("last GPU temperature sample differs from terminal temperature")
    if max(temperatures) != gpu["maximumObservedTemperatureC"]:
        _fail("maximum GPU temperature was not recomputed from samples")
    all_temperatures = [
        pre_runtime["temperatureCBeforeRuntimeImport"],
        gpu["temperatureCBeforeLoad"],
        *temperatures,
    ]
    if any(
        type(value) is not int or value < 0 or value > maximum
        for value in all_temperatures
    ):
        _fail("GPU temperature evidence violates the fixed policy")
    _positive_int(pre_runtime["freeMiBBeforeRuntimeImport"], "pre-runtime free MiB")
    _positive_int(pre_runtime["totalMiB"], "pre-runtime total MiB")
    if pre_runtime["freeMiBBeforeRuntimeImport"] > pre_runtime["totalMiB"]:
        _fail("pre-runtime free GPU memory exceeds total GPU memory")
    if pre_runtime["freeMiBBeforeRuntimeImport"] < int(float(minimum_free) * 1024):
        _fail("pre-runtime GPU free memory violates the fixed policy")
    _positive_int(gpu["freeBytesBeforeLoad"], "pre-load free bytes")
    if gpu["freeBytesBeforeLoad"] < int(float(minimum_free) * 1024**3):
        _fail("pre-load GPU free memory violates the fixed policy")
    for key in ("totalBytes", "peakReservedBytesTraining"):
        _positive_int(
            gpu[key], f"gpu.{key}", allow_zero=key == "peakReservedBytesTraining"
        )
    if gpu["freeBytesBeforeLoad"] > gpu["totalBytes"]:
        _fail("pre-load free GPU memory exceeds total GPU memory")
    if gpu["peakReservedBytesTraining"] > gpu["totalBytes"]:
        _fail("peak reserved GPU memory exceeds total GPU memory")
    for key in ("name", "computeCapability", "torchVersion", "cudaRuntime"):
        if not isinstance(gpu[key], str) or not gpu[key]:
            _fail(f"gpu.{key} must be nonempty string")


def _validate_training(observed: Any) -> None:
    training = _mapping(observed, "training")
    _exact_keys(training, {"durationSeconds", "metrics"}, "training")
    _finite_number(
        training["durationSeconds"], "training.durationSeconds", allow_zero=False
    )
    metrics = _mapping(training["metrics"], "training.metrics")
    if "train_loss" not in metrics:
        _fail("training.metrics must contain train_loss")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            _fail("training metric names must be nonempty strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            _fail(f"training metric {key} has unsupported value")
        if isinstance(value, float) and not math.isfinite(value):
            _fail(f"training metric {key} is non-finite")
    _finite_number(metrics["train_loss"], "training.metrics.train_loss")


def _validate_adapter(
    observed: Any,
    expected_aggregate_sha256: str,
    expected_files: Sequence[Mapping[str, Any]],
) -> None:
    aggregate = _sha256(expected_aggregate_sha256, "expected adapter aggregate")
    adapter = _mapping(observed, "adapter")
    _exact_keys(
        adapter,
        {"relativePath", "formatPolicy", "aggregateSha256", "files"},
        "adapter",
    )
    _exact(adapter["relativePath"], "adapter", "adapter relative path")
    _exact(
        adapter["formatPolicy"],
        "PARSED_SAFETENSORS_AND_ALLOWLISTED_METADATA",
        "adapter format policy",
    )
    _exact(adapter["aggregateSha256"], aggregate, "adapter aggregate digest")
    if (
        not isinstance(expected_files, Sequence)
        or isinstance(expected_files, (str, bytes, bytearray))
        or not expected_files
    ):
        _fail("expected adapter files must be a nonempty sequence")
    expected_file_list = [
        _mapping(item, f"expected adapter file {index}")
        for index, item in enumerate(expected_files)
    ]
    if not isinstance(adapter["files"], list):
        _fail("adapter.files must be an array")
    if adapter["files"] != expected_file_list:
        _fail("adapter file evidence differs from the supervisor recomputation")
    paths = [item.get("path") for item in expected_file_list]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail("adapter file evidence must have unique sorted paths")
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(paths):
        _fail("adapter file evidence omits required files")
    for item in expected_file_list:
        if (
            not isinstance(item.get("path"), str)
            or "/" in item["path"]
            or "\\" in item["path"]
        ):
            _fail("adapter file path is not a flat allowlisted name")
        if item["path"] not in ALLOWED_ADAPTER_FILES:
            _fail("adapter file path is not a flat allowlisted name")
        suffix = item["path"].rsplit(".", 1)[-1]
        expected_keys = {"path", "bytes", "sha256"}
        if suffix == "json":
            expected_keys.add("jsonKeys")
        elif suffix == "safetensors":
            expected_keys.add("tensorCount")
        _exact_keys(item, expected_keys, f"adapter file {item['path']}")
        _positive_int(
            item["bytes"], f"adapter file {item['path']} bytes", allow_zero=True
        )
        if item["bytes"] > 256 * 1024 * 1024:
            _fail(f"adapter file {item['path']} exceeds the size policy")
        _sha256(item.get("sha256"), f"adapter file {item.get('path')} digest")
        if suffix == "json":
            _positive_int(
                item["jsonKeys"],
                f"adapter file {item['path']} JSON key count",
                allow_zero=True,
            )
        elif suffix == "safetensors":
            _positive_int(
                item["tensorCount"], f"adapter file {item['path']} tensor count"
            )
    weights = next(
        item
        for item in expected_file_list
        if item["path"] == "adapter_model.safetensors"
    )
    _positive_int(weights["bytes"], "adapter weights bytes")


def validate_successful_report(
    report: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    expected_source_revision: str,
    expected_source_bundle: Mapping[str, Any],
    expected_supervisor_run_id: str,
    expected_gpu_uuid: str,
    expected_worker_source_sha256: str,
    expected_adapter_aggregate_sha256: str,
    expected_adapter_files: Sequence[Mapping[str, Any]],
    expected_run_kind: str,
) -> ValidatedTrainingReport:
    """Validate one successful child report against supervisor-observed evidence.

    The function is intentionally strict: unknown fields, type coercion, failure
    reports, partial output, smoke/full confusion, and every promotion boundary
    are rejected.  The caller must pass independently obtained adapter evidence.
    """

    candidate_dict = _validate_candidate(candidate)
    report_dict = _mapping(report, "training report")
    _exact_keys(report_dict, TOP_LEVEL_KEYS, "training report")
    if (
        not isinstance(expected_source_revision, str)
        or HEX_40.fullmatch(expected_source_revision) is None
    ):
        _fail("expected source revision must be exact lowercase Git SHA")
    if (
        not isinstance(expected_supervisor_run_id, str)
        or HEX_32.fullmatch(expected_supervisor_run_id) is None
    ):
        _fail("expected supervisor run ID must be exactly 32 lowercase hex characters")
    if not isinstance(expected_run_kind, str):
        _fail("expected run kind must be smoke/SMOKE or full/FULL")
    run_kind = expected_run_kind.upper()
    if run_kind not in SUCCESS_STATE_BY_KIND or expected_run_kind not in {
        run_kind,
        run_kind.lower(),
    }:
        _fail("expected run kind must be smoke/SMOKE or full/FULL")

    _exact(report_dict["schema"], REPORT_SCHEMA, "training report schema")
    _exact(report_dict["candidateId"], candidate_dict["candidate_id"], "candidate ID")
    _exact(
        report_dict["supervisorRunId"], expected_supervisor_run_id, "supervisor run ID"
    )
    _exact(report_dict["runKind"], run_kind, "run kind")
    _exact(
        report_dict["state"], SUCCESS_STATE_BY_KIND[run_kind], "child terminal state"
    )
    _exact(report_dict["hostClass"], "LOCAL_GPU_RUNNER_REDACTED", "host class")
    _validate_timestamp(report_dict["measuredAt"])

    policy_sha = sha256_json(candidate_dict["supervision_policy"])
    recipe_sha = sha256_json(candidate_dict["training_recipe"])
    _exact(
        report_dict["supervisionPolicySha256"], policy_sha, "supervision policy digest"
    )
    _exact(report_dict["trainingRecipeSha256"], recipe_sha, "training recipe digest")
    _exact(
        report_dict["workerSourceSha256"],
        _sha256(expected_worker_source_sha256, "expected worker source digest"),
        "worker source digest",
    )

    expected_source = {
        "repository": "szl-holdings/szl-forge",
        "revision": expected_source_revision,
        "branch": "main",
        "originIdentityVerified": True,
        "freshRemoteMainObserved": False,
        "freshRemoteMainObservationDelegatedToSupervisor": True,
        "cachedRemoteTrackingMatches": True,
        "workingTreeClean": True,
        "commitSignatureVerifiedByThisTool": False,
    }
    _exact(report_dict["source"], expected_source, "worker source identity")
    source_bundle = _validate_source_bundle(
        report_dict["sourceBundle"], expected_source_bundle, candidate_dict
    )
    _exact(
        report_dict["implementation"],
        candidate_dict["actual_training_base"],
        "training implementation",
    )
    _exact(
        report_dict["runtimePackages"], candidate_dict["runtime_lock"], "runtime lock"
    )

    recipe = _mapping(candidate_dict["training_recipe"], "candidate training recipe")
    training_data = _mapping(candidate_dict["training_data"], "candidate training data")
    expected_steps = (
        recipe["smoke_optimizer_steps"]
        if run_kind == "SMOKE"
        else recipe["full_optimizer_steps"]
    )
    expected_examples = expected_steps * recipe["gradient_accumulation_steps"]
    if run_kind == "FULL":
        _exact(
            expected_examples,
            recipe["full_scheduled_examples"],
            "full scheduled examples",
        )
        _exact(
            expected_examples,
            training_data["train_rows"] * recipe["full_epochs_over_unique_rows"],
            "full epochs over unique rows",
        )
    _exact(
        report_dict["uniqueTrainingRows"], training_data["train_rows"], "training rows"
    )
    _exact(
        report_dict["uniqueTrainingRows"],
        source_bundle["uniqueTrainingRows"],
        "report/source-bundle training rows",
    )
    _exact(report_dict["optimizerSteps"], expected_steps, "optimizer steps")
    _exact(report_dict["scheduledExamples"], expected_examples, "scheduled examples")
    _validate_configuration(report_dict["configuration"], recipe, run_kind)
    _validate_gpu(report_dict["gpu"], recipe, expected_gpu_uuid)
    _validate_training(report_dict["training"])
    _validate_adapter(
        report_dict["adapter"],
        expected_adapter_aggregate_sha256,
        expected_adapter_files,
    )

    is_full = run_kind == "FULL"
    _exact(
        report_dict["integrityDigestIsAuthentication"],
        False,
        "digest/authentication boundary",
    )
    _exact(
        report_dict["authenticatedTrainingEnvelopePresent"],
        False,
        "training authentication boundary",
    )
    _exact(report_dict["qualificationEligible"], is_full, "qualification eligibility")
    _exact(report_dict["receiptEligible"], False, "receipt eligibility")
    _exact(report_dict["publicationEligible"], False, "publication eligibility")
    _exact(report_dict["autonomyEligible"], False, "autonomy eligibility")
    _exact(report_dict["claimBoundary"], CLAIM_BOUNDARY, "claim boundary")

    report_sha = _sha256(report_dict["reportSha256"], "training report self-digest")
    unsigned = dict(report_dict)
    unsigned.pop("reportSha256")
    if sha256_json(unsigned) != report_sha:
        _fail("training report self-digest is invalid")

    return ValidatedTrainingReport(
        observation_state=OBSERVATION_STATE_BY_KIND[run_kind],
        run_kind=run_kind,
        candidate_id=candidate_dict["candidate_id"],
        report_sha256=report_sha,
        source_revision=expected_source_revision,
        source_bundle_sha256=source_bundle["bundleSha256"],
        supervisor_run_id=expected_supervisor_run_id,
        gpu_uuid=expected_gpu_uuid,
        worker_source_sha256=expected_worker_source_sha256,
        training_recipe_sha256=recipe_sha,
        supervision_policy_sha256=policy_sha,
        adapter_aggregate_sha256=expected_adapter_aggregate_sha256,
        local_evaluation_input_binding_satisfied=is_full,
    )
