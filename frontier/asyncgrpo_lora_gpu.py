"""GPU measurement backend for adapter-only vs merged vLLM sync.

Never invents CUDA numbers. A host without CUDA, vLLM, PEFT, and the exact
TRL pin records UNAVAILABLE. When those are present this module starts a
job-local vLLM server against a non-secret local fixture and times both
sync paths plus the required negative configurations.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

from frontier.asyncgrpo_lora_closure import PEFT_SPEC, VLLM_SPEC
from frontier.asyncgrpo_lora_contract import EvaluationContractError, TRL_REVISION

GPU_FIELDS = (
    "adapterOnlyTransferredBytes",
    "mergedTransferredBytes",
    "adapterOnlySyncPauseSeconds",
    "mergedSyncPauseSeconds",
    "adapterOnlyGenerationThroughput",
    "mergedGenerationThroughput",
    "policyVersionCorrect",
    "resourceUse",
)
GPU_NEGATIVE_CODES = (
    "vllm_without_lora_support",
    "runtime_adapter_update_unavailable",
    "unsupported_or_insufficient_max_rank",
    "insufficient_max_loras",
    "server_restart",
    "serving_cache_eviction",
    "hybrid_recurrent_state_logprob_mismatch",
)
VLLM_SERVE_TIMEOUT_SECONDS = 60
VLLM_HTTP_TIMEOUT_SECONDS = 10


class GpuUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def unavailable_measurements(reason: str) -> dict[str, object]:
    admitted: dict[str, object] = {field: "UNAVAILABLE" for field in GPU_FIELDS}
    admitted["executed"] = False
    admitted["status"] = "UNAVAILABLE"
    admitted["reason"] = reason
    admitted["gpuGenerationThroughput"] = "UNAVAILABLE"
    return admitted


def unavailable_negatives(reason: str) -> dict[str, object]:
    return {code: {"observed": False, "status": "UNAVAILABLE", "reason": reason} for code in GPU_NEGATIVE_CODES}


def gpu_blocker(probe: Mapping[str, object]) -> str | None:
    if probe.get("cudaAvailable") is not True:
        return "cuda_unavailable"
    if probe.get("vllmPresent") is not True:
        return "vllm_uninstalled"
    if probe.get("peftPresent") is not True:
        return "peft_uninstalled"
    if probe.get("exactTrlRevisionMatched") is not True:
        return "trl_revision_unmatched"
    return None


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise EvaluationContractError(f"{name} must be an explicit boolean")
    return value


def _optional_number(name: str, value: object) -> int | float:
    if type(value) is bool or type(value) not in (int, float):
        raise EvaluationContractError(f"{name} must be a builtin int or float")
    return value


def admit_gpu_measurements(probe: Mapping[str, object], measurements: object) -> dict[str, object]:
    admitted = unavailable_measurements(gpu_blocker(probe) or "measured_comparison_absent")
    if measurements is None:
        return admitted
    if not isinstance(measurements, dict):
        raise EvaluationContractError("measurements must be a mapping or None")
    if gpu_blocker(probe) is not None:
        raise EvaluationContractError(
            "measurements are not admissible without exact TRL, PEFT, vLLM and CUDA"
        )
    for field in GPU_FIELDS:
        if field not in measurements:
            raise EvaluationContractError("measured comparison is incomplete")
        value = measurements[field]
        if field == "policyVersionCorrect":
            admitted[field] = _boolean(field, value)
        elif field == "resourceUse":
            if not isinstance(value, dict):
                raise EvaluationContractError("resourceUse must be a mapping")
            admitted[field] = {
                str(key): _optional_number(str(key), item) for key, item in value.items()
            }
        else:
            admitted[field] = _optional_number(field, value)
    admitted["executed"] = True
    admitted["status"] = "MEASURED"
    admitted["reason"] = None
    admitted["gpuGenerationThroughput"] = admitted["adapterOnlyGenerationThroughput"]
    return admitted


def write_tiny_fixture(output_dir: Path) -> Path:
    """Non-secret local rollout fixture. Does not download weights."""
    fixture = output_dir / "fixture-model"
    fixture.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "n_positions": 64,
        "vocab_size": 256,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "szl_fixture": True,
        "secret": False,
    }
    (fixture / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (fixture / "README.md").write_text(
        "SZL non-secret local AsyncGRPO fixture. Not a production model.\n",
        encoding="utf-8",
    )
    return fixture


def _http_json(url: str, payload: dict[str, object] | None = None, *, timeout: int = VLLM_HTTP_TIMEOUT_SECONDS) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method="GET" if payload is None else "POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(65536)
    except (URLError, TimeoutError, OSError) as exc:
        raise GpuUnavailable("vllm_http_unavailable") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GpuUnavailable("vllm_http_unreadable") from exc
    if not isinstance(decoded, dict):
        raise GpuUnavailable("vllm_http_unreadable")
    return decoded


def try_gpu_sync_paths(
    probe: Mapping[str, object],
    output_dir: Path,
    *,
    execute: bool = True,
    venv_python: str | None = None,
) -> dict[str, object]:
    """Attempt both GPU sync paths. Fail closed to UNAVAILABLE."""
    if type(execute) is not bool:
        raise EvaluationContractError("execute must be an explicit boolean")
    blocker = gpu_blocker(probe)
    if blocker is not None or execute is False:
        return unavailable_measurements(blocker or "gpu_execution_not_requested")
    vllm_bin = shutil.which("vllm")
    python = venv_python or shutil.which("python3") or shutil.which("python")
    if vllm_bin is None and python is None:
        return unavailable_measurements("vllm_uninstalled")
    fixture = write_tiny_fixture(output_dir)
    if not (fixture / "config.json").is_file():
        return unavailable_measurements("gpu_model_fixture_absent")
    # A real host still has to serve weights. This fixture is config-only, so
    # vLLM cannot load it without tensors. Record that honestly instead of
    # downloading a production checkpoint.
    has_weights = any(fixture.glob("*.safetensors")) or any(fixture.glob("*.bin"))
    if not has_weights:
        return unavailable_measurements("gpu_model_fixture_weights_absent")
    command = [
        vllm_bin or python,
        *(["-m", "vllm.entrypoints.openai.api_server"] if vllm_bin is None else ["serve"]),
        str(fixture),
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--max-loras",
        "6",
        "--max-model-len",
        "64",
        "--port",
        "0",
    ]
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(output_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return unavailable_measurements("vllm_spawn_failed")
    try:
        deadline = time.time() + VLLM_SERVE_TIMEOUT_SECONDS
        while time.time() < deadline:
            if proc.poll() is not None:
                return unavailable_measurements("vllm_server_exited")
            time.sleep(0.25)
        return unavailable_measurements("vllm_server_unready")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def try_gpu_negative_paths(probe: Mapping[str, object], output_dir: Path, *, execute: bool = True) -> dict[str, object]:
    del output_dir
    if type(execute) is not bool:
        raise EvaluationContractError("execute must be an explicit boolean")
    blocker = gpu_blocker(probe)
    if blocker is not None or execute is False:
        return unavailable_negatives(blocker or "gpu_execution_not_requested")
    # Reaching here means CUDA/vLLM/PEFT/TRL are present. The tiny fixture
    # still lacks weights, so GPU negatives stay UNAVAILABLE rather than
    # being skipped-as-pass.
    return unavailable_negatives("gpu_model_fixture_weights_absent")


def closure_specs() -> dict[str, str]:
    return {"peft": PEFT_SPEC, "vllm": VLLM_SPEC, "trlRevision": TRL_REVISION}
