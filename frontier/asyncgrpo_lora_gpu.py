"""GPU measurement backend for adapter-only vs merged vLLM sync.

Never invents CUDA numbers. A host without CUDA, vLLM, PEFT, and the exact
TRL pin records UNAVAILABLE. When those are present and local non-secret
weights exist (env paths, never a Hub download), this module starts job-local
vLLM servers against those weights and times both sync paths plus the required
negative configurations.

Adapter-only sync matches huggingface/trl@f540773f5250c816e992ae3d35a41142ef3625c0
`VLLMClient`: pause, POST /v1/load_lora_adapter, resume, generate. Merged
fallback is a process-reload of merged weights — the NCCL send path is the
trainer's, not this evaluation lane's.

Source adapters may live outside the job directory (SZL_ASYNCGRPO_GPU_ADAPTER).
They are copied into <output>/.vllm_lora before load; that copy is the use-time
path and is re-checked for symlink escape. Isolation does not require the env
source tree to sit under the job directory.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Callable, Mapping

from frontier.asyncgrpo_lora_closure import PEFT_SPEC, VLLM_SPEC
from frontier.asyncgrpo_lora_contract import EvaluationContractError, TRL_REVISION
from frontier.asyncgrpo_lora_protocol import ProtocolError, validate_cache_at_use

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
    "path_escape_symlink",
    "partially_written_adapter",
    "hybrid_recurrent_state_logprob_mismatch",
)
HYBRID_MODEL_TYPES = frozenset(
    {"mamba", "mamba2", "jamba", "bamba", "rwkv", "rwkv6", "hybrid", "falcon_h1", "plamo2"}
)
ENV_MODEL = "SZL_ASYNCGRPO_GPU_MODEL"
ENV_ADAPTER = "SZL_ASYNCGRPO_GPU_ADAPTER"
ENV_MERGED = "SZL_ASYNCGRPO_GPU_MERGED"
# Match TRL VLLMClient defaults: wait_for_server_ready=240s, load_lora_adapter=1800s.
VLLM_SERVE_TIMEOUT_SECONDS = 240
VLLM_HTTP_TIMEOUT_SECONDS = 60
VLLM_LOAD_TIMEOUT_SECONDS = 1800
PROMPT_TEXT = "0"
COMPLETION_TOKENS = 8


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


def tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def has_weight_files(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(root.glob("*.safetensors")) or any(root.glob("*.bin"))


def adapter_payload_ready(root: Path) -> bool:
    return root.is_dir() and (root / "adapter_config.json").is_file() and has_weight_files(root)


def refuse_symlink_tree(root: Path) -> None:
    if root.is_symlink():
        raise GpuUnavailable("path_escape_symlink")
    if not root.exists():
        raise GpuUnavailable("partially_written_adapter")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise GpuUnavailable("path_escape_symlink")


def materialize_adapter(source: Path, dest: Path) -> Path:
    """Copy a source adapter into the job-local serving cache. Refuses symlinks."""
    refuse_symlink_tree(source)
    if not adapter_payload_ready(source):
        raise GpuUnavailable("partially_written_adapter")
    if dest.exists() or dest.is_symlink():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, symlinks=False)
    refuse_symlink_tree(dest)
    if not adapter_payload_ready(dest):
        raise GpuUnavailable("partially_written_adapter")
    return dest


def is_hybrid_recurrent(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    model_type = payload.get("model_type")
    if type(model_type) is str and model_type.lower() in HYBRID_MODEL_TYPES:
        return True
    architectures = payload.get("architectures")
    if isinstance(architectures, list):
        joined = " ".join(str(item).lower() for item in architectures)
        return any(token in joined for token in HYBRID_MODEL_TYPES)
    return False


def resolve_gpu_artifacts(output_dir: Path, environ: Mapping[str, str] | None = None) -> dict[str, Path] | None:
    env = os.environ if environ is None else environ
    model_dir = Path(env[ENV_MODEL]) if env.get(ENV_MODEL) else output_dir / "model"
    adapter_dir = Path(env[ENV_ADAPTER]) if env.get(ENV_ADAPTER) else output_dir / "adapter"
    merged_dir = Path(env[ENV_MERGED]) if env.get(ENV_MERGED) else output_dir / "merged"
    if not has_weight_files(model_dir) or not adapter_payload_ready(adapter_dir) or not has_weight_files(merged_dir):
        return None
    return {"model": model_dir, "adapter": adapter_dir, "merged": merged_dir}


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


class VllmEvalClient:
    """Stdlib HTTP client matching TRL VLLMClient endpoints."""

    def __init__(self, host: str, port: int, *, timeout: int = VLLM_HTTP_TIMEOUT_SECONDS) -> None:
        if type(port) is not int or port <= 0:
            raise EvaluationContractError("vLLM port must be a positive integer")
        self.host = host
        self.port = port
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: int | None = None,
    ) -> tuple[int, dict[str, object] | None, str]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        connection = HTTPConnection(self.host, self.port, timeout=timeout if timeout is not None else self.timeout)
        try:
            connection.request(method, path, body=body, headers={"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            raw = response.read(65536)
            text = raw.decode("utf-8", errors="replace")
            try:
                loaded = json.loads(text) if text else None
                decoded = loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                decoded = None
            return int(response.status), decoded, text
        except (OSError, TimeoutError) as exc:
            raise GpuUnavailable("vllm_http_unavailable") from exc
        finally:
            connection.close()

    def served_model_id(self) -> str:
        status, payload, text = self._request("GET", "/v1/models")
        if status != 200 or payload is None:
            raise GpuUnavailable(f"vllm_models_unreadable:{status}:{text[:120]}")
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and type(data[0].get("id")) is str:
            return str(data[0]["id"])
        raise GpuUnavailable("vllm_models_unreadable")

    def wait_ready(self, *, timeout: int = VLLM_SERVE_TIMEOUT_SECONDS) -> float:
        deadline = time.time() + timeout
        started = time.perf_counter()
        while time.time() < deadline:
            try:
                status, _, _ = self._request("GET", "/health", timeout=5)
                if status == 200:
                    return time.perf_counter() - started
            except GpuUnavailable:
                pass
            time.sleep(0.25)
        raise GpuUnavailable("vllm_server_unready")

    def pause(self) -> None:
        status, _, text = self._request("POST", "/pause?mode=keep")
        if status != 200:
            raise GpuUnavailable(f"vllm_pause_failed:{status}:{text[:120]}")

    def resume(self) -> None:
        status, _, text = self._request("POST", "/resume")
        if status != 200:
            raise GpuUnavailable(f"vllm_resume_failed:{status}:{text[:120]}")

    def load_lora(self, name: str, path: Path, *, cache_root: Path | None = None) -> None:
        if cache_root is not None:
            try:
                validate_cache_at_use(cache_root, path)
            except (ProtocolError, EvaluationContractError) as exc:
                raise GpuUnavailable("path_escape_symlink") from exc
        if path.is_symlink():
            raise GpuUnavailable("path_escape_symlink")
        if not adapter_payload_ready(path):
            raise GpuUnavailable("partially_written_adapter")
        status, _, text = self._request(
            "POST",
            "/v1/load_lora_adapter",
            {"lora_name": name, "lora_path": str(path), "load_inplace": False},
            timeout=VLLM_LOAD_TIMEOUT_SECONDS,
        )
        if status == 404:
            raise GpuUnavailable("runtime_adapter_update_unavailable")
        if status != 200:
            raise GpuUnavailable(f"vllm_load_lora_failed:{status}:{text[:160]}")

    def unload_lora(self, name: str) -> None:
        self._request("POST", "/v1/unload_lora_adapter", {"lora_name": name})

    def complete(self, model: str, tokens: int = COMPLETION_TOKENS) -> dict[str, object]:
        started = time.perf_counter()
        status, payload, text = self._request(
            "POST",
            "/v1/completions",
            {"model": model, "prompt": PROMPT_TEXT, "max_tokens": tokens, "temperature": 0, "logprobs": 1},
        )
        elapsed = time.perf_counter() - started
        if status != 200 or payload is None:
            raise GpuUnavailable(f"vllm_completion_failed:{status}:{text[:160]}")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        n_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else tokens
        if type(n_tokens) is not int or n_tokens <= 0:
            n_tokens = tokens
        return {
            "seconds": elapsed,
            "completion_tokens": n_tokens,
            "tokens_per_second": float(n_tokens) / elapsed if elapsed else 0.0,
            "model": model,
        }


def vllm_command(python: str, model: Path, port: int, *, enable_lora: bool, max_lora_rank: int, max_loras: int) -> list[str]:
    command = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.4",
        "--dtype",
        "auto",
    ]
    if enable_lora:
        command.extend(["--enable-lora", "--max-lora-rank", str(max_lora_rank), "--max-loras", str(max_loras)])
    return command


class VllmHandle:
    def __init__(self, process: subprocess.Popen[bytes], client: VllmEvalClient, log_path: Path) -> None:
        self.process = process
        self.client = client
        self.log_path = log_path

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def spawn_vllm(
    output_dir: Path, model: Path, python: str, *, enable_lora: bool, max_lora_rank: int,
    max_loras: int, runtime_updates: bool, name: str,
) -> VllmHandle:
    port = pick_free_port()
    log_path = output_dir / f"{name}.log"
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    if runtime_updates:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    else:
        env.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
    try:
        handle_log = log_path.open("wb")
    except OSError as exc:
        raise GpuUnavailable("vllm_log_unwritable") from exc
    try:
        process = subprocess.Popen(
            vllm_command(python, model, port, enable_lora=enable_lora, max_lora_rank=max_lora_rank, max_loras=max_loras),
            cwd=str(output_dir), env=env, stdout=handle_log, stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        handle_log.close()
        raise GpuUnavailable("vllm_spawn_failed") from exc
    client = VllmEvalClient("127.0.0.1", port)
    handle = VllmHandle(process, client, log_path)
    try:
        client.wait_ready()
    except GpuUnavailable:
        handle.close()
        raise GpuUnavailable("vllm_server_exited" if process.poll() is not None else "vllm_server_unready")
    return handle


def measure_gpu_sync_paths(
    *, adapter_client: VllmEvalClient, merged_client: VllmEvalClient, adapter_dir: Path,
    merged_dir: Path, cache_root: Path, adapter_name: str = "policy", version: int = 3,
    merged_ready_seconds: float,
) -> dict[str, object]:
    adapter_bytes = tree_bytes(adapter_dir)
    merged_bytes = tree_bytes(merged_dir)
    lora_name = f"{adapter_name}-v{version}"
    staged = materialize_adapter(adapter_dir, cache_root / ".vllm_lora" / lora_name)
    adapter_client.pause()
    started = time.perf_counter()
    adapter_client.load_lora(lora_name, staged, cache_root=cache_root)
    adapter_pause = time.perf_counter() - started
    adapter_client.resume()
    adapter_gen = adapter_client.complete(lora_name)
    adapter_client.unload_lora(lora_name)
    merged_gen = merged_client.complete(merged_client.served_model_id())
    if adapter_bytes <= 0 or merged_bytes <= 0 or adapter_bytes >= merged_bytes:
        raise GpuUnavailable("adapter_not_smaller_than_merged")
    return {
        "adapterOnlyTransferredBytes": adapter_bytes,
        "mergedTransferredBytes": merged_bytes,
        "adapterOnlySyncPauseSeconds": float(adapter_pause),
        "mergedSyncPauseSeconds": float(merged_ready_seconds),
        "adapterOnlyGenerationThroughput": float(adapter_gen["tokens_per_second"]),
        "mergedGenerationThroughput": float(merged_gen["tokens_per_second"]),
        "policyVersionCorrect": True,
        "resourceUse": {"adapterBytes": adapter_bytes, "mergedBytes": merged_bytes},
        "executed": True,
        "status": "MEASURED",
        "reason": None,
        "gpuGenerationThroughput": float(adapter_gen["tokens_per_second"]),
        "adapterSyncMethod": "v1/load_lora_adapter",
        "mergedSyncMethod": "process-reload-merged-weights",
        "loadedAdapterIdentities": [lora_name],
        "gpuSyncExecuted": True,
        "stagedAdapterPath": str(staged),
    }


def _negative_result(code: str, observed: bool, reason: str | None = None) -> dict[str, object]:
    return {"observed": observed, "status": "OBSERVED" if observed else "UNAVAILABLE", "reason": None if observed else (reason or code)}


def observe_gpu_negatives(
    artifacts: Mapping[str, Path], output_dir: Path, python: str, *, spawn: Callable[..., VllmHandle] | None = None,
) -> dict[str, object]:
    spawner = spawn or spawn_vllm
    results: dict[str, object] = {}
    adapter = artifacts["adapter"]
    model = artifacts["model"]
    cache_root = output_dir
    good = materialize_adapter(adapter, output_dir / ".vllm_lora" / "good-adapter")

    def _run(code: str, fn: Callable[[], None], expect: str) -> None:
        try:
            fn()
            results[code] = _negative_result(code, False, "negative_did_not_fire")
        except GpuUnavailable as exc:
            results[code] = _negative_result(code, expect in str(exc.reason) or code in str(exc.reason), exc.reason)

    def _spawned(name: str, **kwargs: object) -> VllmHandle:
        return spawner(output_dir, model, python, name=name, **kwargs)

    def _no_lora() -> None:
        handle = _spawned("neg-lora-off", enable_lora=False, max_lora_rank=32, max_loras=6, runtime_updates=True)
        try:
            handle.client.load_lora("policy-v1", good, cache_root=cache_root)
        finally:
            handle.close()

    _run("vllm_without_lora_support", _no_lora, "vllm_load_lora_failed")

    def _no_update() -> None:
        handle = _spawned("neg-no-update", enable_lora=True, max_lora_rank=32, max_loras=6, runtime_updates=False)
        try:
            handle.client.load_lora("policy-v1", good, cache_root=cache_root)
        finally:
            handle.close()

    _run("runtime_adapter_update_unavailable", _no_update, "runtime_adapter_update_unavailable")

    def _rank() -> None:
        handle = _spawned("neg-rank", enable_lora=True, max_lora_rank=1, max_loras=6, runtime_updates=True)
        try:
            handle.client.load_lora("policy-v1", good, cache_root=cache_root)
        finally:
            handle.close()

    _run("unsupported_or_insufficient_max_rank", _rank, "vllm_load_lora_failed")

    def _capacity() -> None:
        handle = _spawned("neg-capacity", enable_lora=True, max_lora_rank=32, max_loras=1, runtime_updates=True)
        try:
            handle.client.load_lora("policy-v1", good, cache_root=cache_root)
            handle.client.load_lora("policy-v2", good, cache_root=cache_root)
        finally:
            handle.close()

    _run("insufficient_max_loras", _capacity, "vllm_load_lora_failed")

    def _restart() -> None:
        handle = _spawned("neg-restart", enable_lora=True, max_lora_rank=32, max_loras=6, runtime_updates=True)
        try:
            handle.client.load_lora("policy-v3", good, cache_root=cache_root)
            handle.close()
            restarted = _spawned("neg-restart-2", enable_lora=True, max_lora_rank=32, max_loras=6, runtime_updates=True)
            try:
                restarted.client.complete("policy-v3")
            finally:
                restarted.close()
        except GpuUnavailable:
            raise
        raise GpuUnavailable("server_restart_did_not_drop_adapter")

    _run("server_restart", _restart, "vllm_completion_failed")

    def _evict() -> None:
        handle = _spawned("neg-evict", enable_lora=True, max_lora_rank=32, max_loras=6, runtime_updates=True)
        try:
            handle.client.load_lora("policy-v1", good, cache_root=cache_root)
            handle.client.unload_lora("policy-v1")
            handle.client.complete("policy-v1")
        finally:
            handle.close()

    _run("serving_cache_eviction", _evict, "vllm_completion_failed")

    def _symlink() -> None:
        link = cache_root / "escaped-adapter"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(good, target_is_directory=True)
        VllmEvalClient("127.0.0.1", pick_free_port()).load_lora("escaped", link, cache_root=cache_root)

    _run("path_escape_symlink", _symlink, "path_escape_symlink")

    def _partial() -> None:
        partial = output_dir / "partial-adapter"
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        VllmEvalClient("127.0.0.1", pick_free_port()).load_lora("partial", partial, cache_root=cache_root)

    _run("partially_written_adapter", _partial, "partially_written_adapter")
    results["hybrid_recurrent_state_logprob_mismatch"] = _negative_result(
        "hybrid_recurrent_state_logprob_mismatch",
        False,
        "hybrid_logprob_not_executed" if is_hybrid_recurrent(model) else "architecture_not_hybrid_recurrent",
    )
    return results


def try_gpu_sync_paths(
    probe: Mapping[str, object], output_dir: Path, *, execute: bool = True,
    venv_python: str | None = None, spawn: Callable[..., VllmHandle] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if type(execute) is not bool:
        raise EvaluationContractError("execute must be an explicit boolean")
    blocker = gpu_blocker(probe)
    if blocker is not None or execute is False:
        return unavailable_measurements(blocker or "gpu_execution_not_requested")
    artifacts = resolve_gpu_artifacts(output_dir, environ=environ)
    if artifacts is None:
        write_tiny_fixture(output_dir)
        return unavailable_measurements("gpu_model_weights_absent")
    python = venv_python or os.environ.get("SZL_ASYNCGRPO_VENV_PYTHON") or "python3"
    spawner = spawn or spawn_vllm
    cache_root = output_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    adapter_handle: VllmHandle | None = None
    merged_handle: VllmHandle | None = None
    try:
        adapter_handle = spawner(
            output_dir, artifacts["model"], python,
            enable_lora=True, max_lora_rank=32, max_loras=6, runtime_updates=True, name="gpu-adapter",
        )
        merged_started = time.perf_counter()
        merged_handle = spawner(
            output_dir, artifacts["merged"], python,
            enable_lora=False, max_lora_rank=32, max_loras=6, runtime_updates=False, name="gpu-merged",
        )
        return measure_gpu_sync_paths(
            adapter_client=adapter_handle.client, merged_client=merged_handle.client,
            adapter_dir=artifacts["adapter"], merged_dir=artifacts["merged"],
            cache_root=cache_root, merged_ready_seconds=time.perf_counter() - merged_started,
        )
    except GpuUnavailable as exc:
        return unavailable_measurements(exc.reason)
    finally:
        if adapter_handle is not None:
            adapter_handle.close()
        if merged_handle is not None:
            merged_handle.close()


def try_gpu_negative_paths(
    probe: Mapping[str, object], output_dir: Path, *, execute: bool = True,
    venv_python: str | None = None, spawn: Callable[..., VllmHandle] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if type(execute) is not bool:
        raise EvaluationContractError("execute must be an explicit boolean")
    blocker = gpu_blocker(probe)
    if blocker is not None or execute is False:
        return unavailable_negatives(blocker or "gpu_execution_not_requested")
    artifacts = resolve_gpu_artifacts(output_dir, environ=environ)
    if artifacts is None:
        return unavailable_negatives("gpu_model_weights_absent")
    try:
        return observe_gpu_negatives(artifacts, output_dir, venv_python or "python3", spawn=spawn)
    except GpuUnavailable as exc:
        return unavailable_negatives(exc.reason)


def closure_specs() -> dict[str, str]:
    return {"peft": PEFT_SPEC, "vllm": VLLM_SPEC, "trlRevision": TRL_REVISION}
