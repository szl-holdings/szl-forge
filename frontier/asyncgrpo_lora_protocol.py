"""In-process AsyncGRPO LoRA / vLLM serving protocol.

This is not a GPU vLLM process. It executes the admitted serving contract
from huggingface/trl@f540773f5250c816e992ae3d35a41142ef3625c0 against real
files: adapter-only vs merged transfer, load-before-evict, path isolation,
partial writes, restart, and cache eviction.

GPU generation throughput is not produced here. Callers must keep that field
UNAVAILABLE unless a separate CUDA/vLLM host records it.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from pathlib import Path

from frontier.asyncgrpo_lora_contract import (
    ALLOWED_MAX_LORA_RANKS,
    EvaluationContractError,
    required_max_loras,
    serving_cache_path,
)

MERGED_BASE_BYTES = 4096
ADAPTER_BYTE_UNIT = 256
ADAPTER_HEADER = b"SZL-LORA-V1\n"
MERGED_HEADER = b"SZL-MERGED-V1\n"


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def adapter_payload_bytes(rank: int, name: str, version: int) -> bytes:
    body_len = rank * ADAPTER_BYTE_UNIT
    digest = hashlib.sha256(f"{name}:{version}:{rank}".encode("utf-8")).digest()
    body = (digest * ((body_len // len(digest)) + 1))[:body_len]
    return ADAPTER_HEADER + body


def merged_payload_bytes(rank: int, name: str, version: int) -> bytes:
    return MERGED_HEADER + (b"\x00" * MERGED_BASE_BYTES) + adapter_payload_bytes(rank, name, version)


def validate_cache_at_use(output_dir: Path, candidate: Path) -> Path:
    """Re-check isolation at use time. Point-in-time resolve is not enough."""
    if not isinstance(output_dir, Path) or not isinstance(candidate, Path):
        raise EvaluationContractError("cache paths must be pathlib.Path objects")
    cursor = candidate
    for _ in range(256):
        if cursor.is_symlink():
            raise ProtocolError("path_escape_symlink", "serving cache rejects symlinks at use time")
        if cursor == output_dir:
            break
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    try:
        resolved = serving_cache_path(output_dir, candidate)
    except EvaluationContractError as exc:
        raise ProtocolError("path_escape_symlink", "serving cache cannot resolve inside output directory") from exc
    if candidate.exists() and candidate.is_symlink():
        raise ProtocolError("path_escape_symlink", "serving cache rejects symlinks at use time")
    return resolved


class ProtocolServer:
    """Bounded replica of vLLM LoRA runtime-update semantics."""

    def __init__(
        self,
        output_dir: Path,
        *,
        enable_lora: bool,
        max_lora_rank: int,
        max_loras: int,
        max_staleness: int,
        runtime_updates: bool,
        checkpoints_enabled: bool,
    ) -> None:
        if type(enable_lora) is not bool or type(runtime_updates) is not bool or type(checkpoints_enabled) is not bool:
            raise EvaluationContractError("protocol flags must be explicit booleans")
        if max_lora_rank not in ALLOWED_MAX_LORA_RANKS:
            raise EvaluationContractError("max_lora_rank is not an upstream-supported capacity")
        self.output_dir = output_dir.resolve()
        self.cache_root = self.output_dir / ".vllm_lora"
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self.max_loras = max_loras
        self.max_staleness = max_staleness
        self.runtime_updates = runtime_updates
        self.checkpoints_enabled = checkpoints_enabled
        self.loaded: OrderedDict[str, dict[str, object]] = OrderedDict()
        self.in_flight: set[str] = set()
        self.paused = False
        self.alive = True
        self.evicted: list[str] = []
        self.generation_logprobs: dict[str, str] = {}
        self.merged_slot: dict[str, object] | None = None

    def ensure_layout(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if self.checkpoints_enabled:
            self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    def cache_path(self, candidate: Path) -> Path:
        return validate_cache_at_use(self.output_dir, candidate)

    def write_adapter(self, name: str, rank: int, version: int, *, complete: bool = True) -> Path:
        self.ensure_layout()
        path = serving_cache_path(self.output_dir, self.cache_root / f"{name}-v{version}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = adapter_payload_bytes(rank, name, version)
        if not complete:
            payload = payload[: len(ADAPTER_HEADER)]
        path.write_bytes(payload)
        return path

    def write_merged(self, name: str, rank: int, version: int) -> Path:
        self.ensure_layout()
        path = serving_cache_path(self.output_dir, self.cache_root / f"{name}-merged-v{version}")
        path.write_bytes(merged_payload_bytes(rank, name, version))
        return path

    def write_checkpoint(self, name: str, rank: int, version: int) -> Path | None:
        if not self.checkpoints_enabled:
            return None
        self.ensure_layout()
        path = self.checkpoint_root / f"{name}-v{version}.ckpt"
        path.write_bytes(adapter_payload_bytes(rank, name, version))
        return path

    def _require_alive(self) -> None:
        if not self.alive:
            raise ProtocolError("server_restarted", "server is not running")

    def pause(self) -> None:
        self._require_alive()
        self.paused = True

    def resume(self) -> None:
        self._require_alive()
        self.paused = False

    def load_adapter(self, name: str, path: Path, *, rank: int, version: int) -> dict[str, object]:
        self._require_alive()
        if not self.enable_lora:
            raise ProtocolError("vllm_without_lora_support", "server started without --enable-lora")
        if not self.runtime_updates:
            raise ProtocolError("runtime_adapter_update_unavailable", "VLLM_ALLOW_RUNTIME_LORA_UPDATING is not set")
        if rank > self.max_lora_rank:
            raise ProtocolError("unsupported_or_insufficient_max_rank", "adapter rank exceeds --max-lora-rank")
        resolved = self.cache_path(path)
        if not resolved.is_file() or resolved.is_symlink():
            raise ProtocolError("load_failure", "adapter path is missing")
        payload = resolved.read_bytes()
        if not payload.startswith(ADAPTER_HEADER) or len(payload) <= len(ADAPTER_HEADER):
            raise ProtocolError("partially_written_adapter", "adapter is incomplete")
        expected = required_max_loras(self.max_staleness)
        if self.max_loras < expected:
            raise ProtocolError("insufficient_max_loras", "max_loras is below max_staleness+2")
        while len(self.loaded) >= self.max_loras:
            victim = next(iter(self.loaded))
            if victim in self.in_flight:
                raise ProtocolError("load_before_evict_ordering", "cannot evict an in-flight policy")
            self.unload_adapter(victim)
        slot = {
            "name": name,
            "path": str(resolved),
            "rank": rank,
            "version": version,
            "bytes": len(payload),
            "logprobSeed": hashlib.sha256(payload).hexdigest(),
            "mode": "adapter-only",
        }
        self.loaded[name] = slot
        self.loaded.move_to_end(name)
        self.generation_logprobs[name] = str(slot["logprobSeed"])
        self.merged_slot = None
        return dict(slot)

    def load_merged(self, name: str, path: Path, *, rank: int, version: int) -> dict[str, object]:
        self._require_alive()
        resolved = self.cache_path(path)
        if not resolved.is_file() or resolved.is_symlink():
            raise ProtocolError("load_failure", "merged path is missing")
        payload = resolved.read_bytes()
        if not payload.startswith(MERGED_HEADER) or len(payload) <= len(MERGED_HEADER):
            raise ProtocolError("load_failure", "merged payload is incomplete")
        slot = {
            "name": name,
            "path": str(resolved),
            "rank": rank,
            "version": version,
            "bytes": len(payload),
            "logprobSeed": hashlib.sha256(payload).hexdigest(),
            "mode": "merged",
        }
        self.merged_slot = slot
        return dict(slot)

    def unload_adapter(self, name: str) -> None:
        self._require_alive()
        if name in self.in_flight:
            raise ProtocolError("load_before_evict_ordering", "cannot evict an in-flight policy")
        if name in self.loaded:
            del self.loaded[name]
            self.evicted.append(name)

    def generate(self, name: str, tokens: int = 8) -> dict[str, object]:
        self._require_alive()
        if self.paused:
            raise ProtocolError("server_paused", "generation refused while paused")
        if name not in self.loaded:
            raise ProtocolError("adapter_not_loaded", "requested policy is not loaded")
        self.in_flight.add(name)
        started = time.perf_counter()
        seed = str(self.loaded[name]["logprobSeed"])
        stored = self.generation_logprobs.get(name)
        if stored is not None and stored != seed:
            self.in_flight.discard(name)
            raise ProtocolError(
                "hybrid_recurrent_state_logprob_mismatch",
                "recurrent-state/log-prob seed drifted",
            )
        acc = seed.encode("utf-8")
        for index in range(tokens):
            acc = hashlib.sha256(acc + index.to_bytes(2, "big")).digest()
        elapsed = time.perf_counter() - started
        self.in_flight.discard(name)
        return {
            "name": name,
            "tokens": tokens,
            "seconds": elapsed,
            "digest": acc.hex(),
            "policyVersion": int(self.loaded[name]["version"]),
            "mode": "adapter-only",
        }

    def generate_merged(self, tokens: int = 8) -> dict[str, object]:
        self._require_alive()
        if self.paused:
            raise ProtocolError("server_paused", "generation refused while paused")
        if self.merged_slot is None:
            raise ProtocolError("adapter_not_loaded", "merged weights are not loaded")
        started = time.perf_counter()
        seed = str(self.merged_slot["logprobSeed"])
        acc = seed.encode("utf-8")
        for index in range(tokens):
            acc = hashlib.sha256(acc + index.to_bytes(2, "big")).digest()
        elapsed = time.perf_counter() - started
        return {
            "name": self.merged_slot["name"],
            "tokens": tokens,
            "seconds": elapsed,
            "digest": acc.hex(),
            "policyVersion": int(self.merged_slot["version"]),
            "mode": "merged",
        }

    def restart(self) -> None:
        self.loaded.clear()
        self.in_flight.clear()
        self.generation_logprobs.clear()
        self.merged_slot = None
        self.paused = False
        self.alive = True

    def evict_cache_files(self) -> None:
        if self.cache_root.exists():
            for child in self.cache_root.iterdir():
                if child.is_file() and not child.is_symlink():
                    child.unlink()


def measure_sync_paths(server: ProtocolServer, *, name: str, rank: int, version: int) -> dict[str, object]:
    """Measure adapter-only vs merged transfer under equivalent local conditions."""
    adapter_path = server.write_adapter(name, rank, version)
    merged_path = server.write_merged(name, rank, version)
    checkpoint = server.write_checkpoint(name, rank, version)

    server.pause()
    adapter_started = time.perf_counter()
    adapter_slot = server.load_adapter(name, adapter_path, rank=rank, version=version)
    adapter_pause = time.perf_counter() - adapter_started
    server.resume()
    adapter_gen = server.generate(name, tokens=8)

    server.unload_adapter(name)
    server.pause()
    merged_started = time.perf_counter()
    merged_slot = server.load_merged(name, merged_path, rank=rank, version=version)
    merged_pause = time.perf_counter() - merged_started
    server.resume()
    merged_gen = server.generate_merged(tokens=8)

    adapter_bytes = int(adapter_slot["bytes"])
    merged_bytes = int(merged_slot["bytes"])
    adapter_tps = float(adapter_gen["tokens"]) / float(adapter_gen["seconds"]) if adapter_gen["seconds"] else 0.0
    merged_tps = float(merged_gen["tokens"]) / float(merged_gen["seconds"]) if merged_gen["seconds"] else 0.0
    return {
        "adapterOnlyTransferredBytes": adapter_bytes,
        "mergedTransferredBytes": merged_bytes,
        "adapterOnlySyncPauseSeconds": float(adapter_pause),
        "mergedSyncPauseSeconds": float(merged_pause),
        "fixtureAdapterTokensPerSecond": adapter_tps,
        "fixtureMergedTokensPerSecond": merged_tps,
        "policyVersionCorrect": adapter_gen["policyVersion"] == version and merged_gen["policyVersion"] == version,
        "checkpointPresent": checkpoint is not None and checkpoint.is_file(),
        "loadedAdapterIdentities": list(server.loaded),
        "evictedAdapterIdentities": list(server.evicted),
        "adapterOnlySmallerThanMerged": adapter_bytes < merged_bytes,
        "equivalentLocalConditions": True,
        "gpuGenerationThroughput": "UNAVAILABLE",
        "gpuSyncExecuted": False,
        "resourceUse": {
            "adapterBytes": adapter_bytes,
            "mergedBytes": merged_bytes,
        },
    }


def run_negative_paths(output_dir: Path) -> dict[str, bool]:
    """Execute every required negative path against real files. No GPU."""
    observed: dict[str, bool] = {}

    def _saw(code: str, fn: object) -> None:
        try:
            fn()  # type: ignore[operator]
            observed[code] = False
        except ProtocolError as exc:
            observed[code] = exc.code == code
        except EvaluationContractError:
            observed[code] = code in {"path_escape_symlink", "shared_path_unavailable"}

    bare = ProtocolServer(
        output_dir / "neg-lora-off",
        enable_lora=False,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    path = bare.write_adapter("policy", 8, 1)

    def _no_lora() -> None:
        bare.load_adapter("policy", path, rank=8, version=1)

    _saw("vllm_without_lora_support", _no_lora)

    frozen = ProtocolServer(
        output_dir / "neg-no-update",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=False,
        checkpoints_enabled=True,
    )
    frozen_path = frozen.write_adapter("policy", 8, 1)

    def _no_update() -> None:
        frozen.load_adapter("policy", frozen_path, rank=8, version=1)

    _saw("runtime_adapter_update_unavailable", _no_update)

    rank_limited = ProtocolServer(
        output_dir / "neg-rank",
        enable_lora=True,
        max_lora_rank=8,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    rank_path = rank_limited.write_adapter("policy", 8, 1)

    def _rank() -> None:
        rank_limited.load_adapter("policy", rank_path, rank=32, version=1)

    _saw("unsupported_or_insufficient_max_rank", _rank)

    small = ProtocolServer(
        output_dir / "neg-capacity",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=2,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    small_path = small.write_adapter("policy", 8, 1)

    def _capacity() -> None:
        small.load_adapter("policy", small_path, rank=8, version=1)

    _saw("insufficient_max_loras", _capacity)

    missing_root = output_dir / "neg-missing" / "gone"
    missing = ProtocolServer(
        output_dir / "neg-missing",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )

    def _missing() -> None:
        missing.load_adapter("policy", missing_root / "adapter", rank=8, version=1)

    try:
        _missing()
        observed["shared_path_unavailable"] = False
    except (ProtocolError, EvaluationContractError):
        observed["shared_path_unavailable"] = True

    escape = ProtocolServer(
        output_dir / "neg-escape",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    escape.ensure_layout()
    outside = output_dir / "outside-adapter"
    outside.write_bytes(adapter_payload_bytes(8, "escaped", 1))
    link = escape.cache_root / "escaped"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(outside)

    def _escape() -> None:
        escape.load_adapter("escaped", link, rank=8, version=1)

    try:
        _escape()
        observed["path_escape_symlink"] = False
    except (ProtocolError, EvaluationContractError):
        observed["path_escape_symlink"] = True

    partial = ProtocolServer(
        output_dir / "neg-partial",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    partial_path = partial.write_adapter("policy", 8, 1, complete=False)

    def _partial() -> None:
        partial.load_adapter("policy", partial_path, rank=8, version=1)

    _saw("partially_written_adapter", _partial)

    load_fail = ProtocolServer(
        output_dir / "neg-load",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    load_fail.ensure_layout()
    bogus = load_fail.cache_root / "empty"
    bogus.write_bytes(b"not-an-adapter")

    def _load_fail() -> None:
        load_fail.load_adapter("policy", bogus, rank=8, version=1)

    try:
        _load_fail()
        observed["load_failure"] = False
    except ProtocolError as exc:
        observed["load_failure"] = exc.code in {"load_failure", "partially_written_adapter"}

    inflight = ProtocolServer(
        output_dir / "neg-inflight",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=2,
        max_staleness=0,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    a = inflight.write_adapter("p1", 8, 1)
    b = inflight.write_adapter("p2", 8, 2)
    inflight.load_adapter("p1", a, rank=8, version=1)
    inflight.load_adapter("p2", b, rank=8, version=2)
    inflight.in_flight.add("p1")
    inflight.in_flight.add("p2")
    c = inflight.write_adapter("p3", 8, 3)

    def _inflight() -> None:
        inflight.load_adapter("p3", c, rank=8, version=3)

    _saw("load_before_evict_ordering", _inflight)

    restart = ProtocolServer(
        output_dir / "neg-restart",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    rpath = restart.write_adapter("policy", 8, 1)
    restart.load_adapter("policy", rpath, rank=8, version=1)
    restart.restart()
    try:
        restart.generate("policy")
        observed["server_restart"] = False
    except ProtocolError as exc:
        observed["server_restart"] = exc.code == "adapter_not_loaded"

    eviction = ProtocolServer(
        output_dir / "neg-evict",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    epath = eviction.write_adapter("policy", 8, 1)
    eviction.load_adapter("policy", epath, rank=8, version=1)
    eviction.evict_cache_files()
    eviction.restart()
    try:
        eviction.load_adapter("policy", epath, rank=8, version=1)
        observed["serving_cache_eviction"] = False
    except ProtocolError:
        observed["serving_cache_eviction"] = True

    nocheck = ProtocolServer(
        output_dir / "neg-ckpt",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=False,
    )
    observed["checkpoint_missing"] = nocheck.write_checkpoint("policy", 8, 1) is None

    hybrid = ProtocolServer(
        output_dir / "neg-hybrid",
        enable_lora=True,
        max_lora_rank=32,
        max_loras=6,
        max_staleness=4,
        runtime_updates=True,
        checkpoints_enabled=True,
    )
    hpath = hybrid.write_adapter("policy", 8, 1)
    hybrid.load_adapter("policy", hpath, rank=8, version=1)
    hybrid.generation_logprobs["policy"] = "tampered"
    try:
        hybrid.generate("policy")
        observed["hybrid_recurrent_state_logprob_mismatch"] = False
    except ProtocolError as exc:
        observed["hybrid_recurrent_state_logprob_mismatch"] = (
            exc.code == "hybrid_recurrent_state_logprob_mismatch"
        )
    ckpt = hybrid.write_checkpoint("policy", 8, 1)
    cache_file = hybrid.cache_root / "policy-v1"
    observed["serving_cache_ephemeral_not_durable_artifact"] = (
        ckpt is not None and cache_file.exists() and ckpt.exists() and cache_file.resolve() != ckpt.resolve()
    )
    return observed
