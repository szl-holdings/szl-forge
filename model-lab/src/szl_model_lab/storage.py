# SPDX-License-Identifier: Apache-2.0
"""Storage-aware research primitives, not a production engine or model release.

Original SZL implementation informed by Edge0's published streaming design.
Unlike routing-replacement modes, hints here NEVER select executed experts.
Cache accounting covers serialized payloads only, not process/GPU/OS memory.
No network, model download, training, provider routing or publication occurs.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

from .safeio import read_regular, strict_json

GIB = 1 << 30
MAX_CHUNK = 64 << 20
MAX_EXPERTS = 4096
MAX_ROUTE = 64
SPARK_REPO = "XHToken/Spark-X2.5-4B"
# This is the existing SZL intake identity, not a newly attested download.
SPARK_INTAKE_REVISION = "5e10fcc0286756aebf7c41dc52c1e42d95c70281"
SPARK_INTAKE_PATH = "frontier/waves/2026-09-08-kimi-hy4-spark.json"


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _int(value: object, low: int, high: int, code: str) -> int:
    _need(type(value) is int and low <= value <= high, code)
    return value


def _sha(value: object, length: int = 64) -> str:
    _need(type(value) is str and re.fullmatch(r"[a-f0-9]{%d}" % length, value) is not None,
          "exact_content_identity_required")
    return value


def authority() -> dict[str, bool]:
    return dict.fromkeys(("training", "inference_dispatch", "publication", "promotion",
                          "provider_calls", "tool_execution"), False)


@dataclass(frozen=True)
class SparkShape:
    """Validated dense Spark dimensions; config inspection executes no custom code."""
    layer_types: tuple[str, ...]
    kv_heads: int
    head_dim: int
    window: int
    maximum_context: int

    def __post_init__(self) -> None:
        _need(type(self.layer_types) is tuple and 1 <= len(self.layer_types) <= 256,
              "layer_types_required")
        _need(all(type(t) is str and t in {"full_attention", "sliding_attention"}
                  for t in self.layer_types), "unsupported_attention")
        _int(self.kv_heads, 1, 256, "invalid_kv_heads")
        _int(self.head_dim, 1, 1024, "invalid_head_dim")
        _int(self.window, 1, 1 << 20, "invalid_window")
        _int(self.maximum_context, 1, 1 << 22, "invalid_context_cap")
        _need(self.window <= self.maximum_context, "window_exceeds_context")

    @classmethod
    def from_config(cls, raw: bytes, expected_sha256: str) -> "SparkShape":
        _sha(expected_sha256)
        _need(type(raw) is bytes and 0 < len(raw) <= 65536, "bounded_config_required")
        _need(hashlib.sha256(raw).hexdigest() == expected_sha256, "config_digest_mismatch")
        doc = strict_json(raw)
        pending = [doc]
        while pending:
            value = pending.pop()
            if type(value) is float:
                _need(math.isfinite(value), "nonfinite_config_number")
            elif type(value) is dict:
                pending.extend(value.values())
            elif type(value) is list:
                pending.extend(value)
        _need(type(doc) is dict and doc.get("model_type") == "spark2_5", "not_spark_config")
        _need(doc.get("architectures") == ["Spark2_5ForCausalLM"], "unsupported_spark_class")
        # Reject expert declarations instead of claiming a generic dense/MoE converter.
        _need(not any(k in doc for k in ("num_experts", "num_local_experts", "num_experts_per_tok")),
              "moe_config_not_supported_by_dense_planner")
        layers = doc.get("layer_types")
        _need(type(layers) is list, "layer_types_required")
        _need(_int(doc.get("num_hidden_layers"), 1, 256, "invalid_layer_count") == len(layers),
              "layer_count_mismatch")
        heads = _int(doc.get("num_attention_heads"), 1, 256, "invalid_query_heads")
        kv_heads = _int(doc.get("num_key_value_heads"), 1, 256, "invalid_kv_heads")
        _need(heads % kv_heads == 0, "invalid_grouped_query_heads")
        return cls(tuple(layers), kv_heads, doc.get("head_dim"), doc.get("sliding_window"),
                   doc.get("max_position_embeddings"))

    def kv_bytes(self, tokens: int, *, batch: int = 1, bytes_per_element: int = 2,
                 cache_layout: str = "full_allocation") -> int:
        _int(tokens, 1, self.maximum_context, "context_outside_model_limit")
        _int(batch, 1, 128, "invalid_batch")
        _need(type(bytes_per_element) is int and bytes_per_element in (1, 2, 4), "invalid_kv_dtype")
        _need(cache_layout in ("full_allocation", "bounded_sliding"), "unsupported_cache_layout")
        # A sliding attention mask alone does not prove a runtime crops its cache.
        slots = sum(min(tokens, self.window) if cache_layout == "bounded_sliding" and
                    kind == "sliding_attention" else tokens for kind in self.layer_types)
        return 2 * batch * slots * self.kv_heads * self.head_dim * bytes_per_element


def memory_plan(shape: SparkShape, *, tokens: int, batch: int = 1,
                bytes_per_element: int = 2, resident_weight_bytes: int | None = None,
                runtime_reserve_bytes: int | None = None, memory_budget_bytes: int | None = None,
                model_revision: str | None = None, config_sha256: str | None = None) -> dict:
    """Calculate declared budgets, never infer free memory or engine support.

    Both cache layouts are shown. Conservative admission arithmetic uses full
    allocation until an external engine qualification proves otherwise. Even a
    true arithmetic fit never grants execution authority.
    """
    if model_revision is not None:
        _sha(model_revision, 40)
    if config_sha256 is not None:
        _sha(config_sha256)
    for value in (resident_weight_bytes, runtime_reserve_bytes, memory_budget_bytes):
        if value is not None:
            _int(value, 0, 1 << 50, "invalid_memory_declaration")
    full = shape.kv_bytes(tokens, batch=batch, bytes_per_element=bytes_per_element)
    bounded = shape.kv_bytes(tokens, batch=batch, bytes_per_element=bytes_per_element,
                             cache_layout="bounded_sliding")
    complete = resident_weight_bytes is not None and runtime_reserve_bytes is not None
    required = full + resident_weight_bytes + runtime_reserve_bytes if complete else None
    fit = required <= memory_budget_bytes if required is not None and memory_budget_bytes is not None else None
    return {
        "schema": "szl.storage-memory-plan/v1", "state": "CALCULATED_NOT_QUALIFIED",
        "model_id": SPARK_REPO, "model_revision_declared": model_revision,
        "config_sha256": config_sha256, "tokens_including_output": tokens, "batch": batch,
        "kv_bytes_per_element": bytes_per_element,
        "full_attention_layers": shape.layer_types.count("full_attention"),
        "sliding_attention_layers": shape.layer_types.count("sliding_attention"),
        "sliding_window": shape.window, "kv_full_allocation_bytes": full,
        "kv_if_bounded_sliding_bytes": bounded,
        "resident_weight_bytes_declared": resident_weight_bytes,
        "runtime_reserve_bytes_declared": runtime_reserve_bytes,
        "memory_budget_bytes_declared": memory_budget_bytes,
        "conservative_required_bytes": required, "fits_declared_budget": fit,
        "observed_free_memory_bytes": None, "engine_qualified": False,
        "sliding_cache_allocation_verified": False, "weights_verified": False,
        "production_disposition": "HOLD", "authority": authority(),
        "bounds": ["Arithmetic, not a measured allocation or throughput result.",
                   "Tokens include prompt, generated output and tool history.",
                   "Quantized KV needs separate scale/metadata and quality evidence.",
                   "Allocator, prefill workspace and other processes need explicit headroom.",
                   "The dense Spark model has no MoE experts to stream."]}


def reference_plan(tokens: int = 32768, batch: int = 1) -> dict:
    """Transcribed public config dimensions, NOT immutable provider-byte readback."""
    shape = SparkShape(("sliding_attention",) * 3 + ("full_attention",), 4, 256, 512, 1048576)
    shape = SparkShape(shape.layer_types * 9, 4, 256, 512, 1048576)
    result = memory_plan(shape, tokens=tokens, batch=batch)
    result.update(reference="PUBLIC_CONFIG_DIMENSIONS_NOT_RUNTIME_OBSERVATION",
                  source="https://huggingface.co/XHToken/Spark-X2.5-4B/blob/main/config.json",
                  existing_intake_revision=SPARK_INTAKE_REVISION,
                  existing_intake_path=SPARK_INTAKE_PATH)
    return result


@dataclass(frozen=True)
class ExpertChunk:
    """One reviewed per-expert Safetensors payload in an owner-controlled store."""
    expert_id: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _need(type(self.expert_id) is str and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", self.expert_id) is not None,
              "invalid_expert_identity")
        _sha(self.sha256)
        _int(self.size, 1, MAX_CHUNK, "chunk_size_limit")


def bundle_digest(chunks: Sequence[ExpertChunk]) -> str:
    """Bind expert names, serialized hashes and sizes; not a model-lineage attestation."""
    _need(type(chunks) in (tuple, list) and 1 <= len(chunks) <= MAX_EXPERTS
          and all(type(c) is ExpertChunk for c in chunks), "invalid_chunk_manifest")
    _need(len({c.expert_id for c in chunks}) == len(chunks), "duplicate_expert_identity")
    manifest = {"format": "szl.cpu-f32-expert-bundle/v1", "experts": [
        {"id": c.expert_id, "sha256": c.sha256, "size": c.size}
        for c in sorted(chunks, key=lambda c: c.expert_id)]}
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


class ExpertStore:
    """Read/hash exact local bytes. No pickle, URLs, code import, or file mutation.

    This research format is NOT Edge0's packed INT4 layout. A production importer
    and backend need separate qualification; callers control root and ancestors.
    """
    def __init__(self, root: Path, bundle_sha256: str, chunks: Sequence[ExpertChunk]):
        _need(isinstance(root, Path), "store_path_required")
        _sha(bundle_sha256)
        _need(bundle_digest(chunks) == bundle_sha256, "bundle_manifest_digest_mismatch")
        _need(1 <= len(chunks) <= MAX_EXPERTS and all(type(c) is ExpertChunk for c in chunks),
              "invalid_chunk_manifest")
        mapping = {c.expert_id: c for c in chunks}
        _need(len(mapping) == len(chunks), "duplicate_expert_identity")
        self.root = root
        self.bundle_sha256 = bundle_sha256
        self.chunks = MappingProxyType(mapping)

    def chunk(self, expert_id: str) -> ExpertChunk:
        _need(type(expert_id) is str and expert_id in self.chunks, "unknown_expert")
        return self.chunks[expert_id]

    def read(self, expert_id: str) -> bytes:
        spec = self.chunk(expert_id)
        body = read_regular(self.root / (spec.sha256 + ".safetensors"), spec.size)
        _need(len(body) == spec.size and hashlib.sha256(body).hexdigest() == spec.sha256,
              "expert_integrity_failure")
        return body


class ExactExpertCache:
    """Per-session locked byte cache; hints use spare space and never change routes.

    Stage is synchronous I/O preparation, NOT demonstrated I/O/compute overlap.
    One cache is bound to one store; allocate a separate instance per tenant/run.
    Resident payload accounting excludes tensor copies, Python overhead and OS
    page cache. No API here claims bounded total RAM or VRAM.
    """
    def __init__(self, store: ExpertStore, capacity_bytes: int, max_hints: int = 16):
        _need(type(store) is ExpertStore, "exact_store_required")
        self.store = store
        self.capacity = _int(capacity_bytes, 1, 1 << 32, "invalid_cache_budget")
        self.max_hints = _int(max_hints, 0, MAX_ROUTE, "invalid_hint_limit")
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._pinned: dict[str, int] = {}
        self._lock = RLock()
        self._reads = self._hits = self._bytes_read = self._peak = self._skipped = 0

    @property
    def resident_bytes(self) -> int:
        return sum(len(b) for b in self._cache.values())

    def _load(self, key: str) -> bytes:
        body = self.store.read(key)
        self._cache[key] = body
        self._reads += 1
        self._bytes_read += len(body)
        self._peak = max(self._peak, self.resident_bytes)
        return body

    def stage(self, hints: Sequence[str]) -> None:
        """Hints can be wrong. Refuse unbounded work; never evict demanded data."""
        _need(type(hints) in (list, tuple) and len(hints) <= self.max_hints, "bounded_hints_required")
        _need(all(type(key) is str for key in hints) and len(set(hints)) == len(hints), "invalid_hint_set")
        with self._lock:
            for key in hints:
                self.store.chunk(key)
            for key in hints:
                if key in self._cache:
                    continue
                if self.resident_bytes + self.store.chunk(key).size > self.capacity:
                    self._skipped += 1
                    continue
                self._load(key)

    @contextmanager
    def lease(self, exact_route: Sequence[str]) -> Iterator[Mapping[str, bytes]]:
        _need(type(exact_route) in (list, tuple) and 1 <= len(exact_route) <= MAX_ROUTE,
              "bounded_exact_route_required")
        _need(all(type(key) is str for key in exact_route) and len(set(exact_route)) == len(exact_route),
              "duplicate_or_invalid_exact_route")
        with self._lock:
            specs = [self.store.chunk(key) for key in exact_route]
            _need(sum(s.size for s in specs) <= self.capacity, "exact_route_exceeds_cache_budget")
            protected = set(exact_route) | set(self._pinned)
            required = sum(self.store.chunk(k).size for k in protected)
            _need(required <= self.capacity, "leased_payloads_exceed_cache_budget")
            for key in exact_route:
                self._pinned[key] = self._pinned.get(key, 0) + 1
            try:
                result = {}
                for key in exact_route:
                    if key in self._cache:
                        self._hits += 1
                        self._cache.move_to_end(key)
                    else:
                        while self.resident_bytes + self.store.chunk(key).size > self.capacity:
                            victim = next((k for k in self._cache if k not in protected), None)
                            _need(victim is not None, "cache_accounting_failure")
                            del self._cache[victim]
                        self._load(key)
                    result[key] = self._cache[key]
                # The lock intentionally serializes this reference implementation.
                yield MappingProxyType(result)
            finally:
                for key in exact_route:
                    self._pinned[key] -= 1
                    if self._pinned[key] == 0:
                        del self._pinned[key]

    def stats(self) -> dict:
        with self._lock:
            return {"resident_serialized_bytes": self.resident_bytes,
                    "peak_serialized_bytes": self._peak, "capacity_bytes": self.capacity,
                    "file_reads": self._reads, "serialized_bytes_read": self._bytes_read,
                    "demand_cache_hits": self._hits, "hints_skipped_for_budget": self._skipped,
                    "process_peak_bytes": None, "gpu_peak_bytes": None,
                    "physical_storage_type": "UNOBSERVED", "async_overlap_measured": False}


def streamed_moe_reference(cache: ExactExpertCache, x, exact_route: Sequence[tuple[str, float]],
                           *, bundle_sha256: str):
    """CPU FP32 SwiGLU MoE reference on explicit small per-expert files.

    No learned router is installed. The caller's independently selected route
    and weights are authoritative for this function; stage hints cannot replace
    either. Packed INT4, Recover-LoRA and upstream Edge0 adapters are unsupported.
    """
    import torch
    from safetensors.torch import load
    _need(type(cache) is ExactExpertCache and _sha(bundle_sha256) == cache.store.bundle_sha256,
          "bundle_identity_mismatch")
    _need(isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dtype == torch.float32 and
          x.ndim == 2 and 1 <= x.shape[0] <= 64 and 1 <= x.shape[1] <= 4096 and
          torch.isfinite(x).all().item(), "bounded_finite_cpu_input_required")
    _need(type(exact_route) in (list, tuple) and 1 <= len(exact_route) <= MAX_ROUTE,
          "invalid_exact_route")
    for pair in exact_route:
        _need(type(pair) in (list, tuple) and len(pair) == 2 and type(pair[0]) is str,
              "invalid_route_pair")
        _need(type(pair[1]) in (float, int) and math.isfinite(pair[1]) and 0 <= pair[1] <= 1,
              "invalid_route_weight")
    _need(math.isclose(sum(p[1] for p in exact_route), 1.0, rel_tol=0, abs_tol=1e-6),
          "normalized_route_required")
    keys = [pair[0] for pair in exact_route]
    with torch.inference_mode(), cache.lease(keys) as blobs:
        output = torch.zeros_like(x)
        for key, weight in exact_route:
            tensors = load(blobs[key])
            _need(set(tensors) == {"gate", "up", "down"}, "unsupported_expert_layout")
            gate, up, down = (tensors[n] for n in ("gate", "up", "down"))
            _need(all(t.dtype == torch.float32 and t.ndim == 2 and torch.isfinite(t).all().item()
                      for t in (gate, up, down)), "invalid_expert_tensors")
            _need(gate.shape == up.shape and gate.shape[1] == x.shape[1] and
                  down.shape == (x.shape[1], gate.shape[0]), "expert_shape_mismatch")
            value = torch.nn.functional.silu(x @ gate.T) * (x @ up.T)
            output.add_((value @ down.T) * weight)
        _need(torch.isfinite(output).all().item(), "nonfinite_reference_output")
        return output
