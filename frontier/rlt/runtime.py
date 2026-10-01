"""Controller-owned CPU/FP32 research sessions with complete-state continuity.

Integrity is not authenticity. Expected checkpoint digests, policy, tenant and
source-evidence digests must come from the trusted controller, not the same
untrusted producer as a checkpoint. No external provider, tool or pickle loader.
No claim of isolation from hostile code in the same Python process.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import platform
import struct
import sys
import threading
from typing import Iterator

from pathlib import Path
from safetensors import SafetensorError
from safetensors.torch import load, save
import torch
from torch import Tensor

from .model import CompleteState, KV, TinyRLT, detached_state

SCHEMA = "szl.rlt.complete-state.v1"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


class ContinuityError(ValueError):
    """Invalid state or changed history requires fresh replay, never reuse."""


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checked_digest(value: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContinuityError("a full lowercase SHA-256 digest is required")
    return value


@dataclass(frozen=True)
class Scope:
    tenant_sha256: str
    policy_sha256: str
    evidence_sha256: str
    tokenizer_sha256: str
    template_sha256: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            checked_digest(value)


def tensor_digest(value: Tensor) -> str:
    if value.device.type != "cpu" or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ContinuityError("only finite CPU float32 tensors are supported")
    header = canonical({"shape": list(value.shape), "dtype": str(value.dtype)})
    # Explicit little-endian bytes; no pickle and no dependency on NumPy.
    data = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
    return digest(struct.pack("<Q", len(header)) + header + data)


def state_tensors(state: CompleteState) -> dict[str, Tensor]:
    return {"encoder_k": state.encoder_kv.k, "encoder_v": state.encoder_kv.v,
            "memory_k": state.encoder_memory.k, "memory_v": state.encoder_memory.v,
            "decoder_k": state.decoder_kv.k, "decoder_v": state.decoder_kv.v,
            "output": state.output}


def token_list(values: object, vocab: int) -> tuple[int, ...]:
    if type(values) not in (tuple, list) or any(type(x) is not int or not 0 <= x < vocab for x in values):
        raise ContinuityError("tokens must be a list/tuple of in-vocabulary integers")
    return tuple(values)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContinuityError("duplicate JSON key")
        result[key] = value
    return result


class Session:
    """One owned model, one ordered history, no cross-request shared state.

    A deep copy prevents outside mutation of the supplied model from silently
    changing a session. Inference snapshots are never training checkpoints.
    Calls are single-flight and append commits only after complete validation.
    """
    def __init__(self, model: TinyRLT, scope: Scope) -> None:
        if type(model) is not TinyRLT or type(scope) is not Scope:
            raise ContinuityError("exact TinyRLT and Scope types required")
        if sys.byteorder != "little":
            raise ContinuityError("this reference supports little-endian hosts only")
        if model.max_tokens > 512 or model.dim > 64 or model.vocab > 512:
            raise ContinuityError("research resource bounds exceeded")
        self._model = deepcopy(model).eval()
        self._scope = scope
        self._lock = threading.Lock()
        self._tokens: tuple[int, ...] = ()
        self._pending: int | None = None
        self._state = detached_state(self._model.empty_state(1))
        self._identity = self._identity_now()
        self._seal = self._components(self._state)

    def _identity_now(self) -> str:
        if torch.is_autocast_enabled("cpu"):
            raise ContinuityError("CPU autocast is outside this execution contract")
        m = self._model
        content = {name: tensor_digest(value) for name, value in m.state_dict().items()}
        config = {"vocab": m.vocab, "dim": m.dim, "heads": m.shared_attention.heads,
                  "window": m.window, "max_tokens": m.max_tokens}
        return digest(canonical({"scope": asdict(self._scope), "weights": content, "config": config,
            "torch": torch.__version__, "machine": platform.machine(), "python": platform.python_version(),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "mkldnn": torch.backends.mkldnn.enabled, "deterministic": torch.are_deterministic_algorithms_enabled(),
            "source_files": {name: digest(Path(__file__).with_name(name).read_bytes())
                             for name in ("model.py", "runtime.py")},
            "matmul_precision": torch.get_float32_matmul_precision(),
            "execution": "szl.rlt.cpu-fp32-inference.v1"}))

    def _components(self, state: CompleteState) -> dict[str, str]:
        if type(state.consumed) is not int or not 0 <= state.consumed <= self._model.max_tokens:
            raise ContinuityError("invalid consumed state count")
        m, n = self._model, state.consumed
        h = m.shared_attention.heads
        global_shape = (1, h, n, m.dim // h)
        local_shape = (1, h, min(n, m.window - 1), m.dim // h)
        result = {}
        for name, value in state_tensors(state).items():
            shape = (1, 1, m.dim) if name == "output" else local_shape if name.startswith("decoder") else global_shape
            if tuple(value.shape) != shape:
                raise ContinuityError("incomplete or inconsistent state shapes")
            result[name] = tensor_digest(value)
        return result

    @contextmanager
    def _guard(self) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise ContinuityError("session already in use")
        try:
            if self._identity_now() != self._identity or self._components(self._state) != self._seal:
                raise ContinuityError("model, execution or state changed: replay required")
            yield
        finally:
            self._lock.release()

    @property
    def consumed(self) -> int:
        return len(self._tokens)

    def append(self, tokens: list[int] | tuple[int, ...]) -> Tensor:
        with self._guard(), torch.no_grad():
            extra = token_list(tokens, self._model.vocab)
            if not extra or len(extra) + self.consumed > self._model.max_tokens:
                raise ContinuityError("empty input or context budget exceeded")
            if self._pending is not None and extra[0] != self._pending:
                raise ContinuityError("pending emitted token must be consumed exactly once first")
            state, outputs = self._state, []
            for token in extra:
                logits, state = self._model.step(torch.tensor([[token]], dtype=torch.long), state)
                if not bool(torch.isfinite(logits).all()):
                    raise ContinuityError("nonfinite model output")
                outputs.append(logits)
            seal = self._components(state)
            self._state, self._seal = detached_state(state), seal
            self._tokens += extra
            self._pending = None
            return torch.cat(outputs, dim=1).clone()

    def emit_greedy(self) -> int:
        with self._guard(), torch.no_grad():
            if not self.consumed or self._pending is not None or self.consumed >= self._model.max_tokens:
                raise ContinuityError("generation needs context, budget and no pending token")
            logits = self._model.lm_head(self._model.output_norm(self._state.output))
            if not bool(torch.isfinite(logits).all()):
                raise ContinuityError("nonfinite model output")
            token = int(logits.argmax(dim=-1).item())
            self._pending = token
            return token

    def checkpoint(self) -> bytes:
        with self._guard():
            passport = {"schema": SCHEMA, "identity": self._identity,
                        "prefix_sha256": digest(canonical(self._tokens)), "consumed": self.consumed,
                        "pending": self._pending, "components": self._seal,
                        "purpose": "INFERENCE_ONLY", "execution_authority": "NONE"}
            tensors = {name: value.detach().contiguous().clone() for name, value in state_tensors(self._state).items()}
            artifact = save(tensors, metadata={"passport": canonical(passport).decode("utf-8")})
            if len(artifact) > MAX_ARTIFACT_BYTES:
                raise ContinuityError("checkpoint resource bound exceeded")
            return artifact

    @classmethod
    def restore(cls, model: TinyRLT, scope: Scope, artifact: bytes, *, expected_sha256: str,
                consumed_tokens: list[int] | tuple[int, ...], pending_token: int | None = None) -> Session:
        """Restore bytes only with an independently held expected digest/history."""
        checked_digest(expected_sha256)
        if type(artifact) is not bytes or not 8 < len(artifact) <= MAX_ARTIFACT_BYTES:
            raise ContinuityError("invalid checkpoint length")
        if digest(artifact) != expected_sha256:
            raise ContinuityError("checkpoint bytes do not match the trusted expected digest")
        session = cls(model, scope)
        tokens = token_list(consumed_tokens, model.vocab)
        if pending_token is not None:
            token_list([pending_token], model.vocab)
        header_size = struct.unpack("<Q", artifact[:8])[0]
        if header_size > min(len(artifact) - 8, 65536):
            raise ContinuityError("invalid checkpoint header length")
        try:
            header = json.loads(artifact[8:8 + header_size], object_pairs_hook=_unique_object)
            metadata = header["__metadata__"]
            if set(metadata) != {"passport"}:
                raise ContinuityError("unexpected checkpoint metadata")
            passport = json.loads(metadata["passport"], object_pairs_hook=_unique_object)
            expected = {"schema": SCHEMA, "identity": session._identity,
                        "prefix_sha256": digest(canonical(tokens)), "consumed": len(tokens),
                        "pending": pending_token, "purpose": "INFERENCE_ONLY", "execution_authority": "NONE"}
            if set(passport) != set(expected) | {"components"}:
                raise ContinuityError("unexpected passport fields")
            # Canonical byte equality distinguishes true/1 and rejects NaN.
            if any(canonical(passport[key]) != canonical(value) for key, value in expected.items()):
                raise ContinuityError("history, evidence, identity or pending token changed: replay required")
            tensors = load(artifact)
            if set(tensors) != set(state_tensors(session._state)):
                raise ContinuityError("missing or extra state tensors")
            state = CompleteState(KV(tensors["encoder_k"], tensors["encoder_v"]),
                                  KV(tensors["memory_k"], tensors["memory_v"]),
                                  KV(tensors["decoder_k"], tensors["decoder_v"]), tensors["output"], len(tokens))
            seal = session._components(state)
            if canonical(seal) != canonical(passport["components"]):
                raise ContinuityError("state component digest mismatch")
        except (KeyError, TypeError, ValueError, OverflowError, SafetensorError) as exc:
            raise ContinuityError("checkpoint validation failed") from exc
        session._state, session._seal, session._tokens, session._pending = detached_state(state), seal, tokens, pending_token
        return session

    def replay(self, scope: Scope, history: list[int] | tuple[int, ...]) -> Session:
        """Fresh reconstruction after edited context/evidence; old session unchanged."""
        with self._guard():
            new = Session(self._model, scope)
            new.append(history)
            return new

    def observation(self) -> dict[str, object]:
        """Sanitized observer projection. No raw tokens, state tensors or reasoning."""
        with self._guard():
            return {"schema": "szl.rlt.observation.v1", "identity_sha256": self._identity,
                    "consumed": self.consumed, "pending_present": self._pending is not None,
                    "state_sha256": digest(canonical(self._seal)), "status": "RESEARCH_REFERENCE",
                    "task_verified": False, "execution_authority": "NONE", "receipt_status": "UNSIGNED"}
