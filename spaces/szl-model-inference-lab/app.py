from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import platform
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SPACE_ID = "SZLHOLDINGS/szl-model-inference-lab"
MODEL_REPO = "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF"
MODEL_REVISION = "67d60ec577730747055491640cfb91fc4a4b5d25"
MODEL_FILE = "SZL-Khipu-1.5B-Q4_K_M.gguf"
MODEL_SIZE = 986_047_904
MODEL_SHA256 = "13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a"
OPENAI_MODEL_ID = f"{MODEL_REPO}@{MODEL_REVISION}"
# Timestamp of the immutable Hub revision used by this runtime, not a claim about
# the creation date of the underlying base model.
MODEL_REVISION_CREATED_UNIX = 1_784_133_947
RECEIPT_FILES = (
    "training_receipt.signed.json",
    "eval_receipt.signed.json",
    "owner_pubkey.json",
)
MAX_INPUT_CHARS = 1_200
MAX_CHAT_MESSAGES = 12
MAX_PROMPT_TOKENS = 800
MAX_NEW_TOKENS = 32
INFERENCE_BUDGET_SECONDS = 45.0
BODY_READ_TIMEOUT_SECONDS = 10.0
RESERVED_CHAT_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
SYSTEM_PROMPT = (
    "You are a bounded research demo. Answer briefly. If evidence is missing, "
    "say so; do not claim independent benchmarking or safety certification."
)
SOURCE_ROOT = Path(__file__).resolve().parent
SOURCE_REVISION_ENV = "SZL_GITHUB_SOURCE_REVISION"


state: dict[str, Any] = {
    "status": "STARTING",
    "failure_code": None,
    "model_path": None,
    "model_sha256": None,
    "source_integrity": False,
    "receipt_status": "NOT_CHECKED",
    "llama_cpp_version": None,
}
llm: Any = None
inference_lock = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_file(path: Path) -> str:
    """Hash the Git-canonical text bytes on every checkout platform.

    The release manifest is generated from Git blobs, whose text files use LF.
    Git may materialize those same files with CRLF on Windows. Normalizing that
    checkout-only transform preserves source integrity without weakening the
    byte comparison performed for binary model artifacts.
    """

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_release_manifest() -> dict[str, Any]:
    manifest = json.loads((SOURCE_ROOT / "release.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_files"].items():
        if sha256_source_file(SOURCE_ROOT / relative) != expected:
            raise RuntimeError("SOURCE_INTEGRITY_MISMATCH")
    return manifest


def artifact_path(filename: str) -> Path:
    override = os.getenv("MODEL_DIR_OVERRIDE")
    if override:
        return Path(override) / filename
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            revision=MODEL_REVISION,
            local_files_only=True,
            token=False,
        )
    )


def verify_receipts() -> dict[str, Any]:
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    declared_key = json.loads(artifact_path("owner_pubkey.json").read_text(encoding="utf-8"))
    receipts: dict[str, dict[str, Any]] = {}
    for filename in RECEIPT_FILES[:2]:
        receipt = json.loads(artifact_path(filename).read_text(encoding="utf-8"))
        canonical = json.dumps(
            receipt["payload"], ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if canonical != receipt["canonical"]:
            raise RuntimeError("RECEIPT_CANONICAL_MISMATCH")
        if receipt["publicKeySpkiBase64"] != declared_key["publicKeySpkiBase64"]:
            raise RuntimeError("RECEIPT_KEY_MISMATCH")
        public_key = load_der_public_key(base64.b64decode(receipt["publicKeySpkiBase64"]))
        public_key.verify(
            base64.b64decode(receipt["signatureBase64"]), canonical.encode("utf-8")
        )
        receipts[filename] = receipt
    training_digest = hashlib.sha256(
        receipts["training_receipt.signed.json"]["canonical"].encode("utf-8")
    ).hexdigest()
    if (
        receipts["eval_receipt.signed.json"]["payload"]["trainingReceiptSha256"]
        != training_digest
    ):
        raise RuntimeError("RECEIPT_CHAIN_MISMATCH")
    return {
        "status": "DECLARED_KEY_SIGNATURES_VALID",
        "key_id": declared_key["keyId"],
        "training_canonical_sha256": training_digest,
        "eval_canonical_sha256": hashlib.sha256(
            receipts["eval_receipt.signed.json"]["canonical"].encode("utf-8")
        ).hexdigest(),
    }


def initialize() -> None:
    global llm
    state["status"] = "STARTING"
    state["failure_code"] = None
    try:
        manifest = load_release_manifest()
        state["source_integrity"] = True
        state["release_id"] = manifest["release_id"]

        model_path = artifact_path(MODEL_FILE)
        if model_path.stat().st_size != MODEL_SIZE:
            raise RuntimeError("MODEL_SIZE_MISMATCH")
        actual_sha = sha256_file(model_path)
        if actual_sha != MODEL_SHA256:
            raise RuntimeError("MODEL_SHA256_MISMATCH")
        state["model_path"] = str(model_path)
        state["model_sha256"] = actual_sha

        receipt = verify_receipts()
        state["receipt_status"] = receipt["status"]
        state["receipt_evidence"] = receipt

        import llama_cpp
        from llama_cpp import Llama

        state["llama_cpp_version"] = llama_cpp.__version__
        threads = max(1, min(int(os.getenv("CPU_CORES", "2")), 2))
        llm = Llama(
            model_path=str(model_path),
            n_ctx=1024,
            n_batch=64,
            n_threads=threads,
            n_threads_batch=threads,
            seed=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        state["status"] = "READY"
    except Exception as exc:  # keep /health available for honest diagnostics
        state["status"] = "FAILED"
        state["failure_code"] = str(exc) if str(exc).isupper() else type(exc).__name__


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["status"] = "STARTING"
    threading.Thread(target=initialize, name="model-initializer", daemon=True).start()
    yield


app = FastAPI(
    title="SZL Model Inference Lab",
    version="1.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


class BodyLimitMiddleware:
    """Enforce a body limit while consuming ASGI chunks, then replay valid bytes."""

    def __init__(
        self,
        app: Any,
        max_bytes: int = 8192,
        read_timeout_seconds: float = BODY_READ_TIMEOUT_SECONDS,
    ) -> None:
        self.application = app
        self.max_bytes = max_bytes
        self.read_timeout_seconds = read_timeout_seconds

    @staticmethod
    async def _reject(
        send: Any, status: int, detail: str, path: str = ""
    ) -> None:
        payload: dict[str, Any]
        if path.startswith("/v1/"):
            payload = {
                "error": {
                    "message": detail,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "request_body_error",
                }
            }
        else:
            payload = {"detail": detail}
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.application(scope, receive, send)
            return

        path = scope.get("path", "")
        header_map = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = header_map.get(b"content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length)
            except ValueError:
                await self._reject(send, 400, "invalid content-length", path)
                return
            if declared < 0:
                await self._reject(send, 400, "invalid content-length", path)
                return
            if declared > self.max_bytes:
                await self._reject(send, 413, "request body too large", path)
                return

        chunks: list[bytes] = []
        total = 0
        try:
            async with asyncio.timeout(self.read_timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    if message["type"] != "http.request":
                        continue
                    chunk = message.get("body", b"")
                    total += len(chunk)
                    if total > self.max_bytes:
                        await self._reject(send, 413, "request body too large", path)
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await self._reject(send, 408, "request body timeout", path)
            return

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.application(scope, replay_receive, send)


app.add_middleware(
    BodyLimitMiddleware,
    max_bytes=8192,
    read_timeout_seconds=BODY_READ_TIMEOUT_SECONDS,
)


@app.middleware("http")
async def bounded_requests(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def openai_error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = provenance_headers()
    response_headers["Cache-Control"] = "no-store"
    response_headers.update(headers or {})
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        status_code=status_code,
        headers=response_headers,
    )


@app.exception_handler(RequestValidationError)
async def bounded_validation_error_handler(
    request: Request, exc: RequestValidationError
):
    if not request.url.path.startswith("/v1/"):
        return await request_validation_exception_handler(request, exc)
    first = exc.errors()[0] if exc.errors() else {}
    location = [str(value) for value in first.get("loc", ()) if value != "body"]
    param = ".".join(location) or None
    message = first.get("msg", "request failed validation")
    return openai_error_response(
        422,
        message,
        "invalid_request_error",
        "validation_error",
        param,
    )


@app.exception_handler(HTTPException)
async def bounded_http_error_handler(request: Request, exc: HTTPException):
    if not request.url.path.startswith("/v1/"):
        return await http_exception_handler(request, exc)
    if exc.status_code == 429:
        error_type, code = "rate_limit_error", "concurrency_limit"
    elif exc.status_code == 503:
        error_type, code = "server_error", "runtime_not_ready"
    else:
        error_type, code = "invalid_request_error", "request_rejected"
    return openai_error_response(
        exc.status_code,
        str(exc.detail),
        error_type,
        code,
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def bounded_unhandled_error_handler(request: Request, _: Exception):
    """Return a non-leaking, non-cacheable envelope for unexpected failures."""

    if request.url.path.startswith("/v1/"):
        return openai_error_response(
            500,
            "internal server error",
            "server_error",
            "internal_error",
        )
    return JSONResponse(
        {"detail": "internal server error"},
        status_code=500,
        headers={"Cache-Control": "no-store"},
    )


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    max_new_tokens: int = Field(default=24, ge=1, le=MAX_NEW_TOKENS)

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("prompt must contain visible text and no NUL bytes")
        if any(token in value for token in RESERVED_CHAT_TOKENS):
            raise ValueError("prompt contains a reserved chat control token")
        return value


class ChatMessage(BaseModel):
    """Supported string-only subset of an OpenAI chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("message content must contain visible text and no NUL bytes")
        if any(token in value for token in RESERVED_CHAT_TOKENS):
            raise ValueError("message content contains a reserved ChatML control token")
        return value


class ChatCompletionRequest(BaseModel):
    """Bounded, deterministic, non-streaming OpenAI-compatible request subset."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_CHAT_MESSAGES)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS)
    max_completion_tokens: int | None = Field(
        default=None, ge=1, le=MAX_NEW_TOKENS
    )
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    top_p: float = Field(default=1.0, ge=1.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stream: Literal[False] = False
    tools: None = None
    tool_choice: None = None
    parallel_tool_calls: None = None

    @field_validator("max_tokens", "max_completion_tokens", "n", mode="before")
    @classmethod
    def require_json_integer(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("field must be a JSON integer")
        return value

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def require_json_number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("field must be a JSON number")
        return value

    @field_validator("stream", mode="before")
    @classmethod
    def require_json_boolean(cls, value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError("stream must be a JSON boolean")
        return value

    @model_validator(mode="after")
    def enforce_bounded_chat_contract(self) -> "ChatCompletionRequest":
        if self.model != OPENAI_MODEL_ID:
            raise ValueError(f"model must be the immutable id {OPENAI_MODEL_ID}")
        if self.messages[-1].role != "user":
            raise ValueError("the final message must have role user")
        total_chars = sum(len(message.content) for message in self.messages)
        if total_chars > MAX_INPUT_CHARS:
            raise ValueError(
                f"combined message content exceeds {MAX_INPUT_CHARS} characters"
            )
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError(
                "max_tokens and max_completion_tokens must match when both are supplied"
            )
        return self

    @property
    def generation_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 24


def formatted_chat(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def formatted_chat_messages(messages: list[ChatMessage]) -> str:
    """Render the exact ChatML string that is counted and sent to llama.cpp."""

    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for message in messages:
        parts.append(
            f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        )
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def enforce_prompt_budget(prompt_tokens: int) -> None:
    if prompt_tokens > MAX_PROMPT_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"formatted prompt exceeds {MAX_PROMPT_TOKENS} tokens",
        )


def identity_payload() -> dict[str, Any]:
    manifest = json.loads((SOURCE_ROOT / "release.json").read_text(encoding="utf-8"))
    return {
        "schema": "szl.hf-free-inference-identity/v1",
        "status": state["status"],
        "space": {
            "id": os.getenv("SPACE_ID", SPACE_ID),
            "release_id": manifest["release_id"],
            "release_manifest_sha256": sha256_file(SOURCE_ROOT / "release.json"),
            "source_integrity": state["source_integrity"],
            "source_integrity_meaning": (
                "internal release-file checksum consistency; not external authorship evidence"
            ),
            "license": "Apache-2.0",
        },
        "hardware": {
            "required": "cpu-basic",
            "accelerator_observed": os.getenv("ACCELERATOR", "none"),
            "cpu_cores_observed": os.getenv("CPU_CORES", "unknown"),
            "memory_observed": os.getenv("MEMORY", "unknown"),
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "size": MODEL_SIZE,
            "sha256_expected": MODEL_SHA256,
            "sha256_loaded": state["model_sha256"],
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "license": "Apache-2.0",
        },
        "runtime": {
            "python": platform.python_version(),
            "llama_cpp_python": state["llama_cpp_version"],
            "concurrency": 1,
            "max_input_chars": MAX_INPUT_CHARS,
            "max_chat_messages": MAX_CHAT_MESSAGES,
            "max_formatted_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "inference_budget_seconds": INFERENCE_BUDGET_SECONDS,
            "inference_budget_semantics": (
                "best-effort cutoff checked between streamed token chunks; not a hard wall-clock deadline"
            ),
            "dependency_boundary": (
                "Python package versions are pinned; llama CPU wheel is hash-pinned; "
                "a full system-package/SBOM attestation is not claimed"
            ),
            "max_request_body_bytes": 8192,
            "body_read_timeout_seconds": BODY_READ_TIMEOUT_SECONDS,
            "openai_compatible_subset": {
                "models": "GET /v1/models",
                "chat_completions": "POST /v1/chat/completions",
                "model_id": OPENAI_MODEL_ID,
                "streaming": False,
                "tools": False,
                "choices": 1,
                "decoding": "greedy temperature=0, top_p=1",
            },
        },
        "receipt_boundary": {
            "status": state["receipt_status"],
            "evidence": state.get("receipt_evidence"),
            "covers": (
                "training/eval signatures verify against the repository-declared key, "
                "including their canonical-payload hash chain"
            ),
            "does_not_cover": [
                "independent identity or key-ownership binding",
                "Space source authorship",
                "GGUF quantization quality",
                "this runtime's outputs",
                "independent benchmarking",
                "safety certification",
            ],
        },
        "failure_code": state["failure_code"],
    }


def build_info_payload() -> dict[str, Any]:
    """Expose the exact Git source bound by the governed deploy workflow.

    The Hugging Face variable is non-secret. A missing or malformed revision
    stays UNKNOWN; it is never inferred from the Space repository revision or
    the internal release manifest.
    """

    raw_revision = os.getenv(SOURCE_REVISION_ENV, "").strip().lower()
    revision_observed = (
        len(raw_revision) == 40
        and all(character in "0123456789abcdef" for character in raw_revision)
    )
    return {
        "schema": "szl.build-info/v1",
        "service": "szl-model-inference-lab",
        "build": {
            "state": "OBSERVED" if revision_observed else "UNKNOWN",
            "revision": raw_revision if revision_observed else None,
            "revision_source": (
                f"Hugging Face Space variable {SOURCE_REVISION_ENV}"
                if revision_observed
                else "UNAVAILABLE"
            ),
        },
        "runtime": {
            "state": state["status"],
            "model_revision": MODEL_REVISION,
            "model_sha256_verified": state["model_sha256"] == MODEL_SHA256,
            "source_integrity": state["source_integrity"],
        },
        # Execution records are deliberately unsigned. Source binding is
        # lineage evidence, not a minted cryptographic release receipt.
        "receipt_minted": False,
    }


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def provenance_headers(execution_record_sha256: str | None = None) -> dict[str, str]:
    """Compact, namespaced boundary markers for API clients and gateways."""

    headers = {
        "X-SZL-Provenance-Schema": "szl.openai-compat-provenance/v1",
        "X-SZL-Model-Revision": MODEL_REVISION,
        "X-SZL-Model-SHA256": MODEL_SHA256,
        "X-SZL-Output-Signature": "none",
        "X-SZL-Native-Provider-Mapping": "not-claimed",
        "X-SZL-Service-Level": "best-effort-no-sla",
    }
    if execution_record_sha256 is not None:
        headers["X-SZL-Execution-Record-SHA256"] = execution_record_sha256
    return headers


def unsigned_provenance(
    termination_reason: str | None = None,
    execution_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Truthful provenance for this execution without implying output signing."""

    manifest = json.loads((SOURCE_ROOT / "release.json").read_text(encoding="utf-8"))
    return {
        "schema": "szl.openai-compat-provenance/v1",
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
        },
        "runtime": {
            "space": os.getenv("SPACE_ID", SPACE_ID),
            "release_id": manifest["release_id"],
            "service_level": "BEST_EFFORT_NO_SLA",
            "native_hugging_face_provider_mapping": "NOT_CLAIMED",
        },
        "receipts": {
            "status": state["receipt_status"],
            "scope": "repository-declared key continuity only",
            "covers_this_output": False,
        },
        "output": {
            "signature_status": "UNSIGNED",
            "signature": None,
            "termination_reason": termination_reason,
        },
        "execution_record": execution_record,
    }


def build_unsigned_execution_record(
    request: ChatCompletionRequest,
    result: dict[str, Any],
    request_id: str,
    created_unix: int,
) -> dict[str, Any]:
    """Build a content-addressed record without retaining prompt or output text."""

    canonical_request = {
        "schema": "szl.openai-chat-request/v1",
        "model": request.model,
        "messages": [message.model_dump(mode="json") for message in request.messages],
        "max_completion_tokens": request.generation_limit,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "tools": None,
    }
    manifest_path = SOURCE_ROOT / "release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = {
        "schema": "szl.unsigned-execution-record/v1",
        "request_id": request_id,
        "created_unix": created_unix,
        "canonical_request_sha256": sha256_json(canonical_request),
        "output_sha256": hashlib.sha256(
            result["output"].encode("utf-8")
        ).hexdigest(),
        "model": {
            "id": OPENAI_MODEL_ID,
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
        },
        "source": {
            "space_id": os.getenv("SPACE_ID", SPACE_ID),
            "release_id": manifest["release_id"],
            "release_manifest_sha256": sha256_file(manifest_path),
        },
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "elapsed_ms": result["elapsed_ms"],
        "termination": {
            "reason": result["finish_reason"],
            "time_budget_reached": result["finish_reason"] == "time_budget",
        },
        "signature_status": "UNSIGNED",
        "signature": None,
        "authenticity_not_established": True,
        "persistence": {
            "application_record_storage": "NOT_PERSISTED",
            "boundary": "platform or network logging outside this source is not asserted",
        },
        "record_sha256_scope": (
            "canonical UTF-8 JSON with sorted keys and compact separators, "
            "excluding record_sha256"
        ),
    }
    record["record_sha256"] = sha256_json(record)
    return record


@app.get("/health")
def health() -> JSONResponse:
    code = 200 if state["status"] == "READY" else 503
    return JSONResponse(
        {
            "status": state["status"],
            "model_sha256_verified": state["model_sha256"] == MODEL_SHA256,
            "source_integrity": state["source_integrity"],
            "receipt_status": state["receipt_status"],
            "failure_code": state["failure_code"],
        },
        status_code=code,
    )


@app.get("/live")
def live() -> JSONResponse:
    code = 503 if state["status"] == "FAILED" else 200
    return JSONResponse({"status": state["status"]}, status_code=code)


@app.get("/api/v1/identity")
def identity() -> dict[str, Any]:
    return identity_payload()


@app.get("/api/build-info")
def build_info() -> JSONResponse:
    return JSONResponse(build_info_payload())


@app.get("/.well-known/szl-inference-contract.json")
def inference_contract() -> JSONResponse:
    return JSONResponse(
        {
            "schema": "szl.inference-contract/v1",
            "status": state["status"],
            "execution_surface": "public Hugging Face Docker Space on cpu-basic",
            "model": {
                "id": OPENAI_MODEL_ID,
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "file": MODEL_FILE,
                "size": MODEL_SIZE,
                "sha256": MODEL_SHA256,
            },
            "openai_compatible_subset": {
                "base_path": "/v1",
                "models": {"method": "GET", "path": "/v1/models"},
                "chat_completions": {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "streaming": False,
                    "tools": False,
                    "choices": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                "model_id": OPENAI_MODEL_ID,
            },
            "authentication": {
                "required_by_application": False,
                "client_compatibility_dummy_key": "not-a-secret",
                "warning": (
                    "Do not send real Hugging Face, OpenAI, or other access tokens; "
                    "this public demo does not use them."
                ),
            },
            "limits": {
                "concurrency": 1,
                "max_request_body_bytes": 8192,
                "body_read_timeout_seconds": BODY_READ_TIMEOUT_SECONDS,
                "max_messages": MAX_CHAT_MESSAGES,
                "max_total_message_chars": MAX_INPUT_CHARS,
                "max_formatted_prompt_tokens": MAX_PROMPT_TOKENS,
                "max_completion_tokens": MAX_NEW_TOKENS,
                "inference_budget_seconds": INFERENCE_BUDGET_SECONDS,
            },
            "privacy": {
                "application_prompt_storage": "NOT_INTENTIONALLY_PERSISTED",
                "sensitive_prompt_guidance": (
                    "Do not submit secrets, credentials, regulated data, personal data, "
                    "or other sensitive content to this public best-effort demo."
                ),
                "boundary": "platform or network logging outside this source is not asserted",
            },
            "provenance": {
                "execution_record": "content-addressed and not persisted by this source",
                "signature_status": "UNSIGNED",
                "authenticity_not_established": True,
                "native_hugging_face_provider_mapping": "NOT_CLAIMED",
                "service_level": "BEST_EFFORT_NO_SLA",
            },
        },
        headers=provenance_headers(),
    )


@app.get("/v1/models")
def openai_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": OPENAI_MODEL_ID,
                    "object": "model",
                    "created": MODEL_REVISION_CREATED_UNIX,
                    "owned_by": "SZLHOLDINGS",
                    "szl_created_basis": "pinned_revision_last_modified_at",
                    "szl_provenance": unsigned_provenance(),
                }
            ],
        },
        headers=provenance_headers(),
    )


def run_bounded_completion(prompt: str, max_new_tokens: int) -> dict[str, Any]:
    """Single generation path shared by the native and compatibility APIs."""

    if state["status"] != "READY" or llm is None:
        raise HTTPException(status_code=503, detail="runtime is not ready")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"max generated tokens must be between 1 and {MAX_NEW_TOKENS}",
        )
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="one inference is already running",
            headers={"Retry-After": str(int(INFERENCE_BUDGET_SECONDS))},
        )
    started = time.monotonic()
    try:
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        enforce_prompt_budget(prompt_tokens)
        stream = llm.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )
        chunks: list[str] = []
        finish_reason = None
        timed_out = False
        try:
            for event in stream:
                if time.monotonic() - started >= INFERENCE_BUDGET_SECONDS:
                    timed_out = True
                    break
                choice = event["choices"][0]
                chunks.append(choice.get("text") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        output = "".join(chunks).strip()
        completion_tokens = (
            len(
                llm.tokenize(
                    output.encode("utf-8"), add_bos=False, special=False
                )
            )
            if output
            else 0
        )
        return {
            "output": output,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "time_budget" if timed_out else finish_reason,
        }
    finally:
        inference_lock.release()


@app.post("/api/v1/infer")
def infer(request: InferenceRequest) -> dict[str, Any]:
    result = run_bounded_completion(
        formatted_chat(request.prompt), request.max_new_tokens
    )
    return {
        **result,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
        },
        "determinism": "greedy temperature=0; exact model bytes are disclosed",
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(request: ChatCompletionRequest) -> JSONResponse:
    prompt = formatted_chat_messages(request.messages)
    result = run_bounded_completion(prompt, request.generation_limit)
    request_id = f"chatcmpl-szl-{uuid.uuid4().hex}"
    created_unix = int(time.time())
    execution_record = build_unsigned_execution_record(
        request, result, request_id, created_unix
    )
    internal_finish_reason = result["finish_reason"]
    if internal_finish_reason in {"length", "time_budget"}:
        finish_reason: str | None = "length"
    elif internal_finish_reason == "stop":
        finish_reason = "stop"
    else:
        finish_reason = None

    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": created_unix,
            "model": OPENAI_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["output"],
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"]
                + result["completion_tokens"],
            },
            "szl_provenance": unsigned_provenance(
                internal_finish_reason, execution_record
            ),
        },
        headers={
            **provenance_headers(execution_record["record_sha256"]),
            "Cache-Control": "no-store",
        },
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SZL Model Inference Lab</title><style>
body{margin:0;background:#080b13;color:#e9efff;font:16px/1.5 system-ui,sans-serif}main{max-width:820px;margin:auto;padding:40px 22px}
.card{background:#101728;border:1px solid #2a385b;border-radius:18px;padding:24px;box-shadow:0 18px 60px #0007}h1{margin:.1em 0;color:#8ce8ff}
textarea{box-sizing:border-box;width:100%;min-height:130px;background:#080d19;color:#fff;border:1px solid #40537f;border-radius:10px;padding:12px}
button{margin-top:12px;background:#7ce6ff;color:#06101a;border:0;border-radius:999px;padding:10px 18px;font-weight:700;cursor:pointer}pre{white-space:pre-wrap;background:#080d19;padding:14px;border-radius:10px;min-height:52px}code{color:#a7f3d0}.fine{color:#aebbd4;font-size:.9rem}a{color:#8ce8ff}</style></head>
<body><main><div class="card"><p class="fine">FREE CPU BASIC · ONE REQUEST AT A TIME · IMMUTABLE Q4_K_M</p><h1>SZL Model Inference Lab</h1>
<p>Real, bounded CPU inference for <code>SZL-Khipu-1.5B-GGUF</code>. Max 1,200 input characters and 32 generated tokens.</p>
<textarea id="prompt" maxlength="1200">Reply with one short sentence describing what a cryptographic receipt can prove.</textarea><br>
<button id="run" disabled aria-disabled="true">Run bounded inference</button><pre id="out" role="status" aria-live="polite">Checking runtime…</pre>
<p class="fine">Upstream receipt signatures verify against the repository-declared key only; no independent identity or key-ownership binding is claimed. No independent benchmark, post-quantization quality claim, or safety certification. Prompts are not intentionally stored. <a href="/api/v1/identity">Machine identity</a> · <a href="/health">Health</a></p></div></main>
<script>
const b=document.querySelector('#run'),o=document.querySelector('#out'),p=document.querySelector('#prompt');
let running=false,hasResult=false;
function renderStatus(status,failure){const ready=status==='READY';b.disabled=!ready||running;b.setAttribute('aria-disabled',String(b.disabled));if(running)return;if(ready){if(!hasResult)o.textContent='Runtime READY. You can run bounded inference.';return}hasResult=false;o.textContent=status==='FAILED'?'Runtime FAILED'+(failure?': '+failure:'')+'.':status==='STARTING'?'Runtime STARTING…':'Runtime status unavailable; retrying…'}
async function checkHealth(){try{const r=await fetch('/health',{cache:'no-store'}),j=await r.json();renderStatus(j.status,j.failure_code)}catch(_){renderStatus('UNAVAILABLE')}}
b.onclick=async()=>{if(b.disabled)return;running=true;hasResult=false;b.disabled=true;b.setAttribute('aria-disabled','true');o.textContent='Running on free CPU…';try{const r=await fetch('/api/v1/infer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:p.value,max_new_tokens:24})}),j=await r.json();hasResult=true;o.textContent=r.ok?j.output:JSON.stringify(j)}catch(e){hasResult=true;o.textContent='Request failed: '+e}finally{running=false;await checkHealth()}};
checkHealth();setInterval(checkHealth,5000);
</script></body></html>"""
