from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import platform
import threading
import time
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
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
RECEIPT_SHA256 = {
    "training_receipt.signed.json": (
        "7af76dd4f26dcd122012bfd1e47a0f55481a952b86aee28956cf7cfaaf59bd04"
    ),
    "eval_receipt.signed.json": (
        "32edd2d862fd5abac390bee3d30950f4718afedc41f4da4e24f3d0dfe67f8450"
    ),
    "owner_pubkey.json": (
        "843d0958392b4ee11ad8e36519261bebf841ee20caec479cbbc4bb9e8c991031"
    ),
}
MAX_INPUT_CHARS = 1_200
MAX_CHAT_MESSAGES = 12
MAX_PROMPT_TOKENS = 800
MAX_NEW_TOKENS = 32
INFERENCE_BUDGET_SECONDS = 45.0
BODY_READ_TIMEOUT_SECONDS = 10.0
DISKCACHE_ADVISORY = "CVE-2025-69872"
RESERVED_CHAT_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
SYSTEM_PROMPT = (
    "You are a bounded research demo. Answer briefly. If evidence is missing, "
    "say so; do not claim independent benchmarking or safety certification."
)
SOURCE_ROOT = Path(__file__).resolve().parent
# Vendored SZL Kanchay design system (byte-for-byte v1.0.0 export). The landing
# page links only these files. URL path -> (file under kanchay/, media type);
# nothing else below kanchay/ is served and no request value reaches a path.
KANCHAY_ROOT = SOURCE_ROOT / "kanchay"
KANCHAY_ASSETS: dict[str, tuple[str, str]] = {
    "/kanchay/kanchay.css": ("kanchay.css", "text/css; charset=utf-8"),
    "/kanchay/fonts/SpaceGrotesk-latin.woff2": (
        "fonts/SpaceGrotesk-latin.woff2",
        "font/woff2",
    ),
    "/kanchay/fonts/Inter-latin.woff2": ("fonts/Inter-latin.woff2", "font/woff2"),
    "/kanchay/fonts/JetBrainsMono-latin.woff2": (
        "fonts/JetBrainsMono-latin.woff2",
        "font/woff2",
    ),
}
SOURCE_REVISION_ENV = "SZL_GITHUB_SOURCE_REVISION"
ARTIFACT_ROOT = Path("/opt/szl/model-artifacts")
SERVICE_NAME = "szl-model-inference-lab"
SURFACE_NAME = "model-inference"


state: dict[str, Any] = {
    "status": "STARTING",
    "failure_code": None,
    "model_path": None,
    "model_sha256": None,
    "source_integrity": False,
    "receipt_status": "NOT_CHECKED",
    "llama_cpp_version": None,
    "prompt_cache_status": "NOT_CHECKED",
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
    if filename not in {MODEL_FILE, *RECEIPT_FILES}:
        raise RuntimeError("ARTIFACT_NOT_ALLOWLISTED")
    path = ARTIFACT_ROOT / filename
    if path.is_symlink():
        raise RuntimeError("ARTIFACT_SYMLINK_REJECTED")
    if not path.is_file():
        raise RuntimeError("ARTIFACT_NOT_REGULAR")
    return path


def verify_receipts() -> dict[str, Any]:
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    paths = {filename: artifact_path(filename) for filename in RECEIPT_FILES}
    for filename, path in paths.items():
        if sha256_file(path) != RECEIPT_SHA256[filename]:
            raise RuntimeError("RECEIPT_FILE_SHA256_MISMATCH")
    declared_key = json.loads(paths["owner_pubkey.json"].read_text(encoding="utf-8"))
    receipts: dict[str, dict[str, Any]] = {}
    for filename in RECEIPT_FILES[:2]:
        receipt = json.loads(paths[filename].read_text(encoding="utf-8"))
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


def enforce_prompt_cache_disabled(model: Any) -> None:
    """Fail closed if llama.cpp's optional pickle-backed cache is reachable.

    ``llama-cpp-python`` currently requires DiskCache even when applications do
    not use prompt caching. DiskCache through 5.6.3 has no patched release for
    CVE-2025-69872, so this public runtime must never attach any cache object.
    Requiring the upstream ``cache`` attribute to exist also turns a future
    dependency contract change into a startup failure instead of an assumption.
    """

    if not hasattr(model, "cache") or model.cache is not None:
        raise RuntimeError("PROMPT_CACHE_MUST_REMAIN_DISABLED")


def initialize() -> None:
    global llm
    state["status"] = "STARTING"
    state["failure_code"] = None
    state["prompt_cache_status"] = "NOT_CHECKED"
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
        enforce_prompt_cache_disabled(llm)
        state["prompt_cache_status"] = "DISABLED"
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
            "prompt_cache": {
                "status": state["prompt_cache_status"],
                "advisory": DISKCACHE_ADVISORY,
                "boundary": (
                    "llama-cpp-python requires DiskCache, but this runtime never "
                    "attaches a prompt cache and fails startup if one is present"
                ),
            },
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


def observed_source_revision() -> str | None:
    """Return only an exact deploy-bound Git revision.

    The Space repository SHA, release identifier, and model revision are all
    different identities.  None is allowed to stand in for the governed
    GitHub source revision.
    """

    raw_revision = os.getenv(SOURCE_REVISION_ENV, "").strip().lower()
    if len(raw_revision) != 40:
        return None
    if any(character not in "0123456789abcdef" for character in raw_revision):
        return None
    return raw_revision


def build_info_payload() -> dict[str, Any]:
    """Expose the exact Git source bound by the governed deploy workflow.

    The Hugging Face variable is non-secret. A missing or malformed revision
    stays UNKNOWN; it is never inferred from the Space repository revision or
    the internal release manifest.
    """

    revision = observed_source_revision()
    revision_observed = revision is not None
    return {
        "schema": "szl.build-info/v1",
        "service": SERVICE_NAME,
        "build": {
            "state": "OBSERVED" if revision_observed else "UNKNOWN",
            "revision": revision,
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


def version_payload() -> dict[str, Any]:
    revision = observed_source_revision()
    return {
        "schemaVersion": "szl.vertical-conformance.version.v1",
        "service": SERVICE_NAME,
        "surface": SURFACE_NAME,
        "gitSha": revision,
        "evidenceState": "MEASURED" if revision is not None else "UNAVAILABLE",
    }


def evidence_payload() -> dict[str, Any]:
    revision = observed_source_revision()
    receipts_verified = (
        state["receipt_status"] == "DECLARED_KEY_SIGNATURES_VALID"
    )
    return {
        "schemaVersion": "szl.vertical-conformance.evidence.v1",
        "service": SERVICE_NAME,
        "surface": SURFACE_NAME,
        "gitSha": revision,
        "evidenceState": (
            "MEASURED"
            if revision is not None and state["source_integrity"] and receipts_verified
            else "UNAVAILABLE"
        ),
        "runtime": {
            "status": state["status"],
            "ready": state["status"] == "READY",
            "sourceIntegrity": state["source_integrity"],
            "modelSha256Verified": state["model_sha256"] == MODEL_SHA256,
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
        },
        "receipts": [
            {
                "kind": "training",
                "status": state["receipt_status"],
                "canonicalSha256": (
                    state.get("receipt_evidence", {}).get(
                        "training_canonical_sha256"
                    )
                ),
                "scope": "repository-declared key continuity only",
            },
            {
                "kind": "evaluation",
                "status": state["receipt_status"],
                "canonicalSha256": (
                    state.get("receipt_evidence", {}).get("eval_canonical_sha256")
                ),
                "scope": "repository-declared key continuity only",
            },
        ],
        "outputProvenance": {
            "signatureStatus": "UNSIGNED",
            "authenticityEstablished": False,
            "record": "content-addressed and returned to the caller; not persisted",
        },
        "limitations": [
            "No independent benchmark or safety certification is claimed.",
            "Training and evaluation receipts do not cover this runtime output.",
            "The public Space is best-effort and has no service-level agreement.",
        ],
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


@app.get("/readyz")
def readyz() -> JSONResponse:
    return health()


@app.get("/live")
def live() -> JSONResponse:
    code = 503 if state["status"] == "FAILED" else 200
    return JSONResponse({"status": state["status"]}, status_code=code)


@app.get("/healthz")
def healthz() -> JSONResponse:
    return live()


@app.get("/version")
def version() -> JSONResponse:
    payload = version_payload()
    return JSONResponse(payload, status_code=200 if payload["gitSha"] else 503)


@app.get("/evidence")
def evidence() -> JSONResponse:
    payload = evidence_payload()
    return JSONResponse(
        payload,
        status_code=200 if payload["evidenceState"] == "MEASURED" else 503,
    )


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


def kanchay_asset(url_path: str) -> Response:
    """Return one allowlisted Kanchay file; release.json binds its bytes."""

    relative, media_type = KANCHAY_ASSETS[url_path]
    return Response((KANCHAY_ROOT / relative).read_bytes(), media_type=media_type)


@app.get("/kanchay/kanchay.css", include_in_schema=False)
def kanchay_stylesheet() -> Response:
    return kanchay_asset("/kanchay/kanchay.css")


@app.get("/kanchay/fonts/SpaceGrotesk-latin.woff2", include_in_schema=False)
def kanchay_display_font() -> Response:
    return kanchay_asset("/kanchay/fonts/SpaceGrotesk-latin.woff2")


@app.get("/kanchay/fonts/Inter-latin.woff2", include_in_schema=False)
def kanchay_sans_font() -> Response:
    return kanchay_asset("/kanchay/fonts/Inter-latin.woff2")


@app.get("/kanchay/fonts/JetBrainsMono-latin.woff2", include_in_schema=False)
def kanchay_mono_font() -> Response:
    return kanchay_asset("/kanchay/fonts/JetBrainsMono-latin.woff2")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en" data-screenshot-ready="false"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="description" content="A bounded, source-bound GGUF inference instrument with visible model, receipt, and unsigned-output boundaries.">
<meta property="og:type" content="website"><meta property="og:title" content="SZL Model Inference Lab - Khipu Loom">
<meta property="og:description" content="Every token leaves a thread: inspect exact model bytes, deploy identity, receipt evidence, and bounded execution together.">
<meta property="og:url" content="https://szlholdings-szl-model-inference-lab.hf.space/">
<link rel="canonical" href="https://szlholdings-szl-model-inference-lab.hf.space/">
<title>SZL Model Inference Lab - Khipu Loom</title>
<link rel="stylesheet" href="/kanchay/kanchay.css">
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;min-height:100vh;background:var(--color-a11oy-bg);color:var(--color-a11oy-text);font:16px/1.55 var(--font-sans);-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
::selection{background:color-mix(in srgb,var(--color-a11oy-gold) 22%,transparent);color:var(--color-a11oy-text)}a{color:inherit}:focus-visible{outline:var(--border-focus) solid var(--color-focus);outline-offset:2px}
.skip{position:fixed;left:var(--space-4);top:-80px;z-index:var(--z-toast);display:inline-flex;align-items:center;min-height:var(--szl-touch-target);padding:0 var(--space-4);border:var(--border-hairline) solid transparent;border-radius:var(--radius-sm);background:var(--color-a11oy-gold);color:var(--color-on-accent);font-weight:600;text-decoration:none}.skip:focus{top:var(--space-4)}
.shell{width:min(1180px,calc(100% - 2 * var(--szl-theatre-gutter)));margin:auto;position:relative}.nav{display:flex;align-items:center;justify-content:space-between;gap:var(--space-6);padding:var(--space-4) 0;border-bottom:var(--border-hairline) solid var(--color-a11oy-border-subtle)}.brand{display:flex;align-items:center;gap:var(--space-3);min-height:var(--szl-touch-target);border-radius:var(--radius-sm);text-decoration:none}
.mark{position:relative;flex:none;width:34px;height:34px;border:var(--border-hairline) solid var(--color-a11oy-gold);border-radius:var(--radius-full)}.mark:before,.mark:after{content:"";position:absolute}.mark:before{left:16px;top:-7px;height:46px;border-left:var(--border-hairline) solid var(--color-ink-signal)}.mark:after{left:-7px;top:16px;width:46px;border-top:var(--border-hairline) solid var(--color-ink-signal)}
.brand strong{font-family:var(--font-display);font-weight:600;letter-spacing:.06em}.brand small{display:block;color:var(--color-a11oy-text-ghost);font:500 10px/1.2 var(--font-mono);letter-spacing:.18em;text-transform:uppercase}.links{display:flex;gap:var(--space-2);align-items:center}.links a{display:inline-flex;align-items:center;min-height:var(--szl-touch-target);padding:0 var(--space-3);border-radius:var(--radius-md);color:var(--color-a11oy-text-sub);font-size:13px;font-weight:500;text-decoration:none;transition:background-color var(--duration-fast) var(--ease-standard),color var(--duration-fast) var(--ease-standard)}.links a:hover{background:var(--color-a11oy-overlay);color:var(--color-a11oy-text)}
.hero{position:relative;display:grid;grid-template-columns:minmax(0,1.02fr) minmax(360px,.98fr);gap:var(--space-12);align-items:center;padding:var(--space-20) 0 var(--space-12)}.hero:before{content:"";position:absolute;inset:0;pointer-events:none;background-image:linear-gradient(color-mix(in srgb,var(--color-a11oy-text) 2.5%,transparent) 1px,transparent 1px),linear-gradient(90deg,color-mix(in srgb,var(--color-a11oy-text) 2.5%,transparent) 1px,transparent 1px);background-size:56px 56px;-webkit-mask-image:radial-gradient(ellipse at center,var(--color-a11oy-bg) 30%,transparent 75%);mask-image:radial-gradient(ellipse at center,var(--color-a11oy-bg) 30%,transparent 75%)}.hero>*{position:relative}
.eyebrow,.mono{font-family:var(--font-mono)}.eyebrow{color:var(--color-a11oy-gold);font-size:11px;font-weight:500;line-height:16px;letter-spacing:.2em;text-transform:uppercase}.hero h1{margin:var(--space-5) 0 var(--space-6);max-width:760px;font-family:var(--font-display);font-size:clamp(2.75rem,6.4vw,5.2rem);font-weight:300;line-height:1.02;letter-spacing:-.035em;text-wrap:balance}.hero h1 span{background:linear-gradient(120deg,var(--color-a11oy-text) 20%,var(--color-a11oy-gold) 90%);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.lede{max-width:660px;color:var(--color-a11oy-text-sub);font-size:clamp(17px,1.6vw,20px);line-height:1.65;text-wrap:pretty}.hero-actions{display:flex;flex-wrap:wrap;gap:var(--space-3);margin-top:var(--space-8)}.button,.ghost{display:inline-flex;align-items:center;justify-content:center;min-height:var(--szl-touch-target);padding:0 var(--space-6);border:var(--border-hairline) solid transparent;border-radius:var(--radius-sm);font:600 15px/20px var(--font-sans);text-decoration:none;transition:background-color var(--duration-fast) var(--ease-standard),border-color var(--duration-fast) var(--ease-standard)}
.button{background:var(--color-a11oy-gold);color:var(--color-on-accent);box-shadow:var(--shadow-sm);cursor:pointer}.button:not(:disabled):hover{background:var(--gold-bright)}.button:disabled{cursor:not-allowed;opacity:.5;box-shadow:none}.ghost{background:transparent;color:var(--color-a11oy-gold);border-color:var(--gold-line)}.ghost:hover{background:color-mix(in srgb,var(--color-a11oy-gold) 8%,transparent);border-color:color-mix(in srgb,var(--color-a11oy-gold) 35%,transparent)}
.loom{position:relative;min-height:470px;overflow:hidden;border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);background:var(--color-a11oy-deep)}.loom:before{content:"";position:absolute;inset:var(--space-6);border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-sm)}.cord{position:absolute;left:13%;right:13%;height:0;border-top:var(--border-hairline) solid var(--teal-line);transform-origin:center}.cord.c1{top:23%;transform:rotate(14deg)}.cord.c2{top:43%;transform:rotate(-9deg)}.cord.c3{top:65%;transform:rotate(5deg)}.cord.c4{top:78%;transform:rotate(-15deg)}
.knot{position:absolute;width:18px;height:18px;border:var(--border-thick) solid var(--color-a11oy-gold);border-radius:var(--radius-full);background:var(--color-a11oy-deep)}.k1{left:24%;top:29%}.k2{left:52%;top:39%;border-color:var(--color-ink-signal)}.k3{left:72%;top:58%;border-color:var(--color-a11oy-text-sub)}.k4{left:34%;top:70%}
.loom-label{position:absolute;padding:var(--space-2) var(--space-3);border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-sm);background:color-mix(in srgb,var(--color-a11oy-bg) 86%,transparent);font:500 10px/1.2 var(--font-mono);letter-spacing:.15em;text-transform:uppercase}.l1{left:9%;top:12%;color:var(--color-ink-signal)}.l2{right:8%;top:33%;color:var(--color-a11oy-gold)}.l3{left:12%;bottom:13%;color:var(--color-a11oy-text-sub)}
.loom-core{position:absolute;left:50%;top:50%;display:grid;place-items:center;width:130px;height:130px;transform:translate(-50%,-50%);border:var(--border-hairline) solid var(--color-a11oy-border);border-radius:var(--radius-full);background:var(--color-a11oy-surface);text-align:center}.loom-core b{font:600 12px/1.2 var(--font-mono);letter-spacing:.15em}.loom-core small{display:block;margin-top:var(--space-1);color:var(--color-a11oy-text-sub)}
.status-strip{display:grid;grid-template-columns:repeat(4,1fr);margin:var(--space-1) 0 var(--space-16);overflow:hidden;border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);background:var(--color-a11oy-surface)}.metric{padding:var(--space-5);border-right:var(--border-hairline) solid var(--color-a11oy-border-subtle)}.metric:last-child{border:0}.metric b{display:block;margin-top:var(--space-1);font:500 14px/1.25 var(--font-mono);font-variant-numeric:tabular-nums;word-break:break-word}.metric small{color:var(--color-a11oy-text-sub);font:500 11px/16px var(--font-mono);letter-spacing:.15em;text-transform:uppercase}
.state,.metric .state{display:flex;align-items:center;gap:var(--space-2)}.state:before{content:"";flex:none;width:8px;height:8px;border-radius:var(--radius-full);background:var(--color-warning)}.state.ready{color:var(--color-ink-signal)}.state.ready:before{background:var(--color-success)}.state.failed{color:var(--color-ink-danger)}.state.failed:before{background:var(--color-ink-danger)}
.section{padding:var(--space-16) 0;border-top:var(--border-hairline) solid var(--color-a11oy-border)}.section-head{display:flex;align-items:end;justify-content:space-between;gap:var(--space-8);margin-bottom:var(--space-8)}.section h2{margin:var(--space-2) 0;font-family:var(--font-display);font-size:clamp(2rem,4.4vw,3.25rem);font-weight:300;line-height:1.05;letter-spacing:-.03em;text-wrap:balance}.section-copy{max-width:600px;color:var(--color-a11oy-text-sub);text-wrap:pretty}
.composer{display:grid;grid-template-columns:1.08fr .92fr;overflow:hidden;border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);background:var(--color-a11oy-surface)}.input-pane,.output-pane{padding:var(--space-6)}.output-pane{border-left:var(--border-hairline) solid var(--color-a11oy-border-subtle);background:var(--color-a11oy-deep)}label{display:block;margin-bottom:var(--space-2);font-weight:600}
textarea{width:100%;min-height:180px;resize:vertical;padding:var(--space-4);border:var(--border-hairline) solid var(--color-control-border);border-radius:var(--radius-sm);background:var(--color-a11oy-deep);color:var(--color-a11oy-text);box-shadow:var(--shadow-sm);font:15px/1.55 var(--font-mono);transition:border-color var(--duration-fast) var(--ease-standard)}textarea:focus-visible{border-color:var(--color-focus)}
.composer-meta{display:flex;justify-content:space-between;gap:var(--space-3);margin:var(--space-2) 2px;color:var(--color-a11oy-text-sub);font-size:12px;font-variant-numeric:tabular-nums}.run-row{display:flex;align-items:center;justify-content:space-between;gap:var(--space-4);margin-top:var(--space-4)}.output-label{color:var(--color-a11oy-text-ghost);font:500 11px/16px var(--font-mono);letter-spacing:.15em;text-transform:uppercase}.output-pane pre{min-height:220px;margin:var(--space-4) 0 0;white-space:pre-wrap;color:var(--color-a11oy-text);font:15px/1.7 var(--font-mono)}.thread-id{color:var(--color-a11oy-text-sub);font-size:11px;word-break:break-all}
.evidence-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:var(--space-4)}.evidence-card{min-height:220px;padding:var(--space-5);border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);background:var(--color-a11oy-surface);transition:border-color var(--duration-fast) var(--ease-standard)}.evidence-card:hover{border-color:color-mix(in srgb,var(--color-a11oy-gold) 20%,transparent)}.evidence-card .number{color:var(--color-a11oy-gold);font:500 11px/16px var(--font-mono);letter-spacing:.15em;text-transform:uppercase}
.evidence-card h3{margin:var(--space-10) 0 var(--space-2);font-family:var(--font-display);font-size:20px;font-weight:600;letter-spacing:-.02em}.evidence-card p{color:var(--color-a11oy-text-sub);font-size:14px}.evidence-card a{display:inline-flex;align-items:center;min-height:var(--szl-touch-target);color:var(--color-ink-signal);font:500 13px/1.2 var(--font-mono);text-underline-offset:4px}
.boundary{display:grid;grid-template-columns:1fr 1fr;gap:var(--space-4)}.boundary article{padding:var(--space-6);border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);background:var(--color-a11oy-surface)}.boundary article:first-child{border-color:var(--teal-line);background:var(--teal-soft)}.boundary h3{margin-top:0;font-family:var(--font-display);font-size:20px;font-weight:600;letter-spacing:-.02em}.boundary ul{padding-left:var(--space-5);color:var(--color-a11oy-text-sub)}footer{display:flex;justify-content:space-between;gap:var(--space-6);padding:var(--space-10) 0 var(--space-12);border-top:var(--border-hairline) solid var(--color-a11oy-border);color:var(--color-a11oy-text-ghost);font-size:12px}
@media(max-width:850px){.links{display:none}.hero{grid-template-columns:1fr;padding-top:var(--space-12)}.loom{min-height:390px}.status-strip{grid-template-columns:1fr 1fr}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:var(--border-hairline) solid var(--color-a11oy-border-subtle)}.composer{grid-template-columns:1fr}.output-pane{border-left:0;border-top:var(--border-hairline) solid var(--color-a11oy-border-subtle)}.evidence-grid{grid-template-columns:1fr}.boundary{grid-template-columns:1fr}.section-head{display:block}}@media(max-width:520px){.hero{gap:var(--space-8)}.loom{min-height:330px}.loom-core{width:104px;height:104px}.l2{top:22%}.status-strip{grid-template-columns:1fr}.metric{border-right:0;border-bottom:var(--border-hairline) solid var(--color-a11oy-border-subtle)!important}.metric:last-child{border-bottom:0!important}.input-pane,.output-pane{padding:var(--space-5)}.run-row{align-items:stretch;flex-direction:column}.button{width:100%}footer{display:block}}
@media(pointer:coarse){.skip,.brand,.links a,.button,.ghost,.evidence-card a{min-height:var(--szl-touch-target-coarse)}}@media(forced-colors:active){.skip,.button,.ghost{border-color:CanvasText}.hero h1 span{background:none;-webkit-text-fill-color:currentColor}.state:before,.state.ready:before,.state.failed:before{forced-color-adjust:none;background:CanvasText}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*:before,*:after{animation:none!important;transition:none!important}}
</style></head>
<body><a class="skip" href="#main">Skip to inference lab</a><div class="shell">
<nav class="nav" aria-label="Primary"><a class="brand" href="/"><span class="mark" aria-hidden="true"></span><span><strong>SZL / KHIPU LOOM</strong><small>Bounded inference instrument</small></span></a><div class="links"><a href="#run-lab">Run</a><a href="#evidence">Evidence</a><a href="/api/v1/identity">Identity</a><a href="/.well-known/szl-inference-contract.json">API contract</a></div></nav>
<main id="main"><section class="hero"><div><p class="eyebrow">Formula genome 01 / public cpu instrument</p><h1>Every token leaves a <span>thread.</span></h1><p class="lede">A bounded inference surface where the exact model bytes, source revision, receipt chain, limits, and missing guarantees remain visible together.</p><div class="hero-actions"><a class="button" href="#run-lab">Enter the loom</a><a class="ghost" href="/evidence">Inspect machine evidence</a></div></div>
<div class="loom" aria-label="Abstract Khipu provenance loom showing source, model, and receipt threads"><i class="cord c1"></i><i class="cord c2"></i><i class="cord c3"></i><i class="cord c4"></i><i class="knot k1"></i><i class="knot k2"></i><i class="knot k3"></i><i class="knot k4"></i><span class="loom-label l1">source / exact sha</span><span class="loom-label l2">model / q4_k_m</span><span class="loom-label l3">output / unsigned</span><div class="loom-core"><div><b>KHIPU<br>1.5B</b><small>CPU bound</small></div></div></div></section>
<section class="status-strip" aria-label="Live runtime summary"><div class="metric"><small>Runtime</small><b id="runtime-state" class="state">CHECKING</b></div><div class="metric"><small>Git source</small><b id="source-sha">UNAVAILABLE</b></div><div class="metric"><small>Model pin</small><b>67d60ec...f4a4b5d25</b></div><div class="metric"><small>Output proof</small><b>UNSIGNED / HASHED</b></div></section>
<section class="section" id="run-lab"><div class="section-head"><div><p class="eyebrow">Bounded generation</p><h2>Pull one thread.</h2></div><p class="section-copy">One request at a time. At most 1,200 characters, 800 formatted prompt tokens, and 32 generated tokens. Greedy decoding. No tools, streaming, or hidden fallback.</p></div>
<div class="composer"><div class="input-pane"><label for="prompt">Prompt</label><textarea id="prompt" maxlength="1200">Reply with one short sentence describing what a cryptographic receipt can prove.</textarea><div class="composer-meta"><span id="char-count">0 / 1,200 characters</span><span>24 output tokens</span></div><div class="run-row"><button class="button" id="run" disabled aria-disabled="true">Run bounded inference</button><span class="mono thread-id" id="request-state">Waiting for readiness</span></div></div><div class="output-pane"><span class="output-label">Model output</span><pre id="out" role="status" aria-live="polite">Checking the runtime and evidence threads...</pre></div></div></section>
<section class="section" id="evidence"><div class="section-head"><div><p class="eyebrow">Inspectable by default</p><h2>The evidence bay.</h2></div><p class="section-copy">The interface does not turn provenance into decoration. Every status below resolves to a machine-readable surface.</p></div><div class="evidence-grid"><article class="evidence-card"><span class="number">01 / SOURCE</span><h3>Exact deploy identity</h3><p id="source-detail">The Git revision must come from the governed Space variable. Missing or malformed identity fails closed.</p><a href="/version">Open /version</a></article><article class="evidence-card"><span class="number">02 / RECEIPTS</span><h3>Declared-key chain</h3><p id="receipt-detail">Training and evaluation receipts are verified against the repository-declared key and canonical hash chain.</p><a href="/evidence">Open /evidence</a></article><article class="evidence-card"><span class="number">03 / RUNTIME</span><h3>Exact model bytes</h3><p>Q4_K_M GGUF bytes are fetched at image build from an immutable revision, size-checked, hash-checked, and loaded offline.</p><a href="/api/v1/identity">Open identity</a></article></div></section>
<section class="section"><div class="boundary"><article><p class="eyebrow">What is measured</p><h3>Internal integrity and bounded execution</h3><ul><li>Exact GGUF revision, size, and SHA-256</li><li>Exact deploy-bound Git revision when configured</li><li>Declared-key training/evaluation receipt chain</li><li>Runtime readiness and deterministic request limits</li></ul></article><article><p class="eyebrow">What is not claimed</p><h3>Authenticity and quality remain bounded</h3><ul><li>No independent identity or key-ownership binding</li><li>No post-quantization quality or safety certification</li><li>No signed output or reproducible execution claim</li><li>No provider SLA, sensitive-data handling guarantee, or autonomy</li></ul></article></div></section></main>
<footer><span>SZL Holdings / Khipu Loom / Apache-2.0</span><span class="mono">MEASURED, REPORTED, UNKNOWN, or UNAVAILABLE - never implied</span></footer></div>
<script>
const b=document.querySelector('#run'),o=document.querySelector('#out'),p=document.querySelector('#prompt'),root=document.documentElement;
const runtimeState=document.querySelector('#runtime-state'),sourceSha=document.querySelector('#source-sha'),requestState=document.querySelector('#request-state'),charCount=document.querySelector('#char-count'),sourceDetail=document.querySelector('#source-detail'),receiptDetail=document.querySelector('#receipt-detail');
let running=false,hasResult=false;
function short(value){return typeof value==='string'&&value.length>12?value.slice(0,7)+'...'+value.slice(-7):value||'UNAVAILABLE'}
function countChars(){charCount.textContent=p.value.length.toLocaleString()+' / 1,200 characters'}
function renderStatus(status,failure){const ready=status==='READY';runtimeState.textContent=status||'UNAVAILABLE';runtimeState.className='state '+(ready?'ready':status==='FAILED'?'failed':'');b.disabled=!ready||running;b.setAttribute('aria-disabled',String(b.disabled));root.dataset.screenshotReady=String(ready);if(running)return;if(!hasResult)requestState.textContent=ready?'Runtime ready':status==='FAILED'?'Runtime failed':'Warming exact model bytes';if(ready){if(!hasResult)o.textContent='Runtime READY. Pull a thread when you are ready.';return}if(!hasResult)o.textContent=status==='FAILED'?'Runtime FAILED'+(failure?': '+failure:'')+'.':status==='STARTING'?'Runtime STARTING. Verifying exact model bytes and receipts...':'Runtime status unavailable; retrying...'}
async function getJson(path){const r=await fetch(path,{cache:'no-store'});let j={};try{j=await r.json()}catch(_){j={}}return{ok:r.ok,status:r.status,json:j}}
async function refresh(){const [health,version,evidence]=await Promise.allSettled([getJson('/health'),getJson('/version'),getJson('/evidence')]);if(health.status==='fulfilled')renderStatus(health.value.json.status,health.value.json.failure_code);else renderStatus('UNAVAILABLE');if(version.status==='fulfilled'){const sha=version.value.json.gitSha;sourceSha.textContent=short(sha);sourceDetail.textContent=sha?'Governed Git source '+sha+' is bound to this deployment.':'Exact governed Git source is unavailable; /version fails closed.'}if(evidence.status==='fulfilled'){const receipts=evidence.value.json.receipts||[];const verified=receipts.filter(x=>x.status==='DECLARED_KEY_SIGNATURES_VALID').length;receiptDetail.textContent=verified===2?'Two declared-key receipts are visible and chain-verified. Runtime outputs remain unsigned.':'Receipt evidence is not yet available; no green state is inferred.'}}
p.addEventListener('input',countChars);b.addEventListener('click',async()=>{if(b.disabled)return;running=true;hasResult=false;b.disabled=true;b.setAttribute('aria-disabled','true');requestState.textContent='Inference in progress';o.textContent='Pulling the bounded model thread on free CPU...';try{const r=await fetch('/api/v1/infer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:p.value,max_new_tokens:24})});const j=await r.json();hasResult=true;o.textContent=r.ok?j.output:JSON.stringify(j);requestState.textContent=r.ok?'Completed / output unsigned':'Request refused / '+r.status}catch(_){hasResult=true;o.textContent='Request unavailable. No result was fabricated.';requestState.textContent='Network unavailable'}finally{running=false;await refresh()}});
countChars();refresh();setInterval(refresh,5000);
</script></body></html>"""
