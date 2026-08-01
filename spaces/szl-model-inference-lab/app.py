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
RESERVED_CHAT_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
SYSTEM_PROMPT = (
    "You are a bounded research demo. Answer briefly. If evidence is missing, "
    "say so; do not claim independent benchmarking or safety certification."
)
SOURCE_ROOT = Path(__file__).resolve().parent
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en" data-screenshot-ready="false"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0c0811"><meta name="color-scheme" content="dark">
<meta name="description" content="A bounded, source-bound GGUF inference instrument with visible model, receipt, and unsigned-output boundaries.">
<meta property="og:type" content="website"><meta property="og:title" content="SZL Model Inference Lab - Khipu Loom">
<meta property="og:description" content="Every token leaves a thread: inspect exact model bytes, deploy identity, receipt evidence, and bounded execution together.">
<meta property="og:url" content="https://szlholdings-szl-model-inference-lab.hf.space/">
<link rel="canonical" href="https://szlholdings-szl-model-inference-lab.hf.space/">
<title>SZL Model Inference Lab - Khipu Loom</title>
<style>
:root{--ink:#0c0811;--ink-2:#120d19;--panel:#171020;--line:#3b2a45;--text:#f7f0f4;--muted:#bcaebc;--fiber:#ef86d7;--ember:#ffbd69;--mint:#63e6d4;--danger:#ff6f82;--radius:22px;--shadow:0 30px 90px #0009}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 78% 12%,#301733 0,transparent 34%),radial-gradient(circle at 7% 78%,#172b30 0,transparent 30%),var(--ink);color:var(--text);font:16px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.2;background-image:linear-gradient(#fff1 1px,transparent 1px),linear-gradient(90deg,#fff1 1px,transparent 1px);background-size:48px 48px;mask-image:linear-gradient(to bottom,#000,transparent 82%)}
a{color:inherit}.skip{position:fixed;left:16px;top:-60px;background:var(--text);color:var(--ink);padding:10px 14px;z-index:20}.skip:focus{top:16px}
.shell{width:min(1180px,calc(100% - 40px));margin:auto;position:relative}.nav{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:22px 0;border-bottom:1px solid #ffffff14}.brand{display:flex;align-items:center;gap:12px;text-decoration:none}.mark{width:34px;height:34px;border:1px solid var(--fiber);border-radius:50%;position:relative;box-shadow:inset 0 0 18px #ef86d744}.mark:before,.mark:after{content:"";position:absolute;background:var(--mint)}.mark:before{width:1px;height:46px;left:16px;top:-7px}.mark:after{width:46px;height:1px;left:-7px;top:16px}.brand strong{letter-spacing:.08em}.brand small{display:block;color:var(--muted);font:700 10px/1.2 ui-monospace,monospace;letter-spacing:.18em;text-transform:uppercase}.links{display:flex;gap:22px;align-items:center}.links a{color:var(--muted);font-size:13px;text-decoration:none}.links a:hover{color:var(--text)}
.hero{display:grid;grid-template-columns:minmax(0,1.02fr) minmax(360px,.98fr);gap:58px;align-items:center;padding:82px 0 50px}.eyebrow,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.eyebrow{color:var(--mint);font-size:12px;font-weight:700;letter-spacing:.18em;text-transform:uppercase}.hero h1{font-size:clamp(52px,7vw,92px);line-height:.94;letter-spacing:-.06em;margin:20px 0 26px;max-width:760px}.hero h1 span{color:var(--ember);font-family:Georgia,serif;font-style:italic;font-weight:500}.lede{font-size:clamp(18px,2vw,22px);color:var(--muted);max-width:660px}.hero-actions{display:flex;flex-wrap:wrap;gap:12px;margin-top:30px}.button,.ghost{display:inline-flex;align-items:center;justify-content:center;min-height:48px;padding:0 20px;border-radius:999px;font-weight:800;text-decoration:none}.button{border:0;background:var(--text);color:var(--ink);cursor:pointer}.button:hover{background:var(--ember)}.button:disabled{cursor:not-allowed;opacity:.48}.ghost{border:1px solid var(--line);color:var(--text);background:#ffffff08}.ghost:hover{border-color:var(--fiber)}
.loom{min-height:470px;border:1px solid #ffffff1b;border-radius:32px;background:linear-gradient(145deg,#ffffff0d,#ffffff03);box-shadow:var(--shadow);position:relative;overflow:hidden}.loom:before{content:"";position:absolute;inset:28px;border:1px solid #ffffff12;border-radius:24px}.cord{position:absolute;left:13%;right:13%;height:1px;transform-origin:center;background:linear-gradient(90deg,transparent,var(--fiber),var(--ember),var(--mint),transparent);box-shadow:0 0 14px #ef86d766}.cord.c1{top:23%;transform:rotate(14deg)}.cord.c2{top:43%;transform:rotate(-9deg)}.cord.c3{top:65%;transform:rotate(5deg)}.cord.c4{top:78%;transform:rotate(-15deg)}.knot{position:absolute;width:18px;height:18px;border-radius:50%;background:var(--panel);border:3px solid var(--ember);box-shadow:0 0 0 7px #ffbd6917,0 0 28px #ffbd6955}.k1{left:24%;top:29%}.k2{left:52%;top:39%;border-color:var(--fiber)}.k3{left:72%;top:58%;border-color:var(--mint)}.k4{left:34%;top:70%}.loom-label{position:absolute;padding:9px 12px;border:1px solid var(--line);border-radius:12px;background:#0c0811db;font:700 10px/1.2 ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase}.l1{left:9%;top:12%;color:var(--fiber)}.l2{right:8%;top:33%;color:var(--ember)}.l3{left:12%;bottom:13%;color:var(--mint)}.loom-core{position:absolute;left:50%;top:50%;width:130px;height:130px;transform:translate(-50%,-50%);border:1px solid #ffffff20;border-radius:50%;display:grid;place-items:center;background:#100b17cc;box-shadow:0 0 0 22px #ffffff05,0 0 80px #ef86d72b;text-align:center}.loom-core b{font:700 12px/1.2 ui-monospace,monospace;letter-spacing:.13em}.loom-core small{display:block;color:var(--muted);margin-top:5px}
.status-strip{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:20px;background:#0c0811b8;overflow:hidden;margin:4px 0 72px}.metric{padding:20px 22px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font:800 14px/1.25 ui-monospace,monospace;word-break:break-word}.metric small{color:var(--muted);font-size:11px;letter-spacing:.1em;text-transform:uppercase}.state{display:inline-flex;align-items:center;gap:8px}.state:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--ember);box-shadow:0 0 14px currentColor}.state.ready{color:var(--mint)}.state.ready:before{background:var(--mint)}.state.failed{color:var(--danger)}.state.failed:before{background:var(--danger)}
.section{padding:72px 0;border-top:1px solid #ffffff12}.section-head{display:flex;align-items:end;justify-content:space-between;gap:30px;margin-bottom:30px}.section h2{font-size:clamp(34px,5vw,58px);line-height:1;margin:10px 0;letter-spacing:-.04em}.section-copy{color:var(--muted);max-width:600px}.composer{display:grid;grid-template-columns:1.08fr .92fr;border:1px solid var(--line);border-radius:28px;overflow:hidden;background:var(--panel);box-shadow:var(--shadow)}.input-pane,.output-pane{padding:28px}.output-pane{background:#0a0710;border-left:1px solid var(--line)}label{display:block;font-weight:800;margin-bottom:10px}textarea{box-sizing:border-box;width:100%;min-height:180px;resize:vertical;background:#0b0710;color:var(--text);border:1px solid #4a3554;border-radius:16px;padding:16px;font:15px/1.55 ui-monospace,monospace}textarea:focus{outline:2px solid var(--fiber);outline-offset:2px}.composer-meta{display:flex;justify-content:space-between;gap:14px;color:var(--muted);font-size:12px;margin:10px 2px}.run-row{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-top:18px}.output-label{color:var(--mint);font:700 11px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase}.output-pane pre{white-space:pre-wrap;margin:18px 0 0;min-height:220px;color:#e8dfe8;font:15px/1.7 ui-monospace,monospace}.thread-id{color:var(--muted);font-size:11px;word-break:break-all}
.evidence-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.evidence-card{min-height:220px;border:1px solid var(--line);border-radius:20px;background:#ffffff08;padding:22px}.evidence-card .number{color:var(--ember);font:700 12px ui-monospace,monospace}.evidence-card h3{font-size:20px;margin:40px 0 10px}.evidence-card p{color:var(--muted);font-size:14px}.evidence-card a{color:var(--mint)}
.boundary{display:grid;grid-template-columns:1fr 1fr;gap:16px}.boundary article{padding:26px;border-radius:22px;border:1px solid var(--line)}.boundary article:first-child{background:#17302e55}.boundary article:last-child{background:#34182255}.boundary h3{margin-top:0}.boundary ul{padding-left:20px;color:var(--muted)}footer{display:flex;justify-content:space-between;gap:24px;padding:38px 0 56px;color:var(--muted);font-size:12px;border-top:1px solid #ffffff12}
:focus-visible{outline:2px solid var(--mint);outline-offset:3px}@media(max-width:850px){.links{display:none}.hero{grid-template-columns:1fr;padding-top:56px}.loom{min-height:390px}.status-strip{grid-template-columns:1fr 1fr}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}.composer{grid-template-columns:1fr}.output-pane{border-left:0;border-top:1px solid var(--line)}.evidence-grid{grid-template-columns:1fr}.boundary{grid-template-columns:1fr}.section-head{display:block}}@media(max-width:520px){.shell{width:min(100% - 24px,1180px)}.hero{gap:34px}.hero h1{font-size:52px}.loom{min-height:330px}.status-strip{grid-template-columns:1fr}.metric{border-right:0;border-bottom:1px solid var(--line)!important}.metric:last-child{border-bottom:0!important}.input-pane,.output-pane{padding:20px}.run-row{align-items:stretch;flex-direction:column}.button{width:100%}footer{display:block}}
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
function renderStatus(status,failure){const ready=status==='READY';runtimeState.textContent=status||'UNAVAILABLE';runtimeState.className='state '+(ready?'ready':status==='FAILED'?'failed':'');b.disabled=!ready||running;b.setAttribute('aria-disabled',String(b.disabled));root.dataset.screenshotReady=String(ready);requestState.textContent=ready?'Runtime ready':status==='FAILED'?'Runtime failed':'Warming exact model bytes';if(running)return;if(ready){if(!hasResult)o.textContent='Runtime READY. Pull a thread when you are ready.';return}hasResult=false;o.textContent=status==='FAILED'?'Runtime FAILED'+(failure?': '+failure:'')+'.':status==='STARTING'?'Runtime STARTING. Verifying exact model bytes and receipts...':'Runtime status unavailable; retrying...'}
async function getJson(path){const r=await fetch(path,{cache:'no-store'});let j={};try{j=await r.json()}catch(_){j={}}return{ok:r.ok,status:r.status,json:j}}
async function refresh(){const [health,version,evidence]=await Promise.allSettled([getJson('/health'),getJson('/version'),getJson('/evidence')]);if(health.status==='fulfilled')renderStatus(health.value.json.status,health.value.json.failure_code);else renderStatus('UNAVAILABLE');if(version.status==='fulfilled'){const sha=version.value.json.gitSha;sourceSha.textContent=short(sha);sourceDetail.textContent=sha?'Governed Git source '+sha+' is bound to this deployment.':'Exact governed Git source is unavailable; /version fails closed.'}if(evidence.status==='fulfilled'){const receipts=evidence.value.json.receipts||[];const verified=receipts.filter(x=>x.status==='DECLARED_KEY_SIGNATURES_VALID').length;receiptDetail.textContent=verified===2?'Two declared-key receipts are visible and chain-verified. Runtime outputs remain unsigned.':'Receipt evidence is not yet available; no green state is inferred.'}}
p.addEventListener('input',countChars);b.addEventListener('click',async()=>{if(b.disabled)return;running=true;hasResult=false;b.disabled=true;b.setAttribute('aria-disabled','true');requestState.textContent='Inference in progress';o.textContent='Pulling the bounded model thread on free CPU...';try{const r=await fetch('/api/v1/infer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:p.value,max_new_tokens:24})});const j=await r.json();hasResult=true;o.textContent=r.ok?j.output:JSON.stringify(j);requestState.textContent=r.ok?'Completed / output unsigned':'Request refused / '+r.status}catch(_){hasResult=true;o.textContent='Request unavailable. No result was fabricated.';requestState.textContent='Network unavailable'}finally{running=false;await refresh()}});
countChars();refresh();setInterval(refresh,5000);
</script></body></html>"""
