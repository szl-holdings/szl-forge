"""Bounded public ASGI adapter for the unchanged Foundation v0.4 release.

Copyright SZL Holdings LLC. SPDX-License-Identifier: Apache-2.0
The adapter executes exploratory synthetic trials, never training or promotion.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
import time
from typing import Any, Callable
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03"
MANIFEST_SHA256 = "03a13779b09f2e8ad3dd53d928ac460a395d329742f9f43f7e5892fba672c877"
PUBLIC_HOST = "szlholdings-szl-foundation-confirmation.hf.space"
PUBLIC_ORIGIN = "https://" + PUBLIC_HOST
FAMILIES = ("clean", "shared_bias", "independent_outliers", "reference_outliers", "bias_drift", "compositional")
POLICIES = ("no_confirmation", "independent_bias", "source_aware", "cheap_only", "reference_only", "random_mixed", "learned")
MODEL_SEEDS = (17, 23, 41)
MAX_BODY = 4096
BODY_DEADLINE = 10.0
MAX_RECEIPT_BYTES = 65536
MAX_RECEIPTS = 128
RECEIPT_TTL = 86400
RATE_PER_MINUTE = 12
RATE_BURST = 2


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_identity() -> dict:
    revision = os.getenv("SZL_GITHUB_SOURCE_REVISION", "")
    observed = bool(re.fullmatch(r"[0-9a-f]{40}", revision))
    return {"state": "OBSERVED" if observed else "UNKNOWN", "repository": "szl-holdings/szl-forge",
            "revision": revision if observed else None}


def json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def decode(raw: bytes):
    def nonfinite(_):
        raise ValueError("Nonfinite JSON")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=json_object, parse_constant=nonfinite)


def validate_request(value):
    if not isinstance(value, dict) or set(value) != {"seed", "index", "family", "policy", "model_seed"}:
        raise ValueError("Provide exactly seed, index, family, policy and model_seed")
    for name in ("seed", "index"):
        if type(value[name]) is not int or not 0 <= value[name] <= 2**32 - 1:
            raise ValueError("World seed and episode index must be unsigned 32-bit integers")
    if type(value["model_seed"]) is not int or value["model_seed"] not in MODEL_SEEDS:
        raise ValueError("Select checkpoint 17, 23 or 41")
    if type(value["family"]) is not str or value["family"] not in FAMILIES:
        raise ValueError("Unknown observation environment")
    if type(value["policy"]) is not str or value["policy"] not in POLICIES:
        raise ValueError("Unknown decision policy")
    return dict(value)


def release_files(archive: Path) -> dict[str, bytes]:
    """Admit all regular bytes before importing any archived Python source."""
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Frozen archive digest mismatch")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if len(names) != 71 or len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
            raise ValueError("Unexpected or duplicate release members")
        if sum(info.file_size for info in infos) > 128 * 1024 * 1024:
            raise ValueError("Release expansion limit exceeded")
        for info in infos:
            name = info.filename
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or str(path) != name or "\\" in name or ":" in name
                    or any(part in (".", "..") for part in path.parts) or any(ord(char) < 32 for char in name)
                    or info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Noncanonical release member")
        files = {info.filename: bundle.read(info) for info in infos}
    manifest_raw = files["release-manifest.json"]
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen manifest digest mismatch")
    manifest = decode(manifest_raw)
    entries = manifest.get("files", {})
    if len(entries) != 70 or set(files) != set(entries) | {"release-manifest.json"}:
        raise ValueError("Release manifest file set mismatch")
    for name, expected in entries.items():
        data = files[name]
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Release member digest mismatch")
    return files


def extract_release(archive: Path, output: Path) -> None:
    files = release_files(archive)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Extraction requires an empty destination")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


@dataclass
class QualifiedRuntime:
    runtime: Any
    benchmark: dict
    startup_probes: list[dict]
    temporary_directory: Any = None


def qualify_release(archive: Path, root: Path | None = None) -> QualifiedRuntime:
    files = release_files(archive)
    temporary = None
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="szl-confirmation-")
        root = Path(temporary.name)
        for name, data in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    root = root.resolve(strict=True)
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if observed != set(files):
        raise ValueError("Extracted release file set mismatch")
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or path.read_bytes() != expected:
            raise ValueError("Extracted release byte mismatch")
    spec = importlib.util.spec_from_file_location("szl_confirmation_frozen_runtime", root / "server.py")
    if spec is None or spec.loader is None:
        raise ValueError("Frozen runtime loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.ConfirmationRuntime(root)
    benchmark = runtime.benchmark()
    if (benchmark.get("verification", {}).get("status") != "SOURCE_BOUND_AND_RECOMPUTED"
            or benchmark.get("verification", {}).get("rows_verified") != 5184
            or benchmark["data"]["primary"]["overall_pass"] is not False):
        raise ValueError("Registered benchmark qualification failed")
    probes = []
    for seed in MODEL_SEEDS:
        request = {"seed": 20260929, "index": 9821, "family": "shared_bias", "policy": "learned", "model_seed": seed}
        result, binding = runtime.run(request)
        if (not binding.get("checkpoint_sha256") or not binding.get("model_fingerprint")
                or result.get("policy") != "learned" or not isinstance(result.get("history"), list)):
            raise ValueError("Checkpoint execution qualification failed")
        probes.append({"role": "STARTUP_EXECUTION_PROBE", "model_seed": seed,
                       "request_sha256": digest(request), "result_sha256": digest(result), "binding": binding})
    if runtime.status().get("models_loaded") != list(MODEL_SEEDS):
        raise ValueError("All three trained models must be loaded")
    return QualifiedRuntime(runtime, benchmark, probes, temporary)


class EphemeralReceipts:
    """Capped process-local receipts. No listing or cross-restart persistence."""
    def __init__(self, *, capacity=MAX_RECEIPTS, ttl=RECEIPT_TTL, max_bytes=MAX_RECEIPT_BYTES, clock=time.monotonic):
        self.capacity = capacity
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.clock = clock
        self.rows: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self.lock = threading.Lock()

    def _expire(self):
        now = self.clock()
        for ident, (created, _) in list(self.rows.items()):
            if now - created >= self.ttl:
                del self.rows[ident]

    def save(self, payload: dict) -> dict:
        receipt = {**payload, "receipt_sha256": digest(payload)}
        raw = canonical(receipt)
        if len(raw) > self.max_bytes:
            raise ValueError("Receipt size limit exceeded")
        with self.lock:
            self._expire()
            self.rows[receipt["id"]] = (self.clock(), raw)
            while len(self.rows) > self.capacity:
                self.rows.popitem(last=False)
        return receipt

    def get(self, ident: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", ident):
            raise KeyError("Receipt unavailable")
        with self.lock:
            self._expire()
            _, raw = self.rows[ident]
        receipt = decode(raw)
        payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("id") != ident or receipt.get("receipt_sha256") != digest(payload):
            raise ValueError("Receipt integrity check failed")
        if receipt.get("status") == "COMPLETE" and receipt.get("result_sha256") != digest(receipt["result"]):
            raise ValueError("Result integrity check failed")
        return receipt

    def count(self) -> int:
        with self.lock:
            self._expire()
            return len(self.rows)


class TokenBucket:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.last = clock()
        self.tokens = float(RATE_BURST)
        self.lock = threading.Lock()

    def take(self) -> tuple[bool, int]:
        with self.lock:
            now = self.clock()
            self.tokens = min(RATE_BURST, self.tokens + max(0.0, now - self.last) * RATE_PER_MINUTE / 60)
            self.last = now
            if self.tokens < 1:
                return False, max(1, math.ceil((1 - self.tokens) * 60 / RATE_PER_MINUTE))
            self.tokens -= 1
            return True, 0


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


class ConfirmationApp:
    def __init__(self, *, archive: Path | None = None, release_root: Path | None = None,
                 allow_local: bool = False, qualifier: Callable[[], QualifiedRuntime] | None = None,
                 body_deadline: float = BODY_DEADLINE):
        self.archive = archive or HERE / "release.zip"
        self.release_root = release_root
        self.allow_local = allow_local
        self.qualifier = qualifier or (lambda: qualify_release(self.archive, self.release_root))
        self.body_deadline = body_deadline
        self.qualified: QualifiedRuntime | None = None
        self.state = "STARTING"
        self.startup_error = None
        self.trial_lock = threading.Lock()
        self.receipts = EphemeralReceipts()
        self.rate = TokenBucket()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="confirmation-runtime")
        self.started_at = utc_now()
        self.instance_id = uuid.uuid4().hex

    async def startup(self):
        if self.state != "STARTING":
            return
        try:
            self.qualified = await asyncio.get_running_loop().run_in_executor(self.executor, self.qualifier)
            self.state = "READY"
        except Exception:
            # Public status never returns exception messages or local filesystem paths.
            self.state = "FAILED"
            self.startup_error = "Frozen release or trained-model execution could not be qualified"

    async def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.qualified and self.qualified.temporary_directory:
            self.qualified.temporary_directory.cleanup()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await self.startup()
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await self.shutdown()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "http":
            try:
                await self.http(scope, receive, send)
            except RequestError as exc:
                await self.reply(send, exc.status, {"error": exc.message})
        else:
            await send({"type": "websocket.close", "code": 1008})

    async def reply(self, send, status, value, *, mime="application/json; charset=utf-8", extra_headers=()):
        raw = value if isinstance(value, bytes) else canonical(value)
        headers = [(b"content-type", mime.encode("ascii")), (b"content-length", str(len(raw)).encode("ascii")),
                   (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                   (b"referrer-policy", b"no-referrer"),
                   (b"content-security-policy", b"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self' https://huggingface.co"),
                   *extra_headers]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": raw})

    def admission(self, scope):
        headers = {}
        for key, value in scope.get("headers", []):
            headers.setdefault(key.lower(), []).append(value.decode("latin-1"))
        hosts = headers.get(b"host", [])
        if len(hosts) != 1:
            raise RequestError(403, "Use the declared workbench origin")
        host = hosts[0]
        origin = PUBLIC_ORIGIN
        loopback = re.fullmatch(r"(?:127\.0\.0\.1|localhost)(?::([0-9]{1,5}))?", host)
        local = bool(loopback and (loopback.group(1) is None or 1 <= int(loopback.group(1)) <= 65535))
        health = scope["method"] == "GET" and scope["path"] in ("/live", "/healthz")
        if host != PUBLIC_HOST:
            if not local or not (self.allow_local or health):
                raise RequestError(403, "Use the declared workbench origin")
            origin = "http://" + host
        origins = headers.get(b"origin", [])
        if (origins and origins != [origin]) or (scope["method"] == "POST" and origins != [origin]):
            raise RequestError(403, "Use a same-origin trial request")
        if scope.get("query_string", b""):
            raise RequestError(400, "Query parameters are unsupported")
        return headers

    def status(self):
        return {"ready": self.state == "READY", "state": self.state, "error": self.startup_error,
                "busy": self.trial_lock.locked(), "scope": "EXPLORATORY_SYNTHETIC_CPU_INFERENCE",
                "archive_sha256": ARCHIVE_SHA256, "manifest_sha256": MANIFEST_SHA256,
                "checkpoints_verified": len(MODEL_SEEDS) if self.qualified else 0,
                "models_loaded": list(MODEL_SEEDS) if self.qualified else [],
                "startup_execution_probes": self.qualified.startup_probes if self.qualified else [],
                "startup_probes_are_user_receipts": False, "scientific_overall_gate": "FAILED",
                "families": FAMILIES, "policies": POLICIES, "model_seeds": MODEL_SEEDS,
                "limits": {"max_body_bytes": MAX_BODY, "body_deadline_seconds": BODY_DEADLINE,
                           "active_trials": 1, "queued_trials": 0, "global_trials_per_minute": RATE_PER_MINUTE,
                           "global_burst": RATE_BURST, "episode_resource_units": 4},
                "retention": {"mode": "EPHEMERAL_PROCESS_MEMORY", "max_receipts": MAX_RECEIPTS,
                              "ttl_seconds": RECEIPT_TTL, "max_receipt_bytes": MAX_RECEIPT_BYTES,
                              "retained": self.receipts.count(), "listing_available": False,
                              "lost_on_restart": True, "capacity_eviction": "OLDEST_FIRST"},
                "instance_id": self.instance_id, "started_at": self.started_at, "receipt_minted": False}

    async def body(self, headers, receive):
        lengths = headers.get(b"content-length", [])
        if len(lengths) > 1 or (lengths and b"transfer-encoding" in headers):
            raise RequestError(400, "Ambiguous request length")
        declared = None
        if lengths:
            if not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
                raise RequestError(400, "Invalid request length")
            declared = int(lengths[0])
            if declared > MAX_BODY:
                raise RequestError(413, "Trial request exceeds 4096 bytes")
        types = headers.get(b"content-type", [])
        if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise RequestError(415, "Use application/json")
        chunks = []
        total = 0
        try:
            async with asyncio.timeout(self.body_deadline):
                while True:
                    message = await receive()
                    if message["type"] != "http.request":
                        raise RequestError(400, "Incomplete trial request")
                    part = message.get("body", b"")
                    total += len(part)
                    if total > MAX_BODY:
                        raise RequestError(413, "Trial request exceeds 4096 bytes")
                    chunks.append(part)
                    if not message.get("more_body", False):
                        break
        except TimeoutError as exc:
            raise RequestError(408, "Trial request body deadline exceeded") from exc
        if not total or (declared is not None and declared != total):
            raise RequestError(400, "Incomplete trial request")
        try:
            return validate_request(decode(b"".join(chunks)))
        except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
            raise RequestError(400, "Invalid trial request or JSON fields") from exc

    async def http(self, scope, receive, send):
        headers = self.admission(scope)
        method, path = scope["method"], scope["path"]
        if method == "GET":
            if path in ("/live", "/healthz"):
                return await self.reply(send, 200, {"status": "UP", "service": "szl-foundation-confirmation", "receipt_minted": False})
            if path == "/readyz":
                return await self.reply(send, 200 if self.state == "READY" else 503, self.status())
            if path == "/api/status":
                return await self.reply(send, 200, self.status())
            if path == "/api/build-info":
                return await self.reply(send, 200, {"build": source_identity(),
                        "archive_sha256": ARCHIVE_SHA256, "runtime_state": self.state, "receipt_minted": False})
            if path == "/api/results":
                if not self.qualified:
                    return await self.reply(send, 503, {"available": False, "error": "Registered evidence is not qualified"})
                return await self.reply(send, 200, self.qualified.benchmark)
            if path.startswith("/api/trials/"):
                try:
                    receipt = self.receipts.get(path.removeprefix("/api/trials/"))
                except KeyError:
                    return await self.reply(send, 404, {"error": "Receipt unavailable or expired"})
                except ValueError:
                    return await self.reply(send, 503, {"error": "Receipt integrity check failed"})
                return await self.reply(send, 200, receipt)
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in static:
                filename, mime = static[path]
                return await self.reply(send, 200, (HERE / "web" / filename).read_bytes(), mime=mime)
            return await self.reply(send, 404, {"error": "Unknown route"})
        if method != "POST":
            return await self.reply(send, 405, {"error": "Unsupported method"})
        if path != "/api/trial":
            return await self.reply(send, 404, {"error": "Unknown route"})
        request = await self.body(headers, receive)
        if self.state != "READY" or self.qualified is None:
            return await self.reply(send, 503, {"error": "Trained-model execution is unavailable"})
        if not self.trial_lock.acquire(blocking=False):
            return await self.reply(send, 429, {"error": "Another exploratory trial is running; try again shortly"}, extra_headers=[(b"retry-after", b"1")])
        allowed, retry = self.rate.take()
        if not allowed:
            self.trial_lock.release()
            return await self.reply(send, 429, {"error": "Shared CPU trial rate reached; try again shortly"}, extra_headers=[(b"retry-after", str(retry).encode("ascii"))])
        # The worker owns the lock until execution finishes, including disconnects.
        future = asyncio.get_running_loop().run_in_executor(self.executor, self.run_trial, request)
        code, receipt = await asyncio.shield(future)
        return await self.reply(send, code, receipt)

    def run_trial(self, request):
        ident, created = uuid.uuid4().hex, utc_now()
        payload = {"schema": "szl.confirmation.public-exploratory-receipt/v1", "id": ident,
                   "created_at": created, "scope": "EXPLORATORY_SYNTHETIC_CPU_INFERENCE",
                   "request": request, "request_sha256": digest(request),
                   "source": source_identity(),
                   "archive_sha256": ARCHIVE_SHA256, "scientific_overall_gate": "FAILED",
                   "unsigned": True, "authenticity_established": False,
                   "retention": {"mode": "EPHEMERAL_PROCESS_MEMORY", "ttl_seconds": RECEIPT_TTL,
                                 "max_receipts": MAX_RECEIPTS, "lost_on_restart": True}}
        try:
            try:
                result, binding = self.qualified.runtime.run(request)
                payload.update({"status": "COMPLETE", "binding": binding, "result": result, "result_sha256": digest(result)})
                code = 201
            except Exception:
                payload.update({"status": "FAILED", "error": "Trial execution or immutable source admission failed"})
                code = 503
            payload["completed_at"] = utc_now()
            try:
                receipt = self.receipts.save(payload)
            except (ValueError, TypeError, OverflowError):
                return 503, {"error": "A bounded trial receipt could not be retained"}
            return code, receipt
        finally:
            self.trial_lock.release()


def make_app() -> ConfirmationApp:
    configured_host = os.getenv("SPACE_HOST", PUBLIC_HOST)
    if configured_host != PUBLIC_HOST:
        raise ValueError("Unexpected public Space identity")
    root = os.getenv("FOUNDATION_RELEASE_ROOT")
    return ConfirmationApp(release_root=Path(root) if root else None, allow_local=os.getenv("FOUNDATION_ALLOW_LOCAL") == "1")


app = make_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", nargs=2, metavar=("ARCHIVE", "OUTPUT"))
    args = parser.parse_args()
    if not args.extract:
        parser.error("Use Uvicorn for the ASGI service; --extract prepares verified immutable bytes")
    extract_release(Path(args.extract[0]), Path(args.extract[1]))
