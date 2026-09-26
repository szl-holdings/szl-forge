"""Loopback-only review API with local durable receipts and bounded model access."""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import threading
import time
from urllib.parse import urlsplit
import uuid

from . import __version__, engine, model
from .benchmark import examples

LIMITATIONS = [
    "Explicit adjacent form-label association only; not a full accessibility audit or conformance certificate.",
    "Human accessibility and assistive-technology evaluation remain required.",
    "Local history contains repaired markup. Use synthetic or approved non-sensitive source only.",
    "Receipt hashes detect accidental changes; no independent signer or production authorization is claimed.",
    "No new specialist model weights have been trained or qualified by this release.",
]
MAX_RECEIPT_BYTES = 262144
RUN_STATES = frozenset({"RUNNING", "INTERRUPTED", "VERIFIED_SCOPED_REPAIR", "NO_CHANGE", "REVIEW_REQUIRED", "MODEL_UNAVAILABLE"})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RequestError(ValueError):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(code)


def read_receipt(raw, run_id):
    """Reject damaged local records; hashes are not a hostile-owner trust boundary."""
    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RECEIPT_BYTES:
            raise ValueError("receipt size")
        result = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(result, dict) or result.get("run_id") != run_id or str(uuid.UUID(run_id)) != run_id:
            raise ValueError("receipt identity")
        if result.get("state") not in RUN_STATES or result.get("mode") not in {"deterministic", "ollama"}:
            raise ValueError("receipt state")
        if not isinstance(result.get("created_at"), str) or datetime.fromisoformat(result["created_at"]).tzinfo is None:
            raise ValueError("receipt timestamp")
        for field in ("source_sha256", "receipt_sha256"):
            if not isinstance(result.get(field), str) or re.fullmatch(r"[0-9a-f]{64}", result[field]) is None:
                raise ValueError("receipt digest missing")
        original = result.pop("receipt_sha256")
        if not secrets.compare_digest(original, digest(result)):
            raise ValueError("receipt digest mismatch")
        result["receipt_sha256"] = original
        if result["state"] not in {"RUNNING", "INTERRUPTED"}:
            if (not isinstance(result.get("output_html"), str)
                    or result.get("output_sha256") != hashlib.sha256(result["output_html"].encode("utf-8")).hexdigest()):
                raise ValueError("receipt output mismatch")
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise RequestError(409, "LOCAL_RECEIPT_INTEGRITY_FAILED") from None


def receipt_summary(result):
    """An allowlisted source-free export, not a replacement for the full receipt."""
    summary = {key: result.get(key) for key in (
        "run_id", "state", "mode", "created_at", "source_sha256", "output_sha256", "receipt_sha256",
    )}
    summary.update(schema="szl-access-receipt-summary-v1", exporter_version=__version__,
                   contains_source=False, independent_attestation=False,
                   limitations=[
                       "Summary omits source, output, bindings, and model details; retain the full local receipt for review.",
                       "Hashes and run metadata may still identify a known document; this is not an anonymization guarantee.",
                       "A narrow static repair is not accessibility certification or a browser/assistive-technology result.",
                       "Export hash covers this summary without export_sha256; receipt_sha256 refers to the separate full receipt.",
                   ])
    summary["export_sha256"] = digest(summary)
    return summary


def trusted_directory(value):
    """Trusted local parent is a precondition, not protection against a hostile owner."""
    raw = os.fspath(value)
    if not raw or any(part in {".", ".."} for part in raw.replace("\\", "/").split("/")):
        raise ValueError("state path must not contain dot traversal")
    path = Path(os.path.abspath(raw))
    if path == Path(path.anchor) or (os.name == "nt" and path.drive.startswith("\\\\")):
        raise ValueError("state path must be a local non-root directory")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        kernel.GetDriveTypeW.restype = wintypes.UINT
        if len(path.drive) != 2 or kernel.GetDriveTypeW(path.anchor) != 3:
            raise ValueError("state must use a fixed local Windows drive")
        if any(part.rstrip(" .") != part or ":" in part or Path(part).is_reserved() for part in path.parts[1:]):
            raise ValueError("state path has an invalid Windows name")
    for parent in reversed(path.parents):
        observed = parent.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & 0x400:
            raise ValueError("state parent must be trusted, existing and not a reparse point")
    path.mkdir(mode=0o700, exist_ok=True)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & 0x400:
        raise ValueError("state directory must not be a link or reparse point")
    database = path / "access.sqlite3"
    if os.path.lexists(database):
        observed = database.lstat()
        if not stat.S_ISREG(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & 0x400:
            raise ValueError("state database must be a regular file")
    return path


class Store:
    def __init__(self, directory):
        self.path = trusted_directory(directory) / "access.sqlite3"
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, input_hash TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)")
            for run_id, raw in connection.execute("SELECT run_id,substr(payload,1,?) FROM runs", (MAX_RECEIPT_BYTES + 1,)).fetchall():
                result = read_receipt(raw, run_id)
                if result["state"] == "RUNNING":
                    result.update(state="INTERRUPTED", limitations=LIMITATIONS, error="previous process ended before completion")
                    result.pop("receipt_sha256")
                    result["receipt_sha256"] = digest(result)
                    connection.execute("UPDATE runs SET payload=? WHERE run_id=?", (canonical(result), run_id))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            with connection:
                yield connection
        finally:
            connection.close()

    def claim(self, request_id, source, mode):
        binding = digest({"source": source, "mode": mode})
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT run_id,input_hash FROM runs WHERE request_id=?", (request_id,)).fetchone()
            if old:
                if old[1] != binding:
                    raise RequestError(409, "IDEMPOTENCY_CONFLICT")
                return self.get(old[0]), False
            if connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] >= 1000:
                raise RequestError(409, "LOCAL_HISTORY_CAPACITY_REACHED")
            result = {"run_id": str(uuid.uuid4()), "state": "RUNNING", "mode": mode, "created_at": utc_now(),
                      "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "limitations": LIMITATIONS}
            result["receipt_sha256"] = digest(result)
            connection.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (result["run_id"], request_id, binding, result["created_at"], canonical(result)))
            return result, True

    def finish(self, result):
        result.pop("receipt_sha256", None)
        result["receipt_sha256"] = digest(result)
        read_receipt(canonical(result), result["run_id"])
        with self.connect() as connection:
            connection.execute("UPDATE runs SET payload=? WHERE run_id=?", (canonical(result), result["run_id"]))
        return result

    def get(self, run_id):
        with self.connect() as connection:
            row = connection.execute("SELECT substr(payload,1,?) FROM runs WHERE run_id=?", (MAX_RECEIPT_BYTES + 1, run_id)).fetchone()
        if not row:
            raise RequestError(404, "RUN_NOT_FOUND")
        return read_receipt(row[0], run_id)

    def recent(self):
        with self.connect() as connection:
            ids = [r[0] for r in connection.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 50")]
        return [{k: value.get(k) for k in ("run_id", "state", "mode", "created_at", "receipt_sha256")}
                for value in (self.get(run_id) for run_id in ids)]


class StateLease:
    """OS-held single-writer lease, automatically released after process exit."""
    def __init__(self, directory):
        path = trusted_directory(directory) / "access.lock"
        if os.path.lexists(path):
            observed = path.lstat()
            if not stat.S_ISREG(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & 0x400:
                raise ValueError("state lease must be a regular local file")
        self.handle = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0), 0o600), "r+b")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise ValueError("state directory already has an active writer") from None

    def close(self):
        self.handle.close()


class AccessServer(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = False

    def __init__(self, port, directory, key, ollama_model=None):
        if not isinstance(key, str) or len(key) < 32 or len(key) > 256 or not key.isascii() or any(c.isspace() for c in key):
            raise ValueError("SZL_ACCESS_API_KEY must be 32-256 non-whitespace ASCII characters")
        self.key = key
        self.model_name = model.model_name(ollama_model) if ollama_model else None
        self.work_lock = threading.Lock()
        self.connection_slots = threading.BoundedSemaphore(16)
        self.ui = resources.files("szl_access").joinpath("static/index.html").read_bytes()
        script_hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(block).digest()).decode() + "'"
                         for block in re.findall(rb"<script\b[^>]*>(.*?)</script>", self.ui, re.S)]
        style_hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(block).digest()).decode() + "'"
                        for block in re.findall(rb"<style\b[^>]*>(.*?)</style>", self.ui, re.S)]
        self.csp = ("default-src 'none'; connect-src 'self'; img-src 'none'; frame-src 'none'; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'none'; script-src " + " ".join(script_hashes or ["'none'"]) +
                    "; style-src " + " ".join(style_hashes or ["'none'"]))
        self.lease = StateLease(directory)
        try:
            self.store = Store(directory)
            super().__init__(("127.0.0.1", port), AccessHandler)
        except BaseException:
            self.lease.close()
            raise

    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()

    def server_close(self):
        try:
            super().server_close()
        finally:
            self.lease.close()


class AccessHandler(BaseHTTPRequestHandler):
    server_version = "SZLAccess/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # No submitted source, prompts, token, result bodies or URL query logs.
        return

    def reply(self, status, value, *, html=False, attachment=None):
        self.discard_rejected_body()
        raw = value if html else canonical(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if attachment is not None:
            self.send_header("Content-Disposition", 'attachment; filename="' + attachment + '"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", self.server.csp)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(raw)

    def discard_rejected_body(self):
        """Avoid resetting an early-denied POST while its bounded body is in flight.

        Never interpret rejected content. Invalid/ambiguous framing is closed
        without draining. The loop has a one-second total time and 64-KiB cap.
        """
        if self.command != "POST" or getattr(self, "body_consumed", False):
            return
        self.body_consumed = True
        lengths = self.headers.get_all("Content-Length", [])
        if (self.headers.get_all("Transfer-Encoding") or len(lengths) != 1
                or re.fullmatch(r"[0-9]{1,5}", lengths[0]) is None):
            return
        remaining = int(lengths[0])
        if not 0 < remaining <= 65536:
            return
        original_timeout = self.connection.gettimeout()
        deadline = time.monotonic() + 1
        try:
            while remaining:
                allowance = deadline - time.monotonic()
                if allowance <= 0:
                    break
                self.connection.settimeout(allowance)
                chunk = self.rfile.read1(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass
        finally:
            self.connection.settimeout(original_timeout)

    def validate_request(self, *, public=False):
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get_all("Host") != [host]:
            raise RequestError(403, "HOST_DENIED")
        origins = self.headers.get_all("Origin", [])
        if origins and origins != ["http://" + host]:
            raise RequestError(403, "ORIGIN_DENIED")
        if not public:
            values = self.headers.get_all("Authorization", [])
            if len(values) != 1 or not values[0].isascii() or not secrets.compare_digest(values[0], "Bearer " + self.server.key):
                raise RequestError(401, "AUTHENTICATION_REQUIRED")

    def do_GET(self):
        try:
            path = urlsplit(self.path)
            if path.query or path.fragment:
                raise RequestError(400, "QUERY_NOT_SUPPORTED")
            self.validate_request(public=path.path in {"/", "/api/health"})
            if path.path == "/":
                self.reply(200, self.server.ui, html=True)
            elif path.path == "/api/health":
                self.reply(200, {"ok": True, "service": "szl_access_lab", "version": __version__,
                                "model": {"enabled": bool(self.server.model_name), "name": self.server.model_name,
                                          "trained_for_accessibility": False}, "limitations": LIMITATIONS})
            elif path.path == "/api/examples":
                self.reply(200, {"examples": examples()})
            elif path.path == "/api/runs":
                self.reply(200, {"runs": self.server.store.recent()})
            elif re.fullmatch(r"/api/runs/[0-9a-f-]{36}/summary", path.path):
                run_id = path.path.split("/")[3]
                summary = receipt_summary(self.server.store.get(run_id))
                self.reply(200, summary, attachment=f"szl-access-{run_id}-summary.json")
            elif re.fullmatch(r"/api/runs/[0-9a-f-]{36}", path.path):
                self.reply(200, self.server.store.get(path.path.rsplit("/", 1)[1]))
            else:
                raise RequestError(404, "NOT_FOUND")
        except RequestError as exc:
            self.reply(exc.status, {"ok": False, "error": exc.code})
        except (OSError, ValueError, sqlite3.Error):
            self.reply(503, {"ok": False, "error": "LOCAL_STATE_UNAVAILABLE"})

    def do_POST(self):
        try:
            self.validate_request()
            if self.path != "/api/runs":
                raise RequestError(404, "NOT_FOUND")
            if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                raise RequestError(400, "BODY_FRAMING_REJECTED")
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise RequestError(415, "JSON_REQUIRED")
            try:
                length = int(self.headers["Content-Length"])
            except (TypeError, ValueError):
                raise RequestError(400, "BODY_LENGTH_REJECTED") from None
            if not 0 < length <= 65536:
                raise RequestError(413, "BODY_SIZE_REJECTED")
            self.body_consumed = True
            payload = json.loads(self.rfile.read(length), object_pairs_hook=unique_object)
            if not isinstance(payload, dict) or set(payload) != {"source", "mode", "request_id"}:
                raise RequestError(400, "REQUEST_SCHEMA_REJECTED")
            source, mode, request_id = payload["source"], payload["mode"], payload["request_id"]
            if not isinstance(source, str) or not 0 < len(source.encode()) <= 32768 or mode not in {"deterministic", "ollama"}:
                raise RequestError(400, "REQUEST_VALUE_REJECTED")
            if not isinstance(request_id, str) or str(uuid.UUID(request_id)) != request_id:
                raise RequestError(400, "REQUEST_ID_REJECTED")
            if mode == "ollama" and not self.server.model_name:
                raise RequestError(409, "LOCAL_MODEL_DISABLED")
            if not self.server.work_lock.acquire(blocking=False):
                raise RequestError(429, "LOCAL_WORKER_BUSY")
            try:
                result, claimed = self.server.store.claim(request_id, source, mode)
                if claimed:
                    try:
                        result.update(self.execute(source, mode))
                    except (ValueError, TypeError, KeyError):
                        result.update(state="REVIEW_REQUIRED", output_html=source,
                                      output_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                      checks={}, applied_bindings=[], limitations=LIMITATIONS,
                                      error="proposal did not satisfy the checked repair contract")
                    result = self.server.store.finish(result)
            finally:
                self.server.work_lock.release()
            self.reply(200, result)
        except RequestError as exc:
            self.reply(exc.status, {"ok": False, "error": exc.code})
        except (ValueError, UnicodeError, TypeError, RecursionError):
            self.reply(400, {"ok": False, "error": "INVALID_JSON_OR_REQUEST"})
        except (OSError, sqlite3.Error):
            self.reply(503, {"ok": False, "error": "LOCAL_STATE_UNAVAILABLE"})

    def execute(self, source, mode):
        analysis = engine.analyze(source)
        evidence = {"enabled": False, "name": None, "trained_for_accessibility": False}
        if mode == "ollama" and analysis["status"] == "SUPPORTED" and analysis["candidates"]:
            try:
                proposal, evidence = model.propose(analysis, self.server.model_name)
            except model.ModelUnavailable:
                return {"state": "MODEL_UNAVAILABLE", "output_html": source, "output_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "applied_bindings": [], "checks": {}, "limitations": LIMITATIONS,
                        "model": {"enabled": True, "name": self.server.model_name, "trained_for_accessibility": False},
                        "error": "local model failed; no deterministic fallback or repair was performed"}
        else:
            proposal = engine.deterministic_proposal(source)
            if mode == "ollama":
                evidence = {"enabled": True, "name": self.server.model_name, "invoked": False,
                            "reason": "no_supported_candidates", "trained_for_accessibility": False}
        result = engine.apply_proposal(source, proposal)
        result["limitations"] = list(dict.fromkeys(result.get("limitations", []) + LIMITATIONS))
        result["model"] = evidence
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local form-label review workbench; not accessibility certification.")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    with AccessServer(args.port, args.state_dir, os.environ.get("SZL_ACCESS_API_KEY", ""),
                      os.environ.get("SZL_ACCESS_OLLAMA_MODEL")) as server:
        print(f"SZL Access Lab listening at http://127.0.0.1:{server.server_port}; local-only scoped repair", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
