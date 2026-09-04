from __future__ import annotations

"""Bounded live-shadow result transports for Owned Agent Clinical Control.

The adapters in this module receive result messages and hand them to the
existing kernel.  They are not a device-control plane.  An MLLP receiver can
only return an HL7 acknowledgement after the kernel reports durable ingest.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import ipaddress
import json
import queue
import re
import shutil
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Mapping, Protocol


MLLP_START = b"\x0b"
MLLP_END = b"\x1c\r"
DEFAULT_QUEUE_CAPACITY = 128
MAX_QUEUE_CAPACITY = 4096
DEFAULT_MAX_FRAME_BYTES = 1024 * 1024
MAX_MAX_FRAME_BYTES = 1024 * 1024
LIVE_SHADOW_MODE = "live-shadow"
VALID_ACK_CODES = frozenset({"AA", "AE", "AR"})
SAFE_CODE_RE = re.compile(r"^[A-Z0-9_]{1,64}$")
TRANSPORT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ClinicalKernelProtocol(Protocol):
    def run(
        self,
        command: str,
        args: Mapping[str, Any] | None = None,
        *,
        state_dir: str | Path | None = None,
    ) -> dict[str, Any]: ...


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hl7_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S+0000")


def _safe_decode(value: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def _safe_code(value: Any, fallback: str = "UNSPECIFIED") -> str:
    candidate = str(value or "").strip().upper().replace("-", "_")
    return candidate if SAFE_CODE_RE.fullmatch(candidate) else fallback


def _safe_hl7_field(value: Any, fallback: str, maximum: int = 199) -> str:
    candidate = str(value or "").strip()
    if not candidate or len(candidate) > maximum:
        return fallback
    if any(ord(char) < 32 or ord(char) > 126 or char in "|\r\n" for char in candidate):
        return fallback
    return candidate


def _redacted_token(value: str | bytes, length: int = 16) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:length]


def _bounded_int(value: Any, default: int, minimum: int, maximum: int, name: str) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise TransportFailure(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TransportFailure(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise TransportFailure(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _bounded_float(value: Any, default: float, minimum: float, maximum: float, name: str) -> float:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise TransportFailure(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TransportFailure(f"{name} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise TransportFailure(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _extract_messages_from_response(payload: str) -> list[tuple[bytes, dict[str, Any]]]:
    normalized = payload.strip()
    if not normalized:
        return []
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return [(normalized.encode("utf-8"), {"kind": "raw_text"})]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    messages: list[tuple[bytes, dict[str, Any]]] = []
    for index, item in enumerate(parsed):
        if isinstance(item, dict):
            raw = item.get("hl7") or item.get("message") or item.get("payload") or item.get("raw")
            if isinstance(raw, str):
                messages.append((raw.encode("utf-8"), {"kind": "json_payload", "item_index": index}))
        elif isinstance(item, str):
            messages.append((item.encode("utf-8"), {"kind": "json_text", "item_index": index}))
    return messages


@dataclass(frozen=True)
class HL7Envelope:
    valid: bool
    control_id: str
    message_type: str
    sender_application: str
    sender_facility: str
    version: str


def _extract_hl7_envelope(raw: bytes) -> HL7Envelope:
    payload = raw[1:] if raw.startswith(MLLP_START) else raw
    if payload.endswith(MLLP_END):
        payload = payload[:-2]
    first_segment = payload.split(b"\r", 1)[0]
    try:
        msh = first_segment.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return HL7Envelope(False, "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "2.5.1")
    if not msh.startswith("MSH|"):
        return HL7Envelope(False, "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "2.5.1")
    fields = msh.split("|")
    control_id = _safe_hl7_field(fields[9] if len(fields) > 9 else "", "UNKNOWN")
    message_type = _safe_hl7_field(fields[8] if len(fields) > 8 else "", "UNKNOWN", 64)
    sender_application = _safe_hl7_field(fields[2] if len(fields) > 2 else "", "UNKNOWN", 128)
    sender_facility = _safe_hl7_field(fields[3] if len(fields) > 3 else "", "UNKNOWN", 128)
    version = _safe_hl7_field(fields[11] if len(fields) > 11 else "", "2.5.1", 32)
    return HL7Envelope(
        control_id != "UNKNOWN" and message_type != "UNKNOWN",
        control_id,
        message_type,
        sender_application,
        sender_facility,
        version,
    )


def binding_lookup_token(raw: bytes) -> str:
    """Derive a PHI-safe filename token from PID-3 plus OBR-2/3.

    The token is only an index into a site-managed binding registry. It is not
    patient identity resolution, deidentification, or authorization evidence.
    """

    payload = raw[1:] if raw.startswith(MLLP_START) else raw
    if payload.endswith(MLLP_END):
        payload = payload[:-2]
    try:
        segments = [segment for segment in payload.decode("utf-8").split("\r") if segment]
    except UnicodeDecodeError as exc:
        raise IngestRejected("BINDING_KEY_UNAVAILABLE") from exc
    pid = next((segment for segment in segments if segment.startswith("PID|")), "")
    obr = next((segment for segment in segments if segment.startswith("OBR|")), "")
    pid_fields = pid.split("|")
    obr_fields = obr.split("|")
    subject = pid_fields[3] if len(pid_fields) > 3 else ""
    order = ""
    if len(obr_fields) > 2:
        order = obr_fields[2]
    if not order and len(obr_fields) > 3:
        order = obr_fields[3]
    if not subject or not order or len(subject) > 256 or len(order) > 256:
        raise IngestRejected("BINDING_KEY_UNAVAILABLE")
    if any(ord(char) < 32 or char in "|\r\n" for char in subject + order):
        raise IngestRejected("BINDING_KEY_UNAVAILABLE")
    return hashlib.sha256((subject + "\x00" + order).encode("utf-8")).hexdigest()


def build_mllp_ack(
    inbound: bytes,
    acknowledgement_code: str,
    error_code: str = "NONE",
    *,
    timestamp: str | None = None,
    version_override: str | None = None,
) -> bytes:
    """Build the configured R33 ACK without echoing patient/result content.

    MSA-2 echoes inbound MSH-10.  The outgoing MSH-10 is deterministic for the
    inbound control ID and acknowledgement code.  AE and AR include ERR.
    """

    ack_code = str(acknowledgement_code).upper()
    if ack_code not in VALID_ACK_CODES:
        raise ValueError("acknowledgement_code must be AA, AE, or AR")
    envelope = _extract_hl7_envelope(inbound)
    safe_error = _safe_code(error_code, "UNSPECIFIED")
    digest_material = f"{envelope.control_id}|{ack_code}".encode("utf-8")
    ack_control_id = f"OACACK{hashlib.sha256(digest_material).hexdigest()[:20].upper()}"
    message_timestamp = _safe_hl7_field(timestamp or _hl7_timestamp(), _hl7_timestamp(), 32)
    response_version = _safe_hl7_field(version_override, envelope.version, 32) if version_override else envelope.version
    message = (
        "MSH|^~\\&|OAC-LIVE-SHADOW|OAC|"
        f"{envelope.sender_application}|{envelope.sender_facility}|{message_timestamp}||"
        f"ACK^R33^ACK|{ack_control_id}|P|{response_version}\r"
        f"MSA|{ack_code}|{envelope.control_id}|{safe_error}\r"
    )
    if ack_code in {"AE", "AR"}:
        message += f"ERR|||{safe_error}^OAC^99OAC|E\r"
    return MLLP_START + message.encode("ascii") + MLLP_END


class TransportFailure(RuntimeError):
    pass


class IngestRejected(TransportFailure):
    def __init__(self, code: str) -> None:
        self.code = _safe_code(code, "INGEST_REJECTED")
        super().__init__(self.code)


@dataclass(frozen=True)
class IngestOutcome:
    ok: bool
    code: str


@dataclass
class _WorkItem:
    raw: bytes
    metadata: dict[str, Any]
    completion: threading.Event
    outcome: IngestOutcome | None = None


@dataclass
class TransportStatus:
    transport_id: str
    kind: str
    running: bool
    start_at: str
    messages_total: int
    messages_succeeded: int
    messages_failed: int
    queue_rejected: int
    queue_depth: int
    queue_capacity: int
    last_error: str | None
    last_seen_utc: str | None
    listener_port: int | None
    transport_security: str
    operational_mode: str = LIVE_SHADOW_MODE
    site_validated: bool = False
    clinical_use_authorized: bool = False
    device_commands_enabled: bool = False
    profile_assertion: str = "configured-profile-not-site-validated"

    def asdict(self) -> dict[str, Any]:
        return {
            "transport_id": self.transport_id,
            "kind": self.kind,
            "running": self.running,
            "start_at": self.start_at,
            "messages_total": self.messages_total,
            "messages_succeeded": self.messages_succeeded,
            "messages_failed": self.messages_failed,
            "queue_rejected": self.queue_rejected,
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "last_error": self.last_error,
            "last_seen_utc": self.last_seen_utc,
            "listener_port": self.listener_port,
            "transport_security": self.transport_security,
            "operational_mode": self.operational_mode,
            "site_validated": self.site_validated,
            "clinical_use_authorized": self.clinical_use_authorized,
            "device_commands_enabled": self.device_commands_enabled,
            "profile_assertion": self.profile_assertion,
        }


class BaseTransport:
    canonical_kind = "base"
    profile_assertion = "configured-profile-not-site-validated"

    def __init__(
        self,
        transport_id: str,
        config: dict[str, Any],
        ingest: Callable[[bytes, dict[str, Any]], Any],
        *,
        stop_event: threading.Event,
        poll_interval: float = 2.0,
    ) -> None:
        self.transport_id = transport_id
        self.config = dict(config)
        self._ingest = ingest
        self._stop_event = stop_event
        self._poll_interval = _bounded_float(poll_interval, 2.0, 0.05, 3600.0, "poll_interval")
        self._queue_capacity = _bounded_int(
            config.get("queue_capacity"), DEFAULT_QUEUE_CAPACITY, 1, MAX_QUEUE_CAPACITY, "queue_capacity"
        )
        self._queue_put_timeout = _bounded_float(
            config.get("queue_put_timeout"), 0.25, 0.0, 5.0, "queue_put_timeout"
        )
        self._startup_timeout = _bounded_float(
            config.get("startup_timeout"), 10.0, 0.1, 30.0, "startup_timeout"
        )
        self._thread: threading.Thread | None = None
        self._started_at = _utc_now_iso()
        self._messages_total = 0
        self._messages_succeeded = 0
        self._messages_failed = 0
        self._queue_rejected = 0
        self._last_error: str | None = None
        self._last_seen: str | None = None
        self._metrics_lock = threading.Lock()
        self._startup_event = threading.Event()
        self._startup_error: str | None = None
        self._inbound_queue: queue.Queue[_WorkItem] = queue.Queue(maxsize=self._queue_capacity)
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"oac-transport-worker-{_redacted_token(transport_id, 10)}",
            daemon=True,
        )

    @property
    def transport_kind(self) -> str:
        return self.canonical_kind

    @property
    def transport_security(self) -> str:
        return "not-applicable"

    def _mark_ready(self) -> None:
        self._startup_event.set()

    def _set_error(self, code: str, exc: BaseException | None = None) -> None:
        safe = _safe_code(code, "TRANSPORT_ERROR")
        suffix = f":{type(exc).__name__}" if exc is not None else ""
        with self._metrics_lock:
            self._last_error = f"{safe}{suffix}"

    def _record_rejection(self, code: str) -> None:
        with self._metrics_lock:
            self._messages_total += 1
            self._messages_failed += 1
            self._last_error = _safe_code(code, "TRANSPORT_REJECTED")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._worker.start()
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=f"oac-transport-{_redacted_token(self.transport_id, 10)}",
            daemon=True,
        )
        self._thread.start()
        if not self._startup_event.wait(self._startup_timeout):
            self.stop()
            raise TransportFailure("transport startup timed out")
        if self._startup_error is not None:
            self.stop()
            raise TransportFailure(f"transport startup failed: {self._startup_error}")

    def _run_guarded(self) -> None:
        try:
            self.run_loop()
        except Exception as exc:  # noqa: BLE001 - fail closed at the transport boundary
            self._startup_error = _safe_code(type(exc).__name__, "TRANSPORT_ERROR")
            self._set_error("TRANSPORT_LOOP_FAILED", exc)
        finally:
            self._startup_event.set()

    def _request_stop(self) -> None:
        return

    def stop(self) -> None:
        self._stop_event.set()
        self._request_stop()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._startup_timeout))
        self._worker.join(timeout=5.0)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _worker_loop(self) -> None:
        while True:
            if self._stop_event.is_set() and self._inbound_queue.empty():
                break
            try:
                item = self._inbound_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            outcome = IngestOutcome(False, "INGEST_EXCEPTION")
            try:
                result = self._ingest(item.raw, dict(item.metadata))
                if isinstance(result, Mapping) and result.get("ok") is False:
                    raise IngestRejected(str(result.get("code") or "INGEST_REJECTED"))
                outcome = IngestOutcome(True, "INGESTED")
            except IngestRejected as exc:
                outcome = IngestOutcome(False, exc.code)
                self._set_error(exc.code)
            except Exception as exc:  # noqa: BLE001 - isolate the receiver from kernel failure
                self._set_error("INGEST_EXCEPTION", exc)
            finally:
                with self._metrics_lock:
                    self._messages_total += 1
                    if outcome.ok:
                        self._messages_succeeded += 1
                        self._last_seen = _utc_now_iso()
                    else:
                        self._messages_failed += 1
                item.outcome = outcome
                item.completion.set()
                self._inbound_queue.task_done()

    def _submit(
        self,
        raw: bytes,
        metadata: Mapping[str, Any],
        *,
        wait: bool,
        timeout: float | None = None,
    ) -> IngestOutcome:
        if self._stop_event.is_set():
            return IngestOutcome(False, "TRANSPORT_STOPPING")
        item = _WorkItem(bytes(raw), dict(metadata), threading.Event())
        try:
            self._inbound_queue.put(item, timeout=self._queue_put_timeout)
        except queue.Full:
            with self._metrics_lock:
                self._messages_total += 1
                self._messages_failed += 1
                self._queue_rejected += 1
                self._last_error = "INGEST_QUEUE_FULL"
            return IngestOutcome(False, "INGEST_QUEUE_FULL")
        if not wait:
            return IngestOutcome(True, "QUEUED")
        if not item.completion.wait(timeout):
            self._set_error("INGEST_WAIT_TIMEOUT")
            return IngestOutcome(False, "INGEST_WAIT_TIMEOUT")
        return item.outcome or IngestOutcome(False, "INGEST_OUTCOME_MISSING")

    def status(self) -> TransportStatus:
        with self._metrics_lock:
            return TransportStatus(
                transport_id=self.transport_id,
                kind=self.transport_kind,
                running=self.running,
                start_at=self._started_at,
                messages_total=self._messages_total,
                messages_succeeded=self._messages_succeeded,
                messages_failed=self._messages_failed,
                queue_rejected=self._queue_rejected,
                queue_depth=self._inbound_queue.qsize(),
                queue_capacity=self._queue_capacity,
                last_error=self._last_error,
                last_seen_utc=self._last_seen,
                listener_port=getattr(self, "bound_port", None),
                transport_security=self.transport_security,
                profile_assertion=self.profile_assertion,
            )

    def run_loop(self) -> None:
        raise NotImplementedError


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class FileDropTransport(BaseTransport):
    canonical_kind = "file-drop"
    profile_assertion = "filesystem-result-drop-not-site-validated"

    @property
    def transport_security(self) -> str:
        return "local-filesystem-boundary"

    def __init__(
        self,
        transport_id: str,
        config: dict[str, Any],
        ingest: Callable[[bytes, dict[str, Any]], Any],
        *,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            transport_id, config, ingest, stop_event=stop_event,
            poll_interval=float(config.get("poll_interval", 1.0)),
        )
        self.watch_dir = Path(config.get("watch_dir", "inbox")).expanduser().resolve()
        self.pattern = str(config.get("pattern", "*.hl7"))
        self.archive_dir = Path(config.get("archive_dir", self.watch_dir / ".oac-archive")).expanduser().resolve()
        self.quarantine_dir = Path(config.get("quarantine_dir", self.watch_dir / ".oac-quarantine")).expanduser().resolve()
        self.settle_seconds = _bounded_float(config.get("settle_seconds"), 0.5, 0.0, 60.0, "settle_seconds")
        self.ingest_timeout = _bounded_float(config.get("ingest_timeout"), 60.0, 1.0, 600.0, "ingest_timeout")
        self.max_file_bytes = _bounded_int(
            config.get("max_file_bytes"), DEFAULT_MAX_FRAME_BYTES, 1, MAX_MAX_FRAME_BYTES, "max_file_bytes"
        )
        self._candidates: dict[str, tuple[int, int, float]] = {}

    def _destination(self, root: Path, payload: bytes) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return root / f"{stamp}-{_redacted_token(payload)}-{uuid.uuid4().hex[:8]}.hl7"

    def _move(self, path: Path, root: Path, payload: bytes) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        destination = self._destination(root, payload)
        shutil.move(str(path), str(destination))
        return destination

    def _quarantine(self, path: Path, payload: bytes, code: str) -> None:
        try:
            destination = self._move(path, self.quarantine_dir, payload)
            destination.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "code": _safe_code(code, "INGEST_REJECTED"),
                        "quarantined_at": _utc_now_iso(),
                        "operational_mode": LIVE_SHADOW_MODE,
                        "site_validated": False,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            self._set_error("FILE_QUARANTINE_FAILED", exc)

    def run_loop(self) -> None:
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._mark_ready()
        while not self._stop_event.is_set():
            now = time.monotonic()
            paths = sorted(self.watch_dir.glob(self.pattern))
            visible: set[str] = set()
            for path in paths:
                if self._stop_event.is_set():
                    break
                if (
                    path.is_symlink()
                    or not _is_under(path, self.watch_dir)
                    or not path.is_file()
                    or _is_under(path, self.archive_dir)
                    or _is_under(path, self.quarantine_dir)
                ):
                    continue
                key = str(path.resolve())
                visible.add(key)
                try:
                    stat = path.stat()
                except OSError as exc:
                    self._set_error("FILE_STAT_FAILED", exc)
                    continue
                snapshot = (stat.st_size, stat.st_mtime_ns)
                prior = self._candidates.get(key)
                if prior is None or prior[:2] != snapshot:
                    self._candidates[key] = (snapshot[0], snapshot[1], now)
                    continue
                if now - prior[2] < self.settle_seconds:
                    continue
                try:
                    payload = path.read_bytes()
                except OSError as exc:
                    self._set_error("FILE_READ_FAILED", exc)
                    continue
                try:
                    after_read = path.stat()
                except OSError as exc:
                    self._set_error("FILE_STAT_FAILED", exc)
                    continue
                if (after_read.st_size, after_read.st_mtime_ns) != snapshot:
                    self._candidates[key] = (after_read.st_size, after_read.st_mtime_ns, now)
                    continue
                if not payload or len(payload) > self.max_file_bytes:
                    self._record_rejection("INVALID_FILE_SIZE")
                    self._quarantine(path, payload, "INVALID_FILE_SIZE")
                    self._candidates.pop(key, None)
                    continue
                outcome = self._submit(
                    payload,
                    {
                        "kind": "file-drop",
                        "source_file_token": _redacted_token(path.name),
                        "payload_size": len(payload),
                    },
                    wait=True,
                    timeout=self.ingest_timeout,
                )
                if outcome.ok:
                    try:
                        self._move(path, self.archive_dir, payload)
                    except OSError as exc:
                        self._set_error("FILE_ARCHIVE_FAILED", exc)
                elif outcome.code in {"INGEST_QUEUE_FULL", "TRANSPORT_STOPPING"}:
                    pass  # Retryable backpressure: leave the source untouched.
                else:
                    self._quarantine(path, payload, outcome.code)
                self._candidates.pop(key, None)
            for key in set(self._candidates) - visible:
                self._candidates.pop(key, None)
            self._stop_event.wait(self._poll_interval)


class MLLPReceiverBase(BaseTransport):
    profile_assertion = "configured-oru-r30-ack-r33-profile-not-site-validated"

    def __init__(
        self,
        transport_id: str,
        config: dict[str, Any],
        ingest: Callable[[bytes, dict[str, Any]], Any],
        *,
        stop_event: threading.Event,
        poll_interval: float,
    ) -> None:
        super().__init__(transport_id, config, ingest, stop_event=stop_event, poll_interval=poll_interval)
        self.max_frame_bytes = _bounded_int(
            config.get("max_frame_bytes"), DEFAULT_MAX_FRAME_BYTES, 256, MAX_MAX_FRAME_BYTES, "max_frame_bytes"
        )
        self.socket_timeout = _bounded_float(config.get("socket_timeout"), 0.5, 0.05, 30.0, "socket_timeout")
        self.idle_timeout = _bounded_float(config.get("idle_timeout"), 30.0, 1.0, 3600.0, "idle_timeout")
        self.ingest_timeout = _bounded_float(config.get("ingest_timeout"), 60.0, 1.0, 600.0, "ingest_timeout")
        single_message = config.get("single_message_per_connection", False)
        if not isinstance(single_message, bool):
            raise TransportFailure("single_message_per_connection must be a boolean")
        self.single_message_per_connection = single_message
        required_version = str(config.get("required_hl7_version", "")).strip()
        if required_version and not re.fullmatch(r"[0-9.]{1,10}", required_version):
            raise TransportFailure("required_hl7_version is invalid")
        self.required_hl7_version = required_version or None
        required_message_type = str(
            config.get("required_message_type", "ORU^R30^ORU_R30")
        ).strip()
        self.required_message_type = _safe_hl7_field(required_message_type, "", 64) or None

    def _send_ack(self, connection: socket.socket, frame: bytes, code: str, error: str) -> bool:
        try:
            connection.sendall(
                build_mllp_ack(
                    frame,
                    code,
                    error,
                    version_override=self.required_hl7_version,
                )
            )
            return True
        except OSError as exc:
            self._set_error("MLLP_ACK_SEND_FAILED", exc)
            return False

    def _consume_connection(self, connection: socket.socket) -> None:
        connection.settimeout(self.socket_timeout)
        buffer = b""
        last_activity = time.monotonic()
        while not self._stop_event.is_set():
            try:
                chunk = connection.recv(min(65536, self.max_frame_bytes + 2))
            except socket.timeout:
                if time.monotonic() - last_activity >= self.idle_timeout:
                    return
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    self._set_error("MLLP_RECEIVE_FAILED", exc)
                return
            if not chunk:
                return
            last_activity = time.monotonic()
            buffer += chunk
            while True:
                start = buffer.find(MLLP_START)
                if start < 0:
                    if len(buffer) > self.max_frame_bytes:
                        self._record_rejection("INVALID_MLLP_PREFIX")
                        return
                    break
                if start > 0:
                    buffer = buffer[start:]
                end = buffer.find(MLLP_END, 1)
                if end < 0:
                    if len(buffer) > self.max_frame_bytes + len(MLLP_END):
                        self._record_rejection("MLLP_FRAME_TOO_LARGE")
                        self._send_ack(connection, buffer, "AR", "MLLP_FRAME_TOO_LARGE")
                        return
                    break
                frame = buffer[: end + len(MLLP_END)]
                buffer = buffer[end + len(MLLP_END) :]
                if len(frame) > self.max_frame_bytes:
                    self._record_rejection("MLLP_FRAME_TOO_LARGE")
                    if not self._send_ack(connection, frame, "AR", "MLLP_FRAME_TOO_LARGE"):
                        return
                    continue
                envelope = _extract_hl7_envelope(frame)
                if not envelope.valid:
                    self._record_rejection("INVALID_HL7_ENVELOPE")
                    if not self._send_ack(connection, frame, "AR", "INVALID_HL7_ENVELOPE"):
                        return
                    continue
                if self.required_hl7_version and envelope.version != self.required_hl7_version:
                    self._record_rejection("HL7_VERSION_DENIED")
                    if not self._send_ack(connection, frame, "AR", "HL7_VERSION_DENIED"):
                        return
                    continue
                if self.required_message_type and envelope.message_type != self.required_message_type:
                    self._record_rejection("HL7_MESSAGE_TYPE_DENIED")
                    if not self._send_ack(connection, frame, "AR", "HL7_MESSAGE_TYPE_DENIED"):
                        return
                    continue
                outcome = self._submit(
                    frame,
                    {
                        "kind": self.transport_kind,
                        "frame_size": len(frame),
                        "message_control_token": _redacted_token(envelope.control_id),
                    },
                    wait=True,
                    timeout=self.ingest_timeout,
                )
                # Once framing/profile admission succeeds, inability to queue,
                # store, or process is a host application error (AE), not AR.
                ack_code = "AA" if outcome.ok else "AE"
                if not self._send_ack(connection, frame, ack_code, outcome.code):
                    return
                if self.single_message_per_connection:
                    return


def _loopback_host(host: str) -> bool:
    lowered = host.strip().lower()
    return lowered == "localhost" or lowered == "::1" or lowered.startswith("127.")


class MLLPListenerTransport(MLLPReceiverBase):
    canonical_kind = "mllp-listener"

    def __init__(
        self,
        transport_id: str,
        config: dict[str, Any],
        ingest: Callable[[bytes, dict[str, Any]], Any],
        *,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            transport_id, config, ingest, stop_event=stop_event,
            poll_interval=float(config.get("accept_interval", 0.25)),
        )
        self.bind_host = str(config.get("bind_host", "127.0.0.1")).strip()
        self.bind_port = _bounded_int(config.get("bind_port"), 2100, 0, 65535, "bind_port")
        if not self.bind_host:
            raise TransportFailure("bind_host is required")
        if not _loopback_host(self.bind_host) and not bool(config.get("allow_non_loopback", False)):
            raise TransportFailure("non-loopback MLLP binding requires allow_non_loopback=true")
        self.max_connections = _bounded_int(config.get("max_connections"), 4, 1, 32, "max_connections")
        allowed_peer_values = config.get("allowed_peer_ips", [])
        if not isinstance(allowed_peer_values, list) or not all(
            isinstance(value, str) for value in allowed_peer_values
        ):
            raise TransportFailure("allowed_peer_ips must be an array of IP address strings")
        try:
            self.allowed_peer_ips = frozenset(
                str(ipaddress.ip_address(value.strip())) for value in allowed_peer_values
            )
        except ValueError as exc:
            raise TransportFailure("allowed_peer_ips contains an invalid IP address") from exc
        self.tls_enabled = bool(config.get("tls_enabled", False))
        self.tls_certfile = Path(str(config.get("tls_certfile", ""))).expanduser() if self.tls_enabled else None
        self.tls_keyfile = Path(str(config.get("tls_keyfile", ""))).expanduser() if self.tls_enabled else None
        self.tls_handshake_timeout = _bounded_float(
            config.get("tls_handshake_timeout"), 10.0, 1.0, 60.0, "tls_handshake_timeout"
        )
        if self.tls_enabled and (
            self.tls_certfile is None or self.tls_keyfile is None
            or not self.tls_certfile.is_file() or not self.tls_keyfile.is_file()
        ):
            raise TransportFailure("TLS listener requires existing tls_certfile and tls_keyfile")
        self.bound_port: int | None = None
        self._listener: socket.socket | None = None
        self._listener_lock = threading.Lock()
        self._connection_slots = threading.BoundedSemaphore(self.max_connections)
        self._connection_threads: set[threading.Thread] = set()
        self._connection_lock = threading.Lock()
        # Parse and release certificate/key files before any background thread
        # starts. This makes configuration failure synchronous and prevents a
        # startup-timeout race from leaving certificate handles behind.
        self._tls_context = self._build_tls_context()

    @property
    def transport_security(self) -> str:
        return "tls-server-minimum-1.2" if self.tls_enabled else "plaintext"

    def _request_stop(self) -> None:
        with self._listener_lock:
            listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def _connection_worker(self, connection: socket.socket) -> None:
        try:
            with connection:
                if self._tls_context is not None:
                    connection.settimeout(self.tls_handshake_timeout)
                    try:
                        receiver = self._tls_context.wrap_socket(connection, server_side=True)
                    except (OSError, ssl.SSLError) as exc:
                        self._set_error("TLS_HANDSHAKE_FAILED", exc)
                        return
                    try:
                        self._consume_connection(receiver)
                    finally:
                        receiver.close()
                else:
                    self._consume_connection(connection)
        finally:
            self._connection_slots.release()
            with self._connection_lock:
                self._connection_threads.discard(threading.current_thread())

    def _build_tls_context(self) -> ssl.SSLContext | None:
        if not self.tls_enabled:
            return None
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(
                certfile=str(self.tls_certfile),
                keyfile=str(self.tls_keyfile),
            )
            ca_file = self.config.get("tls_ca_file")
            if ca_file:
                context.load_verify_locations(cafile=str(Path(str(ca_file)).expanduser()))
            if bool(self.config.get("tls_require_client_cert", False)):
                context.verify_mode = ssl.CERT_REQUIRED
            return context
        except (OSError, ssl.SSLError) as exc:
            raise TransportFailure("TLS context configuration failed") from exc

    def run_loop(self) -> None:
        family = socket.AF_INET6 if ":" in self.bind_host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.bind_host, self.bind_port))
            listener.listen(self.max_connections)
            listener.settimeout(self._poll_interval)
            self.bound_port = int(listener.getsockname()[1])
            with self._listener_lock:
                self._listener = listener
            self._mark_ready()
            while not self._stop_event.is_set():
                if not self._connection_slots.acquire(timeout=self._poll_interval):
                    continue
                try:
                    connection, peer = listener.accept()
                except socket.timeout:
                    self._connection_slots.release()
                    continue
                except OSError as exc:
                    self._connection_slots.release()
                    if not self._stop_event.is_set():
                        self._set_error("MLLP_ACCEPT_FAILED", exc)
                    continue
                peer_ip = str(ipaddress.ip_address(str(peer[0])))
                if self.allowed_peer_ips and peer_ip not in self.allowed_peer_ips:
                    connection.close()
                    self._connection_slots.release()
                    self._record_rejection("MLLP_PEER_DENIED")
                    continue
                thread = threading.Thread(
                    target=self._connection_worker,
                    args=(connection,),
                    name=f"oac-mllp-peer-{uuid.uuid4().hex[:8]}",
                    daemon=True,
                )
                with self._connection_lock:
                    self._connection_threads.add(thread)
                thread.start()
        finally:
            with self._listener_lock:
                self._listener = None
            try:
                listener.close()
            except OSError:
                pass
            with self._connection_lock:
                threads = list(self._connection_threads)
            for thread in threads:
                thread.join(timeout=1.0)


class MLLPClientReceiverTransport(MLLPReceiverBase):
    """Receiver that connects to an upstream server; not the analyzer listener."""

    canonical_kind = "mllp-client-receiver"
    profile_assertion = "upstream-mllp-client-receiver-not-site-validated"

    @property
    def transport_security(self) -> str:
        return "plaintext"

    def __init__(
        self,
        transport_id: str,
        config: dict[str, Any],
        ingest: Callable[[bytes, dict[str, Any]], Any],
        *,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            transport_id, config, ingest, stop_event=stop_event,
            poll_interval=float(config.get("reconnect_interval", 2.0)),
        )
        self.remote_host = str(config.get("remote_host", "")).strip()
        self.remote_port = _bounded_int(config.get("remote_port"), 2100, 1, 65535, "remote_port")
        self.connect_timeout = _bounded_float(config.get("connect_timeout"), 5.0, 0.1, 60.0, "connect_timeout")
        if not self.remote_host:
            raise TransportFailure("remote_host is required for mllp-client-receiver")
        if not _loopback_host(self.remote_host) and not bool(
            config.get("allow_plaintext_upstream", False)
        ):
            raise TransportFailure(
                "non-loopback mllp-client-receiver requires allow_plaintext_upstream=true"
            )

    def run_loop(self) -> None:
        self._mark_ready()
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(
                    (self.remote_host, self.remote_port), timeout=self.connect_timeout
                ) as connection:
                    self._consume_connection(connection)
            except OSError as exc:
                if not self._stop_event.is_set():
                    self._set_error("MLLP_UPSTREAM_CONNECT_FAILED", exc)
            self._stop_event.wait(self._poll_interval)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so credentials and trust do not cross origins."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, code, "redirect denied", headers, fp)


class RestPollingTransport(BaseTransport):
    canonical_kind = "rest-poll"
    profile_assertion = "read-only-http-result-poll-not-site-validated"

    def __init__(
        self,
        transport_id: str,
        config: dict[str, Any],
        ingest: Callable[[bytes, dict[str, Any]], Any],
        *,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(
            transport_id, config, ingest, stop_event=stop_event,
            poll_interval=float(config.get("poll_interval", 5.0)),
        )
        self.endpoint = str(config.get("endpoint", "")).strip()
        self.token = config.get("authorization_bearer")
        self.max_response_bytes = _bounded_int(
            config.get("max_response_bytes"), DEFAULT_MAX_FRAME_BYTES, 1, MAX_MAX_FRAME_BYTES,
            "max_response_bytes",
        )
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise TransportFailure("endpoint must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise TransportFailure("endpoint URL credentials are forbidden")
        if parsed.scheme == "http" and not _loopback_host(parsed.hostname) and not bool(
            config.get("allow_insecure_http", False)
        ):
            raise TransportFailure("non-loopback HTTP polling requires allow_insecure_http=true")
        self._endpoint_token = _redacted_token(f"{parsed.scheme}://{parsed.hostname}:{parsed.port or ''}")
        self._endpoint_scheme = parsed.scheme
        self._opener = urllib.request.build_opener(NoRedirectHandler())

    @property
    def transport_security(self) -> str:
        return "https" if self._endpoint_scheme == "https" else "http"

    def run_loop(self) -> None:
        self._mark_ready()
        headers = {
            "User-Agent": "oac-live-shadow-transport/2.0",
            "Accept": "application/json,text/plain",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        while not self._stop_event.is_set():
            request = urllib.request.Request(self.endpoint, method="GET", headers=headers)
            try:
                with self._opener.open(request, timeout=10) as response:
                    raw = response.read(self.max_response_bytes + 1)
                    if len(raw) > self.max_response_bytes:
                        self._record_rejection("HTTP_RESPONSE_TOO_LARGE")
                    else:
                        for message, extra in _extract_messages_from_response(_safe_decode(raw)):
                            metadata = {
                                "kind": self.transport_kind,
                                "endpoint_token": self._endpoint_token,
                                "http_status": int(getattr(response, "status", 200)),
                            }
                            metadata.update(extra)
                            self._submit(message, metadata, wait=False)
            except (urllib.error.URLError, OSError, socket.timeout) as exc:
                self._set_error("REST_POLL_FAILED", exc)
            self._stop_event.wait(self._poll_interval)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class LiveTransportRuntime:
    """Container for explicitly non-clinical, live-shadow result transports."""

    def __init__(self, kernel: ClinicalKernelProtocol, *, state_dir: Path) -> None:
        self.kernel = kernel
        self.state_dir = Path(state_dir).resolve()
        self._transports: dict[str, BaseTransport] = {}
        self._stop_signals: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def _ingest_callback(
        self, transport_id: str, config: dict[str, Any]
    ) -> Callable[[bytes, dict[str, Any]], dict[str, Any]]:
        source_id = str(config["source_id"])
        fixed_binding = config.get("binding_path")
        binding_dir_value = config.get("binding_dir")
        binding_dir = (
            Path(str(binding_dir_value)).expanduser().resolve()
            if binding_dir_value is not None
            else None
        )
        state_dir = Path(config.get("state_dir", str(self.state_dir))).resolve()
        transport_token = _redacted_token(str(transport_id))

        def _callback(raw: bytes, message_meta: dict[str, Any]) -> dict[str, Any]:
            # Keep transport bytes memory-only until the same narrow admission
            # profile used by the core has accepted them.  Import lazily so the
            # integration module can continue to construct this bridge without
            # creating a controller/integration import cycle.
            from owned_agent_clinical_control import (
                ControlError,
                parse_roche_liat_hl7,
                require_deidentified_live_shadow_message,
            )

            try:
                parsed = parse_roche_liat_hl7(raw)
                require_deidentified_live_shadow_message(raw, parsed)
            except ControlError as exc:
                raise IngestRejected(
                    _safe_code(getattr(exc, "code", None), "LIVE_SHADOW_ADMISSION_REJECTED")
                ) from None
            if binding_dir is not None:
                lookup_token = binding_lookup_token(raw)
                selected_binding = binding_dir / f"{lookup_token}.json"
                if not selected_binding.is_file():
                    raise IngestRejected("BINDING_NOT_FOUND")
            elif fixed_binding is not None:
                selected_binding = Path(str(fixed_binding)).expanduser().resolve()
            else:
                raise IngestRejected("BINDING_CONFIGURATION_MISSING")
            inbound = state_dir / "clinical-inbound"
            processing = inbound / "processing"
            rejected = inbound / "quarantine"
            processing.mkdir(parents=True, exist_ok=True)
            rejected.mkdir(parents=True, exist_ok=True)
            payload_name = f"{transport_token}-{uuid.uuid4().hex}.hl7"
            hl7_path = processing / payload_name
            hl7_path.write_bytes(raw)
            try:
                result = self.kernel.run(
                    "clinical-ingest",
                    {
                        "source_id": source_id,
                        "hl7_file": str(hl7_path),
                        "binding": str(selected_binding),
                        "state_dir": str(state_dir),
                    },
                    state_dir=str(state_dir),
                )
            except Exception:
                if hl7_path.exists():
                    hl7_path.replace(rejected / payload_name)
                raise
            if not isinstance(result, dict):
                if hl7_path.exists():
                    hl7_path.replace(rejected / payload_name)
                raise IngestRejected("INVALID_KERNEL_RESPONSE")
            inner = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            result_code = _safe_code(
                inner.get("code") or result.get("code") or ("OK" if result.get("ok") else "INGEST_REJECTED"),
                "INGEST_REJECTED",
            )
            summary = {
                "ok": bool(result.get("ok")),
                "code": result_code,
                "return_code": result.get("return_code"),
                "transport_token": transport_token,
                "transport_kind": _safe_code(config.get("kind"), "UNKNOWN").lower(),
                "payload_token": _redacted_token(raw),
                "payload_size": len(raw),
                "ingested_at": _utc_now_iso(),
                "operational_mode": LIVE_SHADOW_MODE,
                "site_validated": False,
                "clinical_use_authorized": False,
                "device_commands_enabled": False,
                "source_token": _redacted_token(source_id),
                "message_kind": _safe_code(message_meta.get("kind"), "UNKNOWN").lower(),
            }
            try:
                _atomic_json(state_dir / "oac_transport_last_payload.json", summary)
            except OSError:
                pass
            if result.get("ok"):
                hl7_path.unlink(missing_ok=True)
                return {"ok": True, "code": result_code}
            if hl7_path.exists():
                hl7_path.replace(rejected / payload_name)
            raise IngestRejected(result_code)

        return _callback

    def _create_transport(
        self,
        transport_id: str,
        kind: str,
        config: Mapping[str, Any],
        source_id: str,
        binding_path: Path | None,
        state_dir: Path,
    ) -> BaseTransport:
        callback = self._ingest_callback(
            transport_id,
            {
                "kind": kind,
                "source_id": source_id,
                "binding_path": str(binding_path) if binding_path is not None else None,
                "binding_dir": config.get("binding_dir"),
                "state_dir": str(state_dir),
            },
        )
        event = threading.Event()
        self._stop_signals[transport_id] = event
        transport_config = dict(config)
        if kind == "file-drop":
            return FileDropTransport(transport_id, transport_config, callback, stop_event=event)
        if kind in {"mllp", "mllp-listener"}:
            return MLLPListenerTransport(transport_id, transport_config, callback, stop_event=event)
        if kind == "mllp-client-receiver":
            return MLLPClientReceiverTransport(transport_id, transport_config, callback, stop_event=event)
        if kind in {"rest-poll", "http-poll"}:
            return RestPollingTransport(transport_id, transport_config, callback, stop_event=event)
        if kind in {"roche_cobas_liat_v2.0", "roche-cobas-liat-v2.0"}:
            if not bool(transport_config.get("tls_enabled", False)):
                raise TransportFailure("configured Roche-compatible profile requires tls_enabled=true")
            transport_config["required_hl7_version"] = "2.5"
            transport_config["required_message_type"] = "ORU^R30^ORU_R30"
            transport_config["single_message_per_connection"] = True
            if not transport_config.get("allowed_peer_ips"):
                raise TransportFailure(
                    "configured Roche-compatible profile requires allowed_peer_ips"
                )
            transport = MLLPListenerTransport(transport_id, transport_config, callback, stop_event=event)
            transport.profile_assertion = (
                "configured-roche-liat-host-interface-v11.3-sw3.4-3.5-not-site-validated"
            )
            return transport
        raise TransportFailure(f"unsupported transport kind {kind}")

    def start(
        self,
        transport_id: str,
        kind: str,
        config: Mapping[str, Any],
        source_id: str,
        binding_path: str | Path | None = None,
        state_dir: str | Path | None = None,
    ) -> TransportStatus:
        transport_id = str(transport_id).strip()
        kind = str(kind).strip().lower()
        if not TRANSPORT_ID_RE.fullmatch(transport_id):
            raise TransportFailure("transport_id must contain only letters, digits, dot, underscore, or hyphen")
        if not str(source_id).strip():
            raise TransportFailure("source_id is required")
        supported = {
            "file-drop", "mllp", "mllp-listener", "mllp-client-receiver",
            "rest-poll", "http-poll", "roche_cobas_liat_v2.0", "roche-cobas-liat-v2.0",
        }
        if kind not in supported:
            raise TransportFailure(f"unsupported transport kind: {kind}")
        resolved_state_dir = Path(state_dir or self.state_dir).resolve()
        resolved_state_dir.mkdir(parents=True, exist_ok=True)
        resolved_binding: Path | None = None
        if binding_path is not None:
            resolved_binding = Path(binding_path).expanduser().resolve()
            if not resolved_binding.is_file():
                raise TransportFailure("binding_path must point to an existing JSON file")
        binding_dir_value = config.get("binding_dir")
        if binding_dir_value is not None:
            binding_dir = Path(str(binding_dir_value)).expanduser().resolve()
            if not binding_dir.is_dir():
                raise TransportFailure("binding_dir must point to an existing directory")
            if resolved_binding is not None:
                raise TransportFailure("binding_path and binding_dir are mutually exclusive")
        elif resolved_binding is None:
            raise TransportFailure("binding_path or binding_dir is required")
        with self._lock:
            if transport_id in self._transports:
                raise TransportFailure(f"transport {transport_id} is already running")
            try:
                transport = self._create_transport(
                    transport_id, kind, config, str(source_id), resolved_binding, resolved_state_dir
                )
            except Exception:
                self._stop_signals.pop(transport_id, None)
                raise
            self._transports[transport_id] = transport
        try:
            transport.start()
        except Exception:
            with self._lock:
                self._transports.pop(transport_id, None)
                self._stop_signals.pop(transport_id, None)
            raise
        return transport.status()

    def stop(self, transport_id: str) -> TransportStatus | None:
        with self._lock:
            transport = self._transports.get(transport_id)
            if transport is None:
                return None
            event = self._stop_signals.pop(transport_id, None)
            if event is not None:
                event.set()
        transport.stop()
        status = transport.status()
        with self._lock:
            self._transports.pop(transport_id, None)
        return status

    def status(self, transport_id: str | None = None) -> list[TransportStatus] | TransportStatus | None:
        with self._lock:
            if transport_id is not None:
                transport = self._transports.get(str(transport_id))
                return transport.status() if transport is not None else None
            return [transport.status() for transport in self._transports.values()]

    def list(self) -> list[TransportStatus]:
        with self._lock:
            return [transport.status() for transport in self._transports.values()]
