from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ipaddress
from pathlib import Path
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


sys.path.insert(0, str(Path(__file__).resolve().parent))

from oac_live_transport_bridge import (  # noqa: E402
    FileDropTransport,
    IngestOutcome,
    IngestRejected,
    LiveTransportRuntime,
    MLLPClientReceiverTransport,
    MLLPListenerTransport,
    NoRedirectHandler,
    RestPollingTransport,
    TransportFailure,
    binding_lookup_token,
    build_mllp_ack,
)


def hl7_frame(
    control_id: str = "CONTROL-123",
    patient_id: str = "PATIENT-SECRET",
    order_id: str = "ORDER-SECRET",
    version: str = "2.5.1",
) -> bytes:
    body = (
        "MSH|^~\\&|LIAT|LAB|HOST|FAC|20260904120000-0400||"
        f"ORU^R30^ORU_R30|{control_id}|P|{version}\r"
        f"PID|||{patient_id}\r"
        "ORC|RE\r"
        f"OBR|1|{order_id}\r"
        "NTE|1\r"
        "OBX|1\r"
    ).encode("ascii")
    return b"\x0b" + body + b"\x1c\r"


def live_shadow_frame(
    *,
    include_patient_name: bool = False,
    subject_token: str = "DEID-SUBJECT01",
    order_token: str = "DEID-ORDER0001",
    report_token: str = "DEID-REPORT001",
    control_id: str = "TECH-MSG-001",
) -> bytes:
    msh = ["MSH"] + [""] * 17
    msh[1] = "^~\\&"
    msh[2] = "LIAT-DEID"
    msh[3] = "SITE-DEID"
    msh[6] = "20260904090000-0400"
    msh[8] = "ORU^R30^ORU_R30"
    msh[9] = control_id
    msh[10] = "P"
    msh[11] = "2.5"
    msh[17] = "UNICODE UTF-8"

    pid = ["PID"] + [""] * 5
    pid[3] = subject_token
    if include_patient_name:
        pid[5] = "PERSON^NAME"

    orc = ["ORC", "NW"]
    obr = ["OBR"] + [""] * 25
    obr[1] = "1"
    obr[2] = order_token
    obr[3] = report_token
    obr[4] = "SITE-FLU^Site configured influenza assay^urn:szl:site:assay"
    obr[7] = "20260904085900-0400"
    obr[11] = "O"
    obr[22] = "20260904090000-0400"
    obr[25] = "F"
    nte = [
        "NTE",
        "1",
        "L",
        "Run=RUN001;Device=LIAT001;Version=3.5.0;Tube=TUBE001;TubeExp=2099-01-01",
    ]
    obx = ["OBX"] + [""] * 11
    obx[1] = "1"
    obx[2] = "ST"
    obx[3] = "FLU-A-B^Influenza A/B result^urn:szl:site:observation"
    obx[5] = "Not detected"
    obx[11] = "F"
    body = ("\r".join("|".join(segment) for segment in (msh, pid, orc, obr, nte, obx)) + "\r").encode()
    return b"\x0b" + body + b"\x1c\r"


def wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def write_ephemeral_localhost_certificate(root: Path) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path = root / "localhost-test-cert.pem"
    key_path = root / "localhost-test-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


class AckTests(unittest.TestCase):
    def test_ack_is_deterministic_and_does_not_echo_patient_data(self) -> None:
        inbound = hl7_frame()
        first = build_mllp_ack(inbound, "AA", timestamp="20260904160000+0000")
        second = build_mllp_ack(inbound, "AA", timestamp="20260904160000+0000")
        self.assertEqual(first, second)
        self.assertIn(b"ACK^R33^ACK", first)
        self.assertIn(b"MSA|AA|CONTROL-123|NONE", first)
        self.assertNotIn(b"PATIENT-SECRET", first)
        self.assertNotIn(b"ERR|", first)

    def test_error_ack_has_err_segment(self) -> None:
        result = build_mllp_ack(
            hl7_frame(), "AE", "STORE_FAILED", timestamp="20260904160000+0000"
        )
        self.assertIn(b"MSA|AE|CONTROL-123|STORE_FAILED", result)
        self.assertIn(b"ERR|||STORE_FAILED^OAC^99OAC|E", result)


class ListenerTests(unittest.TestCase):
    def _exchange(
        self,
        ingest,
        control_id: str = "CONTROL-123",
        *,
        version: str = "2.5.1",
        extra_config: dict[str, object] | None = None,
    ) -> tuple[bytes, MLLPListenerTransport]:
        stop = threading.Event()
        config: dict[str, object] = {
            "bind_host": "127.0.0.1",
            "bind_port": 0,
            "socket_timeout": 0.1,
            "idle_timeout": 1.0,
            "ingest_timeout": 2.0,
            "queue_capacity": 2,
        }
        config.update(extra_config or {})
        transport = MLLPListenerTransport(
            "listener-test",
            config,
            ingest,
            stop_event=stop,
        )
        transport.start()
        self.assertIsNotNone(transport.bound_port)
        with socket.create_connection(("127.0.0.1", int(transport.bound_port)), timeout=2.0) as client:
            client.settimeout(2.0)
            client.sendall(hl7_frame(control_id=control_id, version=version))
            response = b""
            while not response.endswith(b"\x1c\r"):
                response += client.recv(4096)
        transport.stop()
        return response, transport

    def test_listener_sends_aa_only_after_ingest_success(self) -> None:
        seen: list[bytes] = []

        def ingest(raw: bytes, metadata: dict[str, object]) -> dict[str, object]:
            seen.append(raw)
            self.assertEqual(metadata["kind"], "mllp-listener")
            return {"ok": True}

        response, transport = self._exchange(ingest)
        self.assertEqual(seen, [hl7_frame()])
        self.assertIn(b"MSA|AA|CONTROL-123|INGESTED", response)
        status = transport.status().asdict()
        self.assertEqual(status["messages_succeeded"], 1)
        self.assertEqual(status["operational_mode"], "live-shadow")
        self.assertFalse(status["site_validated"])
        self.assertFalse(status["clinical_use_authorized"])
        self.assertFalse(status["device_commands_enabled"])

    def test_listener_sends_ae_with_err_after_ingest_failure(self) -> None:
        def ingest(_raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
            return {"ok": False, "code": "STORE_FAILED"}

        response, transport = self._exchange(ingest, "CONTROL-FAIL")
        self.assertIn(b"MSA|AE|CONTROL-FAIL|STORE_FAILED", response)
        self.assertIn(b"ERR|||STORE_FAILED^OAC^99OAC|E", response)
        self.assertEqual(transport.status().messages_failed, 1)

    def test_status_never_includes_exception_details(self) -> None:
        def ingest(_raw: bytes, _metadata: dict[str, object]) -> None:
            raise RuntimeError(r"patient PATIENT-SECRET at C:\secret\result.hl7")

        _response, transport = self._exchange(ingest, "CONTROL-REDACT")
        status_text = str(transport.status().asdict())
        self.assertNotIn("PATIENT-SECRET", status_text)
        self.assertNotIn("result.hl7", status_text)
        self.assertIn("INGEST_EXCEPTION:RuntimeError", status_text)

    def test_non_loopback_requires_explicit_gate(self) -> None:
        with self.assertRaises(TransportFailure):
            MLLPListenerTransport(
                "unsafe-bind",
                {"bind_host": "0.0.0.0", "bind_port": 0},
                lambda _raw, _meta: None,
                stop_event=threading.Event(),
            )

    def test_pinned_version_mismatch_is_rejected_before_ingest(self) -> None:
        seen: list[bytes] = []

        def ingest(raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
            seen.append(raw)
            return {"ok": True}

        response, transport = self._exchange(
            ingest,
            "CONTROL-VERSION",
            version="2.5.1",
            extra_config={"required_hl7_version": "2.5"},
        )
        self.assertEqual(seen, [])
        self.assertIn(b"MSA|AR|CONTROL-VERSION|HL7_VERSION_DENIED", response)
        self.assertIn(b"|2.5\rMSA|", response)
        self.assertEqual(transport.status().messages_failed, 1)

    def test_single_message_connection_processes_only_first_buffered_frame(self) -> None:
        seen: list[bytes] = []

        def ingest(raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
            seen.append(raw)
            return {"ok": True}

        transport = MLLPListenerTransport(
            "single-message-test",
            {
                "bind_host": "127.0.0.1",
                "bind_port": 0,
                "socket_timeout": 0.1,
                "idle_timeout": 1.0,
                "ingest_timeout": 2.0,
                "single_message_per_connection": True,
            },
            ingest,
            stop_event=threading.Event(),
        )
        self.assertTrue(transport.single_message_per_connection)
        transport.start()
        self.assertIsNotNone(transport.bound_port)
        first = hl7_frame(control_id="CONTROL-FIRST")
        second = hl7_frame(control_id="CONTROL-SECOND")
        with socket.create_connection(("127.0.0.1", int(transport.bound_port)), timeout=2.0) as client:
            client.settimeout(2.0)
            client.sendall(first + second)
            response = b""
            while not response.endswith(b"\x1c\r"):
                response += client.recv(4096)
        self.assertTrue(wait_for(lambda: len(seen) == 1))
        transport.stop()
        self.assertEqual(seen, [first])
        self.assertIn(b"MSA|AA|CONTROL-FIRST|INGESTED", response)
        self.assertNotIn(b"CONTROL-SECOND", response)

    def test_queue_failure_returns_ae_not_ar(self) -> None:
        transport = MLLPListenerTransport(
            "queue-failure-test",
            {
                "bind_host": "127.0.0.1",
                "bind_port": 0,
                "socket_timeout": 0.1,
                "idle_timeout": 1.0,
                "single_message_per_connection": True,
            },
            lambda _raw, _metadata: {"ok": True},
            stop_event=threading.Event(),
        )
        transport._submit = lambda *_args, **_kwargs: IngestOutcome(  # type: ignore[method-assign]
            False, "INGEST_QUEUE_FULL"
        )
        transport.start()
        self.assertIsNotNone(transport.bound_port)
        with socket.create_connection(("127.0.0.1", int(transport.bound_port)), timeout=2.0) as client:
            client.settimeout(2.0)
            client.sendall(hl7_frame(control_id="CONTROL-QUEUE"))
            response = b""
            while not response.endswith(b"\x1c\r"):
                response += client.recv(4096)
        transport.stop()
        self.assertIn(b"MSA|AE|CONTROL-QUEUE|INGEST_QUEUE_FULL", response)
        self.assertIn(b"ERR|||INGEST_QUEUE_FULL^OAC^99OAC|E", response)
        self.assertNotIn(b"MSA|AR|", response)


class FileDropTests(unittest.TestCase):
    def _run_timeout_case(self, result: dict[str, object], destination_name: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inbox, archive, quarantine = (root / name for name in ("inbox", "archive", "quarantine"))
            inbox.mkdir()
            entered, release = threading.Event(), threading.Event()
            calls: list[bytes] = []

            def ingest(raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
                calls.append(raw)
                entered.set()
                release.wait(30.0)
                return result

            transport = FileDropTransport(
                "pending-file-test",
                {"watch_dir": str(inbox), "archive_dir": str(archive),
                 "quarantine_dir": str(quarantine), "poll_interval": 0.05,
                 "settle_seconds": 0.0, "ingest_timeout": 1.0},
                ingest, stop_event=threading.Event(),
            )
            source = inbox / "result.hl7"
            payload = hl7_frame()
            transport.start()
            try:
                source.write_bytes(payload)
                self.assertTrue(entered.wait(2.0))
                self.assertTrue(wait_for(lambda: transport.status().last_error == "INGEST_WAIT_TIMEOUT"))
                time.sleep(0.2)  # Allow additional polls while the same work is unresolved.
                self.assertEqual(source.read_bytes(), payload)
                self.assertEqual(list(archive.glob("*.hl7")), [])
                self.assertEqual(list(quarantine.glob("*.hl7")), [])
                self.assertEqual(calls, [payload])
                self.assertEqual(transport.status().messages_total, 0)
                release.set()
                destination = archive if destination_name == "archive" else quarantine
                self.assertTrue(
                    wait_for(lambda: not transport._pending_files and not source.exists()
                             and len(list(destination.glob("*.hl7"))) == 1, timeout=10.0),
                    {"status": transport.status().asdict(), "calls": len(calls),
                     "pending_count": len(transport._pending_files)},
                )
                self.assertEqual(calls, [payload])
                self.assertEqual(transport.status().messages_total, 1)
                if destination_name == "quarantine":
                    receipt = list(quarantine.glob("*.json"))[0].read_text(encoding="utf-8")
                    self.assertIn("BINDING_REJECTED", receipt)
                    self.assertNotIn("INGEST_WAIT_TIMEOUT", receipt)
            finally:
                release.set()
                transport.stop()

    def test_wait_timeout_retains_one_pending_ingest_until_success(self) -> None:
        self._run_timeout_case({"ok": True}, "archive")

    def test_wait_timeout_quarantines_only_after_actual_rejection(self) -> None:
        self._run_timeout_case({"ok": False, "code": "BINDING_REJECTED"}, "quarantine")

    def test_archive_retry_does_not_repeat_completed_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inbox, archive = root / "inbox", root / "archive"
            inbox.mkdir()
            calls: list[bytes] = []

            def ingest(raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
                calls.append(raw)
                return {"ok": True}

            transport = FileDropTransport(
                "archive-retry-test",
                {"watch_dir": str(inbox), "archive_dir": str(archive),
                 "poll_interval": 0.05, "settle_seconds": 0.0},
                ingest, stop_event=threading.Event(),
            )
            original_move = transport._move
            move_calls = 0

            def fail_once(path: Path, destination: Path, payload: bytes) -> Path:
                nonlocal move_calls
                move_calls += 1
                if move_calls == 1:
                    raise OSError("temporary archive failure")
                return original_move(path, destination, payload)

            source = inbox / "result.hl7"
            with patch.object(transport, "_move", side_effect=fail_once):
                transport.start()
                try:
                    source.write_bytes(hl7_frame())
                    self.assertTrue(wait_for(lambda: not transport._pending_files and not source.exists()
                                             and len(list(archive.glob("*.hl7"))) == 1))
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(move_calls, 2)
                finally:
                    transport.stop()

    def test_pending_archive_backlog_is_bounded_and_leaves_new_sources_intact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inbox, archive = root / "inbox", root / "archive"
            inbox.mkdir()
            calls: list[bytes] = []
            allow_archive = threading.Event()

            def ingest(raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
                calls.append(raw)
                return {"ok": True}

            transport = FileDropTransport(
                "bounded-pending-file-test",
                {"watch_dir": str(inbox), "archive_dir": str(archive),
                 "poll_interval": 0.05, "settle_seconds": 0.0, "queue_capacity": 1},
                ingest, stop_event=threading.Event(),
            )
            original_move = transport._move

            def hold_archive(path: Path, destination: Path, payload: bytes) -> Path:
                if not allow_archive.is_set():
                    raise OSError("archive unavailable")
                return original_move(path, destination, payload)

            first, second = inbox / "a.hl7", inbox / "b.hl7"
            first_payload = hl7_frame(control_id="FIRST")
            second_payload = hl7_frame(control_id="SECOND")
            first.write_bytes(first_payload)
            second.write_bytes(second_payload)
            with patch.object(transport, "_move", side_effect=hold_archive):
                transport.start()
                try:
                    self.assertTrue(wait_for(
                        lambda: transport.status().last_error == "FILE_PENDING_CAPACITY", timeout=10.0
                    ))
                    time.sleep(0.15)
                    self.assertEqual(len(transport._pending_files), 1)
                    self.assertEqual(calls, [first_payload])
                    self.assertEqual(first.read_bytes(), first_payload)
                    self.assertEqual(second.read_bytes(), second_payload)
                    self.assertEqual(list(archive.glob("*.hl7")), [])
                    allow_archive.set()
                    self.assertTrue(
                        wait_for(lambda: not first.exists() and not second.exists(), timeout=10.0),
                        {"status": transport.status().asdict(), "calls": len(calls),
                         "pending_count": len(transport._pending_files)},
                    )
                    self.assertEqual(len(list(archive.glob("*.hl7"))), 2)
                    self.assertEqual(calls, [first_payload, second_payload])
                finally:
                    allow_archive.set()
                    transport.stop()

    def _run_case(self, result: dict[str, object], destination_name: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inbox = root / "inbox"
            archive = root / "archive"
            quarantine = root / "quarantine"
            inbox.mkdir()
            stop = threading.Event()

            def ingest(_raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
                return result

            transport = FileDropTransport(
                "file-test",
                {
                    "watch_dir": str(inbox),
                    "archive_dir": str(archive),
                    "quarantine_dir": str(quarantine),
                    "poll_interval": 0.05,
                    "settle_seconds": 0.0,
                    "ingest_timeout": 2.0,
                },
                ingest,
                stop_event=stop,
            )
            transport.start()
            source = inbox / "patient-name-must-not-survive.hl7"
            source.write_bytes(hl7_frame())
            destination = archive if destination_name == "archive" else quarantine
            self.assertTrue(
                wait_for(lambda: not transport._pending_files and not source.exists()
                         and len(list(destination.glob("*.hl7"))) == 1, timeout=10.0),
                {"status": transport.status().asdict(), "pending_count": len(transport._pending_files)},
            )
            transport.stop()
            self.assertFalse(source.exists())
            self.assertEqual(len(list(destination.glob("*.hl7"))), 1)
            self.assertNotIn("patient-name", list(destination.glob("*.hl7"))[0].name)

    def test_success_archives_only_after_ingest(self) -> None:
        self._run_case({"ok": True}, "archive")

    def test_failure_moves_source_to_quarantine(self) -> None:
        self._run_case({"ok": False, "code": "BINDING_REJECTED"}, "quarantine")

    def test_source_is_not_moved_before_ingest_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inbox = root / "inbox"
            archive = root / "archive"
            inbox.mkdir()
            entered = threading.Event()
            release = threading.Event()

            def ingest(_raw: bytes, _metadata: dict[str, object]) -> dict[str, object]:
                entered.set()
                self.assertTrue(release.wait(2.0))
                return {"ok": True}

            transport = FileDropTransport(
                "lossless-file-test",
                {
                    "watch_dir": str(inbox),
                    "archive_dir": str(archive),
                    "poll_interval": 0.05,
                    "settle_seconds": 0.0,
                    "ingest_timeout": 3.0,
                },
                ingest,
                stop_event=threading.Event(),
            )
            transport.start()
            source = inbox / "result.hl7"
            source.write_bytes(hl7_frame())
            self.assertTrue(entered.wait(2.0))
            self.assertTrue(source.exists())
            self.assertEqual(list(archive.glob("*.hl7")), [])
            release.set()
            self.assertTrue(wait_for(lambda: not source.exists() and len(list(archive.glob("*.hl7"))) == 1))
            transport.stop()


class RuntimeGateTests(unittest.TestCase):
    def test_loopback_exemption_requires_a_complete_literal_ip_address(self) -> None:
        for host in ("localhost", "127.0.0.1.example.invalid", "127.example.invalid", "127.1", "127.0.0.999"):
            constructors = (
                (MLLPListenerTransport, {"bind_host": host, "bind_port": 0}),
                (MLLPClientReceiverTransport, {"remote_host": host}),
                (RestPollingTransport, {"endpoint": f"http://{host}/results"}),
            )
            for constructor, config in constructors:
                with self.subTest(host=host, transport=constructor.__name__):
                    with self.assertRaises(TransportFailure):
                        constructor("literal-ip-test", config, lambda *_args: {"ok": True},
                                    stop_event=threading.Event())
        for host in ("127.0.0.1", "127.0.0.2", "::1"):
            with self.subTest(valid_literal=host):
                transport = MLLPListenerTransport(
                    "literal-valid-test", {"bind_host": host, "bind_port": 0},
                    lambda *_args: {"ok": True}, stop_event=threading.Event(),
                )
                self.assertEqual(transport.bind_host, host)

    def test_security_flags_reject_non_boolean_values_even_on_loopback(self) -> None:
        cases = (
            (MLLPListenerTransport, {"bind_host": "127.0.0.1"}, "allow_non_loopback"),
            (MLLPListenerTransport, {"bind_host": "127.0.0.1"}, "tls_enabled"),
            (MLLPListenerTransport, {"bind_host": "127.0.0.1"}, "tls_require_client_cert"),
            (MLLPClientReceiverTransport, {"remote_host": "127.0.0.1"}, "allow_plaintext_upstream"),
            (RestPollingTransport, {"endpoint": "https://example.invalid/results"}, "allow_insecure_http"),
        )
        for constructor, base_config, flag in cases:
            for value in ("false", "true", 0, 1, None, [], {}):
                with self.subTest(flag=flag, value=value):
                    with self.assertRaisesRegex(TransportFailure, f"{flag} must be a boolean"):
                        constructor("strict-boolean-test", {**base_config, flag: value},
                                    lambda *_args: {"ok": True}, stop_event=threading.Event())

    def test_client_certificate_requirement_cannot_be_silently_disabled_with_tls(self) -> None:
        with self.assertRaisesRegex(TransportFailure, "requires tls_enabled=true"):
            MLLPListenerTransport(
                "required-mtls-test", {"tls_enabled": False, "tls_require_client_cert": True},
                lambda *_args: {"ok": True}, stop_event=threading.Event(),
            )

    def test_rest_poll_rejects_url_credentials_and_redirects(self) -> None:
        stop = threading.Event()
        with self.assertRaisesRegex(TransportFailure, "credentials"):
            RestPollingTransport(
                "poll-with-credentials",
                {"endpoint": "https://user:password@example.invalid/results"},
                lambda *_args: {"ok": True},
                stop_event=stop,
            )

        handler = NoRedirectHandler()
        request = urllib.request.Request("https://source.example.invalid/results")
        with self.assertRaisesRegex(urllib.error.HTTPError, "redirect denied"):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://target.example.invalid/results",
            )

    def test_fixed_and_dynamic_bindings_are_mutually_exclusive(self) -> None:
        class Kernel:
            def run(self, *_args, **_kwargs):  # pragma: no cover - start is rejected first
                return {"ok": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "binding.json"
            binding.write_text("{}", encoding="utf-8")
            binding_dir = root / "bindings"
            binding_dir.mkdir()
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            with self.assertRaisesRegex(TransportFailure, "mutually exclusive"):
                runtime.start(
                    "ambiguous-binding",
                    "mllp-listener",
                    {"binding_dir": str(binding_dir)},
                    "source",
                    binding,
                )
            self.assertEqual(runtime.list(), [])

    def test_roche_compatible_profile_fails_closed_without_tls(self) -> None:
        class Kernel:
            def run(self, *_args, **_kwargs):  # pragma: no cover - start is rejected first
                return {"ok": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "binding.json"
            binding.write_text("{}", encoding="utf-8")
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            with self.assertRaises(TransportFailure):
                runtime.start(
                    "roche-shadow",
                    "roche-cobas-liat-v2.0",
                    {},
                    "source",
                    binding,
                )
            self.assertEqual(runtime.list(), [])
            self.assertEqual(runtime._stop_signals, {})

    def test_roche_compatible_profile_requires_peer_allowlist(self) -> None:
        class Kernel:
            def run(self, *_args, **_kwargs):  # pragma: no cover - start is rejected first
                return {"ok": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "binding.json"
            binding.write_text("{}", encoding="utf-8")
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            with self.assertRaisesRegex(TransportFailure, "allowed_peer_ips"):
                runtime.start(
                    "roche-no-peers",
                    "roche-cobas-liat-v2.0",
                    {"tls_enabled": True},
                    "source",
                    binding,
                )
            self.assertEqual(runtime.list(), [])
            self.assertEqual(runtime._stop_signals, {})

    def test_invalid_tls_material_fails_before_background_start(self) -> None:
        class Kernel:
            def run(self, *_args, **_kwargs):  # pragma: no cover - start is rejected first
                return {"ok": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "binding.json"
            binding.write_text("{}", encoding="utf-8")
            certificate = root / "invalid.crt"
            private_key = root / "invalid.key"
            certificate.write_text("not a certificate", encoding="utf-8")
            private_key.write_text("not a key", encoding="utf-8")
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            with self.assertRaisesRegex(TransportFailure, "TLS context"):
                runtime.start(
                    "roche-invalid-tls",
                    "roche-cobas-liat-v2.0",
                    {
                        "tls_enabled": True,
                        "tls_certfile": str(certificate),
                        "tls_keyfile": str(private_key),
                        "allowed_peer_ips": ["127.0.0.1"],
                    },
                    "source",
                    binding,
                )
            self.assertEqual(runtime.list(), [])
            self.assertEqual(runtime._stop_signals, {})

    def test_dynamic_binding_directory_selects_hashed_subject_order_file(self) -> None:
        calls: list[dict[str, object]] = []

        class Kernel:
            def run(self, _command, args, **_kwargs):
                calls.append(dict(args))
                return {"ok": True, "return_code": 0, "payload": {"ok": True}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_dir = root / "bindings"
            binding_dir.mkdir()
            raw = live_shadow_frame()
            lookup_token = binding_lookup_token(raw)
            selected = binding_dir / f"{lookup_token}.json"
            selected.write_text("{}", encoding="utf-8")
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            callback = runtime._ingest_callback(
                "binding-lookup-test",
                {
                    "kind": "mllp-listener",
                    "source_id": "source",
                    "binding_path": None,
                    "binding_dir": str(binding_dir),
                    "state_dir": str(root),
                },
            )
            result = callback(raw, {"kind": "mllp-listener"})
            self.assertTrue(result["ok"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(str(calls[0]["binding"])).resolve(), selected.resolve())

            missing_raw = live_shadow_frame(
                subject_token="DEID-UNKNOWN01",
                order_token="DEID-UNKNOWN02",
                report_token="DEID-UNKNOWN03",
                control_id="TECH-MSG-002",
            )
            with self.assertRaisesRegex(IngestRejected, "BINDING_NOT_FOUND"):
                callback(missing_raw, {"kind": "mllp-listener"})

    def test_deidentification_gate_runs_before_any_raw_disk_write(self) -> None:
        calls: list[dict[str, object]] = []
        observed_raw: list[bytes] = []

        class Kernel:
            def run(self, _command, args, **_kwargs):
                calls.append(dict(args))
                message_path = Path(str(args["hl7_file"]))
                observed_raw.append(message_path.read_bytes())
                return {"ok": True, "return_code": 0, "payload": {"ok": True}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = root / "binding.json"
            binding.write_text("{}", encoding="utf-8")
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            callback = runtime._ingest_callback(
                "pre-persistence-gate-test",
                {
                    "kind": "mllp-listener",
                    "source_id": "source",
                    "binding_path": str(binding),
                    "binding_dir": None,
                    "state_dir": str(root),
                },
            )

            rejected = live_shadow_frame(include_patient_name=True)
            with self.assertRaisesRegex(
                IngestRejected, "LIVE_SHADOW_DEIDENTIFICATION_REQUIRED"
            ):
                callback(rejected, {"kind": "mllp-listener"})
            self.assertEqual(calls, [])
            inbound = root / "clinical-inbound"
            self.assertFalse(inbound.exists())

            accepted = live_shadow_frame()
            result = callback(accepted, {"kind": "mllp-listener"})
            self.assertTrue(result["ok"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(observed_raw, [accepted])
            self.assertEqual(list(inbound.rglob("*.hl7")), [])

    def test_roche_alias_negotiates_tls_and_returns_r33_ack(self) -> None:
        calls: list[dict[str, object]] = []

        class Kernel:
            def run(self, _command, args, **_kwargs):
                calls.append(dict(args))
                return {"ok": True, "return_code": 0, "payload": {"ok": True}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certificate, private_key = write_ephemeral_localhost_certificate(root)
            binding = root / "binding.json"
            binding.write_text("{}", encoding="utf-8")
            runtime = LiveTransportRuntime(Kernel(), state_dir=root)
            status = runtime.start(
                "roche-tls-loopback",
                "roche-cobas-liat-v2.0",
                {
                    "bind_host": "127.0.0.1",
                    "bind_port": 0,
                    "tls_enabled": True,
                    "tls_certfile": str(certificate),
                    "tls_keyfile": str(private_key),
                    "allowed_peer_ips": ["127.0.0.1"],
                    "socket_timeout": 0.1,
                    "idle_timeout": 5.0,
                    "ingest_timeout": 10.0,
                },
                "source",
                binding,
            )
            self.assertEqual(status.transport_security, "tls-server-minimum-1.2")
            self.assertIn("roche-liat-host-interface-v11.3", status.profile_assertion)
            self.assertIsNotNone(status.listener_port)

            client_context = ssl.create_default_context(
                ssl.Purpose.SERVER_AUTH,
                cafile=str(certificate),
            )
            client_context.minimum_version = ssl.TLSVersion.TLSv1_2
            try:
                with socket.create_connection(
                    ("127.0.0.1", int(status.listener_port)), timeout=10.0
                ) as plain_socket:
                    with client_context.wrap_socket(
                        plain_socket, server_hostname="localhost"
                    ) as client:
                        negotiated = client.version()
                        # TLS and durable temporary-file admission share the
                        # host scheduler; the outer deadline exceeds ingestion.
                        client.settimeout(15.0)
                        client.sendall(live_shadow_frame())
                        response = b""
                        while not response.endswith(b"\x1c\r"):
                            chunk = client.recv(4096)
                            self.assertTrue(chunk, "connection closed before a complete ACK")
                            response += chunk
            finally:
                stopped = runtime.stop("roche-tls-loopback")

            self.assertIn(negotiated, {"TLSv1.2", "TLSv1.3"})
            self.assertIn(b"ACK^R33^ACK", response)
            self.assertIn(b"MSA|AA|TECH-MSG-001|INGESTED", response)
            self.assertEqual(len(calls), 1)
            self.assertIsNotNone(stopped)
            self.assertFalse(stopped.running)


if __name__ == "__main__":
    unittest.main()
