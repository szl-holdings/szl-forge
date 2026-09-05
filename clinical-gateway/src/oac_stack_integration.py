from __future__ import annotations

"""Stack integration helpers for Owned Agent Clinical Control.

This module is the programmatic integration path for the existing
`owned_agent_clinical_control.py` payload. It keeps the one-file control engine
untouched and wraps it in:

* safe subprocess execution,
* immutable local JSONL dataset capture,
* a compact non-clinical operational-evidence score,
* and a kernel that turns CLI-style commands into structured operations.
"""

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from typing import Any, Mapping


DEFAULT_SCRIPT = Path(__file__).resolve().with_name("owned_agent_clinical_control.py")
DEFAULT_STATE_DIR = Path(os.environ.get("OAC_STATE_DIR", "state")).resolve()
DEFAULT_COMMAND_TIMEOUT_SECONDS = max(
    5.0,
    min(300.0, float(os.environ.get("OAC_COMMAND_TIMEOUT_SECONDS", "60"))),
)
_DATASET_LOCKS: dict[str, threading.Lock] = {}
_DATASET_LOCKS_GUARD = threading.Lock()
DATASET_SCHEMA = "owned-agent-clinical-control/operational-observation/v1"
DATASET_SEMANTICS = "operational_observability_not_clinical_evidence"
OPERATIONAL_PAYLOAD_KEYS = frozenset(
    {
        "clinical_use_authorized",
        "code",
        "deidentified",
        "device_commands_enabled",
        "device_control",
        "direct_device_transport",
        "integrity",
        "ledger_correlation_verified",
        "mode",
        "ok",
        "operation",
        "operation_status",
        "real_phi_authorized",
        "review_trust_sealed",
        "site_validated",
        "synthetic",
        "truth_boundary",
    }
)
OPERATIONAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,160}")
STATELESS_CLI_COMMANDS = frozenset(
    {
        "clinical-capabilities",
        "clinical-self-test",
        "self-test",
    }
)
TRANSPORT_CONFIG_PATH_KEYS = frozenset(
    {
        "archive_dir",
        "binding_dir",
        "quarantine_dir",
        "tls_ca_file",
        "tls_certfile",
        "tls_keyfile",
        "watch_dir",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_operational_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only bounded scalar control facts; never persist result content."""

    projected: dict[str, Any] = {}
    for key in sorted(OPERATIONAL_PAYLOAD_KEYS):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool) or value is None:
            projected[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            projected[key] = value
        elif isinstance(value, str):
            projected[key] = (
                value if OPERATIONAL_TOKEN_RE.fullmatch(value) else "UNSAFE_VALUE_REDACTED"
            )
    return projected


def _parse_output(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = raw.strip()
    if not text:
        return None, "empty output"

    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass

    candidate_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(candidate_lines):
        try:
            return json.loads(line), ""
        except json.JSONDecodeError:
            continue

    return None, text[:4096]


def run_owned_command(
    command: str,
    args: Mapping[str, Any],
    script: Path = DEFAULT_SCRIPT,
) -> tuple[dict[str, Any], str, int]:
    state_dir = Path(args["state_dir"]).resolve()
    argv = [str(Path(sys.executable)), "-I", "-B", str(script), command]
    if command not in STATELESS_CLI_COMMANDS:
        argv.extend(["--state-dir", str(state_dir)])

    for key, raw_value in args.items():
        if key == "state_dir":
            continue
        if raw_value is None:
            continue
        if isinstance(raw_value, bool):
            if raw_value:
                argv.append(f"--{key.replace('_', '-')}")
            continue
        if isinstance(raw_value, (list, tuple)):
            for value in raw_value:
                argv.extend([f"--{key.replace('_', '-')}", str(value)])
            continue
        argv.extend([f"--{key.replace('_', '-')}", str(raw_value)])

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_payload = {
            "code": "COMMAND_TIMEOUT",
            "error": f"command exceeded {DEFAULT_COMMAND_TIMEOUT_SECONDS:g} seconds",
            "ok": False,
            "operation": command,
        }
        return timeout_payload, str(exc.stdout or ""), 124
    except OSError as exc:
        start_payload = {
            "code": "COMMAND_START_FAILED",
            "error": str(exc),
            "ok": False,
            "operation": command,
        }
        return start_payload, "", 126
    parsed, error = _parse_output(proc.stdout)
    if parsed is None:
        parsed = {
            "ok": False,
            "operation": command,
            "error": "non-json output",
            "stderr": proc.stderr.strip() or error or "",
        }
    return parsed, proc.stdout, proc.returncode


class ControlEvidenceModel:
    """Deterministic operational score; never a clinical risk/confidence model."""

    def score(self, result: Mapping[str, Any], command: str) -> float:
        if not isinstance(result, Mapping):
            return 0.0
        score = 0.0

        if result.get("ok"):
            score += 0.35
        if result.get("clinical_use_authorized") is False:
            score += 0.20
        if result.get("device_control") is False:
            score += 0.05
        if result.get("direct_device_transport") is False:
            score += 0.05
        if result.get("ledger_correlation_verified"):
            score += 0.20
        if result.get("truth_boundary") in {
            "synthetic_offline_artifact_not_clinical_delivery",
            "deidentified_live_shadow_artifact_not_clinical_delivery",
        }:
            score += 0.10
        if result.get("operation_status", "").startswith("VERIFIED_"):
            score += 0.25

        if command in {"clinical-self-test", "self-test"}:
            score -= 0.10
        if result.get("code") in {"SELF_TEST_PROCESS_NOT_RUNNING", "SELF_TEST_SUPERVISOR_EXIT"}:
            score = 0.0

        return max(0.0, min(1.0, score))


# Compatibility alias for integrations that imported the earlier local name.
ClinicalRiskModel = ControlEvidenceModel


@dataclass(frozen=True)
class DatasetRecord:
    schema: str
    dataset_semantics: str
    event_id: str
    created_utc: str
    command: str
    state_dir: str
    return_code: int
    request_hash: str
    ok: bool
    operation: str
    control_evidence_score: float
    score_semantics: str
    payload: dict[str, Any]


class ClinicalDataset:
    """JSONL event log used as the stack data set."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "oac_integration_dataset.jsonl"
        key = str(self.path.resolve())
        with _DATASET_LOCKS_GUARD:
            self._lock = _DATASET_LOCKS.setdefault(key, threading.Lock())

    def append(self, record: DatasetRecord) -> Path:
        encoded = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as fp:
                fp.write(encoded)
                fp.flush()
                os.fsync(fp.fileno())
        return self.path

    def tail(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        max_lines = max(1, int(limit))
        lines = deque[str](maxlen=max_lines)
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if text:
                    lines.append(text)
        out: list[dict[str, Any]] = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def summary(self, limit: int = 50) -> dict[str, Any]:
        rows = self.tail(limit=limit)
        ok_count = sum(1 for row in rows if bool(row.get("ok")))
        return {
            "schema": DATASET_SCHEMA,
            "dataset_semantics": DATASET_SEMANTICS,
            "dataset_rows": len(rows),
            "ok_rows": ok_count,
            "bad_rows": len(rows) - ok_count,
            "latest": rows[-1] if rows else None,
            "truth_boundary": "local_observability_jsonl_not_clinical_evidence_store",
        }


class ClinicalKernel:
    """High-level orchestrator used by API/front-end integration points."""

    def __init__(
        self,
        state_dir: Path | str | None = None,
        script: Path | None = None,
        data_root: Path | str | None = None,
    ) -> None:
        self.default_state_dir = Path(state_dir or DEFAULT_STATE_DIR).resolve()
        self.data_root = Path(data_root or self.default_state_dir.parent).resolve()
        try:
            self.default_state_dir.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError("default state_dir must stay within data_root") from exc
        self.script = Path(script or DEFAULT_SCRIPT).resolve()
        self.model = ControlEvidenceModel()
        self.default_dataset = ClinicalDataset(self.default_state_dir)

        # Import here to avoid a circular import if the transport layer imports this
        # orchestrator class.
        from oac_live_transport_bridge import LiveTransportRuntime

        self._transport_runtime_class = LiveTransportRuntime
        self._runtime_by_state: dict[str, Any] = {}
        self._runtime_lock = threading.Lock()

    def _dataset_for(self, state_dir: Path) -> ClinicalDataset:
        return ClinicalDataset(state_dir)

    def _resolve_data_path(self, value: Path | str, label: str) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} must not be empty")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.data_root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError(f"{label} must stay within data_root") from exc
        return resolved

    def _resolve_state_dir(self, state_dir: Path | str | None = None) -> Path:
        return self._resolve_data_path(
            state_dir or self.default_state_dir,
            "state_dir",
        )

    def normalize_transport_config(
        self,
        config: Mapping[str, Any] | None,
        *,
        binding_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Normalize bridge configuration without starting a transport.

        Relative filesystem paths are rooted at ``data_root`` and existing
        symlink components are resolved before the confinement check.  The
        former ``inbox_dir`` spelling is rejected because the bridge consumes
        ``watch_dir``.
        """

        normalized = dict(config or {})
        if "inbox_dir" in normalized:
            raise ValueError("config.inbox_dir is unsupported; use config.watch_dir")

        configured_binding_dir = normalized.get("binding_dir")
        if binding_dir is not None and configured_binding_dir is not None:
            requested = self._resolve_data_path(binding_dir, "binding_dir")
            configured = self._resolve_data_path(configured_binding_dir, "binding_dir")
            if requested != configured:
                raise ValueError("binding_dir is defined more than once with different values")
            normalized["binding_dir"] = str(requested)
        elif binding_dir is not None:
            normalized["binding_dir"] = str(
                self._resolve_data_path(binding_dir, "binding_dir")
            )

        for key in TRANSPORT_CONFIG_PATH_KEYS:
            value = normalized.get(key)
            if value is not None:
                normalized[key] = str(self._resolve_data_path(value, key))
        return normalized

    def _transport_runtime(self, state_dir: Path | str | None = None) -> Any:
        state_root = str(self._resolve_state_dir(state_dir))
        with self._runtime_lock:
            runtime = self._runtime_by_state.get(state_root)
            if runtime is None:
                runtime = self._transport_runtime_class(self, state_dir=Path(state_root))
                self._runtime_by_state[state_root] = runtime
            return runtime

    def transport_start(
        self,
        *,
        transport_id: str,
        kind: str,
        source_id: str,
        binding_path: str | Path | None = None,
        binding_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        state_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        target_state = self._resolve_state_dir(state_dir)
        normalized_config = self.normalize_transport_config(
            config,
            binding_dir=binding_dir,
        )
        resolved_binding: Path | None = None
        if binding_path is not None:
            resolved_binding = self._resolve_data_path(binding_path, "binding_path")
        has_binding_dir = normalized_config.get("binding_dir") is not None
        if resolved_binding is None and not has_binding_dir:
            raise ValueError("binding_path or binding_dir is required")
        if resolved_binding is not None and has_binding_dir:
            raise ValueError("binding_path and binding_dir are mutually exclusive")

        runtime = self._transport_runtime(target_state)
        status = runtime.start(
            transport_id=transport_id,
            kind=kind,
            config=normalized_config,
            source_id=source_id,
            binding_path=resolved_binding,
            state_dir=target_state,
        )
        return {"ok": True, "operation": "transport-start", "status": status.asdict()}

    def transport_stop(self, *, transport_id: str, state_dir: Path | str | None = None) -> dict[str, Any]:
        runtime = self._transport_runtime(self._resolve_state_dir(state_dir))
        status = runtime.stop(transport_id=transport_id)
        if status is None:
            return {
                "code": "TRANSPORT_NOT_FOUND",
                "ok": False,
                "operation": "transport-stop",
                "transport_id": transport_id,
            }
        return {"ok": True, "operation": "transport-stop", "status": status.asdict()}

    def transport_status(
        self,
        transport_id: str | None = None,
        state_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        runtime = self._transport_runtime(self._resolve_state_dir(state_dir))
        status = runtime.status(transport_id)
        if status is None:
            return {
                "ok": False,
                "operation": "transport-status",
                "transport_id": transport_id,
            }
        if isinstance(status, list):
            payload = [item.asdict() for item in status]
        else:
            payload = status.asdict()
        return {"ok": True, "operation": "transport-status", "transports": payload}

    def transport_list(
        self,
        state_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        target_state = self._resolve_state_dir(state_dir)
        runtime = self._transport_runtime(target_state)
        return {
            "ok": True,
            "operation": "transport-list",
            "transports": [item.asdict() for item in runtime.list()],
            "state_dir": str(target_state),
        }

    def run(
        self,
        command: str,
        args: Mapping[str, Any] | None = None,
        *,
        state_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        arguments = dict(args or {})
        target_state = self._resolve_state_dir(
            state_dir or arguments.pop("state_dir", None) or self.default_state_dir
        )
        arguments["state_dir"] = str(target_state)

        payload, stdout, rc = run_owned_command(command, arguments, self.script)
        operation = str(payload.get("operation", command))
        ok = bool(payload.get("ok")) and rc == 0
        payload_hash = stable_hash({"command": command, "args": arguments, "stdout": stdout, "rc": rc})
        evidence_score = self.model.score(payload, command)

        record = DatasetRecord(
            schema=DATASET_SCHEMA,
            dataset_semantics=DATASET_SEMANTICS,
            event_id=stable_hash(
                {
                    "state_dir": str(target_state),
                    "command": command,
                    "ts": utc_now_iso(),
                    "rc": rc,
                }
            ),
            created_utc=utc_now_iso(),
            command=command,
            state_dir=str(target_state),
            return_code=rc,
            request_hash=payload_hash,
            ok=ok,
            operation=operation,
            control_evidence_score=evidence_score,
            score_semantics="deterministic_operational_evidence_not_clinical_confidence",
            payload=project_operational_payload(payload),
        )

        dataset = self._dataset_for(target_state)
        dataset.append(record)

        result = {
            "ok": ok,
            "operation": operation,
            "command": command,
            "state_dir": str(target_state),
            "control_evidence_score": evidence_score,
            "return_code": rc,
            "dataset_event_id": record.event_id,
            "dataset_path": str(dataset.path),
            "dataset_schema": DATASET_SCHEMA,
            "dataset_semantics": DATASET_SEMANTICS,
            "payload": payload,
            "score_semantics": "deterministic_operational_evidence_not_clinical_confidence",
            "truth_boundary": "local_command_observation_not_clinical_validation",
        }
        return result

    def run_with_default(self, command: str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.run(command, args=args, state_dir=self.default_state_dir)

    def dataset_tail(self, state_dir: Path | str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        target_state = self._resolve_state_dir(state_dir)
        return self._dataset_for(target_state).tail(limit=limit)

    def dataset_summary(self, state_dir: Path | str | None = None) -> dict[str, Any]:
        target_state = self._resolve_state_dir(state_dir)
        return self._dataset_for(target_state).summary()
