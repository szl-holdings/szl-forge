"""Synthetic native-observation fixtures for offline tests; never run evidence."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location(
    "szl_host_memory_fixture_guard", Path(__file__).with_name("host_memory_guard.py")
)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)


def add_host_memory_fixture(supervisor, source_revision, run_id):
    """Add a deterministic healthy fixture and return its separately sealed report."""
    names = guard.unit_names(run_id)
    source = supervisor.setdefault("source", {"revision": source_revision})
    components = source.setdefault("components", {})
    for name in ("host_memory_guard.py", "windows_host_memory_sampler.ps1"):
        components.setdefault(
            name, {"bytes": 1, "sha256": guard.digest({"syntheticComponent": name})}
        )
    launch = supervisor.setdefault("launch", {})
    launch.setdefault("workerControlGroup", "/user.slice/" + names["worker"])
    launch.setdefault("startedMonotonicNs", 2_000_000_000)
    launch.setdefault("endedMonotonicNs", 122_000_000_000)
    lease = guard.MemoryLease(
        run_id, source_revision, components["windows_host_memory_sampler.ps1"]["sha256"]
    )
    origin = datetime(2026, 8, 13, 11, 59, 59, tzinfo=timezone.utc)
    for sequence in range(63):
        at = origin + timedelta(seconds=sequence * 2)
        lease.observe(
            {
                "schema": guard.SAMPLE_SCHEMA,
                "runId": run_id,
                "sourceRevision": source_revision,
                "samplerSourceSha256": lease.sampler_sha256,
                "observedAt": at.isoformat(),
                "sequence": sequence,
                "nativeMonotonicTicks": 10_000_000 + sequence * 20_000_000,
                "nativeMonotonicFrequency": 10_000_000,
                "availablePhysicalBytes": 12 * 1024**3,
                "totalPhysicalBytes": 32 * 1024**3,
                "committedBytes": 12 * 1024**3,
                "commitLimitBytes": 48 * 1024**3,
                "availableCommitBytes": 36 * 1024**3,
            },
            received_ns=1_000_000_000 + sequence * 2_000_000_000,
            now=at,
        )

    def identity(kind):
        return {
            "unit": names[kind],
            "invocationId": "d" * 32,
            "controlGroup": "/user.slice/" + names[kind],
        }

    units = SimpleNamespace(
        identity={
            "helper": identity("host-memory"),
            "supervisor": identity("supervisor"),
        },
        worker_identity={
            **identity("worker"),
            "controlGroup": launch["workerControlGroup"],
        },
    )
    session = guard.GuardSession(
        lease, None, units, components["host_memory_guard.py"]["sha256"]
    )
    request = {
        "runId": run_id,
        "sourceRevision": source_revision,
        "requestedMonotonicNs": 123_000_000_000,
        "workerStartedMonotonicNs": launch["startedMonotonicNs"],
        "workerEndedMonotonicNs": launch["endedMonotonicNs"],
    }
    receipt = session.receipt("FINAL_HEALTHY", now_ns=125_000_000_000, request=request)
    receipt["observedAt"] = (origin + timedelta(seconds=124)).isoformat()
    receipt["reportSha256"] = guard.digest(receipt)
    raw = guard.canonical_bytes(receipt) + b"\n"
    supervisor["hostMemoryGuard"] = {
        "relativePath": "reports/host-memory-terminal.json",
        "fileSha256": guard.hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "state": "FINAL_HEALTHY",
        "helperExitConfirmed": True,
    }
    return receipt


def evaluation_guard_linkage(receipt):
    return {"state": "FINAL_HEALTHY", "terminalReportSha256": receipt["reportSha256"]}
