"""Offline controls for native-host samples and a sticky fail-closed lease."""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from host_memory_test_fixture import add_host_memory_fixture

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ra3_host_memory_guard_test", HERE / "host_memory_guard.py"
)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)
RUN = "a" * 32
SOURCE = "b" * 40
SAMPLER = "c" * 64


def sample(sequence=0, native_ticks=10_000_000, **updates):
    result = {
        "schema": "szl.windows-host-memory-sample/v1",
        "runId": RUN,
        "sourceRevision": SOURCE,
        "samplerSourceSha256": SAMPLER,
        "observedAt": "2026-10-04T12:00:00+00:00",
        "sequence": sequence,
        "nativeMonotonicTicks": native_ticks,
        "nativeMonotonicFrequency": 10_000_000,
        "availablePhysicalBytes": 12 * 1024**3,
        "totalPhysicalBytes": 32 * 1024**3,
        "committedBytes": 12 * 1024**3,
        "commitLimitBytes": 48 * 1024**3,
        "availableCommitBytes": 36 * 1024**3,
    }
    result.update(updates)
    return result


NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


class NativeMemoryLeaseTests(unittest.TestCase):
    def lease(self):
        return guard.MemoryLease(
            run_id=RUN, source_revision=SOURCE, sampler_sha256=SAMPLER
        )

    def test_native_physical_and_system_commit_are_both_required(self):
        for field in ("availablePhysicalBytes", "availableCommitBytes"):
            lease = self.lease()
            row = sample(**{field: 8 * 1024**3 - 1})
            if field == "availableCommitBytes":
                row["committedBytes"] = row["commitLimitBytes"] - row[field]
            with self.assertRaisesRegex(guard.HostMemoryError, "HOST_MEMORY_PRESSURE"):
                lease.observe(row, received_ns=1_000_000_000, now=NOW)
            self.assertEqual(lease.failure, "HOST_MEMORY_PRESSURE")

    def test_missing_stale_future_and_bad_native_samples_deny(self):
        for update in (
            {"availableCommitBytes": None},
            {"availableCommitBytes": True},
            {"observedAt": "2026-10-04T11:59:30+00:00"},
            {"observedAt": "2026-10-04T12:00:30+00:00"},
            {"runId": "d" * 32},
            {"sourceRevision": "d" * 40},
            {"samplerSourceSha256": "d" * 64},
            {"commitLimitBytes": 1},
            {"availablePhysicalBytes": 40 * 1024**3},
        ):
            with self.subTest(update=update):
                with self.assertRaises(guard.HostMemoryError):
                    self.lease().observe(
                        sample(**update), received_ns=1_000_000_000, now=NOW
                    )

    def test_a_failure_latches_and_healthy_bytes_cannot_clear_it(self):
        lease = self.lease()
        lease.observe(sample(), received_ns=1_000_000_000, now=NOW)
        lease.fail("HOST_MEMORY_WATCHER_EXITED")
        with self.assertRaisesRegex(
            guard.HostMemoryError, "HOST_MEMORY_WATCHER_EXITED"
        ):
            lease.observe(sample(1, 30_000_000), received_ns=3_000_000_000, now=NOW)
        with self.assertRaises(guard.HostMemoryError):
            lease.assert_fresh(3_000_000_000)

    def test_missing_lease_and_dead_watcher_fail_closed(self):
        with self.assertRaises(guard.HostMemoryError):
            self.lease().assert_fresh(1_000_000_000)
        lease = self.lease()
        lease.observe(sample(), received_ns=1_000_000_000, now=NOW)
        with self.assertRaisesRegex(guard.HostMemoryError, "HOST_MEMORY_LEASE_EXPIRED"):
            lease.assert_fresh(9_000_000_001)

    def test_replay_sequence_native_clock_and_receive_gap_fail(self):
        for next_row, at in (
            (sample(), 2_000_000_000),
            (sample(1, 9_000_000), 3_000_000_000),
            (sample(1, 30_000_000), 10_000_000_000),
        ):
            lease = self.lease()
            lease.observe(sample(), received_ns=1_000_000_000, now=NOW)
            with self.assertRaises(guard.HostMemoryError):
                lease.observe(next_row, received_ns=at, now=NOW)

    def test_exact_floor_admits_but_is_not_capacity_or_release_clearance(self):
        lease = self.lease()
        row = sample(
            availablePhysicalBytes=8 * 1024**3,
            availableCommitBytes=8 * 1024**3,
            committedBytes=40 * 1024**3,
        )
        lease.observe(row, received_ns=1_000_000_000, now=NOW)
        lease.assert_fresh(1_000_000_000)
        public = lease.public()
        self.assertEqual(public["minimumAvailablePhysicalBytes"], 8 * 1024**3)
        self.assertFalse(public["fullRunCapacityProven"])
        self.assertFalse(public["qualificationEligible"])


class GuardEvidenceTests(unittest.TestCase):
    def test_strict_native_json_rejects_duplicate_and_nonfinite_keys(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"x":NaN}', b"[]", b"\xff"):
            with self.assertRaises(guard.HostMemoryError):
                guard.decode_object(raw)

    def test_failure_witness_precedes_exact_worker_stop_and_outer_fallback(self):
        events = []

        class Evidence:
            def abort(self, reason):
                events.append(("abort", reason))

            def lease(self, lease):
                events.append(("lease", lease.failure))

            def terminal(self, receipt):
                events.append(("terminal", receipt["state"]))

        class Units:
            worker_identity = None
            identity = {}

            def stop_worker(self):
                events.append(("stop", "worker"))
                return {"cgroupEmptyConfirmed": False}

            def stop_outer(self):
                events.append(("stop", "outer"))

        session = guard.GuardSession(
            guard.MemoryLease(RUN, SOURCE, SAMPLER), Evidence(), Units()
        )
        session.fail("HOST_MEMORY_PRESSURE")
        self.assertLess(
            events.index(("lease", "HOST_MEMORY_PRESSURE")),
            events.index(("stop", "worker")),
        )
        self.assertLess(
            events.index(("stop", "worker")), events.index(("stop", "outer"))
        )
        self.assertLess(
            events.index(("terminal", "FAILED")), events.index(("stop", "outer"))
        )

    def test_an_abort_is_write_once_and_vetoes_a_healthy_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = guard.Evidence(Path(folder), RUN, SOURCE)
            evidence.abort("HOST_MEMORY_ABORTED")
            original = evidence.abort_path.read_bytes()
            evidence.abort("HOST_MEMORY_PRESSURE")
            self.assertEqual(evidence.abort_path.read_bytes(), original)
            with self.assertRaises(guard.HostMemoryError):
                evidence.require_no_abort()

    def test_lease_and_terminal_artifacts_are_bound_and_not_mutable_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = guard.Evidence(Path(folder), RUN, SOURCE)
            lease = guard.MemoryLease(RUN, SOURCE, SAMPLER)
            lease.observe(sample(), received_ns=1_000_000_000, now=NOW)
            evidence.lease(lease)
            self.assertEqual(json.loads(evidence.lease_path.read_bytes())["runId"], RUN)
            evidence.terminal({"state": "FAILED"})
            with self.assertRaises(FileExistsError):
                evidence.terminal({"state": "FINAL_HEALTHY"})

    def test_report_parser_rejects_resource_and_identity_tampering_after_rehash(self):
        supervisor = {"runId": RUN}
        original = add_host_memory_fixture(supervisor, SOURCE, RUN)
        mutations = (
            lambda r: r["samples"][-1]["native"].update(availablePhysicalBytes=1),
            lambda r: r["samples"][-1]["native"].update(
                availableCommitBytes=1, committedBytes=48 * 1024**3 - 1
            ),
            lambda r: r["samples"][-1]["native"].update(
                observedAt="2026-08-13T00:00:00+00:00"
            ),
            lambda r: r["samples"][-1]["native"].update(sequence=0),
            lambda r: r["samples"][-1]["native"].update(nativeMonotonicTicks=1),
            lambda r: r["samples"][-1]["native"].update(sourceRevision="f" * 40),
            lambda r: r.update(observedMonotonicNs=150_000_000_000),
            lambda r: r.update(qualificationEligible=True),
            lambda r: r.update(workerCgroupEmptyConfirmed=False),
            lambda r: r["workerIdentity"].update(unit="unrelated.service"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                receipt, parent = copy.deepcopy(original), copy.deepcopy(supervisor)
                mutate(receipt)
                receipt["lease"]["lastSample"] = copy.deepcopy(receipt["samples"][-1])
                receipt["lease"]["samplesSha256"] = guard.digest(receipt["samples"])
                receipt.pop("reportSha256")
                receipt["reportSha256"] = guard.digest(receipt)
                raw = guard.canonical_bytes(receipt) + b"\n"
                parent["hostMemoryGuard"].update(
                    fileSha256=guard.hashlib.sha256(raw).hexdigest(), bytes=len(raw)
                )
                with self.assertRaises(guard.HostMemoryError):
                    guard.validate_report_bundle(parent, receipt, SOURCE)

    def test_guard_exit_and_complete_runtime_coverage_are_required(self):
        parent = {"runId": RUN}
        receipt = add_host_memory_fixture(parent, SOURCE, RUN)
        parent["hostMemoryGuard"]["helperExitConfirmed"] = False
        with self.assertRaises(guard.HostMemoryError):
            guard.validate_report_bundle(parent, receipt, SOURCE)
        parent["hostMemoryGuard"]["helperExitConfirmed"] = True
        parent["launch"]["startedMonotonicNs"] = 0
        receipt["finalizeRequest"]["workerStartedMonotonicNs"] = 0
        receipt.pop("reportSha256")
        receipt["reportSha256"] = guard.digest(receipt)
        raw = guard.canonical_bytes(receipt) + b"\n"
        parent["hostMemoryGuard"].update(
            fileSha256=guard.hashlib.sha256(raw).hexdigest(), bytes=len(raw)
        )
        with self.assertRaises(guard.HostMemoryError):
            guard.validate_report_bundle(parent, receipt, SOURCE)

    def test_evidence_failure_does_not_delay_exact_worker_stop(self):
        writes, units = mock.Mock(), mock.Mock()
        writes.abort.side_effect = OSError("synthetic disk full")
        writes.lease.side_effect = OSError("synthetic disk full")
        writes.terminal.side_effect = OSError("synthetic disk full")
        units.stop_worker.return_value = {"cgroupEmptyConfirmed": True}
        units.identity, units.worker_identity = {}, None
        session = guard.GuardSession(
            guard.MemoryLease(RUN, SOURCE, SAMPLER), writes, units
        )
        session.fail("HOST_MEMORY_WATCHER_EXITED")
        units.stop_worker.assert_called_once()
        units.stop_outer.assert_not_called()

    def test_healthy_seal_requires_exact_completion_request_and_successful_helper_exit(
        self,
    ):
        for scenario in (
            "success",
            "crash_after_seal",
            "wrong_request",
            "empty_helper_invocation",
            "missing_helper_invocation",
        ):
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as folder,
            ):
                supervisor = {"runId": RUN}
                receipt = add_host_memory_fixture(supervisor, SOURCE, RUN)
                reports = Path(folder)
                source = SimpleNamespace(
                    revision=SOURCE,
                    component=lambda name: SimpleNamespace(
                        sha256=supervisor["source"]["components"][name]["sha256"]
                    ),
                )
                client = guard.GuardClient(
                    policy={},
                    attempt=SimpleNamespace(reports=reports, run_id=RUN),
                    source=source,
                )
                client.units.identity = receipt["units"]
                if scenario == "wrong_request":
                    receipt["finalizeRequest"]["requestedMonotonicNs"] += 1
                    receipt.pop("reportSha256")
                    receipt["reportSha256"] = guard.digest(receipt)
                guard.write_json(client.evidence.terminal_path, receipt)
                properties = {
                    "LoadState": "loaded",
                    "Result": "success",
                    "ExecMainStatus": "0",
                    "MainPID": "0",
                    "ActiveState": "active",
                    "SubState": "exited",
                    "InvocationID": receipt["units"]["helper"]["invocationId"],
                }
                if scenario == "crash_after_seal":
                    properties.update(
                        Result="signal",
                        ExecMainStatus="9",
                        ActiveState="failed",
                        SubState="failed",
                    )
                if scenario == "empty_helper_invocation":
                    properties["InvocationID"] = ""
                if scenario == "missing_helper_invocation":
                    properties.pop("InvocationID")
                with (
                    mock.patch.object(client, "assert_healthy"),
                    mock.patch.object(
                        client.units, "properties", return_value=properties
                    ),
                    mock.patch.object(
                        guard.time, "monotonic_ns", return_value=123_000_000_000
                    ),
                ):
                    if scenario == "success":
                        self.assertTrue(
                            client.finalize(supervisor["launch"])["helperExitConfirmed"]
                        )
                    else:
                        reason = (
                            "WATCHER_EXITED"
                            if scenario == "crash_after_seal"
                            else "IDENTITY_MISMATCH"
                        )
                        with self.assertRaisesRegex(guard.HostMemoryError, reason):
                            client.finalize(supervisor["launch"])


class UnitStopControls(unittest.TestCase):
    def properties(self, invocation="d" * 32):
        names = guard.unit_names(RUN)
        return {
            "Id": names["worker"],
            "InvocationID": invocation,
            "ControlGroup": "/user.slice/" + names["worker"],
            "LoadState": "loaded",
            "BindsTo": names["supervisor"] + " " + names["host-memory"],
            "After": names["supervisor"] + " " + names["host-memory"],
            "KillMode": "control-group",
            "SendSIGKILL": "yes",
        }

    def test_replacement_invocation_is_never_signalled(self):
        units = guard.UnitControl(RUN)
        units.worker_identity = units.identify(self.properties())
        with (
            mock.patch.object(
                units, "properties", return_value=self.properties("e" * 32)
            ),
            mock.patch.object(guard.subprocess, "run") as run,
        ):
            result = units.stop_worker()
        self.assertFalse(result["cgroupEmptyConfirmed"])
        run.assert_not_called()

    def test_stop_timeout_escalates_only_the_same_worker_and_confirms_empty(self):
        units = guard.UnitControl(RUN)
        with (
            mock.patch.object(units, "properties", return_value=self.properties()),
            mock.patch.object(units, "empty", return_value=True),
            mock.patch.object(
                guard.subprocess,
                "run",
                side_effect=[
                    subprocess.TimeoutExpired("systemctl", 20),
                    subprocess.CompletedProcess([], 0),
                ],
            ) as run,
        ):
            result = units.stop_worker()
        self.assertTrue(result["cgroupEmptyConfirmed"])
        self.assertEqual(
            run.call_args_list[0].args[0][-1], guard.unit_names(RUN)["worker"]
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                guard.SYSTEMCTL,
                "--user",
                "kill",
                "--signal=KILL",
                guard.unit_names(RUN)["worker"],
            ],
        )

    def test_identity_change_during_stop_prevents_kill_escalation(self):
        units = guard.UnitControl(RUN)
        with (
            mock.patch.object(
                units,
                "properties",
                side_effect=[self.properties(), self.properties("e" * 32)],
            ),
            mock.patch.object(
                guard.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("systemctl", 20),
            ) as run,
        ):
            result = units.stop_worker()
        self.assertFalse(result["cgroupEmptyConfirmed"])
        self.assertEqual(run.call_count, 1)

    def test_missing_guard_dependency_is_an_identity_failure(self):
        units = guard.UnitControl(RUN)
        value = self.properties()
        value["BindsTo"] = guard.unit_names(RUN)["supervisor"]
        with (
            mock.patch.object(units, "properties", return_value=value),
            self.assertRaises(guard.HostMemoryError),
        ):
            units.observe_worker()

    def test_real_dummy_worker_stops_under_pressure_without_touching_unrelated_process(
        self,
    ):
        self.assert_dummy_worker_stopped("HOST_MEMORY_PRESSURE")

    def test_native_watcher_death_stops_a_real_dummy_worker(self):
        self.assert_dummy_worker_stopped("HOST_MEMORY_WATCHER_EXITED")

    def assert_dummy_worker_stopped(self, reason):
        # Synthetic native frame, real pipe and subprocesses; systemd/cgroups are
        # represented by the exact-unit adapter here, never claimed as hardware proof.
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                reports = Path(directory) / RUN / "reports"
                reports.mkdir(parents=True)
                events = []
                units = mock.Mock()
                units.identity, units.worker_identity = {}, None

                def stop_worker():
                    witness = json.loads(
                        (reports / "host-memory-lease.json").read_text()
                    )
                    events.append(witness["failure"])
                    worker.terminate()
                    worker.wait(timeout=3)
                    return {"cgroupEmptyConfirmed": True}

                units.stop_worker.side_effect = stop_worker
                source = SimpleNamespace(
                    revision=SOURCE, component=lambda _: SimpleNamespace(sha256=SAMPLER)
                )
                available = 1 if reason == "HOST_MEMORY_PRESSURE" else 12 * 1024**3
                row = sample(
                    availablePhysicalBytes=available,
                    observedAt=datetime.now(timezone.utc).isoformat(),
                )
                tail = (
                    "sys.stdin.read()"
                    if reason == "HOST_MEMORY_PRESSURE"
                    else "sys.exit(74)"
                )
                script = (
                    "import sys; print("
                    + repr(json.dumps(row))
                    + ", flush=True); "
                    + tail
                )
                with (
                    mock.patch.object(guard, "RUNS_ROOT", Path(directory)),
                    mock.patch.object(guard, "UnitControl", return_value=units),
                    mock.patch.object(
                        guard,
                        "validated_interop_socket",
                        return_value="/run/WSL/123_interop",
                    ),
                    mock.patch.object(
                        guard,
                        "native_command",
                        return_value=[sys.executable, "-c", script],
                    ),
                    mock.patch.object(guard.signal, "signal"),
                ):
                    self.assertEqual(guard.run_guard(source, RUN), 81)
                self.assertEqual(events, [reason])
                self.assertIsNotNone(worker.poll())
                self.assertIsNone(unrelated.poll())
                self.assertEqual(
                    json.loads((reports / "host-memory-terminal.json").read_text())[
                        "state"
                    ],
                    "FAILED",
                )
        finally:
            for process in (worker, unrelated):
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
