# SPDX-License-Identifier: Apache-2.0
"""SIMULATED source fixtures only; no live incumbent or GPU qualification."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "frontier/msq650_admission/coexistence.py"
SPEC = importlib.util.spec_from_file_location("_msq650_coexistence_tests", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture():
    identity = {"hostname": "fixture-host", "boot_id": "11111111-1111-1111-1111-111111111111",
        "gpu_uuid": "GPU-22222222-2222-2222-2222-222222222222", "pid": 12, "start_ticks": 500,
        "executable_sha256": "a" * 64, "model_sha256": "b" * 64}
    plan = {"schema": "szl.msq650.coexistence-plan/v1", "candidate_id": MODULE.CANDIDATE,
        "manifest_sha256": MODULE.MANIFEST_SHA256, "campaign_id": "33333333-3333-3333-3333-333333333333",
        "lease_id": "44444444-4444-4444-4444-444444444444", "not_before_epoch": 1000,
        "expires_at_epoch": 1100, "incumbent": identity, "worker_stop_scope": "OWNED_WORKER_SESSION_ONLY",
        "resources": {"worker_gpu_memory_bytes": 10, "incumbent_extra_gpu_reserve_bytes": 10,
            "gpu_safety_bytes": 5, "worker_host_memory_bytes": 10, "host_safety_bytes": 5,
            "worker_disk_bytes": 10, "disk_safety_bytes": 5, "worker_cpu_threads": 1}}
    before = {"schema": "szl.msq650.prelaunch-snapshot/v1", "observed_epoch": 1001,
        "incumbent": copy.deepcopy(identity), "compute_processes": [{"pid": 12, "start_ticks": 500}],
        "gpu_total_bytes": 100, "gpu_free_bytes": 50, "host_available_bytes": 100,
        "disk_free_bytes": 100, "cpu_threads_available": 4, "health": {"ready": True, "active_requests": 0}}
    after = copy.deepcopy(before)
    after["observed_epoch"] = 1002
    return plan, before, after


def observe(plan, before, after, now=1003):
    raw = MODULE.canonical(plan)
    return MODULE.observe_coexistence(raw, MODULE.canonical(before), MODULE.canonical(after),
        expected_plan_sha256=hashlib.sha256(raw).hexdigest(), now_epoch=now)


class CoexistenceTests(unittest.TestCase):
    def test_consistent_fixture_never_admits_or_coerces(self):
        result = observe(*fixture())
        self.assertTrue(result.plan_consistent)
        self.assertFalse(result.execution_authority or result.training_admitted or result.publication_eligible)
        self.assertEqual((result.evidence_class, result.state, result.isolation_backend), ("MODELED", "BLOCKED", "UNAVAILABLE"))
        with self.assertRaises(TypeError):
            bool(result)
        with self.assertRaises(FrozenInstanceError):
            result.training_admitted = True

    def test_pid_reuse_or_identity_change_denied(self):
        for key, value in (("pid", 13), ("start_ticks", 501), ("hostname", "other"),
            ("boot_id", "55555555-5555-5555-5555-555555555555"),
            ("gpu_uuid", "GPU-55555555-5555-5555-5555-555555555555"),
            ("model_sha256", "c" * 64), ("executable_sha256", "d" * 64)):
            p, b, a = fixture()
            a["incumbent"][key] = value
            with self.subTest(key=key):
                self.assertFalse(observe(p, b, a).plan_consistent)

    def test_competing_or_reused_gpu_process_denied(self):
        for processes in ([], [{"pid": 12, "start_ticks": 501}],
                          [{"pid": 12, "start_ticks": 500}] * 2,
                          [{"pid": 13, "start_ticks": 500}]):
            p, b, a = fixture()
            a["compute_processes"] = processes
            self.assertFalse(observe(p, b, a).plan_consistent)

    def test_health_and_activity_are_required(self):
        for health in ({"ready": False, "active_requests": 0}, {"ready": True, "active_requests": 1},
                       {"ready": 1, "active_requests": 0}, {"ready": True, "active_requests": False}):
            p, b, a = fixture()
            a["health"] = health
            self.assertFalse(observe(p, b, a).plan_consistent)

    def test_stale_future_unordered_and_expired_denied(self):
        for before, after, now in ((1001, 1002, 1100), (1001, 1002, 1040),
                                    (1001, 1004, 1003), (1002, 1001, 1003),
                                    (1001, 1001, 1003), (999, 1002, 1003)):
            p, b, a = fixture()
            b["observed_epoch"], a["observed_epoch"] = before, after
            self.assertFalse(observe(p, b, a, now).plan_consistent)

    def test_all_resource_checks_fail_closed(self):
        for key, value in (("gpu_free_bytes", 24), ("gpu_free_bytes", 101),
                           ("host_available_bytes", 14), ("disk_free_bytes", 14),
                           ("cpu_threads_available", 1), ("gpu_total_bytes", 101)):
            p, b, a = fixture()
            a[key] = value
            self.assertFalse(observe(p, b, a).plan_consistent)

    def test_capacity_equality_is_not_authority(self):
        p, b, a = fixture()
        for snapshot in (b, a):
            snapshot.update(gpu_free_bytes=25, host_available_bytes=15, disk_free_bytes=15)
        result = observe(p, b, a)
        self.assertTrue(result.plan_consistent)
        self.assertFalse(result.execution_authority)

    def test_boolean_and_unbounded_integer_inputs_denied(self):
        for key in fixture()[0]["resources"]:
            for value in (True, -1, 0, 1 << 80, "1"):
                p, b, a = fixture()
                p["resources"][key] = value
                self.assertFalse(observe(p, b, a).plan_consistent)
        self.assertFalse(observe(*fixture(), now=True).plan_consistent)

    def test_claimed_qualification_or_control_expansion_denied(self):
        for name in ("isolation_qualified", "signer_trusted", "training_admitted", "stop_bonsai"):
            p, b, a = fixture()
            p[name] = True
            self.assertFalse(observe(p, b, a).plan_consistent)
        p, b, a = fixture()
        p["worker_stop_scope"] = "ANY_GPU_PROCESS"
        self.assertFalse(observe(p, b, a).plan_consistent)

    def test_plan_hash_and_candidate_domain_are_bound(self):
        p, b, a = fixture()
        raw = MODULE.canonical(p)
        self.assertFalse(MODULE.observe_coexistence(raw, MODULE.canonical(b), MODULE.canonical(a),
            expected_plan_sha256="0" * 64, now_epoch=1003).plan_consistent)
        for key in ("candidate_id", "manifest_sha256", "schema"):
            p, b, a = fixture()
            p[key] = "legacy-v4"
            self.assertFalse(observe(p, b, a).plan_consistent)

    def test_byte_and_parser_limits(self):
        p, b, a = fixture()
        for raw in (b"", b"x" * (MODULE.MAX_BYTES + 1), b'{"x":1,"x":2}',
                    b'{"x":NaN}', b'{"x":1.0}', b"\xff", b"[" * 1500 + b"]" * 1500):
            result = MODULE.observe_coexistence(raw, MODULE.canonical(b), MODULE.canonical(a),
                expected_plan_sha256="a" * 64, now_epoch=1003)
            self.assertFalse(result.plan_consistent)

    def test_lease_cannot_expand_or_be_ambiguous(self):
        for expires in (1000, 2201, True):
            p, b, a = fixture()
            p["expires_at_epoch"] = expires
            self.assertFalse(observe(p, b, a).plan_consistent)


if __name__ == "__main__":
    unittest.main()
