"""SIMULATED control fixtures, not runtime, signature provenance or GPU proof."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_runtime as runtime


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def policy_fixture(config, expected):
    return {
        "schema": runtime.POLICY_SCHEMA, "key_trust": "INDEPENDENTLY_PINNED",
        "public_keys": {"fixture-not-an-owner": "ab" * 32}, "revoked_key_ids": [],
        "max_admission_age_seconds": 300, "expected_payload_sha256": digest(expected),
        "paths_sha256": digest(runtime.canonical(config["paths"])),
        "qualification_sha256": "c" * 64, "source_revision": "a" * 40,
        "source_tree": "b" * 40, "smoke_authorized": True,
        "recipe_sha256": digest(runtime.canonical(runtime.NATIVE_RECIPE)),
        "limits": {**runtime.MAX_BOUNDS, **runtime.MIN_BOUNDS},
    }


class ReadFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="szl-v4-native-binding-")
        self.directory = Path(self.temp.name)
        self.path = self.directory / "input.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_duplicate_unknown_scalar_depth_and_nonfinite_rejected(self):
        cases = [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":1.0}', b'[]',
                 b'{"x":' + b'[' * 20 + b'0' + b']' * 20 + b'}', b'{"x":9223372036854775808}']
        for raw in cases:
            with self.subTest(raw=raw[:40]):
                self.path.write_bytes(raw)
                with self.assertRaises(runtime.RuntimeBindingError):
                    runtime.read_json(self.path)

    def test_bounded_unaliased_regular_file(self):
        self.path.write_bytes(b'{"x":1}')
        self.assertEqual(runtime.read_json(self.path), {"x": 1})
        self.assertEqual(runtime.sha256_file(self.path), digest(b'{"x":1}'))
        self.assertEqual(runtime.sha256_file(self.path, max_bytes=512 * 1024**2), digest(b'{"x":1}'))
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "FILE_BYTE_CAP"):
            runtime.read_json(self.path, maximum=2)
        with self.assertRaises(runtime.RuntimeBindingError):
            runtime.sha256_file(self.directory)

    def test_metrics_have_separate_finite_float_reader_not_owner_policy(self):
        self.path.write_bytes(b'{"token_weighted_loss":1.25,"gradient_norm_before_clip":0.5}')
        self.assertEqual(runtime.read_worker_result(self.path)["token_weighted_loss"], 1.25)
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "JSON_SCALAR_FORBIDDEN"):
            runtime.read_json(self.path)
        for raw in (b'{"loss":NaN}', b'{"loss":Infinity}', b'{"loss":1e1000}',
                    b'{"loss":1.0,"loss":2.0}'):
            self.path.write_bytes(raw)
            with self.assertRaises(runtime.RuntimeBindingError):
                runtime.read_worker_result(self.path)

    def test_hardlink_is_not_an_independent_snapshot(self):
        self.path.write_bytes(b'{}')
        linked = self.directory / "linked.json"
        os.link(self.path, linked)
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "REGULAR_UNALIASED_FILE_REQUIRED"):
            runtime.read_json(linked)

    @unittest.skipUnless(os.name == "posix", "Linux symlink fixture; Windows privilege varies")
    def test_alias_and_parent_alias_forbidden(self):
        actual = self.directory / "actual"
        actual.mkdir()
        (actual / "x.json").write_bytes(b'{}')
        link = self.directory / "alias"
        link.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "PATH_ALIAS_FORBIDDEN"):
            runtime.read_json(link / "x.json")

    def test_traversal_and_relative_paths_forbidden(self):
        for path in (Path("relative"), self.directory / ".." / "x"):
            with self.assertRaises(runtime.RuntimeBindingError):
                runtime.check_path(path, may_not_exist=True)

    def test_source_denial_before_unavailable_model_policy_and_signature(self):
        config = {"schema": runtime.REQUEST_SCHEMA,
                  "paths": {name: str(self.directory / (name + "-unavailable"))
                            for name in runtime.PATH_NAMES}}
        config["paths"]["source_root"] = str(runtime.SOURCE_ROOT)
        self.path.write_bytes(runtime.canonical(config))
        with mock.patch.object(runtime, "validate_policy", side_effect=AssertionError("not reached")):
            with self.assertRaisesRegex(runtime.RuntimeBindingError, "SOURCE_TRAINING_INELIGIBLE"):
                runtime.load_bound_run(self.path, owner_policy_sha256="f" * 64)
        self.assertFalse((self.directory / ".v4-nonces").exists())

    def test_policy_pin_is_out_of_band_and_not_request_field(self):
        self.path.write_bytes(runtime.canonical({"schema": runtime.REQUEST_SCHEMA,
                                                "paths": {}, "owner_policy_sha256": "a" * 64}))
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "RUN_REQUEST_FIELDS_INVALID"):
            runtime.load_bound_run(self.path, owner_policy_sha256="f" * 64)
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "OUT_OF_BAND_OWNER_POLICY_PIN_REQUIRED"):
            runtime.load_bound_run(self.path, owner_policy_sha256="")

    def test_source_closure_is_bijective_and_bytes_bound(self):
        self.path.write_bytes(b'{}')
        paths = {"source": self.path}
        entry = {"id": "source", "sha256": digest(b'{}')}
        runtime.check_closure([entry], paths)
        for entries in ([], [entry, entry], [{**entry, "id": "../source"}],
                        [{**entry, "sha256": "a" * 64}], [{**entry, "extra": 1}]):
            with self.assertRaises(runtime.RuntimeBindingError):
                runtime.check_closure(entries, paths)


class PolicyFixtures(unittest.TestCase):
    def setUp(self):
        self.config = {"paths": {"fixed": "/fixed"}}
        self.expected = b'{"fixed":1}'
        self.policy = policy_fixture(self.config, self.expected)

    def test_closed_policy_context(self):
        runtime.validate_policy(self.policy, self.config, self.expected)
        for key, value in (("smoke_authorized", 1), ("key_trust", "REPO_DECLARED"),
                           ("recipe_sha256", "a" * 64), ("expected_payload_sha256", "b" * 64),
                           ("paths_sha256", "c" * 64), ("extra", 1),
                           ("max_admission_age_seconds", True)):
            altered = {**self.policy, key: value}
            with self.subTest(field=key), self.assertRaises(runtime.RuntimeBindingError):
                runtime.validate_policy(altered, self.config, self.expected)

    def test_known_public_fixture_key_never_owner_pin(self):
        self.policy["public_keys"] = {"SIMULATED": runtime.PUBLIC_TEST_KEY}
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "OWNER_PUBLIC_PIN_INVALID_OR_TEST_KEY"):
            runtime.validate_policy(self.policy, self.config, self.expected)

    def test_no_self_declared_or_unbounded_resource_authority(self):
        for key, value in (("wall_seconds", 2**63 - 1), ("kill_seconds", True),
                           ("disk_reserve_bytes", 1), ("extra", 1)):
            policy = copy.deepcopy(self.policy)
            policy["limits"][key] = value
            with self.subTest(field=key), self.assertRaises(runtime.RuntimeBindingError):
                runtime.validate_policy(policy, self.config, self.expected)

    def test_admission_may_only_tighten_owner_limits(self):
        limits = self.policy["limits"]
        runtime.validate_bounds(dict(limits), limits)
        for key, value in (("wall_seconds", limits["wall_seconds"] + 1),
                           ("disk_reserve_bytes", limits["disk_reserve_bytes"] - 1),
                           ("host_memory_bytes", True), ("sequence_tokens", 0)):
            with self.subTest(field=key), self.assertRaises(runtime.RuntimeBindingError):
                runtime.validate_bounds({**limits, key: value}, limits)

    def test_captured_context_properties_are_fresh_not_mutable_permits(self):
        config = {"paths": {name: "/fixed/" + name for name in runtime.PATH_NAMES}}
        run = runtime.BoundRun(Path("/request"), "a" * 64, b'{"x":1}',
                               runtime.canonical(config), b'{"python_prefix":"/python"}')
        altered = run.config
        altered["paths"]["model_dir"] = "/evil"
        self.assertEqual(str(run.model_dir).replace("\\", "/"), "/fixed/model_dir")
        recipe = run.recipe
        recipe["peft"]["r"] = 1024
        self.assertEqual(run.recipe["peft"]["r"], 16)
        with self.assertRaises(runtime.RuntimeBindingError):
            run.path("unknown")
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "BOUND_RUN_TYPE_INVALID"):
            runtime.revalidate_bound_run(object())


class QualificationFixtures(unittest.TestCase):
    def setUp(self):
        self.digest = "a" * 64
        self.gpu = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        self.qualification = {
            "schema": runtime.QUALIFICATION_SCHEMA, "qualified_at_epoch": 100,
            "device_uuid": self.gpu, "python_prefix": str(HERE), "python_sha256": "b" * 64,
            "packages": {name: {"version": "SIMULATED", "tree_sha256": "c" * 64,
                                "root": str(HERE)} for name in (
                "torch", "transformers", "peft", "bitsandbytes", "safetensors", "tokenizers",
                "accelerate", "jsonschema", "cryptography")},
            "lora_target_names": ["model.language_model.layers.0.self_attn.q_proj"],
            "loader_missing_keys": [], "loader_unexpected_keys": [], "loader_mismatched_keys": [],
            "loader_error_msgs": [], "systemd_properties": {},
            "containment_boundary": "TRUSTED_SOURCE_BOUND_PROCESS_NOT_HOSTILE_SAME_UID_SANDBOX",
            "containment_policy_sha256": "d" * 64, "container_kind": "UNAVAILABLE",
            "control_executable_sha256": {name: "e" * 64 for name in (
                "/usr/bin/systemd-run", "/usr/bin/systemctl", "/usr/bin/nvidia-smi", "/usr/bin/env")},
            "runner_rate_minor_units_per_hour": 57, "cost_rate_source": "SIMULATED_NOT_PROVIDER_PROOF",
            **{field: True for field in runtime.BARRIER_OBSERVATIONS},
        }
        self.payload = {
            "runner": {"backend": "LOCAL_ISOLATED", "qualified_at_epoch": 100,
                       "max_qualification_age_seconds": 300, "device_id": self.gpu,
                       "containment_policy_sha256": "d" * 64,
                       **{field: self.digest for field in (
                           "device_qualification_sha256", "telemetry_qualification_sha256",
                           "containment_qualification_sha256")}},
            "tokens": {"cpu_qualification_sha256": self.digest},
            "bounds": {**runtime.MAX_BOUNDS, **runtime.MIN_BOUNDS},
        }

    def check(self, qualification=None, payload=None, now=110):
        runtime.validate_qualification(self.qualification if qualification is None else qualification,
                                       self.payload if payload is None else payload, self.digest, now)

    def test_qualification_shape_is_not_container_or_dependency_proof(self):
        # This checks shape/bindings only. Production loader still unconditionally
        # denies native container identity; the worker separately denies missing
        # installed dependency identity. These synthetic values authorize nothing.
        self.check()
        self.check({**self.qualification, "container_kind": "OCI_IMAGE"})

    def test_all_barriers_exact_true_not_truthy(self):
        for field in runtime.BARRIER_OBSERVATIONS:
            for value in (False, 1, "true", None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                        runtime.RuntimeBindingError, "QUALIFIED_BARRIER_UNAVAILABLE"):
                    self.check({**self.qualification, field: value})

    def test_qualification_pin_and_device_freshness_are_bound(self):
        for now in (99, 401):
            with self.assertRaisesRegex(runtime.RuntimeBindingError, "QUALIFICATION_STALE_OR_FUTURE"):
                self.check(now=now)
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "QUALIFIED_DEVICE_MISMATCH"):
            self.check({**self.qualification, "device_uuid": "GPU-wrong"})
        altered = copy.deepcopy(self.payload)
        altered["runner"]["device_qualification_sha256"] = "f" * 64
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "QUALIFICATION_RECORD_BINDING_MISMATCH"):
            self.check(payload=altered)

    def test_loader_dependency_and_control_coverage_closed(self):
        for field, value in (("loader_missing_keys", ["missing"]),
                             ("packages", {}), ("control_executable_sha256", {}),
                             ("lora_target_names", ["x", "x"]), ("unknown", True)):
            with self.subTest(field=field), self.assertRaises(runtime.RuntimeBindingError):
                self.check({**self.qualification, field: value})

    def test_budget_includes_kill_drain_and_startup_not_numeric_cap_only(self):
        altered = copy.deepcopy(self.payload)
        altered["bounds"]["cost_cap_minor_units"] = 1
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "COST_CAP_INSUFFICIENT"):
            self.check(payload=altered)
        with self.assertRaisesRegex(runtime.RuntimeBindingError, "COST_QUALIFICATION_UNAVAILABLE"):
            self.check({**self.qualification, "runner_rate_minor_units_per_hour": True})


if __name__ == "__main__":
    unittest.main()
