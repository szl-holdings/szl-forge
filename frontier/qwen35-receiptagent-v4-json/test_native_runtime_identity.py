# SPDX-License-Identifier: Apache-2.0
"""SIMULATED opaque files; not runtime completeness or execution qualification."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
import native_identity as files
import native_wheel_identity as wheels
import native_runtime_identity as runtime
import native_signed_admission as admission
from test_signed_admission import simulated_shape


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="szl-native-runtime-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).absolute()
        self.runtime = self.parent / "runtime"
        self.installed = self.parent / "installed"
        self.wheel_root = self.parent / "wheels"
        for path in (self.runtime, self.installed, self.wheel_root):
            path.mkdir()
        self.runtime_manifest = self.parent / "runtime.json"
        self.installed_manifest = self.parent / "installed.json"
        self.inventory = self.parent / "inventory.json"
        self.descriptor = self.parent / "descriptor.json"
        self.runtime_members = {
            "bin/python": b"not an executable",
            "lib/libpython.so": b"not a shared library",
            "stdlib/os.py": b"raise AssertionError('must never execute')\n",
        }
        self.dependencies = {
            "package/__init__.py": b"raise AssertionError('must never import')\n",
            "example-1.0.dist-info/RECORD": b"opaque record",
        }
        runtime_pin = self.write_tree(self.runtime, self.runtime_manifest, self.runtime_members)
        installed_pin = self.write_tree(self.installed, self.installed_manifest, self.dependencies)
        wheel = self.wheel_root / "example-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
            for name, raw in self.dependencies.items():
                archive.writestr(name, raw)
        raw = canonical({"schema": wheels.INVENTORY_SCHEMA, "native_manifest_sha256": installed_pin,
                         "wheels": [{"path": wheel.name, "size_bytes": wheel.stat().st_size,
                                     "sha256": sha(wheel.read_bytes())}]})
        self.inventory.write_bytes(raw)
        self.document = {
            "schema": runtime.DESCRIPTOR_SCHEMA,
            "runtime_manifest_sha256": runtime_pin,
            "installed_manifest_sha256": installed_pin,
            "wheel_inventory_sha256": sha(raw),
            "roles": {"interpreter": "bin/python", "stdlib": ["stdlib/os.py"],
                      "native_libraries": ["lib/libpython.so"]},
        }
        self.bind_descriptor()

    def write_tree(self, root, manifest, members):
        for name, raw in members.items():
            path = root.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        raw = canonical({"schema": files.MANIFEST_SCHEMA, "files": [
            {"path": name, "size_bytes": len(data), "sha256": sha(data)}
            for name, data in sorted(members.items())]})
        manifest.write_bytes(raw)
        return sha(raw)

    def bind_descriptor(self, document=None, raw=None):
        raw = canonical(self.document if document is None else document) if raw is None else raw
        self.descriptor.write_bytes(raw)
        self.pin = sha(raw)

    def verify(self, **changes):
        arguments = {
            "runtime_root": self.runtime, "runtime_manifest_path": self.runtime_manifest,
            "wheel_root": self.wheel_root, "inventory_path": self.inventory,
            "installed_root": self.installed, "installed_manifest_path": self.installed_manifest,
            "descriptor_path": self.descriptor, "expected_descriptor_sha256": self.pin,
        }
        return runtime.verify_native_runtime_descriptor(**{**arguments, **changes})

    def deny(self, code=None, **changes):
        with self.assertRaises(runtime.NativeRuntimeIdentityError) as error:
            self.verify(**changes)
        if code:
            self.assertEqual(error.exception.code, code)

    def native_payload(self, schema=None):
        payload = simulated_shape(admission.payload_schema() if schema is None else schema)
        payload["identity"].update(issued_at_epoch=100, not_before_epoch=110, expires_at_epoch=200)
        payload["runner"].update(qualified_at_epoch=100, max_qualification_age_seconds=30)
        payload["bounds"].update(microbatch_rows=1, accumulation_batches=360,
                                 sequence_tokens=1, total_tokens=360)
        if schema is None:
            payload["runtime"].update(
                native_descriptor_sha256=self.pin,
                **{field: self.document[field] for field in (
                    "runtime_manifest_sha256", "wheel_inventory_sha256", "installed_manifest_sha256")})
        return payload

    def verify_payload(self, payload=None, raw=None):
        raw = canonical(self.native_payload() if payload is None else payload) if raw is None else raw
        return runtime.verify_native_payload_runtime(
            raw, self.runtime, self.runtime_manifest, self.wheel_root, self.inventory,
            self.installed, self.installed_manifest, self.descriptor)

    def test_payload_bridge_observes_actual_bytes_without_signature_calls(self):
        with mock.patch.object(admission, "_signature_observation", side_effect=AssertionError("no crypto")):
            result = self.verify_payload()
        self.assertEqual(result, self.verify())
        with self.assertRaises(TypeError):
            bool(result)

    def test_payload_bridge_checks_all_four_identity_commitments(self):
        for field in ("native_descriptor_sha256", "runtime_manifest_sha256",
                      "wheel_inventory_sha256", "installed_manifest_sha256"):
            payload = self.native_payload()
            payload["runtime"][field] = "f" * 64
            with self.subTest(field=field), self.assertRaises(runtime.NativeRuntimeIdentityError):
                self.verify_payload(payload)

    def test_payload_bridge_rejects_container_and_malformed_profile_before_io(self):
        container_payload = self.native_payload(admission.container.payload_schema())
        hybrid = self.native_payload()
        hybrid["runtime"]["container_sha256"] = "a" * 64
        wrong_schema = self.native_payload()
        wrong_schema["identity"]["schema_version"] = "unknown"
        for payload in (container_payload, hybrid, wrong_schema, {}):
            with self.subTest(payload=payload.get("identity")), mock.patch.object(
                    runtime, "verify_native_runtime_descriptor", side_effect=AssertionError("no I/O")):
                with self.assertRaises(runtime.NativeRuntimeIdentityError):
                    self.verify_payload(payload)

    def test_payload_bridge_rejects_noncanonical_and_nonbytes_input(self):
        for raw in (canonical(self.native_payload()) + b"\n", "not bytes", b'{}', b'[]'):
            with self.subTest(raw=str(raw)[:20]), self.assertRaises(runtime.NativeRuntimeIdentityError):
                self.verify_payload(raw=raw)

    def test_payload_bridge_still_rejects_modified_runtime_bytes(self):
        (self.runtime / "bin/python").write_bytes(b"changed")
        with self.assertRaises(runtime.NativeRuntimeIdentityError):
            self.verify_payload()

    def test_exact_bytes_return_frozen_non_authorizing_observation(self):
        before = tuple(sys.path)
        result = self.verify()
        self.assertEqual(result.descriptor_sha256, self.pin)
        self.assertEqual(result.runtime_manifest_sha256, self.document["runtime_manifest_sha256"])
        self.assertEqual((result.runtime_file_count, result.wheel_count, result.dependency_file_count), (3, 1, 2))
        self.assertEqual(result.runtime_total_bytes, sum(map(len, self.runtime_members.values())))
        self.assertEqual(result.dependency_total_bytes, sum(map(len, self.dependencies.values())))
        self.assertEqual(tuple(sys.path), before)
        with self.assertRaises(TypeError):
            bool(result)
        with self.assertRaises(AttributeError):
            result.runtime_file_count = 99

    def test_external_pin_required_and_cannot_be_self_declared(self):
        for pin in (None, "", True, "A" * 64, "f" * 64):
            with self.subTest(pin=pin):
                self.deny(expected_descriptor_sha256=pin)
        self.bind_descriptor({**self.document, "expected_descriptor_sha256": self.pin})
        self.deny("DESCRIPTOR_FIELDS_INVALID")

    def test_descriptor_exact_schema_canonical_and_strict_json(self):
        for raw in (b'[]', b'{"schema":1,"schema":2}', b'{"x":NaN}',
                    canonical(self.document) + b'\n', b'{"x":' + b'[' * 1100 + b'0' + b']' * 1100 + b'}'):
            with self.subTest(raw=raw[:30]):
                self.bind_descriptor(raw=raw)
                self.deny()
        for key, value in (("schema", "szl.receiptagent.native-file-manifest/v1"),
                           ("wheel_inventory_sha256", True), ("container_sha256", "a" * 64)):
            with self.subTest(key=key):
                self.bind_descriptor({**self.document, key: value})
                self.deny()

    def test_role_partition_must_be_exact_nonempty_unique_and_ordered(self):
        for roles in ({}, {**self.document["roles"], "stdlib": []},
                      {**self.document["roles"], "native_libraries": []},
                      {**self.document["roles"], "interpreter": "missing"},
                      {**self.document["roles"], "stdlib": ["bin/python", "stdlib/os.py"]},
                      {**self.document["roles"], "stdlib": ["stdlib/os.py", "stdlib/os.py"]},
                      {**self.document["roles"], "stdlib": ["stdlib/os.py", "../escape"]},
                      {**self.document["roles"], "extra": []}):
            with self.subTest(roles=roles):
                self.bind_descriptor({**self.document, "roles": roles})
                self.deny()

    def test_runtime_extra_missing_and_modified_files_deny(self):
        path = self.runtime / "stdlib/os.py"
        path.write_bytes(b"tampered")
        self.deny("FILE_SIZE_MISMATCH")
        path.write_bytes(self.runtime_members["stdlib/os.py"])
        extra = self.runtime / "extra"
        extra.mkdir()
        self.deny("TREE_INVENTORY_MISMATCH")
        extra.rmdir()
        path.unlink()
        self.deny("TREE_INVENTORY_MISMATCH")

    def test_installed_and_wheel_bytes_are_actually_verified(self):
        target = self.installed / "package/__init__.py"
        target.write_bytes(b"tampered")
        self.deny()
        target.write_bytes(self.dependencies["package/__init__.py"])
        wheel = next(self.wheel_root.iterdir())
        wheel.write_bytes(wheel.read_bytes()[:-1] + b"x")
        self.deny("WHEEL_DIGEST_MISMATCH")

    def test_extra_wheel_or_empty_directory_is_not_ignored(self):
        extra = self.wheel_root / "unlisted.whl"
        extra.write_bytes(b"unlisted")
        self.deny("WHEEL_ROOT_INVENTORY_MISMATCH")
        extra.unlink()
        extra.mkdir()
        self.deny("WHEEL_ROOT_INVENTORY_MISMATCH")

    def test_control_files_outside_roots_and_pairwise_distinct(self):
        for key in ("descriptor_path", "runtime_manifest_path", "inventory_path", "installed_manifest_path"):
            with self.subTest(key=key):
                self.deny("CONTROL_PATH_OVERLAP", **{key: self.runtime / "bin/python"})
        self.deny("CONTROL_PATH_OVERLAP", inventory_path=self.runtime_manifest)

    def test_roots_disjoint_and_paths_absolute(self):
        self.deny("DISJOINT_ROOTS_REQUIRED", runtime_root=self.installed)
        self.deny("DISJOINT_ROOTS_REQUIRED", runtime_root=self.parent)
        self.deny("ABSOLUTE_PATH_REQUIRED", runtime_root=Path("relative"))
        self.deny("PATH_TRAVERSAL_FORBIDDEN", runtime_root=self.parent / ".." / "runtime")

    def deny_simulated_bind_alias(self, first, second):
        """Normalize existing validated stamps as a bind alias, without mounting.

        Both paths remain real, separately readable fixture files/directories.
        Only their observed device/inode identity is made equal; path shape,
        regular-file link counts, byte caps and before/after checks still run.
        """
        original = files._stable
        first_meta, second_meta = first.lstat(), second.lstat()
        first_id, second_id = original(first_meta)[:2], original(second_meta)[:2]
        self.assertNotEqual(first_id, second_id)
        if first.is_file():
            self.assertEqual(first_meta.st_nlink, 1)
        if second.is_file():
            self.assertEqual(second_meta.st_nlink, 1)
        def alias(meta):
            stamp = original(meta)
            return first_id + stamp[2:] if stamp[:2] == second_id else stamp
        with mock.patch.object(files, "_stable", side_effect=alias), mock.patch.object(
                files, "verify_native_file_manifest", side_effect=AssertionError("no byte verification")), \
                mock.patch.object(wheels, "verify_native_wheel_inventory",
                                  side_effect=AssertionError("no byte verification")):
            self.deny("FILESYSTEM_IDENTITY_OVERLAP")

    def test_cross_root_file_bind_aliases_reject_with_single_links(self):
        roots = (self.runtime, self.wheel_root, self.installed)
        self.assertEqual(len({(root.stat().st_dev, root.stat().st_ino) for root in roots}), 3)
        paths = (self.runtime / "bin/python", next(self.wheel_root.iterdir()),
                 self.installed / "package/__init__.py")
        for index, first in enumerate(paths):
            for second in paths[index + 1:]:
                with self.subTest(first=first.name, second=second.name):
                    self.deny_simulated_bind_alias(first, second)

    def test_cross_root_directory_bind_aliases_reject(self):
        self.deny_simulated_bind_alias(self.runtime / "lib", self.installed / "package")

    def test_control_to_control_and_control_to_tree_bind_aliases_reject(self):
        for first, second in ((self.descriptor, self.runtime_manifest),
                              (self.runtime / "bin/python", self.descriptor)):
            with self.subTest(first=first.name, second=second.name):
                self.deny_simulated_bind_alias(first, second)

    def test_within_tree_file_bind_aliases_reject(self):
        self.deny_simulated_bind_alias(self.runtime / "bin/python", self.runtime / "lib/libpython.so")

    def test_unavailable_underlying_identity_rejects(self):
        original = files._stable
        target = original((self.runtime / "bin/python").lstat())[:2]
        def unavailable(meta):
            stamp = original(meta)
            return (stamp[0], 0, *stamp[2:]) if stamp[:2] == target else stamp
        with mock.patch.object(files, "_stable", side_effect=unavailable):
            self.deny("FILESYSTEM_IDENTITY_UNAVAILABLE")

    def test_hardlinks_deny(self):
        alias = self.parent / "alias.json"
        os.link(self.descriptor, alias)
        self.deny("REGULAR_UNALIASED_FILE_REQUIRED")

    @unittest.skipUnless(os.name == "posix", "POSIX symlink fixture; Windows requires privilege")
    def test_symbolic_root_denies(self):
        alias = self.parent / "alias"
        alias.symlink_to(self.runtime, target_is_directory=True)
        self.deny("PATH_ALIAS_FORBIDDEN", runtime_root=alias)

    def test_all_manifest_pins_checked(self):
        for field in ("runtime_manifest_sha256", "installed_manifest_sha256", "wheel_inventory_sha256"):
            with self.subTest(field=field):
                self.bind_descriptor({**self.document, field: "f" * 64})
                self.deny()

    def test_change_after_runtime_verification_is_detected(self):
        original = wheels.verify_native_wheel_inventory
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.runtime / "bin/python"
            path.write_bytes(b"changed afterward")
            return result
        with mock.patch.object(wheels, "verify_native_wheel_inventory", side_effect=mutate):
            self.deny("INPUT_CHANGED_DURING_READ")

    def test_descriptor_change_during_verification_is_detected(self):
        original = wheels.verify_native_wheel_inventory
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.descriptor.write_bytes(canonical({**self.document, "unknown": True}))
            return result
        with mock.patch.object(wheels, "verify_native_wheel_inventory", side_effect=mutate):
            self.deny("INPUT_CHANGED_DURING_READ")

    def test_descriptor_byte_cap_before_parse(self):
        self.bind_descriptor(raw=b" " * (runtime.MAX_DESCRIPTOR_BYTES + 1))
        self.deny("FILE_BYTE_CAP")


if __name__ == "__main__":
    unittest.main()
