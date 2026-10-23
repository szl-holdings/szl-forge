# SPDX-License-Identifier: Apache-2.0
"""Compare explicit native-runtime and wheel-installed byte trees, without use.

The externally pinned descriptor binds the two native manifests and the wheel
inventory in one distinct native profile. Runtime file roles are DECLARED:
partition coverage does not establish that a file is an interpreter or that the
selected stdlib, native-library and dependency sets are semantically complete.
In particular this does not authenticate the currently executing interpreter,
its prior startup, ambient native loader or already imported code.

Use from an independently trusted stdlib bootstrap. All inputs are opaque bytes;
none is imported, installed, executed or added to an import path. Existing file
and wheel verifiers provide bounded exact byte comparisons. Inputs are compared
again across stages to detect ordinary change under a trusted single writer;
this is not an immutable snapshot or hostile-same-UID sandbox. Separately freeze
and qualify inputs before consumption. No manifest generation, signatures,
trust creation, network, nonce, process, GPU, admission or launch interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import stat

import native_identity as native
import native_wheel_identity as wheels

DESCRIPTOR_SCHEMA = "szl.receiptagent.native-runtime-descriptor/v1"
MAX_DESCRIPTOR_BYTES = 1024**2
MAX_ROLE_FILES = 8192
_PINS = ("runtime_manifest_sha256", "wheel_inventory_sha256", "installed_manifest_sha256")


class NativeRuntimeIdentityError(ValueError):
    """Fixed value-free normal-display code; not introspection secrecy."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if condition is not True:
        raise NativeRuntimeIdentityError(code)


def _descriptor(raw):
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=native._pairs,
                           parse_float=native._forbidden_number,
                           parse_constant=native._forbidden_number)
    except native.NativeIdentityError as error:
        raise NativeRuntimeIdentityError(error.code) from None
    except (ValueError, UnicodeError, RecursionError):
        raise NativeRuntimeIdentityError("DESCRIPTOR_JSON_INVALID") from None
    _require(type(value) is dict and set(value) == {"schema", "roles", *_PINS}
             and value["schema"] == DESCRIPTOR_SCHEMA, "DESCRIPTOR_FIELDS_INVALID")
    _require(all(type(value[field]) is str and native._HEX.fullmatch(value[field]) is not None
                 for field in _PINS), "DESCRIPTOR_PIN_INVALID")
    roles = value["roles"]
    _require(type(roles) is dict and set(roles) == {"interpreter", "stdlib", "native_libraries"},
             "RUNTIME_ROLE_FIELDS_INVALID")
    names = [native._relative_name(roles["interpreter"])]
    for field in ("stdlib", "native_libraries"):
        entries = roles[field]
        _require(type(entries) is list and 1 <= len(entries) <= MAX_ROLE_FILES,
                 "RUNTIME_ROLE_COVERAGE_INVALID")
        parsed = [native._relative_name(name) for name in entries]
        _require(parsed == sorted(set(parsed)), "RUNTIME_ROLE_ORDER_OR_DUPLICATE")
        names.extend(parsed)
    _require(len({name.casefold() for name in names}) == len(names), "RUNTIME_ROLE_OVERLAP")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                           allow_nan=False).encode("ascii")
    _require(raw == canonical, "NONCANONICAL_DESCRIPTOR")
    return value, frozenset(names)


@dataclass(frozen=True)
class NativeRuntimeIdentity:
    """Byte comparison counts and pins only, never an authorization capability."""
    descriptor_sha256: str
    runtime_manifest_sha256: str
    wheel_inventory_sha256: str
    installed_manifest_sha256: str
    runtime_file_count: int
    runtime_total_bytes: int
    wheel_count: int
    dependency_file_count: int
    dependency_total_bytes: int

    def __bool__(self):
        raise TypeError("NATIVE_RUNTIME_OBSERVATION_IS_NOT_AUTHORIZATION")


def verify_native_runtime_descriptor(runtime_root: Path, runtime_manifest_path: Path,
                                     wheel_root: Path, inventory_path: Path,
                                     installed_root: Path, installed_manifest_path: Path,
                                     descriptor_path: Path, *,
                                     expected_descriptor_sha256: str) -> NativeRuntimeIdentity:
    """Verify one externally pinned, closed native byte descriptor.

    All roots must be separate plain directories. Descriptor/manifests/inventory
    are distinct unaliased files outside every root. Every runtime-manifest file
    must have exactly one declared role; every installed file must be represented
    by the pinned root-layout wheel union. The wheel directory itself is closed,
    with no unlisted wheels, files or subdirectories. Role labels and matching
    bytes confer neither semantic completeness nor executing-process identity.
    """
    _require(type(expected_descriptor_sha256) is str
             and native._HEX.fullmatch(expected_descriptor_sha256) is not None,
             "EXTERNAL_DESCRIPTOR_PIN_REQUIRED")
    roots = (runtime_root, wheel_root, installed_root)
    controls = (descriptor_path, runtime_manifest_path, inventory_path, installed_manifest_path)
    try:
        for path in roots + controls:
            native._checked_path(path)
        for index, root in enumerate(roots):
            _require(stat.S_ISDIR(root.lstat().st_mode), "PLAIN_DIRECTORY_REQUIRED")
            for other in roots[index + 1:]:
                _require(root != other and root not in other.parents and other not in root.parents
                         and (root.stat().st_dev, root.stat().st_ino)
                         != (other.stat().st_dev, other.stat().st_ino), "DISJOINT_ROOTS_REQUIRED")
        _require(len(set(controls)) == len(controls)
                 and all(path != root and root not in path.parents
                         for path in controls for root in roots), "CONTROL_PATH_OVERLAP")
        descriptor_digest, descriptor_raw, descriptor_stamp = native._capture_file(
            descriptor_path, maximum=MAX_DESCRIPTOR_BYTES, retain=True)
        _require(descriptor_digest == expected_descriptor_sha256, "DESCRIPTOR_PIN_MISMATCH")
        descriptor, role_files = _descriptor(descriptor_raw)
        captured = {descriptor_path: descriptor_stamp}
        parsed = {}
        for field, path, cap in (
                ("runtime_manifest_sha256", runtime_manifest_path, native.MAX_MANIFEST_BYTES),
                ("installed_manifest_sha256", installed_manifest_path, native.MAX_MANIFEST_BYTES),
                ("wheel_inventory_sha256", inventory_path, wheels.MAX_INVENTORY_BYTES)):
            digest, raw, stamp = native._capture_file(path, maximum=cap, retain=True)
            _require(digest == descriptor[field], "DESCRIPTOR_INPUT_PIN_MISMATCH")
            captured[path] = stamp
            parsed[field] = raw
        runtime_entries, _, _ = native._manifest(parsed["runtime_manifest_sha256"])
        _require(role_files == frozenset(runtime_entries), "RUNTIME_ROLE_COVERAGE_INVALID")
        wheel_entries = wheels._inventory(parsed["wheel_inventory_sha256"],
                                          descriptor["installed_manifest_sha256"])
        before = {root: native._inventory(root) for root in roots}
        _require(set(before[wheel_root][0]) == {entry["path"] for entry in wheel_entries}
                 and set(before[wheel_root][1]) == {""}, "WHEEL_ROOT_INVENTORY_MISMATCH")
        runtime_result = native.verify_native_file_manifest(
            runtime_root, runtime_manifest_path,
            expected_manifest_sha256=descriptor["runtime_manifest_sha256"])
        wheel_result = wheels.verify_native_wheel_inventory(
            wheel_root, inventory_path, installed_root, installed_manifest_path,
            expected_inventory_sha256=descriptor["wheel_inventory_sha256"],
            expected_native_manifest_sha256=descriptor["installed_manifest_sha256"])
        for root in roots:
            native._checked_path(root)
            _require(native._inventory(root) == before[root], "INPUT_CHANGED_DURING_READ")
        for path, stamp in captured.items():
            native._checked_path(path)
            _require(native._stamp(path.lstat()) == stamp, "INPUT_CHANGED_DURING_READ")
        return NativeRuntimeIdentity(
            descriptor_digest, runtime_result.manifest_sha256, wheel_result.inventory_sha256,
            wheel_result.native_manifest_sha256, runtime_result.file_count, runtime_result.total_bytes,
            wheel_result.wheel_count, wheel_result.file_count, wheel_result.total_bytes)
    except (native.NativeIdentityError, wheels.NativeWheelIdentityError) as error:
        raise NativeRuntimeIdentityError(error.code) from None
    except OSError:
        raise NativeRuntimeIdentityError("FILESYSTEM_UNAVAILABLE") from None


def verify_native_payload_runtime(payload_raw: bytes, runtime_root: Path,
                                   runtime_manifest_path: Path, wheel_root: Path,
                                   inventory_path: Path, installed_root: Path,
                                   installed_manifest_path: Path,
                                   descriptor_path: Path) -> NativeRuntimeIdentity:
    """Compare actual bytes with the commitments in one closed native payload.

    Canonical/structural validation is not signature verification, trusted
    context matching, freshness or authority. The caller must separately check
    the exact payload against its independent expected context and trusted
    signing policy. A payload-supplied descriptor hash alone establishes no
    external trust. Other runtime fields (bootstrap/import/dependency-graph and
    read-only-consumption policies) are NOT authenticated by this comparison.
    No cryptographic backend is invoked and no input artifact is executed.
    """
    import native_signed_admission as admission

    try:
        payload = admission.parse_canonical_bytes(payload_raw)
        admission._validate_payload(payload)
    except admission.AdmissionError as error:
        raise NativeRuntimeIdentityError(error.code) from None
    expected = payload["runtime"]
    result = verify_native_runtime_descriptor(
        runtime_root, runtime_manifest_path, wheel_root, inventory_path,
        installed_root, installed_manifest_path, descriptor_path,
        expected_descriptor_sha256=expected["native_descriptor_sha256"])
    _require(result.runtime_manifest_sha256 == expected["runtime_manifest_sha256"]
             and result.wheel_inventory_sha256 == expected["wheel_inventory_sha256"]
             and result.installed_manifest_sha256 == expected["installed_manifest_sha256"],
             "NATIVE_PAYLOAD_IDENTITY_MISMATCH")
    return result
