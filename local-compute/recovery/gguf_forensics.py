#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only, bounded GGUF tensor fingerprints for Forge issue #264.

No inference, tensor deserialization, downloads, API calls, shell, package installs,
model mutation or recovery-attempt changes. Hashes establish byte identity only;
identical tensors are not proof of a healthy, original or qualified model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import struct
import time
from typing import BinaryIO
import uuid

VERSION = "1.2.1"
SPEC_COMMIT = "7840aaba1989c6deeefede1d77d5aaf8f52b947e"
# Provider-reported FROM identities in the inspected owner recovery report.
# These are NOT installed-tag digests and NOT claims about current local files.
BLOBS = {
    "szl1:latest": "202993bbb4a645b75905ebd18052707968cdc4f647ad12d9d153d57f7ddc2199",
    "szl-sovereign-qwen:latest": "4028d616d241d719db96727fda79f45aed7d964b43fb7f3837cb2c29ac2ca20f",
}
CHUNK = 1024 * 1024
MAX_FILE = 8 * 1024**3
MAX_HEADER = 64 * 1024**2
MAX_TENSORS = 10000
MAX_METADATA = 10000
MAX_ARRAY_ITEMS = 2_000_000
MAX_VALUE_NODES = 4_000_000
# Only the unquantized F16/F32 tensor types relevant to this investigation.
TENSOR_TYPES = {0: ("F32", 4), 1: ("F16", 2)}
PRIMITIVES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
              6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


class ForensicsError(ValueError):
    """Stable error code without source paths or raw metadata."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ForensicsError(code)


class UnsupportedTensorType(ForensicsError):
    """A bounded header observation, not an authenticated file/weight result."""
    def __init__(self, kind: int, tensor_index: int, type_field_offset: int):
        require(type(kind) is int and 0 <= kind <= 0xffffffff,
                "INVALID_TENSOR_TYPE_OBSERVATION")
        require(type(tensor_index) is int and 0 <= tensor_index < MAX_TENSORS,
                "INVALID_TENSOR_TYPE_OBSERVATION")
        require(type(type_field_offset) is int and 0 <= type_field_offset <= MAX_HEADER - 4,
                "INVALID_TENSOR_TYPE_OBSERVATION")
        super().__init__(f"UNSUPPORTED_TENSOR_TYPE:ggml_type_id={kind}")
        # Fixed keys and integers only; never retain tensor names or raw headers.
        self.observation = {
            "ggml_type_id": kind,
            "tensor_index_zero_based": tensor_index,
            "type_field_byte_offset": type_field_offset,
            "supported_ggml_type_ids": sorted(TENSOR_TYPES),
            "scope": "PARTIAL_HEADER_NOT_FULL_FILE_AUTHENTICATED",
            "historical_blob_sha256_verification": "NOT_COMPLETED",
            "inferred_from_model_label": False,
        }


def rejected_model(error: Exception) -> dict:
    """Keep rejection separate from successful layout and full-file checks."""
    result = {"state": "UNAVAILABLE_OR_REJECTED",
              "error_code": str(error) if isinstance(error, ForensicsError) else type(error).__name__}
    if isinstance(error, UnsupportedTensorType):
        result["tensor_type_observation"] = dict(error.observation)
    return result


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def windows_drive_type(anchor: str) -> int:
    """Query the OS drive classification; never probe a remote file/share.

    Load only the system32 DLL. Unknown drive types are NOT a local fallback.
    This is a cooperative admission check, not protection against a concurrent
    drive remap or a hostile storage driver.
    """
    require(os.name == "nt", "WINDOWS_DRIVE_QUERY_REQUIRED")
    import ctypes
    try:
        kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)
        query = kernel.GetDriveTypeW
        query.argtypes = [ctypes.c_wchar_p]
        query.restype = ctypes.c_uint
        return int(query(anchor))
    except (AttributeError, OSError, TypeError, ValueError):
        raise ForensicsError("LOCAL_DRIVE_TYPE_UNAVAILABLE") from None


def require_local_windows_path(path: str | os.PathLike[str]) -> None:
    """Admit fixed local drive-letter paths BEFORE lstat/is_dir/open/mkdir.

    A mapped network drive may be spelled with a drive letter, not a UNC path. Checking
    only the spelling would allow network I/O before the historical byte pin.
    Both the input root and the report home must pass this independent gate.
    """
    raw = os.fspath(path)
    require(isinstance(raw, str) and "\x00" not in raw, "LOCAL_DRIVE_PATH_REQUIRED")
    parsed = PureWindowsPath(raw)
    require(parsed.is_absolute() and re.fullmatch(r"[A-Za-z]:", parsed.drive) is not None
            and ".." not in parsed.parts
            and all(":" not in part for part in parsed.parts[1:]), "LOCAL_DRIVE_PATH_REQUIRED")
    require(windows_drive_type(parsed.anchor) == 3, "FIXED_LOCAL_DRIVE_REQUIRED")


def plain_path(path: Path) -> bool:
    """Reject all existing symlink/reparse components, including broken links."""
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return False
    return True


def snapshot(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def require_stat_match(before: os.stat_result, after: os.stat_result,
                       stage: str, *, same_api: bool) -> None:
    """Compare equivalent metadata, retaining ctime in same-API observations.

    CPython 3.12.10 on Windows maps path-stat ctime to CreationTime, while
    fstat exposes ChangeTime. Comparing those fields across APIs can reject
    an unchanged file. Cross-API comparison uses explicit birthtime instead.
    Descriptor-to-descriptor and path-to-path checks STILL compare ctime;
    there is no tolerance, zero-ID fallback, or exception-to-success path.
    """
    fields = ["st_dev", "st_ino", "st_size", "st_mtime_ns"]
    if os.name == "nt":
        fields.append("st_birthtime_ns")
    if same_api or os.name != "nt":
        fields.append("st_ctime_ns")
    missing = [name for name in fields
               if type(getattr(before, name, None)) is not int
               or type(getattr(after, name, None)) is not int]
    require(not missing, "FILE_METADATA_UNAVAILABLE:" + ",".join(missing))
    require(before.st_ino != 0 and after.st_ino != 0, "FILE_IDENTITY_UNAVAILABLE")
    require(stat.S_ISREG(before.st_mode) and stat.S_ISREG(after.st_mode),
            "REGULAR_FILE_REQUIRED")
    changed = [name for name in fields if getattr(before, name) != getattr(after, name)]
    # Field names only: never print local paths, IDs, timestamps, or raw metadata.
    require(not changed, stage + ":" + ",".join(changed))


class Reader:
    """Sequential bounded reads. Every byte participates in the container hash."""
    def __init__(self, stream: BinaryIO, size: int, seconds: float):
        self.stream, self.size, self.offset = stream, size, 0
        self.header = True
        self.deadline = time.monotonic() + seconds
        self.file_hash = hashlib.sha256()
        self.value_hash = None
        self.nodes = 0

    def take(self, count: int) -> bytes:
        require(0 <= count <= CHUNK, "READ_SIZE_BOUND")
        require(time.monotonic() < self.deadline, "READ_DEADLINE")
        require(self.offset + count <= self.size, "TRUNCATED_FILE")
        require(not self.header or self.offset + count <= MAX_HEADER, "HEADER_LIMIT")
        data = self.stream.read(count)
        require(len(data) == count, "SHORT_READ")
        self.offset += count
        self.file_hash.update(data)
        if self.value_hash is not None:
            self.value_hash.update(data)
        return data

    def number(self, fmt: str) -> int | float:
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def text(self, maximum: int = CHUNK) -> str:
        length = self.number("Q")
        require(length <= maximum, "STRING_LIMIT")
        try:
            return self.take(length).decode("utf-8", errors="strict")
        except UnicodeError:
            raise ForensicsError("INVALID_UTF8") from None

    def value(self, kind: int, depth: int = 0) -> object:
        self.nodes += 1
        require(self.nodes <= MAX_VALUE_NODES and depth <= 4, "VALUE_BUDGET")
        if kind in PRIMITIVES:
            result = self.number(PRIMITIVES[kind])
            require(kind != 7 or result in (0, 1), "INVALID_BOOLEAN")
            require(not isinstance(result, float) or math.isfinite(result), "NONFINITE_METADATA")
            return result
        if kind == 8:
            return self.text()
        if kind == 9:
            element_kind, count = self.number("I"), self.number("Q")
            require(element_kind in set(PRIMITIVES) | {8, 9}, "UNSUPPORTED_METADATA_TYPE")
            require(count <= MAX_ARRAY_ITEMS, "ARRAY_LIMIT")
            # Do not retain vocabularies or other untrusted metadata arrays.
            for _ in range(count):
                self.value(element_kind, depth + 1)
            return None
        raise ForensicsError("UNSUPPORTED_METADATA_TYPE")

    def span(self, count: int, content_hash=None) -> None:
        while count:
            block = self.take(min(count, CHUNK))
            if content_hash is not None:
                content_hash.update(block)
            count -= len(block)


def fingerprint_stream(stream: BinaryIO, size: int, seconds: float = 600) -> dict:
    """Parse an admitted little-endian GGUF v2/v3, retaining only hashes/counts."""
    require(type(size) is int and 24 <= size <= MAX_FILE, "FILE_SIZE_BOUND")
    require(type(seconds) in (int, float) and math.isfinite(seconds)
            and 0 < seconds <= 1800, "DEADLINE_BOUND")
    r = Reader(stream, size, seconds)
    require(r.take(4) == b"GGUF", "BAD_MAGIC")
    version = r.number("I")
    require(version in (2, 3), "UNSUPPORTED_VERSION_OR_ENDIAN")
    tensor_count, metadata_count = r.number("Q"), r.number("Q")
    require(1 <= tensor_count <= MAX_TENSORS, "TENSOR_COUNT_BOUND")
    require(1 <= metadata_count <= MAX_METADATA, "METADATA_COUNT_BOUND")
    groups: dict[str, list] = {k: [] for k in ("all", "tokenizer", "chat_template", "architecture")}
    keys = set()
    alignment, architecture = 32, None
    for _ in range(metadata_count):
        key = r.text(65535)
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", key) is not None, "METADATA_KEY_BOUND")
        require(key not in keys, "DUPLICATE_METADATA_KEY")
        keys.add(key)
        r.value_hash = hashlib.sha256()
        kind = r.number("I")
        value = r.value(kind)
        entry = [digest(key.encode()), r.value_hash.hexdigest()]
        r.value_hash = None
        groups["all"].append(entry)
        if key.startswith("tokenizer.chat_template"):
            groups["chat_template"].append(entry)
        elif key.startswith("tokenizer."):
            groups["tokenizer"].append(entry)
        if key == "general.architecture":
            require(kind == 8 and value == "qwen2", "OUTSIDE_QWEN2_SCOPE")
            architecture = value
        if key == "general.architecture" or key.startswith("qwen2."):
            groups["architecture"].append(entry)
        if key == "general.alignment":
            require(kind == 4 and type(value) is int and 8 <= value <= 4096
                    and value % 8 == 0, "ALIGNMENT_BOUND")
            alignment = value
        if key.startswith("split."):
            raise ForensicsError("SHARDED_GGUF_NOT_SUPPORTED")
    require(architecture == "qwen2", "ARCHITECTURE_MISSING")
    tensors, names = [], set()
    for _ in range(tensor_count):
        name = r.text(64)
        require(bool(name) and name not in names, "EMPTY_OR_DUPLICATE_TENSOR_NAME")
        names.add(name)
        dimensions = r.number("I")
        require(1 <= dimensions <= 4, "DIMENSION_COUNT_BOUND")
        shape = [r.number("Q") for _ in range(dimensions)]
        require(all(0 < x <= 2**31 for x in shape), "DIMENSION_SIZE_BOUND")
        kind = r.number("I")
        if kind not in TENSOR_TYPES:
            raise UnsupportedTensorType(kind, len(tensors), r.offset - 4)
        offset = r.number("Q")
        nbytes = math.prod(shape) * TENSOR_TYPES[kind][1]
        require(nbytes <= MAX_FILE and offset <= MAX_FILE and offset % alignment == 0,
                "TENSOR_RANGE_OR_ALIGNMENT")
        tensors.append({"name_sha256": digest(name.encode()), "shape": shape,
                        "type": TENSOR_TYPES[kind][0], "bytes": nbytes, "offset": offset})
    r.header = False
    data_start = ((r.offset + alignment - 1) // alignment) * alignment
    require(data_start <= size, "TRUNCATED_PADDING")
    end = data_start
    for tensor in sorted(tensors, key=lambda item: item["offset"]):
        start = data_start + tensor["offset"]
        require(start >= end, "OVERLAPPING_TENSORS")
        end = start + tensor["bytes"]
        require(end <= size, "TRUNCATED_TENSOR")
    # One sequential pass through tensors/gaps: no full model copy or mmap.
    for tensor in sorted(tensors, key=lambda item: item["offset"]):
        r.span(data_start + tensor["offset"] - r.offset)
        content_hash = hashlib.sha256()
        r.span(tensor["bytes"], content_hash)
        tensor["data_sha256"] = content_hash.hexdigest()
        del tensor["offset"]
    r.span(size - r.offset)
    tensors.sort(key=lambda item: item["name_sha256"])
    metadata = {name: {"fields": len(entries),
                       "sha256": digest(canonical(sorted(entries))) if entries else None}
                for name, entries in groups.items()}
    return {"gguf_version": version, "architecture": architecture, "bytes": size,
            "container_sha256": r.file_hash.hexdigest(), "tensor_count": len(tensors),
            "tensor_bytes": sum(t["bytes"] for t in tensors),
            "tensor_set_sha256": digest(canonical(tensors)), "tensors": tensors,
            "metadata_fingerprints": metadata,
            "tensor_values_or_health_checked": False,
            "boundary": "Bitwise F16/F32 comparison only; no numerical, provenance or model-quality qualification."}


def inspect_file(path: Path, expected: str, seconds: float = 600) -> dict:
    """Bind parsed bytes to a historical blob digest, not just a filename."""
    require(re.fullmatch(r"[a-f0-9]{64}", expected) is not None, "EXPECTED_DIGEST_REQUIRED")
    require(plain_path(path), "LINKED_INPUT_REFUSED")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "REGULAR_FILE_REQUIRED")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require_stat_match(before, opened, "INPUT_CHANGED_AT_OPEN", same_api=False)
        result = fingerprint_stream(stream, opened.st_size, seconds)
        require_stat_match(opened, os.fstat(stream.fileno()),
                           "INPUT_CHANGED_DURING_READ", same_api=True)
    require(plain_path(path), "INPUT_CHANGED_AFTER_READ:linked_path")
    require_stat_match(before, path.stat(), "INPUT_CHANGED_AFTER_READ", same_api=True)
    require(result["container_sha256"] == expected, "BLOB_DIGEST_MISMATCH")
    result["matches_historical_from_digest"] = True
    result["file_identity_observation"] = {
        "protocol": "equivalent-fields-and-same-api-stability/v1",
        "windows_cross_api_ctime_difference":
            (before.st_ctime_ns != opened.st_ctime_ns) if os.name == "nt" else None,
        "descriptor_and_path_stability_checked": True,
    }
    return result


def compare(left: dict, right: dict) -> dict:
    """Only called for two fully inspected files. Missing groups remain unknown."""
    left_map = {t["name_sha256"]: t for t in left["tensors"]}
    right_map = {t["name_sha256"]: t for t in right["tensors"]}
    shared = left_map.keys() & right_map.keys()
    equal = [key for key in shared if left_map[key] == right_map[key]]
    def group_match(name: str):
        a, b = (x["metadata_fingerprints"][name]["sha256"] for x in (left, right))
        return (a == b) if a is not None and b is not None else None
    return {"same_container_bytes": left["container_sha256"] == right["container_sha256"],
            "same_tensor_set_bytes": left["tensor_set_sha256"] == right["tensor_set_sha256"],
            "shared_tensor_names": len(shared), "identical_tensors": len(equal),
            "different_shared_tensors": len(shared) - len(equal),
            "left_only_tensors": len(left_map.keys() - right_map.keys()),
            "right_only_tensors": len(right_map.keys() - left_map.keys()),
            "same_serialized_tokenizer_metadata": group_match("tokenizer"),
            "same_serialized_chat_template_metadata": group_match("chat_template"),
            "same_serialized_architecture_metadata": group_match("architecture"),
            "root_cause_proven": False, "model_qualified": False}



def write_evidence(path: Path, value: dict) -> None:
    """Create one flushed, fsynced evidence file; never replace or retry it.

    Serialize before opening so invalid data does not create an empty record.
    A write/fsync failure stops admission of further reads. Partial files remain
    evidence of uncertainty, not permission to resume or repeat an experiment.
    """
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(plain_path(path), "LINKED_OUTPUT_REFUSED")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint(out: Path, report: dict, sequence: int, phase: str) -> None:
    """Preserve read intent/results independently of final-report completion."""
    write_evidence(out / f"checkpoint-{sequence:03d}.json", {
        "schema": "szl.gguf-forensics-checkpoint/v1", "sequence": sequence,
        "phase": phase, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "automatic_resume_authorized": False, "report": report,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-blobs", action="store_true", help="Explicitly read both historical blobs; CPU/disk only")
    parser.add_argument("--blob-root", type=Path, help="Owner-selected local blob directory; defaults to ~/.ollama/models/blobs")
    args = parser.parse_args(argv)
    if not args.inspect_blobs:
        print(json.dumps({"state": "PLAN_ONLY_NO_BLOB_READ", "targets": list(BLOBS),
                          "inference": False, "training": False}))
        return 0
    require(os.name == "nt" and os.environ.get("COMPUTERNAME", "").casefold() == "betterwithage",
            "BETTERWITHAGE_WINDOWS_ONLY")
    home = Path.home()
    root = args.blob_root or home / ".ollama" / "models" / "blobs"
    # Admit both storage locations before any path metadata read or report write.
    require_local_windows_path(home)
    require_local_windows_path(root)
    require(plain_path(root) and root.is_dir(), "LOCAL_BLOB_DIRECTORY_REQUIRED")
    require(plain_path(home), "LINKED_HOME_REFUSED")
    out = home / ("szl-gguf-forensics-" + uuid.uuid4().hex)
    require(not out.is_relative_to(root), "OUTPUT_INSIDE_BLOB_ROOT")
    out.mkdir(exist_ok=False)
    report = {"schema": "szl.gguf-forensics/v1", "version": VERSION,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256": digest(Path(__file__).read_bytes()), "spec_commit": SPEC_COMMIT,
              "state": "INCOMPLETE", "models": {}, "comparison": None,
              "inference": False, "training": False, "model_weights_changed": False,
              "paired_attempts_touched": False, "publication_eligible": False,
              "expected_historical_blobs": dict(BLOBS), "active_model": None,
              "automatic_resume_authorized": False}
    checkpoint(out, report, 0, "INSPECTION_PLANNED")
    print("Evidence directory:", out, flush=True)
    for index, (name, expected) in enumerate(BLOBS.items(), start=1):
        report["active_model"] = name
        checkpoint(out, report, 2 * index - 1, "BLOB_READ_INTENT")
        print("Reading historical blob for " + name + " (no inference).", flush=True)
        try:
            report["models"][name] = {"state": "BYTE_IDENTITY_VERIFIED_LAYOUT_INSPECTED",
                                      **inspect_file(root / ("sha256-" + expected), expected)}
        except (ForensicsError, OSError, ValueError, struct.error) as error:
            report["models"][name] = rejected_model(error)
        report["active_model"] = None
        checkpoint(out, report, 2 * index, "BLOB_RESULT_RECORDED")
    models = list(report["models"].values())
    if all(m["state"] == "BYTE_IDENTITY_VERIFIED_LAYOUT_INSPECTED" for m in models):
        report["comparison"] = compare(*models)
        report["state"] = "FORENSICS_COMPLETED_NOT_MODEL_QUALIFICATION"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_evidence(out / "gguf-forensics.json", report)
    model_status = {name: {key: item.get(key) for key in
                          ("state", "error_code", "tensor_type_observation")}
                    for name, item in report["models"].items()}
    print(json.dumps({"state": report["state"], "models": model_status,
                      "comparison": report["comparison"]}, indent=2))
    print("Report:", out / "gguf-forensics.json")
    return 0 if report["comparison"] is not None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ForensicsError, OSError) as error:
        print(json.dumps({"state": "STOPPED_NO_MODEL_MUTATION",
                          "error_code": str(error) if isinstance(error, ForensicsError) else type(error).__name__}))
        raise SystemExit(1) from None
