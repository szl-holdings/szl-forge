#!/usr/bin/env python3
"""Copy one completed ReceiptAgent adapter to a private local archive.

This is byte preservation, not model validation, checkpoint restartability,
OneDrive sync confirmation, remote restore, publication, or promotion.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import uuid


HERE = Path(__file__).resolve().parent
LOCAL_RESULTS_ROOT = HERE / "results"
REQUEST_SCHEMA = "szl.native-archive-request/v1"
COPY_SCHEMA = "szl.native-archive-copy/v1"
REPORT_SCHEMA = "szl.native-bf16-continuation/v1"
BASE_REPO = "unsloth/Qwen3.5-0.8B"
BASE_REVISION = "23c69c53358a07516b5827588b3fdb12ae78fd65"
BASE_MANIFEST_SHA256 = "38838f6e620416c7d29b360de7d6b674cb85fd4e99ece5c9144f9e8034bbc0e6"
PARENT_REPO = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2"
PARENT_REVISION = "46f54373c6bf8f17a288b4c8799e9fbe4b82ecc1"
PARENT_MANIFEST_SHA256 = "c0068e21b3a686dd9a9136520a9c5c4a01693e7e5616683b41d60155420cdaa6"
PARENT_WEIGHT_SHA256 = "885fc29fcb4cf55c280dc085fdb0a40f40d6b946fee400dd5e4ed3459fe6334f"

MAX_REPORT_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 65536
MAX_FILES = 64
MAX_FILE_BYTES = 384 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MIN_ARCHIVE_FREE_BYTES = 512 * 1024**2
ARCHIVE_HEADROOM_BYTES = 64 * 1024**2
CHUNK_BYTES = 1024 * 1024
_UNMATERIALIZED = 0x1000 | 0x40000 | 0x400000  # OFFLINE, RECALL_ON_OPEN/DATA_ACCESS
_REPARSE = 0x400
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REQUEST_KEYS = {"schema", "report_sha256", "source_revision", "runner_sha256",
                 "source_provenance_sha256", "rights_review_sha256",
                 "leakage_review_sha256", "owner_release_sha256", "rights_status",
                 "leakage_status", "owner_release_status"}
_REPORT_ALLOWED_KEYS = {
    "schema", "state", "started_at", "finished_at", "elapsed_seconds",
    "source_revision", "runner_sha256", "candidate_id", "provider_cost_usd",
    "electricity_cost", "publication_eligible", "autonomy_eligible", "steps",
    "recipe", "claim_boundary", "durable_report_written", "disk_admission",
    "base", "adapter_parent", "curriculum_hashes", "unique_training_rows",
    "training_rows_oversampled", "heldout_rows", "exact_prompt_overlap",
    "versions", "gpu", "tokenization", "tracking", "trainable_parameters",
    "base_heldout_loss", "baseline_heldout_loss", "post_heldout_loss",
    "candidate_files", "parent_artifacts_unchanged",
    "candidate_reloaded_and_generation_tested", "heldout_loss_delta",
    "report_sha256",
}
_REVIEW_KEYS = ("source_provenance_sha256", "rights_review_sha256",
                "leakage_review_sha256", "owner_release_sha256")
_GATES = {"rights_status": "APPROVED_FOR_PRIVATE_ARCHIVE",
          "leakage_status": "PASS", "owner_release_status": "RELEASED_FOR_PRIVATE_ARCHIVE"}
_FILE_SUFFIXES = {".json", ".txt", ".jinja", ".model", ".vocab", ".safetensors"}
_SENSITIVE_CONTENT = re.compile(
    rb"(?i)(?:[a-z]:[\\/]|\\\\|/users/|/home/|"
    rb"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|"
    rb"(?:ghp_|gho_|github_pat_|hf_|sk-)[a-z0-9_-]{12,})"
)


class GateError(ValueError):
    """A stable error code without local paths or private content."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise GateError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strict_json(raw: bytes, limit: int) -> object:
    require(len(raw) <= limit, "JSON_TOO_LARGE")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def nonfinite(_):
        raise GateError("NONFINITE_JSON")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise GateError("INVALID_JSON") from None


def _hex(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and bool((_HEX64 if length == 64 else _HEX40).fullmatch(value))


def _native_metadata(path: Path) -> tuple[int, int]:
    """Inspect one exact Windows entry without opening or recalling its payload."""
    class FindData(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
                    ("access", wintypes.FILETIME), ("write", wintypes.FILETIME),
                    ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD),
                    ("tag", wintypes.DWORD), ("reserved", wintypes.DWORD),
                    ("name", wintypes.WCHAR * 260), ("alternate", wintypes.WCHAR * 14)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    name = str(path.absolute())
    bare = name[4:] if name.startswith("\\\\?\\") else name
    require("*" not in bare and "?" not in bare, "PATH_WILDCARD_REJECTED")
    if not name.startswith("\\\\?\\"):
        name = "\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name
    if path.absolute() == Path(path.absolute().anchor):
        attributes = kernel.GetFileAttributesW
        attributes.argtypes = (wintypes.LPCWSTR,)
        attributes.restype = wintypes.DWORD
        value = int(attributes(name))
        if value == 0xffffffff:
            raise GateError("PATH_METADATA_UNAVAILABLE")
        require(not value & _REPARSE, "REPARSE_ANCHOR_REJECTED")
        return value, 0
    find = kernel.FindFirstFileW
    find.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(FindData))
    find.restype = wintypes.HANDLE
    close = kernel.FindClose
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    data = FindData()
    handle = find(name, ctypes.byref(data))
    if handle == ctypes.c_void_p(-1).value:
        raise GateError("PATH_METADATA_UNAVAILABLE")
    try:
        return int(data.attributes), int(data.tag) if data.attributes & _REPARSE else 0
    finally:
        if not close(handle):
            raise GateError("PATH_METADATA_UNAVAILABLE")


def _entry_metadata(path: Path) -> tuple[int, int]:
    if os.name == "nt":
        return _native_metadata(path)
    entry = path.lstat()
    return getattr(entry, "st_file_attributes", 0), getattr(entry, "st_reparse_tag", 0)


def _placeholder_state(attributes: int, tag: int) -> int:
    if os.name != "nt":
        return 0xffffffff
    try:
        function = ctypes.WinDLL("CldApi.dll").CfGetPlaceholderStateFromAttributeTag
        function.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
        function.restype = ctypes.c_uint32
        return int(function(attributes, tag))
    except (AttributeError, OSError):
        return 0xffffffff


def _safe_existing(path: Path, *, allow_hydrated_cloud: bool = False) -> None:
    # Inspect the lexical path and every ancestor before any resolve operation.
    for part in (path.absolute(), *path.absolute().parents):
        try:
            mode = part.lstat().st_mode
            attributes, tag = _entry_metadata(part)
        except OSError:
            raise GateError("PATH_METADATA_UNAVAILABLE") from None
        require(not stat.S_ISLNK(mode), "SYMLINK_OR_REPARSE_REJECTED")
        require(not attributes & _UNMATERIALIZED, "PATH_NOT_FULLY_MATERIALIZED")
        if attributes & _REPARSE:
            state = _placeholder_state(attributes, tag) if allow_hydrated_cloud else 0xffffffff
            require(allow_hydrated_cloud and tag & 0xffff0fff == 0x9000001a
                    and state != 0xffffffff and bool(state & 0x3) and not state & 0x30,
                    "SYMLINK_OR_REPARSE_REJECTED")


def _regular(path: Path, *, allow_hydrated_cloud: bool = False) -> os.stat_result:
    _safe_existing(path, allow_hydrated_cloud=allow_hydrated_cloud)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "FILE_NOT_REGULAR")
    return before


def _same_file(a: os.stat_result, b: os.stat_result) -> bool:
    return ((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns)
            == (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns))


def _read_small(path: Path, limit: int, *, allow_hydrated_cloud: bool = False) -> bytes:
    before = _regular(path, allow_hydrated_cloud=allow_hydrated_cloud)
    require(0 < before.st_size <= limit, "FILE_SIZE_LIMIT")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode) and _same_file(before, opened), "FILE_CHANGED_BEFORE_READ")
        raw = stream.read(limit + 1)
        require(_same_file(opened, os.fstat(stream.fileno())) and len(raw) == before.st_size,
                "FILE_CHANGED_DURING_READ")
    _safe_existing(path, allow_hydrated_cloud=allow_hydrated_cloud)
    return raw


def _digest_file(path: Path, expected_size: int, *, allow_hydrated_cloud: bool = False,
                 reject_sensitive: bool = False) -> str:
    before = _regular(path, allow_hydrated_cloud=allow_hydrated_cloud)
    require(before.st_size == expected_size, "FILE_SIZE_MISMATCH")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(_same_file(before, opened), "FILE_CHANGED_BEFORE_READ")
        count = 0
        previous = b""
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            count += len(chunk)
            require(count <= expected_size, "FILE_SIZE_MISMATCH")
            if reject_sensitive:
                require(not _SENSITIVE_CONTENT.search(previous + chunk),
                        "CANDIDATE_CONTENT_MAY_CONTAIN_PRIVATE_DATA")
                previous = chunk[-256:]
            digest.update(chunk)
        require(count == expected_size and _same_file(opened, os.fstat(stream.fileno())),
                "FILE_CHANGED_DURING_READ")
    _safe_existing(path, allow_hydrated_cloud=allow_hydrated_cloud)
    return digest.hexdigest()


def _relative_candidate_file(value: object) -> Path:
    require(isinstance(value, str), "UNSAFE_CANDIDATE_PATH")
    require(value and "\\" not in value and ":" not in value and not value.startswith("/"),
            "UNSAFE_CANDIDATE_PATH")
    parts = value.split("/")
    require(len(parts) <= 4 and all(_PART.fullmatch(part) and part not in (".", "..") for part in parts),
            "UNSAFE_CANDIDATE_PATH")
    parsed = PurePosixPath(value)
    require(parsed.suffix in _FILE_SUFFIXES, "CANDIDATE_FILE_TYPE_REJECTED")
    require(parsed.suffix != ".safetensors" or value == "adapter_model.safetensors",
            "CANDIDATE_WEIGHT_LAYOUT_REJECTED")
    return Path(*parts)


def _load_request(path: Path) -> tuple[dict, str]:
    raw = _read_small(path, MAX_REQUEST_BYTES)
    request = _strict_json(raw, MAX_REQUEST_BYTES)
    require(isinstance(request, dict) and set(request) == _REQUEST_KEYS,
            "ARCHIVE_REQUEST_SCHEMA")
    require(request["schema"] == REQUEST_SCHEMA, "ARCHIVE_REQUEST_SCHEMA")
    for key in ("report_sha256", "runner_sha256", *_REVIEW_KEYS):
        require(_hex(request[key]), "ARCHIVE_REQUEST_DIGEST_REQUIRED")
    require(_hex(request["source_revision"], 40), "SOURCE_REVISION_REQUIRED")
    for key, expected in _GATES.items():
        require(request[key] == expected, "ARCHIVE_REVIEW_GATE_NOT_RELEASED")
    return request, _sha(raw)


def _verify_report(raw: bytes, request: dict, run: Path) -> tuple[dict, dict[str, dict], int]:
    require(_sha(raw) == request["report_sha256"], "REPORT_BYTES_MISMATCH")
    report = _strict_json(raw, MAX_REPORT_BYTES)
    require(isinstance(report, dict), "TRAINING_REPORT_SCHEMA")
    require(set(report) <= _REPORT_ALLOWED_KEYS, "TRAINING_REPORT_UNKNOWN_FIELD")
    claimed = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    require(_hex(claimed) and claimed == _sha(_canonical(unsigned)), "REPORT_SELF_HASH_MISMATCH")
    require(report.get("schema") == REPORT_SCHEMA and
            report.get("state") == "MEASURED_LOCAL_CONTINUATION_COMPLETED" and
            report.get("durable_report_written") is True and
            report.get("publication_eligible") is False and
            report.get("autonomy_eligible") is False and
            report.get("parent_artifacts_unchanged") is True and
            report.get("candidate_reloaded_and_generation_tested") is False and
            report.get("provider_cost_usd") == 0 and
            report.get("candidate_id") == run.name,
            "TRAINING_REPORT_NOT_COMPLETED_UNQUALIFIED")
    require(report.get("source_revision") == request["source_revision"] and
            report.get("runner_sha256") == request["runner_sha256"],
            "TRAINING_SOURCE_PROVENANCE_MISMATCH")
    for key, repo, revision, manifest in (
        ("base", BASE_REPO, BASE_REVISION, BASE_MANIFEST_SHA256),
        ("adapter_parent", PARENT_REPO, PARENT_REVISION, PARENT_MANIFEST_SHA256),
    ):
        parent = report.get(key)
        require(isinstance(parent, dict) and parent.get("repo_id") == repo and
                parent.get("revision") == revision and parent.get("manifest_sha256") == manifest,
                "TRAINING_PARENT_PROVENANCE_MISMATCH")
    weights = report["adapter_parent"].get("files")
    require(isinstance(weights, dict) and
            isinstance(weights.get("adapter_model.safetensors"), dict) and
            weights["adapter_model.safetensors"].get("sha256") == PARENT_WEIGHT_SHA256,
            "TRAINING_PARENT_PROVENANCE_MISMATCH")
    hashes = report.get("curriculum_hashes")
    require(isinstance(hashes, dict) and len(hashes) == 6 and all(
        isinstance(name, str) and _hex(digest) for name, digest in hashes.items()),
        "TRAINING_CURRICULUM_PROVENANCE_MISSING")
    entries = report.get("candidate_files")
    require(isinstance(entries, list) and 1 <= len(entries) <= MAX_FILES,
            "CANDIDATE_FILE_COUNT")
    declared = {}
    total = len(raw)
    for item in entries:
        require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
                "CANDIDATE_ENTRY_SCHEMA")
        name = item["path"]
        _relative_candidate_file(name)
        size, digest = item["bytes"], item["sha256"]
        require(name not in declared and type(size) is int and 0 < size <= MAX_FILE_BYTES and
                _hex(digest), "CANDIDATE_ENTRY_INVALID")
        total += size
        require(total <= MAX_TOTAL_BYTES, "CANDIDATE_TOTAL_LIMIT")
        declared[name] = {"bytes": size, "sha256": digest}
    require("adapter_model.safetensors" in declared, "CANDIDATE_WEIGHTS_MISSING")
    return report, declared, total


def _actual_files(adapter: Path, declared: dict[str, dict]) -> set[str]:
    actual = set()
    allowed_directories = {""}
    for name in declared:
        path = PurePosixPath(name)
        allowed_directories.update(str(parent) for parent in path.parents if str(parent) != ".")
    for folder, dirs, files in os.walk(adapter, topdown=True, followlinks=False):
        folder_path = Path(folder)
        _safe_existing(folder_path)
        relative_folder = folder_path.relative_to(adapter).as_posix()
        require(relative_folder in allowed_directories or relative_folder == ".",
                "CANDIDATE_INVENTORY_MISMATCH")
        for name in dirs:
            directory = folder_path / name
            _safe_existing(directory)
            require(directory.relative_to(adapter).as_posix() in allowed_directories,
                    "CANDIDATE_INVENTORY_MISMATCH")
        for name in files:
            entry = folder_path / name
            _regular(entry)
            relative = entry.relative_to(adapter).as_posix()
            _relative_candidate_file(relative)
            actual.add(relative)
            require(len(actual) <= MAX_FILES, "CANDIDATE_FILE_COUNT")
    return actual


def _disk_guard(root: Path, remaining: int) -> None:
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        raise GateError("ARCHIVE_DISK_PROBE_UNAVAILABLE") from None
    require(type(free) is int and free >= MIN_ARCHIVE_FREE_BYTES + remaining + ARCHIVE_HEADROOM_BYTES,
            "ARCHIVE_DISK_LIMIT")


def _copy_verified(source: Path, destination: Path, item: dict, *,
                   destination_cloud: bool) -> None:
    before = _regular(source)
    require(before.st_size == item["bytes"], "CANDIDATE_BYTES_MISMATCH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _safe_existing(destination.parent, allow_hydrated_cloud=destination_cloud)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    with os.fdopen(os.open(source, flags), "rb") as reader:
        opened = os.fstat(reader.fileno())
        require(_same_file(before, opened), "CANDIDATE_CHANGED_BEFORE_COPY")
        with destination.open("xb") as writer:
            count = 0
            for chunk in iter(lambda: reader.read(CHUNK_BYTES), b""):
                count += len(chunk)
                require(count <= item["bytes"], "CANDIDATE_SIZE_CHANGED_DURING_COPY")
                require(writer.write(chunk) == len(chunk), "ARCHIVE_SHORT_WRITE")
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        require(count == item["bytes"] and _same_file(opened, os.fstat(reader.fileno())),
                "CANDIDATE_CHANGED_DURING_COPY")
    _safe_existing(source)
    require(digest.hexdigest() == item["sha256"], "CANDIDATE_BYTES_MISMATCH")
    require(_digest_file(destination, item["bytes"],
                         allow_hydrated_cloud=destination_cloud) == item["sha256"],
            "LOCAL_ARCHIVE_COPY_MISMATCH")


def _copy_via_plain_stage(source: Path, destination: Path, item: dict,
                          stage: Path, archive_root: Path) -> None:
    """Keep unverified source bytes off the sync volume even on failure."""
    _disk_guard(stage.parent, item["bytes"])
    _copy_verified(source, stage, item, destination_cloud=False)
    _disk_guard(archive_root, item["bytes"])
    _copy_verified(stage, destination, item, destination_cloud=True)
    stage.unlink()


def archive_run(run: Path, request_path: Path, archive_root: Path) -> dict:
    """Archive report and adapter only; leave all originals and failures intact."""
    require(all(".." not in Path(p).parts for p in (run, request_path, archive_root)),
            "ARCHIVE_PATH_TRAVERSAL_REJECTED")
    run, request_path, archive_root = map(lambda p: Path(p).absolute(),
                                          (run, request_path, archive_root))
    _safe_existing(run)
    _safe_existing(request_path)
    _safe_existing(archive_root, allow_hydrated_cloud=True)
    require(run.is_dir() and archive_root.is_dir(), "RUN_OR_ARCHIVE_ROOT_UNAVAILABLE")
    results = LOCAL_RESULTS_ROOT.absolute()
    _safe_existing(results)
    results, run, request_path, archive_root = (p.resolve(strict=True) for p in
                                                (results, run, request_path, archive_root))
    repo = HERE.parent.resolve(strict=True)
    require(run.parent == results and run != results and
            not archive_root.is_relative_to(repo) and not repo.is_relative_to(archive_root) and
            not run.is_relative_to(archive_root) and
            not request_path.is_relative_to(repo) and
            not request_path.is_relative_to(archive_root), "ARCHIVE_PATH_PLACEMENT_REJECTED")
    request, request_hash = _load_request(request_path)
    report_path = run / "training-report.json"
    raw = _read_small(report_path, MAX_REPORT_BYTES)
    require(not _SENSITIVE_CONTENT.search(raw), "REPORT_MAY_CONTAIN_PRIVATE_DATA")
    report, declared, total = _verify_report(raw, request, run)
    adapter = run / "adapter"
    _safe_existing(adapter)
    require(adapter.is_dir(), "CANDIDATE_ADAPTER_MISSING")
    require(_actual_files(adapter, declared) == set(declared), "CANDIDATE_INVENTORY_MISMATCH")
    for name, item in declared.items():
        require(_digest_file(adapter / _relative_candidate_file(name), item["bytes"],
                             reject_sensitive=True) ==
                item["sha256"], "CANDIDATE_BYTES_MISMATCH")
    _disk_guard(archive_root, total)
    parent = archive_root / "native-receiptagent"
    if parent.exists() or parent.is_symlink():
        _safe_existing(parent, allow_hydrated_cloud=True)
        require(parent.is_dir(), "ARCHIVE_PARENT_UNAVAILABLE")
    else:
        parent.mkdir(exist_ok=False)
        _safe_existing(parent, allow_hydrated_cloud=True)
    destination = parent / ("candidate-" + request["report_sha256"])
    require(not destination.exists() and not destination.is_symlink(),
            "ARCHIVE_DESTINATION_EXISTS")
    pending = parent / (".pending-" + destination.name + "-" + uuid.uuid4().hex)
    pending.mkdir(exist_ok=False)
    _safe_existing(pending, allow_hydrated_cloud=True)
    files = {"training-report.json": {"bytes": len(raw), "sha256": request["report_sha256"]}}
    with tempfile.TemporaryDirectory(prefix="szl-receiptagent-archive-") as temporary:
        stage_root = Path(temporary).absolute()
        _safe_existing(stage_root)
        stage_root = stage_root.resolve(strict=True)
        require(not stage_root.is_relative_to(repo) and
                not stage_root.is_relative_to(archive_root) and
                not stage_root.is_relative_to(run), "ARCHIVE_STAGE_PLACEMENT_REJECTED")
        stage = stage_root / "verified-payload"
        _copy_via_plain_stage(report_path, pending / "training-report.json",
                              files["training-report.json"], stage, archive_root)
        remaining = total - len(raw)
        for name, item in sorted(declared.items()):
            _disk_guard(archive_root, remaining)
            _copy_via_plain_stage(adapter / _relative_candidate_file(name),
                                  pending / "adapter" / _relative_candidate_file(name),
                                  item, stage, archive_root)
            files["adapter/" + name] = item
            remaining -= item["bytes"]
    receipt = {"schema": COPY_SCHEMA, "state": "LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED",
               "artifact_kind": "NATIVE_RECEIPTAGENT_ADAPTER_AND_REPORT",
               "checkpoint_completeness": "NOT_RESTARTABLE",
               "local_copy_verified": True, "remote_restore_verified": False,
               "publication_eligible": False, "autonomy_eligible": False,
               "model_reload_verified": False,
               "review_status": "OPERATOR_ASSERTED_NOT_INDEPENDENTLY_VERIFIED",
               "request_sha256": request_hash, "training_report_sha256": request["report_sha256"],
               "source_revision": report["source_revision"], "runner_sha256": report["runner_sha256"],
               "base_manifest_sha256": report["base"]["manifest_sha256"],
               "parent_adapter_manifest_sha256": report["adapter_parent"]["manifest_sha256"],
               **{key: request[key] for key in (*_REVIEW_KEYS, *_GATES)},
               "files": files}
    receipt_raw = _canonical(receipt)
    require(len(receipt_raw) <= MAX_REQUEST_BYTES, "ARCHIVE_RECEIPT_TOO_LARGE")
    _disk_guard(archive_root, len(receipt_raw))
    receipt_path = pending / "archive-copy-receipt.json"
    with receipt_path.open("xb") as stream:
        require(stream.write(receipt_raw) == len(receipt_raw), "ARCHIVE_SHORT_WRITE")
        stream.flush()
        os.fsync(stream.fileno())
    require(_sha(_read_small(receipt_path, MAX_REQUEST_BYTES, allow_hydrated_cloud=True)) ==
            _sha(receipt_raw), "LOCAL_ARCHIVE_RECEIPT_MISMATCH")
    _safe_existing(pending, allow_hydrated_cloud=True)
    require(not destination.exists() and not destination.is_symlink(),
            "ARCHIVE_DESTINATION_EXISTS")
    pending.rename(destination)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="completed local-compute/results child; never a remote path")
    parser.add_argument("--request", type=Path, required=True,
                        help="private review request outside the checkout")
    args = parser.parse_args()
    value = os.environ.get("SZL_ARCHIVE_ROOT")
    if not value:
        parser.error("SZL_ARCHIVE_ROOT is required")
    try:
        receipt = archive_run(args.run, args.request, Path(value))
    except (GateError, OSError) as error:
        code = str(error) if isinstance(error, GateError) else "ARCHIVE_LOCAL_IO_FAILED"
        print(json.dumps({"state": "FAILED_CLOSED", "error_code": code,
                          "remote_restore_verified": False, "publication_eligible": False}))
        return 1
    print(json.dumps({"state": receipt["state"],
                      "training_report_sha256": receipt["training_report_sha256"],
                      "remote_restore_verified": False, "publication_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
