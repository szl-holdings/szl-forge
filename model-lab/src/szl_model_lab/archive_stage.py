"""Bounded local archive staging and candidate copy; no cloud API or training.

The request and its review digests are operator assertions, not signatures. The
stage proves local byte identity only. A sync folder is not remote restore proof.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile

from .safeio import canonical_bytes, read_regular, strict_json, write_new
from .catalog import track_for


REQUEST_SCHEMA = "szl.model-lab.archive-request/v1"
STAGE_SCHEMA = "szl.model-lab.archive-stage/v1"
COPY_SCHEMA = "szl.model-lab.archive-copy/v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_REQUEST_KEYS = {"schema", "track", "archive_relative_path", "bytes", "sha256",
                 "source_provenance_sha256", "rights_review_sha256",
                 "leakage_review_sha256", "owner_release_sha256", "rights_status",
                 "leakage_status", "owner_release_status"}
_REVIEW_KEYS = ("source_provenance_sha256", "rights_review_sha256",
                "leakage_review_sha256", "owner_release_sha256")
_GATE_STATUS = {"rights_status": "APPROVED_FOR_RESEARCH_TRAINING",
                "leakage_status": "PASS", "owner_release_status": "RELEASED_FOR_RESEARCH_TRAINING"}
_CANDIDATE_LIMITS = {"config.json": 65536, "model.safetensors": 1024 * 1024,
                     "metrics.json": 65536, "README.md": 65536,
                     "manifest.json": 65536}
_UNMATERIALIZED = 0x1000 | 0x40000 | 0x400000  # Windows offline/recall attributes
_SENSITIVE_TEXT = re.compile(
    r"(?i)(?:[a-z]:[\\/]|\\\\|/users/|/home/|"
    r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|"
    r"(?:ghp_|gho_|github_pat_|hf_|sk-)[a-z0-9_-]{12,})"
)
_TENSOR_SHAPES = {"network.0.weight": [16, 8], "network.0.bias": [16],
                  "network.2.weight": [1, 16], "network.2.bias": [1]}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


def _windows_path_metadata(path: Path) -> tuple[int, int]:
    """Query the reparse point itself, without opening a payload or recalling it.

    Python 3.11 lstat and handle queries can hide Cloud Files tags. Query the
    exact directory entry instead; never enumerate with a wildcard. Drive/share
    roots use native attributes, and reparse roots are conservatively rejected.
    Native lookup errors are terminal; never fall back to masked lstat data.
    """
    import ctypes
    from ctypes import wintypes

    class FindData(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD),
                    ("creation", wintypes.FILETIME), ("access", wintypes.FILETIME),
                    ("write", wintypes.FILETIME), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("tag", wintypes.DWORD),
                    ("reserved", wintypes.DWORD), ("name", wintypes.WCHAR * 260),
                    ("alternate", wintypes.WCHAR * 14)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    lookup = kernel.FindFirstFileW
    lookup.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(FindData))
    lookup.restype = wintypes.HANDLE
    close = kernel.FindClose
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    absolute = path.absolute()
    name = str(absolute)
    payload_name = name[4:] if name.startswith("\\\\?\\") else name
    if "*" in payload_name or "?" in payload_name:
        raise ValueError("native_metadata_wildcard_rejected")
    if not name.startswith("\\\\?\\"):
        name = ("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\")
                else "\\\\?\\" + name)
    if absolute == Path(absolute.anchor):
        attributes = kernel.GetFileAttributesW
        attributes.argtypes = (wintypes.LPCWSTR,)
        attributes.restype = wintypes.DWORD
        value = int(attributes(name))
        if value == 0xffffffff:
            raise OSError(ctypes.get_last_error(), "native_path_metadata_unavailable")
        if value & 0x400:
            raise ValueError("native_reparse_root_rejected")
        return value, 0
    info = FindData()
    handle = lookup(name, ctypes.byref(info))
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "native_path_metadata_unavailable")
    try:
        return int(info.attributes), int(info.tag) if info.attributes & 0x400 else 0
    finally:
        if not close(handle):
            raise OSError(ctypes.get_last_error(), "native_path_metadata_close_failed")


def _file_attributes(path: Path) -> int:
    if os.name == "nt":
        return _windows_path_metadata(path)[0]
    return getattr(path.lstat(), "st_file_attributes", 0)


def _reparse_tag(path: Path) -> int:
    if os.name == "nt":
        return _windows_path_metadata(path)[1]
    return getattr(path.lstat(), "st_reparse_tag", 0)


def _placeholder_state(attributes: int, tag: int) -> int:
    if os.name != "nt":
        return 0xffffffff
    import ctypes
    try:
        function = ctypes.WinDLL("CldApi.dll").CfGetPlaceholderStateFromAttributeTag
        function.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
        function.restype = ctypes.c_uint32
        return int(function(attributes, tag))
    except (AttributeError, OSError):
        return 0xffffffff


def _safe_existing_path(path: Path, *, allow_cloud: bool = False) -> None:
    for part in (path, *path.parents):
        attributes = _file_attributes(part)
        if part.is_symlink():
            raise ValueError("symlink_or_reparse_path_rejected")
        if attributes & _UNMATERIALIZED:
            raise ValueError("path_not_fully_materialized")
        if attributes & 0x400:
            tag = _reparse_tag(part)
            state = _placeholder_state(attributes, tag) if allow_cloud else 0xffffffff
            if (not allow_cloud or tag & 0xffff0fff != 0x9000001a
                    or state == 0xffffffff or not state & 0x3
                    or state & 0x30):
                raise ValueError("symlink_or_reparse_path_rejected")


def _read_archive_regular(path: Path, limit: int) -> bytes:
    """Read a hydrated Cloud Files item without allowing name-surrogate reparses."""
    _safe_existing_path(path, allow_cloud=True)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError("archive_file_size_or_type")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino, opened.st_size)
                != (before.st_dev, before.st_ino, before.st_size)):
            raise ValueError("archive_file_changed_before_read")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size
            or (opened.st_mtime_ns, opened.st_size) != (after.st_mtime_ns, after.st_size)):
        raise ValueError("archive_file_changed_during_read")
    _safe_existing_path(path, allow_cloud=True)
    return data


def _safe_new_root(path: Path, *, forbidden: Path) -> Path:
    original = path.absolute()
    _safe_existing_path(original.parent)
    path = original.resolve(strict=False)
    if original.exists() or original.is_symlink():
        raise ValueError("output_must_be_new")
    if path.is_relative_to(forbidden.resolve()):
        raise ValueError("output_inside_source_rejected")
    repo = Path(__file__).resolve().parents[3]
    if path.is_relative_to(repo):
        raise ValueError("output_inside_source_checkout_rejected")
    return path


def _local_root() -> Path:
    value = os.environ.get("SZL_STAGE_ROOT")
    if not value:
        raise ValueError("SZL_STAGE_ROOT_required")
    return Path(value)


def require_local_workspace(path: Path, local_root: Path, archive_root: Path) -> None:
    """Keep staged data and training output on explicit plain local storage."""
    local_root = local_root.absolute()
    archive_root = archive_root.absolute()
    _safe_existing_path(local_root)
    _safe_existing_path(archive_root, allow_cloud=True)
    if not local_root.is_dir() or not archive_root.is_dir():
        raise ValueError("local_or_archive_root_unavailable")
    local_resolved, archive_resolved = local_root.resolve(), archive_root.resolve()
    repo = Path(__file__).resolve().parents[3]
    if (archive_resolved.is_relative_to(repo)
            or repo.is_relative_to(archive_resolved)):
        raise ValueError("archive_root_overlaps_source_checkout_rejected")
    if (local_resolved.is_relative_to(archive_resolved)
            or archive_resolved.is_relative_to(local_resolved)
            or local_resolved.is_relative_to(repo)):
        raise ValueError("local_root_must_be_separate")
    original = path.absolute()
    _safe_existing_path(original if original.exists() else original.parent)
    resolved = original.resolve(strict=False)
    if resolved.is_relative_to(repo):
        raise ValueError("path_inside_source_checkout_rejected")
    if not resolved.is_relative_to(local_resolved) or resolved == local_resolved:
        raise ValueError("path_outside_local_stage_root")


def _archive_root() -> Path:
    value = os.environ.get("SZL_ARCHIVE_ROOT")
    if not value:
        raise ValueError("SZL_ARCHIVE_ROOT_required")
    return Path(value)


def require_stage_output_disjoint(output: Path, manifest_path: Path) -> None:
    """Protect the exact two-file input stage before model imports or fitting."""
    if ".." in output.parts or ".." in manifest_path.parts:
        raise ValueError("stage_output_path_traversal_rejected")
    original = output.absolute()
    stage = manifest_path.absolute().parent
    _safe_existing_path(stage)
    _safe_existing_path(original if original.exists() or original.is_symlink()
                        else original.parent)
    if original.resolve(strict=False).is_relative_to(stage.resolve(strict=True)):
        raise ValueError("candidate_output_inside_input_stage_rejected")


def _relative_archive_path(value: object) -> Path:
    if (not isinstance(value, str) or not value.endswith(".jsonl")
            or "\\" in value or ":" in value or value.startswith("/")):
        raise ValueError("unsafe_archive_relative_path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("unsafe_archive_relative_path")
    return Path(*parts)


def stage_archive(archive_root: Path, request_path: Path, output: Path,
                  local_root: Path) -> dict:
    """Copy one reviewed, size/hash-pinned JSONL into a new local staging root."""
    require_local_workspace(output, local_root, archive_root)
    request_path = request_path.absolute()
    _safe_existing_path(request_path)
    if request_path.resolve(strict=True).is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("request_inside_source_checkout_rejected")
    request_raw = read_regular(request_path, 65536)
    request = strict_json(request_raw)
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        raise ValueError("archive_request_fields_required")
    if request["schema"] != REQUEST_SCHEMA or request["track"] not in ("router", "invariant"):
        raise ValueError("archive_request_schema_or_track")
    if (type(request["bytes"]) is not int or not 0 < request["bytes"] <= 10 * 1024 * 1024
            or not _hex(request["sha256"]) or any(not _hex(request[k]) for k in _REVIEW_KEYS)
            or any(request[key] != value for key, value in _GATE_STATUS.items())):
        raise ValueError("archive_request_digests_or_size")
    relative = _relative_archive_path(request["archive_relative_path"])
    archive_root = archive_root.absolute()
    if not archive_root.is_dir():
        raise ValueError("archive_root_unavailable")
    _safe_existing_path(archive_root, allow_cloud=True)
    source = archive_root / relative
    raw = _read_archive_regular(source, 10 * 1024 * 1024)
    if len(raw) != request["bytes"] or _sha(raw) != request["sha256"]:
        raise ValueError("archive_source_bytes_mismatch")
    output = _safe_new_root(output, forbidden=archive_root)
    if shutil.disk_usage(output.parent).free < len(raw) * 2 + 65536:
        raise ValueError("insufficient_local_stage_space")
    manifest = {"schema": STAGE_SCHEMA, "state": "STAGED_LOCAL_BYTES_VERIFIED",
                "track": request["track"], "data_file": "data.jsonl",
                "data_bytes": len(raw), "data_sha256": _sha(raw),
                "request_sha256": _sha(request_raw),
                "review_status": "OPERATOR_ASSERTED_NOT_INDEPENDENTLY_VERIFIED",
                **{key: request[key] for key in _REVIEW_KEYS}, **_GATE_STATUS}
    output.mkdir(exist_ok=False)
    write_new(output / "data.jsonl", raw)
    manifest_raw = canonical_bytes(manifest)
    write_new(output / "stage-manifest.json", manifest_raw)
    result = verify_stage(output / "stage-manifest.json", _sha(manifest_raw),
                          output / "data.jsonl", request["track"])
    return {"state": result["state"], "manifest_sha256": _sha(manifest_raw),
            "data_sha256": result["data_sha256"]}


def verify_stage(manifest_path: Path, expected_manifest_sha256: str,
                 data_path: Path, track: str) -> dict:
    """Rehash a fully local stage at the training boundary."""
    if not _hex(expected_manifest_sha256):
        raise ValueError("stage_manifest_digest_required")
    manifest_path = manifest_path.absolute()
    data_path = data_path.absolute()
    if manifest_path.name != "stage-manifest.json" or data_path != manifest_path.parent / "data.jsonl":
        raise ValueError("stage_paths_mismatch")
    if _file_attributes(data_path) & _UNMATERIALIZED:
        raise ValueError("stage_data_not_fully_materialized")
    _safe_existing_path(manifest_path)
    _safe_existing_path(data_path)
    if {p.name for p in manifest_path.parent.iterdir()} != {"data.jsonl", "stage-manifest.json"}:
        raise ValueError("stage_contains_unexpected_files")
    raw = read_regular(manifest_path, 65536)
    if _sha(raw) != expected_manifest_sha256:
        raise ValueError("stage_manifest_mismatch")
    manifest = strict_json(raw)
    expected_keys = {"schema", "state", "track", "data_file", "data_bytes",
                     "data_sha256", "request_sha256", "review_status", *_REVIEW_KEYS,
                     *_GATE_STATUS}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("stage_manifest_fields")
    if (manifest["schema"] != STAGE_SCHEMA or manifest["state"] != "STAGED_LOCAL_BYTES_VERIFIED"
            or manifest["track"] != track or manifest["data_file"] != "data.jsonl"
            or manifest["review_status"] != "OPERATOR_ASSERTED_NOT_INDEPENDENTLY_VERIFIED"
            or type(manifest["data_bytes"]) is not int
            or not 0 < manifest["data_bytes"] <= 10 * 1024 * 1024
            or any(not _hex(manifest[key]) for key in ("data_sha256", "request_sha256", *_REVIEW_KEYS))
            or any(manifest[key] != value for key, value in _GATE_STATUS.items())):
        raise ValueError("stage_manifest_invalid")
    data = read_regular(data_path, 10 * 1024 * 1024)
    if len(data) != manifest["data_bytes"] or _sha(data) != manifest["data_sha256"]:
        raise ValueError("stage_data_mismatch")
    return {key: manifest[key] for key in ("state", "track", "data_sha256",
            "request_sha256", "review_status", *_REVIEW_KEYS, *_GATE_STATUS)} | {
            "stage_manifest_sha256": expected_manifest_sha256}


def _verify_small_safetensors(raw: bytes) -> None:
    """Validate this exact 161-parameter Model Lab layout without importing Torch."""
    if len(raw) < 9:
        raise ValueError("candidate_safetensors_invalid")
    header_size = int.from_bytes(raw[:8], "little")
    if not 0 < header_size <= 65536 or 8 + header_size >= len(raw):
        raise ValueError("candidate_safetensors_invalid")
    header = strict_json(raw[8:8 + header_size])
    if not isinstance(header, dict) or set(header) != set(_TENSOR_SHAPES):
        raise ValueError("candidate_tensor_layout_invalid")
    payload = raw[8 + header_size:]
    spans = []
    for name, shape in _TENSOR_SHAPES.items():
        item = header[name]
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("candidate_tensor_layout_invalid")
        offsets = item["data_offsets"]
        actual_shape = item["shape"]
        if (item["dtype"] != "F32" or not isinstance(actual_shape, list)
                or any(type(value) is not int for value in actual_shape)
                or actual_shape != shape
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * 4
                or offsets[1] > len(payload)):
            raise ValueError("candidate_tensor_layout_invalid")
        spans.append(tuple(offsets))
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload[offsets[0]:offsets[1]])):
            raise ValueError("candidate_tensor_nonfinite")
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("candidate_tensor_offsets_invalid")
        cursor = end
    if cursor != len(payload):
        raise ValueError("candidate_tensor_offsets_invalid")


def archive_candidate(candidate: Path, archive_root: Path, local_root: Path) -> dict:
    """Copy only a completed local Model Lab candidate; record no remote proof."""
    require_local_workspace(candidate, local_root, archive_root)
    candidate = candidate.absolute()
    archive_root = archive_root.absolute()
    _safe_existing_path(candidate)
    _safe_existing_path(archive_root, allow_cloud=True)
    candidate = candidate.resolve(strict=True)
    archive_root = archive_root.resolve(strict=True)
    if not archive_root.is_dir() or candidate.is_relative_to(archive_root):
        raise ValueError("archive_root_unavailable_or_source_inside_archive")
    if {p.name for p in candidate.iterdir()} != set(_CANDIDATE_LIMITS):
        raise ValueError("candidate_file_allowlist_mismatch")
    for name in _CANDIDATE_LIMITS:
        path = candidate / name
        if _file_attributes(path) & _UNMATERIALIZED:
            raise ValueError("candidate_not_fully_materialized")
        _safe_existing_path(path)
    bodies = {name: read_regular(candidate / name, limit)
              for name, limit in _CANDIDATE_LIMITS.items()}
    _verify_small_safetensors(bodies["model.safetensors"])
    for name in ("config.json", "metrics.json", "README.md", "manifest.json"):
        if _SENSITIVE_TEXT.search(bodies[name].decode("utf-8")):
            raise ValueError("candidate_metadata_may_contain_private_content")
    manifest = strict_json(bodies["manifest.json"])
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema", "files", "signed", "publication_eligible"}
            or manifest["schema"] != "szl.model-lab.candidate/v1"
            or manifest["signed"] is not False or manifest["publication_eligible"] is not False
            or not isinstance(manifest["files"], dict)
            or set(manifest["files"]) != set(_CANDIDATE_LIMITS) - {"manifest.json"}
            or any(manifest["files"][name] != _sha(bodies[name]) for name in manifest["files"])):
        raise ValueError("candidate_manifest_invalid")
    config = strict_json(bodies["config.json"])
    config_keys = {"schema", "track", "features", "architecture", "normalization_id",
                   "source", "dataset", "recipe", "state", "purpose",
                   "publication_eligible", "runtime_qualified", "calibration_validated"}
    if (not isinstance(config, dict) or set(config) != config_keys
            or config.get("schema") != "szl.model-lab.candidate/v1"
            or config.get("track") not in ("router", "invariant")
            or config.get("features") != list(track_for(config["track"]).features)
            or config.get("architecture") != "tabular-mlp-8x16x1-v1"
            or not isinstance(config.get("normalization_id"), str)
            or not 0 < len(config["normalization_id"]) <= 128
            or config.get("state") != "TRAINED_UNQUALIFIED"
            or config.get("purpose") != "RESEARCH_ADVISORY_ONLY"
            or config.get("publication_eligible") is not False
            or config.get("runtime_qualified") is not False
            or config.get("calibration_validated") is not False
            or not isinstance(config.get("source"), dict)
            or config["source"].get("verification") != "GIT_CLEAN"
            or not isinstance(config["source"].get("revision"), str)
            or not _HEX40.fullmatch(config["source"]["revision"])
            or not isinstance(config.get("dataset"), dict)
            or not _hex(config["dataset"].get("dataset_sha256"))
            or config["dataset"].get("normalization_id") != config["normalization_id"]
            or not isinstance(config.get("recipe"), dict)
            or config["recipe"].get("device") != "cpu"
            or not isinstance(config["dataset"].get("archive_stage"), dict)):
        raise ValueError("candidate_not_completed_unqualified_run")
    stage = config["dataset"]["archive_stage"]
    if (stage.get("state") != "STAGED_LOCAL_BYTES_VERIFIED"
            or stage.get("track") != config["track"]
            or stage.get("data_sha256") != config["dataset"]["dataset_sha256"]
            or stage.get("review_status") != "OPERATOR_ASSERTED_NOT_INDEPENDENTLY_VERIFIED"
            or any(not _hex(stage.get(key)) for key in
                   ("stage_manifest_sha256", "request_sha256", *_REVIEW_KEYS))
            or any(stage.get(key) != value for key, value in _GATE_STATUS.items())):
        raise ValueError("candidate_archive_stage_gate_missing")
    metrics = strict_json(bodies["metrics.json"])
    if not isinstance(metrics, dict) or metrics.get("split") != "validation" or metrics.get("test_evaluated") is not False:
        raise ValueError("candidate_metrics_not_bounded")
    total = sum(map(len, bodies.values()))
    if shutil.disk_usage(archive_root).free < total * 2 + 65536:
        raise ValueError("insufficient_archive_space")
    parent = archive_root / "model-lab-candidates"
    parent.mkdir(exist_ok=True)
    _safe_existing_path(parent, allow_cloud=True)
    destination = parent / ("candidate-" + _sha(bodies["manifest.json"])[:16])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("archive_candidate_destination_exists")
    pending = Path(tempfile.mkdtemp(prefix=".pending-" + destination.name + "-", dir=parent))
    _safe_existing_path(pending, allow_cloud=True)
    for name, raw in bodies.items():
        write_new(pending / name, raw)
        if _sha(_read_archive_regular(pending / name, _CANDIDATE_LIMITS[name])) != _sha(raw):
            raise ValueError("local_archive_copy_mismatch")
    receipt = {"schema": COPY_SCHEMA, "state": "LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED",
               "local_copy_verified": True, "remote_restore_verified": False,
               "artifact_kind": "SMALL_ADVISORY_INFERENCE_CANDIDATE",
               "checkpoint_completeness": "NOT_RESTARTABLE",
               "candidate_manifest_sha256": _sha(bodies["manifest.json"]),
               "stage_manifest_sha256": stage["stage_manifest_sha256"],
               "files": {name: {"bytes": len(raw), "sha256": _sha(raw)}
                         for name, raw in sorted(bodies.items())},
               "publication_eligible": False}
    receipt_raw = canonical_bytes(receipt)
    write_new(pending / "archive-copy-receipt.json", receipt_raw)
    if _sha(_read_archive_regular(pending / "archive-copy-receipt.json", 65536)) != _sha(receipt_raw):
        raise ValueError("local_archive_receipt_mismatch")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("archive_candidate_destination_exists")
    pending.rename(destination)
    return receipt
