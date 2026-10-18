# SPDX-License-Identifier: Apache-2.0
"""Data-only comparison of pinned root-layout wheel bytes with a pinned tree.

This deliberately narrow profile is NOT a wheel installer, provenance check,
dependency resolver, signed admission or launch permission. The caller must
separately review the selected inventory and supply both expected SHA-256 pins.
Completeness of that inventory (including transitive distributions), bootstrap,
interpreter, stdlib, loaded native libraries and runtime suitability are external
obligations. A match does not authenticate code that has already been imported.

Supported archives are single-disk ZIP32 with contiguous local records followed
by their central directory and uncommented EOCD, with no preamble, gaps, overlays,
extras, comments, encryption or data descriptors. Only STORED and DEFLATED entries
with portable ASCII names and regular-file/directory attributes are supported.
Every root-layout file, including RECORD, startup hooks and shared libraries, is
opaque and must exactly match the native manifest. No .data relocation, rewritten
RECORD, generated wrapper/bytecode or unexplained installed file is accepted. This
is a strict byte-layout subset, not validation of general wheel metadata or tags.

No input is extracted, imported, executed, installed or discovered through package
metadata. Use an independently trusted stdlib-only bootstrap before site startup;
importing this module cannot undo startup code. All reads are bounded, and wheel
hashing and parsing use the same descriptor. Revalidation detects ordinary change
under a trusted single writer, not a hostile same-UID writer or immutable snapshot.
Keep inputs separately read-only before consumption. Success returns immutable
counts and digests only, never a capability or production-readiness verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import zipfile
import zlib

import native_identity as native

INVENTORY_SCHEMA = "szl.receiptagent.native-wheel-inventory/v1"
MAX_INVENTORY_BYTES = 1024**2
MAX_WHEELS = 128
MAX_WHEEL_BYTES = 2**32 - 1
MAX_TOTAL_WHEEL_BYTES = 32 * 1024**3
MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024**2
MAX_MEMBERS = 50000
MAX_TOTAL_MEMBERS = 200000
MAX_NAMESPACE_ENTRIES = 400000
MAX_MEMBER_BYTES = 2**32 - 1
MAX_MEMBER_COMPRESSED_BYTES = 2**32 - 1
MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024**3
MAX_COMPRESSION_RATIO = 200
HASH_CHUNK_BYTES = 1024**2
_HEX = re.compile(r"[0-9a-f]{64}")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_LOCAL = struct.Struct("<4s5H3L2H")
_EOCD = struct.Struct("<4s4H2LH")


class NativeWheelIdentityError(ValueError):
    """Value-free normal display, not secrecy against Python introspection."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if condition is not True:
        raise NativeWheelIdentityError(code)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "DUPLICATE_JSON_FIELD")
        result[key] = value
    return result


def _forbidden_number(_value):
    raise NativeWheelIdentityError("INTEGER_ONLY_INVENTORY")


def _inventory(raw: bytes, native_pin: str) -> list[dict]:
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs,
                           parse_float=_forbidden_number, parse_constant=_forbidden_number)
    except NativeWheelIdentityError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise NativeWheelIdentityError("INVENTORY_JSON_INVALID") from None
    _require(type(value) is dict and set(value) == {"schema", "native_manifest_sha256", "wheels"}
             and value["schema"] == INVENTORY_SCHEMA, "INVENTORY_FIELDS_INVALID")
    _require(value["native_manifest_sha256"] == native_pin, "INVENTORY_NATIVE_PIN_MISMATCH")
    wheels = value["wheels"]
    _require(type(wheels) is list and 1 <= len(wheels) <= MAX_WHEELS, "WHEEL_COUNT_CAP")
    names, folded, total = [], set(), 0
    for entry in wheels:
        _require(type(entry) is dict and set(entry) == {"path", "size_bytes", "sha256"},
                 "WHEEL_FIELDS_INVALID")
        name = native._relative_name(entry["path"])
        _require("/" not in name and name.endswith(".whl"), "WHEEL_BASENAME_REQUIRED")
        _require(name.casefold() not in folded, "WHEEL_NAME_ALIAS")
        names.append(name)
        folded.add(name.casefold())
        size, digest = entry["size_bytes"], entry["sha256"]
        _require(type(size) is int and 0 < size <= MAX_WHEEL_BYTES, "WHEEL_BYTE_CAP")
        _require(type(digest) is str and _HEX.fullmatch(digest) is not None, "WHEEL_DIGEST_INVALID")
        total += size
        _require(total <= MAX_TOTAL_WHEEL_BYTES, "TOTAL_WHEEL_BYTE_CAP")
    _require(names == sorted(names), "WHEEL_ORDER_INVALID")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                           allow_nan=False).encode("ascii")
    _require(raw == canonical, "NONCANONICAL_INVENTORY")
    return wheels


@dataclass(frozen=True)
class _Member:
    raw_name: bytes
    path: str
    directory: bool
    made_by: int
    version: int
    flags: int
    method: int
    time: int
    date: int
    crc: int
    compressed: int
    size: int
    internal_attr: int
    external_attr: int
    offset: int

    @property
    def data_offset(self) -> int:
        return self.offset + _LOCAL.size + len(self.raw_name)


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    _require(len(value) == size, "ZIP_TRUNCATED")
    return value


def _register(names: dict, path: str, directory: bool) -> None:
    """Shared identical directory names are allowed; file overlaps never are."""
    parts = path.split("/")
    for index in range(1, len(parts) + 1):
        name = "/".join(parts[:index])
        is_directory = index < len(parts) or directory
        key = name.casefold()
        previous = names.get(key)
        if previous is not None:
            _require(previous == (name, True) and is_directory, "ZIP_PATH_COLLISION")
        else:
            _require(len(names) < MAX_NAMESPACE_ENTRIES, "ZIP_NAMESPACE_ENTRY_CAP")
            names[key] = (name, is_directory)


def _preflight_zip(stream, size: int) -> tuple[_Member, ...]:
    """Bound and parse actual central records BEFORE ZipFile allocates them."""
    _require(size >= _EOCD.size, "ZIP_EOCD_INVALID")
    stream.seek(size - _EOCD.size)
    signature, disk, cd_disk, on_disk, count, cd_size, cd_offset, comment = _EOCD.unpack(
        _read_exact(stream, _EOCD.size))
    _require(signature == b"PK\x05\x06" and disk == cd_disk == comment == 0
             and on_disk == count and 0 < count < 65535, "ZIP_EOCD_INVALID")
    _require(count <= MAX_MEMBERS, "ZIP_MEMBER_COUNT_CAP")
    _require(cd_size != 0xFFFFFFFF and cd_offset != 0xFFFFFFFF
             and 0 < cd_size <= MAX_CENTRAL_DIRECTORY_BYTES, "ZIP_DIRECTORY_BYTE_CAP")
    _require(0 < cd_offset and cd_offset + cd_size == size - _EOCD.size, "ZIP_LAYOUT_INVALID")
    stream.seek(cd_offset)
    members, seen, names, total = [], set(), {}, 0
    while stream.tell() < cd_offset + cd_size:
        _require(len(members) < MAX_MEMBERS, "ZIP_MEMBER_COUNT_CAP")
        _require(stream.tell() + _CENTRAL.size <= cd_offset + cd_size, "ZIP_DIRECTORY_INVALID")
        (signature, made_by, version, flags, method, time, date, crc, compressed, unpacked,
         name_length, extra, member_comment, disk_start, internal_attr, external_attr,
         offset) = _CENTRAL.unpack(_read_exact(stream, _CENTRAL.size))
        _require(signature == b"PK\x01\x02" and extra == member_comment == disk_start == 0,
                 "ZIP_METADATA_UNSUPPORTED")
        _require(version <= 20 and made_by >> 8 in (0, 3) and flags in (0, 0x800)
                 and method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "ZIP_FEATURE_UNSUPPORTED")
        _require(0 < name_length <= native.MAX_PATH_BYTES + 1
                 and stream.tell() + name_length <= cd_offset + cd_size, "ZIP_NAME_INVALID")
        raw_name = _read_exact(stream, name_length)
        _require(b"\x00" not in raw_name, "ZIP_NAME_INVALID")
        try:
            name = raw_name.decode("ascii")
        except UnicodeError:
            raise NativeWheelIdentityError("ZIP_NAME_INVALID") from None
        directory = name.endswith("/")
        path = native._relative_name(name[:-1] if directory else name)
        _require(not path.split("/")[0].lower().endswith(".data"), "WHEEL_RELOCATION_UNSUPPORTED")
        _require(name not in seen, "ZIP_DUPLICATE_MEMBER")
        seen.add(name)
        _register(names, path, directory)
        mode = stat.S_IFMT(external_attr >> 16)
        _require(mode in ((0, stat.S_IFDIR) if directory else (0, stat.S_IFREG))
                 and (not external_attr & 0x10 or directory)
                 and external_attr & 0xFFFF & ~0x37 == 0
                 and internal_attr in (0, 1), "ZIP_FILE_TYPE_UNSUPPORTED")
        _require(0 <= compressed < 0xFFFFFFFF and compressed <= MAX_MEMBER_COMPRESSED_BYTES
                 and 0 <= unpacked < 0xFFFFFFFF and unpacked <= MAX_MEMBER_BYTES,
                 "ZIP_MEMBER_BYTE_CAP")
        _require(offset < 0xFFFFFFFF, "ZIP64_UNSUPPORTED")
        _require(unpacked <= MAX_COMPRESSION_RATIO * max(1, compressed), "ZIP_COMPRESSION_RATIO_CAP")
        _require(method != zipfile.ZIP_STORED or compressed == unpacked, "ZIP_STORED_SIZE_INVALID")
        if directory:
            _require(compressed == unpacked == crc == 0 and method == zipfile.ZIP_STORED,
                     "ZIP_DIRECTORY_CONTENT_INVALID")
        total += unpacked
        _require(total <= MAX_TOTAL_UNCOMPRESSED_BYTES, "ZIP_TOTAL_BYTE_CAP")
        members.append(_Member(raw_name, path, directory, made_by, version, flags, method,
                               time, date, crc, compressed, unpacked, internal_attr,
                               external_attr, offset))
    _require(stream.tell() == cd_offset + cd_size and len(members) == count, "ZIP_DIRECTORY_COUNT_INVALID")
    cursor = 0
    for member in members:
        _require(member.offset == cursor and cursor + _LOCAL.size <= cd_offset, "ZIP_LAYOUT_INVALID")
        stream.seek(cursor)
        local = _LOCAL.unpack(_read_exact(stream, _LOCAL.size))
        _require(local == (b"PK\x03\x04", member.version, member.flags, member.method,
                           member.time, member.date, member.crc, member.compressed,
                           member.size, len(member.raw_name), 0), "ZIP_LOCAL_HEADER_MISMATCH")
        _require(member.data_offset <= cd_offset
                 and _read_exact(stream, len(member.raw_name)) == member.raw_name, "ZIP_LOCAL_NAME_MISMATCH")
        cursor = member.data_offset + member.compressed
        _require(cursor <= cd_offset, "ZIP_LAYOUT_INVALID")
    _require(cursor == cd_offset, "ZIP_LAYOUT_INVALID")
    return tuple(members)


def _zipfile_agrees(info, member: _Member) -> None:
    _require(info.filename == info.orig_filename == member.raw_name.decode("ascii")
             and info.header_offset == member.offset and info.flag_bits == member.flags
             and info.compress_type == member.method and info.compress_size == member.compressed
             and info.file_size == member.size and info.CRC == member.crc
             and info.create_system == member.made_by >> 8
             and info.external_attr == member.external_attr and info.internal_attr == member.internal_attr
             and info.extra == info.comment == b"", "ZIP_PARSER_DISAGREEMENT")


def _deflated_digest(stream, member: _Member) -> str:
    # ZipExtFile truncates output at declared file_size. Validate the complete raw
    # stream ourselves, with bounded output, EOF, no suffix and exact CRC/length.
    stream.seek(member.data_offset)
    decoder, hasher, crc, observed = zlib.decompressobj(-15), hashlib.sha256(), 0, 0
    remaining = member.compressed
    while remaining:
        block = _read_exact(stream, min(HASH_CHUNK_BYTES, remaining))
        remaining -= len(block)
        pending = block
        while pending:
            previous = len(pending)
            output = decoder.decompress(pending, HASH_CHUNK_BYTES)
            pending = decoder.unconsumed_tail
            observed += len(output)
            _require(observed <= member.size, "ZIP_OBSERVED_SIZE_MISMATCH")
            hasher.update(output)
            crc = zlib.crc32(output, crc)
            _require(not decoder.unused_data and not (decoder.eof and remaining), "ZIP_DEFLATE_SUFFIX")
            _require(output != b"" or len(pending) < previous, "ZIP_DEFLATE_STALLED")
    _require(decoder.eof and not decoder.unused_data and observed == member.size
             and crc == member.crc, "ZIP_DEFLATE_CONTENT_INVALID")
    return hasher.hexdigest()


def _stored_digest(archive, info, member: _Member) -> str:
    hasher, observed = hashlib.sha256(), 0
    with archive.open(info, "r") as stream:
        while True:
            block = stream.read(min(HASH_CHUNK_BYTES, member.size - observed + 1))
            if not block:
                break
            observed += len(block)
            _require(observed <= member.size, "ZIP_OBSERVED_SIZE_MISMATCH")
            hasher.update(block)
    _require(observed == member.size, "ZIP_OBSERVED_SIZE_MISMATCH")
    return hasher.hexdigest()


def _unchanged(path: Path, stream, path_before, fd_before) -> None:
    native._checked_path(path)
    path_after, fd_after = path.lstat(), os.fstat(stream.fileno())
    _require(native._stamp(path_before) == native._stamp(path_after)
             and native._stamp(fd_before) == native._stamp(fd_after)
             and native._stable(path_after) == native._stable(fd_after), "WHEEL_CHANGED_DURING_READ")


def _read_wheel(path: Path, entry: dict, names: dict, files: dict,
                totals: list[int]) -> tuple:
    native._checked_path(path)
    before_path = path.lstat()
    _require(native._plain(before_path) and stat.S_ISREG(before_path.st_mode)
             and before_path.st_nlink == 1, "REGULAR_UNALIASED_FILE_REQUIRED")
    _require(before_path.st_size == entry["size_bytes"] <= MAX_WHEEL_BYTES, "WHEEL_SIZE_MISMATCH")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before_fd = os.fstat(stream.fileno())
        _require(native._stable(before_path) == native._stable(before_fd), "WHEEL_CHANGED_DURING_READ")
        hasher, observed = hashlib.sha256(), 0
        while True:
            block = stream.read(min(HASH_CHUNK_BYTES, entry["size_bytes"] - observed + 1))
            if not block:
                break
            observed += len(block)
            _require(observed <= entry["size_bytes"], "WHEEL_BYTE_CAP")
            hasher.update(block)
        _require(observed == entry["size_bytes"], "WHEEL_SIZE_MISMATCH")
        _require(hasher.hexdigest() == entry["sha256"], "WHEEL_DIGEST_MISMATCH")
        _unchanged(path, stream, before_path, before_fd)
        members = _preflight_zip(stream, observed)
        _unchanged(path, stream, before_path, before_fd)
        totals[0] += len(members)
        totals[1] += sum(member.size for member in members)
        _require(totals[0] <= MAX_TOTAL_MEMBERS and totals[1] <= MAX_TOTAL_UNCOMPRESSED_BYTES,
                 "WHEEL_UNION_RESOURCE_CAP")
        with zipfile.ZipFile(stream, "r", allowZip64=False) as archive:
            infos = archive.infolist()
            _require(len(infos) == len(members), "ZIP_PARSER_DISAGREEMENT")
            for info, member in zip(infos, members):
                _zipfile_agrees(info, member)
                _require(member.directory or member.path not in files, "WHEEL_FILE_OVERLAP")
                _register(names, member.path, member.directory)
                digest = (_stored_digest(archive, info, member) if member.method == zipfile.ZIP_STORED
                          else _deflated_digest(stream, member))
                if not member.directory:
                    files[member.path] = (member.size, digest)
        _unchanged(path, stream, before_path, before_fd)
    return native._stamp(before_path)


@dataclass(frozen=True)
class NativeWheelIdentity:
    """Observed immutable counts/digests, not completeness or launch authority."""
    inventory_sha256: str
    native_manifest_sha256: str
    wheel_count: int
    file_count: int
    total_bytes: int


def verify_native_wheel_inventory(wheel_root: Path, inventory_path: Path,
                                  installed_root: Path, native_manifest_path: Path, *,
                                  expected_inventory_sha256: str,
                                  expected_native_manifest_sha256: str) -> NativeWheelIdentity:
    """Verify exact pinned wheel union, native manifest and installed file bytes.

    Both external pins are mandatory. This checks only the named wheel inventory;
    it does not discover distributions or prove that the chosen set is complete.
    Inputs must remain under a trusted single writer and be separately frozen
    before any later consumption. No partial result is returned on failure.
    """
    for pin in (expected_inventory_sha256, expected_native_manifest_sha256):
        _require(type(pin) is str and _HEX.fullmatch(pin) is not None, "EXTERNAL_PIN_REQUIRED")
    try:
        for path in (wheel_root, inventory_path, installed_root, native_manifest_path):
            native._checked_path(path)
        _require(stat.S_ISDIR(wheel_root.lstat().st_mode), "PLAIN_DIRECTORY_REQUIRED")
        _require(wheel_root != installed_root and wheel_root not in installed_root.parents
                 and installed_root not in wheel_root.parents, "DISJOINT_ROOTS_REQUIRED")
        inventory_digest, raw, inventory_stamp = native._capture_file(
            inventory_path, maximum=MAX_INVENTORY_BYTES, retain=True)
        _require(inventory_digest == expected_inventory_sha256, "INVENTORY_PIN_MISMATCH")
        wheels = _inventory(raw, expected_native_manifest_sha256)
        native_digest, native_raw, native_stamp = native._capture_file(
            native_manifest_path, maximum=native.MAX_MANIFEST_BYTES, retain=True)
        _require(native_digest == expected_native_manifest_sha256, "NATIVE_MANIFEST_PIN_MISMATCH")
        entries, directories, total = native._manifest(native_raw)
        files, names, stamps, totals = {}, {}, {}, [0, 0]
        for entry in wheels:
            path = wheel_root / entry["path"]
            stamps[path] = _read_wheel(path, entry, names, files, totals)
        observed_dirs = {name for name, directory in names.values() if directory}
        _require(files == entries and observed_dirs == directories, "WHEEL_NATIVE_MANIFEST_MISMATCH")
        result = native.verify_native_file_manifest(installed_root, native_manifest_path,
                                                    expected_manifest_sha256=expected_native_manifest_sha256)
        _require(result.file_count == len(files) and result.total_bytes == total, "TREE_RESULT_MISMATCH")
        for path, stamp in {**stamps, inventory_path: inventory_stamp,
                            native_manifest_path: native_stamp}.items():
            native._checked_path(path)
            _require(native._stamp(path.lstat()) == stamp, "INPUT_CHANGED_DURING_READ")
        return NativeWheelIdentity(inventory_digest, native_digest, len(wheels), len(files), total)
    except native.NativeIdentityError as error:
        raise NativeWheelIdentityError(error.code) from None
    except OSError:
        raise NativeWheelIdentityError("FILESYSTEM_UNAVAILABLE") from None
    except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError, EOFError):
        raise NativeWheelIdentityError("ZIP_CONTENT_INVALID") from None
