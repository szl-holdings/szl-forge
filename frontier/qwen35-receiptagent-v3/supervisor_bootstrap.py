#!/usr/bin/env python3
"""Stdlib-only bootstrap and evidence primitives for ReceiptAgent v3.

This module deliberately imports no project sibling and no third-party package.
It is intended to run before ``train_candidate.py`` or
``supervisor_validation.py`` is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELATIVE = "frontier/qwen35-receiptagent-v3"
GIT = "/usr/bin/git"
CANONICAL_ORIGINS = frozenset(
    {
        "https://github.com/szl-holdings/szl-forge",
        "https://github.com/szl-holdings/szl-forge.git",
        "git@github.com:szl-holdings/szl-forge.git",
    }
)
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
EVIDENCE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{0,127}")
REQUIRED_COMPONENTS = (
    "launch_supervised_training.py",
    "supervisor_bootstrap.py",
    "supervise_training.py",
    "containment_probe.py",
    "train_candidate.py",
    "supervisor_validation.py",
)
SIBLING_MODULE_NAMES = {
    "train_candidate.py": "szl_ra3_train_candidate",
    "supervisor_validation.py": "szl_ra3_supervisor_validation",
}
MAX_COMPONENT_BYTES = 2 * 1024 * 1024
NAMESPACE_DIRECTORIES = (
    "usr", "bin", "sbin", "lib", "lib64", "etc",
    "home", "home/rosie", "home/rosie/.venvs",
    "home/rosie/.venvs/szl-unsloth", "opt", "opt/szl-ra3",
    "opt/szl-ra3/input", "opt/szl-ra3/output",
    "opt/szl-ra3/cache", "proc", "sys", "dev", "run",
    "tmp", "var", "var/tmp",
)
NAMESPACE_PLACEHOLDERS = ("etc/ld.so.cache",)
MODEL_CACHE_DIRECTORY = "models--unsloth--Qwen3.5-0.8B"


class BootstrapError(RuntimeError):
    """A source, evidence, or admission invariant was not satisfied."""


class SourceVerificationError(BootstrapError):
    """The requested source was not exact, fresh, clean current main."""


class StrictJSONError(BootstrapError):
    """A JSON evidence file was not one strict, bounded JSON object."""


class DuplicateJSONKey(StrictJSONError):
    """A JSON object contained a duplicate member name."""


class PublicationError(BootstrapError):
    """Base class for write-once publication failures."""


class PublicationNotCommitted(PublicationError):
    """The helper did not create and durably commit the final name."""


class PublicationIndeterminate(PublicationError):
    """The final link exists, but its directory commit could not be confirmed."""

    def __init__(
        self,
        message: str,
        *,
        final_path: Path,
        temporary_path: Path,
        sha256: str,
        size: int,
    ) -> None:
        super().__init__(message)
        self.final_path = final_path
        self.temporary_path = temporary_path
        self.sha256 = sha256
        self.size = size


class AdmissionError(BootstrapError):
    """An attempt leaf could not be admitted or prepared."""


class AdmissionCollision(AdmissionError):
    """The requested immutable attempt leaf already exists."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    size: int
    device: int
    inode: int


@dataclass(frozen=True)
class VerifiedComponent:
    filename: str
    path: Path
    sha256: str
    size: int
    source_bytes: bytes


@dataclass(frozen=True)
class VerifiedSource:
    repository: str
    revision: str
    branch: str
    origin: str
    components: tuple[VerifiedComponent, ...]

    def component(self, filename: str) -> VerifiedComponent:
        matches = [item for item in self.components if item.filename == filename]
        if len(matches) != 1:
            raise SourceVerificationError(
                f"verified component identity is unavailable: {filename}"
            )
        return matches[0]

    def public_evidence(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "branch": self.branch,
            "originIdentityVerified": True,
            "freshRemoteMainObserved": True,
            "cachedRemoteTrackingMatches": True,
            "workingTreeClean": True,
            "components": {
                component.filename: {
                    "bytes": component.size,
                    "sha256": component.sha256,
                }
                for component in self.components
            },
        }


@dataclass(frozen=True)
class StrictJSONDocument:
    path: Path
    value: dict[str, Any]
    raw: bytes
    sha256: str
    size: int
    device: int
    inode: int


@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str
    size: int
    committed: bool
    commit_point: str
    cleanup_complete: bool
    cleanup_error: str | None
    temporary_path: Path | None


@dataclass(frozen=True)
class AdmissionPaths:
    run_id: str
    root: Path
    payload: Path
    logs: Path
    reports: Path
    runtime_cache: Path
    namespace_root: Path
    reserve: Path


@dataclass(frozen=True)
class TombstoneResult:
    artifact: PublishedArtifact | None
    reserve_released: bool
    indeterminate: bool
    error: str | None


@dataclass(frozen=True)
class AdmissionResult:
    paths: AdmissionPaths
    prepared: bool
    failed_stage: str | None
    failure_type: str | None
    created_entries: tuple[str, ...]
    reserve_allocated: bool
    tombstone: TombstoneResult | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _run_git(
    repo_root: Path,
    *arguments: str,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            [GIT, *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            shell=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceVerificationError(
            f"fixed git command failed: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        raise SourceVerificationError(
            f"fixed git command returned nonzero: {arguments[0]}"
        )
    return result


def _one_utf8_line(raw: bytes, label: str) -> str:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SourceVerificationError(f"{label} was not UTF-8") from exc
    if len(lines) != 1 or not lines[0]:
        raise SourceVerificationError(f"{label} was not exactly one nonempty line")
    return lines[0]


def _open_flags(*, directory: bool = False, write: bool = False) -> int:
    flags = os.O_WRONLY if write else os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _read_regular_file_once(path: Path, maximum_bytes: int) -> FileSnapshot:
    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool):
        raise BootstrapError("maximum byte count must be an integer")
    if maximum_bytes <= 0:
        raise BootstrapError("maximum byte count must be positive")
    descriptor = os.open(path, _open_flags())
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BootstrapError("input must be a single-link regular file")
        if before.st_size > maximum_bytes:
            raise BootstrapError("input exceeds its fixed byte ceiling")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise BootstrapError("input grew past its fixed byte ceiling")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or total != before.st_size:
            raise BootstrapError("input changed during its single-open read")
        raw = b"".join(chunks)
        return FileSnapshot(
            path=path,
            raw=raw,
            sha256=_sha256(raw),
            size=total,
            device=before.st_dev,
            inode=before.st_ino,
        )
    finally:
        os.close(descriptor)


def verify_exact_source_before_import(
    source_commit: str,
    *,
    repo_root: Path = ROOT,
    component_dir: Path = HERE,
) -> VerifiedSource:
    """Verify fresh, clean exact main and bind every executable component.

    No project sibling or third-party module is imported by this operation.
    Component bytes returned here are the bytes later compiled by
    :func:`verify_and_load_siblings`.
    """

    if SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise SourceVerificationError(
            "source commit must be exactly 40 lowercase hexadecimal characters"
        )
    root = repo_root.resolve(strict=True)
    components_root = component_dir.resolve(strict=True)
    try:
        components_root.relative_to(root)
    except ValueError as exc:
        raise SourceVerificationError(
            "component directory is outside the repository"
        ) from exc

    origin = _one_utf8_line(
        _run_git(root, "remote", "get-url", "origin").stdout,
        "origin URL",
    )
    if origin not in CANONICAL_ORIGINS:
        raise SourceVerificationError(
            "origin is not the canonical szl-forge repository"
        )

    remote_line = _one_utf8_line(
        _run_git(
            root,
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/heads/main",
        ).stdout,
        "fresh remote main",
    )
    remote_parts = remote_line.split()
    if len(remote_parts) != 2 or remote_parts[1] != "refs/heads/main":
        raise SourceVerificationError("fresh remote main response was malformed")
    if remote_parts[0] != source_commit:
        raise SourceVerificationError("fresh remote main differs from requested source")

    head = _one_utf8_line(_run_git(root, "rev-parse", "HEAD").stdout, "HEAD")
    branch = _one_utf8_line(
        _run_git(root, "branch", "--show-current").stdout,
        "current branch",
    )
    cached_main = _one_utf8_line(
        _run_git(root, "rev-parse", "refs/remotes/origin/main").stdout,
        "cached origin/main",
    )
    dirty = _run_git(root, "status", "--porcelain", "--untracked-files=all").stdout
    if head != source_commit:
        raise SourceVerificationError("HEAD differs from requested source")
    if branch != "main":
        raise SourceVerificationError("source verification requires local main")
    if cached_main != source_commit:
        raise SourceVerificationError(
            "cached origin/main differs from requested source"
        )
    if dirty.strip():
        raise SourceVerificationError("source verification requires a clean worktree")

    verified_components: list[VerifiedComponent] = []
    for filename in REQUIRED_COMPONENTS:
        path = components_root / filename
        local = _read_regular_file_once(path, MAX_COMPONENT_BYTES)
        committed = _run_git(
            root,
            "show",
            f"{source_commit}:{RELATIVE}/{filename}",
        ).stdout
        if len(committed) > MAX_COMPONENT_BYTES:
            raise SourceVerificationError(
                f"committed component is too large: {filename}"
            )
        if local.raw != committed:
            raise SourceVerificationError(
                f"worktree component differs from exact source: {filename}"
            )
        verified_components.append(
            VerifiedComponent(
                filename=filename,
                path=path,
                sha256=local.sha256,
                size=local.size,
                source_bytes=local.raw,
            )
        )
    return VerifiedSource(
        repository="szl-holdings/szl-forge",
        revision=source_commit,
        branch=branch,
        origin=origin,
        components=tuple(verified_components),
    )


def _revalidate_verified_component(component: VerifiedComponent) -> None:
    current = _read_regular_file_once(component.path, MAX_COMPONENT_BYTES)
    if current.raw != component.source_bytes or current.sha256 != component.sha256:
        raise SourceVerificationError(
            f"verified component changed before execution: {component.filename}"
        )


def _execute_verified_module(
    component: VerifiedComponent,
    module_name: str,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(component.path)
    module.__package__ = ""
    module.__loader__ = None
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        code = compile(
            component.source_bytes,
            str(component.path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def verify_and_load_siblings(
    source_commit: str,
    *,
    repo_root: Path = ROOT,
    component_dir: Path = HERE,
) -> tuple[VerifiedSource, Mapping[str, types.ModuleType]]:
    """Verify all component bytes, then execute the two verified siblings.

    Every sibling is rechecked before either sibling executes. Compilation uses
    the already verified in-memory bytes, preventing a path-replacement race.
    """

    verified = verify_exact_source_before_import(
        source_commit,
        repo_root=repo_root,
        component_dir=component_dir,
    )
    for filename in SIBLING_MODULE_NAMES:
        _revalidate_verified_component(verified.component(filename))
    loaded = {
        filename: _execute_verified_module(
            verified.component(filename),
            module_name,
        )
        for filename, module_name in SIBLING_MODULE_NAMES.items()
    }
    return verified, loaded


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise DuplicateJSONKey(f"duplicate JSON key: {key}")
        value[key] = member
    return value


def _strict_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise StrictJSONError("non-finite JSON number")
    return value


def read_strict_json_once(
    path: Path,
    *,
    maximum_bytes: int = 2 * 1024 * 1024,
) -> StrictJSONDocument:
    """Read, hash, and parse one bounded JSON object from one file descriptor."""

    try:
        snapshot = _read_regular_file_once(path, maximum_bytes)
    except BootstrapError as exc:
        raise StrictJSONError(str(exc)) from exc
    if b"\x00" in snapshot.raw:
        raise StrictJSONError("JSON evidence contains NUL")
    try:
        text = snapshot.raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_strict_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                StrictJSONError(f"non-finite JSON number: {token}")
            ),
        )
    except DuplicateJSONKey:
        raise
    except StrictJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StrictJSONError("evidence is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise StrictJSONError("JSON evidence must contain exactly one object")
    return StrictJSONDocument(
        path=path,
        value=value,
        raw=snapshot.raw,
        sha256=snapshot.sha256,
        size=snapshot.size,
        device=snapshot.device,
        inode=snapshot.inode,
    )


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write made no progress")
        remaining = remaining[written:]


def _open_directory(path: Path) -> int:
    descriptor = os.open(path, _open_flags(directory=True))
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise BootstrapError("evidence directory is not a directory")
    return descriptor


def _create_exclusive_at(directory_fd: int, name: str, mode: int = 0o600) -> int:
    flags = _open_flags(write=True) | os.O_CREAT | os.O_EXCL
    return os.open(name, flags, mode, dir_fd=directory_fd)


def _cleanup_temporary(
    directory_fd: int,
    temporary_name: str,
) -> str | None:
    try:
        os.unlink(temporary_name, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"unlink:{type(exc).__name__}"
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        return f"cleanup-fsync:{type(exc).__name__}"
    return None


def publish_write_once(
    directory: Path,
    name: str,
    data: bytes,
) -> PublishedArtifact:
    """Publish immutable bytes without replacement and expose the commit point.

    The artifact becomes committed only after the final hard link exists and
    ``fsync(directory)`` succeeds. Failures after that point are returned as
    cleanup metadata and never downgrade the committed final artifact.
    """

    if EVIDENCE_NAME_PATTERN.fullmatch(name) is None:
        raise PublicationNotCommitted("unsafe evidence filename")
    if not isinstance(data, bytes):
        raise PublicationNotCommitted("evidence payload must be bytes")
    if not directory.is_absolute():
        raise PublicationNotCommitted("evidence directory must be absolute")
    digest = _sha256(data)
    temporary_name = f".{name}.{secrets.token_hex(16)}.tmp"
    final_path = directory / name
    temporary_path = directory / temporary_name
    try:
        directory_fd = _open_directory(directory)
    except (OSError, BootstrapError) as exc:
        raise PublicationNotCommitted(
            f"could not open evidence directory: {type(exc).__name__}"
        ) from exc

    temporary_created = False
    try:
        descriptor = _create_exclusive_at(directory_fd, temporary_name)
        temporary_created = True
        try:
            _write_all(descriptor, data)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
    except Exception as exc:
        cleanup_error = (
            _cleanup_temporary(directory_fd, temporary_name)
            if temporary_created
            else None
        )
        os.close(directory_fd)
        suffix = f"; cleanup={cleanup_error}" if cleanup_error else ""
        raise PublicationNotCommitted(
            f"temporary evidence was not committed: {type(exc).__name__}{suffix}"
        ) from exc

    try:
        os.link(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except Exception as exc:
        cleanup_error = _cleanup_temporary(directory_fd, temporary_name)
        os.close(directory_fd)
        suffix = f"; cleanup={cleanup_error}" if cleanup_error else ""
        raise PublicationNotCommitted(
            f"final evidence name was not created: {type(exc).__name__}{suffix}"
        ) from exc

    try:
        os.fsync(directory_fd)
    except Exception as exc:
        try:
            os.close(directory_fd)
        except OSError:
            pass
        raise PublicationIndeterminate(
            f"final evidence link exists without a confirmed directory commit: {type(exc).__name__}",
            final_path=final_path,
            temporary_path=temporary_path,
            sha256=digest,
            size=len(data),
        ) from exc

    cleanup_error = _cleanup_temporary(directory_fd, temporary_name)
    try:
        os.close(directory_fd)
    except OSError as exc:
        close_error = f"close:{type(exc).__name__}"
        cleanup_error = (
            f"{cleanup_error}; {close_error}" if cleanup_error else close_error
        )
    return PublishedArtifact(
        path=final_path,
        sha256=digest,
        size=len(data),
        committed=True,
        commit_point="FINAL_LINK_AND_DIRECTORY_FSYNC",
        cleanup_complete=cleanup_error is None,
        cleanup_error=cleanup_error,
        temporary_path=temporary_path if cleanup_error else None,
    )


def _admission_paths(runs_root: Path, run_id: str) -> AdmissionPaths:
    root = runs_root / run_id
    return AdmissionPaths(
        run_id=run_id,
        root=root,
        payload=root / "payload",
        logs=root / "logs",
        reports=root / "reports",
        runtime_cache=root / "runtime-cache",
        namespace_root=root / "namespace-root",
        reserve=root / ".evidence-reserve",
    )


def _mkdir_at(directory_fd: int, name: str) -> None:
    os.mkdir(name, mode=0o700, dir_fd=directory_fd)


def _prepare_namespace_scaffold(
    paths: AdmissionPaths, created_entries: list[str]
) -> None:
    os.mkdir(paths.namespace_root, mode=0o700)
    created_entries.append("namespace-root")
    for relative in NAMESPACE_DIRECTORIES:
        target = paths.namespace_root / relative
        os.mkdir(target, mode=0o700)
        created_entries.append(f"namespace-root/{relative}")
    for relative in NAMESPACE_PLACEHOLDERS:
        target = paths.namespace_root / relative
        descriptor = os.open(
            target, _open_flags(write=True) | os.O_CREAT | os.O_EXCL, 0o400
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        created_entries.append(f"namespace-root/{relative}")
    for relative in sorted(NAMESPACE_DIRECTORIES, key=lambda value: value.count("/"), reverse=True):
        descriptor = _open_directory(paths.namespace_root / relative)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = _open_directory(paths.namespace_root)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_runtime_model_target(
    paths: AdmissionPaths, created_entries: list[str]
) -> None:
    hf_fd = _open_directory(paths.runtime_cache / "hf")
    try:
        _mkdir_at(hf_fd, "hub")
        created_entries.append("runtime-cache/hf/hub")
        os.fsync(hf_fd)
    finally:
        os.close(hf_fd)
    hub_fd = _open_directory(paths.runtime_cache / "hf" / "hub")
    try:
        _mkdir_at(hub_fd, MODEL_CACHE_DIRECTORY)
        created_entries.append(f"runtime-cache/hf/hub/{MODEL_CACHE_DIRECTORY}")
        os.fsync(hub_fd)
    finally:
        os.close(hub_fd)


def _allocate_reserve(directory_fd: int, reserve_bytes: int) -> None:
    descriptor = _create_exclusive_at(directory_fd, ".evidence-reserve")
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, reserve_bytes)
        else:
            os.ftruncate(descriptor, reserve_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_reports_directory(paths: AdmissionPaths) -> None:
    try:
        os.mkdir(paths.reports, mode=0o700)
    except FileExistsError:
        pass
    descriptor = _open_directory(paths.reports)
    os.close(descriptor)


def write_admission_tombstone(
    paths: AdmissionPaths,
    *,
    failed_stage: str,
    failure_type: str,
    created_entries: Sequence[str],
) -> TombstoneResult:
    """Release any reserve and publish a fail-closed partial-admission record."""

    reserve_released = False
    try:
        paths.reserve.unlink()
        reserve_released = True
        root_fd = _open_directory(paths.root)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    except FileNotFoundError:
        pass
    except OSError:
        # Tombstone publication is still attempted; its own result is explicit.
        pass
    try:
        _ensure_reports_directory(paths)
        tombstone = {
            "schema": "szl.frontier-training-admission-failure/v1",
            "runId": paths.run_id,
            "state": "ADMISSION_FAILED_PARTIAL_LEAF",
            "failedStage": failed_stage,
            "failureType": failure_type,
            "observedAt": _utc_now(),
            "attemptLeafExclusivelyCreated": True,
            "createdEntries": sorted(set(created_entries)),
            "workerLaunched": False,
            "qualificationEligible": False,
            "receiptEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }
        artifact = publish_write_once(
            paths.reports,
            "admission-failure.json",
            _canonical_json(tombstone) + b"\n",
        )
        return TombstoneResult(
            artifact=artifact,
            reserve_released=reserve_released,
            indeterminate=False,
            error=None,
        )
    except PublicationIndeterminate as exc:
        return TombstoneResult(
            artifact=None,
            reserve_released=reserve_released,
            indeterminate=True,
            error=type(exc).__name__,
        )
    except Exception as exc:
        return TombstoneResult(
            artifact=None,
            reserve_released=reserve_released,
            indeterminate=False,
            error=type(exc).__name__,
        )


def _partial_admission_result(
    paths: AdmissionPaths,
    *,
    failed_stage: str,
    failure: Exception,
    created_entries: Sequence[str],
    reserve_allocated: bool,
) -> AdmissionResult:
    tombstone = write_admission_tombstone(
        paths,
        failed_stage=failed_stage,
        failure_type=type(failure).__name__,
        created_entries=created_entries,
    )
    return AdmissionResult(
        paths=paths,
        prepared=False,
        failed_stage=failed_stage,
        failure_type=type(failure).__name__,
        created_entries=tuple(created_entries),
        reserve_allocated=reserve_allocated and not tombstone.reserve_released,
        tombstone=tombstone,
    )


def admit_attempt_atomic(
    runs_root: Path,
    run_id: str,
    reserve_bytes: int,
) -> AdmissionResult:
    """Exclusively admit one leaf and return truthful state for every later fault.

    Collision before leaf creation raises :class:`AdmissionCollision` without
    modifying the existing leaf. Any failure after exclusive leaf creation is
    returned as a partial result with a best-effort immutable tombstone.
    """

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise AdmissionError("run ID must be exactly 32 lowercase hex characters")
    if not isinstance(reserve_bytes, int) or isinstance(reserve_bytes, bool):
        raise AdmissionError("reserve byte count must be an integer")
    if reserve_bytes <= 0:
        raise AdmissionError("reserve byte count must be positive")
    if not runs_root.is_absolute():
        raise AdmissionError("runs root must be absolute")

    paths = _admission_paths(runs_root, run_id)
    try:
        runs_fd = _open_directory(runs_root)
    except Exception as exc:
        raise AdmissionError(f"could not open runs root: {type(exc).__name__}") from exc
    try:
        os.mkdir(run_id, mode=0o700, dir_fd=runs_fd)
    except FileExistsError as exc:
        os.close(runs_fd)
        raise AdmissionCollision("attempt leaf already exists") from exc
    except Exception as exc:
        os.close(runs_fd)
        raise AdmissionError(
            f"attempt leaf was not admitted: {type(exc).__name__}"
        ) from exc

    created_entries: list[str] = []
    try:
        os.fsync(runs_fd)
    except Exception as exc:
        try:
            os.close(runs_fd)
        except OSError:
            pass
        return _partial_admission_result(
            paths,
            failed_stage="FSYNC_ADMITTED_LEAF",
            failure=exc,
            created_entries=created_entries,
            reserve_allocated=False,
        )
    os.close(runs_fd)

    try:
        attempt_fd = _open_directory(paths.root)
    except Exception as exc:
        return _partial_admission_result(
            paths,
            failed_stage="OPEN_ADMITTED_LEAF",
            failure=exc,
            created_entries=created_entries,
            reserve_allocated=False,
        )

    reserve_allocated = False
    stage = "CREATE_REPORTS_DIRECTORY"
    try:
        _mkdir_at(attempt_fd, "reports")
        created_entries.append("reports")
        os.fsync(attempt_fd)

        stage = "ALLOCATE_EVIDENCE_RESERVE"
        _allocate_reserve(attempt_fd, reserve_bytes)
        reserve_allocated = True
        created_entries.append(".evidence-reserve")
        os.fsync(attempt_fd)

        for name, label in (
            ("payload", "CREATE_PAYLOAD_DIRECTORY"),
            ("logs", "CREATE_LOGS_DIRECTORY"),
            ("runtime-cache", "CREATE_RUNTIME_CACHE_DIRECTORY"),
        ):
            stage = label
            _mkdir_at(attempt_fd, name)
            created_entries.append(name)
        os.fsync(attempt_fd)

        stage = "CREATE_RUNTIME_CACHE_SUBDIRECTORIES"
        cache_fd = _open_directory(paths.runtime_cache)
        try:
            for name in (
                "home",
                "xdg",
                "torch",
                "unsloth",
                "triton",
                "numba",
                "cuda",
                "hf",
            ):
                _mkdir_at(cache_fd, name)
                created_entries.append(f"runtime-cache/{name}")
            os.fsync(cache_fd)
        finally:
            os.close(cache_fd)
        stage = "CREATE_RUNTIME_MODEL_BIND_TARGET"
        _prepare_runtime_model_target(paths, created_entries)
        stage = "CREATE_NAMESPACE_ROOT_SCAFFOLD"
        _prepare_namespace_scaffold(paths, created_entries)
        os.fsync(attempt_fd)
    except Exception as exc:
        try:
            os.close(attempt_fd)
        except OSError:
            pass
        return _partial_admission_result(
            paths,
            failed_stage=stage,
            failure=exc,
            created_entries=created_entries,
            reserve_allocated=reserve_allocated,
        )
    os.close(attempt_fd)
    return AdmissionResult(
        paths=paths,
        prepared=True,
        failed_stage=None,
        failure_type=None,
        created_entries=tuple(created_entries),
        reserve_allocated=True,
        tombstone=None,
    )


__all__ = [
    "AdmissionCollision",
    "AdmissionError",
    "AdmissionPaths",
    "AdmissionResult",
    "BootstrapError",
    "DuplicateJSONKey",
    "PublicationIndeterminate",
    "PublicationNotCommitted",
    "PublishedArtifact",
    "SourceVerificationError",
    "StrictJSONDocument",
    "StrictJSONError",
    "TombstoneResult",
    "VerifiedComponent",
    "VerifiedSource",
    "admit_attempt_atomic",
    "publish_write_once",
    "read_strict_json_once",
    "verify_and_load_siblings",
    "verify_exact_source_before_import",
    "write_admission_tombstone",
]
