"""Verify this project's closed pure-Python wheel without installing/executing it.

Expected code/assets come only from a full immutable Git commit. Generated
metadata is validated separately; this is not a signature or clinical release.
"""
from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import tomllib
import zipfile


MAX_WHEEL_BYTES = 2 * 1024 * 1024
MAX_MEMBER_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 256
MAX_COMPRESSION_RATIO = 100
MIN_FREE_BYTES = 512 * 1024 * 1024
ROOT = "clinical-gateway"
DEPENDENCY_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
DEPENDENCY_BOUND = re.compile(r"(?:<=|>=|==|!=|<|>)[0-9]+(?:\.[0-9]+)*")
EXTRA_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
EXTRA_MARKER = re.compile(r"extra\s*==\s*(?P<quote>['\"])(?P<extra>[a-z0-9]+(?:-[a-z0-9]+)*)(?P=quote)")
AUTHORITY = {"clinical_use_authorized": False, "real_phi_authorized": False,
             "site_validated": False, "production_promotion_allowed": False,
             "source_signature_verified": False, "wheel_executed": False}


class VerificationError(ValueError):
    """A sanitized, stable failure code; no raw subprocess/file content."""


def require(condition, code):
    if not condition:
        raise VerificationError(code)


def require_revision(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
            "IMMUTABLE_GIT_REVISION_REQUIRED")
    return value


def safe_path(value):
    require(isinstance(value, str) and bool(value) and len(value) <= 240
            and re.fullmatch(r"[A-Za-z0-9_./-]+", value) is not None, "UNSAFE_ARCHIVE_PATH")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    require(all(part not in {"", ".", ".."} and not part.endswith(".")
                and part.split(".")[0].upper() not in reserved for part in value.split("/")),
            "UNSAFE_ARCHIVE_PATH")
    return value


def bound_identity(value):
    """Compare only the supported numeric bounds, independent of field order."""
    require(type(value) is str and value.isascii(), "UNSUPPORTED_DEPENDENCY_DECLARATION")
    bounds = tuple(sorted(part.strip() for part in value.split(",")))
    require(len(bounds) == len(set(bounds))
            and all(DEPENDENCY_BOUND.fullmatch(part) is not None for part in bounds),
            "UNSUPPORTED_DEPENDENCY_DECLARATION")
    return bounds


def dependency_identity(value, *, allow_extra):
    """Parse only this package's static name + numeric bound dependency form."""
    require(type(value) is str and value.isascii(), "UNSUPPORTED_DEPENDENCY_DECLARATION")
    parts = value.split(";")
    require(len(parts) <= 2, "UNSUPPORTED_DEPENDENCY_DECLARATION")
    requirement = parts[0].strip()
    name = DEPENDENCY_NAME.match(requirement)
    require(name is not None, "UNSUPPORTED_DEPENDENCY_DECLARATION")
    bounds = requirement[name.end():].strip()
    specifiers = ()
    if bounds:
        specifiers = bound_identity(bounds)
    extra = None
    if len(parts) == 2:
        require(allow_extra, "UNSUPPORTED_DEPENDENCY_DECLARATION")
        marker = EXTRA_MARKER.fullmatch(parts[1].strip())
        require(marker is not None, "UNSUPPORTED_DEPENDENCY_DECLARATION")
        extra = marker.group("extra")
    normalized_name = re.sub(r"[-_.]+", "-", name.group(0)).lower()
    return normalized_name, specifiers, extra


def declared_dependencies(project):
    base = project.get("dependencies", [])
    optional = project.get("optional-dependencies", {})
    require(type(base) is list and type(optional) is dict,
            "UNSUPPORTED_DEPENDENCY_DECLARATION")
    expected = [dependency_identity(value, allow_extra=False) for value in base]
    for extra, values in optional.items():
        require(type(extra) is str and EXTRA_NAME.fullmatch(extra) is not None
                and type(values) is list, "UNSUPPORTED_DEPENDENCY_DECLARATION")
        for value in values:
            identity = dependency_identity(value, allow_extra=False)
            expected.append((identity[0], identity[1], extra))
    return sorted(expected), sorted(optional)


def bounded_command(argv, *, env, limit=MAX_MEMBER_BYTES):
    """Drain a trusted local Git command with bounded memory and elapsed time."""
    collected = bytearray()
    errors = []
    with subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, shell=False) as process:
        def consume():
            try:
                while True:
                    chunk = process.stdout.read(min(65536, limit + 1 - len(collected)))
                    if not chunk:
                        return
                    collected.extend(chunk)
                    if len(collected) > limit:
                        process.kill()
                        return
            except OSError:
                errors.append(True)

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=5)
            raise VerificationError("GIT_READ_TIMEOUT") from exc
        finally:
            reader.join(timeout=5)
        require(not reader.is_alive() and not errors and len(collected) <= limit, "GIT_READ_BOUND_EXCEEDED")
        require(process.returncode == 0, "GIT_SOURCE_UNAVAILABLE")
    return bytes(collected)


def git_read(repo, *args, limit=MAX_MEMBER_BYTES):
    # Inherited redirection and replace refs must not substitute another source.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    return bounded_command(["git", "--no-replace-objects", "-C", str(repo), *args], env=env, limit=limit)


def manifest_from_git(repo, revision):
    revision = require_revision(revision)
    require(git_read(repo, "cat-file", "-t", revision).strip() == b"commit", "GIT_COMMIT_REQUIRED")
    tree = git_read(repo, "ls-tree", "-r", "-z", revision, "--", ROOT)
    objects = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split(" ")
        path = safe_path(raw_path.decode("utf-8"))
        require(path not in objects, "DUPLICATE_GIT_PATH")
        objects[path] = (mode, kind, oid)
    require(len(objects) <= 1024, "GIT_TREE_BOUND_EXCEEDED")
    cache = {}

    def blob(path):
        safe_path(path)
        require(path in objects and objects[path][0] in {"100644", "100755"}
                and objects[path][1] == "blob", "GIT_REGULAR_SOURCE_REQUIRED")
        if path not in cache:
            oid = objects[path][2]
            size = git_read(repo, "cat-file", "-s", oid).strip()
            require(re.fullmatch(rb"[0-9]+", size) is not None and int(size) <= MAX_MEMBER_BYTES,
                    "GIT_BLOB_BOUND_EXCEEDED")
            data = git_read(repo, "cat-file", "blob", oid)
            require(len(data) == int(size) and hashlib.sha1(b"blob " + size + b"\0" + data).hexdigest() == oid,
                    "GIT_BLOB_IDENTITY_MISMATCH")
            cache[path] = data
        return cache[path]

    config = tomllib.loads(blob(ROOT + "/pyproject.toml").decode("utf-8"))
    project = config["project"]
    settings = config["tool"]["setuptools"]
    require(config["build-system"]["build-backend"] == "setuptools.build_meta"
            and set(settings) == {"package-dir", "py-modules", "packages"}
            and settings["package-dir"] == {"": "src"} and not project.get("dynamic"),
            "UNSUPPORTED_PACKAGING_DECLARATION")
    name, version = project["name"], project["version"]
    require(name == "szl-oac-clinical-gateway" and isinstance(version, str)
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version) is not None
            and isinstance(project.get("license"), str), "UNSUPPORTED_DISTRIBUTION_IDENTITY")
    dependencies, extras = declared_dependencies(project)
    modules, packages = settings["py-modules"], settings["packages"]
    require(isinstance(modules, list) and isinstance(packages, list) and modules and packages
            and len(modules + packages) == len(set(modules + packages))
            and all(isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value)
                    for value in modules + packages)
            and packages == ["oac_clinical_resources"], "UNSUPPORTED_PACKAGE_LAYOUT")
    normalized = re.sub(r"[-_.]+", "_", name).lower()
    dist_info = normalized + "-" + version + ".dist-info"
    source = {}

    def add(wheel_path, git_path):
        safe_path(wheel_path)
        require(wheel_path not in source, "DUPLICATE_SOURCE_MAPPING")
        source[wheel_path] = (git_path, blob(git_path))

    for module in modules:
        add(module + ".py", ROOT + "/src/" + module + ".py")
    for package in packages:
        prefix = ROOT + "/src/" + package + "/"
        package_files = [path for path in objects if path.startswith(prefix)]
        require(prefix + "__init__.py" in package_files
                and all("/" not in path[len(prefix):] and path.endswith(".py") for path in package_files),
                "UNDECLARED_PACKAGE_DATA_OR_LAYOUT")
        for path in package_files:
            add(package + "/" + path[len(prefix):], path)

    declarations = [node for node in ast.parse(blob(ROOT + "/setup.py").decode("utf-8")).body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_ASSETS" for target in node.targets)]
    require(len(declarations) == 1, "ASSET_DECLARATION_REQUIRED")
    assets = ast.literal_eval(declarations[0].value)
    require(isinstance(assets, tuple) and 0 < len(assets) <= MAX_MEMBERS
            and len(assets) == len(set(assets)) and all(isinstance(path, str) for path in assets),
            "INVALID_ASSET_DECLARATION")
    entries = []
    for relative in assets:
        path = ROOT + "/" + safe_path(relative)
        add("oac_clinical_resources/assets/" + path, path)
        data = source["oac_clinical_resources/assets/" + path][1]
        entries.append({"path": path, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    encoded = (json.dumps({"schema": "szl-oac/portable-workspace-assets/v1", "package_version": version,
                           "files": entries}, sort_keys=True, indent=2) + "\n").encode("utf-8")
    generated = {"oac_clinical_resources/assets/manifest.json": encoded,
                 "oac_clinical_resources/_asset_integrity.py": (
                     '# Generated by the trusted build from canonical source assets.\n'
                     f'MANIFEST_SHA256 = "{hashlib.sha256(encoded).hexdigest()}"\n').encode("utf-8")}
    require(not set(generated).intersection(source), "GENERATED_SOURCE_COLLISION")
    # The current explicit profile contains exactly one source license. Do not
    # silently widen auto-discovery if packaging policy changes later.
    require(project.get("license-files", ["LICENSE"]) == ["LICENSE"], "UNSUPPORTED_LICENSE_LAYOUT")
    add(dist_info + "/licenses/LICENSE", ROOT + "/LICENSE")
    require(len(source) + len(generated) + 5 <= MAX_MEMBERS
            and sum(len(raw) for _, raw in source.values()) + sum(map(len, generated.values())) <= MAX_TOTAL_BYTES,
            "SOURCE_PACKAGE_BOUND_EXCEEDED")
    return {"source_revision": revision, "name": name, "version": version,
            "requires_python": project["requires-python"], "description": project["description"],
            "license": project["license"], "scripts": project["scripts"], "top_levels": set(modules + packages),
            "dependencies": dependencies, "extras": extras,
            "dist_info": dist_info, "wheel_filename": normalized + "-" + version + "-py3-none-any.whl",
            "source": source, "generated": generated}


def metadata_headers(raw):
    raw.decode("utf-8")
    # compat32 retains raw header values. The default policy decodes RFC 2047
    # encoded words, which could launder a changed Name or Requires-Dist field.
    message = BytesParser(policy=policy.compat32).parsebytes(raw)
    require(not message.defects, "MALFORMED_GENERATED_METADATA")
    return message


def single_header(message, name, expected=None):
    values = message.get_all(name, [])
    require(len(values) == 1 and (expected is None or str(values[0]) == expected), "GENERATED_METADATA_MISMATCH")
    return str(values[0])


def validate_generated_metadata(files, contract):
    prefix = contract["dist_info"] + "/"
    metadata = metadata_headers(files[prefix + "METADATA"])
    for field, expected in (("Metadata-Version", "2.4"), ("Name", contract["name"]), ("Version", contract["version"]),
                            ("Summary", contract["description"]),
                            ("License-Expression", contract["license"]), ("License-File", "LICENSE")):
        single_header(metadata, field, expected)
    require(bound_identity(single_header(metadata, "Requires-Python"))
            == bound_identity(contract["requires_python"]), "GENERATED_METADATA_MISMATCH")
    actual_extras = [str(value) for value in metadata.get_all("Provides-Extra", [])]
    require(sorted(actual_extras) == contract["extras"], "GENERATED_EXTRA_MISMATCH")
    actual_dependencies = sorted(dependency_identity(str(value), allow_extra=True)
                                 for value in metadata.get_all("Requires-Dist", []))
    require(actual_dependencies == contract["dependencies"], "GENERATED_DEPENDENCY_MISMATCH")
    wheel = metadata_headers(files[prefix + "WHEEL"])
    require(set(wheel.keys()) == {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"}, "GENERATED_WHEEL_FIELDS_MISMATCH")
    for field, expected in (("Wheel-Version", "1.0"), ("Root-Is-Purelib", "true"), ("Tag", "py3-none-any")):
        single_header(wheel, field, expected)
    require(bool(single_header(wheel, "Generator")), "GENERATED_METADATA_MISMATCH")
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(files[prefix + "entry_points.txt"].decode("utf-8"))
    require(parser.sections() == ["console_scripts"] and not parser.defaults()
            and dict(parser["console_scripts"]) == contract["scripts"], "GENERATED_ENTRY_POINTS_MISMATCH")
    top_levels = files[prefix + "top_level.txt"].decode("utf-8").splitlines()
    require(len(top_levels) == len(set(top_levels)) and set(top_levels) == contract["top_levels"],
            "GENERATED_TOP_LEVEL_MISMATCH")


def verify_wheel_bytes(raw, filename, contract):
    require(filename == contract["wheel_filename"], "WHEEL_FILENAME_MISMATCH")
    require(0 < len(raw) <= MAX_WHEEL_BYTES, "WHEEL_SIZE_BOUND_EXCEEDED")
    files = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            require(0 < len(entries) <= MAX_MEMBERS, "ARCHIVE_MEMBER_BOUND_EXCEEDED")
            aliases = set()
            total = 0
            for entry in entries:
                name = safe_path(entry.filename)
                require(entry.orig_filename == name and name.casefold() not in aliases, "ARCHIVE_PATH_ALIAS")
                aliases.add(name.casefold())
                kind = stat.S_IFMT(entry.external_attr >> 16)
                require(not entry.is_dir() and not entry.external_attr & 0x10 and kind in {0, stat.S_IFREG},
                        "ARCHIVE_REGULAR_FILE_REQUIRED")
                require(not entry.flag_bits & (1 | 64), "ENCRYPTED_ARCHIVE_REJECTED")
                require(entry.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}, "UNSUPPORTED_ZIP_COMPRESSION")
                require(0 <= entry.file_size <= MAX_MEMBER_BYTES and 0 <= entry.compress_size <= MAX_WHEEL_BYTES
                        and entry.file_size <= max(1, entry.compress_size) * MAX_COMPRESSION_RATIO,
                        "ARCHIVE_EXPANSION_BOUND_EXCEEDED")
                total += entry.file_size
                require(total <= MAX_TOTAL_BYTES, "ARCHIVE_TOTAL_BOUND_EXCEEDED")
            for entry in entries:
                with archive.open(entry) as stream:
                    payload = stream.read(MAX_MEMBER_BYTES + 1)
                require(len(payload) == entry.file_size and len(payload) <= MAX_MEMBER_BYTES,
                        "ARCHIVE_READ_SIZE_MISMATCH")
                files[entry.filename] = payload
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError) as exc:
        raise VerificationError("MALFORMED_WHEEL_ARCHIVE") from exc
    prefix = contract["dist_info"] + "/"
    generated_metadata = {prefix + name for name in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD")}
    expected = set(contract["source"]) | set(contract["generated"]) | generated_metadata
    require(set(files) == expected, "WHEEL_CLOSED_FILE_SET_MISMATCH")
    record_path = prefix + "RECORD"
    try:
        rows = list(csv.reader(io.StringIO(files[record_path].decode("utf-8"), newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise VerificationError("MALFORMED_WHEEL_RECORD") from exc
    record = {}
    for row in rows:
        require(len(row) == 3, "MALFORMED_WHEEL_RECORD")
        name, digest, size = row
        safe_path(name)
        require(name not in record, "DUPLICATE_RECORD_ROW")
        record[name] = (digest, size)
    require(set(record) == set(files), "RECORD_CLOSED_FILE_SET_MISMATCH")
    for name, payload in files.items():
        digest, size = record[name]
        if name == record_path:
            require(digest == size == "", "RECORD_SELF_HASH_REJECTED")
        else:
            wanted = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
            require(digest == wanted and size == str(len(payload)), "RECORD_HASH_OR_SIZE_MISMATCH")
    source_files, derived_files = [], []
    for name, (git_path, expected_bytes) in sorted(contract["source"].items()):
        require(files[name] == expected_bytes, "WHEEL_SOURCE_MISMATCH")
        source_files.append({"wheel_path": name, "git_path": git_path, "size_bytes": len(expected_bytes),
                             "sha256": hashlib.sha256(expected_bytes).hexdigest(), "provenance": "EXACT_IMMUTABLE_GIT_BYTES"})
    for name, expected_bytes in sorted(contract["generated"].items()):
        actual_bytes = files[name]
        platform_text = name == "oac_clinical_resources/_asset_integrity.py" \
            and actual_bytes == expected_bytes.replace(b"\n", b"\r\n")
        require(actual_bytes == expected_bytes or platform_text, "DERIVED_ASSET_INTEGRITY_MISMATCH")
        derived_files.append({"wheel_path": name, "size_bytes": len(actual_bytes),
                              "sha256": hashlib.sha256(actual_bytes).hexdigest(),
                              "provenance": "RECONSTRUCTED_FROM_GIT_ASSETS"})
    validate_generated_metadata(files, contract)
    return {"schema": "szl-oac/wheel-source-verification/v1", "state": "WHEEL_SOURCE_VERIFIED", "complete": True,
            "scope": "closed_project_profile_source_bytes_derived_assets_and_validated_generated_metadata",
            "source_revision": contract["source_revision"], "wheel_filename": filename, "wheel_size_bytes": len(raw),
            "wheel_sha256": hashlib.sha256(raw).hexdigest(), "member_count": len(files),
            "record_hashed_members": len(files) - 1, "source_files": source_files, "derived_files": derived_files,
            "generated_metadata_validated": sorted(generated_metadata), **AUTHORITY}


def verify_wheel(wheel, repo, revision):
    require_revision(revision)
    require(shutil.disk_usage(repo).free >= MIN_FREE_BYTES, "LOCAL_STORAGE_RESERVE_REQUIRED")
    observed = wheel.lstat()
    require(stat.S_ISREG(observed.st_mode) and not getattr(observed, "st_file_attributes", 0) & 0x400,
            "REGULAR_WHEEL_FILE_REQUIRED")
    require(0 < observed.st_size <= MAX_WHEEL_BYTES, "WHEEL_SIZE_BOUND_EXCEEDED")
    with wheel.open("rb") as handle:
        raw = handle.read(MAX_WHEEL_BYTES + 1)
        require(os.fstat(handle.fileno()).st_size == observed.st_size and len(raw) == observed.st_size,
                "WHEEL_FILE_CHANGED_DURING_READ")
    return verify_wheel_bytes(raw, wheel.name, manifest_from_git(repo, revision))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output", required=True, help="NEW receipt path; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    revision = args.git_revision if re.fullmatch(r"[0-9a-f]{40}", args.git_revision) else None
    try:
        receipt = verify_wheel(Path(args.wheel), Path(__file__).resolve().parents[2], args.git_revision)
    except (VerificationError, OSError, ValueError, KeyError, TypeError, SyntaxError, RecursionError) as exc:
        receipt = {"schema": "szl-oac/wheel-source-verification/v1", "state": "WHEEL_SOURCE_VERIFICATION_FAILED",
                   "complete": False, "source_revision": revision,
                   "failure_code": str(exc) if isinstance(exc, VerificationError) else "WHEEL_EVIDENCE_UNAVAILABLE",
                   **AUTHORITY}
    encoded = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    print(encoded.decode("utf-8"), end="")
    try:
        with Path(args.output).open("xb") as handle:
            handle.write(encoded)
    except OSError:
        print("WHEEL_RECEIPT_NOT_SAVED_EXISTING_EVIDENCE_PRESERVED", file=sys.stderr)
        return 2
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
