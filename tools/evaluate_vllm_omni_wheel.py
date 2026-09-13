"""Acquire and inspect one admitted vLLM-Omni wheel; never install or import it.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
This is byte/source correspondence evidence, not build attestation or runtime
qualification. The network operation is explicit; ordinary imports are offline.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import platform
import re
import stat
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

UPSTREAM = "aff7d64948f6c0e81e9b3272234623044e215967"
ADMISSION = "fa16050c930a2f5a258619101a98bfc2013cb783"
FILENAME = "vllm_omni-0.29.0rc1-py3-none-any.whl"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/b5/bf/"
    "f7601d4a806ed71dc3f17144462194ef41b6601aa2ca81ea3575e343a039/"
    + FILENAME
)
PYPI_URL = "https://pypi.org/pypi/vllm-omni/0.29.0rc1/json"
WHEEL_SIZE = 7275029
WHEEL_SHA256 = "5c5ab1d7e66b2289e81a5f01511bda41f734ea7f02bae1150534e8df16207d79"
METADATA_SHA256 = "d4a701d5851e05b1ab1594897a35d90cedb0dba6333bedeb8cdda1bb36519023"
DIST = "vllm_omni-0.29.0rc1.dist-info/"
GENERATED = "vllm_omni/_version.py"
MAX_MEMBER = 16 * 1024 * 1024
MAX_EXPANDED = 256 * 1024 * 1024
MAX_FILES = 20000
ROOT = Path(__file__).resolve().parents[1]


class AcquisitionError(ValueError):
    """No successful observation may be emitted for malformed inputs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    # Git object identity only, not a substitute for the outer SHA-256 pin.
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data,
        usedforsecurity=False,
    ).hexdigest()


def safe_name(name: str) -> bool:
    return bool(
        name and not name.startswith("/")
        and re.fullmatch(r"[A-Za-z0-9_./+@-]+", name)
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AcquisitionError("redirects are not admitted for fixed public sources")


def download(url: str, maximum: int, expected_size: int | None = None) -> bytes:
    require(url in (PYPI_URL, WHEEL_URL), "unadmitted acquisition URL")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "identity", "User-Agent": "SZL-Forge-byte-audit/1"}
    )
    deadline = time.monotonic() + 90
    with opener.open(request, timeout=20) as response:
        require(response.status == 200 and response.geturl() == url, "source response mismatch")
        require(response.headers.get("Content-Encoding", "identity") == "identity", "encoded transport")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            require(declared.isdigit() and int(declared) <= maximum, "unbounded content length")
            if expected_size is not None:
                require(int(declared) == expected_size, "declared wheel length drift")
        parts: list[bytes] = []
        length = 0
        while True:
            require(time.monotonic() < deadline, "acquisition deadline exceeded")
            chunk = response.read(min(65536, maximum - length + 1))
            if not chunk:
                break
            length += len(chunk)
            require(length <= maximum, "source exceeded byte bound")
            parts.append(chunk)
        if expected_size is not None:
            require(length == expected_size, "actual wheel length drift")
        return b"".join(parts)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def validate_pypi(raw: bytes) -> dict:
    require(len(raw) <= 1024 * 1024, "PyPI response exceeds bound")
    data = json.loads(raw, object_pairs_hook=_unique_pairs)
    require(type(data) is dict and type(data.get("info")) is dict, "malformed PyPI response")
    info = data["info"]
    require(info.get("name") == "vllm-omni" and info.get("version") == "0.29.0rc1", "release drift")
    require(info.get("yanked") is False, "release is yanked or unknown")
    require(type(data.get("urls")) is list, "missing release assets")
    matches = [x for x in data["urls"] if type(x) is dict and x.get("filename") == FILENAME]
    require(len(matches) == 1, "wheel identity is missing or ambiguous")
    item = matches[0]
    require(
        item.get("url") == WHEEL_URL and type(item.get("size")) is int
        and item["size"] == WHEEL_SIZE and item.get("yanked") is False
        and item.get("packagetype") == "bdist_wheel"
        and type(item.get("digests")) is dict
        and item["digests"].get("sha256") == WHEEL_SHA256,
        "PyPI artifact identity drift",
    )
    return {"responseSha256": sha256(raw), "filename": FILENAME, "yanked": False}


def inspect_wheel(data: bytes) -> tuple[dict, dict[str, str]]:
    """Validate bounded ZIP and every RECORD row without extracting any member."""
    require(len(data) <= WHEEL_SIZE, "wheel exceeds admitted byte bound")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        require(0 < len(infos) <= MAX_FILES, "archive member count exceeds bound")
        require(len({x.filename for x in infos}) == len(infos), "duplicate archive path")
        require(sum(x.file_size for x in infos) <= MAX_EXPANDED, "expanded archive exceeds bound")
        for item in infos:
            require(item.orig_filename == item.filename and safe_name(item.filename), "unsafe archive member path")
            require(not item.is_dir() and not item.flag_bits & 1, "directory or encrypted member")
            kind = stat.S_IFMT(item.external_attr >> 16)
            require(kind in (0, stat.S_IFREG), "non-regular archive member")
            require(0 <= item.file_size <= MAX_MEMBER, "member size exceeds bound")
            require(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "unadmitted compression")
        names = {x.filename for x in infos}
        require(DIST + "RECORD" in names and DIST + "METADATA" in names, "missing wheel metadata")
        members = {}
        source_blobs = {}
        metadata = b""
        record = b""
        generated = None
        for item in infos:
            with archive.open(item) as member:
                body = member.read(item.file_size + 1)
            require(len(body) == item.file_size, "member length mismatch")
            members[item.filename] = {"size": len(body), "sha256": sha256(body)}
            if item.filename.startswith("vllm_omni/"):
                source_blobs[item.filename] = git_blob(body)
            if item.filename == DIST + "METADATA":
                metadata = body
            if item.filename == DIST + "RECORD":
                record = body
            if item.filename == GENERATED:
                generated = {"sha256": sha256(body), "size": len(body)}
        rows = list(csv.reader(io.StringIO(record.decode("utf-8"), newline="")))
        require(len(rows) == len(members), "RECORD does not enumerate every member")
        seen = set()
        for row in rows:
            require(len(row) == 3, "malformed RECORD row")
            name, digest, size = row
            require(name in members and name not in seen, "unknown or duplicate RECORD path")
            seen.add(name)
            if name == DIST + "RECORD":
                require(digest == size == "", "RECORD self-entry must be unhashed")
            else:
                expected = base64.urlsafe_b64encode(bytes.fromhex(members[name]["sha256"])).rstrip(b"=").decode("ascii")
                require(digest == "sha256=" + expected, "RECORD content digest mismatch")
                require(size == str(members[name]["size"]), "RECORD member length mismatch")
        message = BytesParser(policy=default).parsebytes(metadata)
        for key, value in (("Name", "vllm-omni"), ("Version", "0.29.0rc1"), ("License-Expression", "Apache-2.0")):
            require(message.get_all(key) == [value], "wheel identity metadata mismatch")
        requires = message.get_all("Requires-Dist", [])
        require("transformers<5.15,>=5.13.0" in requires, "Transformers boundary changed")
        return {
            "members": members, "metadataSha256": sha256(metadata),
            "requiresPython": message.get("Requires-Python"), "requiresDist": requires,
            "generatedVersionFile": generated, "recordIntegrity": "PASS",
        }, source_blobs


def compare_source(packaged: dict[str, str], tree: bytes) -> dict:
    """Compare wheel package files against Git objects, never caller paths.

    The upstream setup.py explicitly generates _version.py. Record its digest
    separately; do not count it as Git-matched or builder-authenticated evidence.
    """
    tracked = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, digest = header.decode("ascii").split(" ")
        name = raw_name.decode("utf-8")
        require(safe_name(name) and name.startswith("vllm_omni/"), "unexpected Git tree path")
        require(mode in ("100644", "100755") and kind == "blob", "non-regular upstream package object")
        require(re.fullmatch(r"[0-9a-f]{40}", digest) is not None and name not in tracked, "invalid Git object")
        tracked[name] = digest
    require(bool(tracked), "empty upstream package tree")
    compared = {name: digest for name, digest in packaged.items() if name != GENERATED}
    require(bool(compared), "empty packaged source scope")
    changed = sorted(name for name in compared if name in tracked and compared[name] != tracked[name])
    extra = sorted(name for name in compared if name not in tracked)
    missing_python = sorted(name for name in tracked if name.endswith(".py") and name != GENERATED and name not in packaged)
    missing_other = sorted(name for name in tracked if not name.endswith(".py") and name not in packaged)
    return {
        "state": "PASS" if not changed and not extra and not missing_python else "FAIL",
        "scope": "all packaged vllm_omni files except generated _version.py; completeness required for tracked Python only",
        "comparedFiles": len(compared), "changed": changed, "extra": extra,
        "missingPython": missing_python, "unpackagedNonPython": missing_other,
        "generatedFile": GENERATED, "buildAttestationVerified": False,
    }


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=30,
    ).stdout


def run() -> dict:
    upstream = ROOT.parent / "upstream-vllm-omni"
    report = {
        "schema": "szl.forge.vllm-omni-wheel-acquisition.v1", "sourceRevision": None,
        "sourceRepository": "szl-holdings/szl-forge",
        "upstreamRevision": UPSTREAM, "frontierAdmissionMerge": ADMISSION,
        "observedAt": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "state": "FAIL", "disposition": "HOLD",
        "artifactBytesIndependentlyVerified": False, "artifactSourceBound": False,
        "buildAttestationVerified": False, "dependencyClosureCaptured": False,
        "packageInstalled": False, "runtimeExecuted": False, "modelWeightsAcquired": False,
        "productionServingAuthorized": False, "hubPublicationAuthorized": False,
    }
    try:
        source = _git(ROOT, "rev-parse", "HEAD").decode().strip()
        require(re.fullmatch(r"[0-9a-f]{40}", source) is not None, "invalid Forge source")
        report["sourceRevision"] = source
        source_files = {}
        for name in (
            "tools/evaluate_vllm_omni_wheel.py",
            "tests/test_vllm_omni_wheel.py",
            ".github/workflows/vllm-omni-wheel-acquisition.yml",
            "frontier/vllm-omni-029rc1/WHEEL_ACQUISITION.md",
        ):
            admitted = _git(ROOT, "show", source + ":" + name)
            require((ROOT / name).read_bytes() == admitted, "executing source file differs from Git: " + name)
            source_files[name] = sha256(admitted)
        report["sourceFiles"] = source_files
        require(_git(upstream, "rev-parse", "HEAD").decode().strip() == UPSTREAM, "upstream checkout moved")
        report["pypi"] = validate_pypi(download(PYPI_URL, 1024 * 1024))
        wheel = download(WHEEL_URL, WHEEL_SIZE, WHEEL_SIZE)
        require(sha256(wheel) == WHEEL_SHA256, "actual wheel SHA-256 drift")
        report.update(artifactBytesIndependentlyVerified=True, artifactSha256=sha256(wheel), artifactSize=len(wheel))
        inspection, blobs = inspect_wheel(wheel)
        require(inspection["metadataSha256"] == METADATA_SHA256, "wheel metadata SHA-256 drift")
        report["wheel"] = inspection
        comparison = compare_source(blobs, _git(upstream, "ls-tree", "-rz", UPSTREAM, "--", "vllm_omni"))
        report["sourceComparison"] = comparison
        require(comparison["state"] == "PASS", "package source correspondence failed")
        report["state"] = "ACQUISITION_VERIFIED_RUNTIME_NOT_RUN"
    except (AcquisitionError, OSError, ValueError, csv.Error, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="explicitly fetch only the pinned public wheel")
    args = parser.parse_args()
    require(args.acquire, "network acquisition must be explicitly requested")
    result = run()
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    # Exclusive create prevents a stale prior success from being overwritten.
    with (report_dir / "vllm-omni-wheel-acquisition.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({key: result[key] for key in ("state", "sourceRevision", "disposition")}))
    return 0 if result["state"] == "ACQUISITION_VERIFIED_RUNTIME_NOT_RUN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
