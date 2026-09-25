#!/usr/bin/env python3
"""Read-only exact-byte comparison for the fixed invariants source and Hub pair.

No tokens, publication, source imports, retries, HEAD discovery or arbitrary URLs.
Byte equality is not release authorization, current branch state or model fitness.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SOURCE_REPO = "szl-holdings/szl-invariants"
HUB_REPO = "SZLHOLDINGS/szl-invariants"
CONTRACT = "publishing/invariants-source-binding.json"
MAX_FILE_BYTES = 256 * 1024
MAX_REDIRECTS = 3
TIMEOUT_SECONDS = 15
SOURCE_FILES = frozenset(
    f"{root}/szl_invariants/{name}"
    for root in ("build/torch-universal", "torch-ext")
    for name in ("__init__.py", "metadata.json")
)
TARGETS = frozenset(
    [("model", f"build/torch-universal/szl_invariants/{name}",
       f"build/torch-universal/szl_invariants/{name}")
     for name in ("__init__.py", "metadata.json")]
    + [("kernel", f"{root}/szl_invariants/{name}",
        f"build/{variant}/szl_invariants/{name}")
       for root, variant in (("build/torch-universal", "torch-universal"),
                             ("torch-ext", "torch-cpu"))
       for name in ("__init__.py", "metadata.json")]
)
HUB_FILES = {
    "model": frozenset(dst for typ, _, dst in TARGETS if typ == "model"),
    "kernel": frozenset(dst for typ, _, dst in TARGETS if typ == "kernel")
    | frozenset(f"build/{v}/metadata.json" for v in ("torch-cpu", "torch-universal")),
}


class ObservationError(ValueError):
    """Fixed codes only; transport/provider exception contents are not exported."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise ObservationError(code)


def is_sha(value: Any, length: int = 40) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        need(key not in out, "DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def reject_constant(_value: str) -> None:
    raise ObservationError("NONFINITE_JSON_VALUE")


def finite_float(raw: str) -> float:
    value = float(raw)
    need(math.isfinite(value), "NONFINITE_JSON_VALUE")
    return value


def strict_object(raw: bytes) -> dict[str, Any]:
    need(type(raw) is bytes and len(raw) <= MAX_FILE_BYTES, "INVALID_SOURCE_CONTRACT")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                         parse_constant=reject_constant, parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, ObservationError):
            raise
        raise ObservationError("INVALID_SOURCE_CONTRACT") from None
    need(type(obj) is dict, "INVALID_SOURCE_CONTRACT")
    return obj


def validate_contract(raw: bytes) -> dict[str, Any]:
    obj = strict_object(raw)
    need(obj.get("schema") == "szl.invariants-source-binding/v1"
         and obj.get("repo_id") == HUB_REPO
         and obj.get("source_repository") == SOURCE_REPO, "SOURCE_CONTRACT_IDENTITY_MISMATCH")
    paths = obj.get("artifact_files")
    need(type(paths) is list and len(paths) == len(SOURCE_FILES)
         and all(type(p) is str for p in paths) and set(paths) == SOURCE_FILES,
         "SOURCE_FILE_SET_MISMATCH")
    hashes = obj.get("expected_artifact_sha256")
    need(type(hashes) is dict and set(hashes) == SOURCE_FILES
         and all(is_sha(h, 64) for h in hashes.values()), "SOURCE_HASH_SET_MISMATCH")
    targets = obj.get("publication_targets")
    need(type(targets) is list and len(targets) == len(TARGETS), "TARGET_SET_MISMATCH")
    rows = []
    for row in targets:
        need(type(row) is dict and set(row) == {"repo_type", "source_path", "path_in_repo"}
             and all(type(v) is str for v in row.values()), "TARGET_SET_MISMATCH")
        rows.append((row["repo_type"], row["source_path"], row["path_in_repo"]))
    need(len(set(rows)) == len(rows) and set(rows) == TARGETS, "TARGET_SET_MISMATCH")
    claims = obj.get("claims")
    need(type(claims) is dict and claims.get("trained_weights_present") is False,
         "UNSUPPORTED_SOURCE_CLAIMS")
    return obj


@dataclass(frozen=True)
class Resource:
    kind: str
    revision: str
    path: str

    def __post_init__(self) -> None:
        need(is_sha(self.revision), "REVISION_NOT_IMMUTABLE")
        allowed = SOURCE_FILES | {CONTRACT} if self.kind == "source" else HUB_FILES.get(self.kind, ())
        need(self.path in allowed, "UNDECLARED_RESOURCE")

    @property
    def url(self) -> str:
        if self.kind == "source":
            return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{self.revision}/{self.path}"
        prefix = "kernels/" if self.kind == "kernel" else ""
        return f"https://huggingface.co/{prefix}{HUB_REPO}/resolve/{self.revision}/{self.path}"

    def allows(self, url: str) -> bool:
        """Only this resource and immutable revision, including HF cache redirects."""
        if (len(url) > 8192 or any(ord(c) <= 32 or ord(c) == 127 for c in url)
                or "\\" in url):
            return False
        try:
            parts = urllib.parse.urlsplit(url)
            if (parts.scheme != "https" or parts.username is not None
                    or parts.password is not None or parts.port not in (None, 443)
                    or parts.fragment):
                return False
        except ValueError:
            return False
        canonical = urllib.parse.urlsplit(self.url)
        if parts.hostname != canonical.hostname:
            return False
        allowed_paths = {canonical.path}
        if self.kind != "source":
            plural = "kernels" if self.kind == "kernel" else "models"
            allowed_paths.add(f"/api/resolve-cache/{plural}/{HUB_REPO}/{self.revision}/{self.path}")
        # HF cache hints may be in the query; they cannot change the bound path.
        return parts.path in allowed_paths and (self.kind != "source" or not parts.query)


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def public_read(resource: Resource) -> bytes:
    """Anonymous TLS GET only, bounded, without ambient proxy or token fallback."""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        RejectRedirects(),
    )
    url = resource.url
    visited: set[str] = set()
    for _ in range(MAX_REDIRECTS + 1):
        need(resource.allows(url), "UNSAFE_REDIRECT")
        need(url not in visited, "REDIRECT_CYCLE")
        visited.add(url)
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "szl-invariants-byte-observer/1", "Accept-Encoding": "identity",
        })
        try:
            response = opener.open(request, timeout=TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            status = exc.code
            location = exc.headers.get("Location") if exc.headers else None
            exc.close()
            if status in (301, 302, 303, 307, 308):
                need(type(location) is str and bool(location), "REDIRECT_WITHOUT_LOCATION")
                url = urllib.parse.urljoin(url, location)
                continue
            # Anonymous 404 does NOT establish that a private resource is absent.
            codes = {401: "HTTP_401_UNAVAILABLE", 403: "HTTP_403_UNAVAILABLE",
                     404: "HTTP_404_UNAVAILABLE", 429: "HTTP_429_UNAVAILABLE"}
            raise ObservationError(codes.get(status, "HTTP_UNAVAILABLE")) from None
        except Exception:
            raise ObservationError("TRANSPORT_UNAVAILABLE") from None
        with response:
            need(resource.allows(response.geturl()), "UNSAFE_RESPONSE_LOCATION")
            need(response.status == 200, "HTTP_UNAVAILABLE")
            need(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                 "UNSUPPORTED_CONTENT_ENCODING")
            raw = response.read(MAX_FILE_BYTES + 1)
        need(type(raw) is bytes and 0 < len(raw) <= MAX_FILE_BYTES, "FILE_SIZE_INVALID")
        return raw
    raise ObservationError("REDIRECT_LIMIT")


def expected_hub_files(source: Mapping[str, bytes]) -> dict[tuple[str, str], bytes]:
    """Reconstruct the closed staging metadata without importing the publisher."""
    expected = {(kind, dest): source[src] for kind, src, dest in TARGETS}
    for variant in ("torch-cpu", "torch-universal"):
        prefix = f"build/{variant}/"
        files = {path[len(prefix):]: raw for (kind, path), raw in expected.items()
                 if kind == "kernel" and path.startswith(prefix)}
        tree = hashlib.sha256()
        for relative, raw in sorted(files.items()):
            tree.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(raw).digest())
        metadata = {
            "name": "szl-invariants",
            "id": f"_szl_invariants_{variant.replace('-', '_')}_{tree.hexdigest()[:8]}",
            "version": 1, "license": "Apache-2.0", "python-depends": [],
            "backend": {"type": "cpu"},
            "digest": {"algorithm": "sha256", "files": {
                rel: base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
                for rel, raw in sorted(files.items())}},
        }
        expected[("kernel", f"{prefix}metadata.json")] = (
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    return expected


def atomic_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def observe(*, source_revision: str, model_revision: str, kernel_revision: str,
            report_path: Path, enabled: bool = False,
            reader: Callable[[Resource], bytes] = public_read) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "szl.invariants-public-byte-observation/v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None, "status": "NOT_OBSERVED", "reason": None,
        "scope": "FIXED_PUBLIC_FILES_AT_SUPPLIED_IMMUTABLE_REVISIONS",
        "observation_complete": False, "all_bytes_equal": None,
        "independent_provider_byte_comparison": False,
        "publication_verified": False, "signature_independently_verified": False,
        "write_authorization_verified": False, "branch_membership_verified": False,
        "freshness_verified": False, "repository_mutation": "NOT_ATTEMPTED",
        "source_files": [], "comparisons": [],
    }
    # A fresh non-success marker must reach disk before any network access.
    atomic_report(report_path, report)
    try:
        need(all(is_sha(v) for v in (source_revision, model_revision, kernel_revision)),
             "REVISION_NOT_IMMUTABLE")
        report["pins"] = {"source": source_revision, "model": model_revision, "kernel": kernel_revision}
        if enabled:
            report["status"] = "OBSERVING"
            atomic_report(report_path, report)
            raw_contract = reader(Resource("source", source_revision, CONTRACT))
            contract = validate_contract(raw_contract)
            report["source_contract_sha256"] = digest(raw_contract)
            source: dict[str, bytes] = {}
            for path in sorted(SOURCE_FILES):
                raw = reader(Resource("source", source_revision, path))
                need(type(raw) is bytes and 0 < len(raw) <= MAX_FILE_BYTES, "FILE_SIZE_INVALID")
                row = {"path": path, "bytes": len(raw), "sha256": digest(raw)}
                report["source_files"].append(row)
                need(row["sha256"] == contract["expected_artifact_sha256"][path],
                     "SOURCE_BYTES_DO_NOT_MATCH_CONTRACT")
                source[path] = raw
            for name in ("__init__.py", "metadata.json"):
                need(source[f"build/torch-universal/szl_invariants/{name}"]
                     == source[f"torch-ext/szl_invariants/{name}"], "SOURCE_VARIANT_MISMATCH")
            atomic_report(report_path, report)
            expected = expected_hub_files(source)
            for (kind, path), wanted in sorted(expected.items()):
                revision = model_revision if kind == "model" else kernel_revision
                row = {"repo_type": kind, "path": path, "revision": revision,
                       "expected_sha256": digest(wanted), "expected_bytes": len(wanted),
                       "observed_sha256": None, "observed_bytes": None,
                       "equal": None, "status": "UNAVAILABLE", "reason": None}
                try:
                    raw = reader(Resource(kind, revision, path))
                    need(type(raw) is bytes and 0 < len(raw) <= MAX_FILE_BYTES, "FILE_SIZE_INVALID")
                    row.update(observed_sha256=digest(raw), observed_bytes=len(raw),
                               equal=(raw == wanted), status="MATCH" if raw == wanted else "DRIFT")
                except ObservationError as exc:
                    # Only errors from the fixed-code local reader are used in production.
                    row["reason"] = safe_reason(exc)
                except Exception:
                    row["reason"] = "OBSERVATION_UNAVAILABLE"
                report["comparisons"].append(row)
                atomic_report(report_path, report)
            complete = all(row["equal"] is not None for row in report["comparisons"])
            equal = all(row["equal"] is True for row in report["comparisons"]) if complete else None
            report.update(observation_complete=complete, all_bytes_equal=equal,
                          independent_provider_byte_comparison=complete,
                          status=("BYTE_ALIGNMENT_VERIFIED_REVIEW_REQUIRED" if equal is True else
                                  "BYTE_DRIFT_OBSERVED" if complete else "OBSERVATION_INCOMPLETE"))
    except ObservationError as exc:
        report.update(status="OBSERVATION_REJECTED", reason=safe_reason(exc))
    except OSError:
        report.update(status="OBSERVATION_INCOMPLETE", reason="LOCAL_IO_FAILED")
    except Exception:
        report.update(status="OBSERVATION_INCOMPLETE", reason="OBSERVATION_UNAVAILABLE")
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_report(report_path, report)
    return report


ERROR_CODES = frozenset({
    "REVISION_NOT_IMMUTABLE", "UNDECLARED_RESOURCE", "UNSAFE_REDIRECT", "REDIRECT_CYCLE",
    "REDIRECT_WITHOUT_LOCATION", "UNSAFE_RESPONSE_LOCATION", "REDIRECT_LIMIT",
    "HTTP_401_UNAVAILABLE", "HTTP_403_UNAVAILABLE", "HTTP_404_UNAVAILABLE",
    "HTTP_429_UNAVAILABLE", "HTTP_UNAVAILABLE", "TRANSPORT_UNAVAILABLE",
    "UNSUPPORTED_CONTENT_ENCODING", "FILE_SIZE_INVALID", "INVALID_SOURCE_CONTRACT",
    "DUPLICATE_JSON_KEY", "NONFINITE_JSON_VALUE", "SOURCE_CONTRACT_IDENTITY_MISMATCH",
    "SOURCE_FILE_SET_MISMATCH", "SOURCE_HASH_SET_MISMATCH", "TARGET_SET_MISMATCH",
    "UNSUPPORTED_SOURCE_CLAIMS", "SOURCE_BYTES_DO_NOT_MATCH_CONTRACT", "SOURCE_VARIANT_MISMATCH",
})


def safe_reason(exc: ObservationError) -> str:
    text = str(exc)
    return text if text in ERROR_CODES else "OBSERVATION_UNAVAILABLE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--kernel-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--observe", action="store_true", help="Opt into anonymous fixed-target GETs")
    args = parser.parse_args(argv)
    try:
        result = observe(source_revision=args.source_revision, model_revision=args.model_revision,
                         kernel_revision=args.kernel_revision, report_path=args.report,
                         enabled=args.observe)
    except Exception:
        print(json.dumps({"status": "REPORT_WRITE_FAILED", "publication_verified": False}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["all_bytes_equal"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
