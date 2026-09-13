#!/usr/bin/env python3
"""Bounded public Kernel Hub metadata observation. No loaders or Hub writes.

Two paginated list passes bracket per-item ref and immutable-revision reads.
Completeness is for this public API scope, never a whole-account or runtime claim.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ORIGIN = "https://huggingface.co"
LIST_URL = ORIGIN + "/api/kernels?author=SZLHOLDINGS&limit=100"
SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
REPO = re.compile(r"SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
MAX_BODY = 524288
MAX_PAGES = 10
MAX_ITEMS = 100
MAX_REQUESTS = 256
MAX_TOTAL_BYTES = 16 * 1024 * 1024


class ObservationError(ValueError):
    """An incomplete response must not turn into a zero or a missing asset."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def repo_id(value: Any) -> str:
    if (not isinstance(value, str) or not REPO.fullmatch(value)
            or ".." in value or "--" in value or value.endswith((".", ".git"))):
        raise ObservationError("INVALID_OR_OUT_OF_SCOPE_REPOSITORY")
    return value


def revision(value: Any) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ObservationError("INVALID_COMMIT_ID")
    return value


def strict_json(raw: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ObservationError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise ObservationError("NONFINITE_JSON")

    def finite(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ObservationError("NONFINITE_JSON")
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)
    except ObservationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ObservationError("INVALID_JSON") from exc


def validate_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in url):
        raise ObservationError("UNSAFE_URL")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "huggingface.co" or parts.fragment:
        raise ObservationError("WRONG_ORIGIN")
    if parts.path == "/api/kernels":
        pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
        query = dict(pairs)
        if (len(query) != len(pairs) or set(query) - {"author", "limit", "cursor"}
                or query.get("author") != "SZLHOLDINGS" or query.get("limit") != "100"
                or ("cursor" in query and not query["cursor"])):
            raise ObservationError("PAGINATION_SCOPE_CHANGED")
    else:
        path = parts.path.removeprefix("/api/kernels/")
        fields = path.split("/")
        if not parts.path.startswith("/api/kernels/") or parts.query or len(fields) not in (3, 4):
            raise ObservationError("UNAPPROVED_ENDPOINT")
        repo_id("/".join(fields[:2]))
        if fields[2:] != ["refs"]:
            if len(fields) != 4 or fields[2] != "revision":
                raise ObservationError("UNAPPROVED_ENDPOINT")
            revision(fields[3])
    return url


def next_link(header: str) -> str | None:
    if not header:
        return None
    if len(header) > 8192:
        raise ObservationError("OVERSIZED_LINK_HEADER")
    found: list[str] = []
    for part in header.split(","):
        match = re.fullmatch(r'\s*<([^<>]+)>\s*;\s*rel="([a-z ]+)"\s*', part)
        if not match:
            raise ObservationError("MALFORMED_LINK_HEADER")
        if "next" in match[2].split():
            target = validate_url(match[1])
            if urlsplit(target).path != "/api/kernels":
                raise ObservationError("PAGINATION_ENDPOINT_CHANGED")
            found.append(target)
    if len(found) > 1:
        raise ObservationError("DUPLICATE_NEXT_LINK")
    return found[0] if found else None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ObservationError("REDIRECT_FORBIDDEN")


class PublicReader:
    """Credentialless GET transport with fixed origin and bounded captured bytes."""
    def __init__(self, *, transport: Callable | None = None):
        self.records: list[dict[str, Any]] = []
        self.total_bytes = 0
        self.deadline = time.monotonic() + 180
        self.transport = transport or self._get
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def _get(self, url: str) -> tuple[int, dict[str, str], bytes]:
        req = Request(url, headers={"Accept": "application/json", "Accept-Encoding": "identity",
                                    "User-Agent": "szl-kernel-metadata-observer/1"}, method="GET")
        try:
            response = self.opener.open(req, timeout=min(12, max(0.1, self.deadline - time.monotonic())))
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read(MAX_BODY + 1)
            return response.code, {k.lower(): v for k, v in response.headers.items()}, raw

    def read(self, url: str) -> tuple[Any, str]:
        validate_url(url)
        if len(self.records) >= MAX_REQUESTS or time.monotonic() >= self.deadline:
            raise ObservationError("REQUEST_BUDGET_EXHAUSTED")
        record: dict[str, Any] = {"url": url, "started_at": utc(), "http_status": None}
        self.records.append(record)
        try:
            status, headers, raw = self.transport(url)
            record["http_status"] = status
            self.total_bytes += len(raw)
            if len(raw) > MAX_BODY or self.total_bytes > MAX_TOTAL_BYTES:
                raise ObservationError("BODY_BUDGET_EXHAUSTED")
            record.update(body_sha256=digest(raw), body_base64=base64.b64encode(raw).decode("ascii"))
            if status != 200:
                raise ObservationError("HTTP_UNAVAILABLE_" + str(status))
            if "json" not in headers.get("content-type", "").lower():
                raise ObservationError("UNEXPECTED_CONTENT_TYPE")
            link = headers.get("link", "")
            if len(link) > 8192:
                raise ObservationError("OVERSIZED_LINK_HEADER")
            record["link"] = link
            return strict_json(raw), link
        except (OSError, URLError, TimeoutError) as exc:
            record["error"] = "TRANSPORT_UNAVAILABLE"
            raise ObservationError("TRANSPORT_UNAVAILABLE") from exc
        except ObservationError as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["completed_at"] = utc()


def listing(reader: PublicReader) -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    url: str | None = LIST_URL
    error = None
    try:
        while url:
            if url in seen or len(seen) >= MAX_PAGES:
                raise ObservationError("PAGINATION_LOOP_OR_LIMIT")
            seen.add(url)
            data, link = reader.read(url)
            if not isinstance(data, list):
                raise ObservationError("EXPECTED_LIST")
            for value in data:
                if not isinstance(value, dict):
                    raise ObservationError("INVALID_LIST_ITEM")
                identity = repo_id(value.get("id"))
                if identity in items or len(items) >= MAX_ITEMS:
                    raise ObservationError("DUPLICATE_ITEM_OR_ITEM_LIMIT")
                if value.get("repoType", "kernel") != "kernel":
                    raise ObservationError("CONTRADICTORY_REPOSITORY_TYPE")
                tags = value.get("tags")
                if tags is not None and (not isinstance(tags, list) or any(not isinstance(t, str) for t in tags)):
                    raise ObservationError("INVALID_TAGS")
                items[identity] = {"repo_id": identity, "listed_revision": None if value.get("sha") is None else revision(value["sha"]),
                                   "tagged_kernel": None if tags is None else "kernel" in tags}
            url = next_link(link)
    except (ObservationError, ValueError) as exc:
        error = str(exc) if isinstance(exc, ObservationError) else "INVALID_PAGINATION"
    return {"complete": error is None, "error": error, "pages": len(seen),
            "items": [items[k] for k in sorted(items)]}


def refs(reader: PublicReader, identity: str) -> dict[str, str]:
    data, link = reader.read(ORIGIN + "/api/kernels/" + repo_id(identity) + "/refs")
    if link or not isinstance(data, dict) or not isinstance(data.get("branches"), list):
        raise ObservationError("INCOMPLETE_OR_INVALID_REFS")
    result: dict[str, str] = {}
    for branch in data["branches"]:
        if not isinstance(branch, dict) or not isinstance(branch.get("name"), str):
            raise ObservationError("INVALID_BRANCH")
        name = branch["name"]
        if name in result:
            raise ObservationError("DUPLICATE_BRANCH")
        result[name] = revision(branch.get("targetCommit"))
    if "main" not in result:
        raise ObservationError("MAIN_NOT_OBSERVED")
    return result


def item_observation(reader: PublicReader, listed: dict[str, Any]) -> dict[str, Any]:
    identity = listed["repo_id"]
    item: dict[str, Any] = dict(listed, complete=False, refs=None, v1_present=None,
                                revision_metadata_verified=[], runtime_loaded=False)
    try:
        before = refs(reader, identity)
        item["refs"] = before
        item["v1_present"] = "v1" in before
        if listed["listed_revision"] is not None and before["main"] != listed["listed_revision"]:
            raise ObservationError("LIST_TO_REFS_DRIFT")
        for sha in sorted(set(before[name] for name in ("main", "v1") if name in before)):
            data, link = reader.read(ORIGIN + "/api/kernels/" + identity + "/revision/" + sha)
            if (link or not isinstance(data, dict) or data.get("id") != identity
                    or data.get("sha") != sha or data.get("repoType", "kernel") != "kernel"):
                raise ObservationError("IMMUTABLE_METADATA_MISMATCH")
            item["revision_metadata_verified"].append(sha)
        if refs(reader, identity) != before:
            raise ObservationError("REFS_MOVED_DURING_OBSERVATION")
        item["complete"] = True
    except ObservationError as exc:
        item["error"] = str(exc)
    return item


def collect(reader: PublicReader, *, source_revision: str) -> dict[str, Any]:
    revision(source_revision)
    started = utc()
    first = listing(reader)
    items = [item_observation(reader, item) for item in first["items"]]
    second = listing(reader)
    stable = first["complete"] and second["complete"] and first["items"] == second["items"]
    complete = bool(stable and all(item["complete"] for item in items))
    known_tags = all(item["tagged_kernel"] is not None for item in items)
    counts = None
    if complete:
        tagged = sum(item["tagged_kernel"] is True for item in items) if known_tags else None
        counts = {"kernel_api_items": len(items), "tagged_kernel": tagged,
                  "untagged_kernel": None if tagged is None else len(items) - tagged,
                  "v1_present": sum(item["v1_present"] is True for item in items),
                  "v1_missing_in_observed_refs": sum(item["v1_present"] is False for item in items)}
    return {"schema": "szl.kernel-fleet-observation/v1", "source_revision": source_revision,
            "observer_sha256": digest(Path(__file__).read_bytes()),
            "scope": {"origin": ORIGIN, "endpoint": "/api/kernels", "author": "SZLHOLDINGS",
                      "authentication": "NONE", "type_basis": "FIRST_CLASS_KERNEL_API_ENDPOINT",
                      "excludes": ["private_assets", "model_mirrors", "runtime_qualification"]},
            "started_at": started, "completed_at": utc(), "observation_complete": complete,
            "atomic_snapshot": False, "list_stable_across_two_passes": bool(stable),
            "status": "OBSERVED_METADATA_ONLY" if complete else "HOLD",
            "fleet_qualification": "HOLD", "counts": counts, "items": items,
            "list_passes": [first, second], "responses": reader.records,
            "runtime_loaded": False, "production_authorization": False,
            "limitations": ["Sequential observations are not an atomic Hub snapshot.",
                            "Metadata is not artifact-byte, loader, numerical, or GPU evidence.",
                            "Public anonymous API scope is not total organization membership."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        revision(args.source_revision)
        # Reserve a new operator-selected file before any requests; never overwrite evidence.
        with args.output.open("x", encoding="utf-8") as handle:
            report = collect(PublicReader(), source_revision=args.source_revision)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ObservationError, OSError) as exc:
        parser.error(type(exc).__name__)
    summary = {k: report[k] for k in ("schema", "source_revision", "status", "observation_complete", "counts", "fleet_qualification")}
    summary["items"] = [{k: v for k, v in item.items() if k != "revision_metadata_verified"} for item in report["items"]]
    summary["list_errors"] = [p["error"] for p in report["list_passes"]]
    print("SZL_KERNEL_FLEET_OBSERVATION=" + json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if report["observation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
