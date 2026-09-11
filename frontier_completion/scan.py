"""Public metadata-only acquisition and conservative material-delta detection.

No weights, datasets, archives, upstream Python, or models are downloaded by this
module. An observed upstream SHA is not an admitted production recipe. Materiality
is about changed evidence, not popularity or an edited collection timestamp.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import PurePosixPath
from typing import Any

from .core import MAX_JSON, EvidenceError, canonical, digest, observation_shell, parse_json, receipt, require, sha256, text, verify_receipt

ALLOWED_HOSTS = {"huggingface.co", "api.github.com", "a-11-oy.com", "a11oy.net", "szlholdings-a11oy.hf.space"}
REPO_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}")


@dataclasses.dataclass(frozen=True)
class Candidate:
    key: str
    repo: str
    kind: str
    lane: str
    disposition: str = "EVALUATE"

    def __post_init__(self) -> None:
        require(type(self.key) is str and bool(re.fullmatch(r"[a-z][a-z0-9-]{1,79}", self.key)), "invalid_candidate_key")
        require(isinstance(self.repo, str) and bool(REPO_PATTERN.fullmatch(self.repo)), "invalid_repository")
        require(".." not in self.repo and "--" not in self.repo and not self.repo.endswith(".git"), "invalid_repository")
        require(self.kind in {"model", "dataset"}, "unsupported_repository_kind")
        text(self.lane, 80)
        require(self.disposition in {"WATCH", "EVALUATE", "HOLD", "REJECT"}, "invalid_disposition")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EvidenceError("redirect_not_admitted")


class PublicClient:
    """Unauthenticated fixed-origin GET, no ambient token and finite retry budget."""
    def __init__(self, timeout: float = 12.0):
        require(type(timeout) in (int, float) and 0 < timeout <= 30, "invalid_timeout")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, url: str) -> tuple[bytes, str]:
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS,
                "origin_not_admitted")
        require(not parsed.username and not parsed.password and parsed.port in (None, 443)
                and not parsed.fragment, "invalid_url")
        require(len(url) <= 2048 and not any(ord(c) < 32 for c in url), "url_bound")
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "SZL-Handoff-Metadata/1", "Accept": "application/json",
            "Cache-Control": "no-cache"})
        for attempt in range(2):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    require(response.status == 200 and response.geturl() == url, "unexpected_response")
                    raw = response.read(MAX_JSON + 1)
                    require(len(raw) <= MAX_JSON, "response_size_bound")
                    return raw, response.headers.get_content_type()
            except urllib.error.HTTPError as exc:
                code = exc.code
                retry = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                # Auth/access and unknown retry dates are never bypassed.
                if code not in {429, 500, 502, 503, 504} or attempt == 1:
                    raise EvidenceError("http_" + str(code)) from exc
                if retry is not None and (not re.fullmatch(r"\d{1,2}", retry) or int(retry) > 10):
                    raise EvidenceError("retry_deferred") from exc
                time.sleep(max(1, int(retry or "1")))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 1:
                    raise EvidenceError("public_transport_unavailable") from exc
                time.sleep(1)
        raise EvidenceError("retry_budget_exhausted")

    def json(self, url: str) -> dict[str, Any]:
        raw, content_type = self.get(url)
        require(content_type == "application/json" or content_type.endswith("+json"), "json_content_type_required")
        return parse_json(raw)


def _identity(raw: dict[str, Any], candidate: Candidate) -> None:
    identities = [raw[key] for key in ("id", "modelId") if key in raw]
    require(bool(identities) and all(identity == candidate.repo for identity in identities), "upstream_identity_mismatch")
    digest(raw.get("sha"), 40)


def _file_facts(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, bool]:
    """Interpret declared file identities only; never call them locally hashed bytes."""
    siblings = raw.get("siblings")
    require(type(siblings) is list and len(siblings) <= 8192, "file_inventory_missing_or_large")
    results, complete, has_code, seen = [], True, False, set()
    for entry in siblings:
        require(type(entry) is dict, "invalid_file_entry")
        name = text(entry.get("rfilename"), 512)
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
                and str(path) == name and name not in seen, "invalid_or_duplicate_file_path")
        seen.add(name)
        lower = name.lower()
        has_code |= lower.endswith((".py", ".so", ".dll", ".dylib", ".sh"))
        license_document = any(part.startswith(("license", "licence", "notice", "terms")) for part in (path.name.lower(),))
        if lower.endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")) and not license_document:
            continue
        lfs = entry.get("lfs")
        identity = entry.get("blobId")
        size = entry.get("size")
        kind = "git_blob_sha1"
        if lfs is not None:
            require(type(lfs) is dict, "invalid_lfs_metadata")
            values = [lfs[key] for key in ("sha256", "oid") if key in lfs]
            for value in values:
                digest(value, 64)
            require(len(set(values)) <= 1, "lfs_identity_conflict")
            identity = values[0] if values else None
            size = lfs.get("size", size)
            kind = "lfs_sha256"
        if identity is None or size is None:
            complete = False
        else:
            digest(identity, 64 if kind == "lfs_sha256" else 40)
            require(type(size) is int and 0 <= size <= 10**15, "invalid_declared_file_size")
        results.append({"path": name, "declaredIdentity": identity, "identityKind": kind, "declaredBytes": size})
    return sorted(results, key=lambda row: row["path"]), complete and bool(results), has_code
