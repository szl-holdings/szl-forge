#!/usr/bin/env python3
"""One-shot public file census. Never source execution, publication, or admission.

GitHub tree objects are reconstructed and checked against their referenced Git
object IDs. Blob IDs enumerate content addresses; blob/LFS bytes are NOT read.
HF file trees are provider metadata only. The two populations remain separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

GH_ORG = "szl-holdings"
HF_ORG = "SZLHOLDINGS"
GH = "https://api.github.com"
HF = "https://huggingface.co"
KINDS = ("models", "datasets", "spaces", "kernels")
MAX_BODY = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_REQUESTS = 1500
MAX_REPOS = 256
MAX_PAGES = 100
MAX_TREE_ENTRIES = 250000
MAX_SUBTREES = 512
REQUEST_LAUNCH_SECONDS = 900
SHA = re.compile(r"[0-9a-f]{40}\Z")
NAME = r"[A-Za-z0-9_.-]{1,100}"


class CensusError(RuntimeError):
    """Fixed diagnostic codes only; never retain credentials or remote error text."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CensusError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def reject(_):
        raise CensusError("NONFINITE_JSON")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise CensusError("NONFINITE_JSON")
        return number

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CensusError("INVALID_JSON") from exc


def repo_id(value: Any, org: str) -> str:
    if not isinstance(value, str) or re.fullmatch(re.escape(org) + "/" + NAME, value) is None:
        raise CensusError("REPOSITORY_SCOPE")
    if ".." in value or value.endswith(("/.", "/-")):
        raise CensusError("REPOSITORY_SCOPE")
    return value


def sha(value: Any) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise CensusError("EXACT_SHA_REQUIRED")
    return value


def safe_path(value: Any) -> str:
    if (not isinstance(value, str) or not value or len(value) > 4096
            or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(p in ("", ".", "..") for p in value.split("/"))
            or len(value.split("/")) > 64):
        raise CensusError("UNSUPPORTED_PATH")
    return value


def validate_url(url: str) -> str:
    try:
        part = urlsplit(url)
        if (part.scheme != "https" or part.username or part.password or part.port is not None
                or part.fragment or part.netloc not in ("api.github.com", "huggingface.co")):
            raise CensusError("ENDPOINT_SCOPE")
        if part.netloc == "api.github.com":
            pattern = rf"/repos/{GH_ORG}/{NAME}(?:/commits/[^/]+|/git/trees/[0-9a-f]{{40}})?"
            if part.path != f"/orgs/{GH_ORG}/repos" and re.fullmatch(pattern, part.path) is None:
                raise CensusError("ENDPOINT_SCOPE")
        else:
            pattern = rf"/api/(?:models|datasets|spaces|kernels)(?:/{HF_ORG}/{NAME}(?:/tree/[0-9a-f]{{40}})?)?"
            if re.fullmatch(pattern, part.path) is None:
                raise CensusError("ENDPOINT_SCOPE")
        params = parse_qsl(part.query, keep_blank_values=True, strict_parsing=True)
        if len(params) != len(dict(params)) or any(k not in {
            "type", "per_page", "sort", "direction", "page", "recursive", "expand",
            "author", "limit", "full", "cursor"
        } for k, _ in params):
            raise CensusError("QUERY_SCOPE")
        if part.netloc == "huggingface.co" and part.path.count("/") == 2:
            if dict(params).get("author") != HF_ORG:
                raise CensusError("AUTHOR_SCOPE")
        return part.netloc
    except ValueError as exc:
        raise CensusError("ENDPOINT_SCOPE") from exc


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CensusError("REDIRECT_REFUSED")


class Client:
    """Fixed-origin GET-only reader. GH credential is never sent to Hugging Face."""
    def __init__(self, token: str | None = None):
        self.token = token
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.started = time.monotonic()
        self.calls = 0
        self.total_bytes = 0
        self.responses: list[dict[str, Any]] = []

    def headers(self, url: str) -> dict[str, str]:
        host = validate_url(url)
        headers = {"Accept": "application/json", "Accept-Encoding": "identity",
                   "User-Agent": "szl-public-file-census/1"}
        if host == "api.github.com":
            headers["X-GitHub-Api-Version"] = "2022-11-28"
            if self.token:
                headers["Authorization"] = "Bearer " + self.token
        return headers

    def get(self, url: str) -> tuple[Any, str]:
        headers = self.headers(url)
        if self.calls >= MAX_REQUESTS:
            raise CensusError("REQUEST_BOUND")
        if time.monotonic() - self.started >= REQUEST_LAUNCH_SECONDS:
            raise CensusError("REQUEST_LAUNCH_DEADLINE")
        if self.total_bytes >= MAX_TOTAL_BYTES:
            raise CensusError("TOTAL_BYTE_BOUND")
        self.calls += 1
        receipt = {"url": url, "started_at": now(), "http_status": None, "body_sha256": None}
        try:
            with self.opener.open(Request(url, headers=headers, method="GET"), timeout=12) as response:
                receipt["http_status"] = response.status
                if response.status != 200:
                    raise CensusError("HTTP_NON_200")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise CensusError("COMPRESSED_RESPONSE_REFUSED")
                length = response.headers.get("Content-Length")
                if length is not None and (not str(length).isdigit() or int(length) > MAX_BODY):
                    raise CensusError("BODY_BOUND")
                body = response.read(min(MAX_BODY, MAX_TOTAL_BYTES - self.total_bytes) + 1)
                self.total_bytes += len(body)
                if len(body) > MAX_BODY or self.total_bytes > MAX_TOTAL_BYTES:
                    raise CensusError("BODY_BOUND")
                if length is not None and len(body) != int(length):
                    raise CensusError("BODY_LENGTH_MISMATCH")
                receipt["body_sha256"] = digest_bytes(body)
                receipt["body_bytes"] = len(body)
                link = response.headers.get("Link", "")
                if len(link) > 8192:
                    raise CensusError("LINK_BOUND")
                return strict_json(body), link
        except HTTPError as exc:
            receipt["http_status"] = exc.code
            raise CensusError(f"HTTP_{exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise CensusError("TRANSPORT_UNAVAILABLE") from exc
        finally:
            receipt["finished_at"] = now()
            self.responses.append(receipt)


def next_page(link: str, initial: str) -> str | None:
    if not link:
        return None
    links = {}
    for piece in link.split(","):
        found = re.fullmatch(r'\s*<([^<>]+)>\s*;\s*rel="(next|prev|first|last)"\s*', piece)
        if found is None or found[2] in links:
            raise CensusError("INVALID_PAGINATION_LINK")
        links[found[2]] = found[1]
    target = links.get("next")
    if target is None:
        return None
    validate_url(target)
    base, following = urlsplit(initial), urlsplit(target)
    if (base.scheme, base.netloc, base.path) != (following.scheme, following.netloc, following.path):
        raise CensusError("PAGINATION_SCOPE")
    params = dict(parse_qsl(following.query, keep_blank_values=True))
    original = dict(parse_qsl(base.query, keep_blank_values=True))
    if {k: v for k, v in params.items() if k != "cursor"} != original:
        raise CensusError("PAGINATION_SCOPE")
    return target


def hf_pages(client: Client, initial: str) -> list[dict[str, Any]]:
    result = []
    url: str | None = initial
    seen = set()
    while url is not None:
        if url in seen or len(seen) >= MAX_PAGES:
            raise CensusError("PAGINATION_BOUND_OR_CYCLE")
        seen.add(url)
        page, link = client.get(url)
        if not isinstance(page, list) or any(not isinstance(x, dict) for x in page):
            raise CensusError("LIST_SHAPE")
        result.extend(page)
        if len(result) > MAX_TREE_ENTRIES:
            raise CensusError("ENTRY_BOUND")
        url = next_page(link, initial)
    return result


def github_members(client: Client) -> dict[str, str]:
    members = {}
    for page in range(1, 5):
        query = urlencode(dict(type="public", per_page=100, sort="full_name", direction="asc", page=page))
        items, _ = client.get(f"{GH}/orgs/{GH_ORG}/repos?{query}")
        if not isinstance(items, list):
            raise CensusError("REPOSITORY_LIST_SHAPE")
        for item in items:
            if not isinstance(item, dict) or item.get("private") is not False:
                raise CensusError("PUBLIC_REPOSITORY_REQUIRED")
            name = repo_id(item.get("full_name"), GH_ORG)
            branch = item.get("default_branch")
            if name in members or not isinstance(branch, str) or not branch or len(branch) > 250:
                raise CensusError("DUPLICATE_OR_INVALID_REPOSITORY")
            members[name] = branch
        if len(members) > MAX_REPOS:
            raise CensusError("REPOSITORY_BOUND")
        if len(items) < 100:
            return members
    raise CensusError("REPOSITORY_PAGE_BOUND")


def tree_digest(entries: list[dict[str, Any]]) -> str:
    ordered = sorted(entries, key=lambda e: e["path"].encode("utf-8") +
                     (b"/" if e["type"] == "tree" else b""))
    raw = b"".join(e["mode"].lstrip("0").encode("ascii") + b" " + e["path"].encode("utf-8")
                   + b"\0" + bytes.fromhex(e["sha"]) for e in ordered)
    return hashlib.sha1(b"tree " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def verify_tree(root_sha: str, entries: list[dict[str, Any]], *, recursive: bool = True) -> list[dict[str, Any]]:
    """Reconstruct every tree object, including empty trees, from inert metadata."""
    sha(root_sha)
    if not isinstance(entries, list) or len(entries) > MAX_TREE_ENTRIES:
        raise CensusError("ENTRY_BOUND")
    rows, paths = [], set()
    expected = {"": root_sha}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    modes = {"040000": "tree", "40000": "tree", "100644": "blob", "100755": "blob",
             "120000": "blob", "160000": "commit"}
    for item in entries:
        if not isinstance(item, dict):
            raise CensusError("TREE_ENTRY_SHAPE")
        path = safe_path(item.get("path"))
        if not recursive and "/" in path:
            raise CensusError("NONRECURSIVE_TREE_SHAPE")
        mode, kind = item.get("mode"), item.get("type")
        if not isinstance(mode, str) or modes.get(mode) != kind or path in paths:
            raise CensusError("TREE_MODE_OR_DUPLICATE")
        paths.add(path)
        size = item.get("size")
        if size is not None and (type(size) is not int or size < 0):
            raise CensusError("INVALID_SIZE")
        row = dict(path=path, mode=mode, type=kind, sha=sha(item.get("sha")), size=size)
        rows.append(row)
        parent, _, name = path.rpartition("/")
        children[parent].append(dict(row, path=name))
        if kind == "tree" and recursive:
            expected[path] = row["sha"]
    if not set(children).issubset(expected):
        raise CensusError("MISSING_PARENT_TREE")
    for path, oid in expected.items():
        if tree_digest(children.get(path, [])) != oid:
            raise CensusError("TREE_MERKLE_MISMATCH")
    return sorted(rows, key=lambda e: e["path"])


def github_tree(client: Client, repo: str, root: str) -> tuple[list[dict[str, Any]], bool]:
    prefix = f"{GH}/repos/{repo_id(repo, GH_ORG)}/git/trees/"
    root = sha(root)
    payload, _ = client.get(prefix + root + "?recursive=1")
    if not isinstance(payload, dict) or payload.get("sha") != root or type(payload.get("truncated")) is not bool:
        raise CensusError("TREE_RESPONSE_SHAPE")
    if not payload["truncated"]:
        return verify_tree(root, payload.get("tree")), False
    # Never reuse a partial recursive result. Traverse complete subtrees instead.
    queue = deque([("", root)])
    rows, cache = [], {}
    expanded = 0
    while queue:
        path, oid = queue.popleft()
        expanded += 1
        if expanded > MAX_TREE_ENTRIES or len(path.split("/")) > 64:
            raise CensusError("TREE_EXPANSION_BOUND")
        if oid not in cache:
            if len(cache) >= MAX_SUBTREES:
                raise CensusError("SUBTREE_BOUND")
            node, _ = client.get(prefix + oid)
            if not isinstance(node, dict) or node.get("sha") != oid or node.get("truncated") is not False:
                raise CensusError("TREE_TRUNCATED_OR_WRONG_ID")
            cache[oid] = verify_tree(oid, node.get("tree"), recursive=False)
        for child in cache[oid]:
            if "/" in child["path"]:
                raise CensusError("NONRECURSIVE_TREE_SHAPE")
            new_path = child["path"] if not path else path + "/" + child["path"]
            rows.append(dict(child, path=safe_path(new_path)))
            if len(rows) > MAX_TREE_ENTRIES:
                raise CensusError("ENTRY_BOUND")
            if child["type"] == "tree":
                queue.append((new_path, child["sha"]))
    return verify_tree(root, rows), True


def path_signals(rows: list[dict[str, Any]]) -> dict[str, int]:
    paths = [r["path"].lower() for r in rows]
    return {
        "frontend_paths": sum(p.endswith((".tsx", ".jsx", ".vue", ".svelte", ".html", ".css")) for p in paths),
        "python_paths": sum(p.endswith(".py") for p in paths),
        "test_paths": sum("test" in p.split("/")[:-1] or "tests" in p.split("/")[:-1]
                          or p.rsplit("/", 1)[-1].startswith("test_") for p in paths),
        "workflow_paths": sum(p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml")) for p in paths),
        "binding_named_paths": sum("source-binding" in p or "provenance" in p for p in paths),
    }


def observe_github_repo(client: Client, repo: str, branch: str) -> dict[str, Any]:
    result: dict[str, Any] = {"repository": repo, "default_branch": branch, "started_at": now(),
                             "complete": False, "tree_complete": False, "file_count": None,
                             "content_bytes_verified": False, "runtime_verified": False, "blockers": []}
    public_rechecked = False
    try:
        info, _ = client.get(f"{GH}/repos/{repo}")
        if not isinstance(info, dict) or info.get("private") is not False or info.get("full_name") != repo:
            raise CensusError("PUBLIC_REPOSITORY_IDENTITY")
        if info.get("default_branch") != branch:
            raise CensusError("DEFAULT_BRANCH_MOVED")
        safe_path(branch)
        target = f"{GH}/repos/{repo}/commits/{quote(branch, safe='')}"
        commit, _ = client.get(target)
        if not isinstance(commit, dict) or not isinstance(commit.get("commit"), dict):
            raise CensusError("COMMIT_SHAPE")
        revision = sha(commit.get("sha"))
        tree = commit["commit"].get("tree")
        root = sha(tree.get("sha") if isinstance(tree, dict) else None)
        rows, fallback = github_tree(client, repo, root)
        files = [r for r in rows if r["type"] == "blob"]
        result.update(revision=revision, tree_sha=root, tree_complete=True, tree_merkle_verified=True,
                      nonrecursive_fallback=fallback, entries=rows, file_count=len(files),
                      submodule_count=sum(r["type"] == "commit" for r in rows),
                      symlink_count=sum(r["mode"] == "120000" for r in files), path_signals=path_signals(files))
        after, _ = client.get(target)
        after_sha = sha(after.get("sha") if isinstance(after, dict) else None)
        result["revision_after"] = after_sha
        public_after, _ = client.get(f"{GH}/repos/{repo}")
        if (not isinstance(public_after, dict) or public_after.get("private") is not False
                or public_after.get("full_name") != repo):
            # Do not retain a tree if the final visibility check is unavailable/changed.
            result.pop("entries", None)
            result.update(file_count=None, tree_complete=False, tree_merkle_verified=False)
            raise CensusError("PUBLIC_VISIBILITY_CHANGED")
        public_rechecked = True
        if public_after.get("default_branch") != branch:
            raise CensusError("DEFAULT_BRANCH_MOVED")
        if revision != after_sha:
            raise CensusError("REF_MOVED")
        result["complete"] = True
    except CensusError as exc:
        if not public_rechecked:
            result.pop("entries", None)
            result.pop("path_signals", None)
            result.update(file_count=None, tree_complete=False, tree_merkle_verified=False)
        result["blockers"].append(str(exc))
    finally:
        result["finished_at"] = now()
    return result


def hf_members(client: Client, kind: str) -> list[str]:
    if kind not in KINDS:
        raise CensusError("REPOSITORY_TYPE_SCOPE")
    rows = hf_pages(client, f"{HF}/api/{kind}?author={HF_ORG}&limit=100")
    ids = []
    for row in rows:
        if row.get("private") is True:
            raise CensusError("PUBLIC_SCOPE_VIOLATION")
        ids.append(repo_id(row.get("id"), HF_ORG))
    if len(ids) > MAX_REPOS or len(ids) != len(set(ids)):
        raise CensusError("ASSET_BOUND_OR_DUPLICATE")
    return sorted(ids)


def hf_file_rows(items: list[dict[str, Any]], *, allow_redacted: bool = False) -> list[dict[str, Any]]:
    """Retain masked gated metadata without inventing a content identity.

    An exact 64-asterisk LFS oid was observed in an anonymous gated-model tree.
    It is not a digest, and is recognized only under explicit gated metadata.
    All other malformed identities still fail. This never downloads LFS bytes.
    """
    if not isinstance(items, list) or len(items) > MAX_TREE_ENTRIES:
        raise CensusError("HF_TREE_ENTRY")
    seen, rows = set(), []
    for item in items:
        if not isinstance(item, dict):
            raise CensusError("HF_TREE_ENTRY")
        path = safe_path(item.get("path"))
        kind = item.get("type")
        if path in seen or kind not in ("file", "directory"):
            raise CensusError("HF_TREE_ENTRY")
        seen.add(path)
        oid, size = item.get("oid"), item.get("size")
        if (not isinstance(oid, str) or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", oid) is None
                or (size is not None and (type(size) is not int or size < 0))):
            raise CensusError("HF_FILE_IDENTITY")
        lfs = item.get("lfs")
        lfs_oid, lfs_state = None, "NOT_REPORTED"
        if lfs is not None:
            if not isinstance(lfs, dict) or kind != "file":
                raise CensusError("HF_LFS_IDENTITY")
            declared = lfs.get("oid")
            if isinstance(declared, str) and re.fullmatch(r"[0-9a-f]{64}", declared):
                lfs_oid, lfs_state = declared, "OBSERVED"
            elif declared == "*" * 64 and allow_redacted is True:
                lfs_state = "REDACTED"
            else:
                raise CensusError("HF_LFS_IDENTITY")
            lfs_size = lfs.get("size")
            if (type(lfs_size) is not int or lfs_size < 0
                    or (size is not None and size != lfs_size)):
                raise CensusError("HF_LFS_SIZE")
            pointer_size = lfs.get("pointerSize")
            if "pointerSize" in lfs and (type(pointer_size) is not int or pointer_size <= 0):
                raise CensusError("HF_LFS_POINTER_SIZE")
        rows.append(dict(path=path, type=kind, oid=oid, size=size, lfs_oid=lfs_oid,
                         lfs_identity_state=lfs_state))
    return sorted(rows, key=lambda e: e["path"])


def observe_hf_repo(client: Client, kind: str, repo: str) -> dict[str, Any]:
    result: dict[str, Any] = {"repo_id": repo, "repo_type_endpoint": kind, "started_at": now(),
                             "complete": False, "tree_complete": False, "file_count": None,
                             "file_metadata_identity_complete": False,
                             "content_bytes_verified": False, "runtime_verified": False, "blockers": []}
    public_rechecked = False
    try:
        repo_id(repo, HF_ORG)
        if kind not in KINDS:
            raise CensusError("REPOSITORY_TYPE_SCOPE")
        url = f"{HF}/api/{kind}/{repo}"
        info, _ = client.get(url)
        if not isinstance(info, dict) or info.get("id") != repo or info.get("private") is not False:
            raise CensusError("HF_PUBLIC_IDENTITY")
        revision = sha(info.get("sha"))
        tree_url = f"{url}/tree/{revision}?recursive=true&expand=false&limit=100"
        allow_redacted = info.get("gated") in ("auto", "manual")
        rows = hf_file_rows(hf_pages(client, tree_url), allow_redacted=allow_redacted)
        runtime = info.get("runtime")
        stage = runtime.get("stage") if isinstance(runtime, dict) else None
        if stage is not None and (not isinstance(stage, str) or re.fullmatch(r"[A-Z_]{1,64}", stage) is None):
            stage = None
        files = [r for r in rows if r["type"] == "file"]
        redacted = sum(r["lfs_identity_state"] == "REDACTED" for r in files)
        result.update(revision=revision, tree_complete=True, entries=rows, file_count=len(files),
                      file_metadata_identity_complete=redacted == 0, redacted_lfs_file_count=redacted,
                      runtime_stage_reported=stage, path_signals=path_signals(files),
                      readme_present=any(r["path"] == "README.md" for r in files))
        if redacted:
            result["blockers"].append("HF_LFS_IDENTITY_REDACTED")
        after, _ = client.get(url)
        if not isinstance(after, dict) or after.get("id") != repo or after.get("private") is not False:
            raise CensusError("HF_PUBLIC_IDENTITY")
        public_rechecked = True
        result["revision_after"] = sha(after.get("sha"))
        if revision != result["revision_after"]:
            raise CensusError("REF_MOVED")
        if redacted and after.get("gated") != info.get("gated"):
            raise CensusError("HF_ACCESS_POLICY_MOVED")
        result["complete"] = not result["blockers"]
    except CensusError as exc:
        if not public_rechecked:
            # Never publish paths/counts from an item whose final public identity
            # could not be rechecked. Preserve only fixed-code failure evidence.
            for field in ("entries", "path_signals", "readme_present", "redacted_lfs_file_count",
                          "runtime_stage_reported"):
                result.pop(field, None)
            result.update(file_count=None, tree_complete=False, file_metadata_identity_complete=False)
        result["blockers"].append(str(exc))
    finally:
        result["finished_at"] = now()
    return result


def summarize(items: list[dict[str, Any]], membership_stable: bool) -> dict[str, Any]:
    complete = membership_stable and all(row["complete"] for row in items)
    known = sum(row["file_count"] for row in items if type(row.get("file_count")) is int)
    return dict(complete=complete, membership_stable=membership_stable, items_observed=len(items),
                items_complete=sum(row["complete"] is True for row in items), known_file_subtotal=known,
                complete_scope_file_count=known if complete else None)


def base_report(lane: str, revision: str) -> dict[str, Any]:
    return dict(schema="szl.public-estate-file-census/v1", lane=lane, source_revision=sha(revision),
                observer_sha256=digest_bytes(Path(__file__).read_bytes()), started_at=now(),
                scope="PUBLIC_DEFAULT_REVISION_FILE_METADATA_ONLY", semantic_review_complete=False,
                source_content_files_read=0, runtime_verified=False, production_authorization=False,
                excludes=["private_assets", "collections", "mutable_buckets", "untracked_files", "other_branches",
                          "submodule_contents", "lfs_payload_bytes", "source_semantics", "browser_acceptance"],
                populations={}, blockers=[])


def run(client: Client, lane: str, revision: str) -> dict[str, Any]:
    report = base_report(lane, revision)
    kinds = ("github",) if lane == "github" else KINDS
    for kind in kinds:
        pop: dict[str, Any] = {"items": [], "complete": False, "complete_scope_file_count": None, "blockers": []}
        report["populations"][kind] = pop
        try:
            before = github_members(client) if kind == "github" else hf_members(client, kind)
            for repo in sorted(before):
                item = (observe_github_repo(client, repo, before[repo]) if kind == "github"
                        else observe_hf_repo(client, kind, repo))
                pop["items"].append(item)
            after = github_members(client) if kind == "github" else hf_members(client, kind)
            pop.update(summarize(pop["items"], before == after))
            if before != after:
                pop["blockers"].append("MEMBERSHIP_MOVED")
        except CensusError as exc:
            pop["blockers"].append(str(exc))
    report["complete"] = all(p["complete"] for p in report["populations"].values())
    report["status"] = "FILE_METADATA_OBSERVED_NOT_QUALIFIED" if report["complete"] else "PARTIAL_OR_UNAVAILABLE"
    report.update(finished_at=now(), requests_attempted=client.calls, response_bytes=client.total_bytes,
                  responses=client.responses)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("github", "huggingface"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        revision = sha(args.source_revision)
        root = Path(__file__).resolve().parents[1]
        # Verify actual executing bytes, not only an operator-declared SHA.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              check=True, timeout=10).stdout.decode().strip()
        committed = subprocess.run(["git", "show", f"{revision}:tools/observe_public_estate_files.py"],
                                   cwd=root, capture_output=True, check=True, timeout=10).stdout
        if head != revision or committed != Path(__file__).read_bytes():
            raise CensusError("OBSERVER_SOURCE_MISMATCH")
        # Reserve a new output before any observation; never overwrite evidence.
        with args.output.open("x", encoding="utf-8") as handle:
            client = Client(token=os.environ.get("GITHUB_TOKEN") if args.lane == "github" else None)
            report = run(client, args.lane, revision)
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary = {k: {key: p.get(key) for key in ("complete", "items_observed", "items_complete", "known_file_subtotal", "complete_scope_file_count", "blockers")}
                   for k, p in report["populations"].items()}
        print(json.dumps({"status": report["status"], "populations": summary}, sort_keys=True))
        return 0 if report["complete"] else 2
    except (CensusError, OSError, subprocess.SubprocessError, UnicodeError):
        # No exception text: subprocesses and errors must not reveal ambient secrets.
        print(json.dumps({"status": "PREFLIGHT_OR_OUTPUT_UNAVAILABLE", "production_authorization": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
