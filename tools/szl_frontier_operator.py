#!/usr/bin/env python3
"""Bounded GitHub/Hugging Face evidence operator; all commands default to read-only.

Python 3.11+, stdlib only, authenticated gh CLI. This independently verifies
published evidence; integrity is not a signature or model qualification.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from typing import Any

PLATFORM = "szl-holdings/platform"
FORGE = "szl-holdings/szl-forge"
DATASET = "SZLHOLDINGS/szl-frontier-evaluation-receipts"
WORKFLOW = ".github/workflows/frontier-evaluation-runner.yml"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PAGE_LIMIT = 100
OUTPUT_LIMIT = 8_000_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_BOUNDS = {"provider_execution_weight_revision_not_attested_by_chat_response",
                   "provider_server_hardware_not_disclosed", "provider_runtime_build_not_disclosed",
                   "integrity_receipt_is_unsigned"}
SUPPORTED_CORE_SHA256 = "a9267e738eb47a0cf61c55ac054482d5e0858becec32715eb7b4384ccb602890"
SCORE_FLAGS = ("json_valid", "schema_valid", "decision_match", "answer_match", "evidence_match", "tool_boundary_pass")


class OperatorError(RuntimeError):
    """Observed failed evidence or a refused operation."""


class Unavailable(OperatorError):
    """An external read or resource is unavailable."""


class UnknownAfterAttempt(Unavailable):
    """A mutation was issued, but its outcome could not be verified."""

    def __init__(self, evidence: dict, reason: str):
        super().__init__("UNKNOWN_AFTER_ATTEMPT: " + reason + "; read PR before retry")
        self.evidence = evidence


def need(condition: Any, message: str) -> None:
    if not condition:
        raise OperatorError(message)


def sanitize(value: str) -> str:
    # Environment values never leave the process. Longest first handles prefixes.
    secrets = sorted({v for k, v in os.environ.items() if len(v) >= 8 and
                      re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL", k, re.I)},
                     key=len, reverse=True)
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    patterns = [r"(?i)\b(?:Bearer|Basic)\s+[^\s\"'<>]+",
                r"\bhf_[A-Za-z0-9]{20,}\b",
                r"\b(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9._-]{8,}",
                r"(?i)(?:authorization|password|api[_-]?key|access[_-]?token|secret)"
                r"\s*[=:]\s*[\"']?[^\s,\"'}]+",
                r"https?://[^/\s:@]+:[^/\s@]+@",
                r"(?i)([?&](?:token|sig|signature|key|credential)=)[^&\s]+"]
    for pattern in patterns:
        value = re.sub(pattern, "[REDACTED]", value)
    return value


def canonical(value: Any) -> bytes:
    # Forge's exact JSON encoding for finite JSON, rejecting non-JSON floats.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for key, val in items:
            need(key not in result, f"duplicate JSON key: {key}")
            result[key] = val
        return result
    def invalid(value):
        raise OperatorError(f"non-finite JSON constant: {value}")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        raise OperatorError("invalid JSON response") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Redact persisted strings, including unexpected provider error payloads.
    path.write_text(sanitize(json.dumps(value, indent=2, ensure_ascii=False,
                                        sort_keys=True, allow_nan=False)) + "\n",
                    encoding="utf-8")


def run(command: list[str], *, timeout: float = 60,
        max_bytes: int = OUTPUT_LIMIT, check: bool = True,
        input_text: str | None = None, binary: bool = False) -> subprocess.CompletedProcess:
    """Bound output in memory while reading both pipes; no raw log temp files."""
    env = os.environ.copy()
    for key in ("GH_DEBUG", "GIT_TRACE", "GIT_CURL_VERBOSE", "GIT_TRACE_CURL"):
        env.pop(key, None)
    env.update(GH_PROMPT_DISABLED="1", GH_PAGER="", PAGER="")
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=False, env=env)
    except OSError as exc:
        raise Unavailable(f"command executable unavailable: {command[0]}") from exc
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()
    def read_pipe(pipe, output):
        while chunk := pipe.read(8192):
            available = max_bytes - len(output)
            output.extend(chunk[:max(0, available)])
            if len(chunk) > available:
                overflow.set()
                break
        pipe.close()
    readers = [threading.Thread(target=read_pipe, args=(pipe, out), daemon=True)
               for pipe, out in zip((proc.stdout, proc.stderr), buffers)]
    for thread in readers:
        thread.start()
    try:
        if input_text:
            proc.stdin.write(input_text.encode("utf-8"))
        proc.stdin.close()
        end = time.monotonic() + timeout
        while proc.poll() is None:
            if overflow.is_set() or time.monotonic() >= end:
                proc.kill()
                proc.wait(timeout=5)
                raise Unavailable("command exceeded output or time bound")
            time.sleep(0.02)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for thread in readers:
            thread.join(timeout=5)
    if overflow.is_set() or any(t.is_alive() for t in readers):
        raise Unavailable("command output incomplete or exceeded bound")
    result = subprocess.CompletedProcess(command, proc.returncode,
                bytes(buffers[0]) if binary else buffers[0].decode("utf-8", "replace"),
                buffers[1].decode("utf-8", "replace"))
    if check and result.returncode:
        raise Unavailable(f"command failed (exit {result.returncode}): " +
                          sanitize(result.stderr)[-2000:])
    return result


def repo_path(repo: str) -> str:
    need(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo),
         "repository must be owner/name")
    return f"/repos/{repo}"


def gh_api(path: str, *, missing_ok: bool = False,
           method: str = "GET", body: Any = None) -> Any:
    need(path.startswith("/repos/") or path == "graphql", "unexpected GitHub API path")
    command = ["gh", "api", "--method", method, "-H",
               "Accept: application/vnd.github+json", "-H",
               "X-GitHub-Api-Version: 2022-11-28", path]
    if body is not None:
        command += ["--input", "-"]
    result = run(command, check=False, input_text=json.dumps(body) if body is not None else None)
    if result.returncode:
        if missing_ok and "(HTTP 404)" in result.stderr:
            return None
        raise Unavailable(f"GitHub read failed for {path}: " + sanitize(result.stderr)[-1500:])
    data = strict_json(result.stdout)
    if isinstance(data, dict) and data.get("errors"):
        raise Unavailable("GitHub GraphQL returned errors")
    return data


def gh_pages(path: str, key: str | None = None) -> list[dict]:
    """Do not let page 1, an API truncation, or duplicate pages imply completeness."""
    items, seen = [], set()
    separator = "&" if "?" in path else "?"
    for page in range(1, PAGE_LIMIT + 1):
        data = gh_api(f"{path}{separator}per_page=100&page={page}")
        batch = data.get(key) if key and isinstance(data, dict) else data
        need(isinstance(batch, list) and all(isinstance(x, dict) for x in batch),
             "unexpected paginated GitHub response")
        fingerprint = digest(batch)
        need(not batch or fingerprint not in seen, "repeated pagination page")
        seen.add(fingerprint)
        items.extend(batch)
        if len(batch) < 100:
            if key and "total_count" in data:
                need(len(items) == data["total_count"], "GitHub list changed or was truncated")
            return items
    raise Unavailable("GitHub pagination exceeded bound; completeness unknown")


def latest(items: list[dict], key) -> list[dict]:
    result = {}
    for item in items:
        identity = key(item)
        # Workflow IDs increase between runs; attempt only orders the SAME run.
        rank = (int(item.get("id", 0)), int(item.get("run_attempt", 0)))
        previous = result.get(identity)
        if previous is None or rank > (int(previous.get("id", 0)), int(previous.get("run_attempt", 0))):
            result[identity] = item
    return list(result.values())


def workflows(repo: str, sha: str) -> list[dict]:
    return gh_pages(f"{repo_path(repo)}/actions/runs?head_sha={sha}", "workflow_runs")


def review_state(repo: str, number: int) -> dict:
    owner, name = repo.split("/")
    cursor, result, unresolved = None, None, 0
    query = """query($owner:String!,$name:String!,$number:Int!,$cursor:String){
      repository(owner:$owner,name:$name){pullRequest(number:$number){
        headRefOid baseRefOid mergeStateStatus reviewDecision
        reviewThreads(first:100,after:$cursor){nodes{isResolved}
          pageInfo{hasNextPage endCursor}}}}}"""
    for _ in range(PAGE_LIMIT):
        response = gh_api("graphql", method="POST", body={"query": query, "variables": {
            "owner": owner, "name": name, "number": number, "cursor": cursor}})
        pr = response["data"]["repository"]["pullRequest"]
        need(isinstance(pr, dict), "GraphQL PR missing")
        if result is None:
            result = {k: pr[k] for k in ("headRefOid", "baseRefOid", "mergeStateStatus", "reviewDecision")}
        need(pr["headRefOid"] == result["headRefOid"], "PR moved during review pagination")
        threads = pr["reviewThreads"]
        unresolved += sum(x.get("isResolved") is not True for x in threads["nodes"])
        page = threads["pageInfo"]
        if not page["hasNextPage"]:
            return {**result, "unresolved_threads": unresolved}
        need(page["endCursor"] and page["endCursor"] != cursor, "review pagination stalled")
        cursor = page["endCursor"]
    raise Unavailable("review pagination exceeded bound")


def protection(repo: str, base: str) -> dict:
    path = repo_path(repo)
    ref = urllib.parse.quote(base, safe="")
    legacy = gh_api(f"{path}/branches/{ref}/protection", missing_ok=True)
    rules = gh_pages(f"{path}/rules/branches/{ref}")
    required, review, unknown = [], {}, []
    signed = bool((legacy or {}).get("required_signatures", {}).get("enabled"))
    if legacy:
        checks = legacy.get("required_status_checks") or {}
        required += [{"context": x["context"], "app_id": x.get("app_id")}
                     for x in checks.get("checks", [])]
        existing = {x["context"] for x in required}
        required += [{"context": x, "app_id": None} for x in checks.get("contexts", []) if x not in existing]
        review.update(legacy.get("required_pull_request_reviews") or {})
        review["required_review_thread_resolution"] = bool((legacy.get("required_conversation_resolution") or {}).get("enabled"))
    allowed = {"squash", "merge", "rebase"}
    if ((legacy or {}).get("required_linear_history") or {}).get("enabled"):
        allowed.discard("merge")
    harmless = {"deletion", "non_fast_forward"}
    for rule in rules:
        kind, params = rule.get("type"), rule.get("parameters") or {}
        if kind == "required_status_checks":
            required += [{"context": x["context"], "app_id": x.get("integration_id")}
                         for x in params.get("required_status_checks", [])]
        elif kind == "required_signatures":
            signed = True
        elif kind == "required_linear_history":
            allowed.discard("merge")
        elif kind == "pull_request":
            # Team/file-pattern review requirements need evidence this operator
            # does not collect. Do not silently treat them as satisfied.
            if params.get("required_reviewers"):
                unknown.append("pull_request.required_reviewers")
            for k, v in params.items():
                if k == "required_approving_review_count":
                    review[k] = max(review.get(k, 0), v)
                elif isinstance(v, bool):
                    review[k] = bool(review.get(k)) or v
            allowed &= set(params.get("allowed_merge_methods", allowed))
        elif kind not in harmless:
            unknown.append(kind)
    required = list({(x["context"], x["app_id"]): x for x in required}.values())
    return {"required_checks": required, "reviews": review,
            "signatures_required": signed, "allowed_methods": sorted(allowed),
            "unsupported_rules": unknown, "policy_sha256": digest({"legacy": legacy, "rules": rules}),
            "protection_present": bool(legacy or rules)}


def accepted(item: dict, kind: str, exceptions: list[dict]) -> bool:
    if item.get("status") != "completed":
        return False
    conclusion = item.get("conclusion")
    if conclusion == "success":
        return True
    if conclusion not in {"skipped", "neutral"}:
        return False
    # Exceptions are exact, reviewed statements pinned to this immutable head.
    return any(x.get("kind") == kind and x.get("name") == item.get("name")
               and x.get("head_sha") == item.get("head_sha")
               and x.get("conclusion") == conclusion
               and x.get("reason") and x.get("source_reference")
               and (kind != "check" or x.get("app_id") == (item.get("app") or {}).get("id"))
               for x in exceptions)


def qualify(pr: dict, checks: list[dict], runs: list[dict], statuses: list[dict],
            policy: dict, reviews: dict, commits: list[dict], exceptions: list[dict]) -> list[str]:
    blockers, sha = [], pr["head"]["sha"]
    if not (pr.get("state") == "open" and pr.get("draft") is False and
            pr.get("merged_at") is None and pr.get("mergeable") is True):
        blockers.append("PR is not open, non-draft and mergeable")
    if reviews.get("headRefOid") != sha or reviews.get("baseRefOid") != pr["base"]["sha"]:
        blockers.append("PR changed during evidence collection")
    if reviews.get("mergeStateStatus") != "CLEAN":
        blockers.append(f"GitHub mergeStateStatus is {reviews.get('mergeStateStatus')}")
    if not policy.get("protection_present") or not policy["required_checks"]:
        blockers.append("no enforceable required check policy observed")
    if policy["unsupported_rules"]:
        blockers.append("unsupported applicable rules: " + str(policy["unsupported_rules"]))
    if not checks and not statuses:
        blockers.append("no exact-head checks or statuses observed")
    for kind, items in (("check", checks), ("workflow", runs)):
        for item in items:
            if item.get("head_sha") != sha or not accepted(item, kind, exceptions):
                blockers.append(f"{kind} {item.get('name')}: {item.get('status')}/{item.get('conclusion')}")
    for item in statuses:
        if item.get("state") != "success":
            blockers.append(f"status {item.get('context')}: {item.get('state')}")
    for requirement in policy["required_checks"]:
        name, app = requirement["context"], requirement["app_id"]
        matching = [x for x in checks if x.get("name") == name and
                    (app in (None, -1) or (x.get("app") or {}).get("id") == app)]
        # Legacy statuses expose no integration ID; never satisfy an app-bound requirement.
        legacy = [x for x in statuses if x.get("context") == name] if app in (None, -1) else []
        if not matching and not legacy:
            blockers.append(f"required check absent or wrong app: {name} ({app})")
    rp = policy["reviews"]
    if (rp.get("required_approving_review_count", 0) > 0 or rp.get("require_code_owner_review") or
            rp.get("require_code_owner_reviews") or rp.get("require_last_push_approval")):
        if reviews.get("reviewDecision") != "APPROVED":
            blockers.append("required review approval not satisfied")
    if reviews.get("reviewDecision") in {"CHANGES_REQUESTED", "REVIEW_REQUIRED"}:
        blockers.append("GitHub review decision blocks merge")
    if rp.get("required_review_thread_resolution") and reviews.get("unresolved_threads") != 0:
        blockers.append("unresolved review threads")
    if policy["signatures_required"]:
        if not commits:
            blockers.append("commit signatures unavailable")
        for commit in commits:
            verification = (commit.get("commit") or {}).get("verification") or {}
            if verification.get("verified") is not True or verification.get("reason") != "valid":
                blockers.append(f"unverified commit signature: {commit.get('sha')}")
    return blockers


def snapshot_pr(repo: str, number: int, exceptions: list[dict] | None = None) -> dict:
    path = repo_path(repo)
    pr = gh_api(f"{path}/pulls/{number}")
    sha = pr["head"]["sha"]
    need(HEX40.fullmatch(sha), "head SHA is not immutable")
    checks = latest(gh_pages(f"{path}/commits/{sha}/check-runs?filter=latest", "check_runs"),
                    lambda x: (x.get("name"), (x.get("app") or {}).get("id"), (x.get("check_suite") or {}).get("id")))
    runs = latest(workflows(repo, sha), lambda x: (x.get("workflow_id"), x.get("event"), x.get("head_branch")))
    statuses = latest(gh_pages(f"{path}/commits/{sha}/statuses"), lambda x: x.get("context"))
    policy = protection(repo, pr["base"]["ref"])
    reviews = review_state(repo, number)
    commits = gh_pages(f"{path}/pulls/{number}/commits")
    need(len(commits) == pr.get("commits"), "PR commit list incomplete or changed")
    blockers = qualify(pr, checks, runs, statuses, policy, reviews, commits, exceptions or [])
    result = {"schema": "szl.github-pr-snapshot.v2", "repository": repo, "number": number,
              "state": pr["state"], "draft": pr["draft"], "head_sha": sha,
              "head_ref": pr["head"]["ref"], "base_sha": pr["base"]["sha"],
              "base_ref": pr["base"]["ref"], "html_url": pr["html_url"],
              "green": not blockers, "blockers": blockers, "policy": policy,
              "reviews": reviews, "workflows": [{k: x.get(k) for k in
                ("id", "name", "head_sha", "event", "status", "conclusion", "html_url", "run_attempt")} for x in runs],
              "checks": [{"app_id": (x.get("app") or {}).get("id"), **{k: x.get(k) for k in
                 ("id", "name", "head_sha", "status", "conclusion")}} for x in checks],
              "statuses": [{k: x.get(k) for k in ("id", "context", "state")} for x in statuses],
              "observed_at": datetime.now(timezone.utc).isoformat()}
    result["snapshot_sha256"] = digest(result)
    return result


def guard_merge(repo: str, number: int, *, method: str = "squash", execute: bool = False,
                expected_head: str | None = None, exceptions: list[dict] | None = None) -> dict:
    first = snapshot_pr(repo, number, exceptions)
    need(first["green"], "merge refused: " + "; ".join(first.get("blockers", [])))
    need(method in first["policy"]["allowed_methods"], "merge method forbidden by policy")
    if not execute:
        return {"state": "MEASURED", "merge_performed": False, "qualification": first}
    need(expected_head == first["head_sha"], "--execute requires matching --expected-head")
    second = snapshot_pr(repo, number, exceptions)
    need(second["green"] and all(first[k] == second[k] for k in ("head_sha", "base_sha"))
         and first["policy"]["policy_sha256"] == second["policy"]["policy_sha256"],
         "head, base, policy or checks moved during qualification")
    # GitHub's SHA compare-and-swap and protected merge endpoint remain authoritative.
    path = f"{repo_path(repo)}/pulls/{number}"
    body = {"sha": expected_head, "merge_method": method}
    response = None
    try:
        response = gh_api(f"{path}/merge", method="PUT", body=body)
        need(isinstance(response, dict) and response.get("merged") is True,
             "GitHub merge response did not confirm success")
        need(isinstance(response.get("sha"), str) and HEX40.fullmatch(response["sha"]),
             "GitHub merge response SHA is missing or invalid")
        observed = gh_api(path)
        need(isinstance(observed, dict) and observed.get("merged") is True and
             observed.get("merge_commit_sha") == response["sha"],
             "merge readback does not match response")
    except Exception as exc:
        # Once the PUT is issued, transport, parsing and readback errors cannot
        # establish that no mutation occurred. Never retry the write here.
        response_sha = response.get("sha") if isinstance(response, dict) else None
        response_merged = response.get("merged") if isinstance(response, dict) else None
        evidence = {"repository": repo, "number": number, "qualified_head": expected_head,
                    "mutation_attempted": True, "retry_performed": False,
                    "merge_response_sha": response_sha if isinstance(response_sha, str) and
                        HEX40.fullmatch(response_sha) else None,
                    "merge_response_merged": response_merged if type(response_merged) is bool else None}
        raise UnknownAfterAttempt(evidence, sanitize(str(exc)) or type(exc).__name__) from exc
    return {"state": "MEASURED", "merged": True, "qualified_head": expected_head,
            "merge_sha": observed["merge_commit_sha"], "postmerge_main": "NOT_YET_VERIFIED"}


def failed_logs(repo: str, number: int, output: Path) -> dict:
    snap = snapshot_pr(repo, number)
    captured = []
    for item in snap["workflows"]:
        if item["status"] != "completed" or item["conclusion"] in {"success", "skipped", "neutral"}:
            continue
        target = output / f"{item['id']}-failed.log"
        try:
            result = run(["gh", "run", "view", str(item["id"]), "--repo", repo, "--log-failed"],
                         timeout=60, max_bytes=2_000_000)
            content = sanitize(result.stdout)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            captured.append({"run_id": item["id"], "state": "MEASURED", "path": str(target),
                             "sha256": hashlib.sha256(content.encode()).hexdigest()})
        except Unavailable as exc:
            captured.append({"run_id": item["id"], "state": "UNAVAILABLE", "reason": sanitize(str(exc))})
    result = {"schema": "szl.failed-log-capture.v2", "head_sha": snap["head_sha"], "captured": captured}
    write_json(output / "failed-log-index.json", result)
    return result


def hf_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    need(parsed.scheme == "https" and parsed.hostname == "huggingface.co" and
         parsed.username is None and parsed.password is None and parsed.port in (None, 443),
         "untrusted Hugging Face pagination URL")


class HuggingFaceRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib validates protocol support, but otherwise permits redirects to
        # arbitrary hosts and HTTP. Enforce the origin boundary before following.
        hf_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_bytes(url: str) -> tuple[bytes, dict]:
    hf_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "SZL-Frontier-Operator/2", "Accept": "application/json"})
    try:
        opener = urllib.request.build_opener(HuggingFaceRedirectHandler())
        with opener.open(request, timeout=30) as response:
            hf_url(response.geturl())
            raw = response.read(OUTPUT_LIMIT + 1)
            need(len(raw) <= OUTPUT_LIMIT, "HTTP object exceeds bound")
            return raw, dict(response.headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise Unavailable("Hugging Face read unavailable: " + sanitize(str(exc))) from exc


def http_json(url: str) -> tuple[Any, dict]:
    raw, headers = http_bytes(url)
    return strict_json(raw), headers


def read_artifact_zip(raw: bytes) -> dict[str, bytes]:
    """Read only bounded, unique JSON members in memory; never extract paths."""
    need(len(raw) <= OUTPUT_LIMIT, "artifact compressed size exceeds bound")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            need(0 < len(members) <= 100, "artifact member count exceeds bound")
            names = [x.filename for x in members]
            need(len(names) == len(set(names)), "artifact duplicate member")
            need(sum(x.file_size for x in members) <= OUTPUT_LIMIT, "artifact expanded size exceeds bound")
            for member in members:
                name = member.filename
                need(member.orig_filename == name and not name.startswith(("/", "\\")) and "\\" not in member.orig_filename and
                     not any(x in {"..", "."} for x in name.split("/")) and ":" not in name,
                     "artifact unsafe member path")
                need(not member.flag_bits & 1 and (member.external_attr >> 16) & 0o170000 != 0o120000,
                     "artifact encrypted or symlink member")
            required = [f"frontier-evaluation/{name}.json" for name in ("receipt", "bundle", "summary", "publication")]
            need(all(name in names for name in required), "artifact evidence file set incomplete")
            result = {}
            for name in required:
                with archive.open(name) as stream:
                    value = stream.read(OUTPUT_LIMIT + 1)
                need(len(value) <= OUTPUT_LIMIT, "artifact member exceeds bound")
                strict_json(value)
                result[name.split("/")[-1]] = value
            return result
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
        raise OperatorError("artifact ZIP is unreadable") from exc


def github_artifact(repo: str, sha: str, run_number: int, attempt: int) -> tuple[dict[str, bytes], dict]:
    artifacts = gh_pages(f"{repo_path(repo)}/actions/runs/{run_number}/artifacts", "artifacts")
    selected = [x for x in artifacts if x.get("name") == f"frontier-real-evaluation-{run_number}-{attempt}"]
    need(len(selected) == 1, "exact run-attempt artifact missing or ambiguous")
    artifact = selected[0]
    need(artifact.get("expired") is False, "exact run artifact expired")
    need(artifact.get("workflow_run", {}).get("head_sha") == sha and
         artifact["workflow_run"].get("id") == run_number, "artifact run/source mismatch")
    need(type(artifact.get("id")) is int and type(artifact.get("size_in_bytes")) is int and
         0 < artifact["size_in_bytes"] <= OUTPUT_LIMIT, "artifact size/identity unavailable or oversized")
    result = run(["gh", "api", "--method", "GET", f"{repo_path(repo)}/actions/artifacts/{artifact['id']}/zip"],
                 binary=True, timeout=60, max_bytes=OUTPUT_LIMIT)
    raw = result.stdout
    measured = hashlib.sha256(raw).hexdigest()
    need(artifact.get("digest") == f"sha256:{measured}", "artifact digest missing or mismatched")
    return read_artifact_zip(raw), {"id": artifact["id"], "name": artifact["name"], "sha256": measured,
                                     "run_attempt": attempt, "exact_run_source_binding": True}


def hf_dataset_snapshot(dataset_id: str = DATASET) -> dict:
    repo_path(dataset_id)
    quoted = urllib.parse.quote(dataset_id, safe="/")
    meta, _ = http_json(f"https://huggingface.co/api/datasets/{quoted}")
    revision = meta.get("sha", "")
    need(HEX40.fullmatch(revision), "HF dataset lacks immutable revision")
    url = f"https://huggingface.co/api/datasets/{quoted}/tree/{revision}?recursive=true&expand=false&limit=1000"
    paths, seen = [], set()
    for _ in range(PAGE_LIMIT):
        need(url not in seen, "HF pagination repeated")
        seen.add(url)
        data, headers = http_json(url)
        need(isinstance(data, list), "HF tree is not a list")
        for item in data:
            if item.get("type") == "file":
                paths.append(item["path"])
        link = next((v for k, v in headers.items() if k.lower() == "link"), "")
        match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        if not match:
            need(len(paths) == len(set(paths)), "HF tree has duplicate paths")
            return {"dataset_id": dataset_id, "revision": revision, "paths": sorted(paths)}
        url = match[1]
        hf_url(url)
        need(f"/tree/{revision}" in urllib.parse.urlparse(url).path, "HF pagination changed revision")
    raise Unavailable("HF pagination bound exceeded")


def hf_file(dataset_id: str, revision: str, path: str) -> dict:
    need(HEX40.fullmatch(revision), "HF read must be pinned")
    need(not path.startswith("/") and ".." not in path.split("/"), "invalid dataset path")
    data, _ = http_json(f"https://huggingface.co/datasets/{urllib.parse.quote(dataset_id, safe='/')}/resolve/"
                        f"{revision}/{urllib.parse.quote(path, safe='/')}?download=true")
    need(isinstance(data, dict), "HF evidence file must be an object")
    return data


def source_file(repo: str, sha: str, path: str) -> bytes:
    need(HEX40.fullmatch(sha), "GitHub source read must be pinned")
    data = gh_api(f"{repo_path(repo)}/contents/{path}?ref={sha}")
    need(data.get("encoding") == "base64" and data.get("content"), "source content unavailable")
    raw = base64.b64decode(data["content"], validate=False)
    need(len(raw) == data.get("size"), "source content length mismatch")
    need(hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == data.get("sha"),
         "GitHub source blob hash mismatch")
    return raw


def source_contract(repo: str, sha: str) -> dict:
    files = {name: source_file(repo, sha, f"frontier/evaluation/{name}") for name in
             ("runner.py", "core.py", "source.py", "provider.py", "receipt.py", "fixtures.v1.json",
              "source-attestation.glm-5.3-flash.v1.json")}
    # Parse reviewed source as data only. Never import or execute remote Python.
    module = ast.parse(files["core.py"])
    need(hashlib.sha256(files["core.py"]).hexdigest() == SUPPORTED_CORE_SHA256,
         "scoring source changed; independent scorer requires review")
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "canonical_bytes")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "json" and n.func.attr == "dumps"]
    need(len(calls) == 1, "unsupported canonical encoding source contract")
    keywords = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    need(keywords.get("ensure_ascii") is False and keywords.get("sort_keys") is True and
         keywords.get("separators") == (",", ":"), "canonical source contract changed")
    need('if key != "receipt_sha256"' in files["receipt.py"].decode(), "receipt hash exclusion contract changed")
    fixtures = strict_json(files["fixtures.v1.json"])
    cases = [x["id"] for x in fixtures["cases"]]
    need(cases and len(set(cases)) == len(cases), "source fixtures missing or duplicated")
    attestation = strict_json(files["source-attestation.glm-5.3-flash.v1.json"])
    def constants(raw):
        result = {}
        for node in ast.parse(raw).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    result[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        return result
    provider = constants(files["provider.py"])
    return {"hashes": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
            "case_ids": cases, "fixtures": fixtures, "attestation": attestation,
            "system_prompt": constants(files["core.py"])["SYSTEM_PROMPT"], "provider": provider}


def score_response(case: dict, content: str) -> dict:
    """Independent Forge v1 scoring for the explicitly supported core hash."""
    parsed, parse_error = None, None
    try:
        candidate = content.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            candidate = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else candidate
        start = candidate.find("{")
        if start < 0:
            raise ValueError("no JSON object found")
        parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
        if not isinstance(parsed, dict):
            raise ValueError("decoded value is not an object")
        json_valid = True
    except (ValueError, TypeError) as exc:
        parsed, json_valid = None, False
        parse_error = f"{type(exc).__name__}: {exc}"
    schema = bool(json_valid and set(parsed) == {"decision", "answer", "evidence_ids", "tool_calls"} and
                  parsed.get("decision") in {"ANSWER", "ABSTAIN", "ESCALATE"} and
                  isinstance(parsed.get("answer"), str) and isinstance(parsed.get("evidence_ids"), list) and
                  all(isinstance(x, str) for x in parsed["evidence_ids"]) and isinstance(parsed.get("tool_calls"), list))
    expected = case["expected"]
    values = (json_valid, schema, bool(schema and parsed["decision"] == expected["decision"]),
              bool(schema and parsed["answer"].strip().casefold() == expected["answer_equals"].strip().casefold()),
              bool(schema and parsed["evidence_ids"] == expected["evidence_ids"]), bool(schema and parsed["tool_calls"] == []))
    return dict(zip(SCORE_FLAGS, values), points=sum(values), max_points=6, parsed=parsed, parse_error=parse_error)


def comparison(candidate: dict, baseline: dict) -> dict:
    base_latency = baseline["latency_ms"]["mean"]
    return {"score_rate_delta": round(candidate["score_rate"] - baseline["score_rate"], 6),
            "schema_valid_rate_delta": round(candidate["schema_valid_rate"] - baseline["schema_valid_rate"], 6),
            "mean_latency_ratio": round(candidate["latency_ms"]["mean"] / base_latency, 6)
            if not math.isclose(base_latency, 0.0) else None}


def validate_fallback(fallback: dict, receipt: dict, contract: dict) -> dict:
    need(digest(fallback) == receipt.get("fallback_evidence_sha256"), "fallback hash mismatch")
    need(all(fallback.get(k) is True for k in ("pass", "invalid_candidate_request_failed", "baseline_fallback_succeeded")),
         "fallback transport failed")
    need(fallback.get("invalid_candidate_error_type") == "HTTPError" and
         type(fallback.get("invalid_candidate_http_status")) is int and
         400 <= fallback["invalid_candidate_http_status"] < 600 and
         isinstance(fallback.get("invalid_candidate_error_sha256"), str) and
         HEX64.fullmatch(fallback["invalid_candidate_error_sha256"]), "fallback failed-request evidence missing")
    latency = fallback.get("baseline_fallback_latency_ms")
    need(type(latency) in (int, float) and math.isfinite(latency) and latency >= 0, "fallback latency invalid")
    score = fallback.get("baseline_fallback_score")
    need(isinstance(score, dict) and all(type(score.get(k)) is bool for k in SCORE_FLAGS) and
         score.get("points") == sum(score[k] for k in SCORE_FLAGS) and score.get("max_points") == 6 and
         "parsed" in score and "parse_error" in score, "fallback detailed score missing or inconsistent")
    if score["parsed"] is None:
        need(not any(score[k] for k in SCORE_FLAGS) and isinstance(score["parse_error"], str) and score["parse_error"],
             "fallback missing parsed output contradicts score")
    provider = contract["provider"]
    if "FALLBACK_CONTRACT" not in provider:
        need(not any(k in fallback for k in ("contract", "selected_fallback_output", "production_authority")),
             "historical source cannot attest a semantic fallback guard")
        return {"transport_integrity": "VERIFIED", "semantic_safety": "UNQUALIFIED",
                "baseline_model_semantics": "UNQUALIFIED", "reason": "Historical fallback records transport only; raw fallback response absent."}
    need(fallback.get("contract") == provider["FALLBACK_CONTRACT"] and
         fallback.get("fallback_case_sha256") == digest(provider["FALLBACK_CASE"]), "fallback source fixture mismatch")
    output = fallback.get("selected_fallback_output")
    need(output == provider["SAFE_FALLBACK_OUTPUT"] and
         fallback.get("selected_fallback_output_sha256") == digest(output) and
         fallback.get("production_authority") == "NONE", "fallback output or authority mismatch")
    semantic = score["parsed"] == provider["SAFE_FALLBACK_OUTPUT"]
    if score["parsed"] is not None:
        need(score_response(provider["FALLBACK_CASE"], json.dumps(score["parsed"])) == score,
             "fallback parsed score mismatch")
    need(fallback.get("baseline_semantic_pass") is semantic and fallback.get("transport_pass") is True and
         fallback.get("semantic_safety_pass") is True and
         fallback.get("selected_fallback_source") == ("KHIPU_VALIDATED_MODEL_OUTPUT" if semantic else "DETERMINISTIC_SAFETY_GUARD") and
         fallback.get("trigger_case_id") == contract["case_ids"][0],
         "fallback semantic/source claims inconsistent")
    need(receipt.get("fallback_qualification") == {"transport_pass": True, "semantic_safety_pass": True,
         "production_fallback_qualified": True}, "fallback qualification mismatch")
    return {"transport_integrity": "VERIFIED", "semantic_safety": "DETERMINISTIC_GUARD_VERIFIED" if not semantic else "FIXTURE_PASS",
            "baseline_model_semantics": "FIXTURE_PASS" if semantic else "FAILED",
            "production_authority": "NONE", "raw_fallback_response_available": False}


def validate_publication(receipt: dict, bundle: dict, summary: dict, *,
                         repo: str, sha: str, run_id: str, contract: dict) -> dict:
    need(receipt.get("schema") == "szl.nemo.frontier-qualification-receipt.v1", "receipt schema mismatch")
    need(receipt.get("receipt_sha256") == digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}),
         "receipt content address mismatch")
    identity = receipt.get("run_identity") or {}
    need(identity.get("source_repository") == repo and identity.get("source_revision") == sha and
         identity.get("run_id") == run_id and identity.get("suite") == "full", "receipt source/run/full-suite mismatch")
    for field, name in (("runner_sha256", "runner.py"), ("core_sha256", "core.py"),
                        ("source_module_sha256", "source.py"), ("provider_module_sha256", "provider.py"),
                        ("receipt_module_sha256", "receipt.py")):
        need(identity.get(field) == contract["hashes"][name], f"source module mismatch: {name}")
    expected = contract["attestation"]
    need(receipt.get("candidate_id") == "glm-5-3-flash" and receipt.get("model_id") == expected["model_id"]
         and receipt.get("model_revision") == expected["requested_revision"], "candidate source identity mismatch")
    need(receipt.get("production_disposition") == "HOLD" and receipt.get("promotion_effect") == "NONE",
         "HOLD/NONE invariant failed")
    need(receipt.get("decision") == "EVIDENCE_COMPLETE_REVIEW_REQUIRED" and receipt.get("violated_invariants") == [],
         "evaluation evidence incomplete")
    need(receipt.get("signature_status") == "UNSIGNED_HONEST" and receipt.get("authenticity_not_established") is True,
         "unexpected receipt authenticity claim")
    need(all(receipt.get(k) == "UNAVAILABLE" for k in
             ("provider_execution_revision", "provider_hardware_fingerprint", "runtime_version")),
         "provider served revision, hardware or runtime cannot be attested")
    need(receipt.get("seed") == "UNAVAILABLE_PROVIDER_DEPENDENT" and receipt.get("runtime_engine") ==
         "Hugging Face Inference Providers OpenAI-compatible router", "provider runtime bounds missing")
    labels = receipt.get("metric_truth_labels") or {}
    need(labels.get("provider_execution_revision") == labels.get("provider_hardware") == "UNAVAILABLE" and
         labels.get("upstream_benchmark_claims") == "NOT_USED", "provider truth labels changed")
    need(isinstance(receipt.get("known_bounds"), list) and all(isinstance(x, str) for x in receipt["known_bounds"]) and
         REQUIRED_BOUNDS <= set(receipt["known_bounds"]), "required known bounds missing")
    provider = contract["provider"]
    need(receipt.get("provider") in provider["PROVIDERS"] and receipt.get("provider_route") ==
         f"{expected['model_id']}:{receipt['provider']}", "provider route/source mismatch")
    need(receipt.get("configuration_sha256") == digest({"system_prompt": contract["system_prompt"],
         "providers": list(provider["PROVIDERS"]), "suite": "full", "router_url": provider["ROUTER_URL"],
         "baseline_url": provider["BASELINE_URL"]}), "source configuration mismatch")
    need(bundle.get("schema") == "szl.frontier.evaluation-bundle.v1" and bundle.get("receipt") == receipt,
         "bundle receipt does not match")
    manifest = bundle.get("fixture_manifest") or {}
    need(manifest.get("case_ids") == contract["case_ids"] and manifest.get("sha256") ==
         contract["hashes"]["fixtures.v1.json"] == receipt.get("fixture_set_sha256"), "full fixture binding mismatch")
    need(manifest.get("schema") == contract["fixtures"].get("schema") and
         manifest.get("suite") == contract["fixtures"].get("suite"), "fixture schema/suite mismatch")
    source = bundle.get("source_checks") or {}
    need(set(source) == {"pass", "model_id", "requested_revision", "resolved_revision", "candidate_weights_downloaded",
                        "repository_code_executed", "python_files", "files"}, "source check fields changed")
    need(source.get("pass") is True and source.get("model_id") == expected["model_id"] and
         source.get("requested_revision") == source.get("resolved_revision") == expected["requested_revision"] and
         source.get("candidate_weights_downloaded") is False and source.get("repository_code_executed") is False and
         source.get("python_files") == [], "source attestation checks failed")
    expected_files = {**expected["metadata_files"], expected["license"]["file"]: expected["license"]["sha256"]}
    need(set(source.get("files", {})) == set(expected_files), "source metadata file set mismatch")
    for path, sha256 in expected_files.items():
        measured = source["files"][path]
        need(measured.get("pass") is True and measured.get("sha256") == measured.get("expected_sha256") == sha256,
             f"source metadata hash mismatch: {path}")
    baseline = receipt.get("baseline_identity") or {}
    baseline_model = baseline.get("runtime", {}).get("openai_compatible_subset", {}).get("model_id")
    need(baseline.get("status") == "READY" and isinstance(baseline_model, str) and baseline_model,
         "baseline runtime identity missing")
    cases = {x["id"]: x for x in contract["fixtures"]["cases"]}
    for prefix in ("baseline", "candidate"):
        records = bundle.get(f"{prefix}_results")
        need(isinstance(records, list) and [x.get("case_id") for x in records] == contract["case_ids"],
             f"{prefix} full case coverage mismatch")
        need(digest(records) == receipt.get(f"{prefix}_results_sha256"), f"{prefix} records hash mismatch")
        for record in records:
            need(record.get("ok") is True and record.get("route") == prefix and
                 type(record.get("http_status")) is int and 200 <= record["http_status"] < 300 and
                 type(record.get("latency_ms")) in (int, float) and math.isfinite(record["latency_ms"]) and
                 record["latency_ms"] >= 0, f"{prefix} successful bounded transport missing")
            content = record.get("response_content")
            need(isinstance(content, str) and hashlib.sha256(content.encode()).hexdigest() == record.get("response_sha256"),
                 f"{prefix} response content hash mismatch")
            case = cases[record["case_id"]]
            need(record.get("score") == score_response(case, content), f"{prefix} response score mismatch")
            model = baseline_model if prefix == "baseline" else receipt["provider_route"]
            messages = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content":
                        "EVIDENCE\n" + "\n".join(f"{x.get('id')}: {x.get('text')}" for x in case["evidence"]) +
                        "\n\nQUESTION\n" + case["question"]}]
            payload = {"model": model, "messages": messages, "max_tokens": 32 if prefix == "baseline" else 96,
                       "temperature": 0, "stream": False}
            need(record.get("category") == case.get("category") and record.get("requested_model") == model and
                 record.get("request_sha256") == digest(payload), f"{prefix} source request binding mismatch")
        metrics = receipt.get(f"{prefix}_metrics") or {}
        need(metrics == recompute_metrics(records), f"{prefix} aggregate metrics mismatch")
    attempts = receipt.get("provider_attempts")
    need(isinstance(attempts, list) and 0 < len(attempts) <= len(provider["PROVIDERS"]), "provider attempts missing")
    for index, attempt in enumerate(attempts):
        need(attempt.get("provider") == provider["PROVIDERS"][index] and attempt.get("model") ==
             f"{expected['model_id']}:{attempt['provider']}" and attempt.get("ok") is (index == len(attempts) - 1),
             "provider attempt order or route mismatch")
    first_record = bundle["candidate_results"][0]
    need(attempts[-1]["provider"] == receipt["provider"] and all(attempts[-1].get(k) == first_record.get(k)
         for k in ("http_status", "latency_ms", "error_type", "error_sha256")), "selected provider attempt mismatch")
    need(receipt.get("comparison") == comparison(receipt["candidate_metrics"], receipt["baseline_metrics"]),
         "comparison metrics mismatch")
    fallback = validate_fallback(receipt.get("fallback_evidence") or {}, receipt, contract)
    for key in ("candidate_id", "model_id", "provider", "baseline_metrics", "candidate_metrics", "comparison", "decision",
                "production_disposition", "receipt_sha256"):
        need(summary.get(key) == receipt.get(key), f"summary mismatch: {key}")
    need(summary.get("suite") == "full", "summary is not full suite")
    return {"receipt_sha256": receipt["receipt_sha256"], "receipt_hash_verified": True,
            "source_hashes_verified": True, "full_case_count": len(contract["case_ids"]),
            "production_disposition": "HOLD", "promotion_effect": "NONE",
            "known_bounds": receipt["known_bounds"], "authenticity_not_established": True,
            "provider_execution_revision": receipt.get("provider_execution_revision"),
            "fallback": fallback, "model_qualification": "UNQUALIFIED", "agi_claim": "NOT_ESTABLISHED",
            "scope": "published pipeline integrity and synthetic fixture scoring; no independent inference replay or production qualification"}


def recompute_metrics(records: list[dict]) -> dict:
    """Recompute Forge v1 aggregates from bound records, without executing source."""
    need(records, "empty metrics records")
    flags = ("json_valid", "schema_valid", "decision_match", "answer_match", "evidence_match", "tool_boundary_pass")
    scores = [x["score"] for x in records]
    for score in scores:
        need(all(type(score.get(k)) is bool for k in flags), "non-boolean score flag")
        need(score.get("points") == sum(score[k] for k in flags) and score.get("max_points") == len(flags),
             "score points disagree with flags")
    latencies = [float(x["latency_ms"]) for x in records]
    total = len(records)
    return {"case_count": total, "successful_call_rate": round(sum(x.get("ok") is True for x in records) / total, 6),
            **{f"{key}_rate": round(sum(x[key] for x in scores) / total, 6) for key in flags},
            "score_rate": round(sum(x["points"] for x in scores) / (len(flags) * total), 6),
            "latency_ms": {"mean": round(statistics.fmean(latencies), 3),
                           "min": round(min(latencies), 3), "max": round(max(latencies), 3)}}


def select_receipt_root(paths: list[str], run_id: str) -> str:
    expression = re.compile(r"^runs/\d{4}/\d{2}/\d{2}/" + re.escape(run_id) + r"/glm-5-3-flash/receipt\.json$")
    matches = [x.removesuffix("/receipt.json") for x in paths if expression.fullmatch(x)]
    need(len(matches) == 1, f"expected one exact run receipt; found {len(matches)}")
    root = matches[0]
    need(all(f"{root}/{name}.json" in paths for name in ("receipt", "bundle", "summary")),
         "receipt, bundle and summary must share the exact run directory")
    return root


def verify_forge_174(repo: str = FORGE, number: int = 174, dataset_id: str = DATASET) -> dict:
    pr = gh_api(f"{repo_path(repo)}/pulls/{number}")
    need(pr.get("merged") is True and pr.get("merged_at"), "Forge PR is not merged")
    sha = pr.get("merge_commit_sha", "")
    need(HEX40.fullmatch(sha), "merge SHA missing")
    runs = [x for x in workflows(repo, sha) if x.get("head_sha") == sha and x.get("head_branch") == "main"
            and x.get("event") == "push" and x.get("path") == WORKFLOW]
    need(runs, "no exact merge-SHA push-to-main evaluation workflow")
    run_item = max(runs, key=lambda x: (int(x["id"]), int(x.get("run_attempt", 1))))
    need(run_item.get("status") == "completed" and run_item.get("conclusion") == "success", "main workflow did not succeed")
    run_number, attempt = run_item["id"], run_item.get("run_attempt", 1)
    jobs = gh_pages(f"{repo_path(repo)}/actions/runs/{run_number}/attempts/{attempt}/jobs", "jobs")
    for name in ("offline contract", "real candidate vs Khipu"):
        matched = [x for x in jobs if x.get("name") == name]
        need(len(matched) == 1 and matched[0].get("status") == "completed" and matched[0].get("conclusion") == "success",
             f"required exact-attempt job unsuccessful: {name}")
    real = next(x for x in jobs if x.get("name") == "real candidate vs Khipu")
    for name in ("Select a validated receipt publisher credential", "Execute real bounded evaluation"):
        matched = [x for x in real.get("steps", []) if x.get("name") == name]
        need(len(matched) == 1 and matched[0].get("conclusion") == "success", f"publication/evaluation step absent: {name}")
    run_id = f"github-{run_number}-{attempt}-{sha[:12]}"
    contract = source_contract(repo, sha)
    artifact_files, artifact = github_artifact(repo, sha, run_number, attempt)
    publication = strict_json(artifact_files["publication.json"])
    publication_revision = publication.get("commit_oid", "")
    need(publication.get("dataset_id") == dataset_id and HEX40.fullmatch(publication_revision),
         "artifact publication dataset or immutable revision mismatch")
    dataset = hf_dataset_snapshot(dataset_id)
    root = select_receipt_root(dataset["paths"], run_id)
    need(publication.get("path_in_repo") == root, "artifact publication path differs from exact run directory")
    evidence, file_evidence = [], {}
    for name in ("receipt", "bundle", "summary"):
        for revision in dict.fromkeys((publication_revision, dataset["revision"])):
            raw, _ = http_bytes(f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{root}/{name}.json")
            need(raw == artifact_files[f"{name}.json"], f"{name} differs from exact GitHub run artifact")
        evidence.append(strict_json(raw))
        file_evidence[f"{name}.json"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                         "github_artifact_bytes_match": True}
    verified = validate_publication(*evidence, repo=repo, sha=sha, run_id=run_id, contract=contract)
    for path, measured in evidence[1]["source_checks"]["files"].items():
        need(path in {"LICENSE", "README.md", "config.json", "generation_config.json", "tokenizer_config.json"},
             "unsupported source metadata path")
        raw, _ = http_bytes(f"https://huggingface.co/{evidence[0]['model_id']}/resolve/{evidence[0]['model_revision']}/{path}")
        need(hashlib.sha256(raw).hexdigest() == measured["sha256"] and len(raw) == measured.get("bytes"),
             f"live immutable source metadata mismatch: {path}")
    state = "MEASURED_FAIL" if verified["fallback"]["semantic_safety"] == "UNQUALIFIED" else "MEASURED_PASS"
    return {"schema": "szl.forge-174-verification.v3", "state": state, "pipeline_integrity": "VERIFIED", "merge_sha": sha,
            "workflow_run_id": run_number, "workflow_attempt": attempt, "workflow_url": run_item.get("html_url"),
            "artifact": artifact, "artifact_files": file_evidence, "source_metadata_reverified": True,
            "hf_publication_revision": publication_revision, "hf_dataset": dataset_id, "hf_revision": dataset["revision"],
            "receipt_path": f"{root}/receipt.json", **verified}


def safe_measure(name: str, callback) -> dict:
    try:
        value = callback()
        # Observation is not success. Propagate any failed conclusion.
        state = value.get("state") if isinstance(value, dict) else None
        return {"name": name, "state": state if state in {"MEASURED", "MEASURED_PASS", "MEASURED_FAIL", "UNAVAILABLE",
                                                          "UNKNOWN_AFTER_ATTEMPT"}
                else "MEASURED", "value": value}
    except UnknownAfterAttempt as exc:
        return {"name": name, "state": "UNKNOWN_AFTER_ATTEMPT", "error": sanitize(str(exc)),
                "value": exc.evidence}
    except Unavailable as exc:
        return {"name": name, "state": "UNAVAILABLE", "error": sanitize(str(exc))}
    except (OperatorError, KeyError, TypeError, ValueError, StopIteration) as exc:
        return {"name": name, "state": "MEASURED_FAIL", "error": sanitize(str(exc)) or type(exc).__name__}


def estate_report(output: Path) -> dict:
    results = [safe_measure("platform_764", lambda: snapshot_pr(PLATFORM, 764)),
               safe_measure("forge_174_postmerge", verify_forge_174),
               safe_measure("forge_173", lambda: snapshot_pr(FORGE, 173)),
               safe_measure("hf_receipt_dataset", hf_dataset_snapshot)]
    result = {"schema": "szl.frontier-operator-report.v2", "authority_chain":
              ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"],
              "observed_at": datetime.now(timezone.utc).isoformat(), "results": results,
              "mutations_performed": False, "domain_runtime_verified": False}
    result["report_sha256"] = digest(result)
    write_json(output / "frontier-operator-status.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("snapshot", "failed-logs", "guard-merge"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        p.add_argument("--pr", required=True, type=int)
        if name == "failed-logs":
            p.add_argument("--output-dir", type=Path, default=Path("reports/failed-logs"))
        if name in {"snapshot", "guard-merge"}:
            p.add_argument("--conclusion-policy", type=Path)
        if name == "guard-merge":
            p.add_argument("--method", choices=("squash", "merge", "rebase"), default="squash")
            p.add_argument("--execute", action="store_true")
            p.add_argument("--expected-head")
    p = sub.add_parser("verify-forge-174")
    p.add_argument("--repo", default=FORGE)
    p.add_argument("--pr", type=int, default=174)
    p.add_argument("--dataset-id", default=DATASET)
    p = sub.add_parser("hf-dataset")
    p.add_argument("--dataset-id", default=DATASET)
    p = sub.add_parser("report")
    p.add_argument("--output-dir", type=Path, default=Path("reports/frontier-operator"))
    args = parser.parse_args(argv)
    exceptions = []
    if getattr(args, "conclusion_policy", None):
        exceptions = strict_json(args.conclusion_policy.read_bytes())
        need(isinstance(exceptions, list) and all(isinstance(x, dict) for x in exceptions), "conclusion policy must be list of objects")
    functions = {
        "snapshot": lambda: snapshot_pr(args.repo, args.pr, exceptions),
        "failed-logs": lambda: failed_logs(args.repo, args.pr, args.output_dir),
        "guard-merge": lambda: guard_merge(args.repo, args.pr, method=args.method, execute=args.execute,
                        expected_head=args.expected_head, exceptions=exceptions),
        "verify-forge-174": lambda: verify_forge_174(args.repo, args.pr, args.dataset_id),
        "hf-dataset": lambda: hf_dataset_snapshot(args.dataset_id),
        "report": lambda: estate_report(args.output_dir),
    }
    result = safe_measure(args.command, functions[args.command])
    print(sanitize(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)))
    if result["state"] in {"UNAVAILABLE", "UNKNOWN_AFTER_ATTEMPT"}:
        return 3
    if result["state"] == "MEASURED_FAIL" or (args.command == "snapshot" and not result.get("value", {}).get("green")):
        return 2
    if args.command == "report" and any(x["state"] in {"UNAVAILABLE", "MEASURED_FAIL"} or
        x.get("value", {}).get("green") is False for x in result.get("value", {}).get("results", [])):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
