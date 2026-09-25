#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline second-reader for existing SZL census receipts and repeatability data.

This is NOT a collector, source reviewer, runtime prober, publisher or authorizer.
It reads local, digest-bound ZIPs from the existing Forge collector. No network,
model load, subprocess, archive extraction, repository write or secret lookup.
Do not install a new estate service: integrate useful checks into existing owners.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any
import zipfile

MAX_ARCHIVE = 16 * 1024 * 1024
MAX_JSON = 32 * 1024 * 1024
MAX_ITEMS = 512
MAX_ENTRIES = 250_000
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REPO = re.compile(r"szl-holdings/[A-Za-z0-9_.-]+\Z")
HF_ID = re.compile(r"SZLHOLDINGS/[A-Za-z0-9_.-]+\Z")
SCHEMA = "szl.public-estate-file-census/v1"
SCOPE = "PUBLIC_DEFAULT_REVISION_FILE_METADATA_ONLY"
FALSE_FIELDS = ("production_authorization", "runtime_verified", "semantic_review_complete")
MODE_TYPES = {"040000": "tree", "100644": "blob", "100755": "blob", "120000": "blob", "160000": "commit"}


class ContractError(ValueError):
    """A fixed diagnostic, never raw input, source text or a private path."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def integer(value: Any, minimum: int = 0, maximum: int = 10**15) -> bool:
    return type(value) is int and minimum <= value <= maximum


def digest(value: Any, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def strict_json(raw: bytes) -> Any:
    need(type(raw) is bytes and 0 < len(raw) <= MAX_JSON, "JSON_SIZE")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            need(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = value
        return out

    def finite(text: str) -> float:
        value = float(text)
        need(math.isfinite(value), "NONFINITE_JSON")
        return value

    def reject(_: str) -> None:
        raise ContractError("NONFINITE_JSON")

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                            parse_float=finite, parse_constant=reject)
        # JSON can decode escaped lone surrogates. They cannot identify Git paths.
        stack = [(result, 0)]
        nodes = 0
        while stack:
            value, depth = stack.pop()
            need(depth <= 256, "JSON_DEPTH_BOUND")
            nodes += 1
            need(nodes <= 4_000_000, "JSON_NODE_BOUND")
            if isinstance(value, str):
                value.encode("utf-8", errors="strict")
            elif isinstance(value, dict):
                stack.extend((key, depth + 1) for key in value.keys())
                stack.extend((item, depth + 1) for item in value.values())
            elif isinstance(value, list):
                stack.extend((item, depth + 1) for item in value)
        return result
    except ContractError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise ContractError("INVALID_JSON") from None


def safe_path(value: Any) -> str:
    need(type(value) is str and 0 < len(value) <= 4096, "PATH_TYPE")
    need("\\" not in value and not any(ord(c) < 32 or ord(c) == 127 for c in value), "PATH_CONTROL")
    parts = value.split("/")
    need(all(p and p not in {".", ".."} for p in parts), "PATH_TRAVERSAL")
    need(not re.match(r"^[A-Za-z]:", value), "PATH_DRIVE")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ContractError("PATH_ENCODING") from None
    return value


def parse_stamp(value: Any) -> datetime:
    need(type(value) is str and len(value) <= 64, "TIME_TYPE")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        need(stamp.tzinfo is not None and stamp.utcoffset() is not None, "TIME_ZONE")
        return stamp.astimezone(timezone.utc)
    except ContractError:
        raise
    except (ValueError, OverflowError):
        raise ContractError("TIME_INVALID") from None


def window(record: dict[str, Any], now: datetime) -> tuple[datetime, datetime]:
    need(type(record) is dict, "TIME_RECORD_OBJECT")
    first, last = parse_stamp(record.get("started_at")), parse_stamp(record.get("finished_at"))
    need(first <= last <= now + timedelta(seconds=120), "TIME_WINDOW")
    return first, last


def read_native_archive(path: Path, expected: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify bounded ZIP and its one JSON member without extracting any paths.

    Expected ZIP digest must come from an independently inspected native artifact
    record. A supplied digest alone is NOT authenticity or a signature.
    """
    need(digest(expected, SHA256), "ARCHIVE_DIGEST_TYPE")
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_ARCHIVE + 1)
        need(0 < len(raw) <= MAX_ARCHIVE, "ARCHIVE_SIZE")
        actual = hashlib.sha256(raw).hexdigest()
        need(actual == expected, "ARCHIVE_DIGEST_MISMATCH")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            need(len(members) == 1, "ARCHIVE_MEMBER_COUNT")
            member = members[0]
            safe_path(member.filename)
            need(member.filename.endswith(".json") and not member.is_dir(), "ARCHIVE_MEMBER_TYPE")
            mode = member.external_attr >> 16
            need(not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG), "ARCHIVE_SPECIAL_FILE")
            need(not (member.flag_bits & 1), "ARCHIVE_ENCRYPTED")
            need(0 < member.file_size <= MAX_JSON, "ARCHIVE_MEMBER_SIZE")
            with archive.open(member) as stream:
                payload = stream.read(MAX_JSON + 1)
            need(len(payload) == member.file_size, "ARCHIVE_MEMBER_LENGTH")
        result = strict_json(payload)
        need(type(result) is dict, "RECEIPT_OBJECT")
        return result, {"archive_sha256": actual, "archive_bytes": len(raw),
                        "json_sha256": hashlib.sha256(payload).hexdigest(), "json_bytes": len(payload)}
    except ContractError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError):
        raise ContractError("ARCHIVE_UNREADABLE") from None


def git_object(kind: str, body: bytes) -> str:
    return hashlib.sha1(kind.encode("ascii") + b" " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest()


def git_tree_body(entries: list[dict[str, Any]]) -> bytes:
    # Git sorts directories as though their names ended in '/'. Sorting full
    # path strings, or retaining the leading 0 in 040000, hashes the wrong tree.
    ordered = sorted(entries, key=lambda e: e["path"].rsplit("/", 1)[-1].encode("utf-8")
                     + (b"/" if e["type"] == "tree" else b""))
    return b"".join(("40000" if e["mode"] == "040000" else e["mode"]).encode("ascii")
                    + b" " + e["path"].rsplit("/", 1)[-1].encode("utf-8") + b"\0"
                    + bytes.fromhex(e["sha"]) for e in ordered)


def verify_git_item(item: dict[str, Any]) -> dict[str, Any]:
    need(type(item) is dict and type(item.get("repository")) is str
         and REPO.fullmatch(item["repository"]) is not None, "GH_ID")
    need(item.get("content_bytes_verified") is False and item.get("runtime_verified") is False, "GH_SCOPE_AMPLIFIED")
    need(type(item.get("complete")) is bool and type(item.get("tree_complete")) is bool, "GH_COMPLETE_TYPE")
    if not item["tree_complete"]:
        need(item["complete"] is False and item.get("file_count") is None, "GH_PARTIAL_PROMOTED")
        need(item.get("entries") == [] and bool(item.get("blockers")), "GH_UNSUPPORTED_PARTIAL")
        return {"id": item["repository"], "revision": item.get("revision"), "files": None,
                "verified_tree_objects": 0, "metadata_complete": False,
                "content_review": "NOT_PERFORMED", "runtime": "NOT_VERIFIED"}
    need(digest(item.get("revision"), SHA1) and digest(item.get("revision_after"), SHA1), "GH_REVISION")
    need(digest(item.get("tree_sha"), SHA1), "GH_ROOT_TREE")
    need(type(item.get("default_branch")) is str and bool(item["default_branch"]), "GH_DEFAULT_BRANCH")
    entries = item.get("entries")
    need(type(entries) is list and len(entries) <= MAX_ENTRIES, "GH_ENTRIES")
    seen: set[str] = set()
    directories = {"": item["tree_sha"]}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = symlinks = submodules = 0
    for row in entries:
        need(type(row) is dict, "GH_ENTRY_OBJECT")
        name = safe_path(row.get("path"))
        need(name not in seen, "GH_DUPLICATE_PATH")
        seen.add(name)
        mode = row.get("mode")
        need(type(mode) is str and mode in MODE_TYPES and row.get("type") == MODE_TYPES[mode], "GH_MODE")
        need(digest(row.get("sha"), SHA1), "GH_OBJECT_ID")
        parent = name.rpartition("/")[0]
        children[parent].append(row)
        if mode == "040000":
            directories[name] = row["sha"]
        elif row["type"] == "blob":
            need(integer(row.get("size")), "GH_BLOB_SIZE")
            files += 1
            symlinks += int(mode == "120000")
        else:
            submodules += 1
    need(set(children).issubset(directories), "GH_PARENT_MISSING")
    for directory, expected in directories.items():
        need(git_object("tree", git_tree_body(children.get(directory, []))) == expected, "GH_TREE_HASH")
    need(integer(item.get("file_count")) and item["file_count"] == files, "GH_FILE_COUNT")
    need(integer(item.get("symlink_count")) and item["symlink_count"] == symlinks, "GH_SYMLINK_COUNT")
    need(integer(item.get("submodule_count")) and item["submodule_count"] == submodules, "GH_SUBMODULE_COUNT")
    need(item.get("tree_merkle_verified") is True, "GH_TREE_DECLARATION")
    if item["complete"]:
        need(item["revision"] == item["revision_after"] and item.get("blockers") == [], "GH_MOVED_OR_BLOCKED")
    else:
        need(bool(item.get("blockers")), "GH_UNEXPLAINED_PARTIAL")
    return {"id": item["repository"], "revision": item["revision"], "files": files,
            "verified_tree_objects": len(directories), "metadata_complete": item["complete"],
            "symlinks_not_followed": symlinks, "submodules_not_followed": submodules,
            "content_review": "NOT_PERFORMED", "runtime": "NOT_VERIFIED"}


def verify_hf_item(item: dict[str, Any], family: str) -> dict[str, Any]:
    need(type(item) is dict and type(item.get("repo_id")) is str
         and HF_ID.fullmatch(item["repo_id"]) is not None, "HF_ID")
    need(item.get("repo_type_endpoint") == family, "HF_WRONG_FAMILY")
    need(item.get("content_bytes_verified") is False and item.get("runtime_verified") is False, "HF_SCOPE_AMPLIFIED")
    need(type(item.get("complete")) is bool and type(item.get("tree_complete")) is bool, "HF_COMPLETE_TYPE")
    if not item["tree_complete"]:
        need(item["complete"] is False and item.get("file_count") is None
             and item.get("entries") == [] and bool(item.get("blockers")), "HF_UNSUPPORTED_PARTIAL")
        return {"id": item["repo_id"], "family": family, "revision": item.get("revision"),
                "files": None, "metadata_complete": False, "redacted_content_ids": None,
                "content_review": "NOT_PERFORMED", "runtime": "NOT_VERIFIED"}
    need(digest(item.get("revision"), SHA1) and digest(item.get("revision_after"), SHA1), "HF_REVISION")
    entries = item.get("entries")
    need(type(entries) is list and len(entries) <= MAX_ENTRIES, "HF_ENTRIES")
    seen: set[str] = set()
    dirs = {""}
    parents: set[str] = set()
    files = redacted = 0
    for row in entries:
        need(type(row) is dict, "HF_ENTRY_OBJECT")
        path = safe_path(row.get("path"))
        need(path not in seen, "HF_DUPLICATE_PATH")
        seen.add(path)
        parents.add(path.rpartition("/")[0])
        need(type(row.get("type")) is str and row["type"] in {"file", "directory"}
             and digest(row.get("oid"), SHA1), "HF_OBJECT_ID")
        need(integer(row.get("size")), "HF_SIZE")
        state, lfs = row.get("lfs_identity_state"), row.get("lfs_oid")
        need(type(state) is str and state in {"NOT_REPORTED", "OBSERVED", "REDACTED"}, "HF_LFS_STATE")
        if row["type"] == "directory":
            dirs.add(path)
            need(state == "NOT_REPORTED" and lfs is None, "HF_DIRECTORY_LFS")
        else:
            files += 1
            if state == "OBSERVED":
                need(digest(lfs, SHA256), "HF_LFS_HASH")
            elif state == "REDACTED":
                need(lfs is None, "HF_REDACTED_HASH_INVENTED")
                redacted += 1
            else:
                need(lfs is None, "HF_UNREPORTED_HASH")
    need(parents.issubset(dirs), "HF_PARENT_MISSING")
    need(integer(item.get("file_count")) and files == item["file_count"], "HF_FILE_COUNT")
    need(integer(item.get("redacted_lfs_file_count"))
         and redacted == item["redacted_lfs_file_count"], "HF_REDACTION_COUNT")
    need(type(item.get("file_metadata_identity_complete")) is bool, "HF_IDENTITY_COMPLETE_TYPE")
    # This checks the normalized census contract; gating permission itself is
    # not carried by this schema and is NOT independently verified here.
    if redacted:
        need(item["file_metadata_identity_complete"] is False and item["complete"] is False
             and "HF_LFS_IDENTITY_REDACTED" in item.get("blockers", []), "HF_REDACTION_PROMOTED")
    else:
        need(item["file_metadata_identity_complete"] is True, "HF_IDENTITY_INCONSISTENT")
    if item["complete"]:
        need(item["revision"] == item["revision_after"] and item.get("blockers") == [], "HF_MOVED_OR_BLOCKED")
    else:
        need(bool(item.get("blockers")), "HF_UNEXPLAINED_PARTIAL")
    return {"id": item["repo_id"], "family": family, "revision": item["revision"],
            "files": files, "metadata_complete": item["complete"], "redacted_content_ids": redacted,
            "stage_reported": item.get("runtime_stage_reported"),
            "content_review": "NOT_PERFORMED", "runtime": "NOT_VERIFIED"}


def verify_receipt(receipt: dict[str, Any], lane: str, now: datetime) -> dict[str, Any]:
    need(type(now) is datetime and now.tzinfo is not None, "NOW_TIME_ZONE")
    need(type(receipt) is dict and receipt.get("schema") == SCHEMA
         and receipt.get("scope") == SCOPE and receipt.get("lane") == lane, "RECEIPT_SCHEMA_SCOPE")
    need(all(receipt.get(key) is False for key in FALSE_FIELDS), "RECEIPT_SCOPE_AMPLIFIED")
    need(type(receipt.get("source_content_files_read")) is int
         and receipt["source_content_files_read"] == 0, "CONTENT_READ_CLAIM")
    need(digest(receipt.get("source_revision"), SHA1)
         and digest(receipt.get("observer_sha256"), SHA256), "OBSERVER_IDENTITY")
    need(type(receipt.get("complete")) is bool and type(receipt.get("blockers")) is list, "RECEIPT_COMPLETION_TYPE")
    start, end = window(receipt, now)
    populations = receipt.get("populations")
    expected = {"github"} if lane == "github" else {"models", "datasets", "spaces", "kernels"}
    need(type(populations) is dict and set(populations) == expected, "POPULATION_COVERAGE")
    output: dict[str, Any] = {}
    for family, population in sorted(populations.items()):
        need(type(population) is dict and type(population.get("items")) is list, "POPULATION_OBJECT")
        items = population["items"]
        need(len(items) <= MAX_ITEMS, "POPULATION_BOUND")
        need(type(population.get("complete")) is bool
             and type(population.get("membership_stable")) is bool
             and type(population.get("blockers")) is list, "POPULATION_FLAGS")
        rows, seen = [], set()
        for item in items:
            first, last = window(item, now)
            need(start <= first <= last <= end, "ITEM_OUTSIDE_WINDOW")
            row = verify_git_item(item) if lane == "github" else verify_hf_item(item, family)
            need(row["id"] not in seen, "DUPLICATE_ASSET")
            seen.add(row["id"])
            rows.append(row)
        known = sum(row["files"] for row in rows if row["files"] is not None)
        complete_items = sum(row["metadata_complete"] for row in rows)
        need(integer(population.get("items_observed")) and population["items_observed"] == len(rows), "POPULATION_ITEM_COUNT")
        need(integer(population.get("items_complete")) and population["items_complete"] == complete_items, "POPULATION_COMPLETE_COUNT")
        need(integer(population.get("known_file_subtotal")) and population["known_file_subtotal"] == known, "POPULATION_FILE_SUBTOTAL")
        valid_complete = (complete_items == len(rows) and population["membership_stable"]
                          and population["blockers"] == [])
        need(population["complete"] == valid_complete, "POPULATION_COMPLETE_INCONSISTENT")
        total = population.get("complete_scope_file_count")
        need((valid_complete and integer(total) and total == known)
             or (not valid_complete and total is None), "PARTIAL_TOTAL_PROMOTED")
        output[family] = {"items_observed": len(rows), "items_metadata_complete": complete_items,
                          "metadata_complete": valid_complete, "known_file_subtotal": known,
                          "complete_scope_file_count": total,
                          "verified_git_tree_objects": sum(row.get("verified_tree_objects", 0) for row in rows),
                          "items": rows}
    expected_complete = all(p["metadata_complete"] for p in output.values()) and receipt["blockers"] == []
    need(receipt["complete"] == expected_complete, "RECEIPT_COMPLETE_INCONSISTENT")
    expected_status = "FILE_METADATA_OBSERVED_NOT_QUALIFIED" if expected_complete else "PARTIAL_OR_UNAVAILABLE"
    need(receipt.get("status") == expected_status, "RECEIPT_STATUS_INCONSISTENT")
    age = max(0.0, (now - end).total_seconds())
    return {"schema": "szl.payload-second-reader/v1", "lane": lane,
            "scope": SCOPE, "status": expected_status,
            "original_started_at": receipt["started_at"], "original_finished_at": receipt["finished_at"],
            "checked_at": now.isoformat(), "historical_age_seconds": round(age, 3),
            "freshness": "HISTORICAL_REQUIRES_REOBSERVATION" if age > 86400 else "RECORDED_NOT_LIVE",
            "observer_source_revision": receipt["source_revision"],
            "observer_byte_hash": receipt["observer_sha256"], "populations": output,
            "remote_membership_independently_queried": False, "signatures_verified": False,
            "source_content_files_read": 0, "semantic_review_complete": False,
            "runtime_verified": False, "production_authorization": False,
            "private_assets_collections_buckets": "OUTSIDE_INPUT_SCHEMA_NOT_ZERO"}


def consistency(payload: dict[str, Any]) -> dict[str, Any]:
    """Descriptive repeatability metrics on a declared, fixed-k task matrix.

    Null is an unobserved run, not success or failure. Full metrics are withheld
    if any scheduled result is missing. No independence or population claim.
    """
    need(type(payload) is dict and set(payload) == {"k", "task_ids", "runs"}, "CONSISTENCY_SCHEMA")
    k, ids, runs = payload["k"], payload["task_ids"], payload["runs"]
    need(integer(k, 1, 100) and type(ids) is list and 0 < len(ids) <= 10_000, "CONSISTENCY_BOUND")
    need(all(type(x) is str and 0 < len(x) <= 128 for x in ids)
         and len(set(ids)) == len(ids), "CONSISTENCY_TASK_IDS")
    need(type(runs) is list and len(runs) <= len(ids) * k, "CONSISTENCY_RUNS")
    values: dict[tuple[str, int], bool | None] = {}
    for row in runs:
        need(type(row) is dict and set(row) == {"task_id", "repeat", "success"}, "CONSISTENCY_ROW")
        task, repeat, outcome = row["task_id"], row["repeat"], row["success"]
        need(type(task) is str and task in ids and integer(repeat, 0, k - 1), "CONSISTENCY_ID")
        need(outcome is None or type(outcome) is bool, "CONSISTENCY_OUTCOME")
        key = (task, repeat)
        need(key not in values, "CONSISTENCY_DUPLICATE")
        values[key] = outcome
    complete = all(values.get((task, repeat)) is not None for task in ids for repeat in range(k))
    observed = [value for value in values.values() if value is not None]
    result = {"k": k, "tasks": len(ids), "scheduled_runs": len(ids) * k,
              "observed_runs": len(observed), "missing_runs": len(ids) * k - len(observed),
              "mean_at_k": None, "pass_power_k": None, "pass_at_k_empirical": None,
              "consistency_gap": None, "complete": complete,
              "meaning": "FIXED_OBSERVED_REPEATS_NOT_IID_ESTIMATE_OR_PRODUCTION_APPROVAL"}
    if complete:
        mean = sum(observed) / len(observed)
        every = sum(all(values[(t, j)] for j in range(k)) for t in ids) / len(ids)
        any_success = sum(any(values[(t, j)] for j in range(k)) for t in ids) / len(ids)
        result.update(mean_at_k=mean, pass_power_k=every, pass_at_k_empirical=any_success,
                      consistency_gap=mean - every)
    return result


def review_queue(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One follow-up row per input asset; presence never becomes semantic review."""
    rows = []
    for lane in ("github", "huggingface"):
        for family, population in report[lane]["populations"].items():
            for item in population["items"]:
                rows.append({"id": item["id"], "kind": family, "observed_revision": item["revision"],
                             "recorded_files": item["files"], "content_review": "PENDING",
                             "runtime_verification": "PENDING", "automatic_execution_allowed": False})
    return rows


def summary_html(report: dict[str, Any]) -> str:
    """Offline private report for the operator, not a new product UI or publisher."""
    table = []
    for row in review_queue(report):
        table.append("<tr>" + "".join("<td>" + html.escape(str(row[k]), quote=True) + "</td>"
                     for k in ("kind", "id", "observed_revision", "recorded_files", "content_review")) + "</tr>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
            'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
            '<title>SZL private audit planning</title><style>'
            'body{font:16px/1.5 system-ui;margin:1rem;max-width:100%}table{border-collapse:collapse;width:100%}'
            'th,td{padding:.5rem;text-align:left;border-bottom:1px solid #888;overflow-wrap:anywhere}'
            '.table{overflow:auto}a{padding:.75rem;display:inline-block}a:focus{outline:3px solid}'
            '@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}'
            '</style></head><body><a href="#inventory">Skip to inventory</a>'
            '<h1>Recorded inventory is not operational qualification</h1>'
            '<p>Private operator planning only. These historical public receipts verify tree/metadata '
            'consistency, not current membership, source content, private coverage, model behavior or deployment.</p>'
            '<div class="table" id="inventory" tabindex="-1"><table><caption>Pending review by recorded asset</caption>'
            '<thead><tr><th>Kind</th><th>Asset</th><th>Recorded revision</th><th>Files</th><th>Content review</th></tr></thead>'
            '<tbody>' + "".join(table) + '</tbody></table></div></body></html>')


def write_new(path: Path, raw: bytes) -> None:
    # Caller selects an operator-owned private directory. Never overwrite.
    with path.open("xb") as stream:
        stream.write(raw)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows ACL administration remains the operator's responsibility.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    verify = actions.add_parser("verify")
    for lane in ("github", "huggingface"):
        verify.add_argument("--" + lane + "-zip", type=Path, required=True)
        verify.add_argument("--" + lane + "-sha256", required=True)
    verify.add_argument("--out-dir", type=Path, required=True)
    repeat = actions.add_parser("consistency")
    repeat.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "consistency":
            with args.input.open("rb") as stream:
                value = strict_json(stream.read(MAX_JSON + 1))
            print(canonical(consistency(value)).decode())
            return 0 if consistency(value)["complete"] else 2
        now = datetime.now(timezone.utc)
        report: dict[str, Any] = {"schema": "szl.payload-audit-reconciliation/v1", "production_authorization": False}
        for lane in ("github", "huggingface"):
            receipt, archive = read_native_archive(getattr(args, lane + "_zip"), getattr(args, lane + "_sha256"))
            report[lane] = verify_receipt(receipt, lane, now)
            report[lane]["archive"] = archive
        # Same native observer/source is required for this paired reconstruction.
        need(report["github"]["observer_source_revision"] == report["huggingface"]["observer_source_revision"]
             and report["github"]["observer_byte_hash"] == report["huggingface"]["observer_byte_hash"], "PAIR_SOURCE_DIFFERS")
        # Do not create directories in the product or overwrite previous evidence.
        args.out_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        write_new(args.out_dir / "reconciliation.json", canonical(report) + b"\n")
        write_new(args.out_dir / "review-queue.json", canonical(review_queue(report)) + b"\n")
        write_new(args.out_dir / "operator-review.html", summary_html(report).encode("utf-8"))
        complete = all(report[lane]["status"] == "FILE_METADATA_OBSERVED_NOT_QUALIFIED"
                       for lane in ("github", "huggingface"))
        print(canonical({"state": "METADATA_CONSISTENT" if complete else "METADATA_CONSISTENT_WITH_RETAINED_GAPS",
                         "production_authorization": False}).decode())
        return 0 if complete else 2
    except ContractError as exc:
        print(canonical({"state": "REJECTED", "diagnostic": str(exc), "production_authorization": False}).decode(), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError):
        print('{"state":"REJECTED","diagnostic":"LOCAL_IO_OR_SHAPE_ERROR","production_authorization":false}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
