#!/usr/bin/env python3
"""Inspect a digest-bound release ZIP without extracting, loading, or publishing.

A consistent receipt is NOT independent Hub readback, signature verification,
current branch authorization, or permission to run a model. All inputs are data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

SOURCE_REPO = "szl-holdings/szl-invariants"
PUBLISHER_REPO = "szl-holdings/szl-forge"
HUB_REPO = "SZLHOLDINGS/szl-invariants"
WORKFLOW = ".github/workflows/publish-szl-invariants.yml"
AUTH_FILE = "invariants-release-authorization.json"
PUBLICATION_FILE = "invariants-publication.json"
ARCHIVE_LIMIT = 2 * 1024 * 1024
MEMBER_LIMIT = 512 * 1024
SOURCE_FILES = frozenset(
    f"{variant}/szl_invariants/{name}"
    for variant in ("build/torch-universal", "torch-ext")
    for name in ("__init__.py", "metadata.json")
)
STAGED_FILES = frozenset(
    f"build/{variant}/{name}"
    for variant in ("torch-universal", "torch-cpu")
    for name in ("szl_invariants/__init__.py", "szl_invariants/metadata.json", "metadata.json")
)


class EvidenceError(ValueError):
    """Carries only fixed diagnostic codes, never supplied receipt content."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def sha(value: Any, length: int = 40) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def object_value(value: Any, code: str) -> dict[str, Any]:
    need(type(value) is dict, code)
    return value


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def reject_constant(_value: str) -> None:
    raise EvidenceError("NONFINITE_JSON_VALUE")


def finite_float(value: str) -> float:
    number = float(value)
    need(math.isfinite(number), "NONFINITE_JSON_VALUE")
    return number


def parse_json(raw: bytes) -> dict[str, Any]:
    try:
        return object_value(json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                                       parse_constant=reject_constant, parse_float=finite_float), "JSON_NOT_OBJECT")
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise EvidenceError("INVALID_JSON") from None


def read_archive(path: Path, expected_digest: str) -> dict[str, dict[str, Any]]:
    """Hash and parse the same bounded bytes; never extract an archive member."""
    need(sha(expected_digest, 64), "INVALID_EXPECTED_ARCHIVE_DIGEST")
    with path.open("rb") as handle:
        raw = handle.read(ARCHIVE_LIMIT + 1)
    need(len(raw) <= ARCHIVE_LIMIT, "ARCHIVE_TOO_LARGE")
    need(hashlib.sha256(raw).hexdigest() == expected_digest, "ARCHIVE_DIGEST_MISMATCH")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            need(len(entries) <= 2, "UNEXPECTED_ARCHIVE_MEMBERS")
            names = [item.filename for item in entries]
            need(len(set(names)) == len(names), "DUPLICATE_ARCHIVE_MEMBER")
            need(set(names) <= {AUTH_FILE, PUBLICATION_FILE}, "UNEXPECTED_ARCHIVE_MEMBERS")
            need(PUBLICATION_FILE in names, "MISSING_PUBLICATION_REPORT")
            need(AUTH_FILE in names, "MISSING_AUTHORIZATION_REPORT")
            reports = {}
            for entry in entries:
                need(entry.orig_filename == entry.filename, "AMBIGUOUS_ARCHIVE_NAME")
                kind = stat.S_IFMT(entry.external_attr >> 16)
                need(kind in (0, stat.S_IFREG) and not entry.is_dir(), "NONREGULAR_ARCHIVE_MEMBER")
                need(not entry.flag_bits & 1, "ENCRYPTED_ARCHIVE_MEMBER")
                need(entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                     "UNSUPPORTED_ARCHIVE_COMPRESSION")
                need(0 < entry.file_size <= MEMBER_LIMIT, "MEMBER_SIZE_INVALID")
                with archive.open(entry) as handle:
                    payload = handle.read(MEMBER_LIMIT + 1)
                need(len(payload) == entry.file_size and len(payload) <= MEMBER_LIMIT,
                     "MEMBER_SIZE_INVALID")
                reports[entry.filename] = parse_json(payload)
            return reports
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
        raise EvidenceError("INVALID_ARCHIVE") from None


def validate_authorization(auth: dict[str, Any], source: str, publisher: str) -> None:
    need(auth.get("schema") == "szl.invariants-release-authorization/v1" and
         auth.get("status") == "AUTHORIZED_PROTECTED_MAIN", "AUTHORIZATION_NOT_SUCCESSFUL")
    for field, repo, revision in (("source", SOURCE_REPO, source),
                                   ("publisher", PUBLISHER_REPO, publisher)):
        row = object_value(auth.get(field), "INVALID_AUTHORIZATION_IDENTITY")
        need(row.get("repository") == repo and row.get("revision") == revision and
             row.get("protected_main") == revision and
             row.get("branch_protection_observed") is True, "AUTHORIZATION_IDENTITY_MISMATCH")
    source_row = auth["source"]
    need(source_row.get("signature_verified") is True and
         source_row.get("signature_reason") == "valid", "SIGNATURE_CLAIM_MISSING")
    try:
        timestamp = datetime.fromisoformat(auth["authorized_at"])
        need(timestamp.tzinfo is not None, "INVALID_AUTHORIZATION_TIMESTAMP")
    except (ValueError, TypeError, KeyError):
        raise EvidenceError("INVALID_AUTHORIZATION_TIMESTAMP") from None
    checks = source_row.get("checks")
    need(type(checks) is list and len(checks) > 0, "REQUIRED_CHECK_EVIDENCE_MISSING")
    names = []
    for row in checks:
        row = object_value(row, "INVALID_CHECK_EVIDENCE")
        need(type(row.get("app_id")) is int and row["app_id"] > 0 and
             type(row.get("run_id")) is int and row["run_id"] > 0 and
             row.get("status") == "completed" and row.get("conclusion") == "success",
             "CHECK_NOT_SUCCESSFUL")
        name = row.get("name")
        need(isinstance(name, str) and bool(name), "INVALID_CHECK_EVIDENCE")
        names.append(name)
    need("verify canonical kernel" in names, "REQUIRED_CHECK_EVIDENCE_MISSING")
    enforcement = object_value(source_row.get("required_check_enforcement"), "MISSING_CHECK_POLICY")
    basis = enforcement.get("basis")
    need(type(basis) is list and bool(basis) and all(
        b in ("classic_branch_protection", "effective_branch_rules") for b in basis), "MISSING_CHECK_POLICY")
    policies = enforcement.get("required_checks_observed")
    contexts = enforcement.get("required_contexts_observed")
    need(type(policies) is list and bool(policies) and type(contexts) is list,
         "MISSING_CHECK_POLICY")
    bindings = set()
    for policy in policies:
        policy = object_value(policy, "INVALID_CHECK_POLICY")
        need("app_id" in policy, "INVALID_CHECK_POLICY")
        name, app = policy.get("context"), policy.get("app_id")
        need(isinstance(name, str) and bool(name) and
             (app is None or (type(app) is int and (app == -1 or app > 0))),
             "INVALID_CHECK_POLICY")
        need((name, app) not in bindings, "DUPLICATE_CHECK_BINDING")
        bindings.add((name, app))
    need(all(isinstance(name, str) for name in contexts) and
         len(set(contexts)) == len(contexts) and
         set(contexts) == {name for name, _ in bindings} and
         "verify canonical kernel" in contexts, "CHECK_POLICY_MISMATCH")
    seen_bindings = set()
    for row in checks:
        need("required_app_id" in row, "INVALID_CHECK_BINDING")
        name, required = row["name"], row.get("required_app_id")
        need(required is None or (type(required) is int and (required == -1 or required > 0)),
             "INVALID_CHECK_BINDING")
        key = (name, required)
        need(key in bindings and key not in seen_bindings, "CHECK_BINDING_MISMATCH")
        need(required in (None, -1) or row["app_id"] == required, "CHECK_APP_MISMATCH")
        seen_bindings.add(key)
    need(seen_bindings == bindings, "REQUIRED_CHECK_EVIDENCE_MISSING")


def validate_files(report: dict[str, Any]) -> None:
    rows = report.get("declared_files")
    need(type(rows) is list and len(rows) == len(SOURCE_FILES), "DECLARED_FILE_SET_MISMATCH")
    by_path = {}
    for row in rows:
        row = object_value(row, "INVALID_DECLARED_FILE")
        path = row.get("path")
        need(isinstance(path, str) and path in SOURCE_FILES and path not in by_path,
             "DECLARED_FILE_SET_MISMATCH")
        need(sha(row.get("sha256"), 64) and type(row.get("bytes")) is int and
             row["bytes"] > 0, "INVALID_DECLARED_FILE")
        by_path[path] = row
    for name in ("__init__.py", "metadata.json"):
        a = by_path[f"build/torch-universal/szl_invariants/{name}"]
        b = by_path[f"torch-ext/szl_invariants/{name}"]
        need((a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"]), "SOURCE_VARIANT_MISMATCH")
    digest = hashlib.sha256()
    for path, row in sorted(by_path.items()):
        digest.update(path.encode("utf-8") + b"\0" + row["sha256"].encode("ascii") + b"\n")
    need(report.get("artifact_tree_sha256") == digest.hexdigest(), "ARTIFACT_TREE_MISMATCH")


def branches(value: Any) -> dict[str, str]:
    row = object_value(value, "INVALID_KERNEL_BRANCHES")
    need(set(row) == {"main", "v1"} and all(sha(v) for v in row.values()),
         "INVALID_KERNEL_BRANCHES")
    return row


def validate_reports(reports: dict[str, dict[str, Any]], *, source: str, publisher: str,
                     run_id: str, run_attempt: str) -> None:
    need(sha(source) and sha(publisher), "INVALID_EXPECTED_REVISION")
    need(all(isinstance(v, str) and re.fullmatch(r"[1-9][0-9]*", v) is not None
             for v in (run_id, run_attempt)), "INVALID_EXPECTED_RUN")
    auth, report = reports[AUTH_FILE], reports[PUBLICATION_FILE]
    validate_authorization(auth, source, publisher)
    need(report.get("schema") == "szl.invariants-publication-report/v1" and
         report.get("mode") == "PUBLISH" and
         report.get("status") == "PUBLISHED_AND_EXACT_READBACK_VERIFIED",
         "PUBLICATION_NOT_SUCCESSFUL")
    need(report.get("repo_id") == HUB_REPO and report.get("source_repository") == SOURCE_REPO and
         report.get("source_revision") == source, "PUBLICATION_SOURCE_MISMATCH")
    embedded = object_value(report.get("authorization"), "AUTHORIZATION_REPORT_MISMATCH")
    need(json.dumps(embedded, sort_keys=True) == json.dumps(auth, sort_keys=True),
         "AUTHORIZATION_REPORT_MISMATCH")
    identity = object_value(report.get("publisher"), "INVALID_PUBLISHER_IDENTITY")
    expected = {
        "repository": PUBLISHER_REPO, "revision": publisher, "workflow_path": WORKFLOW,
        "workflow_ref": f"{PUBLISHER_REPO}/{WORKFLOW}@refs/heads/main",
        "workflow_url": f"https://github.com/{PUBLISHER_REPO}/blob/{publisher}/{WORKFLOW}",
        "run_id": run_id, "run_attempt": run_attempt,
        "run_url": f"https://github.com/{PUBLISHER_REPO}/actions/runs/{run_id}",
    }
    need(identity == expected, "PUBLISHER_RUN_MISMATCH")
    validate_files(report)
    targets = object_value(report.get("targets"), "INVALID_TARGETS")
    need(set(targets) == {"model", "kernel"}, "TARGET_SET_MISMATCH")
    observed = object_value(report.get("observed_before"), "INVALID_PRIOR_OBSERVATION")
    need(set(observed) == {"model", "kernel"}, "INVALID_PRIOR_OBSERVATION")
    for name, success in (("model", "EXACT_READBACK_VERIFIED"),
                           ("kernel", "V1_EXACT_READBACK_VERIFIED")):
        target = object_value(targets[name], "INVALID_TARGETS")
        before = object_value(observed[name], "INVALID_PRIOR_OBSERVATION")
        need(target.get("status") == success, "TARGET_READBACK_INCOMPLETE")
        need(sha(target.get("revision_before")) and sha(target.get("revision_after")),
             "TARGET_REVISION_NOT_IMMUTABLE")
        need(before.get("revision") == target["revision_before"], "TARGET_PARENT_MISMATCH")
    model, kernel = targets["model"], targets["kernel"]
    need(model.get("publication_interface") == "huggingface_hub.create_commit" and
         kernel.get("publication_interface") == "kernel-builder" and
         kernel.get("publication_interface_version") == "0.17.0-dev0" and
         kernel.get("publication_interface_source_revision") ==
         "633246310320d85def0c67d62c7912fd444a842f", "PUBLICATION_INTERFACE_MISMATCH")
    need(kernel.get("branch") == "v1" and observed["kernel"].get("branch") == "v1",
         "KERNEL_BRANCH_MISMATCH")
    prior = branches(kernel.get("branches_before"))
    after = branches(kernel.get("branches_after"))
    verified = branches(kernel.get("verified_branches"))
    need(prior == branches(kernel.get("parents_revalidated_before_upload")) ==
         branches(observed["kernel"].get("branches")), "KERNEL_PARENT_MISMATCH")
    need(after == verified and after["main"] == prior["main"] and
         prior["v1"] == kernel["revision_before"] and after["v1"] == kernel["revision_after"],
         "KERNEL_READBACK_MISMATCH")
    staged = kernel.get("staged_files")
    need(type(staged) is list and len(staged) == len(STAGED_FILES) and
         all(isinstance(p, str) for p in staged) and set(staged) == STAGED_FILES,
         "KERNEL_STAGED_FILE_SET_MISMATCH")


def inspect_archive(path: Path, *, expected_digest: str, source: str, publisher: str,
                    run_id: str, run_attempt: str) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "schema": "szl.invariants-release-evidence-inspection/v1",
        "status": "REJECTED", "reason": None, "evidence_consistent": False,
        "publication_verified": False, "independent_hub_readback": False,
        "signature_independently_verified": False, "repository_mutation": "NOT_ATTEMPTED",
    }
    try:
        reports = read_archive(path, expected_digest)
        validate_reports(reports, source=source, publisher=publisher,
                         run_id=run_id, run_attempt=run_attempt)
        verdict.update(status="EVIDENCE_CONSISTENT_REVIEW_REQUIRED", evidence_consistent=True)
    except EvidenceError as exc:
        verdict["reason"] = str(exc)
    except Exception:
        verdict["reason"] = "EVIDENCE_UNREADABLE_OR_MALFORMED"
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--publisher-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    args = parser.parse_args(argv)
    verdict = inspect_archive(args.archive, expected_digest=args.archive_sha256,
                              source=args.source_revision, publisher=args.publisher_revision,
                              run_id=args.run_id, run_attempt=args.run_attempt)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["evidence_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
