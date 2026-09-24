"""Closed-set, immutable-source Hugging Face provenance reconciliation.

``validate`` is offline and permits a candidate working tree. ``check`` uses an
immutable clean candidate commit and only reads Hub state. ``publish`` additionally
requires the exact current canonical main commit before every provider write.
Receipts are exclusive-create, fsynced JSONL journals, never silently resumed.
No model is loaded, trained, deleted, or represented as independently approved.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from tools import kernel_provenance_evidence as evidence
except ModuleNotFoundError:
    import kernel_provenance_evidence as evidence

MANIFEST = "publishing/kernel-provenance-reconciliation.json"
BUNDLE = "publishing/kernel-provenance"
CANONICAL_REPOSITORY = "szl-holdings/szl-forge"
ENDPOINT = "https://huggingface.co"
RELEASE_SCHEMA = "szl.kernel-provenance-release/v1"
DOCUMENT_STATUS = "ARTIFACT_STATE_RECONCILIATION"
FIRST_LIMIT = ("This document reconciles artifact state; it does not establish "
               "training, evaluation or runtime readiness.")
MANIFEST_KEYS = {"schema", "bundle", "snapshots_sha256", "entries"}
ENTRY_KEYS = {"id", "repository_type", "expected_parent", "destination",
              "source_path", "original_sha256", "output_sha256"}
require = evidence.require


def blob_oid(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def document_for(snapshot):
    document = evidence.proposal_document(snapshot)
    document["status"] = DOCUMENT_STATUS
    document["limits"][0] = FIRST_LIMIT
    return document


@dataclass(frozen=True)
class Target:
    repo_id: str
    expected_parent: str
    original: bytes
    output: bytes
    files: dict


@dataclass(frozen=True)
class Release:
    manifest_sha256: str
    snapshots_sha256: str
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class LiveState:
    revision: str
    files: dict
    document: bytes


def file_identities(rows):
    """Normalize a fully exhausted tree; compare every file, not only metadata."""
    result = {}
    seen = set()
    for row in rows:
        name = evidence.relative_path(row["path"])
        require(name not in seen, "duplicate live tree path")
        seen.add(name)
        require(row["type"] in ("file", "directory"), "unsupported tree entry")
        if row["type"] == "directory":
            continue
        oid = evidence.digest(row["oid"], 40)
        size = row["size"]
        require(type(size) is int and size >= 0, "invalid live file size")
        result[name] = {"oid": oid, "size": size}
    require(evidence.DESTINATION in result, "live provenance file missing")
    require(not any(name.lower().endswith(evidence.WEIGHT_SUFFIXES) for name in result),
            "weight-like artifact present; absence-only repair refused")
    return result


def validate_release(read_bytes):
    """Pure source validation: callback returns bytes for a repository-relative path."""
    manifest_bytes = read_bytes(MANIFEST)
    manifest = evidence.parse(manifest_bytes)
    require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS, "unexpected manifest fields")
    require(manifest["schema"] == RELEASE_SCHEMA and manifest["bundle"] == BUNDLE,
            "unsupported release or bundle")
    snapshots_bytes = read_bytes(BUNDLE + "/snapshots.json")
    require(evidence.sha(snapshots_bytes) == evidence.digest(manifest["snapshots_sha256"], 64),
            "snapshot source bytes changed")
    snapshots = evidence.parse(snapshots_bytes)
    require(isinstance(snapshots, list) and len(snapshots) == len(evidence.TARGETS), "closed snapshot set required")
    require({item["id"] for item in snapshots} == set(evidence.TARGETS), "expanded or duplicate snapshot targets")
    by_id = {item["id"]: item for item in snapshots}
    entries = manifest["entries"]
    require(isinstance(entries, list) and len(entries) == len(evidence.TARGETS), "all eight release targets required")
    require({item["id"] for item in entries} == set(evidence.TARGETS), "expanded or duplicate release targets")
    targets = []
    for entry in sorted(entries, key=lambda item: item["id"]):
        require(set(entry) == ENTRY_KEYS, "unexpected release entry fields")
        repo_id = entry["id"]
        snapshot = by_id[repo_id]
        expected_document = document_for(snapshot)
        expected_source = BUNDLE + "/" + repo_id.split("/")[1] + "/" + evidence.DESTINATION
        require(entry["source_path"] == expected_source, "source path expansion refused")
        require(entry["repository_type"] == "model" and entry["destination"] == evidence.DESTINATION,
                "publication destination expansion refused")
        require(entry["expected_parent"] == snapshot["revision"], "release parent does not match evidence")
        output = read_bytes(expected_source)
        require(0 < len(output) <= evidence.MAX_DOCUMENT_BYTES, "replacement document size outside bound")
        require(evidence.sha(output) == evidence.digest(entry["output_sha256"], 64), "replacement bytes changed")
        require(evidence.canonical(evidence.parse(output)) == evidence.canonical(expected_document),
                "replacement claims differ from deterministic reconciliation")
        original = base64.b64decode(snapshot["original_bytes_base64"], validate=True)
        require(evidence.sha(original) == evidence.digest(entry["original_sha256"], 64), "original hash mismatch")
        targets.append(Target(repo_id, snapshot["revision"], original, output, file_identities(snapshot["tree"])))
    return Release(evidence.sha(manifest_bytes), evidence.sha(snapshots_bytes), tuple(targets))


def expected_output_files(target):
    files = dict(target.files)
    files[evidence.DESTINATION] = {"oid": blob_oid(target.output), "size": len(target.output)}
    return files


def classify(target, state):
    evidence.digest(state.revision, 40)
    if state.document == target.output and state.files == expected_output_files(target):
        return "ALREADY_APPLIED_EXACT"
    require(state.revision == target.expected_parent, "stale provider head; release must be reviewed again")
    require(state.document == target.original and state.files == target.files,
            "provider artifact tree or provenance differs from captured evidence")
    return "READY"


def preflight(release, backend):
    """Complete every read and reject any discrepancy before a first write."""
    states = []
    for target in release.targets:
        state = backend.inspect(target.repo_id)
        states.append((target, state, classify(target, state)))
    return states


class Journal:
    def __init__(self, path):
        # O_EXCL makes an existing, symlink, partial, or unknown receipt a refusal.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.handle = os.fdopen(descriptor, "wb", buffering=0)
        self.sequence = 0

    def append(self, event, **values):
        record = {"schema": "szl.kernel-provenance-journal/v1", "sequence": self.sequence,
                  "observed_at": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        self.handle.write(evidence.canonical(record) + b"\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.sequence += 1

    def close(self):
        self.handle.close()


def execute(release, backend, *, source_revision, receipt, publish=False, authority_check=None):
    """Injectable orchestration. No retry, rollback, deletion, or implicit publish."""
    evidence.digest(source_revision, 40)
    require(not publish or callable(authority_check), "publication authority check required")
    journal = Journal(receipt)
    attempted, verified, already_applied = 0, 0, 0
    phase = "preflight"
    try:
        journal.append("START", mode="publish" if publish else "check", source_revision=source_revision,
                       manifest_sha256=release.manifest_sha256, snapshots_sha256=release.snapshots_sha256,
                       targets=len(release.targets), independent_approval_claimed=False)
        if publish:
            authority_check()
        states = preflight(release, backend)
        for target, state, status in states:
            journal.append("PREFLIGHT", repository=target.repo_id, revision=state.revision, status=status,
                           output_sha256=evidence.sha(target.output))
        if publish:
            backend.authorize([target.repo_id for target in release.targets])
            journal.append("ALL_TARGET_WRITE_ACCESS_CHECKED", targets=len(release.targets))
            phase = "publication"
            for target, state, status in states:
                if status == "ALREADY_APPLIED_EXACT":
                    already_applied += 1
                    journal.append("ALREADY_APPLIED_EXACT", repository=target.repo_id, revision=state.revision,
                                   attribution="Observed exact artifact state; this run did not author that commit.")
                    continue
                # Refresh both authority and target immediately before committing.
                authority_check()
                current = backend.inspect(target.repo_id)
                require(classify(target, current) == "READY", "provider changed after all-target preflight")
                journal.append("WRITE_ATTEMPT", repository=target.repo_id, expected_parent=target.expected_parent,
                               destination=evidence.DESTINATION, output_sha256=evidence.sha(target.output))
                attempted += 1
                commit = evidence.digest(backend.commit(target, source_revision), 40)
                journal.append("COMMIT_RETURNED", repository=target.repo_id, revision=commit)
                readback = backend.inspect(target.repo_id, revision=commit)
                require(readback.revision == commit and classify(target, readback) == "ALREADY_APPLIED_EXACT",
                        "exact committed artifact readback failed")
                current = backend.inspect(target.repo_id)
                require(current.revision == commit and classify(target, current) == "ALREADY_APPLIED_EXACT",
                        "provider main moved or changed during committed readback")
                verified += 1
                journal.append("COMMIT_VERIFIED", repository=target.repo_id, revision=commit,
                               output_sha256=evidence.sha(target.output), all_other_files_unchanged=True,
                               commit_authorship_verified=False,
                               attribution="Returned commit artifacts verified; SDK may return preexisting identical state.")
            phase = "final_readback"
            for target in release.targets:
                final = backend.inspect(target.repo_id)
                require(classify(target, final) == "ALREADY_APPLIED_EXACT", "final provider artifact state changed")
                journal.append("FINAL_ARTIFACT_VERIFIED", repository=target.repo_id, revision=final.revision,
                               output_sha256=evidence.sha(target.output))
        result = {"status": "PUBLISHED_VERIFIED" if publish else "CHECKED_NOT_PUBLISHED",
                  "complete": True, "write_attempts": attempted, "commits_verified": verified,
                  "already_applied_exact": already_applied, "targets_checked": len(states),
                  "source_revision": source_revision, "training_verified": False,
                  "runtime_verified": False, "independent_approval_claimed": False,
                  "commit_authorship_verified": False}
        journal.append("COMPLETE", **result)
        return result
    except BaseException as exc:
        # Provider exception text may include request details; never journal it.
        journal.append("STOPPED", complete=False, phase=phase,
                       status="PARTIAL_OR_UNCERTAIN" if attempted else "REFUSED_BEFORE_WRITE",
                       write_attempts=attempted, commits_verified=verified, already_applied_exact=already_applied,
                       error_type=type(exc).__name__, automatic_retry=False)
        raise
    finally:
        journal.close()


class WorkingTreeSource:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, path):
        evidence.relative_path(path)
        candidate = self.root / path
        current = candidate
        while current != self.root:
            require(not current.is_symlink(), "symlink source refused")
            current = current.parent
        require(candidate.is_file() and candidate.resolve().is_relative_to(self.root), "source outside repository")
        return candidate.read_bytes()


class GitSource:
    def __init__(self, root, revision):
        self.root = Path(root).resolve()
        self.revision = evidence.digest(revision, 40)
        self.assert_clean()

    def command(self, *arguments):
        result = subprocess.run(["git", "-C", str(self.root), *arguments], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout

    def assert_clean(self):
        actual_root = Path(self.command("rev-parse", "--show-toplevel").decode().strip()).resolve()
        require(actual_root == self.root, "repository root must be exact")
        require(self.command("rev-parse", "HEAD").decode().strip() == self.revision, "HEAD differs from source revision")
        require(not self.command("status", "--porcelain", "--untracked-files=all"), "source checkout is not clean")
        environment_revision = os.environ.get("GITHUB_SHA")
        require(environment_revision is None or environment_revision == self.revision, "GITHUB_SHA differs from source revision")
        remote = self.command("remote", "get-url", "origin").decode().strip()
        require(remote in {"https://github.com/" + CANONICAL_REPOSITORY + suffix for suffix in ("", ".git")}
                | {"git@github.com:" + CANONICAL_REPOSITORY + ".git"}, "noncanonical origin refused")

    def read(self, path):
        evidence.relative_path(path)
        record = self.command("ls-tree", "-z", self.revision, "--", path)
        parts = record.split(b"\0")
        require(len(parts) == 2 and parts[-1] == b"", "missing or ambiguous source blob")
        metadata, actual_path = parts[0].split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        require(mode == "100644" and kind == "blob" and actual_path.decode("utf-8") == path,
                "source must be an ordinary immutable Git blob")
        raw = self.command("cat-file", "blob", oid)
        require(blob_oid(raw) == oid, "Git blob bytes do not match object identity")
        return raw

    def assert_publish_authority(self):
        self.assert_clean()
        self.assert_implementation_bound()
        # gh uses the configured GitHub identity without exposing token values.
        query = subprocess.run(["gh", "api", "repos/" + CANONICAL_REPOSITORY], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        repo = evidence.parse(query.stdout)
        require(repo.get("full_name") == CANONICAL_REPOSITORY and repo.get("default_branch") == "main",
                "canonical default branch differs")
        query = subprocess.run(["gh", "api", "repos/" + CANONICAL_REPOSITORY + "/git/ref/heads/main"],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ref = evidence.parse(query.stdout)
        require(ref.get("ref") == "refs/heads/main" and ref.get("object", {}).get("type") == "commit"
                and ref["object"]["sha"] == self.revision, "source revision is not current canonical main")

    def assert_implementation_bound(self):
        for relative, actual in (("tools/reconcile_kernel_provenance.py", __file__),
                                 ("tools/kernel_provenance_evidence.py", evidence.__file__)):
            expected = self.root / relative
            require(Path(actual).resolve() == expected.resolve() and not expected.is_symlink(),
                    "executing implementation is outside canonical source checkout")
            require(expected.read_bytes() == self.read(relative), "executing implementation differs from source revision")


class HubBackend:
    """Official SDK, fixed endpoint, public immutable reads, one bounded addition."""
    def __init__(self, publish=False):
        from huggingface_hub import HfApi
        self.publish_enabled = publish
        token = os.environ.get("HF_TOKEN") if publish else False
        require(not publish or (isinstance(token, str) and bool(token.strip())), "explicit HF_TOKEN required for publication")
        self.api = HfApi(endpoint=ENDPOINT, token=token)

    def authorize(self, repo_ids):
        require(self.publish_enabled and set(repo_ids) == set(evidence.TARGETS)
                and len(repo_ids) == len(evidence.TARGETS), "closed-set write authorization required")
        for repo_id in repo_ids:
            self.api.auth_check(repo_id=repo_id, repo_type="model", write=True)

    def inspect(self, repo_id, revision=None):
        require(repo_id in evidence.TARGETS, "unapproved provider target")
        if revision is not None:
            evidence.digest(revision, 40)
        info = self.api.repo_info(repo_id=repo_id, repo_type="model", revision=revision or "main", token=False)
        require(info.id == repo_id, "provider repository identity changed")
        head = evidence.digest(info.sha, 40)
        require(revision is None or head == revision, "provider immutable revision mismatch")
        from huggingface_hub.hf_api import RepoFile, RepoFolder
        rows = []
        # Exhaust SDK pagination at the immutable revision; an iteration failure
        # aborts rather than accepting a partial inventory.
        for item in self.api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=head,
                                            recursive=True, expand=False, token=False):
            if isinstance(item, RepoFile):
                rows.append({"path": item.path, "type": "file", "oid": item.blob_id, "size": item.size})
            else:
                require(isinstance(item, RepoFolder), "unknown provider tree item")
                rows.append({"path": item.path, "type": "directory"})
        files = file_identities(rows)
        require(info.siblings is not None, "provider metadata file inventory missing")
        siblings = [evidence.relative_path(item.rfilename) for item in info.siblings]
        require(len(siblings) == len(set(siblings)) and set(siblings) == set(files),
                "provider metadata and complete tree disagree")
        url = f"{ENDPOINT}/{repo_id}/resolve/{head}/{evidence.DESTINATION}"
        with urlopen(Request(url, headers={"User-Agent": "SZL-kernel-provenance-reconciliation/1"}), timeout=45) as response:
            require(response.status == 200, "provider immutable provenance GET failed")
            raw = response.read(evidence.MAX_DOCUMENT_BYTES + 1)
        require(0 < len(raw) <= evidence.MAX_DOCUMENT_BYTES, "provider provenance size outside bound")
        require(files[evidence.DESTINATION] == {"oid": blob_oid(raw), "size": len(raw)},
                "provider document and tree object differ")
        if revision is None:
            observed = self.api.repo_info(repo_id=repo_id, repo_type="model", revision="main", token=False)
            require(observed.id == repo_id and observed.sha == head, "provider main moved during observation")
        return LiveState(head, files, raw)

    def commit(self, target, source_revision):
        require(self.publish_enabled and target.repo_id in evidence.TARGETS, "provider writes disabled")
        evidence.digest(source_revision, 40)
        from huggingface_hub import CommitOperationAdd
        commit = self.api.create_commit(
            repo_id=target.repo_id, repo_type="model", revision="main",
            parent_commit=target.expected_parent, create_pr=False,
            operations=[CommitOperationAdd(path_in_repo=evidence.DESTINATION, path_or_fileobj=target.output)],
            commit_message="Reconcile kernel artifact provenance with published files",
            commit_description=f"Canonical source: {CANONICAL_REPOSITORY}@{source_revision}\n"
                               f"Output SHA256: {evidence.sha(target.output)}\n"
                               "Metadata-only correction; no training or independent approval claim.",
        )
        return commit.oid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "check", "publish"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-revision")
    parser.add_argument("--receipt", type=Path, help="NEW append-only JSONL file, outside the source checkout")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            require(args.source_revision is None and args.receipt is None, "validate accepts only --repo-root")
            release = validate_release(WorkingTreeSource(args.repo_root).read)
            result = {"status": "VALIDATED_OFFLINE_NOT_PUBLISHED", "targets": len(release.targets),
                      "manifest_sha256": release.manifest_sha256, "provider_mutations": 0}
        else:
            require(args.source_revision is not None and args.receipt is not None, "source revision and NEW receipt required")
            source = GitSource(args.repo_root, args.source_revision)
            source.assert_implementation_bound()
            release = validate_release(source.read)
            require(not args.receipt.resolve().is_relative_to(source.root), "receipt must be outside source checkout")
            result = execute(release, HubBackend(publish=args.command == "publish"),
                             source_revision=args.source_revision, receipt=args.receipt,
                             publish=args.command == "publish", authority_check=source.assert_publish_authority)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        # Do not echo provider/subprocess exception strings containing request data.
        if isinstance(exc, evidence.InvalidEvidence):
            print("REFUSED: " + str(exc), file=sys.stderr)
        else:
            print("STOPPED: " + type(exc).__name__ + "; inspect the retained journal, if created.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
