"""Offline publication contracts. All provider operations are fake/injected."""
import base64
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import reconcile_kernel_provenance as publisher

evidence = publisher.evidence


def snapshot(repo_id):
    revision = "a" * 40
    original = {"model": {"id": repo_id, "repository_type": "model", "trained_weights_present": True,
                          "surrogate": {"file": evidence.TARGETS[repo_id], "receipt": "training.json"}},
                "claims": {"trained_model": "SURROGATE_PRESENT_MEASURED_FIDELITY"}}
    raw = evidence.canonical(original)
    rows = [{"id": repo_id, "kind": "models", "revision": revision, "path": evidence.DESTINATION,
             "type": "file", "oid": publisher.blob_oid(raw), "size": len(raw)},
            {"id": repo_id, "kind": "models", "revision": revision, "path": "kernel.py",
             "type": "file", "oid": "e" * 40, "size": 512}]
    record = {"id": repo_id, "kind": "models", "revision": revision,
              "metadata": {"id": repo_id, "sha": revision,
                           "siblings": [{"rfilename": row["path"]} for row in rows]},
              "metadata_receipt": {"observed_at": "SYNTHETIC"}, "tree_entries": 2, "file_count": 2,
              "tree_coverage": "COMPLETE_ACCESSIBLE_RESPONSE",
              "tree_pages": [{"status": 200,
                              "requested_url": f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true",
                              "response_sha256": "b" * 64, "next_url": None}]}
    text = {"id": repo_id, "kind": "models", "revision": revision, "path": evidence.DESTINATION,
            "content": original, "receipt": {"status": 200, "response_sha256": evidence.sha(raw), "response_bytes": len(raw)}}
    return {"id": repo_id, "repository_type": "model", "revision": revision, "repository_record": record,
            "tree": rows, "text_record": text, "original_bytes_base64": base64.b64encode(raw).decode()}


def fixture_files():
    snapshots = [snapshot(repo_id) for repo_id in sorted(evidence.TARGETS)]
    snapshots_raw = evidence.canonical(snapshots)
    files = {publisher.BUNDLE + "/snapshots.json": snapshots_raw}
    entries = []
    for item in snapshots:
        path = publisher.BUNDLE + "/" + item["id"].split("/")[1] + "/" + evidence.DESTINATION
        files[path] = evidence.canonical(publisher.document_for(item))
        entries.append({"id": item["id"], "repository_type": "model", "expected_parent": item["revision"],
                        "destination": evidence.DESTINATION, "source_path": path,
                        "original_sha256": item["text_record"]["receipt"]["response_sha256"],
                        "output_sha256": evidence.sha(files[path])})
    files[publisher.MANIFEST] = evidence.canonical({"schema": publisher.RELEASE_SCHEMA, "bundle": publisher.BUNDLE,
                                                  "snapshots_sha256": evidence.sha(snapshots_raw), "entries": entries})
    return files


class FakeBackend:
    def __init__(self, release):
        self.targets = {target.repo_id: target for target in release.targets}
        self.states = {target.repo_id: publisher.LiveState(target.expected_parent, copy.deepcopy(target.files), target.original)
                       for target in release.targets}
        self.events = []
        self.commits = []
        self.fail_commit = None
        self.fail_auth = False
        self.bad_readback = False

    def inspect(self, repo_id, revision=None):
        self.events.append(("inspect", repo_id, revision))
        return self.states[repo_id]

    def authorize(self, repo_ids):
        self.events.append(("authorize", tuple(repo_ids)))
        if self.fail_auth:
            raise PermissionError("SYNTHETIC SECRET MUST NOT APPEAR IN RECEIPT")

    def commit(self, target, source_revision):
        self.events.append(("commit", target.repo_id))
        self.commits.append((target.repo_id, target.expected_parent, source_revision))
        if self.fail_commit == len(self.commits):
            raise TimeoutError("SYNTHETIC SECRET MUST NOT APPEAR IN RECEIPT")
        revision = format(len(self.commits), "040x")
        output = b"CORRUPTED" if self.bad_readback else target.output
        self.states[target.repo_id] = publisher.LiveState(revision, publisher.expected_output_files(target), output)
        return revision


class ReconcilePublisherTests(unittest.TestCase):
    def setUp(self):
        self.files = fixture_files()
        self.release = publisher.validate_release(self.files.__getitem__)
        self.backend = FakeBackend(self.release)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.receipt = Path(self.temporary.name) / "receipt.jsonl"
        self.authority_calls = 0

    def authority(self):
        self.authority_calls += 1

    def execute(self, publish=True):
        return publisher.execute(self.release, self.backend, source_revision="c" * 40,
                                 receipt=self.receipt, publish=publish, authority_check=self.authority)

    def events(self):
        return [json.loads(line) for line in self.receipt.read_text().splitlines()]

    def test_offline_closed_bundle_preserves_history_and_disclaims_training(self):
        self.assertEqual(len(self.release.targets), 8)
        for target in self.release.targets:
            document = evidence.parse(target.output)
            self.assertEqual(document["status"], publisher.DOCUMENT_STATUS)
            self.assertFalse(document["model"]["trained_weights_present"])
            self.assertFalse(document["claims"]["training_verified"])
            self.assertEqual(base64.b64decode(document["historical_evidence"]["original_bytes_base64"]), target.original)

    def test_manifest_expansion_missing_duplicate_and_unknown_fields_refused(self):
        for mutation in ("missing", "duplicate", "extra", "destination", "source", "type", "parent", "unknown"):
            files = dict(self.files)
            manifest = evidence.parse(files[publisher.MANIFEST])
            entries = manifest["entries"]
            if mutation == "missing":
                entries.pop()
            elif mutation == "duplicate":
                entries[-1] = entries[0]
            elif mutation == "extra":
                entries[0]["id"] = "attacker/model"
            elif mutation == "destination":
                entries[0]["destination"] = "README.md"
            elif mutation == "source":
                entries[0]["source_path"] = "../../payload.py"
            elif mutation == "type":
                entries[0]["repository_type"] = "space"
            elif mutation == "parent":
                entries[0]["expected_parent"] = "b" * 40
            else:
                manifest["approval"] = "FAKE"
            files[publisher.MANIFEST] = evidence.canonical(manifest)
            with self.subTest(mutation=mutation), self.assertRaises(evidence.InvalidEvidence):
                publisher.validate_release(files.__getitem__)

    def test_output_tamper_refused_even_with_recomputed_hash(self):
        for update_hash in (False, True):
            files = dict(self.files)
            manifest = evidence.parse(files[publisher.MANIFEST])
            entry = manifest["entries"][0]
            document = evidence.parse(files[entry["source_path"]])
            document["claims"]["training_verified"] = True
            files[entry["source_path"]] = evidence.canonical(document)
            if update_hash:
                entry["output_sha256"] = evidence.sha(files[entry["source_path"]])
                files[publisher.MANIFEST] = evidence.canonical(manifest)
            with self.subTest(update_hash=update_hash), self.assertRaises(evidence.InvalidEvidence):
                publisher.validate_release(files.__getitem__)

    def test_snapshot_byte_tamper_refused(self):
        self.files[publisher.BUNDLE + "/snapshots.json"] += b" "
        with self.assertRaises(evidence.InvalidEvidence):
            publisher.validate_release(self.files.__getitem__)

    def test_check_is_read_only_without_publish_authority(self):
        result = self.execute(publish=False)
        self.assertEqual(result["status"], "CHECKED_NOT_PUBLISHED")
        self.assertEqual(len(self.backend.commits), 0)
        self.assertEqual(self.authority_calls, 0)
        self.assertFalse(any(event[0] == "authorize" for event in self.backend.events))

    def test_all_target_preflight_then_fsynced_attempt_cas_and_readback(self):
        real_commit = self.backend.commit

        def commit(target, source_revision):
            # The attempt must already be readable from a separate file handle.
            self.assertEqual(self.events()[-1]["event"], "WRITE_ATTEMPT")
            self.assertEqual(self.events()[-1]["expected_parent"], "a" * 40)
            return real_commit(target, source_revision)

        self.backend.commit = commit
        result = self.execute()
        self.assertEqual(result["commits_verified"], 8)
        self.assertEqual(result["write_attempts"], 8)
        self.assertEqual(self.authority_calls, 9)
        self.assertTrue(result["complete"])
        first_commit = next(i for i, event in enumerate(self.backend.events) if event[0] == "commit")
        before = self.backend.events[:first_commit]
        self.assertEqual({event[1] for event in before if event[0] == "inspect"}, set(evidence.TARGETS))
        self.assertTrue(any(event[0] == "authorize" for event in before))
        self.assertTrue(all(parent == "a" * 40 for _, parent, _ in self.backend.commits))
        self.assertEqual([row["sequence"] for row in self.events()], list(range(len(self.events()))))
        self.assertEqual(sum(row["event"] == "FINAL_ARTIFACT_VERIFIED" for row in self.events()), 8)

    def test_stale_eighth_head_prevents_every_write(self):
        last = self.release.targets[-1]
        self.backend.states[last.repo_id] = publisher.LiveState("b" * 40, last.files, last.original)
        with self.assertRaises(evidence.InvalidEvidence):
            self.execute()
        self.assertEqual(self.backend.commits, [])
        self.assertEqual(self.events()[-1]["status"], "REFUSED_BEFORE_WRITE")

    def test_new_weight_or_changed_other_file_refuses_all_writes(self):
        for mutation in ("weight", "other_file"):
            backend = FakeBackend(self.release)
            target = self.release.targets[-1]
            files = copy.deepcopy(target.files)
            files["weights.safetensors" if mutation == "weight" else "kernel.py"] = {"oid": "f" * 40, "size": 12}
            backend.states[target.repo_id] = publisher.LiveState(target.expected_parent, files, target.original)
            with self.subTest(mutation=mutation), self.assertRaises(evidence.InvalidEvidence):
                publisher.preflight(self.release, backend)
            self.assertEqual(backend.commits, [])

    def test_write_scope_for_all_targets_checked_before_first_commit(self):
        self.backend.fail_auth = True
        with self.assertRaises(PermissionError):
            self.execute()
        self.assertEqual(self.backend.commits, [])
        self.assertNotIn("SYNTHETIC SECRET", self.receipt.read_text())
        self.assertEqual(self.events()[-1]["status"], "REFUSED_BEFORE_WRITE")

    def test_partial_commit_failure_journal_retains_verified_and_uncertain_counts(self):
        self.backend.fail_commit = 2
        with self.assertRaises(TimeoutError):
            self.execute()
        stopped = self.events()[-1]
        self.assertEqual(stopped["status"], "PARTIAL_OR_UNCERTAIN")
        self.assertEqual(stopped["commits_verified"], 1)
        self.assertEqual(stopped["write_attempts"], 2)
        self.assertFalse(stopped["complete"])
        self.assertFalse(stopped["automatic_retry"])
        self.assertNotIn("SYNTHETIC SECRET", self.receipt.read_text())
        self.assertEqual(len(self.backend.commits), 2)

    def test_readback_mismatch_stops_without_false_success(self):
        self.backend.bad_readback = True
        with self.assertRaises(evidence.InvalidEvidence):
            self.execute()
        self.assertEqual(len(self.backend.commits), 1)
        self.assertEqual(self.events()[-1]["commits_verified"], 0)
        self.assertFalse(any(row["event"] == "COMPLETE" for row in self.events()))

    def test_already_applied_is_noop_only_for_exact_full_tree(self):
        for target in self.release.targets:
            self.backend.states[target.repo_id] = publisher.LiveState("d" * 40, publisher.expected_output_files(target), target.output)
        result = self.execute()
        self.assertEqual(result["already_applied_exact"], 8)
        self.assertEqual(result["commits_verified"], 0)
        self.assertEqual(self.backend.commits, [])
        target = self.release.targets[0]
        files = publisher.expected_output_files(target)
        files["extra.py"] = {"oid": "f" * 40, "size": 0}
        with self.assertRaises(evidence.InvalidEvidence):
            publisher.classify(target, publisher.LiveState("d" * 40, files, target.output))

    def test_sdk_identical_output_race_verifies_artifacts_without_claiming_authorship(self):
        first = self.release.targets[0]
        real_commit = self.backend.commit
        preexisting_revision = "d" * 40

        def commit(target, source_revision):
            if target.repo_id != first.repo_id:
                return real_commit(target, source_revision)
            # Another writer installs identical bytes after our final READY read.
            # The SDK may return that existing commit without creating a new one.
            self.backend.events.append(("sdk_identical_output_noop", target.repo_id))
            self.backend.states[target.repo_id] = publisher.LiveState(
                preexisting_revision, publisher.expected_output_files(target), target.output)
            return preexisting_revision

        self.backend.commit = commit
        result = self.execute()
        self.assertTrue(result["complete"])
        self.assertEqual(result["write_attempts"], 8)
        self.assertEqual(result["commits_verified"], 8)
        self.assertEqual(len(self.backend.commits), 7)
        self.assertFalse(result["commit_authorship_verified"])
        verified = [row for row in self.events() if row["event"] == "COMMIT_VERIFIED"]
        self.assertEqual(len(verified), 8)
        self.assertEqual(verified[0]["revision"], preexisting_revision)
        self.assertTrue(all(row["commit_authorship_verified"] is False for row in verified))
        self.assertIn("preexisting identical state", verified[0]["attribution"])
        self.assertFalse(self.events()[-1]["commit_authorship_verified"])

    def test_reusing_receipt_refuses_without_provider_access(self):
        self.receipt.write_text("unknown old attempt")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(self.backend.events, [])
        self.assertEqual(self.receipt.read_text(), "unknown old attempt")

    def test_authority_loss_midbatch_stops_after_verified_first_write(self):
        def authority():
            self.authority_calls += 1
            if self.authority_calls == 3:
                raise evidence.InvalidEvidence("source main moved")

        with self.assertRaises(evidence.InvalidEvidence):
            publisher.execute(self.release, self.backend, source_revision="c" * 40, receipt=self.receipt,
                              publish=True, authority_check=authority)
        self.assertEqual(len(self.backend.commits), 1)
        self.assertEqual(self.events()[-1]["commits_verified"], 1)

    def test_post_preflight_target_move_is_not_overwritten(self):
        initial_inspect = self.backend.inspect
        counts = {}

        def inspect(repo_id, revision=None):
            counts[repo_id] = counts.get(repo_id, 0) + 1
            state = initial_inspect(repo_id, revision)
            if counts[repo_id] == 2:
                return publisher.LiveState("f" * 40, state.files, state.document)
            return state

        self.backend.inspect = inspect
        with self.assertRaises(evidence.InvalidEvidence):
            self.execute()
        self.assertEqual(self.backend.commits, [])

    def test_final_readback_rechecks_earlier_already_applied_target(self):
        first = self.release.targets[0]
        self.backend.states[first.repo_id] = publisher.LiveState("d" * 40, publisher.expected_output_files(first), first.output)
        initial_inspect = self.backend.inspect
        counts = {}

        def inspect(repo_id, revision=None):
            counts[repo_id] = counts.get(repo_id, 0) + 1
            state = initial_inspect(repo_id, revision)
            if repo_id == first.repo_id and counts[repo_id] > 1:
                return publisher.LiveState("f" * 40, state.files, b"changed")
            return state

        self.backend.inspect = inspect
        with self.assertRaises(evidence.InvalidEvidence):
            self.execute()
        self.assertEqual(self.events()[-1]["phase"], "final_readback")
        self.assertFalse(any(row["event"] == "COMPLETE" for row in self.events()))

    def test_publisher_requires_explicit_authority(self):
        with self.assertRaises(evidence.InvalidEvidence):
            publisher.execute(self.release, self.backend, source_revision="c" * 40, receipt=self.receipt, publish=True)
        self.assertFalse(self.receipt.exists())

    def test_sdk_commit_fixed_destination_and_parent_cas(self):
        # Construct without discovering credentials, and mock the SDK operation.
        calls = []
        target = self.release.targets[0]
        backend = publisher.HubBackend.__new__(publisher.HubBackend)
        backend.publish_enabled = True
        backend.api = SimpleNamespace(create_commit=lambda **kwargs: (calls.append(kwargs) or SimpleNamespace(oid="d" * 40)))
        operation = lambda **kwargs: SimpleNamespace(**kwargs)
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(CommitOperationAdd=operation)}):
            self.assertEqual(backend.commit(target, "c" * 40), "d" * 40)
        self.assertEqual(calls[0]["parent_commit"], target.expected_parent)
        self.assertEqual(calls[0]["revision"], "main")
        self.assertFalse(calls[0]["create_pr"])
        self.assertEqual(len(calls[0]["operations"]), 1)
        self.assertEqual(calls[0]["operations"][0].path_in_repo, "MODEL_PROVENANCE.json")
        self.assertEqual(calls[0]["operations"][0].path_or_fileobj, target.output)

    def test_backend_requires_explicit_token_and_never_auths_readonly(self):
        calls = []
        api = lambda **kwargs: (calls.append(kwargs) or SimpleNamespace())
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(HfApi=api)}), patch.dict(os.environ, {}, clear=True):
            publisher.HubBackend(publish=False)
            self.assertEqual(calls, [{"endpoint": "https://huggingface.co", "token": False}])
            with self.assertRaises(evidence.InvalidEvidence):
                publisher.HubBackend(publish=True)
            self.assertEqual(len(calls), 1)

    def test_backend_authorize_checks_all_targets_and_rejects_expansion(self):
        calls = []
        backend = publisher.HubBackend.__new__(publisher.HubBackend)
        backend.publish_enabled = True
        backend.api = SimpleNamespace(auth_check=lambda **kwargs: calls.append(kwargs))
        backend.authorize(list(evidence.TARGETS))
        self.assertEqual({call["repo_id"] for call in calls}, set(evidence.TARGETS))
        self.assertTrue(all(call["write"] and call["repo_type"] == "model" for call in calls))
        with self.assertRaises(evidence.InvalidEvidence):
            backend.authorize(["attacker/model"])

    def test_file_inventory_rejects_duplicates_unknown_types_weights_and_unsafe_paths(self):
        good = {"path": evidence.DESTINATION, "type": "file", "oid": "a" * 40, "size": 3}
        cases = [[good, good], [good, {**good, "path": "../file"}],
                 [good, {**good, "path": "other", "type": "symlink"}],
                 [good, {**good, "path": "model.safetensors"}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(evidence.InvalidEvidence):
                publisher.file_identities(rows)


class HubInspectionContractTests(unittest.TestCase):
    class RepoFile(SimpleNamespace):
        pass

    class RepoFolder(SimpleNamespace):
        pass

    class Response:
        status = 200

        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return self.raw[:limit]

    def setUp(self):
        self.target = publisher.validate_release(fixture_files().__getitem__).targets[0]
        self.info = SimpleNamespace(
            id=self.target.repo_id, sha=self.target.expected_parent,
            siblings=[SimpleNamespace(rfilename=path) for path in self.target.files])
        self.rows = [self.RepoFolder(path="empty-directory")]
        self.rows.extend(self.RepoFile(path=path, blob_id=identity["oid"], size=identity["size"])
                         for path, identity in self.target.files.items())
        self.backend = publisher.HubBackend.__new__(publisher.HubBackend)
        self.backend.publish_enabled = False
        self.backend.api = SimpleNamespace(repo_info=Mock(return_value=self.info),
                                          list_repo_tree=Mock(return_value=iter(self.rows)))
        sdk = SimpleNamespace(RepoFile=self.RepoFile, RepoFolder=self.RepoFolder)
        self.module_patch = patch.dict("sys.modules", {"huggingface_hub.hf_api": sdk})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.http_patch = patch.object(publisher, "urlopen", return_value=self.Response(self.target.original))
        self.http = self.http_patch.start()
        self.addCleanup(self.http_patch.stop)

    def test_immutable_inspection_binds_metadata_tree_and_document_to_requested_revision(self):
        result = self.backend.inspect(self.target.repo_id, revision=self.target.expected_parent)
        self.assertEqual(result, publisher.LiveState(self.target.expected_parent,
                                                    self.target.files, self.target.original))
        self.backend.api.repo_info.assert_called_once_with(
            repo_id=self.target.repo_id, repo_type="model", revision=self.target.expected_parent, token=False)
        self.backend.api.list_repo_tree.assert_called_once_with(
            repo_id=self.target.repo_id, repo_type="model", revision=self.target.expected_parent,
            recursive=True, expand=False, token=False)
        self.http.assert_called_once()
        request = self.http.call_args.args[0]
        self.assertEqual(request.full_url,
                         f"https://huggingface.co/{self.target.repo_id}/resolve/"
                         f"{self.target.expected_parent}/MODEL_PROVENANCE.json")
        self.assertNotIn("Authorization", request.headers)

    def test_mutable_inspection_rechecks_main_after_immutable_reads(self):
        result = self.backend.inspect(self.target.repo_id)
        self.assertEqual(result.revision, self.target.expected_parent)
        self.assertEqual(self.backend.api.repo_info.call_count, 2)
        for call in self.backend.api.repo_info.call_args_list:
            self.assertEqual(call.kwargs, {"repo_id": self.target.repo_id, "repo_type": "model",
                                           "revision": "main", "token": False})
        self.assertEqual(self.backend.api.list_repo_tree.call_args.kwargs["revision"], self.target.expected_parent)

    def test_metadata_siblings_mismatch_duplicate_and_missing_inventory_refused(self):
        original = self.info.siblings
        cases = {"missing_file": original[:-1],
                 "extra_file": original + [SimpleNamespace(rfilename="unexpected.py")],
                 "duplicate": original + [original[0]], "missing_inventory": None}
        for name, siblings in cases.items():
            self.info.siblings = siblings
            self.backend.api.list_repo_tree.return_value = iter(self.rows)
            with self.subTest(case=name), self.assertRaises(evidence.InvalidEvidence):
                self.backend.inspect(self.target.repo_id, revision=self.target.expected_parent)
        self.http.assert_not_called()

    def test_main_movement_during_observation_refused_after_document_read(self):
        for changed in (SimpleNamespace(id=self.target.repo_id, sha="f" * 40),
                        SimpleNamespace(id="attacker/model", sha=self.target.expected_parent)):
            self.backend.api.repo_info.side_effect = [self.info, changed]
            self.backend.api.list_repo_tree.return_value = iter(self.rows)
            self.http.reset_mock()
            with self.subTest(changed=changed), self.assertRaisesRegex(
                    evidence.InvalidEvidence, "main moved during observation"):
                self.backend.inspect(self.target.repo_id)
            self.http.assert_called_once()

    def test_immutable_revision_mismatch_refused_before_tree_or_document(self):
        self.info.sha = "f" * 40
        with self.assertRaisesRegex(evidence.InvalidEvidence, "immutable revision mismatch"):
            self.backend.inspect(self.target.repo_id, revision=self.target.expected_parent)
        self.backend.api.list_repo_tree.assert_not_called()
        self.http.assert_not_called()

    def test_tree_pagination_failure_is_not_accepted_as_complete_inventory(self):
        def interrupted_tree():
            yield from self.rows
            raise TimeoutError("synthetic pagination failure")

        self.backend.api.list_repo_tree.return_value = interrupted_tree()
        with self.assertRaises(TimeoutError):
            self.backend.inspect(self.target.repo_id, revision=self.target.expected_parent)
        self.http.assert_not_called()


class GitSourceContractTests(unittest.TestCase):
    def source(self, responses):
        source = publisher.GitSource.__new__(publisher.GitSource)
        source.root = Path.cwd().resolve()
        source.revision = "c" * 40
        source.command = lambda *args: responses[args]
        return source

    def test_source_reads_git_blob_not_working_tree(self):
        raw = b"canonical bytes\n"
        oid = publisher.blob_oid(raw)
        path = "publishing/file.json"
        source = self.source({("ls-tree", "-z", "c" * 40, "--", path):
                              f"100644 blob {oid}\t{path}\0".encode(), ("cat-file", "blob", oid): raw})
        self.assertEqual(source.read(path), raw)

    def test_source_rejects_symlink_and_nonblob_objects(self):
        path = "publishing/file.json"
        for mode, kind in (("120000", "blob"), ("040000", "tree")):
            source = self.source({("ls-tree", "-z", "c" * 40, "--", path):
                                  f"{mode} {kind} {'a' * 40}\t{path}\0".encode()})
            with self.subTest(mode=mode), self.assertRaises(evidence.InvalidEvidence):
                source.read(path)

    def test_source_rejects_dirty_head_wrong_origin_and_mismatched_ci_revision(self):
        base = {("rev-parse", "--show-toplevel"): str(Path.cwd()).encode(),
                ("rev-parse", "HEAD"): b"c" * 40,
                ("status", "--porcelain", "--untracked-files=all"): b"",
                ("remote", "get-url", "origin"): b"https://github.com/szl-holdings/szl-forge.git"}
        for mutation in ("dirty", "head", "origin", "environment"):
            responses = dict(base)
            if mutation == "dirty":
                responses[("status", "--porcelain", "--untracked-files=all")] = b" M code.py"
            elif mutation == "head":
                responses[("rev-parse", "HEAD")] = b"a" * 40
            elif mutation == "origin":
                responses[("remote", "get-url", "origin")] = b"https://github.com/attacker/repo.git"
            env = {"GITHUB_SHA": "a" * 40} if mutation == "environment" else {}
            with self.subTest(mutation=mutation), patch.dict(os.environ, env, clear=True), self.assertRaises(evidence.InvalidEvidence):
                self.source(responses).assert_clean()

    def test_publish_requires_fresh_canonical_main(self):
        source = self.source({})
        source.assert_clean = lambda: None
        source.assert_implementation_bound = lambda: None
        repository = {"full_name": publisher.CANONICAL_REPOSITORY, "default_branch": "main"}
        for head in ("a" * 40, "c" * 40):
            ref = {"ref": "refs/heads/main", "object": {"type": "commit", "sha": head}}
            results = [SimpleNamespace(stdout=evidence.canonical(repository)), SimpleNamespace(stdout=evidence.canonical(ref))]
            with self.subTest(head=head), patch.object(subprocess, "run", side_effect=results):
                if head != source.revision:
                    with self.assertRaises(evidence.InvalidEvidence):
                        source.assert_publish_authority()
                else:
                    source.assert_publish_authority()

    def test_running_implementation_must_match_canonical_checkout_and_blob(self):
        source = self.source({})
        source.root = Path(publisher.__file__).resolve().parents[1]
        source.read = lambda path: (source.root / path).read_bytes()
        source.assert_implementation_bound()
        source.read = lambda path: b"tampered"
        with self.assertRaises(evidence.InvalidEvidence):
            source.assert_implementation_bound()


if __name__ == "__main__":
    unittest.main()
