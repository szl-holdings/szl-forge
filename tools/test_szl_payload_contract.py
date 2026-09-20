# SPDX-License-Identifier: Apache-2.0
"""Synthetic offline controls; no runtime, remote source or model qualification."""
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import szl_payload_contract as M

NOW = datetime(2026, 9, 19, 13, tzinfo=timezone.utc)
START = "2026-09-17T10:00:00Z"
END = "2026-09-17T10:01:00Z"


def git_item():
    # Deliberately exercise Git's directory-vs-file sort: a.c sorts before a/.
    entries = [
        {"path": "a.c", "mode": "100644", "type": "blob", "sha": M.git_object("blob", b"x"), "size": 1},
        {"path": "a/z", "mode": "100755", "type": "blob", "sha": M.git_object("blob", b"z"), "size": 1},
        {"path": "link", "mode": "120000", "type": "blob", "sha": M.git_object("blob", b"a.c"), "size": 3},
        {"path": "vendor", "mode": "160000", "type": "commit", "sha": "d" * 40, "size": None},
    ]
    subtree = M.git_object("tree", M.git_tree_body([entries[1]]))
    entries.append({"path": "a", "mode": "040000", "type": "tree", "sha": subtree, "size": None})
    root = M.git_object("tree", M.git_tree_body([e for e in entries if "/" not in e["path"]]))
    return {"repository": "szl-holdings/fixture", "revision": "a" * 40,
            "revision_after": "a" * 40, "tree_sha": root, "default_branch": "main",
            "entries": entries, "file_count": 3, "symlink_count": 1, "submodule_count": 1,
            "tree_complete": True, "tree_merkle_verified": True, "complete": True,
            "content_bytes_verified": False, "runtime_verified": False, "blockers": [],
            "started_at": START, "finished_at": END}


def hf_item(family="models", redacted=False):
    row = {"path": "weights.safetensors", "type": "file", "oid": "b" * 40, "size": 42,
           "lfs_identity_state": "REDACTED" if redacted else "OBSERVED",
           "lfs_oid": None if redacted else "c" * 64}
    return {"repo_id": "SZLHOLDINGS/fixture", "repo_type_endpoint": family,
            "revision": "a" * 40, "revision_after": "a" * 40, "entries": [row],
            "file_count": 1, "tree_complete": True, "file_metadata_identity_complete": not redacted,
            "redacted_lfs_file_count": int(redacted), "complete": not redacted,
            "content_bytes_verified": False, "runtime_verified": False,
            "runtime_stage_reported": "RUNNING" if family == "spaces" else None,
            "blockers": ["HF_LFS_IDENTITY_REDACTED"] if redacted else [],
            "started_at": START, "finished_at": END}


def population(items):
    count = sum(item["complete"] for item in items)
    known = sum(item["file_count"] for item in items if item["file_count"] is not None)
    complete = count == len(items)
    return {"items": items, "items_observed": len(items), "items_complete": count,
            "known_file_subtotal": known, "complete_scope_file_count": known if complete else None,
            "complete": complete, "membership_stable": True, "blockers": []}


def receipt(lane="github", partial=False):
    populations = {"github": population([git_item()])} if lane == "github" else {
        f: population([hf_item(f, partial and f == "models")]) for f in ("models", "datasets", "spaces", "kernels")}
    complete = all(p["complete"] for p in populations.values())
    return {"schema": M.SCHEMA, "scope": M.SCOPE, "lane": lane, "populations": populations,
            "source_revision": "f" * 40, "observer_sha256": "e" * 64,
            "started_at": START, "finished_at": END, "source_content_files_read": 0,
            "complete": complete, "blockers": [], "production_authorization": False,
            "semantic_review_complete": False, "runtime_verified": False,
            "status": "FILE_METADATA_OBSERVED_NOT_QUALIFIED" if complete else "PARTIAL_OR_UNAVAILABLE"}


def archive(path, value, member="receipt.json", second=False, special=False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        if special:
            info = zipfile.ZipInfo(member)
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            out.writestr(info, M.canonical(value))
        else:
            out.writestr(member, M.canonical(value))
        if second:
            out.writestr("extra.json", b"{}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrictInputTests(unittest.TestCase):
    def test_json_duplicate_keys(self):
        with self.assertRaises(M.ContractError): M.strict_json(b'{"x":1,"x":2}')

    def test_json_nonfinite(self):
        for raw in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(M.ContractError): M.strict_json(raw)

    def test_json_unicode_and_lone_surrogate(self):
        for raw in (b'\xff', b'{"x":"\\ud800"}'):
            with self.subTest(raw=raw), self.assertRaises(M.ContractError): M.strict_json(raw)

    def test_json_empty_or_oversized(self):
        with self.assertRaises(M.ContractError): M.strict_json(b'')
        with self.assertRaises(M.ContractError): M.strict_json(b' ' * (M.MAX_JSON + 1))

    def test_json_nested_fail_is_fixed_error(self):
        with self.assertRaises(M.ContractError): M.strict_json(b'[' * 2000 + b']' * 2000)

    def test_valid_json_roundtrip(self):
        self.assertEqual(M.strict_json(M.canonical({"x": "hello", "n": 1})), {"x": "hello", "n": 1})

    def test_invalid_paths(self):
        for path in ("../secret", "/absolute", "a//b", "a/./b", "C:/file", "a\\b", "a\x00b", ""):
            with self.subTest(path=path), self.assertRaises(M.ContractError): M.safe_path(path)

    def test_timestamps_require_zone_and_valid_window(self):
        with self.assertRaises(M.ContractError): M.parse_stamp("2026-09-17T10:00:00")
        value = receipt(); value["finished_at"] = "2030-01-01T00:00:00Z"
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "github", NOW)


class ArchiveTests(unittest.TestCase):
    def test_digest_and_member_are_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "native.zip"; d = archive(p, receipt())
            result, info = M.read_native_archive(p, d)
            self.assertEqual(result, receipt())
            self.assertEqual(info["json_sha256"], hashlib.sha256(M.canonical(receipt())).hexdigest())

    def test_wrong_archive_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "native.zip"; archive(p, receipt())
            with self.assertRaises(M.ContractError): M.read_native_archive(p, "0" * 64)

    def test_traversal_member_never_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "native.zip"; d = archive(p, receipt(), "../secret.json")
            with self.assertRaises(M.ContractError): M.read_native_archive(p, d)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_second_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "native.zip"; d = archive(p, receipt(), second=True)
            with self.assertRaises(M.ContractError): M.read_native_archive(p, d)

    def test_symlink_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "native.zip"; d = archive(p, receipt(), special=True)
            with self.assertRaises(M.ContractError): M.read_native_archive(p, d)

    def test_non_zip_and_missing_file_are_fixed_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"; p.write_bytes(b"not zip")
            with self.assertRaises(M.ContractError): M.read_native_archive(p, hashlib.sha256(p.read_bytes()).hexdigest())
            with self.assertRaises(M.ContractError): M.read_native_archive(Path(tmp)/"missing", "0" * 64)


class GitTreeTests(unittest.TestCase):
    def test_rehashes_nested_trees_without_following_opaque_pointers(self):
        out = M.verify_git_item(git_item())
        self.assertEqual(out["verified_tree_objects"], 2)
        self.assertEqual(out["files"], 3)
        self.assertEqual(out["submodules_not_followed"], 1)
        self.assertEqual(out["symlinks_not_followed"], 1)
        self.assertEqual(out["content_review"], "NOT_PERFORMED")

    def test_git_known_empty_tree(self):
        self.assertEqual(M.git_object("tree", b""), "4b825dc642cb6eb9a060e54bf8d69288fbee4904")

    def test_directory_sort_matches_git_rule(self):
        value = git_item()
        body = M.git_tree_body([e for e in value["entries"] if "/" not in e["path"]])
        self.assertLess(body.index(b"a.c\0"), body.index(b"a\0"))
        self.assertIn(b"40000 a\0", body)
        self.assertNotIn(b"040000 a\0", body)

    def test_changed_referenced_blob_detected(self):
        value = git_item(); value["entries"][0]["sha"] = "0" * 40
        with self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_duplicate_path_detected(self):
        value = git_item(); value["entries"].append(copy.deepcopy(value["entries"][0]))
        with self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_missing_parent_detected(self):
        value = git_item(); value["entries"] = [r for r in value["entries"] if r["path"] != "a"]
        with self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_mode_type_and_size_are_strict(self):
        for field, bad in (("mode", "100777"), ("type", "tree"), ("size", True)):
            value = git_item(); value["entries"][0][field] = bad
            with self.subTest(field=field), self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_wrong_counts_detected(self):
        for name in ("file_count", "symlink_count", "submodule_count"):
            value = git_item(); value[name] += 1
            with self.subTest(name=name), self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_source_movement_not_pass(self):
        value = git_item(); value["revision_after"] = "b" * 40
        with self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_content_claim_cannot_be_amplified(self):
        value = git_item(); value["content_bytes_verified"] = True
        with self.assertRaises(M.ContractError): M.verify_git_item(value)

    def test_unsupported_partial_does_not_appear_complete(self):
        value = git_item(); value.update(tree_complete=False, complete=False, entries=[], file_count=None, blockers=["UNAVAILABLE"])
        self.assertIsNone(M.verify_git_item(value)["files"])
        value["complete"] = True
        with self.assertRaises(M.ContractError): M.verify_git_item(value)


class HuggingFaceTests(unittest.TestCase):
    def test_model_and_kernel_ids_remain_separate(self):
        for family in ("models", "kernels"):
            result = M.verify_hf_item(hf_item(family), family)
            self.assertEqual(result["family"], family)

    def test_wrong_family_rejected(self):
        with self.assertRaises(M.ContractError): M.verify_hf_item(hf_item("models"), "kernels")

    def test_redaction_preserves_metadata_gap(self):
        result = M.verify_hf_item(hf_item(redacted=True), "models")
        self.assertFalse(result["metadata_complete"])
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["redacted_content_ids"], 1)

    def test_redaction_never_uses_pointer_hash(self):
        value = hf_item(redacted=True); value["entries"][0]["lfs_oid"] = value["entries"][0]["oid"]
        with self.assertRaises(M.ContractError): M.verify_hf_item(value, "models")

    def test_redaction_not_promoted(self):
        value = hf_item(redacted=True); value["complete"] = True
        with self.assertRaises(M.ContractError): M.verify_hf_item(value, "models")

    def test_lfs_digest_and_states_strict(self):
        for key, val in (("lfs_oid", "main"), ("lfs_identity_state", []), ("size", False), ("type", [])):
            value = hf_item(); value["entries"][0][key] = val
            with self.subTest(key=key), self.assertRaises(M.ContractError): M.verify_hf_item(value, "models")

    def test_duplicate_paths_rejected(self):
        value = hf_item(); value["entries"] *= 2
        with self.assertRaises(M.ContractError): M.verify_hf_item(value, "models")

    def test_running_does_not_become_runtime_verified(self):
        result = M.verify_hf_item(hf_item("spaces"), "spaces")
        self.assertEqual(result["stage_reported"], "RUNNING")
        self.assertEqual(result["runtime"], "NOT_VERIFIED")

    def test_unknown_tree_retains_null(self):
        value = hf_item(); value.update(tree_complete=False, complete=False, entries=[], file_count=None, blockers=["UNAVAILABLE"])
        self.assertIsNone(M.verify_hf_item(value, "models")["files"])


class ReceiptTests(unittest.TestCase):
    def test_historical_scope_and_counts_preserved(self):
        result = M.verify_receipt(receipt(), "github", NOW)
        self.assertEqual(result["original_started_at"], START)
        self.assertEqual(result["freshness"], "HISTORICAL_REQUIRES_REOBSERVATION")
        self.assertFalse(result["semantic_review_complete"])
        self.assertEqual(result["populations"]["github"]["verified_git_tree_objects"], 2)

    def test_hf_partial_full_total_null(self):
        result = M.verify_receipt(receipt("huggingface", partial=True), "huggingface", NOW)
        self.assertIsNone(result["populations"]["models"]["complete_scope_file_count"])
        self.assertEqual(result["status"], "PARTIAL_OR_UNAVAILABLE")

    def test_boolean_count_rejected(self):
        value = receipt(); value["populations"]["github"]["items_observed"] = True
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "github", NOW)

    def test_required_family_not_inferred_zero(self):
        value = receipt("huggingface"); del value["populations"]["kernels"]
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "huggingface", NOW)

    def test_duplicate_assets_rejected(self):
        value = receipt(); value["populations"]["github"]["items"] *= 2
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "github", NOW)

    def test_partial_count_cannot_be_complete_count(self):
        value = receipt("huggingface", partial=True); value["populations"]["models"]["complete_scope_file_count"] = 1
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "huggingface", NOW)

    def test_global_authority_and_content_zero_strict(self):
        for key, bad in (("production_authorization", True), ("source_content_files_read", False), ("runtime_verified", 0)):
            value = receipt(); value[key] = bad
            with self.subTest(key=key), self.assertRaises(M.ContractError): M.verify_receipt(value, "github", NOW)

    def test_item_window_must_be_inside_receipt(self):
        value = receipt(); value["populations"]["github"]["items"][0]["started_at"] = "2026-09-16T10:00:00Z"
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "github", NOW)

    def test_membership_movement_cannot_be_green(self):
        value = receipt(); value["populations"]["github"]["membership_stable"] = False
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "github", NOW)

    def test_receipt_status_cannot_amplify_completeness(self):
        value = receipt("huggingface", partial=True); value["status"] = "FILE_METADATA_OBSERVED_NOT_QUALIFIED"
        with self.assertRaises(M.ContractError): M.verify_receipt(value, "huggingface", NOW)


class ConsistencyTests(unittest.TestCase):
    def fixture(self):
        return {"k": 2, "task_ids": ["a", "b"], "runs": [
            {"task_id": t, "repeat": j, "success": v} for t, j, v in
            (("a", 0, True), ("a", 1, False), ("b", 0, True), ("b", 1, True))]}

    def test_distinct_metrics(self):
        result = M.consistency(self.fixture())
        self.assertEqual((result["mean_at_k"], result["pass_power_k"], result["pass_at_k_empirical"]), (.75, .5, 1.0))
        self.assertEqual(result["consistency_gap"], .25)

    def test_missing_repetitions_not_ignored(self):
        value = self.fixture(); value["runs"].pop()
        result = M.consistency(value)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_runs"], 1)
        self.assertIsNone(result["mean_at_k"])

    def test_null_result_is_unknown(self):
        value = self.fixture(); value["runs"][0]["success"] = None
        self.assertEqual(M.consistency(value)["missing_runs"], 1)

    def test_duplicate_repeat_and_unknown_task_rejected(self):
        for field, value in (("repeat", 0), ("task_id", "unknown")):
            data = self.fixture(); data["runs"][1][field] = value
            with self.subTest(field=field), self.assertRaises(M.ContractError): M.consistency(data)

    def test_integer_success_not_boolean(self):
        data = self.fixture(); data["runs"][0]["success"] = 1
        with self.assertRaises(M.ContractError): M.consistency(data)

    def test_duplicate_declared_tasks_and_empty_population_rejected(self):
        for ids in (["a", "a"], []):
            data = self.fixture(); data["task_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(M.ContractError): M.consistency(data)


class CliAndReportTests(unittest.TestCase):
    def execute(self, partial=False, mismatch=False, existing=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); gh = root / "gh.zip"; hf = root / "hf.zip"; out = root / "output"
            g = receipt(); h = receipt("huggingface", partial)
            if mismatch: h["source_revision"] = "1" * 40
            gd = archive(gh, g); hd = archive(hf, h)
            if existing: out.mkdir(); (out / "keep").write_text("original")
            args = ["verify", "--github-zip", str(gh), "--github-sha256", gd,
                    "--huggingface-zip", str(hf), "--huggingface-sha256", hd, "--out-dir", str(out)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = M.main(args)
            return code, sorted(p.name for p in out.iterdir()) if out.exists() else []

    def test_actual_cli_complete_still_not_runtime(self):
        code, files = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(files, ["operator-review.html", "reconciliation.json", "review-queue.json"])

    def test_actual_cli_partial_retains_report_exit_two(self):
        self.assertEqual(self.execute(partial=True)[0], 2)

    def test_pair_source_mismatch_rejected_before_output(self):
        self.assertEqual(self.execute(mismatch=True), (1, []))

    def test_existing_output_never_overwritten(self):
        self.assertEqual(self.execute(existing=True), (1, ["keep"]))

    def test_local_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"; M.write_new(path, b"original")
            with self.assertRaises(FileExistsError): M.write_new(path, b"overwrite")
            self.assertEqual(path.read_bytes(), b"original")

    def test_report_escapes_content_and_does_not_execute(self):
        report = {"github": M.verify_receipt(receipt(), "github", NOW),
                  "huggingface": M.verify_receipt(receipt("huggingface"), "huggingface", NOW)}
        report["github"]["populations"]["github"]["items"][0]["id"] = '<script>alert(1)</script>'
        rendered = M.summary_html(report)
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertIn('Recorded inventory is not operational qualification', rendered)
        self.assertNotIn('https://', rendered)


if __name__ == "__main__":
    unittest.main()
