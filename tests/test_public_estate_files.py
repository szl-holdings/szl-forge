"""Offline adversarial tests; no Hub/GitHub reads or repository code execution."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("census", ROOT / "tools/observe_public_estate_files.py")
census = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = census
spec.loader.exec_module(census)


def git_tree(entries):
    # Fixture constructor deliberately uses a direct byte concatenation.
    entries = sorted(entries, key=lambda e: e[1].encode() + (b"/" if e[0] == "40000" else b""))
    raw = b"".join(mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(oid)
                   for mode, name, oid in entries)
    return hashlib.sha1(b"tree " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


BLOB = "a" * 40
SUB = git_tree([("100644", "x.py", BLOB)])
TREE = git_tree([("40000", "src", SUB), ("120000", "link", BLOB), ("160000", "dep", BLOB)])
ROWS = [
    {"path": "src", "mode": "040000", "type": "tree", "sha": SUB},
    {"path": "src/x.py", "mode": "100644", "type": "blob", "sha": BLOB, "size": 0},
    {"path": "link", "mode": "120000", "type": "blob", "sha": BLOB, "size": 3},
    {"path": "dep", "mode": "160000", "type": "commit", "sha": BLOB},
]


class Trees(unittest.TestCase):
    def test_empty_tree_known_git_identity(self):
        self.assertEqual(census.tree_digest([]), "4b825dc642cb6eb9a060e54bf8d69288fbee4904")

    def test_verified_nested_tree_preserves_opaque_entries(self):
        rows = census.verify_tree(TREE, ROWS)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["type"] == "commit" for r in rows), 1)
        self.assertEqual(sum(r["mode"] == "120000" for r in rows), 1)

    def test_git_directory_sort_order(self):
        entries = [("100644", "a.c", BLOB), ("40000", "a", SUB), ("100644", "a0", BLOB)]
        rows = [dict(mode=m, path=n, sha=s, type="tree" if m == "40000" else "blob")
                for m, n, s in entries]
        self.assertEqual(census.tree_digest(rows), git_tree(entries))

    def test_wrong_root_rejected(self):
        with self.assertRaises(census.CensusError):
            census.verify_tree("b" * 40, ROWS)

    def test_missing_child_rejected(self):
        with self.assertRaises(census.CensusError):
            census.verify_tree(TREE, [r for r in ROWS if r["path"] != "src/x.py"])

    def test_missing_directory_rejected(self):
        with self.assertRaises(census.CensusError):
            census.verify_tree(TREE, ROWS[1:])

    def test_duplicate_rejected(self):
        with self.assertRaises(census.CensusError):
            census.verify_tree(TREE, ROWS + ROWS[:1])

    def test_path_bounds(self):
        for path in ("../a", "/a", "a//b", "a/./b", "a\\b", "a\n", "a\0", "a/", "x" * 4097):
            with self.subTest(path=path), self.assertRaises(census.CensusError):
                census.safe_path(path)

    def test_mode_type_mismatch(self):
        for mode, kind in (("120000", "tree"), ("100644", "commit"), ("999999", "blob")):
            with self.subTest(mode=mode), self.assertRaises(census.CensusError):
                census.verify_tree(TREE, [dict(ROWS[0], mode=mode, type=kind)] + ROWS[1:])

    def test_bad_sha_and_bool_size(self):
        for change in ({"sha": "main"}, {"size": True}, {"size": -1}):
            with self.subTest(change=change), self.assertRaises(census.CensusError):
                census.verify_tree(TREE, [ROWS[0], dict(ROWS[1], **change)] + ROWS[2:])

    def test_recursive_truncation_uses_complete_subtrees(self):
        class Fake:
            def __init__(self): self.urls = []
            def get(self, url):
                self.urls.append(url)
                if "recursive=" in url:
                    return {"sha": TREE, "tree": ROWS[:1], "truncated": True}, ""
                if url.endswith(TREE):
                    return {"sha": TREE, "tree": [ROWS[0]] + ROWS[2:], "truncated": False}, ""
                return {"sha": SUB, "tree": [dict(ROWS[1], path="x.py")], "truncated": False}, ""
        fake = Fake()
        rows, fallback = census.github_tree(fake, "szl-holdings/example", TREE)
        self.assertTrue(fallback)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(fake.urls), 3)

    def test_nonrecursive_truncation_is_not_complete(self):
        fake = mock.Mock()
        fake.get.return_value = ({"sha": TREE, "tree": [], "truncated": True}, "")
        with self.assertRaisesRegex(census.CensusError, "TREE_TRUNCATED"):
            census.github_tree(fake, "szl-holdings/example", TREE)

    def test_missing_truncation_flag_rejected(self):
        fake = mock.Mock()
        fake.get.return_value = ({"sha": TREE, "tree": ROWS}, "")
        with self.assertRaises(census.CensusError):
            census.github_tree(fake, "szl-holdings/example", TREE)


class Transport(unittest.TestCase):
    def test_github_token_never_on_hf(self):
        client = census.Client(token="not-a-real-credential")
        self.assertIn("Authorization", client.headers("https://api.github.com/orgs/szl-holdings/repos"))
        self.assertNotIn("Authorization", client.headers("https://huggingface.co/api/models?author=SZLHOLDINGS"))

    def test_hosts_and_resources_constrained(self):
        for url in ("http://api.github.com/orgs/szl-holdings/repos",
                    "https://api.github.com.evil/repos/szl-holdings/a",
                    "https://evil@api.github.com/orgs/szl-holdings/repos",
                    "https://api.github.com:443/orgs/szl-holdings/repos",
                    "https://api.github.com/repos/other/a/commits/main",
                    "https://api.github.com/repos/szl-holdings/a/actions/secrets",
                    "https://huggingface.co/api/models/other/a",
                    "https://huggingface.co/api/models?author=OTHER",
                    "https://huggingface.co/api/models?author=SZLHOLDINGS#x"):
            with self.subTest(url=url), self.assertRaises(census.CensusError):
                census.validate_url(url)

    def test_redirect_denied(self):
        handler = census.NoRedirect()
        with self.assertRaises(census.CensusError):
            handler.redirect_request(None, None, 302, "redirect", {}, "https://example.com")

    def test_duplicate_and_nonfinite_json_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
            with self.subTest(text=text), self.assertRaises(census.CensusError):
                census.strict_json(text.encode())

    def test_next_link_scope_immutable(self):
        base = "https://huggingface.co/api/models?author=SZLHOLDINGS&limit=100"
        good = base + "&cursor=abc"
        self.assertEqual(census.next_page(f'<{good}>; rel="next"', base), good)
        for url in (good.replace("SZLHOLDINGS", "OTHER"), good.replace("/models", "/datasets"),
                    good.replace("https://huggingface.co", "https://evil.example"),
                    good + "&token=x", good + "&author=SZLHOLDINGS"):
            with self.subTest(url=url), self.assertRaises(census.CensusError):
                census.next_page(f'<{url}>; rel="next"', base)

    def test_unrecognized_link_does_not_silently_end(self):
        with self.assertRaises(census.CensusError):
            census.next_page("not a link", "https://huggingface.co/api/models?author=SZLHOLDINGS")

    def test_repeated_cursor_rejected(self):
        url = "https://huggingface.co/api/models?author=SZLHOLDINGS&limit=100"
        fake = mock.Mock()
        fake.get.return_value = ([{"id": "SZLHOLDINGS/a"}], f'<{url}>; rel="next"')
        with self.assertRaises(census.CensusError):
            census.hf_pages(fake, url)

    def test_request_bound_before_network(self):
        c = census.Client(token=None)
        c.calls = census.MAX_REQUESTS
        with self.assertRaisesRegex(census.CensusError, "REQUEST_BOUND"):
            c.get("https://api.github.com/orgs/szl-holdings/repos")

    def test_size_bound(self):
        c = census.Client(token=None)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {"Content-Length": str(census.MAX_BODY + 1)}
        c.opener = mock.Mock()
        c.opener.open.return_value = response
        with self.assertRaisesRegex(census.CensusError, "BODY_BOUND"):
            c.get("https://api.github.com/orgs/szl-holdings/repos")


class HuggingFace(unittest.TestCase):
    def test_missing_sha_does_not_become_zero(self):
        fake = mock.Mock()
        fake.get.return_value = ({"id": "SZLHOLDINGS/a"}, "")
        result = census.observe_hf_repo(fake, "models", "SZLHOLDINGS/a")
        self.assertFalse(result["complete"])
        self.assertIsNone(result["file_count"])

    def test_success_is_metadata_not_execution(self):
        fake = mock.Mock()
        fake.get.side_effect = [
            ({"id": "SZLHOLDINGS/a", "private": False, "sha": BLOB}, ""),
            ([{"path": "README.md", "type": "file", "oid": "b" * 40, "size": 4}], ""),
            ({"id": "SZLHOLDINGS/a", "private": False, "sha": BLOB}, ""),
        ]
        result = census.observe_hf_repo(fake, "models", "SZLHOLDINGS/a")
        self.assertTrue(result["complete"])
        self.assertEqual(result["file_count"], 1)
        self.assertFalse(result["content_bytes_verified"])
        self.assertFalse(result["runtime_verified"])

    def test_revision_movement_retains_exact_tree(self):
        fake = mock.Mock()
        fake.get.side_effect = [({"id": "SZLHOLDINGS/a", "private": False, "sha": BLOB}, ""), ([], ""),
                                ({"id": "SZLHOLDINGS/a", "private": False, "sha": "b" * 40}, "")]
        result = census.observe_hf_repo(fake, "spaces", "SZLHOLDINGS/a")
        self.assertFalse(result["complete"])
        self.assertTrue(result["tree_complete"])
        self.assertIn("REF_MOVED", result["blockers"])

    def test_private_or_wrong_id_not_exposed(self):
        for info in ({"id": "SZLHOLDINGS/a", "private": True, "sha": BLOB},
                     {"id": "OTHER/a", "sha": BLOB}):
            fake = mock.Mock()
            fake.get.return_value = (info, "")
            self.assertFalse(census.observe_hf_repo(fake, "models", "SZLHOLDINGS/a")["complete"])
            self.assertEqual(fake.get.call_count, 1)

    def test_duplicate_file_rejected(self):
        item = {"path": "a", "type": "file", "oid": BLOB, "size": 1}
        with self.assertRaises(census.CensusError):
            census.hf_file_rows([item, item])

    def test_lfs_metadata_not_payload_verification(self):
        rows = census.hf_file_rows([{"path": "model.safetensors", "type": "file", "oid": BLOB,
                                    "size": 5000000000, "lfs": {"oid": "c" * 64, "size": 5000000000}}])
        self.assertEqual(rows[0]["size"], 5000000000)
        self.assertEqual(rows[0]["lfs_oid"], "c" * 64)


class Scope(unittest.TestCase):
    def test_repo_id_scope(self):
        for value in ("OTHER/a", "szl-holdings/a/../../b", None, "szl-holdings/a?x"):
            with self.assertRaises(census.CensusError):
                census.repo_id(value, census.GH_ORG)

    def test_aggregate_unknown_on_partial(self):
        report = census.summarize([{"complete": True, "file_count": 5},
                                   {"complete": False, "file_count": None}], True)
        self.assertIsNone(report["complete_scope_file_count"])
        self.assertEqual(report["known_file_subtotal"], 5)
        self.assertFalse(report["complete"])

    def test_membership_drift_not_complete(self):
        self.assertFalse(census.summarize([{"complete": True, "file_count": 1}], False)["complete"])

    def test_no_production_or_semantic_authority(self):
        report = census.base_report("github", "a" * 40)
        for name in ("semantic_review_complete", "runtime_verified", "production_authorization"):
            self.assertIs(report[name], False)
        self.assertEqual(report["source_content_files_read"], 0)

    def test_source_categories_are_only_path_signals(self):
        signals = census.path_signals([{"path": "src/app.tsx"}, {"path": "tests/test_api.py"},
                                       {"path": "server.py"}, {"path": "README.md"}])
        self.assertEqual(signals["frontend_paths"], 1)
        self.assertEqual(signals["test_paths"], 1)


class GatedLfsMetadata(unittest.TestCase):
    """Synthetic versions of the masked public shape; never real LFS payloads."""
    @staticmethod
    def item(oid=None):
        return {"path": "model.safetensors", "type": "file", "oid": BLOB, "size": 42,
                "lfs": {"oid": "*" * 64 if oid is None else oid, "size": 42, "pointerSize": 133}}

    @staticmethod
    def info(**changes):
        return {"id": "SZLHOLDINGS/a", "private": False, "sha": BLOB, "gated": "auto", **changes}

    def observation(self, *, before=None, after=None, entries=None):
        before = self.info() if before is None else before
        after = self.info() if after is None else after
        entries = [self.item()] if entries is None else entries
        fake = mock.Mock()
        fake.get.side_effect = [(before, ""), (entries, ""),
                                after if isinstance(after, Exception) else (after, "")]
        return census.observe_hf_repo(fake, "models", "SZLHOLDINGS/a"), fake

    def test_mask_is_never_a_digest_or_pointer_substitution(self):
        row = census.hf_file_rows([self.item()], allow_redacted=True)[0]
        self.assertEqual(row["oid"], BLOB)
        self.assertIsNone(row["lfs_oid"])
        self.assertEqual(row["lfs_identity_state"], "REDACTED")

    def test_mask_requires_explicit_gated_context(self):
        for allowed in (False, None, 0, 1, "true", [], {}):
            with self.subTest(allowed=allowed), self.assertRaises(census.CensusError):
                census.hf_file_rows([self.item()], allow_redacted=allowed)

    def test_other_mask_and_digest_shapes_rejected(self):
        values = ("*" * 63, "*" * 65, "*" * 63 + "a", "REDACTED", "sha256:" + "c" * 64,
                  "c" * 63, "C" * 64, "c" * 64 + "\n", True, [], {}, 0, "")
        for value in values:
            with self.subTest(value=value), self.assertRaises(census.CensusError):
                census.hf_file_rows([self.item(value)], allow_redacted=True)

    def test_malformed_lfs_objects_rejected(self):
        for lfs in ({}, {"size": 42}, {"oid": None, "size": 42}, [], "", True, 0):
            entry = self.item()
            entry["lfs"] = lfs
            with self.subTest(lfs=lfs), self.assertRaises(census.CensusError):
                census.hf_file_rows([entry], allow_redacted=True)

    def test_lfs_size_must_be_typed_and_agree(self):
        for size in (None, True, -1, 43, "42", 42.0):
            entry = self.item()
            entry["lfs"]["size"] = size
            with self.subTest(size=size), self.assertRaisesRegex(census.CensusError, "HF_LFS_SIZE"):
                census.hf_file_rows([entry], allow_redacted=True)

    def test_lfs_pointer_size_must_be_positive_integer(self):
        for size in (None, True, 0, -1, "133", 133.0):
            entry = self.item()
            entry["lfs"]["pointerSize"] = size
            with self.subTest(size=size), self.assertRaisesRegex(census.CensusError, "HF_LFS_POINTER_SIZE"):
                census.hf_file_rows([entry], allow_redacted=True)

    def test_directory_cannot_declare_lfs(self):
        entry = dict(self.item(), type="directory")
        with self.assertRaises(census.CensusError):
            census.hf_file_rows([entry], allow_redacted=True)

    def test_regular_and_lfs_states_remain_distinct(self):
        entry = self.item("c" * 64)
        regular = {"path": "README.md", "type": "file", "oid": BLOB, "size": 1}
        rows = census.hf_file_rows([entry, regular])
        self.assertEqual(rows[0]["lfs_identity_state"], "NOT_REPORTED")
        self.assertEqual(rows[1]["lfs_identity_state"], "OBSERVED")
        self.assertEqual(rows[1]["lfs_oid"], "c" * 64)

    def test_input_observation_not_mutated(self):
        import copy
        entry = self.item()
        before = copy.deepcopy(entry)
        census.hf_file_rows([entry], allow_redacted=True)
        self.assertEqual(entry, before)

    def test_redacted_tree_retained_but_item_stays_incomplete(self):
        row, fake = self.observation()
        self.assertEqual(fake.get.call_count, 3)
        self.assertTrue(row["tree_complete"])
        self.assertEqual(row["file_count"], 1)
        self.assertEqual(row["redacted_lfs_file_count"], 1)
        self.assertEqual(row["revision"], row["revision_after"])
        self.assertIs(row["file_metadata_identity_complete"], False)
        self.assertIs(row["complete"], False)
        self.assertEqual(row["blockers"], ["HF_LFS_IDENTITY_REDACTED"])
        self.assertFalse(row["content_bytes_verified"])
        self.assertFalse(row["runtime_verified"])

    def test_nongated_mask_cannot_be_admitted_as_redacted(self):
        for gated in (False, True, None, "unknown", "AUTO", [], {}):
            row, _ = self.observation(before=self.info(gated=gated))
            with self.subTest(gated=gated):
                self.assertFalse(row["complete"])
                self.assertNotIn("entries", row)
                self.assertIn("HF_LFS_IDENTITY", row["blockers"])

    def test_manual_gate_supported_without_completion(self):
        row, _ = self.observation(before=self.info(gated="manual"), after=self.info(gated="manual"))
        self.assertTrue(row["tree_complete"])
        self.assertFalse(row["complete"])

    def test_partial_aggregate_retains_only_known_subtotal(self):
        row, _ = self.observation()
        result = census.summarize([row, {"complete": True, "file_count": 5}], True)
        self.assertEqual(result["known_file_subtotal"], 6)
        self.assertIsNone(result["complete_scope_file_count"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["items_complete"], 1)

    def test_final_visibility_failure_withholds_paths_and_counts(self):
        for after in (self.info(private=True), self.info(private=None), self.info(private=0),
                      self.info(id="OTHER/a"), {}, census.CensusError("HTTP_403")):
            row, _ = self.observation(after=after)
            with self.subTest(after=type(after).__name__):
                self.assertFalse(row["complete"])
                self.assertFalse(row["tree_complete"])
                self.assertIsNone(row["file_count"])
                self.assertNotIn("entries", row)
                self.assertNotIn("redacted_lfs_file_count", row)
                self.assertNotIn("path_signals", row)
                self.assertNotIn("readme_present", row)

    def test_ordinary_tree_also_withheld_when_final_visibility_fails(self):
        row, _ = self.observation(entries=[self.item("c" * 64)], after=self.info(private=True))
        self.assertNotIn("entries", row)
        self.assertIsNone(row["file_count"])
        self.assertIn("HF_PUBLIC_IDENTITY", row["blockers"])

    def test_initial_public_identity_must_be_explicit_false(self):
        for before in (self.info(private=None), self.info(private=0), {"id": "SZLHOLDINGS/a", "sha": BLOB}):
            row, fake = self.observation(before=before)
            self.assertEqual(fake.get.call_count, 1)
            self.assertIn("HF_PUBLIC_IDENTITY", row["blockers"])
            self.assertIsNone(row["file_count"])

    def test_ref_movement_retains_observed_public_tree_not_complete(self):
        row, _ = self.observation(after=self.info(sha="b" * 40))
        self.assertTrue(row["tree_complete"])
        self.assertEqual(row["file_count"], 1)
        self.assertFalse(row["complete"])
        self.assertIn("REF_MOVED", row["blockers"])
        self.assertIn("HF_LFS_IDENTITY_REDACTED", row["blockers"])

    def test_gate_movement_is_recorded_separately(self):
        row, _ = self.observation(after=self.info(gated=False))
        self.assertTrue(row["tree_complete"])
        self.assertFalse(row["complete"])
        self.assertIn("HF_ACCESS_POLICY_MOVED", row["blockers"])

    def test_available_hash_is_not_content_execution(self):
        row, _ = self.observation(entries=[self.item("c" * 64)])
        self.assertTrue(row["complete"])
        self.assertTrue(row["file_metadata_identity_complete"])
        self.assertEqual(row["redacted_lfs_file_count"], 0)
        self.assertFalse(row["content_bytes_verified"])
        self.assertFalse(row["runtime_verified"])

    def test_file_and_directory_counts_remain_separate(self):
        entries = [dict(self.item(), path="adapter/model.safetensors"),
                   {"path": "adapter", "type": "directory", "oid": BLOB, "size": 0}]
        row, _ = self.observation(entries=entries)
        self.assertEqual(len(row["entries"]), 2)
        self.assertEqual(row["file_count"], 1)
        self.assertFalse(row["complete"])


if __name__ == "__main__":
    unittest.main()
