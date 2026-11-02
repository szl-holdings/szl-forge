"""Synthetic HTTP fixtures: these tests do not qualify any model or dataset."""
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("portfolio", Path(__file__).parents[1] / "tools/audit_training_portfolio.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

def row(name="one", sha="a" * 40, **extra):
    return {"id": "owner/" + name, "sha": sha, "private": False, "tags": [], **extra}

def response(rows, link=None):
    return json.dumps(rows).encode(), {"Link": link} if link else {}

class DiscoveryTests(unittest.TestCase):
    def test_follows_every_page_and_records_terminal_response(self):
        first = "https://huggingface.co/api/models?author=owner&limit=100&full=true"
        second = "https://huggingface.co/api/models?author=owner&cursor=opaque"
        requests = []
        def fetch(url):
            requests.append(url)
            return response([row()], "<"+second+'>; rel="next"') if url == first else response([row("two")])
        result = audit.discover("owner", "models", fetch)
        self.assertEqual(requests, [first, second])
        self.assertEqual(len(result["repositories"]), 2)
        self.assertTrue(result["pagination_complete"])
        self.assertIsNone(result["pages"][-1]["next"])
        self.assertTrue(all(len(p["sha256"]) == 64 for p in result["pages"]))

    def test_repeated_page_and_page_limit_fail(self):
        link = 'https://huggingface.co/api/models?author=owner&limit=100&full=true'
        with self.assertRaises(ValueError):
            audit.discover("owner", "models", lambda _: response([row()], "<"+link+'>; rel="next"'))
        with self.assertRaises(ValueError):
            audit.discover("owner", "models", lambda _: response([row()]), max_pages=0)

    def test_pagination_cannot_change_owner_kind_or_host(self):
        for url in ["https://evil.example/api/models?author=owner",
                    "https://huggingface.co/api/models?author=other",
                    "https://huggingface.co/api/datasets?author=owner",
                    "https://token@huggingface.co/api/models?author=owner",
                    "http://huggingface.co/api/models?author=owner"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                audit.discover("owner", "models", lambda _: response([row()], "<"+url+'>; rel="next"'))

    def test_duplicate_or_wrong_owner_is_rejected(self):
        for rows in [[row(), row()], [row(id="other/model")], [row(id="owner/../x")]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit.discover("owner", "models", lambda _: response(rows))

    def test_revision_and_public_visibility_required(self):
        for change in [{"sha":"main"}, {"sha":None}, {"private":True}, {"private":None}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.discover("owner", "models", lambda _: response([row(**change)]))

    def test_non_array_and_malformed_link_fail(self):
        with self.assertRaises(ValueError):
            audit.discover("owner", "models", lambda _: (b'{}', {}))
        for value in ['broken; rel="next"', '<https://x>; rel="next", <https://y>; rel="next"']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.next_link(value)

    def test_empty_public_owner_is_explicit_not_unavailable(self):
        result = audit.discover("owner", "datasets", lambda _: response([]))
        self.assertEqual(result["repositories"], [])
        self.assertEqual(result["scope"], "PUBLIC_API_VISIBILITY_ONLY")

class EvidenceTests(unittest.TestCase):
    def buckets(self, entries=None):
        return [audit.discover("owner", "models", lambda _: response(entries or [row()]))]

    def test_same_revision_reobserved_never_grants_training_or_quality(self):
        first = self.buckets()
        card = {"kind":"models", "repo":"owner/one", "revision":"a"*40, "status":"READ_BACK"}
        report = audit.reconcile(first, self.buckets(), [card])
        self.assertEqual(report["rows"][0]["badges"], ["PUBLIC_REVISION_REOBSERVED", "CARD_BYTES_READ_BACK"])
        for field in ["training_authorized", "promotion_authorized", "rights_verified",
                      "split_qualified", "independent_evaluator_verified"]:
            self.assertIs(report["rows"][0][field], False)
        self.assertFalse(report["training_authorized"])

    def test_missing_changed_and_added_repositories_never_get_stable_badge(self):
        first = self.buckets()
        variants = [[row(sha="b"*40)], [row("two")], [row(), row("two")]]
        for rows in variants:
            with self.subTest(rows=rows):
                report = audit.reconcile(first, self.buckets(rows), [])
                self.assertTrue(all(not r["public_inventory_stable"] and not r["badges"] for r in report["rows"]))

    def test_stale_or_unavailable_card_not_readback_badge(self):
        for card in [{"kind":"models","repo":"owner/one","revision":"b"*40,"status":"READ_BACK"},
                     {"kind":"models","repo":"owner/one","revision":"a"*40,"status":"UNAVAILABLE"}]:
            report = audit.reconcile(self.buckets(), self.buckets(), [card])
            self.assertEqual(report["rows"][0]["badges"], ["PUBLIC_REVISION_REOBSERVED"])

    def test_incomplete_capture_and_coverage_mismatch_rejected(self):
        first = self.buckets()
        first[0]["pagination_complete"] = False
        with self.assertRaises(ValueError):
            audit.reconcile(first, self.buckets(), [])
        with self.assertRaises(ValueError):
            audit.reconcile(self.buckets(), [], [])

    def test_card_requires_matching_commit_header(self):
        for commit in [None, "b"*40]:
            result = audit.read_card("models", row(), lambda _: (b"card", {"x-repo-commit": commit}))
            self.assertEqual(result["status"], "UNAVAILABLE")
        result = audit.read_card("models", row(), lambda _: (b"card", {"X-Repo-Commit": "a"*40}))
        self.assertEqual(result["status"], "READ_BACK")
        self.assertEqual(len(result["sha256"]), 64)

    def test_card_reads_only_pinned_readme_and_failure_is_redacted(self):
        urls = []
        def fetch(url):
            urls.append(url)
            raise RuntimeError("SECRET_EXAMPLE_DO_NOT_REPORT")
        result = audit.read_card("datasets", row(), fetch)
        self.assertEqual(urls, ["https://huggingface.co/datasets/owner/one/resolve/"+"a"*40+"/README.md"])
        self.assertNotIn("SECRET_EXAMPLE", json.dumps(result))

    def test_lanes_distinguish_weights_derivatives_software_and_apps(self):
        cases = [
            ("spaces", row(), "APPLICATION_REVIEW"),
            ("datasets", row(), "SOURCE_AND_SPLIT_REVIEW"),
            ("models", row(library_name="kernels"), "SOFTWARE_CONFORMANCE"),
            ("models", row(siblings=[{"rfilename":"model.gguf"}]), "DERIVATIVE_EVALUATION"),
            ("models", row(siblings=[{"rfilename":"adapter_model.safetensors"}]), "PEFT_EVALUATION_BEFORE_RETRAINING"),
            ("models", row(siblings=[{"rfilename":"model.safetensors"}]), "WEIGHT_LINEAGE_AND_BASELINE_REVIEW"),
            ("models", row(library_name="numpy"), "CPU_BASELINE_REVIEW"),
            ("models", row(), "RECIPE_OR_OTHER_ARTIFACT_REVIEW")]
        for kind, item, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(audit.classify(kind, item), expected)

    def test_permissive_license_does_not_clear_source_rights(self):
        first = self.buckets([row(tags=["license:apache-2.0"])])
        result = audit.reconcile(first, first, [])["rows"][0]
        self.assertEqual(result["declared_licenses"], ["apache-2.0"])
        self.assertFalse(result["rights_verified"])

class CardNamespaceTests(unittest.TestCase):
    def buckets(self):
        return [audit.discover("owner", kind, lambda _: response([row()]))
                for kind in audit.KINDS]

    def test_one_card_cannot_certify_other_repository_kinds(self):
        for kind in audit.KINDS:
            with self.subTest(kind=kind):
                card = audit.read_card(kind, row(), lambda _: (b"card", {"x-repo-commit":"a"*40}))
                report = audit.reconcile(self.buckets(), self.buckets(), [card])
                for result in report["rows"]:
                    self.assertEqual("CARD_BYTES_READ_BACK" in result["badges"], result["kind"] == kind)
                    self.assertEqual(result["card_status"], "READ_BACK" if result["kind"] == kind else "NOT_READ")

    def test_unavailable_card_cannot_overwrite_other_kind_readback(self):
        successful = audit.read_card("models", row(), lambda _: (b"model card", {"x-repo-commit":"a"*40}))
        failed = audit.read_card("datasets", row(), lambda _: (b"dataset card", {}))
        for cards in [[successful, failed], [failed, successful]]:
            with self.subTest(order=[c["status"] for c in cards]):
                rows = {r["kind"]:r for r in audit.reconcile(self.buckets(), self.buckets(), cards)["rows"]}
                self.assertIn("CARD_BYTES_READ_BACK", rows["models"]["badges"])
                self.assertNotIn("CARD_BYTES_READ_BACK", rows["datasets"]["badges"])
                self.assertEqual(rows["datasets"]["card_status"], "UNAVAILABLE")

    def test_card_without_kind_cannot_establish_namespace_readback(self):
        legacy = {"repo":"owner/one", "revision":"a"*40, "status":"READ_BACK"}
        rows = audit.reconcile(self.buckets(), self.buckets(), [legacy])["rows"]
        self.assertTrue(all("CARD_BYTES_READ_BACK" not in r["badges"] for r in rows))

    def test_read_card_records_kind_for_success_and_failure(self):
        for kind in audit.KINDS:
            for headers in [{"x-repo-commit":"a"*40}, {}]:
                with self.subTest(kind=kind,success=bool(headers)):
                    card = audit.read_card(kind, row(), lambda _: (b"card", headers))
                    self.assertEqual(card["kind"], kind)

if __name__ == "__main__":
    unittest.main()

