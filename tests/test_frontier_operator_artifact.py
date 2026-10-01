"""Independent offline adversarial checks for exact-run artifact ingestion."""
from __future__ import annotations

import copy
from email.message import Message
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch
import urllib.request
import urllib.response
import warnings
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "szl_frontier_operator.py"
SPEC = importlib.util.spec_from_file_location("artifact_operator_under_test", SCRIPT)
op = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(op)
SHA = "a" * 40
RUN_NUMBER = 123
ATTEMPT = 2
NAMES = ("receipt", "bundle", "summary", "publication")


def archive_bytes(*, omit=(), replacements=None, extra=()):
    replacements = replacements or {}
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in NAMES:
                if name not in omit:
                    archive.writestr(f"frontier-evaluation/{name}.json", replacements.get(name, b"{}\n"))
            for name, content in extra:
                # ZipInfo normalizes Windows backslashes at construction; retain
                # adversarial archive spelling so the reader sees it unchanged.
                if isinstance(name, str):
                    member = zipfile.ZipInfo()
                    member.filename = name
                    member.orig_filename = name
                    member.compress_type = zipfile.ZIP_DEFLATED
                    name = member
                archive.writestr(name, content)
    return output.getvalue()


def artifact_metadata(raw):
    return {
        "id": 987,
        "name": f"frontier-real-evaluation-{RUN_NUMBER}-{ATTEMPT}",
        "expired": False,
        "size_in_bytes": len(raw),
        "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "workflow_run": {"head_sha": SHA, "id": RUN_NUMBER},
    }


class ArtifactZipBoundaries(unittest.TestCase):
    def test_valid_zip_returns_only_required_json_bytes(self):
        raw = archive_bytes(extra=(("hf-inference-credential.json", b"{}"), ("frontier-evaluation/log.txt", b"log")))
        result = op.read_artifact_zip(raw)
        self.assertEqual(result, {f"{name}.json": b"{}\n" for name in NAMES})

    def test_compressed_size_bound_is_enforced(self):
        raw = archive_bytes()
        with patch.object(op, "OUTPUT_LIMIT", len(raw) - 1):
            with self.assertRaises(op.OperatorError):
                op.read_artifact_zip(raw)

    def test_expanded_size_bound_includes_unconsumed_members(self):
        raw = archive_bytes(extra=(("unused.txt", b"x" * 8192),))
        self.assertLess(len(raw), 2048)
        with patch.object(op, "OUTPUT_LIMIT", 2048):
            with self.assertRaises(op.OperatorError):
                op.read_artifact_zip(raw)

    def test_member_count_is_bounded(self):
        raw = archive_bytes(extra=tuple((f"extra-{index}.txt", b"") for index in range(101)))
        with self.assertRaises(op.OperatorError):
            op.read_artifact_zip(raw)

    def test_duplicate_member_is_rejected(self):
        raw = archive_bytes(extra=(("frontier-evaluation/receipt.json", b'{"second":true}'),))
        with self.assertRaises(op.OperatorError):
            op.read_artifact_zip(raw)

    def test_traversal_and_absolute_paths_are_rejected_even_when_unused(self):
        for name in ("../outside.json", "frontier-evaluation/../../outside.json", "/absolute.json",
                     "C:/outside.json", "C:\\outside.json", "folder\\outside.json", "./extra.json"):
            with self.subTest(name=name), self.assertRaises(op.OperatorError):
                op.read_artifact_zip(archive_bytes(extra=((name, b"{}"),)))

    def test_symlink_member_is_rejected(self):
        symlink = zipfile.ZipInfo("unused-symlink")
        symlink.create_system = 3
        symlink.external_attr = 0o120777 << 16
        with self.assertRaises(op.OperatorError):
            op.read_artifact_zip(archive_bytes(extra=((symlink, b"../../target"),)))

    def test_every_required_json_member_is_required(self):
        for name in NAMES:
            with self.subTest(name=name), self.assertRaises(op.OperatorError):
                op.read_artifact_zip(archive_bytes(omit=(name,)))

    def test_invalid_duplicate_and_nonfinite_json_are_rejected(self):
        for value in (b"not json", b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'"\xff"'):
            with self.subTest(value=value), self.assertRaises(op.OperatorError):
                op.read_artifact_zip(archive_bytes(replacements={"receipt": value}))

    def test_invalid_and_truncated_zip_are_rejected(self):
        raw = archive_bytes()
        for value in (b"not a ZIP", raw[:len(raw) // 2]):
            with self.subTest(length=len(value)), self.assertRaises(op.OperatorError):
                op.read_artifact_zip(value)


class ExactRunArtifactTransport(unittest.TestCase):
    def fetch(self, metadata, raw):
        with patch.object(op, "gh_pages", return_value=metadata) as pages, \
             patch.object(op, "run", return_value=subprocess.CompletedProcess([], 0, raw, b"")) as transport:
            result = op.github_artifact(op.FORGE, SHA, RUN_NUMBER, ATTEMPT)
        return result, pages, transport

    def test_valid_exact_run_uses_authenticated_bounded_binary_download(self):
        raw = archive_bytes()
        (files, binding), pages, transport = self.fetch([artifact_metadata(raw)], raw)
        self.assertEqual(files, {f"{name}.json": b"{}\n" for name in NAMES})
        self.assertEqual(binding["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(binding["run_attempt"], ATTEMPT)
        self.assertTrue(binding["exact_run_source_binding"])
        pages.assert_called_once_with(f"/repos/{op.FORGE}/actions/runs/{RUN_NUMBER}/artifacts", "artifacts")
        transport.assert_called_once_with(
            ["gh", "api", "--method", "GET", f"/repos/{op.FORGE}/actions/artifacts/987/zip"],
            binary=True, timeout=60, max_bytes=op.OUTPUT_LIMIT,
        )

    def test_wrong_sha_run_attempt_expired_and_missing_metadata_refuse_download(self):
        raw = archive_bytes()
        valid = artifact_metadata(raw)
        variants = []
        for key, value in (("expired", True), ("expired", None), ("id", "987"),
                           ("size_in_bytes", 0), ("size_in_bytes", op.OUTPUT_LIMIT + 1),
                           ("name", f"frontier-real-evaluation-{RUN_NUMBER}-1")):
            item = copy.deepcopy(valid)
            item[key] = value
            variants.append([item])
        for key, value in (("head_sha", "b" * 40), ("id", RUN_NUMBER + 1)):
            item = copy.deepcopy(valid)
            item["workflow_run"][key] = value
            variants.append([item])
        variants.extend(([], [valid, copy.deepcopy(valid)]))
        for metadata in variants:
            with self.subTest(metadata=metadata), patch.object(op, "gh_pages", return_value=metadata), \
                 patch.object(op, "run") as transport:
                with self.assertRaises(op.OperatorError):
                    op.github_artifact(op.FORGE, SHA, RUN_NUMBER, ATTEMPT)
                transport.assert_not_called()

    def test_wrong_or_missing_digest_rejects_otherwise_valid_zip(self):
        raw = archive_bytes()
        for digest in (None, "sha256:" + "0" * 64, hashlib.sha256(raw).hexdigest()):
            item = artifact_metadata(raw)
            item["digest"] = digest
            with self.subTest(digest=digest), self.assertRaises(op.OperatorError):
                self.fetch([item], raw)

    def test_digest_matching_invalid_zip_is_still_rejected(self):
        raw = b"untrusted compressed content"
        with self.assertRaises(op.OperatorError):
            self.fetch([artifact_metadata(raw)], raw)

    def test_download_unavailable_cannot_become_verified(self):
        raw = archive_bytes()
        with patch.object(op, "gh_pages", return_value=[artifact_metadata(raw)]), \
             patch.object(op, "run", side_effect=op.Unavailable("download failed")):
            with self.assertRaises(op.Unavailable):
                op.github_artifact(op.FORGE, SHA, RUN_NUMBER, ATTEMPT)


class PublicationArtifactBinding(unittest.TestCase):
    def test_hf_receipt_must_match_artifact_bytes_before_semantic_validation(self):
        run_id = f"github-{RUN_NUMBER}-{ATTEMPT}-{SHA[:12]}"
        root = f"runs/2026/09/08/{run_id}/glm-5-3-flash"
        published_revision, current_revision = "b" * 40, "c" * 40
        files = {f"{name}.json": b"{}\n" for name in NAMES}
        files["publication.json"] = op.canonical({
            "dataset_id": op.DATASET, "commit_oid": published_revision, "path_in_repo": root,
        })
        workflow = {
            "id": RUN_NUMBER, "run_attempt": ATTEMPT, "head_sha": SHA, "head_branch": "main",
            "event": "push", "path": op.WORKFLOW, "status": "completed", "conclusion": "success",
        }
        jobs = [{"name": name, "status": "completed", "conclusion": "success"}
                for name in ("offline contract", "real candidate vs Khipu")]
        jobs[1]["steps"] = [
            {"name": name, "conclusion": "success"}
            for name in ("Select a validated receipt publisher credential", "Execute real bounded evaluation")
        ]
        dataset = {"revision": current_revision, "paths": [f"{root}/{name}.json" for name in NAMES]}
        # Even an equivalent JSON object must fail byte binding. Also reject a
        # current-head replacement when the original publication still matches.
        for responses in (
            [(b'{"tampered":true}', {})],
            [(b"{ }", {})],
            [(files["receipt.json"], {}), (b'{"tampered":true}', {})],
        ):
            with self.subTest(reads=len(responses), first=responses[0][0]), \
                 patch.object(op, "gh_api", return_value={"merged": True, "merged_at": "2026-09-08", "merge_commit_sha": SHA}), \
                 patch.object(op, "workflows", return_value=[workflow]), \
                 patch.object(op, "gh_pages", return_value=jobs), \
                 patch.object(op, "source_contract", return_value={}), \
                 patch.object(op, "github_artifact", return_value=(files, {"exact_run_source_binding": True})) as artifact, \
                 patch.object(op, "hf_dataset_snapshot", return_value=dataset), \
                 patch.object(op, "http_bytes", side_effect=responses) as fetch, \
                 patch.object(op, "validate_publication") as validate:
                with self.assertRaisesRegex(op.OperatorError, "receipt differs from exact GitHub run artifact"):
                    op.verify_forge_174()
                validate.assert_not_called()
                artifact.assert_called_once_with(op.FORGE, SHA, RUN_NUMBER, ATTEMPT)
                self.assertEqual(fetch.call_count, len(responses))
                self.assertIn(f"/resolve/{published_revision}/{root}/receipt.json", fetch.call_args_list[0].args[0])
                if len(responses) == 2:
                    self.assertIn(f"/resolve/{current_revision}/{root}/receipt.json", fetch.call_args_list[1].args[0])


class HuggingFaceTransportBoundaries(unittest.TestCase):
    def fetch(self, destinations):
        requests = []

        class OfflineHTTPS(urllib.request.HTTPSHandler):
            def https_open(self, request):
                requests.append(request.full_url)
                headers = Message()
                index = len(requests) - 1
                if index < len(destinations):
                    headers["Location"] = destinations[index]
                    response = urllib.response.addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
                    response.msg = "Found"
                else:
                    response = urllib.response.addinfourl(io.BytesIO(b"{}"), headers, request.full_url, 200)
                    response.msg = "OK"
                return response

        real_builder = urllib.request.build_opener
        with patch.object(op.urllib.request, "build_opener", side_effect=lambda *handlers:
                          real_builder(*handlers, OfflineHTTPS())), \
             patch.object(urllib.request.HTTPHandler, "http_open", side_effect=AssertionError("HTTP reached")):
            try:
                result = op.http_bytes("https://huggingface.co/datasets/a/b/resolve/main/receipt.json")
            except op.OperatorError:
                # Expose only the requests that actually reached the transport.
                return None, requests
        return result, requests

    def test_untrusted_redirect_is_rejected_before_following(self):
        for destination in ("https://attacker.example/receipt.json", "http://huggingface.co/receipt.json",
                            "https://huggingface.co.attacker.example/receipt.json",
                            "https://user:password@huggingface.co/receipt.json",
                            "https://huggingface.co:444/receipt.json"):
            with self.subTest(destination=destination):
                result, requests = self.fetch([destination])
                self.assertIsNone(result)
                self.assertEqual(len(requests), 1)

    def test_later_redirect_hop_cannot_escape(self):
        result, requests = self.fetch(["https://huggingface.co/api/cache/receipt.json",
                                       "https://attacker.example/receipt.json"])
        self.assertIsNone(result)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(url.startswith("https://huggingface.co/") for url in requests))

    def test_allowed_absolute_and_relative_redirects_still_work(self):
        result, requests = self.fetch(["https://huggingface.co/api/cache/receipt.json", "/api/final.json"])
        self.assertEqual(result[0], b"{}")
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[-1], "https://huggingface.co/api/final.json")

    def test_final_response_url_is_checked_before_reading(self):
        response = Mock()
        response.geturl.return_value = "https://attacker.example/receipt.json"
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=response)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.object(op.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(op.OperatorError, "untrusted Hugging Face"):
                op.http_bytes("https://huggingface.co/api/datasets/a/b")
        response.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
