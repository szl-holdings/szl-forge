"""No-network fail-closed tests for exact public HF Space source-file parity."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "check_public_space_release_parity.py"
spec = importlib.util.spec_from_file_location("space_parity", SOURCE)
parity = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parity)

REVISION = "a" * 40
SPACE = "SZLHOLDINGS/szl-model-inference-lab"
MANIFEST_SCHEMA = "szl.space-source-release/v1"


class PublicSpaceReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {"README.md": b"approved GitHub README\n", "app.py": b"print('ok')\n"}
        self.remote = {}
        self.rebuild()

    def rebuild(self):
        for name, body in self.files.items():
            (self.root / name).write_bytes(body)
        source = {
            "schema": MANIFEST_SCHEMA,
            "release_id": "test-release",
            "source_files": {k: parity.digest(v) for k, v in self.files.items()},
        }
        self.release = json.dumps(source, indent=2).encode("utf-8")
        (self.root / "release.json").write_bytes(self.release)
        self.remote = {"release.json": self.release, **self.files}

    def fake_get(self, url: str, limit: int):
        self.assertIn("/resolve/" + REVISION + "/", url)
        file = url.split("/resolve/" + REVISION + "/", 1)[1]
        self.assertLessEqual(limit, parity.MAX_SOURCE_BYTES)
        if file not in self.remote:
            raise parity.RemoteUnavailable("HTTP_404")
        if self.remote[file] == "RATE_LIMIT":
            raise parity.RemoteUnavailable("HTTP_429")
        return self.remote[file]

    def check(self, space_id=SPACE, sha=REVISION):
        return parity.verify(self.root, space_id, sha, fetch=self.fake_get)

    def test_full_repository_match_is_not_runtime_or_prod_proof(self):
        report = self.check()
        self.assertEqual(report["state"], "REPOSITORY_BYTES_MATCH_ONLY")
        self.assertEqual(report["matched_count"], 2)
        self.assertTrue(report["space_repository_parity"])
        for field in (
            "served_image_verified", "model_inference_verified",
            "receipt_signed_by_this_tool", "owner_authority_verified",
            "release_admitted", "provider_writes",
        ):
            self.assertFalse(report[field], field)

    def test_appended_showcase_section_is_exact_readme_drift(self):
        self.remote["README.md"] += b"\n<!-- szl:showcase-card v1 weaver -->\n"
        report = self.check()
        self.assertEqual(report["state"], "DRIFT")
        self.assertEqual(report["reason"], "REMOTE_SOURCE_DIGEST_DRIFT")
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["mismatches"], ["README.md"])
        self.assertFalse(report["release_admitted"])

    def test_remote_manifest_drift_is_not_hidden_by_matching_source(self):
        self.remote["release.json"] = b"{}"
        report = self.check()
        self.assertEqual(report["state"], "DRIFT")
        self.assertEqual(report["reason"], "RELEASE_MANIFEST_BYTES_DIFFER")
        self.assertEqual(report["mismatches"], ["release.json"])

    def test_local_source_drift_cannot_be_accepted(self):
        (self.root / "README.md").write_bytes(b"tampered README")
        with mock.patch.object(self, "fake_get", side_effect=AssertionError("remote queried")):
            report = self.check()
        self.assertEqual(report["state"], "LOCAL_SOURCE_CONFLICT")
        self.assertEqual(report["local_mismatches"], ["README.md"])
        self.assertFalse(report["space_repository_parity"])

    def test_unavailable_429_release_manifest_keeps_unknown(self):
        self.remote["release.json"] = "RATE_LIMIT"
        report = self.check()
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertEqual(report["reason"], "RELEASE_MANIFEST_HTTP_429")
        self.assertFalse(report["space_repository_parity"])

    def test_unavailable_503_source_file_does_not_turn_green(self):
        self.remote["app.py"] = "RATE_LIMIT"
        def fail503(url, limit):
            if url.endswith("/app.py"):
                raise parity.RemoteUnavailable("HTTP_503")
            return self.fake_get(url, limit)
        report = parity.verify(self.root, SPACE, REVISION, fetch=fail503)
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["unavailable"], [{"path": "app.py", "reason": "UNAVAILABLE_HTTP_503"}])

    def test_manifest_duplicate_json_key_is_rejected(self):
        (self.root / "release.json").write_bytes(
            b'{"schema":"szl.space-source-release/v1","schema":"evil","source_files":{}}'
        )
        report = self.check()
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertEqual(report["reason"], "DUPLICATE_MANIFEST_KEY")

    def test_manifest_traversal_is_rejected(self):
        entry = {"schema": MANIFEST_SCHEMA,
                 "source_files": {"../config": "a" * 64}}
        (self.root / "release.json").write_text(json.dumps(entry), encoding="utf-8")
        report = self.check()
        self.assertEqual(report["reason"], "SOURCE_PATH_INVALID")

    def test_manifest_unbounded_source_file_set_is_rejected(self):
        entry = {"schema": MANIFEST_SCHEMA,
                 "source_files": {f"file{i}.py": "a" * 64 for i in range(parity.MAX_FILES + 1)}}
        (self.root / "release.json").write_text(json.dumps(entry), encoding="utf-8")
        report = self.check()
        self.assertEqual(report["reason"], "SOURCE_FILE_SET")

    def test_manifest_digest_rejects_mutable_tags(self):
        entry = {"schema": MANIFEST_SCHEMA, "source_files": {"README.md": "main"}}
        (self.root / "release.json").write_text(json.dumps(entry), encoding="utf-8")
        report = self.check()
        self.assertEqual(report["reason"], "SOURCE_DIGEST_INVALID")

    def test_rejects_unsafe_space_and_mutable_hf_revision(self):
        for sid, ref in (("other-org/a11oy", REVISION),
                         (SPACE, "main"), (SPACE, REVISION[:8]),
                         ("SZLHOLDINGS/a11oy/../private", REVISION)):
            with self.subTest(sid=sid, ref=ref):
                report = self.check(sid, ref)
                self.assertEqual(report["state"], "UNAVAILABLE")
                self.assertFalse(report["space_repository_parity"])
                self.assertIn(report["reason"], {"SPACE_ID_INVALID", "SPACE_REVISION_INVALID"})

    def test_local_symlink_is_unavailable_not_read(self):
        (self.root / "app.py").unlink()
        (self.root / "app.py").symlink_to(self.root / "README.md")
        report = self.check()
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertEqual(report["reason"], "LOCAL_SOURCE_UNAVAILABLE")

    def test_crlf_checkout_is_normalized_like_runtime(self):
        self.files["README.md"] = b"line one\nline two\n"
        self.rebuild()
        (self.root / "README.md").write_bytes(b"line one\r\nline two\r\n")
        report = self.check()
        self.assertEqual(report["state"], "REPOSITORY_BYTES_MATCH_ONLY")

    def test_remote_byte_cap_is_fail_closed(self):
        self.remote["README.md"] = b"z" * (parity.MAX_SOURCE_BYTES + 1)
        def bounded_get(url, limit):
            if url.endswith("/README.md"):
                if len(self.remote["README.md"]) > limit:
                    raise parity.RemoteUnavailable("BODY_LIMIT")
            return self.fake_get(url, limit)
        report = parity.verify(self.root, SPACE, REVISION, fetch=bounded_get)
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertFalse(report["space_repository_parity"])

    def test_http_client_is_anonymous_get_and_has_size_limit(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, length):
                self._requested_length = length
                return b"ok"

        response = FakeResponse()
        with mock.patch.object(parity.urllib.request, "urlopen", return_value=response) as open_url:
            out = parity.fetch_public_bytes(
                "https://huggingface.co/spaces/SZLHOLDINGS/szl-model-inference-lab/resolve/" + REVISION + "/README.md",
                64,
            )
        self.assertEqual(out, b"ok")
        req = open_url.call_args.args[0]
        self.assertEqual(req.get_method(), "GET")
        self.assertNotIn("Authorization", req.headers)
        self.assertEqual(response._requested_length, 65)

    def test_public_body_exact_limit_is_accepted(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, length):
                return b"A" * length

        with mock.patch.object(parity.urllib.request, "urlopen", return_value=FakeResponse()):
            with self.assertRaises(parity.RemoteUnavailable) as error:
                parity.fetch_public_bytes("https://huggingface.co/path", 32)
        self.assertEqual(error.exception.code, "BODY_LIMIT")


    def test_manifest_rejects_nul_and_trailing_slash(self):
        for name in ("evil\\x00file.py", "folder/", "  README.md", "README.md  "):
            with self.subTest(name=repr(name)):
                source = {"schema": MANIFEST_SCHEMA, "source_files": {name: "a" * 64}}
                with self.assertRaisesRegex(parity.ValidationError, "SOURCE_PATH_INVALID"):
                    parity.source_paths(source)

    def test_manifest_rejects_casefold_collision_on_windows(self):
        source = {"schema": MANIFEST_SCHEMA, "source_files": {
            "App.py": "a" * 64, "app.py": "a" * 64,
        }}
        with self.assertRaisesRegex(parity.ValidationError, "SOURCE_PATH_COLLISION"):
            parity.source_paths(source)

    def test_manifest_rejects_unicode_normalization_collision(self):
        source = {"schema": MANIFEST_SCHEMA, "source_files": {
            "caf\\u00e9.py": "a" * 64, "cafe\\u0301.py": "a" * 64,
        }}
        with self.assertRaisesRegex(parity.ValidationError, "SOURCE_PATH_INVALID"):
            parity.source_paths(source)

    def test_manifest_rejects_case_insensitive_reserved_names(self):
        for name in ("Release.JSON", ".GIT/config", "docs/.Git/objects"):
            with self.subTest(name=name):
                source = {"schema": MANIFEST_SCHEMA, "source_files": {name: "a" * 64}}
                with self.assertRaisesRegex(parity.ValidationError, "SOURCE_PATH_RESERVED"):
                    parity.source_paths(source)

    def test_malformed_manifests_never_fetch_remote(self):
        entry = {"schema": MANIFEST_SCHEMA, "source_files": {
            "README.md": "a" * 64, "Readme.md": "a" * 64,
        }}
        (self.root / "release.json").write_text(json.dumps(entry), encoding="utf-8")
        with mock.patch.object(self, "fake_get", side_effect=AssertionError("remote call")):
            report = self.check()
        self.assertEqual(report["reason"], "SOURCE_PATH_COLLISION")
        self.assertFalse(report["release_admitted"])


if __name__ == "__main__":
    unittest.main()
