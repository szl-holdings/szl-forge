# SPDX-License-Identifier: Apache-2.0
"""SIMULATED private file fixtures; no real package, owner pin or runtime proof."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import unittest
from unittest import mock

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
import native_identity as identity


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).absolute()
        self.root = self.parent / "tree"
        self.root.mkdir()
        self.manifest = self.parent / "manifest.json"
        self.files = {"package/__init__.py": b"", "package/data.bin": b"\x00\xffopaque\x00"}
        self.make_tree(self.files)

    def make_tree(self, files):
        for name, data in files.items():
            path = self.root.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.document = {"schema": identity.MANIFEST_SCHEMA, "files": [
            {"path": name, "size_bytes": len(data), "sha256": sha(data)}
            for name, data in sorted(files.items())]}
        self.write_document()

    def write_document(self, document=None, raw=None):
        raw = canonical(self.document if document is None else document) if raw is None else raw
        self.manifest.write_bytes(raw)
        self.pin = sha(raw)

    def verify(self):
        return identity.verify_native_file_manifest(self.root, self.manifest,
                                                    expected_manifest_sha256=self.pin)

    def assert_denied(self, code=None):
        with self.assertRaises(identity.NativeIdentityError) as caught:
            self.verify()
        if code is not None:
            self.assertEqual(caught.exception.code, code)
        return caught.exception.code

    def test_exact_opaque_bytes_and_empty_files(self):
        result = self.verify()
        self.assertEqual(result, identity.NativeFileIdentity(self.pin, 2, sum(map(len, self.files.values()))))
        with self.assertRaises(AttributeError):
            result.file_count = 100
        self.assertFalse(hasattr(result, "training_eligible"))

    def test_external_pin_required_and_self_pin_field_rejected(self):
        for value in (None, "", "F" * 64, "f" * 63, "f" * 64 + "\n", True):
            with self.subTest(value=value), self.assertRaisesRegex(
                    identity.NativeIdentityError, "EXTERNAL_MANIFEST_PIN_REQUIRED"):
                identity.verify_native_file_manifest(self.root, self.manifest,
                                                     expected_manifest_sha256=value)
        self.pin = "f" * 64
        self.assert_denied("MANIFEST_PIN_MISMATCH")
        self.document["expected_manifest_sha256"] = self.pin
        self.write_document()
        self.assert_denied("MANIFEST_FIELDS_INVALID")

    def test_profile_cannot_be_container_or_admission_document(self):
        for schema in ("szl.receiptagent.v4-training-admission/v1", "OCI_IMAGE", ""):
            with self.subTest(schema=schema):
                self.write_document({**self.document, "schema": schema})
                self.assert_denied("MANIFEST_FIELDS_INVALID")

    def test_duplicate_fields_and_noncanonical_bytes_rejected(self):
        self.write_document(raw=b'{"schema":"x","schema":"y","files":[]}')
        self.assert_denied("DUPLICATE_JSON_FIELD")
        self.write_document(raw=canonical(self.document) + b"\n")
        self.assert_denied("NONCANONICAL_MANIFEST")

    def test_noninteger_sizes_and_unbounded_numbers_rejected(self):
        for value in (True, -1, 0.5, "0", identity.MAX_SINGLE_FILE_BYTES + 1):
            with self.subTest(value=value):
                document = json.loads(canonical(self.document))
                document["files"][0]["size_bytes"] = value
                self.write_document(document)
                self.assert_denied()
        for token in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            with self.subTest(token=token):
                self.write_document(raw=canonical(self.document).replace(b'"size_bytes":0', b'"size_bytes":' + token))
                self.assert_denied("INTEGER_ONLY_MANIFEST")

    def test_path_traversal_alias_and_portability_rejections(self):
        names = ("../outside", "/outside", "C:/outside", "package\\data.bin", "a//b", "a/./b",
                 "a/../b", "a/", "CON.txt", "aux", "x.", "space name", "nonascii\u00e9", "x\x00y",
                 "/".join(["a"] * (identity.MAX_DEPTH + 1)))
        for name in names:
            with self.subTest(name=name):
                document = json.loads(canonical(self.document))
                document["files"][0]["path"] = name
                self.write_document(document)
                self.assert_denied("RELATIVE_PATH_INVALID")

    def test_duplicate_case_alias_directory_collision_and_order(self):
        zero = {"size_bytes": 0, "sha256": sha(b"")}
        for names in (("x", "x"), ("X", "x"), ("a", "a/b"), ("A/x", "a/y"), ("z", "a")):
            with self.subTest(names=names):
                self.write_document({"schema": identity.MANIFEST_SCHEMA,
                                     "files": [{"path": name, **zero} for name in names]})
                self.assert_denied()

    def test_missing_extra_and_extra_empty_directory_rejected(self):
        data = self.root / "package/data.bin"
        data.unlink()
        self.assert_denied("TREE_INVENTORY_MISMATCH")
        data.write_bytes(self.files["package/data.bin"])
        extra = self.root / "extra.bin"
        extra.write_bytes(b"x")
        self.assert_denied("TREE_INVENTORY_MISMATCH")
        extra.unlink()
        (self.root / "empty").mkdir()
        self.assert_denied("TREE_INVENTORY_MISMATCH")

    def test_same_size_changed_bytes_and_wrong_declared_size_rejected(self):
        data = self.root / "package/data.bin"
        data.write_bytes(b"x" * len(self.files["package/data.bin"]))
        self.assert_denied("FILE_DIGEST_MISMATCH")
        data.write_bytes(b"longer data than before")
        self.assert_denied("FILE_SIZE_MISMATCH")

    def test_hardlink_tree_and_manifest_rejected(self):
        try:
            os.link(self.root / "package/data.bin", self.parent / "outside-link")
        except (OSError, NotImplementedError) as exc:
            self.skipTest("Hardlinks unavailable: " + type(exc).__name__)
        self.assert_denied("REGULAR_UNALIASED_FILE_REQUIRED")
        (self.parent / "outside-link").unlink()
        os.link(self.manifest, self.parent / "manifest-link")
        self.assert_denied("REGULAR_UNALIASED_FILE_REQUIRED")

    def test_symlink_tree_root_and_manifest_rejected(self):
        link = self.parent / "tree-link"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("Symlinks unavailable: " + type(exc).__name__)
        with self.assertRaisesRegex(identity.NativeIdentityError, "PATH_ALIAS_FORBIDDEN"):
            identity.verify_native_file_manifest(link, self.manifest, expected_manifest_sha256=self.pin)
        link.unlink()
        (self.root / "link").symlink_to(self.parent / "outside")
        self.assert_denied("PATH_ALIAS_FORBIDDEN")
        (self.root / "link").unlink()
        target = self.parent / "real-manifest"
        self.manifest.rename(target)
        self.manifest.symlink_to(target)
        self.assert_denied("PATH_ALIAS_FORBIDDEN")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX special-file fixture")
    def test_fifo_rejected_without_opening_it(self):
        os.mkfifo(self.root / "fifo")
        self.assert_denied("REGULAR_UNALIASED_FILE_REQUIRED")

    def test_manifest_must_be_outside_tree_and_paths_absolute(self):
        inside = self.root / "manifest.json"
        inside.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(identity.NativeIdentityError, "MANIFEST_MUST_BE_OUTSIDE_TREE"):
            identity.verify_native_file_manifest(self.root, inside, expected_manifest_sha256=self.pin)
        with self.assertRaisesRegex(identity.NativeIdentityError, "ABSOLUTE_PATH_REQUIRED"):
            identity.verify_native_file_manifest(Path("relative"), self.manifest,
                                                 expected_manifest_sha256=self.pin)
        with self.assertRaisesRegex(identity.NativeIdentityError, "PATH_TRAVERSAL_FORBIDDEN"):
            identity.verify_native_file_manifest(self.parent / "tree" / ".." / "tree", self.manifest,
                                                 expected_manifest_sha256=self.pin)

    def test_caps_fail_before_unbounded_reads(self):
        with mock.patch.object(identity, "MAX_MANIFEST_BYTES", 1):
            self.assert_denied("FILE_BYTE_CAP")
        with mock.patch.object(identity, "MAX_FILES", 1):
            self.assert_denied("MANIFEST_FILE_COUNT_INVALID")
        with mock.patch.object(identity, "MAX_TOTAL_FILE_BYTES", 1):
            self.assert_denied("TREE_BYTE_CAP")
        with mock.patch.object(identity, "MAX_TREE_ENTRIES", 1):
            self.assert_denied("TREE_ENTRY_CAP")
        with mock.patch.object(identity, "MAX_SINGLE_FILE_BYTES", 1):
            self.assert_denied("MANIFEST_FILE_SIZE_INVALID")

    def test_multiple_chunks_are_streamed_and_not_retained(self):
        data = b"A" * (3 * identity.HASH_CHUNK_BYTES + 17)
        self.files["package/data.bin"] = data
        self.make_tree(self.files)
        observed_sizes = []
        retained = []
        original_read = identity.os.read
        original_capture = identity._capture_file

        def bounded_read(fd, maximum):
            observed_sizes.append(maximum)
            return original_read(fd, maximum)

        def capture(path, **kwargs):
            result = original_capture(path, **kwargs)
            retained.append((path, result[1]))
            return result

        with mock.patch.object(identity.os, "read", side_effect=bounded_read), mock.patch.object(
                identity, "_capture_file", side_effect=capture):
            self.assertEqual(self.verify().total_bytes, len(data))
        self.assertGreater(len(observed_sizes), 4)
        self.assertLessEqual(max(observed_sizes), identity.HASH_CHUNK_BYTES)
        self.assertTrue(all(raw is None for path, raw in retained if path != self.manifest))
        self.assertGreaterEqual(identity.MAX_SINGLE_FILE_BYTES, 8 * 1024**3)
        self.assertGreaterEqual(identity.MAX_TOTAL_FILE_BYTES, 64 * 1024**3)

    def test_file_changes_during_stream_are_rejected(self):
        data = self.root / "package/data.bin"
        original_read = identity.os.read
        calls = 0

        def mutate(fd, maximum):
            nonlocal calls
            result = original_read(fd, maximum)
            calls += 1
            if calls == 1:
                with data.open("ab") as output:
                    output.write(b"changed")
            return result

        with mock.patch.object(identity.os, "read", side_effect=mutate):
            with self.assertRaises(identity.NativeIdentityError):
                identity._capture_file(data, maximum=identity.MAX_SINGLE_FILE_BYTES)

    def test_inventory_change_after_hashing_is_rejected(self):
        original_capture = identity._capture_file
        calls = 0

        def mutate(path, **kwargs):
            nonlocal calls
            result = original_capture(path, **kwargs)
            calls += 1
            if calls == 3:
                (self.root / "injected.py").write_bytes(b"raise RuntimeError('must not execute')")
            return result

        with mock.patch.object(identity, "_capture_file", side_effect=mutate):
            self.assert_denied("TREE_CHANGED_DURING_READ")

    def test_manifest_change_after_capture_is_rejected(self):
        original_inventory = identity._inventory
        calls = 0

        def mutate(path):
            nonlocal calls
            result = original_inventory(path)
            calls += 1
            if calls == 2:
                self.manifest.write_bytes(b"modified")
            return result

        with mock.patch.object(identity, "_inventory", side_effect=mutate):
            self.assert_denied("MANIFEST_CHANGED_DURING_READ")

    def test_pth_startup_code_and_shared_library_bytes_are_never_executed(self):
        sentinel = self.parent / "executed"
        self.files.update({
            "evil.pth": ("import pathlib; pathlib.Path(" + repr(str(sentinel)) + ").touch()\n").encode(),
            "sitecustomize.py": b"raise RuntimeError('must never import')\n",
            "package/native.so": b"not an executable library",
            "sample-1.0.dist-info/RECORD": b"untrusted data,not wheel provenance,0\n",
        })
        self.make_tree(self.files)
        script = (
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); import native_identity as n; "
            "r=n.verify_native_file_manifest(Path(sys.argv[2]),Path(sys.argv[3]),"
            "expected_manifest_sha256=sys.argv[4]); "
            "print(r.file_count); "
            "sys.exit(0 if 'site' not in sys.modules and 'torch' not in sys.modules "
            "and 'transformers' not in sys.modules and 'sitecustomize' not in sys.modules else 9)"
        )
        result = subprocess.run((sys.executable, "-I", "-S", "-B", "-c", script, str(HERE),
                                 str(self.root), str(self.manifest), self.pin),
                                capture_output=True, text=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(len(self.files)))
        self.assertFalse(sentinel.exists())

    def test_missing_input_returns_value_free_failure(self):
        self.manifest.unlink()
        self.assert_denied("FILESYSTEM_UNAVAILABLE")

    def test_missing_path_absent_from_normal_formatted_traceback(self):
        missing = self.parent / "private-sentinel-path-must-not-appear.json"
        try:
            identity.verify_native_file_manifest(self.root, missing,
                                                 expected_manifest_sha256=self.pin)
        except identity.NativeIdentityError as error:
            rendered = "".join(traceback.format_exception(error))
            self.assertEqual(error.code, "FILESYSTEM_UNAVAILABLE")
            self.assertNotIn(str(missing), rendered)
            self.assertNotIn(missing.name, rendered)
            self.assertTrue(error.__suppress_context__)
        else:
            self.fail("Missing input must not return an identity")


if __name__ == "__main__":
    unittest.main()
