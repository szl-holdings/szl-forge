"""Offline source-package tests. Never runs model-generated code or services.
SPDX-License-Identifier: Apache-2.0
"""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import build_release as bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="szl-bundle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {"lab.py": "# é\r\nprint('not executed')\r\n", "tasks/a.json": "{}\n"}

    def test_installer_preserves_exact_source_bytes(self):
        dest = bundle.install_files(self.root / "new", self.files, bundle.hashes_for(self.files))
        for name, text in self.files.items():
            self.assertEqual((dest / name).read_bytes(), text.encode("utf-8"))
        self.assertEqual(json.loads((dest / "SOURCE_MANIFEST.json").read_text())["files"],
                         bundle.hashes_for(self.files))

    def test_existing_destination_is_untouched(self):
        dest = self.root / "existing"
        dest.mkdir()
        original = dest / "keep.txt"
        original.write_text("user work")
        with self.assertRaises(FileExistsError):
            bundle.install_files(dest, self.files, bundle.hashes_for(self.files))
        self.assertEqual(original.read_text(), "user work")
        self.assertEqual(len(list(dest.iterdir())), 1)

    def test_tampered_bytes_fail_before_directory_creation(self):
        hashes = bundle.hashes_for(self.files)
        self.files["lab.py"] += "# changed"
        with self.assertRaises(ValueError):
            bundle.install_files(self.root / "new", self.files, hashes)
        self.assertFalse((self.root / "new").exists())

    def test_unsafe_paths_fail_before_directory_creation(self):
        for name in ("../escape.py", "/absolute.py", "C:/outside.py", "a\\b", "CON", "a./x", "a//x"):
            files = {name: "data"}
            with self.subTest(name=name), self.assertRaises(ValueError):
                bundle.install_files(self.root / "new", files, bundle.hashes_for(files))
            self.assertFalse((self.root / "new").exists())

    def test_zip_is_deterministic_and_manifest_matches_every_byte(self):
        data = bundle.zip_bytes(self.files)
        self.assertEqual(data, bundle.zip_bytes(dict(reversed(list(self.files.items())))))
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
            self.assertEqual(set(archive.namelist()), set(self.files) | {"SOURCE_MANIFEST.json"})
            for name, expected in manifest["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)

    def test_payload_is_one_valid_python_block_and_reuses_verified_installer(self):
        markdown = bundle.payload(self.files)
        self.assertEqual(markdown.count("~~~~python\n"), 1)
        source = markdown.split("~~~~python\n", 1)[1].rsplit("~~~~\n", 1)[0]
        # This executes only our trusted installer definition; embedded candidate
        # source is inert string data, never imported or executed.
        namespace = {"__name__": "payload_test"}
        exec(compile(source, "trusted_installer", "exec"), namespace)
        dest = namespace["install_files"](self.root / "new", namespace["FILES"], namespace["HASHES"])
        self.assertEqual((dest / "lab.py").read_bytes(), self.files["lab.py"].encode("utf-8"))

    def test_collect_excludes_runtime_and_unknown_files(self):
        for name in bundle.NAMES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        (self.root / "runtime").mkdir()
        (self.root / "runtime" / "secret.log").write_text("not exported")
        (self.root / "unrelated.txt").write_text("not exported")
        self.assertEqual(set(bundle.collect(self.root)), set(bundle.NAMES))

    def test_hosted_link_is_explicit_and_outside_the_installer(self):
        markdown = bundle.payload(self.files)
        introduction, source = markdown.split("~~~~python\n", 1)
        self.assertIn("https://szlholdings-szl-model-inference-lab.hf.space/#run-lab", introduction)
        self.assertIn("separate hosted Khipu 1.5B demonstration", introduction)
        self.assertIn("not this local Qwen3-4B lab", introduction)
        self.assertIn("do not submit credentials or sensitive data", introduction)
        self.assertIn("32 generated tokens", introduction)
        self.assertIn("do not contact it or fall back to remote inference", introduction)
        self.assertNotIn("https://szlholdings-szl-model-inference-lab.hf.space", source)
        self.assertEqual(markdown.count("~~~~python\n"), 1)

    def test_missing_allowlisted_source_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            bundle.collect(self.root)


if __name__ == "__main__":
    unittest.main()
