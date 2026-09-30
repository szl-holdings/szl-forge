# SPDX-License-Identifier: Apache-2.0
"""Independent acceptance check; run with -I from the target collector checkout."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import unittest

sys.path.insert(0, str(Path.cwd()))
from census import file_audit as audit


class PowerShellAcceptance(unittest.TestCase):
    def test_powershell_scripts_modules_and_data_are_source_candidates(self):
        for suffix in ("ps1", "psm1", "psd1", "PS1"):
            path = "scripts/example." + suffix
            with self.subTest(path=path):
                self.assertEqual(audit.classify_path(path), "source")
                self.assertTrue(audit.is_text_candidate(path, 100, 1024))
                self.assertFalse(audit.is_text_candidate(path, 2048, 1024))

    def test_powershell_archive_content_is_inspected_not_only_hashed(self):
        path = "script.ps1"
        raw = b"# SPDX-License-Identifier: Apache-2.0\n# TODO verify input\nWrite-Output 'ok'\n"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            member = tarfile.TarInfo("root/" + path)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
        tree = [{"path": path, "type": "blob", "mode": "100644", "size": len(raw),
                 "sha": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False).hexdigest()}]
        files, findings, _, _ = audit._archive_file_records(
            "szl-holdings/fixture", "public", buffer.getvalue(), tree,
            max_text_bytes=1024, max_repository_bytes=1024,
        )
        self.assertEqual(files[0]["inspection"], "hashed-and-source-inspected")
        self.assertIn("unfinished-marker", {f["rule"] for f in findings})

    def test_unrelated_binary_and_documentation_classification_is_preserved(self):
        self.assertEqual(audit.classify_path("weights.bin"), "artifact")
        self.assertFalse(audit.is_text_candidate("weights.bin", 10, 1024))
        self.assertEqual(audit.classify_path("README.md"), "documentation")


if __name__ == "__main__":
    unittest.main()
