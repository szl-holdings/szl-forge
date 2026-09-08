import hashlib
import tempfile
import unittest
from pathlib import Path

from verify_snapshot import file_hashes


class SnapshotHashes(unittest.TestCase):
    def test_git_blob_and_content_are_distinct_correct_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture"
            path.write_bytes(b"synthetic\n")
            content, blob = file_hashes(path)
            self.assertEqual(content, hashlib.sha256(b"synthetic\n").hexdigest())
            self.assertEqual(blob, hashlib.sha1(b"blob 10\0synthetic\n").hexdigest())


if __name__ == "__main__":
    unittest.main()
