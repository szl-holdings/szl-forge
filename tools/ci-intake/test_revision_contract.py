"""Exact revision admission at every boundary, using original synthetic data."""

import json
import unittest

from offline_ci_intake import parse_and_normalize
from test_ci_intake import fixture


class RevisionContractTests(unittest.TestCase):
    def check_revision(self, revision, valid):
        # Change all three pins together: inequality must not hide bad syntax.
        value = fixture()
        value["policy"]["source_revision"] = revision
        value["intake"]["source_revision"] = revision
        value["intake"]["results"][0]["source_revision"] = revision
        report = parse_and_normalize(json.dumps(value))
        self.assertEqual(report["proposed_pilot_label"],
                         "READY_FOR_REVIEW" if valid else "INSUFFICIENT_EVIDENCE")
        self.assertFalse(report["authorized"])
        self.assertFalse(report["execution_permission"])
        self.assertFalse(report["release_permission"])

    def test_full_lowercase_sha1_and_sha256_are_admitted(self):
        for revision in ("0123456789abcdef" * 2 + "01234567", "0123456789abcdef" * 4):
            with self.subTest(revision=revision):
                self.check_revision(revision, True)

    def test_short_long_and_intermediate_lengths_are_refused(self):
        for size in (0, 7, 39, 41, 63, 65):
            with self.subTest(size=size):
                self.check_revision("a" * size, False)

    def test_uppercase_and_mixed_case_are_refused(self):
        for revision in ("A" * 40, "A" + "a" * 39, "A" * 64, "a" * 63 + "F"):
            with self.subTest(revision=revision):
                self.check_revision(revision, False)

    def test_symbolic_refs_are_refused(self):
        for revision in ("HEAD", "main", "refs/heads/main", "v1.0.0"):
            with self.subTest(revision=revision):
                self.check_revision(revision, False)

    def test_nonhex_and_unicode_lookalikes_are_refused(self):
        for revision in ("g" * 40, "a" * 39 + "-", "\uff41" * 40, "\u0430" * 64):
            with self.subTest(revision=revision):
                self.check_revision(revision, False)

    def test_whitespace_and_trailing_newline_are_refused(self):
        for revision in (" " + "a" * 40, "a" * 40 + " ", "a" * 40 + "\n", "a" * 64 + "\r\n"):
            with self.subTest(revision=revision):
                self.check_revision(revision, False)

    def test_nonstring_revisions_are_refused(self):
        for revision in (None, True, 123, [], {}):
            with self.subTest(revision=revision):
                self.check_revision(revision, False)

    def test_each_boundary_independently_refuses_a_malformed_revision(self):
        for boundary in ("policy", "intake", "result"):
            with self.subTest(boundary=boundary):
                value = fixture()
                target = value[boundary] if boundary != "result" else value["intake"]["results"][0]
                target["source_revision"] = "a" * 40 + "\n"
                report = parse_and_normalize(json.dumps(value))
                self.assertEqual(report["proposed_pilot_label"], "INSUFFICIENT_EVIDENCE")
                expected = {"policy": "INVALID_POLICY", "intake": "INVALID_INTAKE", "result": "MALFORMED_RESULT"}[boundary]
                self.assertIn(expected, {f["code"] for f in report["findings"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
