import unittest
from benchmark_ollama import CASES, admit_model, grade, local_origin, same_typed, strict_json


class Contracts(unittest.TestCase):
    def test_loopback_origins(self):
        for value in ("http://127.0.0.1:11434", "http://[::1]:11434/"):
            self.assertEqual(local_origin(value), value.rstrip("/"))

    def test_no_remote_hostname_credentials_path_or_redirect_origin(self):
        for value in ("https://example.com", "http://127.evil.test:11434", "http://localhost:11434",
                      "http://127.0.0.1:11434/path", "http://u:p@127.0.0.1:11434", "http://127.0.0.1:11434?x=y"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                local_origin(value)

    def test_exact_typed_grading(self):
        self.assertFalse(same_typed(True, 1))
        self.assertFalse(same_typed({"measured": 0}, {"measured": False}))
        self.assertFalse(same_typed([1, 2.0], [1, 2]))
        self.assertFalse(same_typed({"x": 1, "extra": 2}, {"x": 1}))
        self.assertTrue(same_typed([1, False, {"x": "y"}], [1, False, {"x": "y"}]))

    def test_no_markdown_or_prose_cleanup(self):
        self.assertTrue(grade(CASES[0], "102"))
        self.assertFalse(grade(CASES[0], "The answer is 102."))
        self.assertFalse(grade(CASES[1], "```json\n[-3,2,8]\n```"))
        self.assertTrue(grade(CASES[1], "[-3,2,8]"))

    def test_local_artifacts_required(self):
        entry = {"digest": "a" * 64, "size": 10}
        show = {"details": {"format": "gguf"}}
        admit_model(entry, show)
        for remote in ("remote_host", "remote_model", "remote_name"):
            with self.assertRaises(ValueError):
                admit_model(entry, {**show, remote: "cloud"})
        for bad in ({"digest": "abc", "size": 10}, {"digest": "a" * 64, "size": 0}):
            with self.assertRaises(ValueError):
                admit_model(bad, show)

    def test_ambiguous_and_nonstandard_json_rejected(self):
        for raw in ('{"measured":true,"measured":false}', 'NaN', 'Infinity', '-Infinity'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)
        self.assertFalse(grade(CASES[2], '{"status":"UNKNOWN","measured":true,"measured":false}'))


if __name__ == "__main__":
    unittest.main()
