# SPDX-License-Identifier: Apache-2.0
"""A colon-bearing pip option must remain inside a literal YAML run block."""
from pathlib import Path
import unittest


class WorkflowCommandTests(unittest.TestCase):
    def test_binary_policy_is_a_literal_run_command(self):
        text = (Path(__file__).resolve().parents[1] /
                ".github/workflows/tokenizer-qualification.yml").read_text()
        block = text.split("      - name: Install only hash-bound synthetic-control wheel\n", 1)[1]
        block = block.split("      - name:", 1)[0]
        self.assertEqual(block, "        run: |\n"
            "          python -m pip install --no-deps --only-binary=:all: --require-hashes "
            "-r eval/tokenizer-smoke-requirements.txt\n")
        self.assertIn('-p "test_tokenizer_*.py" -v', text)
        self.assertIn("'tests/test_tokenizer_*.py'", text)


if __name__ == "__main__":
    unittest.main()
