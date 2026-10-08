"""M6: determinism -- the same seed gives a byte-identical model.json (CONTRACT section 6).

Train + freeze is run twice from scratch on DEV data: once in this process and once in a fresh
interpreter with a different PYTHONHASHSEED.  Both model.json files (and receipts) must be
byte-identical.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from v2.research import registry
from v2.tests import _support as S

CHILD = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from v2.research import generator
from v2.tests import _support as S
S.freeze_dev(Path(sys.argv[2]), S.build(1e-3, "isotonic", tuple(generator.dev_rows(S.DEV_ROWS)))[0])
"""


class DeterminismTest(unittest.TestCase):
    def test_train_and_freeze_twice_is_byte_identical(self):
        with S.TempDir() as tmp:
            first = S.freeze_dev(tmp / "first", S.build(1e-3, "isotonic")[0])
            env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="12345", PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run(
                [sys.executable, "-B", "-c", CHILD, str(registry.ROOT), str(tmp / "second")],
                capture_output=True, text=True, env=env, timeout=600,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            a = Path(first["model_path"]).read_bytes()
            b = (tmp / "second" / "model.json").read_bytes()
            self.assertEqual(a, b)
            self.assertEqual(Path(first["receipt_path"]).read_bytes(),
                             (tmp / "second" / "artifact_receipt.json").read_bytes())
            self.assertTrue(a.endswith(b"\n"))

    def test_every_calibrator_is_deterministic_in_process(self):
        for kind in ("none", "platt", "isotonic"):
            with self.subTest(calibrator=kind), S.TempDir() as tmp:
                one = S.freeze_dev(tmp / "1", S.build(1e-2, kind)[0])
                two = S.freeze_dev(tmp / "2", S.build(1e-2, kind)[0])
                self.assertEqual(Path(one["model_path"]).read_bytes(), Path(two["model_path"]).read_bytes())
                self.assertEqual(one["model_sha256"], two["model_sha256"])

    def test_dev_rows_are_deterministic(self):
        from v2.research import generator

        a = generator.jsonl_bytes(generator.dev_rows(512))
        b = generator.jsonl_bytes(generator.dev_rows(512))
        self.assertEqual(a, b)
        self.assertEqual(generator.derive_seed("dev"),
                         int.from_bytes(hashlib.sha256(b"oac-ops-health-v2:20260926:dev").digest()[:8], "big"))


if __name__ == "__main__":
    unittest.main()
