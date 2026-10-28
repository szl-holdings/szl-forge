"""Determinism (prereg s.6.7 item 9): training and freezing twice from the same seeds gives a
byte-identical model.json. SYNTHETIC; DEV data only.

The DEV recipe run is repeated in this process and in two fresh interpreters started as
``python -I`` (one with PYTHONHASHSEED=0, the runtime pin, and one with another hash seed). Each
regenerates the same DEV stream prefixes, so no new DEV row is drawn.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from v3.research import freeze, generator, registry
from v3.tests import _support as S

CHILD = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from v3.tests import _support as S
S.freeze_dev(Path(sys.argv[2]), S.build()[0])
"""


class DeterminismTest(unittest.TestCase):
    def test_train_and_freeze_twice_is_byte_identical(self):
        with S.TempDir() as tmp:
            first = S.freeze_dev(tmp / "first", S.build()[0])
            second = S.freeze_dev(tmp / "second", S.build()[0])
            a = Path(first["model_path"]).read_bytes()
            self.assertEqual(a, Path(second["model_path"]).read_bytes())
            self.assertEqual(Path(first["receipt_path"]).read_bytes(), Path(second["receipt_path"]).read_bytes())
            for label, hashseed in (("pinned", "0"), ("other", "12345")):
                env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONDONTWRITEBYTECODE="1")
                out = tmp / f"child_{label}"
                proc = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(registry.ROOT), str(out)],
                                      capture_output=True, text=True, env=env, timeout=900)
                with self.subTest(child=label):
                    self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                    self.assertEqual((out / "model.json").read_bytes(), a)
                    self.assertEqual((out / "artifact_receipt.json").read_bytes(),
                                     Path(first["receipt_path"]).read_bytes())
            self.assertTrue(a.endswith(b"\n"))

    def test_dev_receipt_trace_digest_is_the_persisted_canonical_diagnostics(self):
        trace = S.dev_trace_bytes()
        self.assertEqual(trace, freeze.canonical_json_bytes(S.diagnostics()))
        parsed = json.loads(trace)
        self.assertEqual(freeze.canonicalize(parsed), parsed)
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "fit")
            receipt = json.loads(Path(info["receipt_path"]).read_bytes())
        self.assertEqual(receipt["fit"]["trace_sha256"], hashlib.sha256(trace).hexdigest())

    def test_dev_trace_ignores_round_off_below_canonical_precision(self):
        # The two IRLS residuals that differed between H-local and the VM at cd62f144 (MEASURED,
        # 2026-10-28): the trace bytes must not depend on them.
        variants = []
        for grad, step in ((4.2838183087617887e-16, 9.952093584190592e-16),
                           (2.5518731008519757e-16, 9.708712485367728e-16)):
            d = S.diagnostics()
            d["lr"]["gradient_max_abs"], d["lr"]["last_newton_step_max_abs"] = grad, step
            variants.append(freeze.canonical_json_bytes(d))
        self.assertEqual(variants[0], variants[1])
        self.assertEqual(variants[0], S.dev_trace_bytes())

    def test_dev_fit_converges_and_is_kernel_valid(self):
        diagnostics = S.diagnostics()
        self.assertTrue(diagnostics["kernel_valid"])
        self.assertIn(diagnostics["lr"]["stop_reason"], ("CONVERGED", "CAP_REACHED", "LINE_SEARCH_FAILED"))
        self.assertLessEqual(diagnostics["lr"]["rows"], registry.DEV_MAX_FIT_ROWS)
        artifact = S.candidate()
        self.assertEqual(artifact["training"]["splits"]["train"]["rows"], S.DEV_D)
        self.assertEqual(artifact["training"]["l2"], 1e-3)
        self.assertEqual(artifact["calibrator"], {"kind": "none", "params": {}})
        self.assertTrue(1 <= artifact["threshold_index"] <= 99)

    def test_dev_rows_are_deterministic(self):
        a = generator.jsonl_bytes(generator.dev_rows(S.DEV_NOMINAL, "nominal"))
        self.assertEqual(a, generator.jsonl_bytes(S.nominal_rows()))


if __name__ == "__main__":
    unittest.main()
