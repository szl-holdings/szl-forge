"""Unit tests for acceptance.py (stdlib unittest; synthetic; spy model only)."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acceptance as acc  # noqa: E402
import eight_case_harness as h  # noqa: E402

FROZEN = h.load_frozen(HERE)


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


class AcceptanceTests(unittest.TestCase):
    def _copy(self):
        d = tempfile.mkdtemp(prefix="e2d_t_", dir=HERE)
        self.addCleanup(shutil.rmtree, d, True)
        for n in acc.FROZEN_INPUTS + ("guard_cases.json", "guard_answer_key.json", "GUARD_FREEZE.json"):
            shutil.copy(os.path.join(HERE, n), d)
        shutil.copytree(os.path.join(HERE, "corpus"), os.path.join(d, "corpus"))
        return d

    def test_end_to_end_green(self):
        self.assertEqual(quiet(acc.main, ["--dir", HERE]), 0)

    def test_guard_key_tamper_refused(self):
        d = self._copy()
        p = os.path.join(d, "guard_answer_key.json")
        with open(p, encoding="utf-8") as f:
            k = json.load(f)
        k["cases"]["E8-G5"]["expected_decision"] = "ANSWER"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(k, f)
        self.assertEqual(quiet(acc.main, ["--dir", d]), 3)

    def test_guard_cases_tamper_refused(self):
        d = self._copy()
        with open(os.path.join(d, "guard_cases.json"), "a", encoding="utf-8", newline="\n") as f:
            f.write(" ")
        with self.assertRaises(h.FreezeError):
            acc.load_guard(d)

    def _spy(self, script):
        w = acc.GuardWorld(FROZEN)
        w.spy.script = script
        prompt = h.TEMPLATE.format(q="How often does Lumen-Gate rotate its session keys?",
                                   evidence=FROZEN["corpus"]["TA-D01"]["text"])
        return w.spy.generate(prompt)

    def test_spy_scripts(self):
        self.assertEqual(self._spy(None), "Lumen-Gate rotates its synthetic session keys every 36 hours. [cite:TA-D01]")
        self.assertEqual(self._spy("omit_citation"), "Lumen-Gate rotates its synthetic session keys every 36 hours.")
        self.assertTrue(self._spy("cite_outside_evidence:TB-D02").endswith("[cite:TB-D02]"))
        self.assertIn("reportedly", self._spy("paraphrase_excerpt"))

    def test_guard_world_routes_dispatch_to_guard_spy(self):
        w = acc.GuardWorld(FROZEN)
        self.assertIs(w.ctl.provider, w.spy)
        self.assertIsInstance(w.spy, acc.GuardSpy)

    def test_exception_counts_as_fail(self):
        rows = [{"case_id": "E8-G1", "error": "KeyError", "_world": None}]
        key = json.loads(h._read(os.path.join(HERE, "guard_answer_key.json")).decode("utf-8"))
        self.assertFalse(acc.score_guard(FROZEN, key, rows)[0]["pass"])


if __name__ == "__main__":
    unittest.main()
