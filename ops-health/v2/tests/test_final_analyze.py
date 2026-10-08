"""final_analyze.py: the registered win rule, and an end-to-end run on DEV stand-in scores.

DEV data only.  The score files come from ``final_score.dry_run`` in a temporary directory; the
analysis reads only those files (an audit hook fails the test if a file under v2/data/ or the real
runs/TEST_OPENED.json is opened during the analysis).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import unittest

from v2.research import final_analyze as FA
from v2.research import final_score, registry
from v2.tests import _support as S


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


_FORBIDDEN = tuple(_norm(p) for p in (registry.DATA_DIR,
                                      registry.RUNS_DIR / registry.TEST_OPENED_NAME))
_WATCH = {"active": False, "touched": []}


def _audit(event: str, args) -> None:
    if not _WATCH["active"] or event != "open" or not args:
        return
    if not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = _norm(args[0])
    for forbidden in _FORBIDDEN:
        if path == forbidden or path.startswith(forbidden + os.sep):
            _WATCH["touched"].append(path)
            raise PermissionError(f"analysis touched a protected path: {path}")


sys.addaudithook(_audit)
QUIET = lambda line: None  # noqa: E731


def _set(ba: dict[str, tuple[float, float]], brier=(0.10, 0.12), ece=(0.02, 0.05)) -> dict:
    """A minimal set result: ba = {contender: (lower, upper)}; brier/ece = (v2, v1)."""
    intervals = {f"ba:{c}": {"point": (lo + hi) / 2, "lower": lo, "upper": hi}
                 for c, (lo, hi) in ba.items()}
    return {
        "bootstrap": {"intervals": intervals},
        "contenders": {"v2": {"score_report": {"brier": brier[0], "ece": ece[0]}},
                       "v1": {"score_report": {"brier": brier[1], "ece": ece[1]}}},
    }


GOOD = {"v2": (0.76, 0.78), "v1": (0.74, 0.75), "majority": (0.5, 0.5), "rule": (0.55, 0.57)}


class WinRuleTest(unittest.TestCase):
    def test_all_three_hold(self):
        v = FA.win_conditions(_set(GOOD), _set(GOOD))
        self.assertEqual(v["result"], "WIN")
        self.assertEqual(v["failed"], [])

    def test_overlap_on_test_fails_w1(self):
        v = FA.win_conditions(_set({**GOOD, "v1": (0.74, 0.765)}), _set(GOOD))
        self.assertEqual(v["result"], "DOES NOT WIN")
        self.assertEqual(v["failed"], ["W1"])
        self.assertFalse(v["W1"]["comparisons"]["v1"]["holds"])

    def test_touching_bounds_fail_because_the_rule_is_strict(self):
        v = FA.win_conditions(_set(GOOD), _set({**GOOD, "rule": (0.70, 0.76)}))
        self.assertEqual(v["failed"], ["W2"])
        self.assertEqual(v["W2"]["comparisons"]["rule"]["margin"], 0.0)

    def test_calibration_ties_hold_and_worse_calibration_fails(self):
        self.assertEqual(FA.win_conditions(_set(GOOD, brier=(0.1, 0.1), ece=(0.03, 0.03)),
                                           _set(GOOD))["result"], "WIN")
        v = FA.win_conditions(_set(GOOD, ece=(0.06, 0.05)), _set(GOOD))
        self.assertEqual(v["failed"], ["W3"])
        self.assertTrue(v["W3"]["brier"]["holds"])
        self.assertFalse(v["W3"]["ece"]["holds"])

    def test_undefined_interval_never_wins(self):
        broken = _set(GOOD)
        broken["bootstrap"]["intervals"]["ba:v2"]["lower"] = None
        self.assertEqual(FA.win_conditions(broken, _set(GOOD))["failed"], ["W1"])


class FinalAnalyzeDevTest(unittest.TestCase):
    @classmethod
    @unittest.skipUnless(S.origin_commit_present(), S.ORIGIN_SKIP_REASON)
    def setUpClass(cls):
        cls._tmp = S.TempDir()
        cls.base = cls._tmp.__enter__()
        final_score.dry_run(cls.base / "run", test_rows=240, shift_rows=100, log=QUIET)
        cls.score_dir = cls.base / "run" / "final"
        _WATCH["touched"].clear()
        _WATCH["active"] = True
        try:
            cls.ev = FA.analyze(cls.score_dir, parts_dir=cls.base / "parts", resamples=60, log=QUIET)
            cls.again = FA.analyze(cls.score_dir, parts_dir=cls.base / "parts", resamples=60,
                                   log=QUIET)
            cls.written = FA.write_outputs(cls.ev, out_json=cls.base / "out" / "e.json",
                                           out_md=cls.base / "out" / "r.md",
                                           out_svg=cls.base / "out" / "r.svg")
        finally:
            _WATCH["active"] = False

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def test_reads_no_data_split_and_no_real_marker(self):
        self.assertEqual(_WATCH["touched"], [])

    def test_structure_and_flags(self):
        ev = self.ev
        self.assertEqual(ev["mode"], "DRY_RUN_DEV_STANDIN")
        self.assertFalse(ev["registered_protocol"])
        self.assertEqual(list(ev["sets"]), [FA.TEST, FA.POOLED, *ev["bootstrap_protocol"]
                                            ["pooled_shift_strata"], FA.LEGACY])
        self.assertEqual(ev["sets"][FA.POOLED]["rows"], 400)
        self.assertEqual(ev["sets"][FA.POOLED]["bootstrap"]["strata"], [100, 100, 100, 100])
        self.assertIn(ev["verdict"]["result"], ("WIN", "DOES NOT WIN"))
        for s, r in ev["sets"].items():
            with self.subTest(set=s):
                maj = r["bootstrap"]["intervals"]["ba:majority"]
                self.assertEqual((maj["point"], maj["lower"], maj["upper"]), (0.5, 0.5, 0.5))
                d = r["paired_differences"]["v2_minus_v1"]["ba"]
                iv = r["bootstrap"]["intervals"]
                self.assertAlmostEqual(d["point"], iv["ba:v2"]["point"] - iv["ba:v1"]["point"])
                self.assertEqual(iv["ba:v2"]["n_defined"] + iv["ba:v2"]["n_undefined"], 60)

    def test_invalid_baselines_carry_no_per_class_rates(self):
        for s, r in self.ev["sets"].items():
            for c in ("majority", "rule", "ABL-optimizer"):
                with self.subTest(set=s, contender=c):
                    br = r["contenders"][c]["binary_report"]
                    self.assertEqual(br["per_class_rates"], "INVALID_BASELINE")
                    self.assertNotIn("confusion", br)
                    self.assertNotIn("accuracy", br)

    def test_rerun_reuses_saved_parts_and_is_identical(self):
        a = copy.deepcopy(self.ev)
        b = copy.deepcopy(self.again)
        for x in (a, b):
            x.pop("generated_utc")
        self.assertEqual(a, b)
        self.assertEqual(sorted(p.name for p in (self.base / "parts").iterdir()),
                         sorted(f"{s}.json" for s in self.ev["sets"]))

    def test_outputs_render_without_the_banned_word(self):
        needle = "clin" + "ical"
        for name in ("e.json", "r.md", "r.svg"):
            text = (self.base / "out" / name).read_text(encoding="utf-8")
            self.assertNotIn(needle, text.lower(), name)
        md = (self.base / "out" / "r.md").read_text(encoding="utf-8")
        self.assertIn("DEV DRY RUN", md)
        self.assertIn(f"## Verdict: v2 **{self.ev['verdict']['result']}**", md)
        svg = (self.base / "out" / "r.svg").read_text(encoding="utf-8")
        self.assertTrue(svg.startswith("<svg") and svg.rstrip().endswith("</svg>"))

    def test_tampered_score_file_is_refused(self):
        with S.TempDir() as tmp:
            copy_dir = tmp / "final"
            shutil.copytree(self.score_dir, copy_dir)
            victim = copy_dir / "scores_test.jsonl"
            data = bytearray(victim.read_bytes())
            data[10] ^= 1
            victim.write_bytes(bytes(data))
            with self.assertRaises(FA.AnalysisError):
                FA.load_scores(copy_dir)

    def test_truth_recovery_is_pinned_to_the_manifest(self):
        tr = self.ev["truth_recovery"]
        manifest = json.loads((self.score_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        for c, rel in FA.MODEL_FILES.items():
            self.assertEqual(tr[c]["sha256"], manifest["contenders"][c]["files"][rel])
            self.assertIn(tr[c]["sign_agreement_k"], range(9))

    # -- render-only additions (made after the opening; they change no number) -------------
    def _robustness(self) -> dict:
        return {
            "inputs": {"score_manifest_sha256": self.ev["inputs"]["score_manifest_sha256"],
                       "final_evaluation_sha256": "0" * 64,
                       "script": "logs/v2/final_verify_independent.py", "numpy": "x",
                       "python": "3.11"},
            "part_A_B": {"max_abs_diff_vs_evaluation": 1e-17, "tolerance": 1e-12,
                         "independent_verdict": self.ev["verdict"]["result"]},
            "part_C_post_hoc_sensitivity": {
                "streams_count": 100, "streams": "numpy PCG64 seeds 1..100",
                "streams_where_w1_v1_holds": 61, "margin_min": -0.001, "margin_median": 0.0002,
                "margin_max": 0.001, "registered_stream_margin": 0.0003},
        }

    def test_deciding_numbers_name_the_narrowest_comparison(self):
        md = FA.render_md(self.ev)
        name, other, x = FA.deciding_comparison(self.ev["verdict"])
        margins = [c["margin"] for w in ("W1", "W2")
                   for c in self.ev["verdict"][w]["comparisons"].values() if c["margin"] is not None]
        self.assertEqual(x["margin"], min(margins))
        self.assertIn(f"Deciding numbers (MEASURED): the narrowest interval comparison is {name} "
                      f"against {other}: v2 lower bound {FA._f(x['v2_lower'], 6)} minus {other} "
                      f"upper bound {FA._f(x['other_upper'], 6)} = {FA._f(x['margin'], 6)} BA", md)
        self.assertLess(md.index("Deciding numbers"), md.index("## Provenance"))

    def test_robustness_section_only_when_given(self):
        self.assertNotIn("Robustness of the verdict", FA.render_md(self.ev))
        md = FA.render_md(self.ev, self._robustness())
        self.assertIn("## Robustness of the verdict (post-hoc, NOT registered)", md)
        self.assertIn("61 of 100 streams", md)
        self.assertLess(md.index("## Robustness"), md.index("## Provenance"))
        # the registered numbers and the verdict line are unchanged by the section
        self.assertIn(f"## Verdict: v2 **{self.ev['verdict']['result']}**", md)

    def test_robustness_reading_follows_the_sensitivity_count(self):
        name, other, _ = FA.deciding_comparison(self.ev["verdict"])
        if (name, other) != ("W1", "v1"):
            self.skipTest("the DEV deciding comparison is not W1 against v1")
        rob = self._robustness()
        self.assertIn("held in 61 of 100 other resample streams, so whether it holds depends on "
                      "bootstrap Monte Carlo noise", FA.render_md(self.ev, rob))
        rob["part_C_post_hoc_sensitivity"]["streams_where_w1_v1_holds"] = 100
        md = FA.render_md(self.ev, rob)
        self.assertIn("it held in all 100 other resample streams", md)
        self.assertNotIn("depends on bootstrap Monte Carlo noise", md)

    def test_not_distinguishable_lists_sets_whose_paired_interval_includes_zero(self):
        def s(lo, hi):
            return {"paired_differences": {"v2_minus_v1": {"ba": {"point": 0.0, "lower": lo,
                                                                  "upper": hi}}}}
        sets = {"a": s(0.01, 0.02), "b": s(-0.001, 0.01), "c": s(0.0, 0.1), "d": s(None, 0.1),
                "e": s(-0.2, -0.1)}
        self.assertEqual([n for n, _ in FA.not_distinguishable(sets)], ["b", "c", "d"])
        md = FA.render_md(self.ev)
        same = FA.not_distinguishable(self.ev["sets"])
        if same:
            para = next(line for line in md.splitlines() if line.startswith("Not distinguishable"))
            for name, _ in same:
                self.assertIn(f"{name} (paired v2 − v1 BA ", para)
        else:
            self.assertNotIn("Not distinguishable", md)

    def _recheck(self) -> dict:
        return {
            "inputs": {"score_manifest_sha256": self.ev["inputs"]["score_manifest_sha256"],
                       "committed_evaluation_sha256": "1" * 64,
                       "analysis_script_sha256": "2" * 64, "python": "3.12.10",
                       "script": "logs/v2/final_reanalyze_check.py",
                       "log": "logs/v2/final_reanalyze_check.txt"},
            "comparison": {"numeric_fields_compared": 1234, "numeric_fields_identical": 1234,
                           "max_abs_numeric_diff": 0.0, "timing_fields_excluded": 7,
                           "expected_metadata_differences": ["generated_utc"],
                           "unexpected_differences": [], "verdict_committed": "WIN",
                           "verdict_rechecked": "WIN"},
        }

    def test_corrections_section_lists_every_change_and_the_proof(self):
        self.assertNotIn("## Post-opening corrections", FA.render_md(self.ev))  # DEV, no recheck
        md = FA.render_md(self.ev, recheck=self._recheck())
        self.assertIn("## Post-opening corrections", md)
        for change in FA.POST_OPENING_CHANGES:
            self.assertIn(change["what"], md)
        self.assertIn("Numeric fields compared: 1234; identical: 1234", md)
        self.assertIn("Unexpected differences: 0.", md)
        self.assertLess(md.index("## Post-opening corrections"), md.index("## Provenance"))
        self.assertNotIn("not the version the re-analysis ran", md)
        renderer = {"evaluation_sha256": "3" * 64, "script_sha256": "4" * 64}
        self.assertIn("not the version the re-analysis ran",
                      FA.render_md(self.ev, renderer=renderer, recheck=self._recheck()))
        with S.TempDir() as tmp:
            bad = self._recheck()
            bad["inputs"]["score_manifest_sha256"] = "f" * 64
            path = tmp / "recheck.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(FA.AnalysisError):
                FA.load_recheck(self.ev, path)
            self.assertIsNone(FA.load_recheck(self.ev, tmp / "absent.json"))

    def test_load_robustness_refuses_another_manifest(self):
        with S.TempDir() as tmp:
            bad = self._robustness()
            bad["inputs"]["score_manifest_sha256"] = "f" * 64
            path = tmp / "robustness.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(FA.AnalysisError):
                FA.load_robustness(self.ev, path)
            self.assertIsNone(FA.load_robustness(self.ev, tmp / "absent.json"))

    def test_render_only_recomputes_nothing_and_checks_the_manifest(self):
        with S.TempDir() as tmp:
            _WATCH["active"] = True
            try:
                result = FA.render_only(eval_json=self.base / "out" / "e.json",
                                        score_dir=self.score_dir, out_md=tmp / "r2.md",
                                        out_svg=tmp / "r2.svg", robustness_json=None,
                                        recheck_json=None)
            finally:
                _WATCH["active"] = False
            self.assertEqual(_WATCH["touched"], [])
            self.assertEqual((tmp / "r2.svg").read_bytes(),
                             (self.base / "out" / "r.svg").read_bytes())
            md = (tmp / "r2.md").read_text(encoding="utf-8")
            self.assertIn("re-rendered from the saved `final_evaluation.json`", md)
            self.assertEqual(result["evaluation"]["verdict"], self.ev["verdict"])
            copy_dir = tmp / "final"
            shutil.copytree(self.score_dir, copy_dir)
            manifest = copy_dir / "MANIFEST.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaises(FA.AnalysisError):
                FA.render_only(eval_json=self.base / "out" / "e.json", score_dir=copy_dir,
                               out_md=tmp / "r3.md", out_svg=tmp / "r3.svg", robustness_json=None,
                               recheck_json=None)

    def test_render_only_keeps_the_analysis_time_reporting_order(self):
        # final_evaluation.json is written with sorted keys; the page rendered from it must list
        # sets, splits, contenders and features in the order the in-memory analysis used.
        saved = json.loads((self.base / "out" / "e.json").read_text(encoding="utf-8"))
        self.assertNotEqual(list(saved["sets"]), list(self.ev["sets"]))  # the keys really are sorted
        with S.TempDir() as tmp:
            FA.render_only(eval_json=self.base / "out" / "e.json", score_dir=self.score_dir,
                           out_md=tmp / "r4.md", out_svg=tmp / "r4.svg", robustness_json=None,
                           recheck_json=None)
            md = (tmp / "r4.md").read_text(encoding="utf-8")
        renderer = {"evaluation_sha256": FA.sha256_bytes((self.base / "out" / "e.json").read_bytes()),
                    "script_sha256": FA.sha256_bytes((registry.ROOT / FA.SCRIPT_REL).read_bytes())}
        self.assertEqual(md, FA.render_md(self.ev, renderer=renderer))
        manifest = json.loads((self.score_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        ordered = FA.reporting_order(saved, manifest)
        self.assertEqual(list(ordered["sets"]), list(self.ev["sets"]))
        self.assertEqual(list(ordered["validity"]), list(FA.CONTENDERS))
        self.assertEqual(list(ordered["inputs"]["score_files"]), list(manifest["splits"]))
        self.assertEqual(list(ordered["disaggregation"]["test"]), list(FA.DISAGG_FEATURES))
        self.assertEqual(ordered["sets"], saved["sets"])  # only key order differs
        self.assertEqual(ordered, saved)


if __name__ == "__main__":
    unittest.main()
