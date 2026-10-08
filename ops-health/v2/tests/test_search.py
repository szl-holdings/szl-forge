"""Stage 2: the registered search loop, selection rules, ABL-conformal rule, stage-2 helpers and
the receipt's selection provenance (DEV data only; SYNTHETIC).

Nothing here opens a registered split: the registered loaders are exercised with a recording
fake guard that serves DEV rows, and no file under v2/data/sealed/ is touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from fractions import Fraction
from pathlib import Path

from v2.research import freeze, generator, pipeline, registry, search, stage2, v1_gd
from v2.research.dataio import labels_of, load_rows
from v2.tests import _support as S

K = S.kernel()
FIXED_TS = "2026-09-29T00:00:00Z"
DEV_SIZES = {"train": 2048, "calibration": 1024, "conformal": 1024, "threshold": 1024}


def dev_split_rows() -> tuple[dict, dict]:
    rows = list(S.dev_rows())
    out = {}
    start = 0
    for role in pipeline.SPLIT_ROLES:
        out[role] = rows[start:start + DEV_SIZES[role]]
        start += DEV_SIZES[role]
    return out, {role: f"dev-{role}" for role in pipeline.SPLIT_ROLES}


def fake_record(trial_id, l2, cal, ba, brier, ece, eligible=True):
    return {
        "trial_id": trial_id, "l2": l2, "calibrator": cal, "eligible": eligible,
        "ineligible_reasons": [] if eligible else ["IRLS not converged: CAP_REACHED"],
        "validation": {"balanced_accuracy_exact": [ba.numerator, ba.denominator],
                       "brier": brier, "ece": ece},
    }


class SearchSpaceTest(unittest.TestCase):
    def test_space_is_the_registered_grid_in_order(self):
        space = search.search_space()
        self.assertEqual(len(space), 12)
        self.assertEqual(search.HARD_CAP, 12)
        self.assertEqual([(c["l2"], c["calibrator"]) for c in space],
                         [(l2, cal) for l2 in (0.0, 1e-4, 1e-3, 1e-2)
                          for cal in ("none", "platt", "isotonic")])
        ids = [c["trial_id"] for c in space]
        self.assertEqual(len(set(ids)), 12)
        self.assertTrue(all(len(i) <= 128 for i in ids))
        self.assertEqual(ids[0], "T01_l2_0_none")
        self.assertEqual(ids[-1], "T12_l2_1e-2_isotonic")
        self.assertEqual(search.trial_id_for(1e-3, "platt"), "T08_l2_1e-3_platt")
        self.assertEqual(set(K.REGISTERED_L2_GRID), {c["l2"] for c in space})


class SelectionRuleTest(unittest.TestCase):
    def test_ba_decides_first(self):
        recs = [fake_record("a", 0.0, "none", Fraction(3, 4), 0.1, 0.1),
                fake_record("b", 0.0, "platt", Fraction(3, 4) + Fraction(1, 10**9), 0.9, 0.9)]
        sel = search.select(recs)
        self.assertEqual(sel["selected_trial"], "b")
        self.assertEqual(sel["decided_by"], "validation_balanced_accuracy")

    def test_tie_breaks_in_registered_order(self):
        ba = Fraction(7, 10)
        cases = [
            ([("a", 0.0, "none", 0.20, 0.01), ("b", 0.0, "platt", 0.10, 0.09)], "b",
             "validation_brier"),
            ([("a", 0.0, "none", 0.10, 0.02), ("b", 0.0, "platt", 0.10, 0.01)], "b",
             "validation_ece"),
            ([("a", 1e-4, "none", 0.10, 0.01), ("b", 1e-2, "isotonic", 0.10, 0.01)], "b", "l2"),
            ([("a", 1e-3, "isotonic", 0.10, 0.01), ("b", 1e-3, "platt", 0.10, 0.01),
              ("c", 1e-3, "none", 0.10, 0.01)], "c", "calibrator_order"),
        ]
        for spec, winner, decided in cases:
            with self.subTest(winner=winner, decided=decided):
                sel = search.select([fake_record(t, l2, c, ba, b, e) for t, l2, c, b, e in spec])
                self.assertEqual(sel["selected_trial"], winner)
                self.assertEqual(sel["decided_by"], decided)

    def test_ineligible_trials_are_never_selected(self):
        recs = [fake_record("best_but_capped", 0.0, "none", Fraction(9, 10), 0.0, 0.0,
                            eligible=False),
                fake_record("ok", 0.0, "platt", Fraction(6, 10), 0.2, 0.2)]
        sel = search.select(recs)
        self.assertEqual(sel["selected_trial"], "ok")
        self.assertEqual(sel["ineligible_trials"][0]["trial_id"], "best_but_capped")
        none = search.select([recs[0]])
        self.assertIsNone(none["selected_trial"])
        self.assertEqual(none["status"], "NO_ELIGIBLE_TRIAL")

    def test_am2_eligibility_reasons(self):
        ok = {"converged": True, "stop_reason": "CONVERGED"}
        self.assertEqual(search.eligibility("platt", ok, ok), [])
        self.assertEqual(search.eligibility("none", ok, {"rows": 3}), [])
        for bad in ("CAP_REACHED", "LINE_SEARCH_FAILED"):
            fail = {"converged": False, "stop_reason": bad}
            self.assertEqual(search.eligibility("none", fail, {}), [f"IRLS not converged: {bad}"])
            self.assertEqual(search.eligibility("platt", ok, fail), [f"Platt not converged: {bad}"])
            self.assertEqual(len(search.eligibility("platt", fail, fail)), 2)
        # isotonic has no convergence notion; its diagnostics never make a trial ineligible
        self.assertEqual(search.eligibility("isotonic", ok, {"blocks": 5}), [])

    def test_conformal_keep_rule_boundaries(self):
        def cov(m, c0, c1):
            return {"marginal": m, "per_class": {"0": c0, "1": c1}}
        self.assertTrue(search.conformal_keep(cov(0.88, 0.85, 0.85)))
        self.assertFalse(search.conformal_keep(cov(0.8799999, 0.95, 0.95)))
        self.assertFalse(search.conformal_keep(cov(0.95, 0.8499999, 0.95)))
        self.assertFalse(search.conformal_keep(cov(0.95, 0.95, 0.8499999)))
        self.assertFalse(search.conformal_keep(cov(None, 0.95, 0.95)))
        self.assertFalse(search.conformal_keep(cov(0.95, None, 0.95)))

    def test_exact_balanced_accuracy(self):
        ba = search.exact_balanced_accuracy({"tp": 1, "fn": 2, "tn": 3, "fp": 1})
        self.assertEqual(ba, (Fraction(1, 3) + Fraction(3, 4)) / 2)


class DevSearchTest(unittest.TestCase):
    """The full bounded loop on DEV rows, twice, into temporary directories."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = S.TempDir()
        cls.tmp = cls._tmp.__enter__()
        cls.rows, cls.names = dev_split_rows()
        cls.record, cls.models = search.run_search(
            cls.rows, cls.names, receipts_dir=cls.tmp / "receipts", run_id="dev-search",
            candidates_dir=cls.tmp / "candidates", ts_fn=lambda: FIXED_TS)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def test_loop_exhausts_the_space_under_the_cap(self):
        loop = self.record["loop"]
        self.assertEqual(loop["stop_reason"], "SEARCH_SPACE_EXHAUSTED")
        self.assertEqual(loop["iterations_run"], 12)
        self.assertEqual(loop["max_iterations"], 12)
        self.assertEqual(loop["order"], list(range(12)))
        files = sorted(p.name for p in (self.tmp / "receipts").iterdir())
        self.assertEqual(len(files), 12 + 1 + 1)  # 12 trial receipts, 1 summary, trace.json
        self.assertIn("trace.json", files)
        self.assertEqual(self.record["receipts"]["verify_run_dir_errors"], [])
        self.assertTrue(self.record["receipts"]["verified"])

    def test_receipts_bind_each_trial_record(self):
        ob = search.ouroboros()
        receipts = self.tmp / "receipts"
        for trial in self.record["trials"]:
            rec = json.loads((receipts / trial["receipt_file"]).read_text(encoding="utf-8"))
            payload = {"kind": "trial", "run_id": "dev-search", "iteration": trial["iteration"],
                       "candidate_index": trial["iteration"],
                       "candidate": search.search_space()[trial["iteration"]],
                       "result": trial["record"],
                       "search_space_digest": self.record["loop"]["search_space_digest"]}
            import receipt_adapter as ra  # loaded by the ouroboros adapter  # noqa: PLC0415
            self.assertEqual(ra.verify_receipt(rec, payload, check_payload=True), [])
            self.assertEqual(rec["digest"], trial["receipt_digest"])
        self.assertIsNotNone(ob)

    def test_candidates_match_receipted_hashes_and_kernel(self):
        for trial in self.record["trials"]:
            rec = trial["record"]
            data = (self.tmp / "candidates" / f"{rec['trial_id']}.model.json").read_bytes()
            self.assertEqual(search.sha256_bytes(data), rec["model_sha256"])
            self.assertEqual(data, self.models[rec["trial_id"]])
            artifact = K.validate_artifact(json.loads(data))
            self.assertEqual(artifact["training"]["l2"], rec["l2"])
            self.assertEqual(artifact["calibrator"]["kind"], rec["calibrator"])
            self.assertEqual(artifact["decision_threshold"], rec["validation"]["threshold"])

    def test_selection_is_the_registered_argmin(self):
        records = [t["record"] for t in self.record["trials"]]
        eligible = [r for r in records if r["eligible"]]
        best = min(eligible, key=search.selection_rank)
        self.assertEqual(self.record["selection"]["selected_trial"], best["trial_id"])
        for r in records:
            num, den = r["validation"]["balanced_accuracy_exact"]
            self.assertAlmostEqual(num / den, r["validation"]["balanced_accuracy"], places=15)
            self.assertEqual(r["eligible"], not r["ineligible_reasons"])

    def test_validation_metrics_are_recomputable(self):
        rec = self.record["trials"][4]["record"]
        artifact = json.loads(self.models[rec["trial_id"]])
        again = search.validation_report(artifact, self.rows["threshold"])
        self.assertEqual(again, rec["validation"])

    def test_conformal_decision_on_threshold_rows(self):
        conf = self.record["conformal"]
        self.assertEqual(conf["trial_id"], self.record["selection"]["selected_trial"])
        self.assertEqual(conf["alpha"], 0.10)
        self.assertEqual(conf["keep"], search.conformal_keep(conf["coverage"]))
        self.assertEqual(conf["coverage"]["rows"], DEV_SIZES["threshold"])
        self.assertEqual(conf["without_abstention"]["abstention_rate"], 0.0)

    def test_rerun_is_deterministic_and_refuses_a_used_dir(self):
        again, _models = search.run_search(
            self.rows, self.names, receipts_dir=self.tmp / "receipts2", run_id="dev-search",
            ts_fn=lambda: FIXED_TS)
        self.assertEqual([t["record"] for t in again["trials"]],
                         [t["record"] for t in self.record["trials"]])
        self.assertEqual(again["selection"], self.record["selection"])
        first = self.record["receipts"]["dir_digest"]["files"]
        second = again["receipts"]["dir_digest"]["files"]
        receipts = {n: d for n, d in first.items() if n != "trace.json"}
        self.assertEqual(receipts, {n: d for n, d in second.items() if n != "trace.json"})
        with self.assertRaises(FileExistsError):
            search.run_search(self.rows, self.names, receipts_dir=self.tmp / "receipts",
                              run_id="dev-search", ts_fn=lambda: FIXED_TS)

    def test_directory_digest_detects_tampering(self):
        copy = self.tmp / "receipts_copy"
        shutil.copytree(self.tmp / "receipts", copy)
        before = search.directory_digest(copy)
        self.assertEqual(before["sha256"], search.directory_digest(self.tmp / "receipts")["sha256"])
        victim = sorted(p for p in copy.iterdir() if p.name != "trace.json")[3]
        victim.write_bytes(victim.read_bytes().replace(b'"ns": "oac-frontier"', b'"ns": "x"'))
        self.assertNotEqual(search.directory_digest(copy)["sha256"], before["sha256"])
        self.assertTrue(search.ouroboros().verify_run_dir(str(copy)))

    def test_stage2_trace_checks(self):
        root = self.tmp / "root"
        (root / "runs").mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.tmp / "receipts", root / "runs" / "r")
        shutil.copytree(self.tmp / "candidates", root / "cands")
        trace = dict(self.record, receipts_dir_relpath="runs/r", candidates_dir_relpath="cands",
                     protocol={"amendments_sha256": search.AMENDMENTS_SHA256})
        checked = stage2.check_trace(trace, root=root)
        self.assertEqual(checked["dir_sha256"], self.record["receipts"]["dir_digest"]["sha256"])
        selected = trace["selection"]["selected_trial"]
        data = stage2.candidate_bytes(trace, selected, root=root)
        self.assertEqual(data, self.models[selected])
        block = stage2.selection_block(trace, "a" * 64, procedure="dev", trials=12,
                                       stop_reason="SEARCH_SPACE_EXHAUSTED")
        self.assertEqual(set(block), set(K.SELECTION_KEYS))
        self.assertEqual(block["conformal_keep"], self.record["conformal"]["keep"])
        receipt = freeze.build_receipt(freeze.canonical_model_bytes(json.loads(data)),
                                       commit=S.FAKE_COMMIT, tree_clean=True, selection=block)
        self.assertEqual(K.validate_receipt(receipt)["selection"], block)
        # a tampered candidate is refused
        cand = root / "cands" / f"{selected}.model.json"
        cand.write_bytes(cand.read_bytes() + b" ")
        with self.assertRaises(stage2.Stage2Refused):
            stage2.candidate_bytes(trace, selected, root=root)
        # a tampered receipt is refused
        victim = sorted(p for p in (root / "runs" / "r").iterdir() if p.name != "trace.json")[0]
        victim.write_bytes(victim.read_bytes().replace(b'"seq": 0', b'"seq": 7'))
        with self.assertRaises(stage2.Stage2Refused):
            stage2.check_trace(trace, root=root)
        # no eligible trial: v2 does not exist
        shutil.rmtree(root / "runs" / "r")
        shutil.copytree(self.tmp / "receipts", root / "runs" / "r")
        empty = dict(trace, selection=dict(trace["selection"], selected_trial=None))
        with self.assertRaises(stage2.Stage2Refused):
            stage2.check_trace(empty, root=root)


class RecordingGuard:
    def __init__(self):
        self.calls = []
        self.rows, _names = dev_split_rows()
        self.by_name = {"train": "train", "calibration": "calibration",
                        "conformal": "conformal", "validation": "threshold"}

    def open_split(self, name, purpose, **kwargs):
        self.calls.append((name, purpose))
        if name in registry.SEALED_SPLITS or name == "sealed":
            raise AssertionError("a sealed split was requested")
        return self.rows[self.by_name[name]]


class RegisteredLoaderTest(unittest.TestCase):
    def test_search_opens_only_registered_open_splits_with_registered_purposes(self):
        guard = RecordingGuard()
        rows, names = search.load_registered_rows(guard)
        self.assertEqual(sorted(guard.calls), sorted([
            ("train", "TRAIN"), ("calibration", "CALIBRATION"), ("conformal", "CONFORMAL"),
            ("validation", "REGISTERED_SEARCH")]))
        self.assertEqual(names["threshold"], "validation")
        from v2.research import sealed_guard  # noqa: PLC0415
        for name, purpose in guard.calls:
            self.assertIn(purpose, sealed_guard.OPEN_PURPOSES[name])

    def test_stage2_loaders_use_registered_purposes(self):
        from v2.research import sealed_guard  # noqa: PLC0415
        guard = RecordingGuard()
        stage2._registered_rows({r: "ABLATION" for r in pipeline.SPLIT_ROLES}, guard)
        stage2._registered_rows({"calibration": "CALIBRATION", "conformal": "CONFORMAL",
                                 "threshold": "GATES"}, guard,
                                roles=stage2.RETRAIN_SHARED_ROLES)
        self.assertEqual(len(guard.calls), 7)
        for name, purpose in guard.calls:
            self.assertIn(purpose, sealed_guard.OPEN_PURPOSES[name])

    def test_registered_preconditions_refuse_changed_amendments(self):
        state = {"preregistration_sha256": registry.PREREG_SHA256, "amendments_sha256": "0" * 64,
                 "tree_clean": True, "dirty": [], "test_opened_marker_present": False}
        with self.assertRaises(search.SearchRefused) as ctx:
            search.registered_preconditions(state, "no-such-run-id")
        self.assertIn("AMENDMENTS.md", str(ctx.exception))


class Stage2HelperTest(unittest.TestCase):
    def test_v1_threshold_rule_reproduces_the_published_v1_threshold(self):
        rows = load_rows(registry.LEGACY_DIR / "validation.jsonl")
        scores = [v1_gd.v1_committed_score(generator.normalize(r["features"])) for r in rows]
        chosen, rank = stage2.v1_choose_threshold(labels_of(rows), scores)
        self.assertEqual(chosen, v1_gd.V1_DECISION_THRESHOLD)
        self.assertGreater(rank["balanced_accuracy"], 0.5)

    def test_v1_threshold_rule_range_and_ties(self):
        # every threshold separates nothing: all ranks tie on BA and F1 -> closest to 0.5
        labels = [0, 1, 0, 1]
        chosen, _ = stage2.v1_choose_threshold(labels, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(chosen, 0.5)
        # perfect separation at any t in (0.2, 0.8]: ties on BA/F1, closest to 0.5 wins
        chosen, rank = stage2.v1_choose_threshold([0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8])
        self.assertEqual((chosen, rank["balanced_accuracy"]), (0.5, 1.0))
        # extremes outside 0.05..0.95 are never chosen
        chosen, _ = stage2.v1_choose_threshold([0, 1], [0.01, 0.02])
        self.assertTrue(0.05 <= chosen <= 0.95)

    def test_score_orders_agree(self):
        vectors = [generator.normalize(r["features"]) for r in S.dev_rows()[:500]]
        weights = list(v1_gd.V1_WEIGHTS.values())
        a = stage2.v1_trainer_scores(v1_gd.V1_INTERCEPT, weights, vectors)
        b = stage2.v1_kernel_order_scores(v1_gd.V1_INTERCEPT, weights, vectors)
        c = [v1_gd.v1_committed_score(v) for v in vectors]
        self.assertEqual(b, c)
        self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-14)

    def test_known_good_seed_material_and_rows(self):
        self.assertEqual(stage2.known_good_seed_material(1), "oac-ops-health-v2:20260927:train")
        self.assertEqual(stage2.known_good_seed_material(5), "oac-ops-health-v2:20260931:train")
        seeds = {generator.derive_seed("train", generator.MASTER_SEED + k) for k in range(0, 6)}
        self.assertEqual(len(seeds), 6)
        a = stage2.known_good_rows(2, 5)
        self.assertEqual(a, stage2.known_good_rows(2, 5))
        self.assertNotEqual(a, generator.generate_rows("train", 5, "nominal"))
        self.assertEqual(a[0]["sample_id"], "train-000000")
        self.assertEqual(stage2.KNOWN_GOOD_TRAIN_ROWS, 16384)
        self.assertEqual(stage2.ABL_DATA_ROWS, 768)

    def test_example_input_is_accepted_by_the_kernel_schema(self):
        self.assertEqual(len(K.normalize_features(K.unwrap_payload(stage2.EXAMPLE_INPUT))), 8)


class SelectionProvenanceReceiptTest(unittest.TestCase):
    """The kernel's receipt.selection block carries the stage-2 search provenance."""

    def setUp(self):
        self._tmp = S.TempDir()
        self.tmp = self._tmp.__enter__()
        self.selection = dict(S.SELECTION, search_trace_sha256="a" * 64,
                              receipts_dir_sha256="b" * 64,
                              amendments_sha256=search.AMENDMENTS_SHA256, conformal_keep=True,
                              selected_trial="T08_l2_1e-3_platt", trials=12,
                              stop_reason="SEARCH_SPACE_EXHAUSTED")
        self.model_bytes = freeze.canonical_model_bytes(S.candidate())

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def receipt(self, **changes):
        selection = dict(self.selection, **changes)
        return freeze.build_receipt(self.model_bytes, commit=S.FAKE_COMMIT, tree_clean=True,
                                    selection=selection)

    def test_full_provenance_validates_and_loads(self):
        info = freeze.freeze(S.candidate(), self.tmp / "art", selection=self.selection,
                             source_state_override={"commit": S.FAKE_COMMIT, "tree_clean": True})
        kernel = K.OpsHealthKernel(info["model_path"], info["receipt_path"])
        self.assertEqual(kernel.receipt["selection"], self.selection)
        for keep in (False, None):
            self.assertEqual(K.validate_receipt(self.receipt(conformal_keep=keep))
                             ["selection"]["conformal_keep"], keep)

    def test_malformed_provenance_is_refused(self):
        for key in K.SELECTION_DIGEST_KEYS:
            for bad in ("x", "A" * 64, "a" * 63, 7, True, ["a" * 64]):
                with self.subTest(key=key, bad=bad), self.assertRaises(K.ModelArtifactError):
                    K.validate_receipt(self.receipt(**{key: bad}))
        for bad in ("yes", 1, 0, "true", {}):
            with self.subTest(conformal_keep=bad), self.assertRaises(K.ModelArtifactError):
                K.validate_receipt(self.receipt(conformal_keep=bad))

    def test_missing_or_extra_selection_key_is_refused(self):
        receipt = self.receipt()
        del receipt["selection"]["amendments_sha256"]
        with self.assertRaises(K.ModelArtifactError):
            K.validate_receipt(receipt)
        receipt = self.receipt()
        receipt["selection"]["metrics"] = {"balanced_accuracy": 1.0}
        with self.assertRaises(K.ModelArtifactError):
            K.validate_receipt(receipt)


class AmendmentsOriginTest(unittest.TestCase):
    """freeze.verify_origin checks selection.amendments_sha256 against the committed blob."""

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=oac-test", "-c",
             "user.email=oac-test@localhost", "-c", "commit.gpgsign=false", "-c",
             "core.autocrlf=false", *args],
            check=True, capture_output=True, text=True,
        ).stdout

    def test_amendments_digest_is_verified(self):
        with S.TempDir() as tmp:
            repo = tmp / "repo"
            rels = [rel for _k, rel in freeze.ORIGIN_BLOBS] + [freeze.AMENDMENTS_REL]
            for rel in rels:
                target = repo / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(registry.ROOT / rel, target)
            self._git(repo, "init", "-q")
            self._git(repo, "add", *rels)
            self._git(repo, "commit", "-q", "-m", "fixture")
            head = self._git(repo, "rev-parse", "HEAD").strip()
            actual = S.sha256_file(registry.ROOT / freeze.AMENDMENTS_REL)
            model_bytes = freeze.canonical_model_bytes(S.candidate())
            selection = dict(S.SELECTION, amendments_sha256=actual)
            receipt = freeze.build_receipt(model_bytes, commit=head, tree_clean=True,
                                           selection=selection)
            result = freeze.verify_origin(receipt, root=repo)
            self.assertEqual(result["verified"]["selection.amendments_sha256"], actual)
            wrong = json.loads(json.dumps(receipt))
            wrong["selection"]["amendments_sha256"] = "0" * 64
            with self.assertRaises(freeze.OriginMismatch) as ctx:
                freeze.verify_origin(wrong, root=repo)
            self.assertIn("amendments_sha256", str(ctx.exception))
            # a null digest (DEV receipts) is not checked
            dev = freeze.build_receipt(model_bytes, commit=head, tree_clean=True,
                                       selection=S.SELECTION)
            self.assertNotIn("selection.amendments_sha256",
                             freeze.verify_origin(dev, root=repo)["verified"])


if __name__ == "__main__":
    unittest.main()
