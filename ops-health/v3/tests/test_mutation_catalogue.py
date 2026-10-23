"""Mutation suite v3: the registered catalogue (prereg s.11.4) and the refusals that precede a
run (s.11.3). SYNTHETIC; DEV data only.

The mutants are built on the DEV fit and checked for form; no gate pass is run over them and no
catch rate is computed (that is protocol step 7, on the frozen v3). Receipt mutants that re-point
the source commit are built in a hermetic temporary git repository.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import unittest
from pathlib import Path

from v3.research import freeze, gates, mutation, registry, tolerances
from v3.tests import _support as S

K = S.kernel()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=oac-test", "-c", "user.email=oac-test@localhost",
         "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        check=True, capture_output=True, text=True).stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")


def _commit(repo: Path, files: dict, message: str) -> str:
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class ArtifactCatalogueTest(unittest.TestCase):
    def setUp(self):
        self.artifact = K.validate_artifact(S.candidate())

    def test_37_registered_artifact_mutants(self):
        mutants = mutation.artifact_mutants(self.artifact)
        ids = [m["id"] for m in mutants]
        self.assertEqual(len(ids), 37)
        self.assertEqual(len(set(ids)), 37)
        families = {}
        for m in mutants:
            families[m["family"]] = families.get(m["family"], 0) + 1
        self.assertEqual(families, {"weight x0.5": 8, "weight x1.5": 8, "sign flip": 8, "drop feature": 8,
                                    "intercept": 2, "threshold": 2, "transform table": 1})
        self.assertIn(mutation.TABLE_MUTANT_ID, ids)

    def test_each_mutant_changes_exactly_one_field(self):
        for m in mutation.artifact_mutants(self.artifact):
            with self.subTest(mutant=m["id"]):
                base = json.loads(json.dumps(self.artifact))
                mutated = m["artifact"]
                diffs = []
                for key in base:
                    if key in ("weights", "transform_table"):
                        for name in base[key]:
                            if base[key][name] != mutated[key][name]:
                                diffs.append(f"{key}.{name}")
                    elif base[key] != mutated[key]:
                        diffs.append(key)
                self.assertEqual(len(diffs), 1, diffs)
                self.assertEqual(set(mutated), set(base))
                for name, value in mutated["weights"].items():
                    self.assertEqual(value, round(value, 12))

    def test_kernel_accepts_every_mutant_except_the_table_mutant(self):
        for m in mutation.artifact_mutants(self.artifact):
            with self.subTest(mutant=m["id"]):
                if m["id"] == mutation.TABLE_MUTANT_ID:
                    with self.assertRaises(K.ModelArtifactError) as ctx:
                        K.validate_artifact(m["artifact"])
                    self.assertIn("differs from the registered representation R", str(ctx.exception))
                    self.assertEqual(m["after"], 86400.0)
                    self.assertEqual(m["artifact"]["weights"], self.artifact["weights"])
                else:
                    K.validate_artifact(m["artifact"])

    def test_threshold_mutants_move_the_grid_index(self):
        by_id = {m["id"]: m for m in mutation.artifact_mutants(self.artifact)}
        k = self.artifact["threshold_index"]
        self.assertEqual(by_id["A-threshold+0.1"]["artifact"]["threshold_index"], k + 10)
        self.assertEqual(by_id["A-threshold-0.1"]["artifact"]["threshold_index"], k - 10)
        edge = dict(copy.deepcopy(self.artifact), threshold_index=95)
        with self.assertRaises(mutation.MutationRefused):
            mutation.artifact_mutants(edge)

    def test_descriptive_catalogue(self):
        desc = mutation.descriptive_mutants(self.artifact)
        self.assertEqual(len(desc), 18)
        ids = {m["id"] for m in desc}
        for f in mutation.FEATURES:
            self.assertIn(f"D-scale0.75-{f}", ids)
            self.assertIn(f"D-scale1.25-{f}", ids)
        self.assertTrue({"D-threshold+0.05", "D-threshold-0.05"} <= ids)
        subs = mutation.recipe_substitution_mutants({n: Path(f"{n}.json") for n in mutation.RECIPE_SUBSTITUTIONS})
        self.assertEqual([m["id"] for m in subs], ["D-recipe-ABL-D", "D-recipe-ABL-R", "D-recipe-ABL-DR"])
        with self.assertRaises(mutation.MutationRefused):
            mutation.recipe_substitution_mutants({"ABL-D": Path("x")})

    def test_registered_fit_mutants_need_prepare_run(self):
        """s.11.3 order rule: a non-DEV artifact or receipt gets no mutant before prepare_run."""
        art = copy.deepcopy(self.artifact)
        for role, entry in art["training"]["splits"].items():
            entry["name"] = {"train": "train", "threshold": "validation", "conformal": "conformal"}[role]
        for builder in (mutation.artifact_mutants, mutation.descriptive_mutants):
            with self.subTest(builder=builder.__name__), self.assertRaises(mutation.MutationRefused):
                builder(art)
            with self.subTest(builder=builder.__name__, prepared="unpinned"), \
                    self.assertRaises(mutation.MutationRefused):
                builder(art, prepared={"tolerances": {"tolerances_sha256": "0" * 64},
                                       "runtime_pins": {"pinned": False}})
        prepared = {"tolerances": {"tolerances_sha256": "0" * 64}, "runtime_pins": {"pinned": True}}
        self.assertEqual(len(mutation.artifact_mutants(art, prepared=prepared)), 37)
        receipt = {"fit": {"declared_fit": "main"}, "model_sha256": "0" * 64,
                   "source": {"commit": "1" * 40}}
        with self.assertRaises(mutation.MutationRefused):
            mutation.receipt_mutants(receipt, first_parent_commit="2" * 40, descendant_commit="3" * 40,
                                     squash_duplicate_commit="4" * 40)

    def test_pass_two_receipt_is_consistent(self):
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "ref")
            receipt = json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))
            mutant = mutation.artifact_mutants(self.artifact)[0]["artifact"]
            data = freeze.canonical_model_bytes(mutant)
            model, rec = freeze.write_pair(tmp / "pass2", data, freeze.consistent_receipt(receipt, data))
            K.OpsHealthKernel(model, rec)  # pass 2 bypasses integrity
            model1, rec1 = freeze.write_pair(tmp / "pass1", data, receipt)
            with self.assertRaises(K.ModelArtifactError):
                K.OpsHealthKernel(model1, rec1)  # pass 1: the frozen receipt refuses it


class KernelCatalogueTest(unittest.TestCase):
    def setUp(self):
        self.text = registry.KERNEL_PATH.read_text(encoding="utf-8")

    def test_eight_kernel_mutants_each_edit_occurs_once(self):
        self.assertEqual([m[0] for m in mutation.KERNEL_MUTANTS], [
            "K-no-finiteness", "K-no-range", "K-no-prohibited-screen", "K-inverted-model-hash",
            "K-inverted-self-hash", "K-no-transform-table-check", "K-no-duplicate-key-refusal",
            "K-authority-true"])
        for mutant_id, _desc, edits in mutation.KERNEL_MUTANTS:
            for old, _new in edits:
                self.assertEqual(self.text.count(old), 1, mutant_id)
        sources = mutation.kernel_mutant_sources(self.text)
        self.assertEqual(len(sources), 8)
        for m in sources:
            self.assertNotEqual(m["source"], self.text)
            compile(m["source"], m["id"], "exec")

    def test_each_kernel_mutant_breaks_its_property(self):
        """Semantic check of each copy (the registered catch rule is evaluated at step 7)."""
        art = K.validate_artifact(S.candidate())
        by_id = {m["id"]: m["source"] for m in mutation.kernel_mutant_sources(self.text)}
        valid = dict(gates.VALID_FEATURES)
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "ref")
            model, receipt_path = Path(info["model_path"]), Path(info["receipt_path"])

            def load(mutant_id):
                return S.load_kernel_copy(tmp, by_id[mutant_id])

            m, _ = load("K-no-finiteness")
            m.ArtifactScorer(art).advise({"features": dict(valid, queue_utilization=math.nan)})
            m, _ = load("K-no-range")
            m.ArtifactScorer(art).advise({"features": dict(valid, queue_utilization=1.5)})
            m, _ = load("K-no-prohibited-screen")
            with self.assertRaises(m.ModelInputError) as ctx:
                m.ArtifactScorer(art).advise({"features": dict(valid, patient_id=0)})
            self.assertNotIn("prohibited", str(ctx.exception))
            m, path = load("K-inverted-model-hash")
            rebound = S.rebind_receipt(receipt_path, path, tmp / "inv_model.json")
            with self.assertRaises(m.ModelArtifactError):
                m.OpsHealthKernel(model, rebound)
            m, _ = load("K-inverted-self-hash")
            m.OpsHealthKernel(model, receipt_path)  # receipt binds the real kernel, not the copy
            m, _ = load("K-no-transform-table-check")
            table_mutant = [x for x in mutation.artifact_mutants(art) if x["id"] == mutation.TABLE_MUTANT_ID][0]
            m.ArtifactScorer(table_mutant["artifact"])
            m, _ = load("K-no-duplicate-key-refusal")
            self.assertEqual(m.loads_strict('{"a": 0, "a": 1}'), {"a": 1})
            m, _ = load("K-authority-true")
            out = m.ArtifactScorer(art).advise({"features": valid})
            self.assertTrue(any(out["authority"].values()))

    def test_kernel_copies_never_replace_the_tracked_kernel(self):
        with self.assertRaises(Exception):
            mutation.write_kernel_copy(registry.V3, "ops-health", "x = 1\n")
        self.assertEqual(registry.KERNEL_PATH.read_text(encoding="utf-8"), self.text)


class ReceiptCatalogueTest(unittest.TestCase):
    def test_seven_receipt_mutants_on_a_hermetic_repo(self):
        with S.TempDir() as tmp:
            repo = tmp / "repo"
            _init_repo(repo)
            bound = {"v3/ops-health/ops_health.py": "kernel\n", "v3/research/generator.py": "gen\n"}
            parent = _commit(repo, dict(bound, **{"code.txt": "P\n"}), "P: the code that ran")
            claimed = _commit(repo, {"v3/fits/trace.json": "{}\n"}, "C: the trace on top of P")
            descendant = _commit(repo, {"v3/fits/receipts.txt": "r\n"}, "D: receipts after C")
            self.assertEqual(mutation.first_parent(repo, claimed), parent)
            paths = list(bound)
            self.assertTrue(mutation.bound_blobs_identical(repo, claimed, parent, paths))
            self.assertTrue(mutation.bound_blobs_identical(repo, claimed, descendant, paths))
            duplicate = mutation.build_squash_duplicate(repo, claimed, tmp / "scratch")
            self.assertNotEqual(duplicate["commit"], claimed)
            self.assertEqual(duplicate["tree"], _git(repo, "rev-parse", f"{claimed}^{{tree}}"))
            self.assertEqual(duplicate["first_parent"], parent)
            with self.assertRaises(subprocess.CalledProcessError):
                _git(repo, "cat-file", "-e", duplicate["commit"])  # it exists in the scratch clone only
            with self.assertRaises(mutation.MutationRefused):
                mutation.build_squash_duplicate(repo, claimed, tmp / "scratch")

            info = S.freeze_dev(tmp / "art", commit=claimed)
            receipt = json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))
            mutants = mutation.receipt_mutants(receipt, first_parent_commit=parent,
                                               descendant_commit=descendant,
                                               squash_duplicate_commit=duplicate["commit"])
            self.assertEqual([m["id"] for m in mutants], [
                "R-model-hash-flip", "R-wrong-schema", "R-source-commit-altered", "R-metrics-added",
                "R-first-parent", "R-descendant", "R-squash-duplicate"])
            commits = {m["id"]: m["receipt"]["source"]["commit"] for m in mutants}
            self.assertEqual(commits["R-first-parent"], parent)
            self.assertEqual(commits["R-descendant"], descendant)
            self.assertEqual(commits["R-squash-duplicate"], duplicate["commit"])
            self.assertNotEqual(commits["R-source-commit-altered"], claimed)
            # The kernel alone refuses the first two and the metrics block; the four re-pointing
            # mutants are well-formed, so only the research origin check (s.14.4) can catch them.
            outcome = {}
            for m in mutants:
                model, rec = freeze.write_pair(tmp / m["id"], Path(info["model_path"]).read_bytes(), m["receipt"])
                try:
                    K.OpsHealthKernel(model, rec)
                    outcome[m["id"]] = "kernel accepts"
                except K.ModelArtifactError:
                    outcome[m["id"]] = "kernel refuses"
            self.assertEqual(outcome, {
                "R-model-hash-flip": "kernel refuses", "R-wrong-schema": "kernel refuses",
                "R-metrics-added": "kernel refuses", "R-source-commit-altered": "kernel accepts",
                "R-first-parent": "kernel accepts", "R-descendant": "kernel accepts",
                "R-squash-duplicate": "kernel accepts"})
            with self.assertRaises(mutation.MutationRefused):
                mutation.receipt_mutants(receipt, first_parent_commit=claimed, descendant_commit=descendant,
                                         squash_duplicate_commit=duplicate["commit"])


class ToleranceDigestRefusalTest(unittest.TestCase):
    """s.11.3: the mutation code refuses to run unless the tolerance digest is in a commit that
    precedes it. Hand-made tolerance numbers only."""

    @staticmethod
    def record() -> dict:
        band = [{"train_split": name, "theta_r": [0.5] * 9, "threshold_index": 20, "ba_gates": 0.8,
                 "converged": True} for name in tolerances.BAND_NAMES]
        vector = tolerances.compute(band)
        return {"vector": vector, "sha256": tolerances.digest(vector)}

    def test_refusal_fires_until_the_digest_is_committed(self):
        with S.TempDir() as tmp:
            repo = tmp / "repo"
            _init_repo(repo)
            _commit(repo, {"v3/README.txt": "x\n"}, "base")
            with self.assertRaises(mutation.MutationRefused):
                mutation.require_tolerances_committed(repo)
            with self.assertRaises(mutation.MutationRefused):
                mutation.prepare_run(repo)
            record = self.record()
            target = repo / mutation.TOLERANCES_REL
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(mutation.MutationRefused):  # present but not committed
                mutation.require_tolerances_committed(repo)
            _commit(repo, {}, "tolerances")
            checked = mutation.require_tolerances_committed(repo)
            self.assertEqual(checked["tolerances_sha256"], record["sha256"])
            target.write_text(json.dumps(dict(record, sha256="0" * 64)), encoding="utf-8")
            with self.assertRaises(mutation.MutationRefused):  # changed after its commit
                mutation.require_tolerances_committed(repo)
            _commit(repo, {}, "wrong digest")
            with self.assertRaises(mutation.MutationRefused):  # committed, but digest wrong
                mutation.require_tolerances_committed(repo)

    def test_prepare_run_also_needs_the_runtime_pins(self):
        with S.TempDir() as tmp:
            repo = tmp / "repo"
            _init_repo(repo)
            record = self.record()
            _commit(repo, {mutation.TOLERANCES_REL: json.dumps(record)}, "tolerances")
            from unittest import mock

            from v3.research import guards

            with mock.patch.object(guards, "runtime_pin_status", return_value={"pinned": False}):
                with self.assertRaises(guards.RuntimePinError):
                    mutation.prepare_run(repo)


if __name__ == "__main__":
    unittest.main()
