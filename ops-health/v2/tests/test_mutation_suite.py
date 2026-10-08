"""Section 10 mutation suite and broken baselines (v2/research/mutation.py): fast subsets.

SYNTHETIC.  DEV rows only for every gate computed here; the frozen v2 model.json and receipt are
read (never written) as the mutation source.  No registered split is opened and nothing under
v2/data/sealed/ is read.  The full run (validation split, all mutants, the kernel-relevant tests
against every kernel mutant) is ``python -m v2.research.mutation run``; its output is checked by
test_readiness.py.
"""

from __future__ import annotations

import copy
import json
import unittest

from v2.research import freeze, gates, kernel_bridge, metrics, mutation as M, pipeline, registry
from v2.tests import _support as S

K = S.kernel()
V2_MODEL = registry.V2 / "ops-health" / "model.json"
V2_RECEIPT = registry.V2 / "ops-health" / "artifact_receipt.json"
PROBE_ID = "v2.tests.test_failclosed_probes.FailClosedProbeTest.test_every_probe_is_refused_with_expected_message"


def frozen_v2() -> tuple[dict, dict, bytes]:
    data = V2_MODEL.read_bytes()
    return json.loads(data.decode("utf-8")), json.loads(V2_RECEIPT.read_text(encoding="utf-8")), data


def dev_context() -> M.GateContext:
    rows = list(S.dev_rows())[S.EVAL][:4096]
    return M.GateContext(rows, pipeline.rows_sha256(rows))


def _diff_fields(a: dict, b: dict) -> list[str]:
    out = [key for key in a if key not in ("weights",) and a[key] != b[key]]
    out += [f"weights.{f}" for f in a["weights"] if a["weights"][f] != b["weights"][f]]
    return out


class CatalogueTest(unittest.TestCase):
    def test_artifact_catalogue_on_frozen_v2(self):
        art, _receipt, data = frozen_v2()
        self.assertEqual(freeze.canonical_model_bytes(art), data)
        mutants = M.artifact_mutants(art)
        self.assertEqual(len(mutants), 36)
        families = {}
        for m in mutants:
            families[m["family"]] = families.get(m["family"], 0) + 1
            with self.subTest(mutant=m["id"]):
                self.assertEqual(_diff_fields(art, m["artifact"]), [m["field"]])
                mbytes = freeze.canonical_model_bytes(m["artifact"])  # frozen resolution, valid
                self.assertNotEqual(mbytes, data)
        self.assertEqual(families, {"weight x0.5": 8, "weight x1.5": 8, "sign flip": 8,
                                    "drop feature": 8, "intercept": 2, "threshold": 2})
        by_id = {m["id"]: m for m in mutants}
        t = art["decision_threshold"]
        self.assertEqual(by_id["A-threshold+0.1"]["after"], (round(t * 100) + 10) / 100)
        self.assertEqual(by_id["A-threshold-0.1"]["after"], (round(t * 100) - 10) / 100)
        self.assertEqual(by_id["A-drop-tls_enabled"]["after"], 0.0)
        self.assertEqual(by_id["A-signflip-queue_utilization"]["after"],
                         -art["weights"]["queue_utilization"])
        self.assertEqual(by_id["A-intercept-1"]["after"], round(art["intercept"] - 1, 12))
        self.assertEqual(art, frozen_v2()[0])  # the source artifact is never modified

    @unittest.skipUnless(S.origin_commit_present(), S.ORIGIN_SKIP_REASON)
    def test_receipt_catalogue(self):
        _art, receipt, _data = frozen_v2()
        mutants = {m["id"]: m for m in M.receipt_mutants(receipt)}
        self.assertEqual(list(mutants), ["R-model-hash-flip", "R-wrong-schema",
                                         "R-source-commit-altered", "R-source-commit-other",
                                         "R-metrics-added"])
        flip = mutants["R-model-hash-flip"]["receipt"]["model_sha256"]
        self.assertRegex(flip, r"^[0-9a-f]{64}$")
        self.assertEqual(sum(a != b for a, b in zip(flip, receipt["model_sha256"])), 1)
        altered = mutants["R-source-commit-altered"]["receipt"]["source"]["commit"]
        self.assertRegex(altered, r"^[0-9a-f]{40}$")
        self.assertEqual(sum(a != b for a, b in zip(altered, receipt["source"]["commit"])), 1)
        other = mutants["R-source-commit-other"]["receipt"]["source"]["commit"]
        self.assertEqual(other, M.parent_commit(receipt["source"]["commit"]))
        self.assertTrue(mutants["R-source-commit-other"]["supplementary"])
        self.assertIn("metrics", mutants["R-metrics-added"]["receipt"])
        self.assertNotIn("metrics", receipt)

    def test_kernel_edits_apply_exactly_once(self):
        source = registry.KERNEL_PATH.read_text(encoding="utf-8")
        ids = [k[0] for k in M.KERNEL_MUTANTS]
        self.assertEqual(ids, ["K-no-finiteness", "K-no-range", "K-no-prohibited-screen",
                               "K-inverted-model-hash", "K-inverted-self-hash"])
        for kid, _d, _s, edits in M.KERNEL_MUTANTS:
            with self.subTest(mutant=kid):
                self.assertNotEqual(M.apply_kernel_edits(source, edits), source)
        with self.assertRaises(M.MutationRefused):
            M.apply_kernel_edits(source, (("no such kernel line\n", ""),))


class IntegrityLayerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = S.TempDir()
        self.tmp = self._tmp.__enter__()
        self.art, self.receipt, self.data = frozen_v2()

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    @unittest.skipUnless(S.origin_commit_present(), S.ORIGIN_SKIP_REASON)
    def test_frozen_v2_pair_passes_the_integrity_layer(self):
        layer = M.integrity_layer(V2_MODEL, V2_RECEIPT, self.receipt)
        self.assertFalse(layer["caught"], layer)

    @unittest.skipUnless(S.origin_commit_present(), S.ORIGIN_SKIP_REASON)
    def test_registered_receipt_mutants_are_caught(self):
        expected = {
            "R-model-hash-flip": ("kernel", "model artifact hash does not match receipt"),
            "R-wrong-schema": ("kernel", "artifact receipt schema mismatch"),
            "R-source-commit-altered": ("verify_origin", "is not in the repository"),
            "R-metrics-added": ("kernel", "artifact receipt fields mismatch"),
        }
        for m in M.receipt_mutants(self.receipt):
            if m["id"] not in expected:
                continue
            layer_name, fragment = expected[m["id"]]
            model, rec = M.write_pair(self.tmp / m["id"], self.data, m["receipt"])
            layer = M.integrity_layer(model, rec, m["receipt"])
            with self.subTest(mutant=m["id"]):
                self.assertTrue(layer["caught"])
                self.assertFalse(layer[layer_name]["accepted"])
                self.assertIn(fragment, layer[layer_name]["refusal"])

    @unittest.skipUnless(S.origin_commit_present(), S.ORIGIN_SKIP_REASON)
    def test_artifact_mutant_pass1_refused_pass2_loads(self):
        m = next(x for x in M.artifact_mutants(self.art) if x["id"] == "A-signflip-listener_running")
        mbytes = freeze.canonical_model_bytes(m["artifact"])
        model1, rec1 = M.write_pair(self.tmp / "p1", mbytes, self.receipt)
        p1 = M.integrity_layer(model1, rec1, self.receipt)
        self.assertTrue(p1["caught"])
        self.assertIn("model artifact hash does not match receipt", p1["kernel"]["refusal"])
        regenerated = dict(copy.deepcopy(self.receipt), model_sha256=M.sha256_bytes(mbytes))
        model2, rec2 = M.write_pair(self.tmp / "p2", mbytes, regenerated)
        p2 = M.integrity_layer(model2, rec2, regenerated)
        self.assertFalse(p2["caught"], p2)
        kernel = K.OpsHealthKernel(model2, rec2)
        self.assertEqual(kernel.artifact["weights"]["listener_running"],
                         -self.art["weights"]["listener_running"])


class KernelMutantTest(unittest.TestCase):
    def setUp(self):
        self._tmp = S.TempDir()
        self.tmp = self._tmp.__enter__()
        self.art, self.receipt, self.data = frozen_v2()
        self.source = registry.KERNEL_PATH.read_text(encoding="utf-8")

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def _mutant(self, kid):
        edits = next(k[3] for k in M.KERNEL_MUTANTS if k[0] == kid)
        path = M.write_kernel_copy(self.tmp, kid, M.apply_kernel_edits(self.source, edits))
        return kernel_bridge.load(path), path

    def _rebound(self, path):
        rebound = dict(copy.deepcopy(self.receipt), kernel_sha256=S.sha256_file(path))
        return M.write_pair(self.tmp / f"rebound_{path.parent.name}", self.data, rebound)

    def test_identity_copy_loads_the_frozen_pair(self):
        path = M.write_kernel_copy(self.tmp, "identity", self.source)
        check = M.kernel_check(V2_MODEL, V2_RECEIPT, kernel_bridge.load(path))
        self.assertTrue(check["accepted"], check)

    def test_self_hash_refuses_every_mutant_but_the_inverted_self_hash(self):
        for kid, _d, _s, _e in M.KERNEL_MUTANTS:
            module, _path = self._mutant(kid)
            check = M.kernel_check(V2_MODEL, V2_RECEIPT, module)
            with self.subTest(mutant=kid):
                if kid == "K-inverted-self-hash":
                    # by construction: an inverted self-check accepts a receipt that does NOT
                    # match this file; only the tests can catch this mutant
                    self.assertTrue(check["accepted"])
                else:
                    self.assertIn("kernel sha256 does not match receipt", check["refusal"])

    def test_mutations_do_what_they_claim(self):
        valid = {"features": dict(gates.VALID_FEATURES)}
        module, path = self._mutant("K-no-finiteness")
        kernel = module.OpsHealthKernel(*self._rebound(path))
        out = kernel.advise({"features": dict(gates.VALID_FEATURES, queue_utilization=float("nan"))})
        self.assertEqual(out["advisory"], "ABSTAIN")  # NaN score: empty prediction set
        module, path = self._mutant("K-no-range")
        kernel = module.OpsHealthKernel(*self._rebound(path))
        kernel.advise({"features": dict(gates.VALID_FEATURES, queue_utilization=1.5)})
        module, path = self._mutant("K-no-prohibited-screen")
        kernel = module.OpsHealthKernel(*self._rebound(path))
        with self.assertRaises(module.ModelInputError) as ctx:
            kernel.advise({"features": dict(gates.VALID_FEATURES, patient_id=0)})
        self.assertNotIn(gates.PROHIBITED, str(ctx.exception))  # refused by the schema only
        kernel.advise(valid)
        module, path = self._mutant("K-inverted-model-hash")
        check = M.kernel_check(*self._rebound(path), module)
        self.assertIn("model artifact hash does not match receipt", check["refusal"])

    def test_kernel_test_harness_control_passes_and_mutant_fails(self):
        control = M.write_kernel_copy(self.tmp, "control", self.source)
        ok = M.run_kernel_tests(control, None, self.tmp, (PROBE_ID,))
        self.assertFalse(ok["harness_error"], ok)
        self.assertTrue(ok["ok"], ok)
        self.assertEqual(ok["tests_run"], 1)
        self.assertEqual(ok["kernel_sha256"], S.sha256_file(control))
        _module, path = self._mutant("K-no-range")
        bad = M.run_kernel_tests(path, None, self.tmp, (PROBE_ID,))
        self.assertFalse(bad["harness_error"], bad)
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["failed_tests"], [PROBE_ID])
        self.assertEqual(bad["errors"], [])
        failing_probes = {f.split("probe='")[1].rstrip("')") for f in bad["failures"]}
        self.assertEqual(failing_probes, {"out_of_range_high", "out_of_range_low",
                                          "out_of_range_failures"})


class BehaviorAndBaselineTest(unittest.TestCase):
    """Gates on DEV rows (evaluate_gates with a research scorer) and the broken baselines."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = dev_context()
        cls.art = S.candidate(calibrator="none")
        cls.reference = M.evaluate_gates(K.ArtifactScorer(cls.art), True, cls.ctx)["gates"]

    def _gates(self, artifact):
        return M.evaluate_gates(K.ArtifactScorer(artifact), True, self.ctx)

    def test_dev_reference_passes_every_gate(self):
        self.assertEqual(set(self.reference.values()), {1.0}, self.reference)

    def test_sign_flip_is_caught_by_truth_sign_and_monotone(self):
        m = next(x for x in M.artifact_mutants(self.art) if x["id"] == "A-signflip-consecutive_failures")
        fx = self._gates(m["artifact"])
        below = M.gates_below(fx["gates"], self.reference)
        self.assertIn("g_truth_sign", below)
        self.assertIn("g_monotone", below)
        self.assertEqual(fx["gates"]["g_truth_sign"], 7 / 8)

    def test_threshold_mutant_changes_decisions_only(self):
        m = next(x for x in M.artifact_mutants(self.art) if x["id"] == "A-threshold+0.1")
        fx = self._gates(m["artifact"])
        ref = M.evaluate_gates(K.ArtifactScorer(self.art), True, self.ctx)
        self.assertEqual(fx["details"]["validation"]["ece"], ref["details"]["validation"]["ece"])
        self.assertNotEqual(fx["details"]["validation"]["balanced_accuracy"],
                            ref["details"]["validation"]["balanced_accuracy"])

    def test_broken_baselines_are_invalid_without_per_class_rates(self):
        art, receipt, data = frozen_v2()
        with S.TempDir() as tmp:
            out = M.broken_baselines(tmp, art, receipt, data, self.ctx)
        self.assertEqual(set(out), {"constant_alert", "fail_open_kernel", "majority"})
        expected = {
            "constant_alert": ["iii_decisions_not_constant"],
            "fail_open_kernel": ["i_receipt_verifies", "iv_refused_all_probes"],
            "majority": ["i_receipt_verifies", "iii_decisions_not_constant", "iv_refused_all_probes"],
        }
        for name, record in out.items():
            report = record["report"]
            with self.subTest(fixture=name):
                self.assertTrue(record["invalid_baseline"])
                self.assertTrue(record["no_per_class_rate_reported"])
                self.assertEqual(report["per_class_rates"], metrics.INVALID_BASELINE)
                self.assertNotIn("confusion", report)
                self.assertNotIn("accuracy", report)
                if name == "constant_alert" and not S.origin_commit_present():
                    # its precondition i is the frozen receipt's origin check
                    self.skipTest(S.ORIGIN_SKIP_REASON)
                self.assertEqual(report["baseline_validity"]["failed_preconditions"], expected[name])
        self.assertEqual(out["constant_alert"]["report"]["balanced_accuracy"], 0.5)
        self.assertEqual(out["majority"]["report"]["balanced_accuracy"], 0.5)
        granted = out["fail_open_kernel"]["with_precondition_i_granted"]
        self.assertEqual(granted["baseline_validity"]["failed_preconditions"], ["iv_refused_all_probes"])
        self.assertEqual(granted["per_class_rates"], metrics.INVALID_BASELINE)
        self.assertGreater(len(out["fail_open_kernel"]["failclosed"]["escaped"]), 0)


if __name__ == "__main__":
    unittest.main()
