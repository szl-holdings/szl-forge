"""Software-only public/mock exclusion tests; no model or hidden gate is run.

Nonmock below means the declared generator path, not verified weight loading.
Numeric harness PASS is retained, but public fixtures cannot qualify an artifact.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frontier.harness import heldout_gate, lane_l2, lane_l3


PUBLIC_FIXTURES = {
    "chaski_smoke_v1.jsonl": "1b5afe991c73ebcd0a67c0fd1ffd09f10e5c132981dbf50b5a620fb21b850c0d",
    "khipu_abstain_smoke_v1.jsonl": "138c14e64c76f7392c69adc89528b8add5dea9b0451cbf48ae85f9b85201f8dc",
    "receiptagent_smoke_v1.jsonl": "55e8825ce992b3cae7ff0b367829061b04f3c8c149d3272f1f7e7eccbc7fb230",
}


def response(messages):
    if "JSON object" in messages[0]["content"]:
        return json.dumps({
            "artifact": "sample", "base_model": "sample", "claim": "harness wiring",
            "label": "SAMPLE", "decision": "DRAFT", "approvalRequired": True,
            "executed": False,
        })
    return lane_l2.mock_generate(messages)


class PublicSmokeBoundaryTests(unittest.TestCase):
    def fixture(self, name):
        return ROOT / "frontier" / "harness" / "probes" / name

    def baseline(self, path):
        probes, _ = heldout_gate.load_probes(str(path))
        return {probe["kind"]: 0 for probe in probes}

    def assert_excluded(self, receipt, *, simulated=False, public=True):
        self.assertEqual(receipt["evals"], "SIMULATED" if simulated else "MEASURED")
        self.assertEqual(receipt["evaluation_mode"],
                         "PUBLIC_CI_SMOKE" if public else "MOCK_SMOKE" if simulated
                         else "UNVERIFIED_PROBE_SET")
        self.assertIs(receipt["candidate_evaluated"], False)
        self.assertIs(receipt["generator_invoked"], True)
        self.assertEqual(receipt["generator_kind"],
                         "MOCK" if simulated else "DECLARED_UNBOUND")
        self.assertEqual(receipt["evidence_scope"], "HARNESS_ONLY")
        self.assertIs(receipt["qualification_gate_ran"], False)
        self.assertIs(receipt["publication_eligible"], False)
        self.assertIs(receipt["promotion_eligible"], False)
        self.assertEqual(receipt["promotion_effect"], "NONE")
        self.assertEqual(receipt["authority"], "NONE")

    def test_all_public_fixture_content_is_pinned(self):
        for name, expected in PUBLIC_FIXTURES.items():
            with self.subTest(name=name):
                _, digest = heldout_gate.load_probes(str(self.fixture(name)))
                self.assertEqual(digest, expected)

    def test_direct_public_api_pass_remains_non_qualifying(self):
        for name in PUBLIC_FIXTURES:
            with self.subTest(name=name):
                path = self.fixture(name)
                receipt = heldout_gate.run_gate(
                    artifact="sample/nonmock", probes_path=str(path),
                    generate=response, baseline=self.baseline(path))
                self.assertEqual(receipt["gate"], "PASS")
                self.assert_excluded(receipt)

    def test_renamed_reformatted_public_content_is_still_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unrelated-name.jsonl"
            for name, expected in PUBLIC_FIXTURES.items():
                with self.subTest(name=name):
                    probes, _ = heldout_gate.load_probes(str(self.fixture(name)))
                    # Different JSON member order/spacing and extra blank lines.
                    path.write_text("\n" + "\n\n".join(
                        json.dumps(dict(reversed(list(probe.items()))), separators=(",", ":"))
                        for probe in probes) + "\n", encoding="utf-8")
                    receipt = heldout_gate.run_gate(
                        artifact="sample/nonmock", probes_path=str(path),
                        generate=response, baseline=self.baseline(path))
                    self.assertEqual(receipt["probe_set_sha256"], expected)
                    self.assertEqual(receipt["gate"], "PASS")
                    self.assert_excluded(receipt)

    def test_known_smoke_filename_is_excluded_after_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in PUBLIC_FIXTURES:
                with self.subTest(name=name):
                    path = Path(directory) / name
                    path.write_text(json.dumps({"id": "changed", "kind": "abstain",
                                                "prompt": "An altered software fixture"}) + "\n",
                                    encoding="utf-8")
                    receipt = heldout_gate.run_gate(
                        artifact="sample/nonmock", probes_path=str(path),
                        generate=response, baseline={"abstain": 0})
                    self.assertEqual(receipt["gate"], "PASS")
                    self.assert_excluded(receipt)

    def test_public_rows_stay_public_after_reorder_subset_and_id_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-the-public-filename.jsonl"
            for name in PUBLIC_FIXTURES:
                probes, _ = heldout_gate.load_probes(str(self.fixture(name)))
                variants = (list(reversed(probes)), probes[:1],
                            [dict(probe, id="new-" + probe["id"]) for probe in probes])
                for variant in variants:
                    with self.subTest(name=name, rows=len(variant), first_id=variant[0]["id"]):
                        path.write_text("\n".join(json.dumps(probe) for probe in variant) + "\n",
                                        encoding="utf-8")
                        receipt = heldout_gate.run_gate(
                            artifact="sample/nonmock", probes_path=str(path),
                            generate=response, baseline=self.baseline(path))
                        self.assertEqual(receipt["gate"], "PASS")
                        self.assert_excluded(receipt)

    def test_explicit_mock_public_content_is_simulated(self):
        path = self.fixture("khipu_abstain_smoke_v1.jsonl")
        receipt = heldout_gate.run_gate(
            artifact="sample/mock", probes_path=str(path), generate=response,
            baseline={"abstain": 3}, mock=True)
        self.assertEqual(receipt["abstain"], "6/6")
        self.assertEqual(receipt["gate"], "PASS")
        self.assert_excluded(receipt, simulated=True)

    def test_mock_on_unclassified_content_is_simulated_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unclassified.jsonl"
            path.write_text(json.dumps({"id": "example", "kind": "abstain",
                                        "prompt": "Software example, not a hidden gate"}) + "\n",
                            encoding="utf-8")
            receipt = heldout_gate.run_gate(
                artifact="sample/mock", probes_path=str(path), generate=response,
                baseline={"abstain": 0}, mock=True)
            self.assertEqual(receipt["gate"], "PASS")
            self.assert_excluded(receipt, simulated=True, public=False)

    def test_hash_mismatch_stays_invalid_and_never_calls_generator(self):
        generate = mock.Mock(side_effect=AssertionError("must not generate"))
        for simulated in (False, True):
            with self.subTest(simulated=simulated):
                receipt = heldout_gate.run_gate(
                    artifact="sample", probes_path=str(self.fixture("khipu_abstain_smoke_v1.jsonl")),
                    generate=generate, declared_probe_sha256="0" * 64,
                    baseline={"abstain": 3}, mock=simulated)
                self.assertEqual(receipt["gate"], "INVALID")
                self.assertNotIn("rows", receipt)
                self.assertIs(receipt["publication_eligible"], False)
                self.assertIs(receipt["qualification_gate_ran"], False)
                self.assertIs(receipt["generator_invoked"], False)
        generate.assert_not_called()

    def test_malformed_unknown_probe_hash_mismatch_rejects_before_fingerprinting(self):
        generate = mock.Mock(side_effect=AssertionError("must not generate"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown-malformed.jsonl"
            path.write_text(json.dumps({"id": "missing-prompt", "kind": "abstain"}) + "\n",
                            encoding="utf-8")
            receipt = heldout_gate.run_gate(
                artifact="sample", probes_path=str(path), generate=generate,
                declared_probe_sha256="0" * 64, baseline={"abstain": 0})
        self.assertEqual(receipt["gate"], "INVALID")
        self.assertIs(receipt["generator_invoked"], False)
        self.assertIs(receipt["publication_eligible"], False)
        self.assertNotIn("rows", receipt)
        generate.assert_not_called()

    def test_public_failure_stays_failed_without_qualification(self):
        receipt = heldout_gate.run_gate(
            artifact="sample", probes_path=str(self.fixture("khipu_abstain_smoke_v1.jsonl")),
            generate=lambda _: "NAVIGATE", baseline={"abstain": 3})
        self.assertEqual(receipt["gate"], "FAIL")
        self.assertEqual(receipt["abstain"], "0/6")
        self.assert_excluded(receipt)

    def test_metadata_cannot_override_public_exclusion(self):
        receipt = heldout_gate.run_gate(
            artifact="sample", probes_path=str(self.fixture("khipu_abstain_smoke_v1.jsonl")),
            generate=response, baseline={"abstain": 3},
            env={"qualification_gate_ran": True, "candidate_evaluated": False,
                 "publication_eligible": True, "promotion_eligible": True,
                 "promotion_effect": "PROMOTE", "authority": "MODEL"})
        self.assertEqual(receipt["gate"], "PASS")
        self.assert_excluded(receipt)

    def test_both_mock_wrappers_exclude_unclassified_content(self):
        with tempfile.TemporaryDirectory() as directory:
            for lane, name in ((lane_l2, "khipu_abstain_smoke_v1.jsonl"),
                               (lane_l3, "chaski_smoke_v1.jsonl")):
                with self.subTest(lane=lane.__name__):
                    probes, _ = heldout_gate.load_probes(str(self.fixture(name)))
                    for probe in probes:
                        probe["id"] = "modified-" + probe["id"]
                        probe["prompt"] += " [software-only unclassified variant]"
                    path = Path(directory) / "unclassified.jsonl"
                    path.write_text("\n".join(json.dumps(probe) for probe in probes) + "\n",
                                    encoding="utf-8")
                    output = Path(directory) / "receipt.json"
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = lane.main(["--artifact", "sample", "--probes", str(path),
                                          "--mock", "--out", str(output)])
                    self.assertEqual(code, 0)
                    receipt = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(receipt["gate"], "PASS")
                    self.assert_excluded(receipt, simulated=True, public=False)

    def test_both_nonmock_wrappers_exclude_public_content(self):
        with tempfile.TemporaryDirectory() as directory:
            for lane, name in ((lane_l2, "khipu_abstain_smoke_v1.jsonl"),
                               (lane_l3, "chaski_smoke_v1.jsonl")):
                with self.subTest(lane=lane.__name__):
                    output = Path(directory) / "receipt.json"
                    with mock.patch.object(lane, "load_generate", return_value=response), \
                            contextlib.redirect_stdout(io.StringIO()):
                        code = lane.main(["--artifact", "sample", "--probes", str(self.fixture(name)),
                                          "--generate", "sample:declared_nonmock", "--out", str(output)])
                    self.assertEqual(code, 0)
                    receipt = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(receipt["gate"], "PASS")
                    self.assert_excluded(receipt)

    def test_unclassified_content_does_not_gain_hidden_gate_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unclassified.jsonl"
            path.write_text(json.dumps({"id": "example", "kind": "abstain",
                                        "prompt": "Software example, not a hidden gate"}) + "\n",
                            encoding="utf-8")
            receipt = heldout_gate.run_gate(
                artifact="sample", probes_path=str(path), generate=response,
                baseline={"abstain": 0})
            self.assertEqual(receipt["gate"], "PASS")
            self.assertIs(receipt["baseline_beaten"], True)
            self.assert_excluded(receipt, public=False)

    def test_missing_optional_public_catalogue_still_denies_unclassified_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unclassified.jsonl"
            path.write_text(json.dumps({"id": "example", "kind": "abstain",
                                        "prompt": "Software example, not a hidden gate"}) + "\n",
                            encoding="utf-8")
            original = heldout_gate.load_probes

            def load_without_catalogue(probe_path):
                if Path(probe_path) != path:
                    raise FileNotFoundError("optional public catalogue unavailable")
                return original(probe_path)

            with mock.patch.object(heldout_gate, "load_probes", side_effect=load_without_catalogue):
                receipt = heldout_gate.run_gate(
                    artifact="sample", probes_path=str(path), generate=response,
                    baseline={"abstain": 0})
            self.assertEqual(receipt["gate"], "PASS")
            self.assert_excluded(receipt, public=False)

    def test_mock_provenance_must_be_an_exact_boolean(self):
        generate = mock.Mock(side_effect=AssertionError("must not generate"))
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                heldout_gate.run_gate(
                    artifact="sample", probes_path=str(self.fixture("khipu_abstain_smoke_v1.jsonl")),
                    generate=generate, mock=value)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
