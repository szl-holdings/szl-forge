"""Check measured comparison integrity offline; never invoke inference in CI."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dev12_verifier", ROOT / "inference/verify_minicpm5_dev12.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class Dev12EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "bundle"
        shutil.copytree(d.BUNDLE, self.bundle)

    def update(self, name, value):
        path = self.bundle / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        manifest = d.read_json(self.bundle / "manifest.json")
        manifest["files"][name] = d.sha(path)
        (self.bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_original_result_preserves_local_development_limits(self):
        result = d.verify(self.bundle)
        self.assertEqual((result["baselinePassedCases"], result["candidatePassedCases"], result["caseCount"]),
                         (9, 12, 12))
        self.assertEqual(result["suiteUse"], "DEVELOPMENT_NOT_HELD_OUT")
        self.assertFalse(result["independentWitness"])
        self.assertEqual(result["productionDisposition"], "HOLD")

    def test_rehashed_comparison_cannot_claim_heldout_or_promotion(self):
        for key, value in (("suiteUse", "HELD_OUT"), ("independentWitness", True),
                           ("productionDisposition", "PROMOTE"), ("sourcePatchUnpublished", False),
                           ("inputPreprocessing", "SELECT_AUTHORIZED_DOCUMENT"), ("p50GenerationMs", 1.0)):
            original = d.read_json(d.BUNDLE / "comparison.json")
            original[key] = value
            self.update("comparison.json", original)
            with self.subTest(key=key), self.assertRaises(d.v.ReportError):
                d.verify(self.bundle)

    def test_rehashed_case_comparison_cannot_hide_baseline_failures(self):
        comparison = d.read_json(self.bundle / "comparison.json")
        comparison["caseComparison"][0]["baselinePassed"] = True
        self.update("comparison.json", comparison)
        with self.assertRaises(d.v.ReportError):
            d.verify(self.bundle)

    def test_rehashed_record_cannot_rewrite_source_at_execution(self):
        report = d.read_json(self.bundle / "explicit-lookup-v2.json")
        report["sourceRevision"] = "a" * 40
        report.pop("recordSha256")
        report["recordSha256"] = d.v.q.digest(report)
        self.update("explicit-lookup-v2.json", report)
        with self.assertRaises(d.v.ReportError):
            d.verify(self.bundle)

    def test_rehashed_record_cannot_rewrite_measurement(self):
        report = d.read_json(self.bundle / "explicit-lookup-v2.json")
        report["cases"][0]["generationMs"] = 1.0
        report = d.v.q.finalize(report)
        self.update("explicit-lookup-v2.json", report)
        with self.assertRaises(d.v.ReportError):
            d.verify(self.bundle)

    def test_line_ending_normalization_cannot_replace_executed_bytes(self):
        name = "source/candidate/inference/minicpm5_qualification.py"
        path = self.bundle / name
        raw = path.read_bytes()
        self.assertIn(b"\r\n", raw)
        path.write_bytes(raw.replace(b"\r\n", b"\n"))
        manifest = d.read_json(self.bundle / "manifest.json")
        manifest["files"][name] = d.sha(path)
        (self.bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(d.v.ReportError):
            d.verify(self.bundle)

    def test_manifest_cannot_add_paths_outside_bundle(self):
        manifest = d.read_json(self.bundle / "manifest.json")
        manifest["files"]["../outside"] = "a" * 64
        (self.bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(d.v.ReportError):
            d.verify(self.bundle)


if __name__ == "__main__":
    unittest.main()
