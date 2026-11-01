"""Research boundary tests; no model or dataset downloads."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location("nano_research",Path(__file__).parents[1]/"tools/run_nano_research.py")
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

class ResearchTests(unittest.TestCase):
    def test_exact_source_pin_rejects_modified_code_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"receipt_agent.py").write_text("raise RuntimeError('must not execute')",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Source hash mismatch"):
                runner.load_source(root,"receipt_agent.py")

    def test_overlapping_dev_and_test_data_are_rejected(self):
        x=np.zeros((2,24)); x[1,0]=1
        y=np.array([0,1])
        with self.assertRaisesRegex(ValueError,"Cross-split"):
            runner.split_evidence({"train":(x,y),"dev":(x+2,y),"test":(x+2,y)})

    def test_duplicate_or_empty_split_is_rejected(self):
        for x,y in [(np.zeros((2,24)),np.array([0,0])),(np.empty((0,24)),np.array([]))]:
            with self.subTest(n=len(x)),self.assertRaises(ValueError):
                runner.split_evidence({"train":(x,y),"dev":(x,y),"test":(x,y)})

    def test_metrics_report_named_errors_and_uncertainty(self):
        m=runner.metrics([0,0,2,3],[0,1,2,3])
        self.assertEqual((m["correct"],m["n"]),(3,4))
        self.assertEqual(m["unsafe_allow_count"],1)
        self.assertEqual(m["confusion_gold_by_prediction"][1][0],1)
        self.assertLess(m["wilson_95"][0],0.75)
        self.assertGreater(m["wilson_95"][1],0.75)

    def test_missing_or_unknown_evaluation_cannot_pass(self):
        for pred,gold in [([],[]),([0],[0,1]),([4],[0]),([0],[-1])]:
            with self.subTest(pred=pred,gold=gold),self.assertRaises(ValueError):
                runner.metrics(pred,gold)

    def test_digest_is_independent_of_archive_order_and_rejects_nan(self):
        a={"first":np.array([1.]),"second":np.array([2.])}
        self.assertEqual(runner.tensor_digest(a),runner.tensor_digest(dict(reversed(list(a.items())))))
        with self.assertRaises(ValueError):
            runner.tensor_digest({"bad":np.array([np.nan])})

if __name__ == "__main__":
    unittest.main()

