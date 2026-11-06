"""Preflight unit tests using fake Hub clients - no network, no GPU, no writes."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import preflight
from preflight import PreflightError, run_preflight

STUDENT = "a" * 40
DATASET = "c" * 40


def _rows(count: int, prefix: str) -> str:
    lines = []
    for index in range(count):
        lines.append(json.dumps({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"{prefix}-{index}"},
                {"role": "assistant", "content": f"answer-{prefix}-{index}"},
            ]
        }))
    return "\n".join(lines) + "\n"


class FakeFile:
    def __init__(self, rfilename):
        self.rfilename = rfilename


class FakeApi:
    def __init__(self, files, resolved_id=None, shas=None):
        self._files = files
        self._resolved_id = resolved_id
        self._shas = shas or {}

    def list_repo_files(self, repo, repo_type="model"):
        if "UNREADABLE" in repo:
            raise RuntimeError("404")
        return list(self._files)

    def repo_info(self, repo, repo_type="model"):
        if "UNREADABLE" in repo:
            raise RuntimeError("404")
        return type("I", (), {"id": self._resolved_id or repo})()

    def model_info(self, mid):
        if "UNREADABLE" in mid:
            raise RuntimeError("404")
        return type("I", (), {"sha": self._shas.get(mid, STUDENT)})()

    def dataset_info(self, did, files_metadata=False):
        if "UNREADABLE" in did:
            raise RuntimeError("404")
        return type("I", (), {
            "sha": self._shas.get(did, DATASET),
            "siblings": [FakeFile("data.jsonl")],
        })()


def write_ready(directory: Path, **overrides) -> str:
    train = directory / "train.jsonl"
    holdout = directory / "holdout.jsonl"
    train.write_text(_rows(32, "train"), encoding="utf-8")
    holdout.write_text(_rows(2, "hold"), encoding="utf-8")
    admission = {
        "schema": "szl.forge.training-admission/v1",
        "dataset_id": "ExampleOrg/data",
        "revision": DATASET,
        "split": "train",
        "license": "operator-supplied-license",
        "rights": "ADMITTED_FOR_TRAINING",
        "commercial_use": "ADMITTED",
        "basis": "OPERATOR_ADMISSION_RECEIPT",
        "corpus_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    }
    (directory / "admission.json").write_text(json.dumps(admission), encoding="utf-8")
    config = {
        "schema": "szl.forge.wave-config/v1",
        "kind": "sft",
        "student": {"id": "ExampleOrg/student", "revision": STUDENT},
        "dataset": {"id": "ExampleOrg/data", "revision": DATASET, "split": "train"},
        "train_jsonl": "train.jsonl",
        "holdout_jsonl": "holdout.jsonl",
        "admission": "admission.json",
        "method": "qlora",
        "target_repo": "SZLHOLDINGS/ok-target",
        "out": "out",
    }
    config.update(overrides)
    path = directory / "wave.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


class TestPreflight(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        os.environ["HF_TOKEN"] = "test"

    def tearDown(self):
        self._tmp.cleanup()

    def test_pass(self):
        api = FakeApi(["README.md", "config.json"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            report = run_preflight(write_ready(self.directory))
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["base_model_pin"]["sha"], STUDENT)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["hub_write"])
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["accepted_examples"], 32)

    def test_identity_mismatch_rejected(self):
        api = FakeApi(
            ["README.md", "config.json"],
            resolved_id="SZLHOLDINGS/someone-elses-target",
        )
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(write_ready(self.directory))
        self.assertEqual(caught.exception.code, "PREFLIGHT_TARGET_MISMATCH")

    def test_npz_target_rejected(self):
        api = FakeApi(["README.md", "tiny_khipu.npz"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(write_ready(
                    self.directory, target_repo="SZLHOLDINGS/npz-target"
                ))
        self.assertEqual(caught.exception.code, "PREFLIGHT_NON_LM_TARGET")

    def test_unreadable_target_rejected(self):
        api = FakeApi([])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(write_ready(
                    self.directory, target_repo="SZLHOLDINGS/UNREADABLE"
                ))
        self.assertEqual(caught.exception.code, "PREFLIGHT_TARGET_UNREADABLE")

    def test_unreadable_model_rejected(self):
        api = FakeApi(["README.md"])
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(write_ready(
                    self.directory,
                    student={"id": "ExampleOrg/UNREADABLE", "revision": STUDENT},
                ))
        self.assertEqual(caught.exception.code, "PREFLIGHT_MODEL_UNREADABLE")

    def test_contract_rejection_does_not_call_hub(self):
        os.environ.pop("HF_TOKEN", None)
        blocked = Path(__file__).resolve().parents[1] / "w30" / "distill-r2.blocked.json"
        with mock.patch.object(preflight, "hf_api", side_effect=AssertionError("hub")):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(str(blocked))
        self.assertEqual(caught.exception.code, "LAUNCH_CONTRACT_REJECTED")
        self.assertIn("LAUNCH_PIN_NOT_IMMUTABLE", str(caught.exception))

    def test_special_method_rejected_before_hub(self):
        with mock.patch.object(preflight, "hf_api", side_effect=AssertionError("hub")):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(write_ready(self.directory, method="special-trainer"))
        self.assertEqual(caught.exception.code, "LAUNCH_CONTRACT_REJECTED")
        self.assertIn("LAUNCH_METHOD_NOT_TRAINABLE", str(caught.exception))

    def test_pin_mismatch_rejected(self):
        api = FakeApi(["README.md", "config.json"], shas={"ExampleOrg/student": "b" * 40})
        with mock.patch.object(preflight, "hf_api", return_value=api):
            with self.assertRaises(PreflightError) as caught:
                run_preflight(write_ready(self.directory))
        self.assertEqual(caught.exception.code, "PREFLIGHT_MODEL_PIN_MISMATCH")


if __name__ == "__main__":
    unittest.main()
